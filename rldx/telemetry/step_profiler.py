"""Env-gated per-section latency / jitter profiler for the inference path.

Activate by exporting ``RLDX_PROFILE=1`` (or any truthy value) before launching
the server. When disabled (the default) every entry point is a near-zero-cost
no-op so this can be safely committed.

Background — the eval fast-path commits (95cadaf5 / 736ab811) pointed out
that image preprocessing on the inference path jitters by tens of ms because
the albumentations conversion chain (PIL→numpy→A.Resize→torch→numpy→PIL) and
the fast Qwen image processor both touch CUDA memory pools / take lazy-decode
hits in unpredictable ways. To see which sub-step is the culprit on a given
checkpoint we record per-section wall-clock samples in a rolling window and
print mean / p50 / p99 / jitter (max-min) every N steps.

Env knobs:
  RLDX_PROFILE         truthy → profiler on. Default off.
  RLDX_PROFILE_EVERY   report cadence in ``step()`` count. Default 50.
  RLDX_PROFILE_WINDOW  rolling window length per section. Default 200.
  RLDX_PROFILE_SYNC    truthy → ``torch.cuda.synchronize()`` at section
                       boundaries. Default off (preprocess is mostly CPU; turn
                       on if you suspect CUDA work is bleeding into a section).
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
import os
import sys
import threading
import time
from typing import Any

from rldx.dist import rank_zero_print as _print


def _truthy(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() not in {"", "0", "false", "no", "off"}


class StepProfiler:
    """Rolling-window per-section timer.

    Thread-safe under the inference server's typical access pattern: one
    runtime thread enters sections sequentially, ``mark_step()`` is called
    once per inference step. The lock is held only briefly during sample
    append / report emission.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        every: int,
        window: int,
        sync_cuda: bool,
        heartbeat: bool = False,
    ) -> None:
        self.enabled = enabled
        self.every = max(1, every)
        self.window = max(1, window)
        self.sync_cuda = sync_cuda
        self.heartbeat = heartbeat
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window))
        self._order: list[str] = []
        self._step_count = 0
        self._heartbeat_section = "step.total"
        self._lock = threading.Lock()

    @contextmanager
    def section(self, name: str):
        """Time the wrapped block, recording wall-clock ms into ``name``."""
        if not self.enabled:
            yield
            return
        if self.sync_cuda:
            self._maybe_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda:
                self._maybe_sync()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                if name not in self._samples:
                    self._order.append(name)
                self._samples[name].append(dt_ms)

    def mark_step(self) -> None:
        """Call once per inference step. Emits the periodic report.

        We also fire a one-shot ``first_fire`` report at step 10 so the user
        sees the profiler is alive long before the (potentially large)
        ``every`` window completes its first cycle.
        """
        if not self.enabled:
            return
        with self._lock:
            self._step_count += 1
            step_idx = self._step_count
            first_fire = step_idx == min(10, self.every)
            periodic = step_idx % self.every == 0
            if first_fire or periodic:
                self._emit_report_locked()
            elif self.heartbeat:
                # One short line per step so a hang is immediately visible
                # (the next step's heartbeat simply never prints).
                self._emit_heartbeat_locked(step_idx)

    def mark_step_start(self, sids: Any = None) -> None:
        """Heartbeat at the beginning of a step (paired with mark_step).

        Use case: pinpoint hangs by tag. If ``[PROFILE-HB-START] step=N`` is
        the last line you see, the hang is somewhere inside step N's body
        (before mark_step fires). If the matching ``[PROFILE-HB] step=N`` did
        appear, the step finished and the hang is elsewhere.
        """
        if not (self.enabled and self.heartbeat):
            return
        # Read step_count under the lock so the printed index lines up with
        # the matching end-of-step heartbeat.
        with self._lock:
            next_step = self._step_count + 1
        _print(f"[PROFILE-HB-START] step={next_step} sids={sids}")
        try:
            sys.stdout.flush()
        except Exception:
            pass

    def _emit_heartbeat_locked(self, step_idx: int) -> None:
        samples = self._samples.get(self._heartbeat_section)
        last_ms = samples[-1] if samples else float("nan")
        _print(f"[PROFILE-HB] step={step_idx} {self._heartbeat_section}={last_ms:.1f}ms")
        try:
            sys.stdout.flush()
        except Exception:
            pass

    def _emit_report_locked(self) -> None:
        if not self._samples:
            return
        lines = [
            f"[PROFILE] step={self._step_count} window={self.window} "
            f"(ms: mean / p50 / p99 / jitter[max-min] over last N)"
        ]
        for name in self._order:
            samples = list(self._samples[name])
            if not samples:
                continue
            n = len(samples)
            mean = sum(samples) / n
            srt = sorted(samples)
            p50 = srt[int(0.5 * (n - 1))]
            p99 = srt[min(n - 1, int(0.99 * (n - 1) + 0.5))]
            jitter = srt[-1] - srt[0]
            lines.append(
                f"  {name:<40s} n={n:>4d}  "
                f"mean={mean:>7.2f}  p50={p50:>7.2f}  "
                f"p99={p99:>7.2f}  jitter={jitter:>7.2f}"
            )
        _print("\n".join(lines))
        # Inference servers often run under buffered stdout (no tty); force a
        # flush so the report is not swallowed until the next big chunk.
        try:
            sys.stdout.flush()
        except Exception:
            pass

    @staticmethod
    def _maybe_sync() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()
        except Exception:
            # Profiler must never break the inference path.
            pass


_profiler: StepProfiler | None = None
_profiler_lock = threading.Lock()


def get_profiler() -> StepProfiler:
    """Process-wide singleton. Config is sampled from env on first call."""
    global _profiler
    if _profiler is not None:
        return _profiler
    with _profiler_lock:
        if _profiler is None:
            enabled = _truthy(os.environ.get("RLDX_PROFILE"))
            every = int(os.environ.get("RLDX_PROFILE_EVERY", "50") or "50")
            window = int(os.environ.get("RLDX_PROFILE_WINDOW", "200") or "200")
            sync_cuda = _truthy(os.environ.get("RLDX_PROFILE_SYNC"))
            heartbeat = _truthy(os.environ.get("RLDX_PROFILE_HEARTBEAT"))
            _profiler = StepProfiler(
                enabled=enabled,
                every=every,
                window=window,
                sync_cuda=sync_cuda,
                heartbeat=heartbeat,
            )
            if enabled:
                _print(
                    f"[PROFILE] enabled (every={every}, window={window}, "
                    f"sync_cuda={sync_cuda}, heartbeat={heartbeat})"
                )
    return _profiler


@contextmanager
def profile_section(name: str):
    """Convenience wrapper — ``with profile_section('foo'): ...``."""
    with get_profiler().section(name):
        yield


def mark_step() -> None:
    get_profiler().mark_step()


def mark_step_start(sids: Any = None) -> None:
    get_profiler().mark_step_start(sids)
