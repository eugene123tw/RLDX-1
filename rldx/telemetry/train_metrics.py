"""Rank-local training metrics and scope-gated loader timing."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import contextvars
from dataclasses import asdict, dataclass
from datetime import timedelta
import gc
import json
import logging
import math
import os
from statistics import median, quantiles
import sys
import time
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info


@dataclass(frozen=True)
class WindowSummary:
    n: int
    window: int
    mean: float
    p50: float
    p95: float
    max: float


@dataclass(frozen=True)
class StallReport:
    ts: float
    rank: int | None
    worker: int | None
    tag: str
    waiting: float
    dataset: str | None = None


@dataclass(frozen=True)
class ShardSummary:
    ts: float
    rank: int | None
    worker: int | None
    tag: str
    shards_total: int
    wait: WindowSummary | None
    build: WindowSummary | None
    parquet: WindowSummary | None
    decode: WindowSummary | None
    dataset: str | None = None


ROLLING_WINDOW_SIZE = 200
_logger = logging.getLogger(__name__)


def _metric_failed(name: str, error: Exception) -> None:
    try:
        _logger.warning("%s disabled after %s", name, type(error).__name__, exc_info=True)
    except Exception:
        pass


def summarize_window(values: list[float] | deque[float], window: int) -> dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        value = values[0]
        return {
            "n": 1,
            "window": window,
            "mean": value,
            "p50": value,
            "p95": value,
            "max": value,
        }
    samples = list(values)
    return {
        "n": len(samples),
        "window": window,
        "mean": sum(samples) / len(samples),
        "p50": median(samples),
        "p95": quantiles(samples, n=20, method="inclusive")[18],
        "max": max(samples),
    }


class WindowStats:
    """Rolling window of durations (seconds) with percentile summaries."""

    def __init__(self, window: int = ROLLING_WINDOW_SIZE):
        self._samples: deque[float] = deque(maxlen=window)

    def add(self, seconds: float) -> None:
        self._samples.append(seconds)

    def __len__(self) -> int:
        return len(self._samples)

    def summary(self) -> dict[str, float]:
        return summarize_window(self._samples, self._samples.maxlen)


@dataclass
class StepRecord:
    duration_s: float = 0.0
    samples: int = 0
    padded_input_tokens: int = 0
    fetch_s: float = 0.0
    route_s: float = 0.0


class _TimedIterator:
    """Measure successful pulls without counting exhaustion probes."""

    def __init__(self, iterator, metrics: StepMetrics):
        self._iterator = iterator
        self._metrics = metrics

    def __iter__(self):
        return self

    def __next__(self):
        started = self._metrics._start_timer()
        value = next(self._iterator)
        self._metrics._record_fetch(started)
        return value


class StepMetrics:
    def __init__(self, window: int = ROLLING_WINDOW_SIZE):
        self._window = window
        self.reset()

    def reset(self) -> None:
        self.fetch = WindowStats(self._window)
        self._steps: deque[StepRecord] = deque(maxlen=self._window)
        self._pending = StepRecord()
        self._previous_end: float | None = None
        self._timing_enabled = True
        self._counts_enabled = True

    def _disable_timing(self, error: Exception) -> None:
        if self._timing_enabled:
            self._timing_enabled = False
            _metric_failed("step timing metrics", error)

    def _disable_counts(self, error: Exception) -> None:
        if self._counts_enabled:
            self._counts_enabled = False
            _metric_failed("batch count metrics", error)

    def _start_timer(self) -> float | None:
        if not self._timing_enabled:
            return None
        try:
            return time.perf_counter()
        except Exception as error:
            self._disable_timing(error)
            return None

    def _record_fetch(self, started: float | None) -> None:
        if started is None or not self._timing_enabled:
            return
        try:
            elapsed = time.perf_counter() - started
            self.fetch.add(elapsed)
            self._pending.fetch_s += elapsed
        except Exception as error:
            self._disable_timing(error)

    def timed_iterator(self, iterator):
        return _TimedIterator(iterator, self)

    def add_batches(self, batches) -> None:
        if not self._counts_enabled:
            return
        try:
            samples = 0
            tokens = 0
            for batch in batches:
                inputs = batch["inputs"]
                samples += inputs["action_mask"].shape[0]
                tokens += inputs["input_ids"].numel()
            self._pending.samples += samples
            self._pending.padded_input_tokens += tokens
        except Exception as error:
            self._disable_counts(error)

    @contextmanager
    def routing(self) -> Iterator[None]:
        started = self._start_timer()
        try:
            yield
        finally:
            if started is not None and self._timing_enabled:
                try:
                    self._pending.route_s += time.perf_counter() - started
                except Exception as error:
                    self._disable_timing(error)

    def finish_step(self) -> None:
        if not self._timing_enabled:
            return
        try:
            now = time.perf_counter()
            if self._previous_end is not None:
                self._pending.duration_s = now - self._previous_end
                self._steps.append(self._pending)
            self._pending = StepRecord()
            self._previous_end = now
        except Exception as error:
            self._disable_timing(error)

    def skip_next_step(self) -> None:
        self._previous_end = None

    def resume_after_log(self) -> None:
        if self._previous_end is None or not self._timing_enabled:
            return
        try:
            self._previous_end = time.perf_counter()
        except Exception as error:
            self._disable_timing(error)

    def log_values(self) -> dict[str, float]:
        if not self._timing_enabled:
            return {}
        try:
            logs = {
                f"perf/fetch_{stat}_s": round(value, 6)
                for stat, value in self.fetch.summary().items()
            }
            durations = [step.duration_s for step in self._steps]
            logs.update(
                {
                    f"perf/step_{stat}_s": round(value, 6)
                    for stat, value in summarize_window(durations, self._window).items()
                }
            )
            elapsed = sum(durations)
            if elapsed > 0:
                logs["perf/fetch_share"] = round(
                    sum(step.fetch_s for step in self._steps) / elapsed, 4
                )
                logs["perf/route_share"] = round(
                    sum(step.route_s for step in self._steps) / elapsed, 4
                )
        except Exception as error:
            self._disable_timing(error)
            return {}
        if elapsed > 0 and self._counts_enabled:
            try:
                logs.update(
                    {
                        "perf/samples_per_s": round(
                            sum(step.samples for step in self._steps) / elapsed, 3
                        ),
                        "perf/tokens_per_s": round(
                            sum(step.padded_input_tokens for step in self._steps) / elapsed, 1
                        ),
                    }
                )
            except Exception as error:
                self._disable_counts(error)
        return logs


class GcCollectionStats:
    """Collection time and gen-2 count since the last drain.

    Forked workers inherit the callback list; this counter answers only to
    the process that installed it.
    """

    def __init__(self):
        self._pid: int | None = None
        self._started: float | None = None
        self._enabled = True
        self.seconds = 0.0
        self.gen2_collections = 0

    def __call__(self, phase: str, info: dict) -> None:
        if not self._enabled:
            return
        try:
            if os.getpid() != self._pid:
                return
            if phase == "start":
                self._started = time.perf_counter()
                return
            if self._started is None:
                return
            self.seconds += time.perf_counter() - self._started
            self._started = None
            if info["generation"] == 2:
                self.gen2_collections += 1
        except Exception as error:
            self._enabled = False
            _metric_failed("GC metrics", error)

    def install(self) -> None:
        self._enabled = True
        try:
            self.drain()
            self._pid = os.getpid()
            if self not in gc.callbacks:
                gc.callbacks.append(self)
        except Exception as error:
            self._enabled = False
            _metric_failed("GC metrics", error)

    def uninstall(self) -> None:
        self._pid = None
        try:
            if self in gc.callbacks:
                gc.callbacks.remove(self)
        except Exception as error:
            if self._enabled:
                self._enabled = False
                _metric_failed("GC metrics", error)

    @contextmanager
    def recording(self) -> Iterator[None]:
        self.install()
        try:
            yield
        finally:
            self.uninstall()

    def drain(self) -> tuple[float, int]:
        drained = (self.seconds, self.gen2_collections)
        self.seconds = 0.0
        self.gen2_collections = 0
        return drained

    def log_values(self) -> dict[str, float]:
        if not self._enabled:
            return {}
        try:
            seconds, gen2 = self.drain()
            return {"perf/gc_s": round(seconds, 6), "perf/gc_gen2_n": gen2}
        except Exception as error:
            self._enabled = False
            _metric_failed("GC metrics", error)
            return {}


class CudaMemoryStats:
    def __init__(self):
        self._retries_seen = 0
        self._enabled = True

    def reset(self, device: torch.device) -> None:
        self._enabled = True
        try:
            retries = torch.cuda.memory_stats(device).get("num_alloc_retries", 0)
            torch.cuda.reset_peak_memory_stats(device)
            self._retries_seen = retries
        except Exception as error:
            self._enabled = False
            _metric_failed("CUDA memory metrics", error)

    def log_values(self, device: torch.device) -> dict[str, float]:
        if not self._enabled:
            return {}
        try:
            gib = 2**30
            retries = torch.cuda.memory_stats(device).get("num_alloc_retries", 0)
            values = {
                "mem/peak_gib": round(torch.cuda.max_memory_allocated(device) / gib, 3),
                "mem/allocated_gib": round(torch.cuda.memory_allocated(device) / gib, 3),
                "mem/reserved_gib": round(torch.cuda.memory_reserved(device) / gib, 3),
                "mem/alloc_retries": retries - self._retries_seen,
            }
            torch.cuda.reset_peak_memory_stats(device)
            self._retries_seen = retries
            return values
        except Exception as error:
            self._enabled = False
            _metric_failed("CUDA memory metrics", error)
            return {}


_RANK_METRIC_KEYS = (
    *(
        f"perf/{name}_{stat}_s"
        for name in ("fetch", "step")
        for stat in ("n", "window", "mean", "p50", "p95", "max")
    ),
    "perf/fetch_share",
    "perf/route_share",
    "perf/samples_per_s",
    "perf/tokens_per_s",
    "perf/gc_s",
    "perf/gc_gen2_n",
    "mem/peak_gib",
    "mem/allocated_gib",
    "mem/reserved_gib",
    "mem/alloc_retries",
)


class RankMetricGather:
    def __init__(self):
        self._group = None
        self._enabled = True
        # Gloo workers must never drop the last ref to these: that needs the GIL,
        # which ~ProcessGroupGloo holds while waiting on the workers (deadlock).
        self._tensors = None

    @contextmanager
    def recording(self) -> Iterator[None]:
        self._enabled = True
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            try:
                self._group = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
            except Exception as error:
                self._enabled = False
                _metric_failed("rank metrics", error)
        try:
            yield
        finally:
            if self._group is not None:
                try:
                    dist.destroy_process_group(self._group)
                except Exception as error:
                    _metric_failed("rank metrics", error)
                self._group = None
                self._tensors = None

    def gather(self, values: dict[str, float]) -> dict[str, float]:
        if self._group is None or not self._enabled:
            return {}
        try:
            local = torch.tensor(
                [float(values.get(key, math.nan)) for key in _RANK_METRIC_KEYS],
                dtype=torch.float64,
            )
            rank = dist.get_rank()
            world = dist.get_world_size(self._group)
            gathered = [torch.empty_like(local) for _ in range(world)] if rank == 0 else None
            self._tensors = (local, gathered)
            dist.gather(local, gather_list=gathered, dst=0, group=self._group)
            if gathered is None:
                return {}
            width = max(2, len(str(world - 1)))
            return {
                f"{key}/rank{source:0{width}d}": value
                for source, row in enumerate(gathered)
                for key, value in zip(_RANK_METRIC_KEYS, row.tolist())
                if math.isfinite(value)
            }
        except Exception as error:
            self._enabled = False
            _metric_failed("rank metrics", error)
            return {}


class LoaderMetrics:
    """Scope-gated loader timing windows."""

    def __init__(
        self, window: int = 200, every: int | None = None, stall_after: float | None = None
    ):
        if every is None:
            every = int(os.environ.get("RLDX_LOADER_METRICS_EVERY", "10") or "10")
        if stall_after is None:
            stall_after = float(os.environ.get("RLDX_LOADER_STALL_SEC", "30") or "30")
        self.every = max(1, every)
        self.stall_after = stall_after
        self.shard_wait = WindowStats(window)
        self.shard_build = WindowStats(window)
        self.parquet = WindowStats(window)
        self.view_decode = WindowStats(window)
        self._shards_seen = 0
        self._training_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
            f"loader_metrics_training_{id(self)}", default=False
        )

    @contextmanager
    def training_scope(self) -> Iterator[None]:
        token = self._training_active.set(True)
        try:
            yield
        finally:
            self._training_active.reset(token)

    @contextmanager
    def timed(self, stats: WindowStats) -> Iterator[None]:
        if not self._training_active.get():
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            stats.add(time.perf_counter() - started)

    @staticmethod
    def _window_summary(stats: WindowStats) -> WindowSummary | None:
        summary = stats.summary()
        return WindowSummary(**summary) if summary else None

    @staticmethod
    def _identity() -> tuple[float, int | None, int | None]:
        worker_info = get_worker_info()
        rank = os.environ.get("RANK")
        return (
            time.time(),
            int(rank) if rank else None,
            worker_info.id if worker_info is not None else None,
        )

    def snapshot(self, tag: str, dataset: str | None = None) -> ShardSummary:
        ts, rank, worker = self._identity()
        return ShardSummary(
            ts=ts,
            rank=rank,
            worker=worker,
            tag=tag,
            shards_total=self._shards_seen,
            wait=self._window_summary(self.shard_wait),
            build=self._window_summary(self.shard_build),
            parquet=self._window_summary(self.parquet),
            decode=self._window_summary(self.view_decode),
            dataset=dataset,
        )

    @staticmethod
    def _write(prefix: str, payload: ShardSummary | StallReport) -> None:
        # One write: the line must stay whole across worker processes
        # sharing a pipe (atomic while under PIPE_BUF).
        sys.stdout.write(prefix + json.dumps(asdict(payload)) + "\n")
        sys.stdout.flush()

    def emit(self, snapshot: ShardSummary) -> None:
        self._write("[LOADER] ", snapshot)

    def on_shard(self, tag: str, dataset: str | None = None) -> None:
        self._shards_seen += 1
        if self._shards_seen % self.every == 0:
            self.emit(self.snapshot(tag, dataset=dataset))

    def on_stall(self, tag: str, waiting: float, dataset: str | None = None) -> None:
        ts, rank, worker = self._identity()
        self._write(
            "[LOADER-STALL] ",
            StallReport(ts=ts, rank=rank, worker=worker, tag=tag, waiting=waiting, dataset=dataset),
        )


LOADER_METRICS = LoaderMetrics()
