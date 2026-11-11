# SPDX-License-Identifier: Apache-2.0
"""Periodic augmented-image snapshot utility.

Useful for debugging the train augmentation pipeline: periodically dumps
`(orig, augmented)` PNG pairs so the user can eyeball whether the crop,
rotation and color jitter ranges are sane.

Enable by setting env vars (defaults disabled):
  RLDX_DUMP_AUG_EVERY_N_STEPS=10
  RLDX_DUMP_AUG_DIR=/path/to/dump/dir

Outputs land at:
  {dump_dir}/step_{N:06d}_pid{P}_{view}_sample{k}_orig.png
  {dump_dir}/step_{N:06d}_pid{P}_{view}_sample{k}_aug.png

PID is included in every filename so DataLoader workers (each its own
process after fork) cannot collide on output paths.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any

import numpy as np
from PIL import Image


_ENV_EVERY = "RLDX_DUMP_AUG_EVERY_N_STEPS"
_ENV_DIR = "RLDX_DUMP_AUG_DIR"


def _to_hwc_uint8(arr: Any) -> np.ndarray:
    """Coerce torch tensor / np.ndarray / PIL.Image into ``(H, W, 3) uint8`` HWC."""
    if hasattr(arr, "numpy"):
        a = arr.numpy()
    elif isinstance(arr, np.ndarray):
        a = arr
    else:
        a = np.asarray(arr)
    return a.astype(np.uint8)


class AugSnapshotter:
    """Thread-safe, env-gated periodic snapshot of (orig, augmented, mask).

    Constructed on the processor; cheap and safe to construct unconditionally.
    Becomes a no-op unless both env vars are set.

    Pickle-safe: the internal lock is dropped via ``__getstate__`` /
    ``__setstate__`` so DataLoader worker spawn / fork can pickle the
    enclosing processor without issue.
    """

    def __init__(self) -> None:
        dump_every_str = os.environ.get(_ENV_EVERY, "")
        dump_dir_str = os.environ.get(_ENV_DIR, "")
        if dump_every_str and dump_dir_str:
            self.enabled = True
            self.dump_every_n = int(dump_every_str)
            self.dump_dir: Path | None = Path(dump_dir_str)
            self.dump_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.enabled = False
            self.dump_every_n = 0
            self.dump_dir = None
        self._counter = 0
        self._lock = threading.Lock()

    # --- pickle hooks (DataLoader fork / spawn safety) -------------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)  # threading.Lock is not picklable
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Each unpickled instance gets a fresh lock; counter resumes from
        # the pickled value (post-fork, per-process counters are normal).
        self._lock = threading.Lock()

    # --- public api ------------------------------------------------------
    def maybe_dump(
        self,
        orig_by_view: dict[str, list[Any]],
        xfm_by_view: dict[str, list[Any]],
    ) -> None:
        """Save sample pairs if the counter hits the ``dump_every_n`` boundary.

        Args:
            orig_by_view: ``{view: [pre-aug PIL or HWC ndarray, ...]}``.
            xfm_by_view:  ``{view: [post-aug HWC tensor or ndarray, ...]}``.
        """
        if not self.enabled or self.dump_dir is None:
            return

        with self._lock:
            self._counter += 1
            current = self._counter
            should_dump = (current % self.dump_every_n) == 0
        if not should_dump:
            return

        pid_tag = f"pid{os.getpid()}"
        for view, originals in orig_by_view.items():
            transformed = xfm_by_view.get(view, [])
            n_samples = min(2, len(originals), len(transformed))
            for k in range(n_samples):
                prefix = self.dump_dir / (f"step_{current:06d}_{pid_tag}_{view}_sample{k}")
                Image.fromarray(_to_hwc_uint8(originals[k])).save(str(prefix) + "_orig.png")
                Image.fromarray(_to_hwc_uint8(transformed[k])).save(str(prefix) + "_aug.png")
