from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class NormStats:
    """One joint group's statistics: ``low``/``high`` map onto [-1, 1], ``mean``/``std`` z-score."""

    low: np.ndarray
    high: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    dim: int

    @classmethod
    def from_stats(cls, stats: dict, low: str, high: str, dim: int | None = None) -> NormStats:
        low_values = np.array(stats[low])
        return cls(
            low=low_values,
            high=np.array(stats[high]),
            mean=np.array(stats["mean"]),
            std=np.array(stats["std"]),
            dim=low_values.shape[0] if dim is None else dim,
        )

    @property
    def per_step(self) -> bool:
        return any(v.ndim >= 2 for v in (self.low, self.high, self.mean, self.std))

    def at(self, position) -> NormStats:
        def pick(values):
            return values[position] if values.ndim >= 2 else values

        return replace(
            self,
            low=pick(self.low),
            high=pick(self.high),
            mean=pick(self.mean),
            std=pick(self.std),
        )
