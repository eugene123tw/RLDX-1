# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


class SessionRegistry:
    """Each session's memory between calls. Sessions end when the caller resets them."""

    def __init__(self) -> None:
        self._memory: dict[str, torch.Tensor] = {}

    def load_memory_batch(self, sids: list[str]) -> tuple[torch.Tensor | None, tuple[bool, ...]]:
        """Stack each session's memory; a session without one starts cold from zeros."""
        cached = [self._memory.get(sid) for sid in sids]
        cold = tuple(memory is None for memory in cached)
        ref = next((memory for memory in cached if memory is not None), None)
        if ref is None:
            return None, cold
        return torch.stack([torch.zeros_like(ref) if m is None else m for m in cached]), cold

    def save_memory_batch(self, sids: list[str], stacked: torch.Tensor) -> None:
        for sid, memory in zip(sids, stacked, strict=True):
            self._memory[sid] = memory.detach().clone()

    def reset(self, sids: list[str]) -> list[str]:
        return [sid for sid in sids if self._memory.pop(sid, None) is not None]

    def clear(self) -> list[str]:
        cleared = list(self._memory)
        self._memory.clear()
        return cleared

    def __len__(self) -> int:
        return len(self._memory)
