# SPDX-License-Identifier: Apache-2.0

"""Device transfer primitives shared by runtime implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from rldx.planner.tree import tensor_map, tensors


def move_tensors(value: Any, device: torch.device, *, non_blocking: bool = True) -> Any:
    """Move every tensor leaf to ``device``."""
    return tensor_map(lambda tensor: tensor.to(device=device, non_blocking=non_blocking), value)


class _AsyncDeviceTransferIterator:
    def __init__(self, iterator, device: torch.device):
        self._iterator = iterator
        self._device = (
            torch.device("cuda", torch.cuda.current_device())
            if device.type == "cuda" and device.index is None
            else device
        )
        self._stream = torch.cuda.Stream(device=self._device)
        self._next_batch = None
        self._event = None
        self._preload()

    def __iter__(self):
        return self

    def _preload(self) -> None:
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._next_batch = None
            self._event = None
            return

        with torch.cuda.stream(self._stream):
            self._next_batch = move_tensors(batch, self._device)
            if not isinstance(self._next_batch, Mapping):
                raise TypeError(
                    "prefetched input contract requires a Mapping batch, "
                    f"got {type(self._next_batch).__name__}"
                )
            inputs = self._next_batch.get("inputs")
            if not isinstance(inputs, Mapping):
                raise TypeError(
                    "prefetched input contract requires batch.inputs to be a Mapping, "
                    f"got {type(inputs).__name__}"
                )
            self._event = torch.cuda.Event()
            self._event.record(self._stream)

    def __next__(self):
        if self._next_batch is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream(self._device)
        current_stream.wait_event(self._event)
        batch = self._next_batch
        for tensor in tensors(batch):
            tensor.record_stream(current_stream)
        self._preload()
        return batch


class AsyncDeviceTransfer:
    """Transfer one pinned-host batch ahead on a dedicated CUDA stream."""

    def __init__(self, loader, device: torch.device | str):
        self._loader = loader
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError(f"AsyncDeviceTransfer requires a CUDA device, got {self._device}")

    def __iter__(self):
        return _AsyncDeviceTransferIterator(iter(self._loader), self._device)

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name: str):
        return getattr(self._loader, name)
