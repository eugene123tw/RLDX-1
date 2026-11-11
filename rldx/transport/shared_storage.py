# SPDX-License-Identifier: Apache-2.0

"""Batches whose tensors share one storage, so a loader worker hands each over with one file descriptor."""

from __future__ import annotations

from typing import Any, Callable

import torch

from rldx.planner.tree import flatten


ALIGNMENT = 64


class SharedStorageCollate:
    def __init__(self, collate: Callable[[list], Any]):
        self.collate = collate

    def __call__(self, samples: list) -> Any:
        return share_one_storage(self.collate(samples))


def share_one_storage(batch: Any) -> Any:
    leaves, spec = flatten(batch)
    unique = {id(leaf): leaf for leaf in leaves if isinstance(leaf, torch.Tensor)}
    offsets, size = {}, 0
    for key, tensor in unique.items():
        offsets[key] = size
        size += -(-tensor.nbytes // ALIGNMENT) * ALIGNMENT
    buffer = torch.empty(size, dtype=torch.uint8)
    shared = {
        key: buffer[offsets[key] : offsets[key] + tensor.nbytes]
        .view(tensor.dtype)
        .view(tensor.shape)
        .copy_(tensor)
        for key, tensor in unique.items()
    }
    return spec.unflatten(
        [shared[id(leaf)] if isinstance(leaf, torch.Tensor) else leaf for leaf in leaves]
    )
