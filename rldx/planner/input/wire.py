"""Parse host metadata and tensor descriptions without reading device values."""

from collections.abc import Mapping
from typing import Any

import torch


def parse_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def parse_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parse_tensor(
    value: Any,
    name: str,
    *,
    shape: tuple[int | None, ...],
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected for actual, expected in zip(value.shape, shape)
    ):
        raise ValueError(f"{name} must have shape {shape}")
    return value
