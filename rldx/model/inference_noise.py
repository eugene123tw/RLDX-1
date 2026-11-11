from contextlib import contextmanager
from typing import Callable, Iterator

import torch


NoiseSource = Callable[[tuple[int, ...], torch.dtype, torch.device], torch.Tensor]


def _sampled(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, device=device)


class _Source:
    current: NoiseSource = _sampled


def initial_noise(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return _Source.current(shape, dtype, device)


@contextmanager
def noise_source(source: NoiseSource) -> Iterator[None]:
    previous, _Source.current = _Source.current, source
    try:
        yield
    finally:
        _Source.current = previous
