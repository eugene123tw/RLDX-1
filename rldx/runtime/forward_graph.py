"""One CUDA graph per input shape for the whole inference call."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from rldx.dist import rank_zero_print
from rldx.model.inference_noise import noise_source
from rldx.planner import tree


_CONSTANTS = (str, int, float, bool, torch.dtype, torch.device)


def signature(value: Any) -> tuple[Any, list[torch.Tensor]]:
    leaves, spec = tree.flatten(value)
    return (spec, tuple(map(_leaf_key, leaves))), [
        leaf for leaf in leaves if isinstance(leaf, torch.Tensor)
    ]


def _leaf_key(leaf: Any) -> Any:
    if isinstance(leaf, torch.Tensor):
        return ("tensor", tuple(leaf.shape), leaf.dtype, str(leaf.device))
    if isinstance(leaf, _CONSTANTS):
        return ("value", repr(leaf))
    raise TypeError(f"forward graph input {type(leaf).__name__} is neither a tensor nor a constant")


def _clone(value: Any) -> Any:
    return tree.tensor_map(torch.Tensor.clone, value)


@dataclasses.dataclass(frozen=True)
class _Graph:
    graph: torch.cuda.CUDAGraph
    inputs: list[torch.Tensor]
    noise: list[torch.Tensor]
    outputs: Any


class ForwardGraphs:
    def __init__(self, call: Callable[[dict], Any]) -> None:
        self._call = call
        self._graphs: dict[Any, _Graph] = {}

    def run(self, inputs: dict) -> Any:
        key, leaves = signature(inputs)
        if key not in self._graphs:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self._graphs[key] = self._capture(inputs)
            rank_zero_print(f"[forward graph] captured {len(self._graphs)} input shapes")
        entry = self._graphs[key]
        for static, leaf in zip(entry.inputs, leaves, strict=True):
            static.copy_(leaf)
        for buffer in entry.noise:
            torch.randn(buffer.shape, dtype=buffer.dtype, device=buffer.device, out=buffer)
        entry.graph.replay()
        return _clone(entry.outputs)

    def _capture(self, inputs: dict) -> _Graph:
        static = _clone(inputs)
        _, static_inputs = signature(static)
        shapes: list[tuple[tuple[int, ...], torch.dtype, torch.device]] = []

        def record(shape, dtype, device):
            shapes.append((shape, dtype, device))
            return torch.randn(shape, dtype=dtype, device=device)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            with noise_source(record):
                self._call(dict(static))
            self._call(dict(static))
        torch.cuda.current_stream().wait_stream(side)
        noise = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype, device in shapes]
        handed = iter(noise)

        def hand(shape, dtype, device):
            buffer = next(handed)
            if (tuple(buffer.shape), buffer.dtype) != (tuple(shape), dtype):
                raise RuntimeError(
                    f"noise drawn as {shape} {dtype} during capture, recorded {buffer.shape}"
                )
            return buffer

        graph = torch.cuda.CUDAGraph()
        with noise_source(hand), torch.cuda.graph(graph):
            outputs = self._call(dict(static))
        return _Graph(graph, static_inputs, noise, outputs)
