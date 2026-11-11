from dataclasses import dataclass
from functools import cache, wraps
from types import MethodType
from typing import Callable, get_args

import torch
from torch import nn

from rldx.configs.execution import ExecutionBackend
from rldx.model.compile_region import CompileRegion
from rldx.model.modules.action_model import blocks
from rldx.model.modules.action_model.msat import MSAT
from rldx.model.modules.action_model.ops import apply_rotary_emb
from rldx.planner.tree import tensor_map


_MSAT_FORWARD = MSAT.forward
TRAINING_MSAT_REGION = CompileRegion(dynamic=False, autotune=False)


@dataclass(frozen=True)
class DenoiseSteps:
    euler: Callable
    guided_velocity: Callable


@dataclass(frozen=True)
class ActionModelExecutionKernels:
    msat_forward: Callable
    denoise_steps: DenoiseSteps | None = None


def _normalize_compile_input_layout(value):
    if not isinstance(value, torch.Tensor):
        return value
    value = value.contiguous()
    # contiguous() ignores strides on singleton dimensions. A B=1 cog-token
    # slice can therefore retain the source prompt length in stride(0), making
    # Dynamo specialize on that length even though the input shape is fixed.
    expected = []
    stride = 1
    for size in reversed(value.shape):
        expected.append(stride)
        stride *= max(size, 1)
    if value.stride() != tuple(reversed(expected)):
        return value.clone(memory_format=torch.contiguous_format)
    return value


def _normalize_compile_inputs(value):
    return tensor_map(_normalize_compile_input_layout, value)


def _compile_denoise_step(step: Callable, region: CompileRegion) -> Callable:
    compiled_step = region.compile(step)

    @wraps(step)
    def run(module, *args):
        return compiled_step(module, *map(_normalize_compile_inputs, args))

    return run


def _compile_msat_forward(region: CompileRegion) -> Callable:
    compiled_forward = region.compile(_MSAT_FORWARD)

    @wraps(_MSAT_FORWARD)
    def forward(module, *args, **kwargs):
        args = tuple(_normalize_compile_input_layout(value) for value in args)
        kwargs = {key: _normalize_compile_input_layout(value) for key, value in kwargs.items()}
        return compiled_forward(module, *args, **kwargs)

    return forward


@cache
def build_training_kernels(backend: ExecutionBackend) -> ActionModelExecutionKernels:
    if _compiles(backend):
        return ActionModelExecutionKernels(msat_forward=_compile_msat_forward(TRAINING_MSAT_REGION))
    return ActionModelExecutionKernels(msat_forward=_MSAT_FORWARD)


@cache
def build_serving_kernels(
    backend: ExecutionBackend, steps: DenoiseSteps
) -> ActionModelExecutionKernels:
    if not _compiles(backend):
        return ActionModelExecutionKernels(msat_forward=_MSAT_FORWARD)
    region = CompileRegion(dynamic=None, autotune=False)
    return ActionModelExecutionKernels(
        msat_forward=_MSAT_FORWARD,
        denoise_steps=DenoiseSteps(
            _compile_denoise_step(steps.euler, region),
            _compile_denoise_step(steps.guided_velocity, region),
        ),
    )


def _compiles(backend: ExecutionBackend) -> bool:
    if backend not in get_args(ExecutionBackend):
        raise ValueError(f"unsupported action-model execution backend: {backend!r}")
    return backend == "compile"


def install_action_model_execution_kernels(
    model: nn.Module, kernels: ActionModelExecutionKernels
) -> None:
    block_types = (
        blocks.SingleStreamBlock,
        blocks.DoubleStreamBlock,
        blocks.ExpandedDoubleStreamBlock,
        blocks.ExpandedSingleStreamBlock,
    )
    for module in model.modules():
        if isinstance(module, MSAT):
            module.forward = MethodType(kernels.msat_forward, module)
        elif isinstance(module, block_types):
            module._apply_rope = apply_rotary_emb
