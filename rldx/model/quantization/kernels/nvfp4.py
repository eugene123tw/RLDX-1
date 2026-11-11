from bisect import bisect_left
from dataclasses import dataclass, replace
import math
from typing import Any, get_args, get_type_hints

from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets
import torch
from torch import nn

from rldx.model.quantization.formats import (
    NVFP4_BLOCK,
    Buffers,
    nvfp4_activation_scale,
    nvfp4_global_scale,
    quantize_nvfp4,
    swizzle,
)
from rldx.model.quantization.linear import GemmSelection, LinearKernel, QuantizedLinear


LIBRARY_DEFAULT_TACTIC = -1

PlanKey = tuple[torch.device, str]


def quantize_activation(
    flat: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    codes, block_scale = quantize_nvfp4(flat, scale)
    return codes, swizzle(block_scale)


def _library_quantize(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return nvfp4_quantize(
        x.contiguous(), scale, sfLayout=SfLayout.layout_128x4, do_shuffle=False, enable_pdl=False
    )


def _probe_inputs(device: torch.device, rows: int, k: int, n: int):
    unit = torch.ones(1, device=device)
    codes, block_scale = quantize_activation(
        torch.randn(rows, k, device=device).to(torch.bfloat16), unit
    )
    data, scale = _library_quantize(torch.zeros(n, k, device=device, dtype=torch.bfloat16), unit)
    return codes, block_scale, unit, data.view(torch.uint8), scale.view(torch.uint8)


@dataclass(frozen=True)
class Choice:
    runner: Any
    tactic: int
    workspace: torch.Tensor

    @classmethod
    def from_library(
        cls, device: torch.device, k: int, n: int, backend: str, rows: int = 1
    ) -> "Choice":
        codes, block_scale, unit, data, scale = _probe_inputs(device, rows, k, n)
        tuner = AutoTuner.get()
        choose_one = tuner.choose_one
        choices = []

        def record(custom_op, runners, tuning_config, inputs, **kwargs):
            runner, tactic = choose_one(custom_op, runners, tuning_config, inputs, **kwargs)
            choices.append(cls(runner, tactic, inputs[-1]))
            return runner, tactic

        tuner.choose_one = record
        try:
            mm_fp4(
                codes,
                data.T,
                block_scale,
                scale.T,
                unit,
                torch.bfloat16,
                block_size=NVFP4_BLOCK,
                use_8x4_sf_layout=False,
                backend=backend,
                use_nvfp4=True,
                enable_pdl=True,
            )
        finally:
            del tuner.choose_one
        (choice,) = choices
        return choice


@dataclass(frozen=True)
class Plan:
    row_bounds: tuple[float, ...]
    choices: tuple[Choice, ...]

    @classmethod
    def for_gemm(cls, device: torch.device, k: int, n: int, gemm: GemmSelection) -> "Plan":
        if gemm.max_rows is None:
            choice = Choice.from_library(device, k, n, gemm.backend)
            return cls((math.inf,), (replace(choice, tactic=LIBRARY_DEFAULT_TACTIC),))
        buckets = get_hybrid_num_tokens_buckets(gemm.max_rows)
        bounds = (*(rows for rows in buckets if rows < gemm.max_rows), gemm.max_rows)
        with autotune(True):
            return cls(
                bounds,
                tuple(Choice.from_library(device, k, n, gemm.backend, rows) for rows in bounds),
            )

    def __call__(
        self,
        codes: torch.Tensor,
        block_scale: torch.Tensor,
        alpha: torch.Tensor,
        data: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        rows = codes.shape[0]
        index = bisect_left(self.row_bounds, rows)
        if index == len(self.row_bounds):
            raise ValueError(f"{rows} rows exceed the tuned GEMM range {self.row_bounds[-1]}")
        choice = self.choices[index]
        output = alpha.new_empty((rows, data.shape[0]), dtype=torch.bfloat16)
        inputs = [
            codes,
            data.T,
            block_scale,
            scale.T,
            alpha,
            torch.bfloat16,
            output,
            NVFP4_BLOCK,
            True,
            choice.workspace,
        ]
        return choice.runner(inputs=inputs, tactic=choice.tactic)


PLANS: dict[PlanKey, Plan] = {}


def _library_backends() -> tuple[str, ...]:
    return get_args(get_type_hints(mm_fp4)["backend"])


def _plan(linear: nn.Linear, gemm: GemmSelection) -> str:
    backends = _library_backends()
    if gemm.backend not in backends:
        raise ValueError(f"nvfp4 backend {gemm.backend!r} is not one of {backends}")
    k, n = linear.in_features, linear.out_features
    name = f"{gemm.label}@{k}x{n}"
    key = (linear.weight.device, name)
    if key not in PLANS:
        try:
            PLANS[key] = Plan.for_gemm(linear.weight.device, k, n, gemm)
        except Exception as error:
            raise ValueError(
                f"flashinfer mm_fp4 ({gemm.label}) rejects in_features={k}, "
                f"out_features={n}: {error}"
            ) from error
    return name


@torch.library.custom_op("rldx::nvfp4_gemm", mutates_args=())
def nvfp4_gemm(
    codes: torch.Tensor,
    block_scale: torch.Tensor,
    alpha: torch.Tensor,
    data: torch.Tensor,
    scale: torch.Tensor,
    plan: str,
) -> torch.Tensor:
    return PLANS[codes.device, plan](codes, block_scale, alpha, data, scale)


@nvfp4_gemm.register_fake
def _nvfp4_gemm_fake(
    codes: torch.Tensor,
    block_scale: torch.Tensor,
    alpha: torch.Tensor,
    data: torch.Tensor,
    scale: torch.Tensor,
    plan: str,
) -> torch.Tensor:
    return codes.new_empty((codes.shape[0], data.shape[0]), dtype=torch.bfloat16)


def _quantize_weight(weight: torch.Tensor) -> Buffers:
    weight = weight.to(torch.bfloat16)
    global_scale = nvfp4_global_scale(weight)
    data, block_scale = _library_quantize(weight, global_scale)
    return {
        "data": data.view(torch.uint8),
        "scale": block_scale.view(torch.uint8),
        "global_scale": global_scale,
        "inverse_global_scale": 1 / global_scale,
    }


@torch.library.custom_op("rldx::nvfp4_library_quantize", mutates_args=())
def library_quantize(flat: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    codes, block_scale = _library_quantize(flat, scale)
    columns = flat.shape[1] // NVFP4_BLOCK
    return codes.view(torch.uint8), block_scale.view(torch.uint8).reshape(-1, columns)


@library_quantize.register_fake
def _library_quantize_fake(
    flat: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = flat.shape
    padded = -(-rows // 128) * 128
    return (
        flat.new_empty((rows, k // 2), dtype=torch.uint8),
        flat.new_empty((padded, k // NVFP4_BLOCK), dtype=torch.uint8),
    )


def _run(x: torch.Tensor, layer: QuantizedLinear) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
    activation_scale = (
        nvfp4_activation_scale(flat) if layer.activation_scale is None else layer.activation_scale
    )
    quantize = library_quantize if layer.library_quantize else quantize_activation
    codes, block_scale = quantize(flat, activation_scale)
    alpha = layer.inverse_global_scale / activation_scale
    output = nvfp4_gemm(codes, block_scale, alpha, layer.data, layer.scale, layer.plan)
    if layer.bias is not None:
        output = output + layer.bias
    return output.to(x.dtype).reshape(*x.shape[:-1], -1)


KERNEL = LinearKernel(_run, _quantize_weight, _plan)
