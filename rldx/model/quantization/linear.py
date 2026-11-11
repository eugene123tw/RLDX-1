from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Callable

import torch
from torch import nn

from rldx.configs.execution import QuantFormat
from rldx.model.quantization.formats import WEIGHT_FORMATS, Buffers
from rldx.model.quantization.reference import dequantized_linear, w8a8_reference


@dataclass(frozen=True)
class GemmSelection:
    backend: str = "auto"
    max_rows: int | None = None

    def __post_init__(self) -> None:
        if self.max_rows is not None and self.max_rows < 1:
            raise ValueError(f"invalid max_rows: {self.max_rows!r}")

    @property
    def label(self) -> str:
        return self.backend if self.max_rows is None else f"{self.backend}:{self.max_rows}"


@dataclass(frozen=True)
class LinearKernel:
    run: Callable[[torch.Tensor, "QuantizedLinear"], torch.Tensor]
    quantize: Callable[[torch.Tensor], Buffers] | None = None
    plan: Callable[[nn.Linear, GemmSelection], str] | None = None


TORCH_KERNELS: dict[QuantFormat, LinearKernel] = {
    "fp8_weight": LinearKernel(dequantized_linear),
    "int8_weight": LinearKernel(dequantized_linear),
    "fp8_w8a8": LinearKernel(w8a8_reference),
}
CUDA_KERNEL_MODULES: dict[QuantFormat, str] = {"fp8_w8a8": "scaled_mm", "nvfp4": "nvfp4"}


def _kernel_module(format_name: QuantFormat) -> ModuleType:
    name = f"rldx.model.quantization.kernels.{CUDA_KERNEL_MODULES[format_name]}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            raise
        raise ValueError(f"{format_name} kernel requires the {error.name} package") from error


def kernel_for(format_name: QuantFormat, device: torch.device) -> LinearKernel:
    if device.type == "cuda" and format_name in CUDA_KERNEL_MODULES:
        return _kernel_module(format_name).KERNEL
    if format_name not in TORCH_KERNELS:
        raise ValueError(f"{format_name} has no kernel for {device.type}")
    return TORCH_KERNELS[format_name]


def planned_kernel(
    format_name: QuantFormat, linear: nn.Linear, site: str, gemm: GemmSelection = GemmSelection()
) -> tuple[LinearKernel, str | None]:
    try:
        kernel = kernel_for(format_name, linear.weight.device)
        plan = None if kernel.plan is None else kernel.plan(linear, gemm)
    except ValueError as error:
        raise ValueError(f"{site}: {error}") from error
    return kernel, plan


class _InputGradient(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, layer: "QuantizedLinear") -> torch.Tensor:
        return layer.kernel.run(x, layer)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.layer = inputs[1]

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        if ctx.layer.transposed is None:
            raise RuntimeError(f"{ctx.layer.site} has no input gradient; prepare it first")
        return ctx.layer.transposed(grad), None


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        format_name: QuantFormat,
        site: str = "linear",
        gemm: GemmSelection = GemmSelection(),
        library_quantize: bool = False,
        activation_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.format = format_name
        self.site = site
        self.gemm = gemm
        self.library_quantize = library_quantize
        self.kernel, self.plan = planned_kernel(format_name, linear, site, gemm)
        quantize = self.kernel.quantize or WEIGHT_FORMATS[format_name].quantize
        for name, tensor in quantize(linear.weight.detach()).items():
            self.register_buffer(name, tensor)
        self.register_buffer("bias", linear.bias.detach() if linear.bias is not None else None)
        self.register_buffer(
            "activation_scale",
            None
            if activation_scale is None
            else torch.full((1,), activation_scale, device=linear.weight.device),
        )
        self.register_module("transposed", None)

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        return WEIGHT_FORMATS[self.format].dequantize(self, dtype)

    def weight(self, dtype: torch.dtype) -> torch.Tensor:
        return WEIGHT_FORMATS[self.format].weight(self, dtype)

    def prepare_input_gradient(self) -> None:
        linear = nn.Linear(self.out_features, self.in_features, bias=False, device="meta")
        linear.weight = nn.Parameter(self.weight(torch.float32).t().contiguous(), False)
        self.transposed = QuantizedLinear(
            linear, self.format, f"{self.site} input gradient", self.gemm, self.library_quantize
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _InputGradient.apply(x, self)
