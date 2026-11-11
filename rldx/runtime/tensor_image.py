from functools import cache
from typing import Callable

import torch

from rldx.configs.execution import ExecutionBackend
from rldx.model.compile_region import CompileRegion
from rldx.planner.input.tensor_image import TensorImageSpec


@cache
def build_pixel_values_kernel(spec: TensorImageSpec, backend: ExecutionBackend) -> Callable:
    match backend:
        case "eager":
            return spec.pixel_values
        case "compile":
            return CompileRegion(dynamic=None, autotune=False).compile(spec.pixel_values)
    raise ValueError(f"unsupported tensor image backend: {backend!r}")


class TensorImages:
    def __init__(
        self, spec: TensorImageSpec, device: torch.device, backend: ExecutionBackend = "eager"
    ):
        self.spec = spec
        self._lookup = spec.lookup(device)
        self.pixel_values = build_pixel_values_kernel(spec, backend)

    def to_pixel_values(self, inputs: dict, dtype: torch.dtype) -> None:
        if "raw_images" not in inputs or "raw_image_order" not in inputs:
            raise ValueError("tensor images require raw_images and raw_image_order")
        inputs["pixel_values"] = self.pixel_values(
            self._lookup, inputs.pop("raw_images"), inputs.pop("raw_image_order"), dtype
        )
