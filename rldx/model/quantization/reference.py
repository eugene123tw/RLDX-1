from typing import Any

import torch
import torch.nn.functional as F

from rldx.model.quantization.formats import (
    dequantize_nvfp4,
    nvfp4_activation_scale,
    quantize_nvfp4,
    quantize_rows_fp8,
)


def dequantized_linear(x: torch.Tensor, layer: Any) -> torch.Tensor:
    output = F.linear(x, layer.dequantize(x.dtype)).float() * layer.scale.T
    if layer.bias is not None:
        output = output + layer.bias.float()
    return output.to(x.dtype)


def w8a8_reference(x: torch.Tensor, layer: Any) -> torch.Tensor:
    quant_x, scale_x = quantize_rows_fp8(x.reshape(-1, x.shape[-1]))
    quant_w = layer.data.float()
    output = F.linear(quant_x.float(), quant_w) * scale_x * layer.scale.T
    if layer.bias is not None:
        output = output + layer.bias.float()
    return output.to(x.dtype).reshape(*x.shape[:-1], -1)


def nvfp4_reference(x: torch.Tensor, layer: Any) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
    activation_scale = nvfp4_activation_scale(flat)
    activation = dequantize_nvfp4(*quantize_nvfp4(flat, activation_scale))
    output = F.linear(activation, layer.dequantize(torch.float32))
    output = (output / (activation_scale * layer.global_scale.float())).to(torch.bfloat16)
    if layer.bias is not None:
        output = output + layer.bias
    return output.to(x.dtype).reshape(*x.shape[:-1], -1)
