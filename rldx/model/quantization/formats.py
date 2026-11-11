from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from rldx.configs.execution import QuantFormat


Buffers = dict[str, torch.Tensor]

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP4_MAX = 6.0
NVFP4_BLOCK = 16
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_TIE_ROUNDS_UP = (False, True, False, True, False, True, False)


@dataclass(frozen=True)
class WeightFormat:
    quantize: Callable[[torch.Tensor], Buffers]
    dequantize: Callable[[Any, torch.dtype], torch.Tensor]
    weight: Callable[[Any, torch.dtype], torch.Tensor]


def _channel_scale(weight: torch.Tensor, maximum: float) -> torch.Tensor:
    amax = weight.abs().amax(dim=1, keepdim=True)
    return torch.where(amax == 0, 1.0, amax / maximum)


def _quantize_fp8(weight: torch.Tensor) -> Buffers:
    weight = weight.float()
    scale = _channel_scale(weight, FP8_MAX)
    return {"data": (weight / scale).to(torch.float8_e4m3fn), "scale": scale}


def _dequantize_fp8(layer: Any, dtype: torch.dtype) -> torch.Tensor:
    return layer.data.to(dtype)


def _channel_scaled_weight(layer: Any, dtype: torch.dtype) -> torch.Tensor:
    return (layer.dequantize(torch.float32) * layer.scale).to(dtype)


def _quantize_int8(weight: torch.Tensor) -> Buffers:
    weight = weight.float()
    scale = _channel_scale(weight, 127)
    return {"data": (weight / scale).round().clamp(-127, 127).to(torch.int8), "scale": scale}


def _dequantize_int8(layer: Any, dtype: torch.dtype) -> torch.Tensor:
    return layer.data.to(dtype)


def quantize_rows_fp8(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = flat.abs().amax(dim=1, keepdim=True).float()
    scale = torch.where(amax == 0, 1.0, amax * (1 / FP8_MAX))
    return (flat.float() / scale).to(torch.float8_e4m3fn), scale


def nvfp4_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax().float()
    return torch.where(amax == 0, 1.0, FP8_MAX * FP4_MAX / amax).reshape(1)


def nvfp4_activation_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax(dim=-1).amax().float()
    mantissa, exponent = torch.frexp(amax)
    power = 12 - exponent - (mantissa > 0.65625).int()
    return torch.where(amax == 0, 1.0, torch.ldexp(torch.ones_like(amax), power)).reshape(1)


def encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    magnitude = values.abs().clamp(max=FP4_MAX)
    code = 8 * torch.signbit(values).to(torch.uint8)
    for midpoint, tie_rounds_up in zip(_E2M1_MIDPOINTS, _E2M1_TIE_ROUNDS_UP):
        code = code + (magnitude >= midpoint if tie_rounds_up else magnitude > midpoint)
    return code


def decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    magnitude = torch.tensor(E2M1_VALUES, device=codes.device)[(codes & 7).long()]
    return torch.where(codes >= 8, -magnitude, magnitude)


def unpack(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)


def nvfp4_block_scale(amax: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    block_scale = (scale * (amax * (1 / FP4_MAX))).to(torch.float8_e4m3fn)
    inverse = torch.where(amax == 0, 0.0, 1 / (block_scale.float() * (1 / scale)))
    return block_scale, inverse


def quantize_nvfp4(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = x.shape
    even, odd = x.float().reshape(rows, k // NVFP4_BLOCK, NVFP4_BLOCK // 2, 2).unbind(-1)
    amax = torch.maximum(even.abs(), odd.abs()).amax(-1)
    block_scale, inverse = nvfp4_block_scale(amax, scale)
    inverse = inverse[..., None]
    packed = encode_e2m1(even * inverse) | (encode_e2m1(odd * inverse) << 4)
    return packed.reshape(rows, k // 2), block_scale.view(torch.uint8)


def dequantize_nvfp4(packed: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    values = decode_e2m1(unpack(packed))
    scales = block_scale.view(torch.float8_e4m3fn).float().repeat_interleave(NVFP4_BLOCK, dim=-1)
    return values * scales


def swizzle(block_scale: torch.Tensor) -> torch.Tensor:
    rows, columns = block_scale.shape
    padded = F.pad(block_scale, (0, 0, 0, -rows % 128))
    tiles = padded.reshape(-1, 4, 32, columns // 4, 4).permute(0, 3, 2, 1, 4)
    return tiles.reshape(-1, columns)


def unswizzle(block_scale: torch.Tensor, rows: int) -> torch.Tensor:
    columns = block_scale.shape[-1]
    tiles = block_scale.reshape(-1, columns // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
    return tiles.reshape(-1, columns)[:rows]


def _quantize_nvfp4_weight(weight: torch.Tensor) -> Buffers:
    weight = weight.to(torch.bfloat16)
    scale = nvfp4_global_scale(weight)
    data, block_scale = quantize_nvfp4(weight, scale)
    return {"data": data, "scale": swizzle(block_scale), "global_scale": scale}


def _dequantize_nvfp4_weight(layer: Any, dtype: torch.dtype) -> torch.Tensor:
    block_scale = unswizzle(layer.scale, layer.out_features)
    return dequantize_nvfp4(layer.data, block_scale).to(dtype)


def _nvfp4_weight(layer: Any, dtype: torch.dtype) -> torch.Tensor:
    return (layer.dequantize(torch.float32) / layer.global_scale).to(dtype)


WEIGHT_FORMATS: dict[QuantFormat, WeightFormat] = {
    "fp8_weight": WeightFormat(_quantize_fp8, _dequantize_fp8, _channel_scaled_weight),
    "int8_weight": WeightFormat(_quantize_int8, _dequantize_int8, _channel_scaled_weight),
    "fp8_w8a8": WeightFormat(_quantize_fp8, _dequantize_fp8, _channel_scaled_weight),
    "nvfp4": WeightFormat(_quantize_nvfp4_weight, _dequantize_nvfp4_weight, _nvfp4_weight),
}
