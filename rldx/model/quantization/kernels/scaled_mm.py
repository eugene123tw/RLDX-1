import torch
from torch import nn

from rldx.model.quantization.formats import quantize_rows_fp8
from rldx.model.quantization.linear import GemmSelection, LinearKernel, QuantizedLinear


ACCEPTED: set[tuple[torch.device, int, int]] = set()


def _plan(linear: nn.Linear, gemm: GemmSelection) -> None:
    key = (linear.weight.device, linear.in_features, linear.out_features)
    if key in ACCEPTED:
        return None
    device, k, n = key
    probe = torch.zeros(1, k, device=device, dtype=torch.float8_e4m3fn)
    weight = torch.zeros(n, k, device=device, dtype=torch.float8_e4m3fn)
    try:
        torch._scaled_mm(
            probe,
            weight.t(),
            scale_a=torch.ones(1, 1, device=device),
            scale_b=torch.ones(1, n, device=device),
            out_dtype=torch.bfloat16,
        )
    except RuntimeError as error:
        raise ValueError(
            f"_scaled_mm rejects in_features={k}, out_features={n}: {error}"
        ) from error
    ACCEPTED.add(key)
    return None


def _run(x: torch.Tensor, layer: QuantizedLinear) -> torch.Tensor:
    quantized, scale = quantize_rows_fp8(x.reshape(-1, x.shape[-1]))
    output = torch._scaled_mm(
        quantized,
        layer.data.t(),
        scale_a=scale,
        scale_b=layer.scale.T,
        bias=layer.bias,
        out_dtype=torch.bfloat16,
    )
    return output.to(x.dtype).reshape(*x.shape[:-1], -1)


KERNEL = LinearKernel(_run, plan=_plan)
