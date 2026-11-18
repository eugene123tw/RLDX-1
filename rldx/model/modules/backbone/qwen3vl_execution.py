"""Instance-owned execution kernels for the in-tree Qwen3-VL model."""

from dataclasses import dataclass
from functools import cache, wraps
from types import MethodType
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from rldx.configs.execution import ExecutionBackend
from rldx.model.compile_region import CompileRegion
from rldx.model.modules.backbone import modeling_qwen3_vl as modeling
from rldx.planner.input.text_attention import TextAttentionLayout


TextAttend = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, TextAttentionLayout, float], torch.Tensor
]


_VISION_BLOCK_FORWARD = modeling.Qwen3VLVisionBlock.forward
_TEXT_RMSNORM_FORWARD = modeling.Qwen3VLTextRMSNorm.forward
_TEXT_QK_RMSNORM_FORWARD = modeling.Qwen3VLTextQKRMSNorm.forward
_TEXT_LAYER_VARLEN_FORWARD = modeling.Qwen3VLTextDecoderLayer.forward_varlen
_SMALL_OP_REGION = CompileRegion(dynamic=True, autotune=True)


def _compile_vision_block():
    return CompileRegion(dynamic=False, autotune=True).compile(_VISION_BLOCK_FORWARD)


def _flash_vision_forward(compiled_forward):
    @wraps(_VISION_BLOCK_FORWARD)
    def forward(
        module,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
        *,
        max_seqlen: int,
        **kwargs,
    ):
        torch._dynamo.mark_dynamic(hidden_states, 0)
        torch._dynamo.mark_dynamic(cu_seqlens, 0)
        if rotary_pos_emb is not None:
            torch._dynamo.mark_dynamic(rotary_pos_emb, 0)
        if position_embeddings is not None:
            for embedding in position_embeddings:
                torch._dynamo.mark_dynamic(embedding, 0)
        return compiled_forward(
            module,
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
            **kwargs,
        )

    return forward


def _chunked_vision_forward(compiled_forward):
    @wraps(_VISION_BLOCK_FORWARD)
    def forward(
        module,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
        *,
        max_seqlen: int,
        sequence_lengths=None,
        **kwargs,
    ):
        if sequence_lengths is None:
            sequence_lengths = tuple(cu_seqlens.diff().tolist())
        return compiled_forward(
            module,
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
            **kwargs,
        )

    return forward


@dataclass(frozen=True)
class Qwen3VLExecutionKernels:
    """Regional kernels installed on one Qwen3-VL model instance."""

    text_rmsnorm_forward: Callable
    text_qk_rmsnorm_forward: Callable
    vision_block_forward: Callable
    text_rope: Callable
    text_layer_forward: Callable | None = None


@cache
def build_qwen3vl_execution_kernels(
    backend: ExecutionBackend,
    planned_text_layers: bool = False,
    attention: str = "flash_attention_2",
) -> Qwen3VLExecutionKernels:
    """Build immutable eager or compiled regional kernels once per process."""
    if backend not in ("eager", "compile"):
        raise ValueError(f"unsupported Qwen3-VL execution backend: {backend!r}")
    text_layer_forward = (
        _planned_text_layer_forward(backend, attention) if planned_text_layers else None
    )
    if backend == "eager":
        return Qwen3VLExecutionKernels(
            text_rmsnorm_forward=_TEXT_RMSNORM_FORWARD,
            text_qk_rmsnorm_forward=_TEXT_QK_RMSNORM_FORWARD,
            vision_block_forward=_VISION_BLOCK_FORWARD,
            text_rope=modeling.apply_rotary_pos_emb,
            text_layer_forward=text_layer_forward,
        )
    return Qwen3VLExecutionKernels(
        text_rmsnorm_forward=_SMALL_OP_REGION.compile(_TEXT_RMSNORM_FORWARD),
        text_qk_rmsnorm_forward=_SMALL_OP_REGION.compile(_TEXT_QK_RMSNORM_FORWARD),
        vision_block_forward=_compile_vision_block(),
        text_rope=_SMALL_OP_REGION.compile(modeling.apply_rotary_pos_emb),
        text_layer_forward=text_layer_forward,
    )


def _planned_text_layer_forward(backend: ExecutionBackend, attention: str) -> Callable:
    attend = _text_attention(attention, backend)
    region = CompileRegion(dynamic=None, autotune=True)
    layer_forward = (
        _TEXT_LAYER_VARLEN_FORWARD
        if backend == "eager"
        else _token_dynamic(region.compile(_TEXT_LAYER_VARLEN_FORWARD))
    )

    @wraps(_TEXT_LAYER_VARLEN_FORWARD)
    def forward(layer, hidden_states, *, position_embeddings, attention_layout, **kwargs):
        return layer_forward(layer, hidden_states, position_embeddings, attention_layout, attend)

    return forward


def _token_dynamic(compiled_forward: Callable) -> Callable:
    @wraps(_TEXT_LAYER_VARLEN_FORWARD)
    def forward(layer, hidden_states, position_embeddings, attention_layout, attend):
        torch._dynamo.mark_dynamic(hidden_states, 1)
        for embedding in position_embeddings:
            torch._dynamo.mark_dynamic(embedding, 1)
        return compiled_forward(layer, hidden_states, position_embeddings, attention_layout, attend)

    return forward


def _text_attention(attention: str, backend: ExecutionBackend) -> TextAttend:
    match attention:
        case "flash_attention_2":
            return _flash_varlen_attend(backend)
        case "sdpa":
            return _sdpa_attend
    raise ValueError(f"inference text layers run flash_attention_2 or sdpa, not {attention!r}")


def _flash_varlen_attend(backend: ExecutionBackend) -> TextAttend:
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as error:
        raise ValueError("inference text attention needs the flash-attn package") from error
    if backend == "compile" and not hasattr(torch.ops.flash_attn, "_flash_attn_varlen_forward"):
        raise ValueError(
            "flash-attn does not register its varlen kernel as a torch custom op; "
            "the text layer cannot compile as one region"
        )

    def attend(query, key, value, layout, scale):
        batch, _, length, _ = query.shape
        attended = flash_attn_varlen_func(
            *(states.transpose(1, 2).flatten(0, 1) for states in (query, key, value)),
            cu_seqlens_q=layout.cu_seqlens,
            cu_seqlens_k=layout.cu_seqlens,
            max_seqlen_q=length,
            max_seqlen_k=length,
            softmax_scale=scale,
            causal=True,
        )
        return attended.unflatten(0, (batch, length))

    return attend


def _sdpa_attend(query, key, value, layout, scale):
    batch, _, length, _ = query.shape
    positions = torch.arange(batch * length, device=query.device)
    segment = (layout.cu_seqlens <= positions[:, None]).sum(-1).view(batch, 1, length)
    causal = torch.ones(length, length, dtype=torch.bool, device=query.device).tril()
    # Same-segment causal mask: token i attends to j iff both are in one sequence and j <= i.
    # transpose, not .mT: OpenVINO has no aten::mT conversion rule.
    mask = (segment.transpose(-1, -2) == segment).unsqueeze(1) & causal
    attended = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=scale, enable_gqa=True
    )
    return attended.transpose(1, 2)


def install_qwen3vl_execution_kernels(model: nn.Module, kernels: Qwen3VLExecutionKernels) -> None:
    """Install regional kernels without mutating classes or module globals."""
    layer_forward = kernels.text_layer_forward
    inside_text_layers = set()
    if layer_forward is not None:
        _require_inference_text_layers(model)
        inside_text_layers = _text_layer_members(model)
    for module in model.modules():
        if id(module) in inside_text_layers:
            _install_class_forward(module)
        elif isinstance(module, modeling.Qwen3VLTextDecoderLayer):
            _install_layer_forward(module, layer_forward)
        elif isinstance(module, modeling.Qwen3VLTextQKRMSNorm):
            module.forward = MethodType(kernels.text_qk_rmsnorm_forward, module)
        elif isinstance(module, modeling.Qwen3VLTextRMSNorm):
            module.forward = MethodType(kernels.text_rmsnorm_forward, module)
        elif isinstance(module, modeling.Qwen3VLVisionBlock):
            forward = kernels.vision_block_forward
            if forward is not _VISION_BLOCK_FORWARD:
                if module.attn.config._attn_implementation == "flash_attention_2":
                    from transformers.modeling_flash_attention_utils import (
                        lazy_import_flash_attention,
                    )

                    lazy_import_flash_attention("flash_attention_2")
                    forward = _flash_vision_forward(forward)
                else:
                    forward = _chunked_vision_forward(forward)
            module.forward = MethodType(forward, module)
        elif isinstance(module, modeling.Qwen3VLVisionAttention):
            module._apply_rope = modeling.apply_rotary_pos_emb_vision
        elif isinstance(module, modeling.Qwen3VLTextAttention):
            module._apply_rope = kernels.text_rope


def _require_inference_text_layers(model: nn.Module) -> None:
    for module in model.modules():
        if not isinstance(module, modeling.Qwen3VLTextDecoderLayer):
            continue
        if not module.projections_fused:
            raise ValueError(
                "inference text layers run fused q/k/v and gate/up projections; "
                "fuse_projections() must run before the kernels are installed"
            )


def _text_layer_members(model: nn.Module) -> set[int]:
    return {
        id(member)
        for layer in model.modules()
        if isinstance(layer, modeling.Qwen3VLTextDecoderLayer)
        for member in layer.modules()
        if member is not layer
    }


def _install_class_forward(module: nn.Module) -> None:
    if isinstance(module, modeling.Qwen3VLTextRMSNorm):
        module.__dict__.pop("forward", None)
    elif isinstance(module, modeling.Qwen3VLTextAttention):
        module._apply_rope = modeling.apply_rotary_pos_emb


def _install_layer_forward(layer: nn.Module, forward: Callable | None) -> None:
    if forward is None:
        layer.__dict__.pop("forward", None)
        return
    layer.forward = MethodType(forward, layer)
