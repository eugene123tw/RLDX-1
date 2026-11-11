import torch
import torch.nn as nn

from rldx.planner.input.text_attention import TextAttentionPlan
from rldx.planner.input.token_compression import TokenCompressionPlan


def _masked_scatter_with_cat_promotion(
    destination: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """Scatter with the dtype promotion performed by the legacy ``torch.cat`` path."""
    output_dtype = torch.promote_types(destination.dtype, source.dtype)
    return destination.to(output_dtype).masked_scatter(mask, source.to(output_dtype))


class LayerWrapper(nn.Module):
    def __init__(
        self, layer, layer_idx, internal_projection=4, img_pattern=[151652], motion_token=0
    ):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.internal_projection = internal_projection
        self.motion_token = motion_token
        self.img_pattern = img_pattern
        assert motion_token == 1

    def forward(self, hidden_states, input_ids, *args, **kwargs):
        plan: TokenCompressionPlan | None = kwargs.get("token_compression_plan")
        text_attention: TextAttentionPlan | None = kwargs.pop("text_attention", None)
        if text_attention is not None:
            kwargs["attention_layout"] = (
                text_attention.before_compression
                if self.layer_idx < self.internal_projection
                else text_attention.after_compression
            )

        bsz, seq_len, dim = hidden_states.shape

        is_incremental = (
            "cache_position" in kwargs and kwargs["cache_position"] is not None and seq_len == 1
        )
        if self.layer_idx == self.internal_projection and not is_incremental:
            if plan is None:
                raise ValueError("internal projection requires a host token compression plan")
            if kwargs.get("motion_drop_info") is not None:
                raise ValueError("token compression plan does not support moss token insertion")
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is not None and attention_mask.ndim == 4:
                raise ValueError("token compression plan does not support 4D attention masks")
            if kwargs.get("attention_mask_2") is not None:
                raise ValueError("token compression plan does not support attention_mask_2")
            if plan.extended_len != seq_len:
                raise ValueError(
                    "token compression plan was built for sequence length "
                    f"{plan.extended_len}, forward received {seq_len}"
                )
            gather_index = plan.gather_index
            pad_mask = plan.pad_mask
            motion_slot = plan.motion_slot
            begin_idx = plan.begin.unsqueeze(1)
            end_idx = plan.end.unsqueeze(1)
            if gather_index.ndim != 2 or gather_index.shape[0] != bsz:
                raise ValueError("invalid token compression gather_index shape")
            if pad_mask.shape != gather_index.shape or motion_slot.shape != gather_index.shape:
                raise ValueError("token compression masks must match gather_index")
            token_indices = torch.arange(seq_len, device=hidden_states.device).view(1, -1)
            drop_mask = (token_indices >= begin_idx) & (token_indices < end_idx)
            motion_token = (hidden_states * drop_mask.unsqueeze(-1)).sum(dim=1) / drop_mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
            gathered = hidden_states.gather(1, gather_index.unsqueeze(-1).expand(-1, -1, dim))
            gathered = gathered.masked_fill(pad_mask.unsqueeze(-1), 0)
            source_order = plan.compress.to(torch.int64).argsort(descending=True, stable=True)
            motion_source = motion_token.gather(0, source_order.unsqueeze(1).expand(-1, dim))
            hidden_states = _masked_scatter_with_cat_promotion(
                gathered, motion_slot.unsqueeze(-1).expand(-1, -1, dim), motion_source
            )
            compressed_attention_mask = kwargs.get("compressed_attention_mask")
            if compressed_attention_mask is None:
                raise ValueError("token compression plan is missing compressed metadata")
            if compressed_attention_mask.shape[-1] != gather_index.shape[1]:
                raise ValueError("compressed attention mask has the wrong sequence length")
            # The planner's compressed mask decides which slots are visible; an
            # all-valid batch reaches here without a mask and stays on the dense path.
            if attention_mask is not None:
                kwargs["attention_mask"] = compressed_attention_mask.to(attention_mask.dtype)
            if kwargs.get("position_embeddings") is not None:
                cos, sin = kwargs["position_embeddings"]
                embedding_index = gather_index.unsqueeze(-1).expand(-1, -1, cos.shape[-1])
                cos = cos.gather(1, embedding_index).masked_fill(pad_mask.unsqueeze(-1), 0)
                sin = sin.gather(1, embedding_index).masked_fill(pad_mask.unsqueeze(-1), 0)
                kwargs["position_embeddings"] = (cos, sin)
            if kwargs.get("cache_position") is not None:
                kwargs["cache_position"] = kwargs["cache_position"][: gather_index.shape[1]]
            return self.layer(hidden_states, *args, **kwargs), kwargs
        return self.layer(hidden_states, *args, **kwargs), kwargs
