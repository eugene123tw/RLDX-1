"""MSAT: Multi-Stream Action Transformer (top-level orchestrator).

Submodules:
- msat_ops.py: RoPE, TimestepEncoder, SwiGLUFFN, head utilities
- msat_blocks.py: Modulation, SingleStream, DoubleStream, Expanded blocks
"""

from typing import NamedTuple, Optional

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
import torch
from torch import nn
import torch.nn.functional as F

# Re-export so callers can import ``_print`` from ``msat`` directly.
from rldx.dist import rank_zero_print as _print
from rldx.model.modules.action_model.blocks import (
    DoubleStreamBlock,
    ExpandedDoubleStreamBlock,
    ExpandedSingleStreamBlock,
    Modulation,
    ModulationOut,
    SingleStreamBlock,
)
from rldx.model.modules.action_model.ops import RoPEEmbedder1D, TimestepEncoder
from rldx.model.modules.action_model.padded_swiglu import PaddedSwiGLUFFN
from rldx.model.modules.norms import adaptive_layer_norm
from rldx.model.quantization.sites import LinearSite


__all__ = [
    "JointBase",
    "MSAT",
]


class MSATOutput(NamedTuple):
    action: torch.Tensor
    physics: torch.Tensor | None


class JointBase(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    def _forward_inner(
        self,
        sa_embs,
        vl_embs,
        timesteps,
        return_all_hidden_states=False,
        encoder_attention_mask=None,
        physics_embs=None,
        physics_attention_mask=None,
    ):
        """
        Forward pass for MSAT.
            - sa_embs: (B, N_sa, sa_dim) - concatenated state+action
            - vl_embs: (B, N_vl, vl_dim) - backbone features (cognition tokens, after the VL tokens when cog_mode='full')
            - physics_embs: (B, N_p, sa_dim) - physics signal tokens (only when use_physics=True)
            - physics_attention_mask: (B,) - per-sample physics mask (1=visible, 0=masked)
        """
        temb = self.timestep_encoder(timesteps)

        # Create Time Token
        time_token = None
        has_time_token = False
        if self.temb_type == "input_token":
            # 1. Projection: (B, D) -> (B, 1, D)
            t_emb = self.time_token_proj(temb).unsqueeze(1)

            # 2. Replication: (B, 1, D) -> (B, N, D)
            time_token = t_emb.repeat(1, self.num_temb_tokens, 1)
            has_time_token = True

        # Generate Shared Modulations
        shared_modulations = None
        shared_single_modulation = None
        if self.temb_type == "shared_mod":
            sa_mod1_raw, sa_mod2_raw = self.shared_sa_mod(temb)
            vl_mod1_raw, vl_mod2_raw = self.shared_vl_mod(temb)
            shared_modulations = {
                "sa_mod1_raw": sa_mod1_raw,
                "sa_mod2_raw": sa_mod2_raw,
                "vl_mod1_raw": vl_mod1_raw,
                "vl_mod2_raw": vl_mod2_raw,
            }
            shared_single_mod_raw, _ = self.shared_single_mod(temb)
            shared_single_modulation = ModulationOut(
                shift=self.shared_single_mod_proj(shared_single_mod_raw.shift),
                scale=self.shared_single_mod_proj(shared_single_mod_raw.scale),
                gate=self.shared_single_mod_proj(shared_single_mod_raw.gate),
            )

        if self.use_physics != (physics_embs is not None):
            raise ValueError(
                f"MSAT built with use_physics={self.use_physics} cannot take "
                f"physics_embs={'a tensor' if physics_embs is not None else None}"
            )
        sa, vl = sa_embs.contiguous(), vl_embs.contiguous()
        p = None if physics_embs is None else physics_embs.contiguous()
        n_p = 0 if p is None else p.shape[1]

        # Lower: DoubleStreamBlocks [VL | time | S | A (| P)]
        if has_time_token:
            sa = torch.cat([time_token, sa], dim=1)
        n_time = self.num_temb_tokens if has_time_token else 0
        all_hidden = [sa]

        pe = None
        if self.use_rope:
            n_body = sa.shape[1] - n_time
            first = n_time + (self.positional_embeddings == "rope_vl_sa")
            pe = self.rope_embedder(
                self._position_ids(sa.shape[0], vl.shape[1], n_time, n_body, first, n_p, sa.device)
            )

        block_idx = 0
        for blk in self.double_blocks:
            stream = dict(
                pe=pe,
                shared_modulations=shared_modulations,
                has_time_token=has_time_token,
                block_idx=block_idx,
                encoder_attention_mask=encoder_attention_mask,
            )
            if p is None:
                sa, vl = blk(sa, vl, temb, **stream)
            else:
                sa, vl, p = blk(
                    sa,
                    vl,
                    temb,
                    **stream,
                    p_tokens=p,
                    physics_attention_mask=physics_attention_mask,
                )
            all_hidden.append(sa)
            block_idx += 1

        if has_time_token:
            time_token = sa[:, : self.num_temb_tokens, :]
            sa = sa[:, self.num_temb_tokens :, :]

        # Upper: SingleStreamBlocks [VL' | time | S | A (| P)]
        if len(self.single_blocks) > 0:
            n_vl = vl.shape[1]
            vl_projected = self.vl_proj_to_sa(vl)
            if has_time_token:
                x = torch.cat([vl_projected, time_token, sa], dim=1)
            else:
                x = torch.cat([vl_projected, sa], dim=1)

            pe_single = None
            if self.use_rope:
                pe_single = self.rope_embedder(
                    self._position_ids(
                        x.shape[0], n_vl, n_time, sa.shape[1], n_time + 1, n_p, x.device
                    )
                )

            single_attn_mask = self._single_padding_mask(
                x, n_vl, n_p, encoder_attention_mask, physics_attention_mask
            )

            for blk in self.single_blocks:
                stream = dict(
                    pe=pe_single,
                    shared_modulation=shared_single_modulation,
                    time_token=time_token if has_time_token else None,
                    block_idx=block_idx,
                    attn_mask=single_attn_mask,
                )
                if p is None:
                    x = blk(x, temb, **stream)
                else:
                    x, p = blk(x, temb, **stream, p_tokens=p)
                block_idx += 1

            sa = x[:, -sa.shape[1] :, :]

        out = MSATOutput(
            self._output_projection(sa, temb),
            None if p is None else self._output_projection_physics(p, temb),
        )
        if return_all_hidden_states:
            return out, all_hidden
        return out

    def _position_ids(self, batch, n_vl, n_time, n_body, first, n_p, device):
        """RoPE ids over [VL | time | body | P]: VL on axis 0 (rope_vl_sa only), time and body
        on axis 1 with the body starting at ``first``, P on axis 1 with axis 0 set to 1."""
        ids = torch.zeros(batch, n_vl + n_time + n_body + n_p, 2, dtype=torch.long, device=device)
        if self.positional_embeddings == "rope_vl_sa":
            ids[:, :n_vl, 0] = torch.arange(n_vl, device=device)
        body = n_vl + n_time
        ids[:, n_vl:body, 1] = torch.arange(n_time, device=device)
        ids[:, body : body + n_body, 1] = torch.arange(first, first + n_body, device=device)
        ids[:, body + n_body :, 0] = 1
        ids[:, body + n_body :, 1] = torch.arange(n_p, device=device)
        return ids

    @staticmethod
    def _single_padding_mask(x, n_vl, n_p, encoder_attention_mask, physics_attention_mask):
        if encoder_attention_mask is None and physics_attention_mask is None:
            return None
        batch, n_x = x.shape[:2]
        if encoder_attention_mask is not None:
            rest = torch.ones(
                batch, n_x - n_vl, device=x.device, dtype=encoder_attention_mask.dtype
            )
            kv_mask = torch.cat([encoder_attention_mask, rest], dim=1)
        else:
            kv_mask = torch.ones(batch, n_x, device=x.device, dtype=x.dtype)
        if n_p:
            if physics_attention_mask is not None:
                p_mask = physics_attention_mask[:, None].expand(-1, n_p).to(dtype=kv_mask.dtype)
            else:
                p_mask = torch.ones(batch, n_p, device=x.device, dtype=kv_mask.dtype)
            kv_mask = torch.cat([kv_mask, p_mask], dim=1)
        kv_mask = kv_mask[:, None, None, :]
        return torch.where(
            kv_mask == 0,
            torch.tensor(float("-inf"), device=x.device, dtype=x.dtype),
            torch.tensor(0.0, device=x.device, dtype=x.dtype),
        )

    def _output_projection_physics(self, p, temb):
        """Physics output projection with AdaLN-zero style."""
        shift, scale = self.proj_out_physics_1(F.silu(temb)).chunk(2, dim=1)
        p = self.norm_out_physics(p) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_physics_2(p)

    def _output_projection(self, sa, temb):
        """Output projection with AdaLN-zero style."""
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        sa = adaptive_layer_norm(self.norm_out, sa, scale[:, None], shift[:, None])
        return self.proj_out_2(sa)


def _stream_sites(group, block, stream):
    for name in ("qkv", "proj"):
        yield LinearSite(group, block, f"{stream}_{name}")
    mlp = getattr(block, f"{stream}_mlp")
    if isinstance(mlp, PaddedSwiGLUFFN):
        yield LinearSite(group, mlp, "w12")
        yield LinearSite(group, mlp, "w3", elementwise_input=True)


class MSAT(JointBase):
    """
    Flux-style MSAT with DoubleStreamBlock + SingleStreamBlock architecture.
    """

    def quantization_sites(self):
        for block in self.double_blocks:
            yield from _stream_sites("msat_vl", block, "vl")
            yield from _stream_sites("msat_sa", block, "sa")
        if not isinstance(self.vl_proj_to_sa, nn.Identity):
            yield LinearSite("msat_vl", self, "vl_proj_to_sa")
        for block in self.single_blocks:
            for name in ("linear1", "linear2", "mlp_proj"):
                if getattr(block, name) is not None:
                    yield LinearSite("msat_sa", block, name)

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        depth_multi_stream: int = 12,  # Number of DoubleStreamBlocks
        depth_single_stream: int = 0,  # Number of SingleStreamBlocks (Flux style)
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        attention_bias: Optional[bool] = None,  # If None, defaults to True
        activation_fn: str = "gelu-approximate",
        norm_eps: float = 1e-6,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        action_model_max_seq_len: int = 512,
        sa_dim: int = 1536,
        vl_dim: int = 1536,
        qk_norm: str = "none",
        use_swiglu: bool = False,
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: Optional[
            float
        ] = None,  # If None, use mlp_ratio. Set lower to reduce VL stream params.
        temb_type: str = "layerwise_mod",  # "layerwise_mod", "shared_mod", or "input_token"
        remove_bias: bool = False,  # If True, remove bias from Modulation and projection layers
        pre_norm: str = "layer_norm",  # Pre-normalization type: "none", "layer_norm", or "rms_norm"
        post_norm: str = "none",  # Post-normalization type: "none", "layer_norm", or "rms_norm"
        rope_theta: float = 10000.0,  # Theta parameter for RoPE. Higher values result in slower rotation (smaller angles).
        # Physics (tactile/torque) conditioning
        use_physics: bool = False,
        physics_dim: int = 0,  # Total physics signal dimension (e.g. tactile_dim + torque_dim)
    ):
        super().__init__()
        self.use_physics = use_physics
        self.physics_dim = physics_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )
        self.positional_embeddings = positional_embeddings
        self.attention_head_dim = attention_head_dim
        self.temb_type = temb_type if temb_type is not None else "layerwise_mod"
        self.num_temb_tokens = 1  # Single time token when temb_type="input_token"
        self.attn_dropout = attn_dropout

        # Set default attention_bias if not provided (MSAT default: True)
        if attention_bias is None:
            attention_bias = True

        # If remove_bias=True, override attention_bias to False
        if remove_bias:
            attention_bias = False
            _print(
                "[MSAT] remove_bias=True: overriding attention_bias to False for all attention layers"
            )

        # Create time token projection if temb_type is "input_token"
        # Time token is always added right before action tokens: [VL | S | time_token | A]
        if self.temb_type == "input_token":
            # Double stream: VL, SA - time_token goes before SA (which contains action)
            if self.inner_dim != sa_dim:
                self.time_token_proj = nn.Linear(self.inner_dim, sa_dim, bias=not remove_bias)
            else:
                self.time_token_proj = nn.Identity()

            self.time_token_pos_emb = None
        else:
            self.time_token_proj = None
            self.time_token_pos_emb = None

        self.freq_encoder = None
        self.freq_token_proj = None
        self.num_freq_tokens = 0

        # Create shared modulation modules if temb_type is "shared_mod"
        if self.temb_type == "shared_mod":
            # Double stream: VL, SA
            self.shared_sa_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)
            self.shared_vl_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)

            # For SingleStreamBlocks: use inner_dim as input (same as temb), then project output
            self.shared_single_mod = Modulation(
                self.inner_dim, double=False, remove_bias=remove_bias
            )
            if self.inner_dim != sa_dim:
                self.shared_single_mod_proj = nn.Linear(
                    self.inner_dim, sa_dim, bias=not remove_bias
                )
            else:
                self.shared_single_mod_proj = nn.Identity()

        _print("\nInitializing MSAT...")

        # Initialize RoPE embedder if needed
        if positional_embeddings == "rope_sa_only":
            # RoPE for SA stream only (attention_head_dim assumed to be 64 below)
            # Axis 0 (dim=16): 0 (unused)
            # Axis 1 (dim=48): SA sequence position
            _print(f"[MSAT] RoPE theta: {rope_theta}")
            self.rope_embedder = RoPEEmbedder1D(
                head_dim=attention_head_dim,
                axes_dim=[attention_head_dim // 4, attention_head_dim - attention_head_dim // 4],
                theta=rope_theta,
                max_seq_len=action_model_max_seq_len,
            )
            self.use_rope = True
        elif positional_embeddings == "rope_vl_sa":
            # RoPE for VL and SA streams (attention_head_dim assumed to be 64 below)
            # Axis 0 (dim=48): VL sequence position
            # Axis 1 (dim=16): SA sequence position
            self.rope_embedder = RoPEEmbedder1D(
                head_dim=attention_head_dim,
                axes_dim=[attention_head_dim - attention_head_dim // 4, attention_head_dim // 4],
                theta=rope_theta,
                max_seq_len=action_model_max_seq_len,
            )
            self.use_rope = True
        else:
            self.rope_embedder = None
            self.use_rope = False

        use_pos_emb = (
            positional_embeddings == "sinusoidal" and action_model_max_seq_len is not None
        ) or self.use_rope
        _print(
            f"[MSAT] 'positional_embeddings' of MSAT: {positional_embeddings}, "
            f"action_model_max_seq_len: {action_model_max_seq_len}, enabled: {use_pos_emb}"
        )

        self.sa_dim = sa_dim
        self.vl_dim = vl_dim

        # VL→SA projection (used by both physics and non-physics paths)
        if sa_dim != vl_dim:
            self.vl_proj_to_sa = nn.Linear(vl_dim, sa_dim, bias=not remove_bias)
            _print(f"[MSAT] Projecting VL dimension from {vl_dim} to {sa_dim}")
        else:
            self.vl_proj_to_sa = nn.Identity()

        if self.use_physics and physics_dim <= 0:
            raise ValueError(f"use_physics needs a positive physics_dim, got {physics_dim}")
        block = dict(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation_fn=activation_fn,
            attention_bias=attention_bias,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            use_swiglu=use_swiglu,
            positional_embeddings=positional_embeddings,
            max_seq_length=action_model_max_seq_len,
            temb_type=self.temb_type,
            remove_bias=remove_bias,
            pre_norm=pre_norm,
            post_norm=post_norm,
        )
        if self.use_physics:
            # Lower: ExpandedDoubleStreamBlocks [VL | SA | P], upper: ExpandedSingleStreamBlocks
            # [VL+SA | P]. Physics tokens are projected to sa_dim by the physics head.
            _print(f"\n[MSAT] Physics mode: use_physics=True, physics_dim={physics_dim}")
            self.double_blocks = nn.ModuleList(
                ExpandedDoubleStreamBlock(
                    sa_dim=sa_dim, vl_dim=vl_dim, p_dim=sa_dim, vl_mlp_ratio=vl_mlp_ratio, **block
                )
                for _ in range(depth_multi_stream)
            )
            self.single_blocks = nn.ModuleList(
                ExpandedSingleStreamBlock(hidden_size=sa_dim, p_dim=sa_dim, **block)
                for _ in range(depth_single_stream)
            )
            self.norm_out_physics = nn.LayerNorm(sa_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_physics_1 = nn.Linear(self.inner_dim, 2 * sa_dim, bias=not remove_bias)
            self.proj_out_physics_2 = nn.Linear(sa_dim, output_dim, bias=not remove_bias)
        else:
            # Lower: DoubleStreamBlocks [VL | SA], upper: SingleStreamBlocks [VL_proj | time_token | SA]
            self.double_blocks = nn.ModuleList(
                DoubleStreamBlock(sa_dim=sa_dim, vl_dim=vl_dim, vl_mlp_ratio=vl_mlp_ratio, **block)
                for _ in range(depth_multi_stream)
            )
            self.single_blocks = nn.ModuleList(
                SingleStreamBlock(hidden_size=sa_dim, **block) for _ in range(depth_single_stream)
            )

        # Output projection (AdaLN-zero style) - for action prediction
        self.norm_out = nn.LayerNorm(sa_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * sa_dim, bias=not remove_bias)
        self.proj_out_2 = nn.Linear(sa_dim, output_dim, bias=not remove_bias)

        self._remove_bias = remove_bias

    def forward(
        self,
        hidden_states: torch.Tensor,  # SA tokens
        encoder_hidden_states: torch.Tensor,  # VL tokens
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask: Optional[
            torch.Tensor
        ] = None,  # [B, N_vl] VL attention mask (1=visible, 0=masked)
        physics_embs: Optional[
            torch.Tensor
        ] = None,  # [B, N_p, sa_dim] Physics tokens (when use_physics=True)
        physics_attention_mask: Optional[
            torch.Tensor
        ] = None,  # [B] per-sample physics mask (1=visible, 0=masked)
    ):
        return self._forward_inner(
            hidden_states,
            encoder_hidden_states,
            timestep,
            return_all_hidden_states,
            encoder_attention_mask,
            physics_embs,
            physics_attention_mask,
        )
