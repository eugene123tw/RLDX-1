import torch
import torch.nn as nn
import torch.nn.functional as F

from rldx.model.modules.embodiment_conditioned_mlp import SinusoidalPositionalEncoding
from rldx.model.modules.norms import RMSNorm


class PhysicalSignalEncoder(nn.Module):
    """Encode physics history tokens: (B, T_hist, input_dim) -> (B, T_hist, output_dim)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, output_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.W1(x)
        t_ids = torch.arange(T, device=x.device).float().unsqueeze(0).expand(B, -1)
        pos = self.pos_encoding(t_ids).to(dtype=h.dtype)
        h = F.silu(self.W2(torch.cat([h, pos], dim=-1)))
        return self.W3(h)  # (B, T_hist, output_dim)


class PhysicalSignalDecoder(nn.Module):
    """Decode physics predictions: (B, T, input_dim) -> (B, T, output_dim)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, T, output_dim)


class PhysicsNoiseEncoder(nn.Module):
    """Future tokens: (B, T_fut, input_dim) -> (B, T_fut, output_dim)
    Positional encoding uses diffusion timestep instead of sequential index."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, output_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_fut, input_dim) - noisy future physics
            timesteps: (B,) - diffusion timestep (scalar per sample)
        Returns:
            (B, T_fut, output_dim)
        """
        B, T, _ = x.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)  # (B, T_fut)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,)")
        x_emb = self.W1(x)  # (B, T_fut, hidden_dim)
        t_emb = self.pos_encoding(timesteps).to(dtype=x_emb.dtype)  # (B, T_fut, hidden_dim)
        x = F.silu(self.W2(torch.cat([x_emb, t_emb], dim=-1)))  # (B, T_fut, hidden_dim)
        return self.W3(x)  # (B, T_fut, output_dim)


def _xavier(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _small_noise(m: nn.Linear, std: float) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=std)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _reset_norm_identity(m: nn.Module) -> None:
    if isinstance(m, RMSNorm):
        nn.init.ones_(m.weight)


def init_physics_params_near_zero(physics: nn.Module, msat: nn.Module) -> None:
    """Internal layers get Xavier, exit layers near-zero, so the physics stream starts at ~0."""
    for encoder in (physics.physics_cond_encoder, physics.physics_fut_encoder):
        _xavier(encoder.W1)
        _xavier(encoder.W2)
        _small_noise(encoder.W3, std=1e-5)
    decoder_linears = [m for m in physics.physics_decoder.net if isinstance(m, nn.Linear)]
    _small_noise(decoder_linears[-1], std=1e-4)

    for blk in msat.double_blocks:
        _xavier(blk.p_qkv)
        _small_noise(blk.p_proj, std=1e-4)
        mlp_linears = [m for m in blk.p_mlp.modules() if isinstance(m, nn.Linear)]
        for linear in mlp_linears:
            _xavier(linear)
        _small_noise(mlp_linears[-1], std=1e-4)
        if blk.p_mod is not None:
            blk.p_mod.apply(_xavier)
        for norm in (
            blk.p_norm1,
            blk.p_norm2_attn,
            blk.p_norm2_mlp,
            blk.p_norm3_mlp,
            blk.q_norm_p,
            blk.k_norm_p,
        ):
            _reset_norm_identity(norm)

    for blk in msat.single_blocks:
        _xavier(blk.p_linear1)
        _small_noise(blk.p_linear2, std=1e-4)
        if blk.p_mlp_proj is not None:
            _xavier(blk.p_mlp_proj)
        _reset_norm_identity(blk.p_pre_norm)
        _reset_norm_identity(blk.p_post_norm)

    _small_noise(msat.proj_out_physics_1, std=1e-5)
    _small_noise(msat.proj_out_physics_2, std=1e-4)
