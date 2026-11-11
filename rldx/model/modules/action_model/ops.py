"""MSAT utility ops: RoPE, TimestepEncoder, SwiGLUFFN, head utils (extracted from msat.py).

``TimestepEncoder`` (NVIDIA Isaac GR00T) and ``SwiGLUFFN`` (Meta DINOv2, via
LightningDiT) are vendored from upstream projects and carry their attribution in their
own docstrings. Neither upstream file is reproduced here as a whole, so each notice sits
on the symbol it covers rather than in a file-level copyright header.
"""

from typing import Callable, Optional

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
import torch
from torch import nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute interleaved RoPE as real 2-by-2 rotation matrices.

    Args:
        dim: Head dimension (must be even)
        end: Maximum sequence length
        theta: Base frequency parameter

    Returns:
        Rotation matrices, shape (end, dim//2, 2, 2) as float32.
    """
    assert dim % 2 == 0, f"dim must be even (dimensions rotate in pairs), got {dim}"
    rates = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(end, dtype=torch.float32)
    angles = positions[:, None] * rates[None, :]
    # Preserve the existing table bits; separate cos/sin calls can differ by one ULP.
    freqs_cis = torch.exp(torch.complex(torch.zeros_like(angles), angles))
    return torch.stack(
        (freqs_cis.real, -freqs_cis.imag, freqs_cis.imag, freqs_cis.real), dim=-1
    ).reshape(end, dim // 2, 2, 2)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply interleaved RoPE to Q and K with real rotation matrices.

    Args:
        xq: Query tensor, shape (B, H, N, D) where D=head_dim
        xk: Key tensor, shape (B, H, N, D) where D=head_dim
        freqs_cis: Rotation matrices, shape (B, N, D//2, 2, 2)

    Returns:
        Rotated Q and K, in the dtype they came in as

    Follows FLUX's ``apply_rope``:
    https://github.com/black-forest-labs/flux, src/flux/math.py

    FLUX is licensed under the Apache License, Version 2.0. The upstream file carries no
    copyright line, so none is reproduced here. Modified from the original, per Section
    4(b): the shape checks, the head axis added to ``freqs_cis``, and the reshape folded
    into each rotation.
    """
    assert xq.ndim == 4 and xk.ndim == 4, "q and k must have shape (B, H, N, D)"
    assert xq.shape == xk.shape, f"q shape {xq.shape} != k shape {xk.shape}"
    expected = (xq.shape[0], xq.shape[2], xq.shape[-1] // 2, 2, 2)
    assert freqs_cis.shape == expected, f"freqs_cis shape {freqs_cis.shape} != {expected}"

    freqs_cis = freqs_cis.unsqueeze(1)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = (freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]).reshape(xq.shape)
    xk_out = (freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]).reshape(xk.shape)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEEmbedder1D(nn.Module):
    """
    Generate real RoPE rotation matrices for 1D sequences with multiple axes.
    For joint_attn_v2:
    - rope_sa_only: Axis 0 (dim=16) = 0 (unused), Axis 1 (dim=48) = SA sequence position (includes time tokens and action tokens)
      * Time tokens: axis 1 = 0..num_temb_tokens-1
      * Action tokens: axis 1 = num_temb_tokens..
    - rope_vl_sa: Axis 0 (dim=48) = VL sequence (Fixed), Axis 1 (dim=16) = SA sequence (Scaled)
    """

    def __init__(
        self,
        head_dim: int,
        axes_dim: list[int],
        theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert sum(axes_dim) == head_dim, (
            f"sum(axes_dim)={sum(axes_dim)} must equal head_dim={head_dim}"
        )
        self.head_dim = head_dim
        self.axes_dim = axes_dim
        self.theta = theta
        self.n_axes = len(axes_dim)
        self.max_seq_len = max_seq_len

        for i, axis_dim in enumerate(axes_dim):
            freqs_cis = precompute_freqs_cis(axis_dim, max_seq_len, theta)
            # These buffers are a deterministic function of the constructor
            # arguments and are rebuilt on every module initialization.
            self.register_buffer(f"freqs_cis_{i}", freqs_cis, persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for i, axis_dim in enumerate(self.axes_dim):
            device = getattr(self, f"freqs_cis_{i}").device
            setattr(
                self,
                f"freqs_cis_{i}",
                precompute_freqs_cis(axis_dim, self.max_seq_len, self.theta).to(device),
            )
        return self

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ids: Position IDs (B, N, n_axes)
        """
        n_axes = ids.shape[-1]
        assert n_axes == self.n_axes, f"ids last dim {n_axes} must equal n_axes {self.n_axes}"

        freqs_list = []
        for i in range(n_axes):
            freqs_cis = getattr(self, f"freqs_cis_{i}")
            pos_ids = ids[..., i]
            freqs = freqs_cis[pos_ids]
            freqs_list.append(freqs)

        return torch.cat(freqs_list, dim=-3)


# Encoders ================================================
class TimestepEncoder(nn.Module):
    """
    Copied verbatim from NVIDIA Isaac GR00T, down to the unused ``compute_dtype`` arg:
    https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/modules/dit.py#L61-L71

    At that revision GR00T is licensed under the Apache License, Version 2.0, and the
    upstream file carries a copyright line, so it is reproduced here as Section 4(c)
    requires:

        Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    The notice is deliberately here rather than in a file-level header: this module is
    not a copy of any GR00T file, it borrows eleven lines from one, and a file-level
    copyright line would assert NVIDIA's copyright over code they did not write. Other
    single-symbol copies in this repository are annotated the same way -- see
    ``quat2axisangle`` in rldx/eval/sim/LIBERO/libero_env.py.
    """

    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


# Normalization and FFN ================================================
class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward block taken from LightningDiT, which vendored it from DINOv2.

    Copied from DINOv2:
    https://github.com/facebookresearch/dinov2/blob/81b2b6419385a321287de91e00282ef7cbd26f94/dinov2/layers/swiglu_ffn.py#L14-L34
    Received via LightningDiT:
    https://github.com/hustvl/LightningDiT/blob/0e483457f4f142f57c5ca164f7d172880e1055a6/models/swiglu_ffn.py#L15-L36

    DINOv2 is licensed under the Apache License, Version 2.0, and the upstream file
    carries a copyright line, so it is reproduced here as Section 4(c) requires. (Some
    other upstreams vendored into this repository carry none, and no copyright line is
    invented for those.)

        Copyright (c) Meta Platforms, Inc. and affiliates.

    Modified from the original, per Section 4(b), in two notational ways: ``act_layer``
    is annotated ``Optional[Callable[..., nn.Module]]`` rather than
    ``Callable[..., nn.Module]``, and ``forward`` spells ``torch.Tensor`` rather than the
    imported ``Tensor``. The ``__init__`` and ``forward`` bodies are otherwise identical
    to upstream, variable names (w12, w3, x12, x1, x2, hidden) included.

    No LightningDiT notice is owed for this class. LightningDiT ships the file under
    Meta's header, and its only change to the class is a ``@torch.compile`` decorator on
    ``forward`` -- which this copy does not have. No HUST Vision Lab expression reaches
    this file, so LightningDiT's own MIT license has nothing to attach to; the route is
    recorded above only because it is how the code arrived, and because the original
    one-line docstring named it.

    The DINOv2 pin is 81b2b641 ("Update license everywhere", 2023-08-31). The class body
    is byte-stable from ebc1cba1 (2023-08-30) through main, so the pin is not fragile;
    81b2b641 is chosen because it is the revision that replaced the earlier "All rights
    reserved" header with Apache-2.0, i.e. the license this copy is taken under.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


# RMSNorm, create_norm_layer, create_qk_norm_layers → imported from common.py


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # (B, N, D) -> (B, H, N, Dh)
    B, N, D = x.shape
    Dh = D // num_heads
    return x.view(B, N, num_heads, Dh).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # (B, H, N, Dh) -> (B, N, D)
    B, H, N, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, N, H * Dh)
