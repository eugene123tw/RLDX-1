"""Physics conditioning + flow matching stream for RLDXActionModel."""

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

from rldx.dist import rank_zero_print as _print
from rldx.model.inference_noise import initial_noise
from rldx.model.modules.action_model.physics import (
    PhysicalSignalDecoder,
    PhysicalSignalEncoder,
    PhysicsNoiseEncoder,
)


def _physics_attention_mask(
    inputs: Mapping[str, Any] | None, batch_size: int, masked: bool
) -> torch.Tensor | None:
    mask = None if inputs is None else inputs.get("physics_mask")
    if mask is None:
        if masked:
            raise ValueError("this checkpoint allows missing physics, so physics_mask is required")
        return None
    if not masked:
        raise ValueError("physics_mask is only read by checkpoints that allow missing physics")
    if not torch.is_tensor(mask) or mask.shape not in ((batch_size,), (batch_size, 1)):
        raise ValueError("physics_mask must have shape [B] or [B, 1]")
    return mask.reshape(batch_size)


class PhysicsInput(NamedTuple):
    values: torch.Tensor
    attention_mask: torch.Tensor | None

    @classmethod
    def parse(
        cls,
        inputs: Mapping[str, Any] | None,
        batch_size: int,
        physics_dim: int,
        *,
        masked: bool,
        min_steps: int = 1,
    ) -> "PhysicsInput":
        if inputs is None or "physics" not in inputs:
            raise ValueError(
                "This physics mode requires prepared 'physics' input; process raw "
                "tactile/torque first. Allowed missing observations must be prepared "
                "as zero-filled physics with physics_mask=0."
            )
        values = inputs["physics"]
        if (
            not torch.is_tensor(values)
            or values.ndim != 3
            or values.shape[0] != batch_size
            or values.shape[1] < min_steps
            or values.shape[2] != physics_dim
            or not values.is_floating_point()
        ):
            raise ValueError(
                f"physics must be a floating tensor with shape [{batch_size}, T, {physics_dim}], "
                f"T >= {min_steps}"
            )
        return cls(values, _physics_attention_mask(inputs, batch_size, masked))


class ConditionTraining(NamedTuple):
    tokens: torch.Tensor
    mask: torch.Tensor | None


class FlowTraining(NamedTuple):
    tokens: torch.Tensor
    mask: torch.Tensor | None
    velocity: torch.Tensor


class ConditionState(NamedTuple):
    """Conditioning tokens, encoded once per request."""

    tokens: torch.Tensor
    mask: torch.Tensor | None


class FlowState(NamedTuple):
    """History tokens encoded once per request, and the future the Euler loop integrates."""

    hist_tok: torch.Tensor
    fut: torch.Tensor
    mask: torch.Tensor | None


# --- State dict key remapping for backward compatibility ---

_PHYSICS_KEY_RENAMES = [
    ("physics_encoder.", "physics.physics_cond_encoder."),  # very old → new
    ("physics_cond_encoder.", "physics.physics_cond_encoder."),  # old → new
    ("physics_fut_encoder.", "physics.physics_fut_encoder."),  # old → new
    ("physics_decoder.", "physics.physics_decoder."),  # old → new
]


def remap_physics_keys(state_dict: dict) -> dict:
    """Remap older physics state_dict keys onto the current PhysicsHead layout.

    Handles both ``physics_encoder.*`` and ``physics_cond_encoder.*`` prefixes.
    Keys already in the current ``physics.physics_*`` layout are left
    unchanged, including nested cases like
    ``action_model.physics.physics_cond_encoder.*``.
    """
    remapped = {}
    renamed_count = 0
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in _PHYSICS_KEY_RENAMES:
            # Idempotence: if the target prefix is already present anywhere in
            # the key, this key is already in the new layout — do not rename.
            if new_prefix in key:
                break
            if old_prefix in key:
                new_key = key.replace(old_prefix, new_prefix, 1)
                renamed_count += 1
                break
        remapped[new_key] = value
    if renamed_count > 0:
        _print(f"[Physics] Remapped {renamed_count} older-format keys → physics.* layout")
    return remapped


class PhysicsHead(nn.Module):
    """The physics encoders and decoder; FlowPhysicsHead and ConditionPhysicsHead use them.

    Registered as RLDXActionModel.physics, so the state_dict keys stay physics.*.
    ``masked`` is whether the checkpoint allows missing physics, and so takes physics_mask.
    """

    def __init__(
        self,
        physics_dim: int,
        embed_dim: int,
        msat_output_dim: int,
        physics_delta_indices: list[int] | None,
        physics_loss_weight: float,
        physics_dropout_prob: float = 0.0,
        masked: bool = False,
    ):
        super().__init__()

        self.physics_dim = physics_dim
        self.embed_dim = embed_dim
        self.physics_loss_weight = physics_loss_weight
        self.physics_dropout_prob = physics_dropout_prob
        self.masked = masked

        delta = physics_delta_indices or []
        self.hist_len = sum(1 for d in delta if d <= 0)
        self.fut_len = sum(1 for d in delta if d > 0)

        self.physics_cond_encoder = PhysicalSignalEncoder(physics_dim, embed_dim, embed_dim)
        self.physics_fut_encoder = PhysicsNoiseEncoder(physics_dim, embed_dim, embed_dim)
        self.physics_decoder = PhysicalSignalDecoder(msat_output_dim, embed_dim, physics_dim)

        # Learned mask token used to replace dropped physics conditioning tokens.
        # Only created when dropout is on so checkpoints saved with prob=0 stay clean.
        self.physics_mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, embed_dim)) if physics_dropout_prob > 0 else None
        )

    def msat_inputs(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> dict:
        return {
            "physics_embs": tokens,
            "physics_attention_mask": mask,
        }

    def _parse(self, action_input, batch_size: int, min_steps: int) -> PhysicsInput:
        return PhysicsInput.parse(
            action_input, batch_size, self.physics_dim, masked=self.masked, min_steps=min_steps
        )

    def _maybe_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        """Per-sample dropout: replace a dropped sample's full token slice with
        the learned `physics_mask_token`. Active only during training."""
        if not (
            self.training
            and self.physics_dropout_prob > 0
            and self.physics_mask_token is not None
            and tokens.shape[1] > 0
        ):
            return tokens
        do_dropout = torch.rand(tokens.shape[0], device=tokens.device) < self.physics_dropout_prob
        do_dropout = do_dropout[:, None, None].to(dtype=tokens.dtype)
        return tokens * (1 - do_dropout) + self.physics_mask_token * do_dropout

    def _announce(self, flow: bool) -> None:
        _print(
            f"\n[Physics] Physics stream enabled "
            f"(dim={self.physics_dim}, weight={self.physics_loss_weight})"
        )
        _print(f"[Physics] hist_len={self.hist_len}, fut_len={self.fut_len}, flow_matching={flow}")
        if self.physics_dropout_prob > 0:
            mode = "hist-only" if flow else "all-conditioning"
            _print(f"[Physics] physics_dropout_prob={self.physics_dropout_prob} ({mode})")


class ConditionPhysicsHead(PhysicsHead):
    """Physics as conditioning tokens only; nothing is predicted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _print(
            "[Physics] Flow matching disabled. Physics used as conditioning only (no prediction loss)"
        )
        self._announce(flow=False)

    def prepare_train(self, action_input, t_raw) -> ConditionTraining:
        physics_input = self._parse(action_input, t_raw.shape[0], 1)
        # Dropout the whole sequence since there's no flow-matching target to preserve.
        tokens = self._maybe_dropout(self.physics_cond_encoder(physics_input.values))
        return ConditionTraining(tokens, physics_input.attention_mask)

    def loss(self, physics_output, prepared: ConditionTraining, action_mask) -> None:
        return None

    def prepare_inference(self, action_input, batch_size, device, dtype) -> ConditionState:
        physics_input = self._parse(action_input, batch_size, 1)
        return ConditionState(
            self.physics_cond_encoder(physics_input.values), physics_input.attention_mask
        )

    def tokens(self, state: ConditionState, timesteps_tensor) -> torch.Tensor:
        return state.tokens

    def step(self, state: ConditionState, physics_output, dt) -> ConditionState:
        return state


class FlowPhysicsHead(PhysicsHead):
    """Predicts the future physics alongside the actions; the history conditions both."""

    def __init__(self, *args, action_horizon: int, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.fut_len == action_horizon, (
            f"fut_len ({self.fut_len}) must equal action_horizon "
            f"({action_horizon}) so that action_mask can be reused for physics loss masking"
        )
        self._announce(flow=True)

    def prepare_train(self, action_input, t_raw) -> FlowTraining:
        physics_input = self._parse(action_input, t_raw.shape[0], self.hist_len + self.fut_len)
        physics_hist = physics_input.values[:, : self.hist_len, :]  # (B, H, D)
        physics_fut_gt = physics_input.values[:, self.hist_len :, :]  # (B, F, D)

        t_broad_p = t_raw[:, None, None]  # (B, 1, 1)
        physics_noise = torch.randn_like(physics_fut_gt)
        # noisy_fut = (1-t) * noise + t * gt  (flow matching interpolation)
        noisy_physics_fut = (1 - t_broad_p) * physics_noise + t_broad_p * physics_fut_gt
        velocity = physics_fut_gt - physics_noise

        if self.hist_len > 0:
            hist_tok = self.physics_cond_encoder(physics_hist)
        else:
            hist_tok = torch.zeros(
                physics_hist.shape[0],
                0,
                self.embed_dim,
                dtype=physics_hist.dtype,
                device=physics_hist.device,
            )
        # Hist-only dropout: future tokens are the prediction target and must NOT be masked.
        hist_tok = self._maybe_dropout(hist_tok)
        fut_tok = self.physics_fut_encoder(noisy_physics_fut, t_raw)
        tokens = torch.cat([hist_tok, fut_tok], dim=1)
        return FlowTraining(tokens, physics_input.attention_mask, velocity)

    def loss(self, physics_output, prepared: FlowTraining, action_mask) -> torch.Tensor:
        physics_pred_vel = self.physics_decoder(physics_output[:, -self.fut_len :, :])

        # Combine per-step action_mask (episode boundary) with per-sample physics mask
        step_mask = action_mask.any(dim=-1).float()  # (B, T) — per-step validity from action
        if prepared.mask is not None:
            step_mask = step_mask * prepared.mask.unsqueeze(1)  # (B, T) * (B, 1)
        mask_3d = step_mask.unsqueeze(-1)  # (B, T, 1)

        loss_unreduced = F.mse_loss(physics_pred_vel, prepared.velocity, reduction="none")
        n_valid = mask_3d.sum() * physics_pred_vel.shape[-1]
        return (loss_unreduced * mask_3d).sum() / (n_valid + 1e-6)

    def prepare_inference(self, action_input, batch_size, device, dtype) -> FlowState:
        if self.hist_len > 0:
            physics_input = self._parse(action_input, batch_size, self.hist_len)
            hist_tok = self.physics_cond_encoder(physics_input.values[:, : self.hist_len, :])
            mask = physics_input.attention_mask
        else:
            hist_tok = torch.zeros(batch_size, 0, self.embed_dim, dtype=dtype, device=device)
            mask = _physics_attention_mask(action_input, batch_size, self.masked)
        fut = initial_noise((batch_size, self.fut_len, self.physics_dim), dtype, device)
        return FlowState(hist_tok, fut, mask)

    def tokens(self, state: FlowState, timesteps_tensor) -> torch.Tensor:
        return torch.cat(
            [state.hist_tok, self.physics_fut_encoder(state.fut, timesteps_tensor)], dim=1
        )

    def step(self, state: FlowState, physics_output, dt) -> FlowState:
        velocity = self.physics_decoder(physics_output[:, -self.fut_len :, :])
        return FlowState(state.hist_tok, state.fut + dt * velocity, state.mask)
