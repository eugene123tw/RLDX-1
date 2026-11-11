from itertools import chain
from types import MethodType
from typing import NamedTuple, Optional

from rldx.configs.execution import InferenceRuntimeProfile, RuntimeProfile, TrainingRuntimeProfile
from rldx.configs.model.inference import InferenceConfig, overridden
from rldx.configs.model.rldx import MSAT_DEFAULTS, ActionModelLoraConfig, MemoryConfig, RLDXConfig
from rldx.dist import rank_zero_print as _print
from rldx.model.contracts import (
    ActionModelInput,
    ActionOutput,
    RLDXInputBatch,
    TrainingOutput,
    validate_rldx_input_batch,
)
from rldx.model.frozen_module_policy import FrozenModulePolicy
from rldx.model.inference_noise import initial_noise
from rldx.model.lora import inject_lora
from rldx.model.modules.action_model.action_model_execution import (
    DenoiseSteps,
    build_serving_kernels,
    build_training_kernels,
    install_action_model_execution_kernels,
)
from rldx.model.modules.action_model.msat import MSAT
from rldx.model.modules.action_model.padded_swiglu import restore_padded_swiglu_invariants
from rldx.model.modules.action_model.physics import init_physics_params_near_zero
from rldx.model.modules.action_model.physics_head import (
    ConditionPhysicsHead,
    FlowPhysicsHead,
    PhysicsHead,
)
from rldx.model.modules.action_model.rtc import (
    RtcActions,
    RtcInput,
    build_noisy_trajectory_rtc,
    build_per_token_time,
    guidance_scale,
    rtc_timing,
    sample_training_prefix,
)
from rldx.model.modules.backbone.adapter import VTCQwen3VLBackbone
from rldx.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from rldx.model.modules.hidden_padding import realign_hidden_widths
from rldx.model.modules.memory import TransformerMemory
from rldx.model.quantization.apply import quantize_linears
from rldx.model.quantization.linear import GemmSelection, QuantizedLinear
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature


class DenoiseContext(NamedTuple):
    vl_embeds: torch.Tensor
    encoder_attention_mask: Optional[torch.Tensor]
    state_features: torch.Tensor
    embodiment_id: torch.Tensor
    action_dim_mask: torch.Tensor


def _pin_prefix(values: torch.Tensor, prefix: RtcActions) -> torch.Tensor:
    d = prefix.actions.shape[1]
    present = prefix.present.view(*prefix.present.shape, *(1,) * (values.dim() - 2))
    pinned = torch.where(present, prefix.actions, values[:, :d])
    return torch.cat((pinned, values[:, d:]), dim=1)


MSAT_PHYSICS_LORA_TARGETS = ("p_qkv", "p_proj")


class RLDXActionModel(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def quantization_sites(self):
        yield from self.model.quantization_sites()

    def prepare_for_execution(self, profile: RuntimeProfile) -> None:
        """Install the selected execution backend for MSAT forward."""
        match profile:
            case InferenceRuntimeProfile():
                kernels = build_serving_kernels(
                    profile.backend,
                    DenoiseSteps(RLDXActionModel._euler_step, RLDXActionModel._velocity),
                )
            case TrainingRuntimeProfile():
                kernels = build_training_kernels(profile.backend)
        install_action_model_execution_kernels(self.model, kernels)
        steps = kernels.denoise_steps
        if steps is None:
            self.__dict__.pop("_euler_step", None)
            self.__dict__.pop("_guided_velocity", None)
        else:
            self._euler_step = MethodType(steps.euler, self)
            self._guided_velocity = MethodType(steps.guided_velocity, self)

    def prepare_input_gradients(self) -> None:
        for layer in [m for m in self.model.modules() if isinstance(m, QuantizedLinear)]:
            layer.prepare_input_gradient()

    def __init__(self, config: RLDXConfig, inference: InferenceConfig = InferenceConfig()):
        super().__init__()
        self.config = config
        self.mask_padded_action_dims = overridden(
            config.mask_padded_action_dims, inference.mask_padded_action_dims
        )
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        msat = {
            **MSAT_DEFAULTS,
            "sa_dim": config.input_embedding_dim,
            "vl_dim": config.backbone_embedding_dim,
            **config.diffusion_model_cfg,
        }
        physics = config.physics
        self.model = MSAT(
            **msat,
            attn_dropout=float(config.attn_dropout),
            use_physics=physics is not None,
            physics_dim=0 if physics is None else physics.dim,
        )
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.sample_timestep_from_beta_dist = inference.sample_timestep_from_beta_dist
        self.denoising_timesteps = inference.denoising_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        # These land in whatever the ambient default dtype is, so a
        # from_pretrained(torch_dtype=bfloat16) load leaves them bfloat16 --
        # which Beta cannot sample from. Left as-is on purpose: training builds
        # the model under the fp32 default dtype and the one inference
        # consumer re-casts locally. See get_action_with_features.
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        # Real-Time Chunking. A sample whose whole horizon is frozen has no
        # postfix to learn from, so one position always stays free.
        self.rtc_training_max_delay = config.rtc_training_max_delay
        if not 0 <= self.rtc_training_max_delay < self.action_horizon:
            raise ValueError(
                f"rtc_training_max_delay={self.rtc_training_max_delay} must be in "
                f"[0, {self.action_horizon - 1}]"
            )
        self.rtc_timing = rtc_timing(
            inference.rtc, self.rtc_training_max_delay, self.action_horizon
        )
        if self.rtc_training_max_delay or self.rtc_timing is not None:
            _print(
                f"[RTC] training_max_delay={self.rtc_training_max_delay}, "
                f"inference={self.rtc_timing}"
            )

        # Physics (tactile/torque) stream; with none, nothing is built or wired.
        self.physics: PhysicsHead | None = None
        if physics is not None:
            physics_delta = physics.delta_indices or []
            head = dict(
                physics_dim=physics.dim,
                embed_dim=self.input_embedding_dim,
                msat_output_dim=msat["output_dim"],
                physics_delta_indices=physics_delta,
                physics_loss_weight=physics.loss_weight,
                physics_dropout_prob=physics.dropout_prob,
                masked=physics.allow_missing,
            )
            if physics.use_flow_matching and any(d > 0 for d in physics_delta):
                self.physics = FlowPhysicsHead(**head, action_horizon=self.action_horizon)
            else:
                self.physics = ConditionPhysicsHead(**head)
            init_physics_params_near_zero(self.physics, self.model)

        restore_padded_swiglu_invariants(self)
        self._set_trainable_parameters()
        if config.action_model_lora is not None:
            self.apply_lora(config.action_model_lora)

    def _projector_modules(self) -> list[nn.Module]:
        modules = [self.state_encoder, self.action_encoder, self.action_decoder]
        if self.physics is not None:
            modules.append(self.physics)
        if self.config.add_pos_embed:
            modules.append(self.position_embedding)
        return modules

    def _set_trainable_parameters(self):
        for p in self.parameters():
            p.requires_grad = True
        if not self.config.tune_projector:
            for module in self._projector_modules():
                module.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)

        if not self.config.tune_diffusion_model:
            self.model.requires_grad_(False)

        if not self.config.tune_vlln:
            self.vlln.requires_grad_(False)

        _print(f"[MSAT] Tune action model projector: {self.config.tune_projector}")
        _print(f"[MSAT] Tune action model diffusion model: {self.config.tune_diffusion_model}")
        _print(f"[MSAT] Tune action model vlln: {self.config.tune_vlln}")
        _print(f"[MSAT] Action model LoRA: {self.config.action_model_lora}")

    def restore_frozen_modes(self) -> None:
        if not self.training:
            return
        if not self.config.tune_projector:
            for module in self._projector_modules():
                module.eval()
        if self.config.action_model_lora is not None or not self.config.tune_diffusion_model:
            self.model.eval()

    def apply_lora(self, lora: ActionModelLoraConfig) -> None:
        """Only the adapters train: the whole MSAT freezes first, whatever tune_diffusion_model."""
        targets = [
            target
            for target in lora.target_modules
            if self.physics is not None or target not in MSAT_PHYSICS_LORA_TARGETS
        ]
        present = {name.rpartition(".")[2] for name, _ in self.model.named_modules()}
        missing = [target for target in targets if target not in present]
        if missing:
            raise ValueError(f"MSAT has no {missing} to put LoRA adapters on")
        self.model.requires_grad_(False)
        inject_lora(self.model, lora, targets)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def _sample_beta_denoising_timesteps(self) -> list[float]:
        """Build the Python-controlled inference schedule entirely on the host."""
        if self.beta_dist.concentration1.dtype != torch.float32:
            self.beta_dist = Beta(
                self.beta_dist.concentration1.float(),
                self.beta_dist.concentration0.float(),
            )
        interior = self.sample_time(
            self.num_inference_timesteps - 1,
            device="cpu",
            dtype=torch.float32,
        )
        return [0.0, *interior.sort().values.tolist(), 1.0]

    def _normalize_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """Apply action-head normalization to an already validated output."""
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        return BatchFeature(data={**backbone_output, "backbone_features": backbone_features})

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: ActionModelInput,
    ) -> TrainingOutput:
        """
        Forward pass through the action model.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: canonical execution mask: ``None`` when
                  all tokens are valid, otherwise [B, seq_len]
            action_input: Input containing:
                - state: [B, state_horizon, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]
        """
        backbone_output = self._normalize_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        encoder_attention_mask = backbone_output.get("backbone_attention_mask", None)
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input["embodiment_id"]

        # Embed state.
        state_features = self.state_encoder(action_input["state"], embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            _print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input["action"]
        batch_size = actions.shape[0]
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        if self.mask_padded_action_dims:
            action_dim_mask = action_input["action_dim_mask"]
            actions = actions * action_dim_mask
            noise = noise * action_dim_mask
        t_raw = self.sample_time(batch_size, device=actions.device, dtype=actions.dtype)  # (B,)

        # Training RTC: per-sample prefix with clean ground-truth actions at t=1.
        if self.training and self.rtc_training_max_delay > 0:
            prefix_mask = sample_training_prefix(
                batch_size,
                self.action_horizon,
                self.rtc_training_max_delay,
                device=actions.device,
            )  # (B, H) bool
            t_tok = build_per_token_time(t_raw, prefix_mask)  # (B, H)
            noisy_trajectory = build_noisy_trajectory_rtc(actions, noise, t_tok)
        else:
            prefix_mask = None
            t_tok = t_raw.unsqueeze(1).expand(-1, self.action_horizon).contiguous()
            t = t_raw[:, None, None]
            noisy_trajectory = (1 - t) * noise + t * actions

        velocity = actions - noise
        action_features = self.action_encoder(noisy_trajectory, t_tok, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        # Memory dropout mutates the mask on device, so the host-canonical
        # None-or-padded invariant no longer applies to that configuration.
        if (
            self.config.memory is not None
            and self.config.memory.dropout_prob > 0.0
            and encoder_attention_mask is not None
            and encoder_attention_mask.all()
        ):
            encoder_attention_mask = None

        physics = None if self.physics is None else self.physics.prepare_train(action_input, t_raw)

        # MSAT global temb uses the scalar postfix τ per sample. Per-token
        # time has already been threaded through action_encoder above.
        model_output, _ = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            timestep=t_raw,
            return_all_hidden_states=True,
            encoder_attention_mask=encoder_attention_mask,
            **({} if physics is None else self.physics.msat_inputs(physics.tokens, physics.mask)),
        )

        pred = self.action_decoder(model_output.action, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Action loss. Training RTC masks out the clean-prefix positions so
        # the model is only graded on postfix reconstruction.
        action_mask = action_input["action_mask"]
        loss_mask = action_mask
        if prefix_mask is not None:
            postfix = (~prefix_mask).to(dtype=action_mask.dtype).unsqueeze(-1)
            loss_mask = action_mask * postfix
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * loss_mask
        valid_count = loss_mask.sum(dim=(1, 2)).clamp_min(1)
        loss = (action_loss.sum(dim=(1, 2)) / valid_count).mean()

        # Physics prediction loss (flow matching only; conditioning-only mode has no physics loss)
        # Not ``action_mask``: PAD leaves the episode boundary out of that one, and
        # physics has no label past the end under either policy.
        physics_loss = (
            None
            if physics is None
            else self.physics.loss(
                model_output.physics,
                physics,
                action_input.get("action_boundary_mask", action_mask),
            )
        )
        if physics_loss is not None:
            loss = loss + self.physics.physics_loss_weight * physics_loss
        return TrainingOutput(loss=loss, action_loss=action_loss, physics_loss=physics_loss)

    def _encode_features(
        self,
        backbone_output: BatchFeature,
        action_input: ActionModelInput,
    ) -> BatchFeature:
        """
        Encode features for the action model.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: canonical execution mask: ``None`` when
                  all tokens are valid, otherwise [B, seq_len]
            action_input: Input containing:
                - state: [B, state_horizon, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self._normalize_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        backbone_attention_mask = backbone_output.get("backbone_attention_mask", None)
        embodiment_id = action_input["embodiment_id"]

        # Embed state.
        state_features = self.state_encoder(action_input["state"], embodiment_id)

        return BatchFeature(
            data={
                "backbone_features": vl_embeds,
                "backbone_attention_mask": backbone_attention_mask,
                "state_features": state_features,
            }
        )

    def max_sample_tokens(self, state_tokens: int) -> int:
        return (
            state_tokens
            + self.action_horizon
            + (0 if self.physics is None else self.physics.hist_len + self.physics.fut_len)
        )

    def _velocity(self, actions, t_scalar, t_tok, context: DenoiseContext, physics):
        action_features = self.action_encoder(
            actions * context.action_dim_mask, t_tok, context.embodiment_id
        )
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=actions.device
            )
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
        sa = torch.cat((context.state_features, action_features), dim=1)
        model_output = self.model(
            hidden_states=sa,
            encoder_hidden_states=context.vl_embeds,
            timestep=t_scalar,
            encoder_attention_mask=context.encoder_attention_mask,
            **(
                {}
                if physics is None
                else self.physics.msat_inputs(self.physics.tokens(physics, t_scalar), physics.mask)
            ),
        )
        decoded = self.action_decoder(model_output.action, context.embodiment_id)
        return decoded[:, -actions.shape[1] :] * context.action_dim_mask, model_output.physics

    _guided_velocity = _velocity

    def _euler_update(
        self, actions, velocity, dt, context: DenoiseContext, physics, physics_output
    ):
        actions = (actions + dt * velocity) * context.action_dim_mask
        if physics is not None:
            physics = self.physics.step(physics, physics_output, dt)
        return actions, physics

    def _euler_step(self, actions, t_now, t_next, context: DenoiseContext, physics, prefix=None):
        t_scalar = t_now.to(actions.dtype).expand(actions.shape[0])
        t_tok = t_scalar[:, None].expand(-1, actions.shape[1])
        if prefix is not None:
            # Per-token time t=1 on the prefix is a *training-RTC* trick
            # (arXiv 2512.05964): the model has to be trained to expect a
            # clean prefix marked by t=1.
            actions = _pin_prefix(actions, prefix)
            ones = RtcActions(t_tok.new_ones(prefix.actions.shape[:2]), prefix.present)
            t_tok = _pin_prefix(t_tok, ones)
        velocity, physics_output = self._velocity(actions, t_scalar, t_tok, context, physics)
        actions, physics = self._euler_update(
            actions, velocity, t_next - t_now, context, physics, physics_output
        )
        return (actions if prefix is None else _pin_prefix(actions, prefix)), physics

    def _guided_step(
        self,
        actions,
        t_now: float,
        dt: float,
        context: DenoiseContext,
        physics,
        rtc: RtcInput,
        soft_mask: torch.Tensor,
    ):
        batch_size, horizon = actions.shape[:2]
        prefix_len = rtc.prefix.actions.shape[1]
        dtype = context.vl_embeds.dtype
        mask = context.action_dim_mask
        t_scalar = torch.full((batch_size,), t_now, device=actions.device, dtype=dtype)
        t_tok = t_scalar.unsqueeze(1).expand(-1, horizon).clone()
        # Compute v_guided = v + c · VJP[(Y−Â¹)ᵀ diag(W), ∂Â¹/∂x].
        x_g = actions.detach().requires_grad_(True)
        with torch.enable_grad():
            v, mo = self._guided_velocity(x_g, t_scalar, t_tok, context, physics)
            a_hat = (x_g + (1.0 - t_now) * v) * mask
            # Inpainting target Y (Eq. 5 of arXiv 2506.07339): the prefix, then
            # the previous chunk's predictions on the ramp where a sample has
            # them. Prefix positions that are not frozen carry zero weight.
            Y = a_hat.detach().clone()
            Y[:, :prefix_len] = rtc.prefix.actions
            if rtc.postfix is not None:
                ramp = slice(prefix_len, prefix_len + rtc.postfix.actions.shape[1])
                present = rtc.postfix.present[..., None]
                Y[:, ramp] = torch.where(present, rtc.postfix.actions, Y[:, ramp])
            grad_outputs = soft_mask.unsqueeze(-1) * (Y - a_hat.detach())
            vjp = torch.autograd.grad(
                outputs=a_hat,
                inputs=x_g,
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
            )[0]
        c = guidance_scale(t_now, self.rtc_timing.rtc.jacobian_beta)
        velocity = (v.detach() + c * vjp.detach()).to(dtype=dtype)
        physics_output = None if mo is None else mo.detach()
        with torch.no_grad():
            # Load-bearing: ``vjp`` is a gradient w.r.t. ``x_g``, so it can be
            # non-zero on padded dims even though every forward tensor above is
            # masked (the gradient reaches them through the network from the
            # valid outputs). Do not drop this as "already masked".
            return self._euler_update(
                actions, velocity * mask, dt, context, physics, physics_output
            )

    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: ActionModelInput | None = None,
    ) -> ActionOutput:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model supplying the
                attention mask associated with ``backbone_features``.
            action_input: Optional, used for physics conditioning. When
                Real-Time Chunking is enabled by config it must carry ``rtc``,
                the parsed ``RtcInput`` whose prefix length is the configured
                delay; ``present`` marks the prefix positions that are frozen.
        """
        vl_embeds = backbone_features
        encoder_attention_mask = backbone_output.get("backbone_attention_mask", None)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype
        horizon = self.config.action_horizon

        # ─── RTC setup ──────────────────────────────────────────────────────
        timing = self.rtc_timing
        rtc_input = None if timing is None else action_input["rtc"]

        # ─── Physics ────────────────────────────────────────────────────────
        with torch.no_grad():
            phys_state = (
                None
                if self.physics is None
                else self.physics.prepare_inference(action_input, batch_size, device, dtype)
            )

        # ─── Initial noise ──────────────────────────────────────────────────
        actions = initial_noise((batch_size, horizon, self.action_dim), dtype, device)
        action_dim_mask = (
            action_input["action_dim_mask"].to(dtype)
            if self.mask_padded_action_dims
            else torch.ones_like(actions)
        )
        actions = actions * action_dim_mask

        # Denoising schedule. Priority: explicit list > beta-sampled > uniform.
        # Uniform is the default; the other two are opt-in from the eval server
        # (--denoising-timesteps / --sample-timestep-from-beta-dist).
        if self.denoising_timesteps is not None:
            timesteps_list = list(self.denoising_timesteps) + [1.0]
        elif self.sample_timestep_from_beta_dist:
            # Draw the interior steps from the same Beta the trainer samples t
            # from, so the grid is dense where training saw the most samples.
            # t=0 (pure noise) and t=1 always bracket them.
            #
            # Force float32: this model is served via from_pretrained(
            # torch_dtype=bfloat16), which runs __init__ under a bfloat16
            # default dtype, so self.beta_dist ends up holding bfloat16
            # concentrations -- and Dirichlet (which Beta samples through) has
            # no bfloat16 kernel. Training is unaffected (__init__ runs under
            # the fp32 default dtype), which is why this only ever bites on
            # the inference path.
            timesteps_list = self._sample_beta_denoising_timesteps()
        else:
            n = self.num_inference_timesteps
            timesteps_list = [t / float(n) for t in range(n)] + [1.0]

        context = DenoiseContext(
            vl_embeds, encoder_attention_mask, state_features, embodiment_id, action_dim_mask
        )
        schedule = [torch.full((), t, dtype=torch.float32, device=device) for t in timesteps_list]
        guided = timing is not None and timing.guided
        pinned = None
        soft_mask = None
        if timing is not None and not timing.guided:
            prefix = rtc_input.prefix
            pinned = RtcActions(prefix.actions * action_dim_mask[:, : timing.delay], prefix.present)
        if guided:
            frozen = rtc_input.prefix.present
            free = frozen.new_ones(batch_size, horizon - timing.delay)
            presence = torch.cat((frozen, free), dim=1).to(dtype)
            soft_mask = timing.soft_mask(device=device, dtype=dtype) * presence

        # Run denoising steps.
        for i in range(len(timesteps_list) - 1):
            t_now = float(timesteps_list[i])
            dt = float(timesteps_list[i + 1] - timesteps_list[i])
            if guided and (
                timing.rtc.jacobian_steps_only is None or i < timing.rtc.jacobian_steps_only
            ):
                actions, phys_state = self._guided_step(
                    actions, t_now, dt, context, phys_state, rtc_input, soft_mask
                )
            else:
                with torch.no_grad():
                    actions, phys_state = self._euler_step(
                        actions, schedule[i], schedule[i + 1], context, phys_state, pinned
                    )

        return ActionOutput(action_pred=actions)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: ActionModelInput,
    ) -> ActionOutput:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_horizon, state_dim]
                - embodiment_id: [B] (embodiment IDs)
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input["embodiment_id"],
            backbone_output=BatchFeature(
                data={"backbone_attention_mask": features.get("backbone_attention_mask", None)}
            ),
            action_input=action_input,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class RLDX(PreTrainedModel):
    """RLDX: Vision-Language-Action model with backbone."""

    config_class = RLDXConfig
    supports_gradient_checkpointing = True

    def prepare_for_execution(self, profile: RuntimeProfile, gemm_rows: int | None = None) -> None:
        """Delegate execution-backend preparation to model-owned components."""
        realign_hidden_widths(self, profile.hidden_alignment)
        match profile:
            case InferenceRuntimeProfile(
                quantization=quantization,
                nvfp4_backends=backends,
                nvfp4_library_quantize=library_groups,
            ):
                self.backbone.fuse_text_projections()
                quantize_linears(
                    chain(
                        self.backbone.quantization_sites(),
                        self.action_model.quantization_sites(),
                    ),
                    quantization,
                    {
                        group: GemmSelection(backend, gemm_rows)
                        for group, backend in backends.items()
                    },
                    library_groups,
                )
        self.backbone.prepare_for_execution(profile)
        self.action_model.prepare_for_execution(profile)

    def train(self, mode: bool = True):
        """Set the model mode without repeating unchanged training-tree walks.

        Hugging Face Trainer calls ``model.train()`` for every micro-batch. An
        RLDX training tree deliberately contains frozen subtrees in eval mode,
        so recursively rebuilding that same mixed-mode tree each time is pure
        Python overhead. The constructor establishes the configured mixed-mode
        training tree, so an unchanged mode request can preserve it exactly as
        it is. A real transition into training runs normal PyTorch recursion
        before restoring the frozen subtrees (including the train-mode motion
        block).

        Consequently, callers must perform intentional subtree mode changes
        through the RLDX runtime lifecycle rather than using ``model.train()``
        as a way to repair an already-training tree.
        """
        if mode == self.training:
            return self

        super().train(mode)
        if mode:
            FrozenModulePolicy.apply_training(self)
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Skip the redundant VTC-Qwen3-VL backbone download on inference loads.

        ``HF AutoModel.from_pretrained`` will overwrite the entire backbone
        with the trained checkpoint's state_dict anyway — every ``RLDX-1-{PT,
        FT-*, MT-*}`` ckpt carries the full backbone weights. Without this
        override the backbone constructor first downloads
        ``RLWRLD/RLDX-1-VLM`` (~16 GB) and then HF immediately overwrites
        those weights with the ckpt state_dict, wasting bandwidth and disk.

        Setting ``transformers_loading_kwargs["skip_pretrained_weights"]
        = True`` here builds the backbone from config only; the state_dict
        load downstream fills it in. Direct construction
        (``RLDX(config, ...)``) is unaffected — fresh-train from VLM still
        downloads through the unchanged default in ``__init__``.

        Caller may opt out by passing
        ``transformers_loading_kwargs={"skip_pretrained_weights": False}``.
        Missing backbone keys in the ckpt would surface via the standard HF
        ``missing_keys`` warning, so silent corruption is not introduced.
        """
        tlk = dict(kwargs.pop("transformers_loading_kwargs", None) or {})
        tlk.setdefault("skip_pretrained_weights", True)
        kwargs["transformers_loading_kwargs"] = tlk
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def __init__(
        self,
        config: RLDXConfig,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
        inference: InferenceConfig = InferenceConfig(),
    ):
        """
        Initialize RLDX model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config
        kwargs = {}
        kwargs["cog_mode"] = "cog_only"
        kwargs["n_cog_tokens"] = self.config.n_cog_tokens
        _print(f"\n[MSAT Configs] n_cog_tokens: {kwargs['n_cog_tokens']}")

        if self.config.motion is not None:
            kwargs["motion_config"] = self.config.motion.vision_config()
            _print(f"[motion module] Enabled with config: {kwargs['motion_config']}")

        if config.backbone_model_type != "vtc_qwen3_vl":
            raise ValueError(
                f"Unsupported backbone_model_type={config.backbone_model_type!r}; "
                "only 'vtc_qwen3_vl' is supported."
            )
        self.backbone = VTCQwen3VLBackbone(
            model_name=config.model_name,
            image_max_area=config.image.max_area,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            freeze_cog_tokens=config.freeze_cog_tokens,
            transformers_loading_kwargs=transformers_loading_kwargs,
            skip_pretrained_weights=transformers_loading_kwargs.pop(
                "skip_pretrained_weights", False
            ),
            **kwargs,
        )

        # Initialize action model
        self.action_model = RLDXActionModel(config, inference)

        # After the action model: lora_A init draws from the global RNG.
        if config.backbone_lora is not None:
            self.backbone.apply_lora(config.backbone_lora)

        self.use_memory = config.memory is not None
        self.memory = None

        if self.use_memory:
            self._init_memory(config.memory)

        # Establish the configured mixed-mode training tree once so repeated
        # Trainer ``model.train()`` calls can safely take the constant-time guard.
        FrozenModulePolicy.apply_training(self)

    def _init_memory(self, config: MemoryConfig):
        """Initialize memory module for temporal context aggregation."""
        # Switch backbone to full mode (return VL + cognition tokens)
        self.backbone.cog_mode = "full"

        self._memory_length = config.length
        self._n_cog_tokens = self.backbone.n_cog_tokens

        raw_mem_nq = config.n_cog_tokens
        self._memory_n_cog_tokens = raw_mem_nq if raw_mem_nq is not None else self._n_cog_tokens

        assert self._memory_n_cog_tokens <= self._n_cog_tokens, (
            f"memory_n_cog_tokens ({self._memory_n_cog_tokens}) must be "
            f"<= n_cog_tokens ({self._n_cog_tokens})"
        )

        self._concat_memory = config.concat
        self._memory_dropout_ratio = config.dropout_prob

        # Build memory module
        memory_cfg = dict(config.cfg)
        backbone_hidden_size = self._get_backbone_hidden_size()
        if memory_cfg.get("hidden_size") != backbone_hidden_size:
            _print(
                f"[i] Updating memory hidden_size from {memory_cfg.get('hidden_size')} to {backbone_hidden_size}"
            )
            memory_cfg["hidden_size"] = backbone_hidden_size
            memory_cfg["intermediate_size"] = backbone_hidden_size * 4

        n_mq_mem = self._memory_n_cog_tokens
        memory_cfg["block_attn_size"] = n_mq_mem
        memory_cfg["max_position_embeddings"] = self._memory_length * n_mq_mem

        self.memory = TransformerMemory(**memory_cfg)

        n_out = (self._n_cog_tokens + n_mq_mem) if self._concat_memory else self._n_cog_tokens
        _print(
            f"\n[i] Memory enabled: length={self._memory_length}, "
            f"n_mq_mem={n_mq_mem}, concat={self._concat_memory}, "
            f"dropout={self._memory_dropout_ratio}, output_tokens={n_out}"
        )

        # Memory trainable by default
        for param in self.memory.parameters():
            param.requires_grad = True

    def _get_backbone_hidden_size(self) -> int:
        return self.backbone.qwen_model.model.language_model.config.hidden_size

    @property
    def inference_memory_shape(self) -> tuple[int, int]:
        return self._memory_length * self._memory_n_cog_tokens, self._get_backbone_hidden_size()

    def forward(self, inputs: RLDXInputBatch) -> TrainingOutput:
        validate_rldx_input_batch(inputs)
        backbone_outputs = self.backbone(inputs)

        if self.use_memory:
            backbone_outputs = self._apply_memory_training(backbone_outputs)

        return self.action_model(backbone_outputs, inputs)

    def get_action(self, inputs: RLDXInputBatch) -> ActionOutput:
        inputs = dict(inputs)
        if self.use_memory:
            memory = inputs.pop("memory")
            reset_memory = inputs.pop("reset_memory")

        validate_rldx_input_batch(inputs)
        backbone_outputs = self.backbone(inputs)

        if not self.use_memory:
            return self.action_model.get_action(backbone_outputs, inputs)
        backbone_outputs, memory = self._apply_memory_inference(
            backbone_outputs, memory, reset_memory
        )
        return self.action_model.get_action(backbone_outputs, inputs)._replace(memory=memory)

    # ── Memory processing ──

    def _apply_memory_training(self, backbone_outputs: BatchFeature) -> BatchFeature:
        """Process backbone features through memory for training."""
        backbone_outputs = BatchFeature(data=dict(backbone_outputs))
        backbone_features = backbone_outputs["backbone_features"]  # [B*K, T, d]
        BK, T, d = backbone_features.shape
        K = self._memory_length
        B = BK // K
        n_q = self._n_cog_tokens
        n_mq_mem = self._memory_n_cog_tokens
        n_mq_pass = n_q - n_mq_mem

        mq_all = backbone_features[:, -n_q:, :].contiguous().view(B, K, n_q, d)
        mq_original = mq_all[:, -1, :, :]  # [B, n_q, d]
        mq_for_memory = mq_all[:, :, n_mq_pass:, :]  # [B, K, n_mq_mem, d]

        mq_mem_seq = mq_for_memory.contiguous().view(B, K * n_mq_mem, d)
        mq_memory_out = self.memory(inputs_embeds=mq_mem_seq).last_hidden_state
        mq_augmented = mq_memory_out.view(B, K, n_mq_mem, d)[:, -1, :, :]

        if self._concat_memory:
            backbone_outputs["backbone_features"] = torch.cat([mq_original, mq_augmented], dim=1)
        else:
            if n_mq_pass > 0:
                mq_pass = mq_original[:, :n_mq_pass, :]
                backbone_outputs["backbone_features"] = torch.cat([mq_pass, mq_augmented], dim=1)
            else:
                backbone_outputs["backbone_features"] = mq_augmented

        # Rebuild attention mask
        attn_mask = backbone_outputs.get("backbone_attention_mask")
        if attn_mask is not None:
            attn_mask_last = attn_mask.view(B, K, -1)[:, -1, :]
            mq_mask = attn_mask_last[:, -n_q:]
            if self._concat_memory:
                mem_mask = mq_mask[:, -n_mq_mem:]
                backbone_outputs["backbone_attention_mask"] = torch.cat([mq_mask, mem_mask], dim=1)
            else:
                backbone_outputs["backbone_attention_mask"] = mq_mask

        # Memory dropout (concat mode only)
        if self.training and self._memory_dropout_ratio > 0.0 and self._concat_memory:
            features = backbone_outputs["backbone_features"]
            B_out, seq_len = features.shape[:2]
            attn_mask = backbone_outputs.get("backbone_attention_mask")
            if attn_mask is None:
                attn_mask = torch.ones(B_out, seq_len, dtype=torch.long, device=features.device)
            else:
                attn_mask = attn_mask.clone()
            do_dropout = torch.rand(B_out, device=attn_mask.device) < self._memory_dropout_ratio
            dropout_mask = do_dropout[:, None].expand(-1, n_mq_mem)
            attn_mask[:, -n_mq_mem:] = attn_mask[:, -n_mq_mem:].masked_fill(dropout_mask, 0)
            backbone_outputs["backbone_attention_mask"] = attn_mask

        return backbone_outputs

    def _apply_memory_inference(
        self, backbone_outputs: BatchFeature, memory: torch.Tensor, reset_memory: torch.Tensor
    ) -> tuple[BatchFeature, torch.Tensor]:
        """Process backbone features through memory for inference (single timestep)."""
        backbone_outputs = BatchFeature(data=dict(backbone_outputs))
        backbone_features = backbone_outputs["backbone_features"]
        B = backbone_features.shape[0]
        n_q = self._n_cog_tokens
        n_mq_mem = self._memory_n_cog_tokens
        n_mq_pass = n_q - n_mq_mem

        mq_all = backbone_features[:, -n_q:, :]
        mq_current = mq_all[:, n_mq_pass:, :]

        memory = torch.where(
            reset_memory.view(B, 1, 1),
            mq_current.repeat(1, self._memory_length, 1),
            torch.cat([memory[:, n_mq_mem:, :], mq_current], dim=1),
        )

        mq_memory_out = self.memory(inputs_embeds=memory).last_hidden_state
        mq_augmented = mq_memory_out[:, -n_mq_mem:, :]

        if self._concat_memory:
            backbone_outputs["backbone_features"] = torch.cat([mq_all, mq_augmented], dim=1)
        else:
            if n_mq_pass > 0:
                mq_pass = mq_all[:, :n_mq_pass, :]
                backbone_outputs["backbone_features"] = torch.cat([mq_pass, mq_augmented], dim=1)
            else:
                backbone_outputs["backbone_features"] = mq_augmented

        # The output contains only always-valid cog and memory tokens.
        backbone_outputs["backbone_attention_mask"] = None

        return backbone_outputs, memory

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
