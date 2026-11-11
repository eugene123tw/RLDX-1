# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original NVIDIA Isaac GR00T N1.7.
# Original source: https://github.com/NVIDIA/Isaac-GR00T

from dataclasses import asdict, dataclass, field, replace
import os
from typing import Annotated

import tyro

from rldx.configs.data.data_config import DataArgs, SingleDatasetConfig
from rldx.configs.data.dataset_mix import dataset_mix
from rldx.configs.model.rldx import (
    ActionModelLoraConfig,
    BackboneLoraConfig,
    ImageGeometryConfig,
    MemoryConfig,
    MotionConfig,
    PhysicsConfig,
    RLDXConfig,
    default_color_jitter_params,
)
from rldx.configs.training.runtime import TrainingArgs, WandbConfig
from rldx.data.embodiment_tags import EmbodimentTag


@dataclass(frozen=True)
class DataByPaths:
    """Datasets given by path, all of one embodiment."""

    paths: list[str]
    """LeRobot dataset roots."""

    embodiment_tag: EmbodimentTag
    """Embodiment every dataset is."""

    mix_ratios: list[float] | None = None
    """Relative sampling weight per path, aligned with paths; None weighs them equally."""

    def __post_init__(self) -> None:
        if self.mix_ratios is not None and len(self.mix_ratios) != len(self.paths):
            raise ValueError(f"{len(self.mix_ratios)} mix ratios for {len(self.paths)} paths")
        if self.mix_ratios is not None and any(r <= 0 for r in self.mix_ratios):
            raise ValueError(f"mix ratios must be > 0, got {self.mix_ratios}")

    def specs(self) -> list[SingleDatasetConfig]:
        ratios = self.mix_ratios or [1.0] * len(self.paths)
        return [
            SingleDatasetConfig(
                dataset_paths=[path], embodiment_tag=self.embodiment_tag.value, mix_ratio=ratio
            )
            for path, ratio in zip(self.paths, ratios)
        ]


@dataclass(frozen=True)
class DataByMix:
    """Datasets given by a registered mix (rldx/configs/data/dataset_mix.py), which names each
    dataset, its embodiment and its weight."""

    root: str
    """Directory the mix's dataset names are relative to."""

    name: str
    """Registered mix name."""

    def specs(self) -> list[SingleDatasetConfig]:
        if self.name not in dataset_mix:
            raise ValueError(
                f"Unknown dataset mix {self.name!r}.\nAvailable:\n  "
                + "\n  ".join(sorted(dataset_mix))
            )
        return [
            SingleDatasetConfig(
                dataset_paths=[os.path.join(self.root, entry["dataset_name"])],
                embodiment_tag=entry["embodiment_tag"].value,
                mix_ratio=entry["mix_ratio"],
            )
            for entry in dataset_mix[self.name]
        ]


@dataclass(frozen=True)
class MotionArgs:
    """Motion-Aware Spatio-Temporal Summarization block in the vision encoder."""

    insert_layer: int = MotionConfig.insert_layer
    """Vision encoder layer the block is inserted at."""

    injection_point: str = MotionConfig.injection_point
    """'vision_encoder' (residual add at the insert layer) or 'vl_input' (project and
    prepend as LLM input tokens)."""

    pool_type: str = MotionConfig.pool_type
    """Spatial pooling for vl_input tokens: 'avg' or 'conv' (learned depthwise separable)."""

    drop: bool = MotionConfig.drop
    """Drop the motion tokens at the internal_projection layer; keeping them through every
    LLM layer costs ~40% more compute."""

    gradient_check: bool = MotionConfig.gradient_check
    """Log the motion module's gradient norms during training."""

    def to_config(self, saved: MotionConfig | None) -> MotionConfig:
        return replace(saved or MotionConfig(), **asdict(self))


@dataclass(frozen=True)
class MemoryArgs:
    """Memory-augmented cognition tokens (HAMLET) over past timesteps."""

    length: int = MemoryConfig.length
    """Past timesteps in the memory context window."""

    n_cog_tokens: int | None = MemoryConfig.n_cog_tokens
    """Cognition tokens routed through memory, at most n_cog_tokens; None = all."""

    concat: bool = MemoryConfig.concat
    """Append the memory-augmented tokens after the cognition tokens; --memory.no-concat
    replaces the memory subset instead."""

    blockwise_attn: bool = False
    """Attend bidirectionally within a timestep's tokens while staying causal across
    timesteps, instead of token-level causal attention."""

    dropout_prob: float = MemoryConfig.dropout_prob
    """With concat, the probability of masking the augmented tokens out of attention during
    training."""

    stride: int = MemoryConfig.stride
    """Action steps between memory snapshots; set it to the serving execution horizon."""

    def to_config(self, saved: MemoryConfig | None) -> MemoryConfig:
        base = saved or MemoryConfig()
        return replace(
            base,
            length=self.length,
            n_cog_tokens=self.n_cog_tokens,
            concat=self.concat,
            dropout_prob=self.dropout_prob,
            stride=self.stride,
            cfg={**base.cfg, "use_causal_attn": False} if self.blockwise_attn else base.cfg,
        )


@dataclass(frozen=True)
class PhysicsArgs:
    """Physics (tactile / torque) stream in the action model."""

    keys: list[str]
    """Physics signal keys, e.g. tactile torque."""

    dims: list[int]
    """Per-key dimensions aligned with keys, e.g. 30 7."""

    loss_weight: float = PhysicsConfig.loss_weight
    """Weight of the physics prediction loss."""

    allow_missing: bool = PhysicsConfig.allow_missing
    """Allow datasets without the physics keys; their samples are zero-filled and masked."""

    dropout_prob: float = PhysicsConfig.dropout_prob
    """Per-sample probability of replacing the physics conditioning tokens with a learned
    mask token during training (flow matching drops the history tokens only)."""

    def to_config(self, saved: PhysicsConfig | None) -> PhysicsConfig:
        if saved is None:
            return PhysicsConfig(**asdict(self))
        return replace(saved, **asdict(self))


@dataclass(frozen=True)
class ActionModelLoraArgs:
    """LoRA adapters on the action model (MSAT); only they train there, so
    --tune-diffusion-model is not given with them."""

    rank: int = ActionModelLoraConfig.rank
    alpha: int = ActionModelLoraConfig.alpha
    dropout: float = ActionModelLoraConfig.dropout

    def __post_init__(self):
        # TODO: decide whether LoRA training wants its frozen base in train mode.
        if self.dropout != 0.0:
            raise ValueError(
                f"action-model-lora dropout {self.dropout} has no effect: "
                "the frozen MSAT the adapters sit in runs in eval mode"
            )

    def to_config(self, saved: ActionModelLoraConfig | None) -> ActionModelLoraConfig:
        return replace(saved or ActionModelLoraConfig(), **asdict(self))


@dataclass(frozen=True)
class BackboneLoraArgs:
    """LoRA adapters on the backbone LLM layers; only they train in the backbone, so
    --tune-llm, --tune-visual, --tune-top-llm-layers and --freeze-cog-tokens are not
    given with them."""

    rank: int = BackboneLoraConfig.rank
    alpha: int = BackboneLoraConfig.alpha
    dropout: float = BackboneLoraConfig.dropout

    num_layers: int = BackboneLoraConfig.num_layers
    """Top LLM layers to wrap: -1 = all, N > 0 = the top N."""

    def __post_init__(self):
        # TODO: decide whether LoRA training wants its frozen base in train mode.
        if self.dropout != 0.0:
            raise ValueError(
                f"backbone-lora dropout {self.dropout} has no effect: "
                "the frozen backbone the adapters sit in runs in eval mode"
            )

    def to_config(self, saved: BackboneLoraConfig | None) -> BackboneLoraConfig:
        return replace(saved or BackboneLoraConfig(), **asdict(self))


@dataclass
class ActionModelArgs:
    """MSAT action-model regularizers, written into the model config on every run."""

    attn_dropout: float = 0.0
    """DS-4: attention-map dropout (SDPA dropout_p) in the MSAT action head,
    train-only. 0.0 = off (baseline)."""


@dataclass
class ModelArgs:
    """Written into the model config on every run, a finetune included.

    WARN: a finetune that leaves out n_cog_tokens, action_horizon, image_max_area,
    image_resize_m or conversation_image_first trains with the CLI default, not the
    checkpoint's value. A different n_cog_tokens re-initializes the trained cog_emb, with
    only a shape-mismatch log line to show for it.
    TODO: move those facts about the checkpoint to CheckpointOverrides.
    """

    # Left out, a tune flag takes RLDXConfig's value. TrainArgs rejects one given next to
    # a LoRA on its part.

    tune_llm: tyro.conf.DisallowNone[bool | None] = None
    """If True, fine-tune the language model (LLM) backbone during training (False)."""

    tune_visual: tyro.conf.DisallowNone[bool | None] = None
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone) (False)."""

    tune_projector: bool = RLDXConfig.tune_projector
    """If True, fine-tune the action head's per-embodiment state/action encoders and action
    decoder (plus physics head, position embedding, mask token). Keep True for a robot the
    base checkpoint never trained under its general_embodiment / new_embodiment slot."""

    tune_diffusion_model: tyro.conf.DisallowNone[bool | None] = None
    """If True, fine-tune the diffusion-based action decoder (if present in the model)
    (True)."""

    tune_top_llm_layers: tyro.conf.DisallowNone[int | None] = None
    """Number of top LLM layers to tune (4)."""

    freeze_cog_tokens: tyro.conf.DisallowNone[bool | None] = None
    """If True, freeze the cog_emb parameter in the backbone to prevent VLM layer
    backpropagation (False)."""

    mask_padded_action_dims: bool = RLDXConfig.mask_padded_action_dims
    """Hold padded action dims at 0 during training: the ground-truth action and
    the noise are both multiplied by ``action_dim_mask``, so dims an embodiment
    does not use contribute no gradient. Written into ``config.json`` and read
    back by evaluation, so the two stages cannot disagree."""

    state_dropout_prob: float = RLDXConfig.state_dropout_prob
    """Dropout probability applied to state inputs for regularization during
    training. Treated as the *initial* probability when
    --state-dropout-prob-final is set — the actual per-step value is
    annealed from state_dropout_prob → state_dropout_prob_final on the
    --state-dropout-schedule curve."""

    rtc_training_max_delay: int = RLDXConfig.rtc_training_max_delay
    """Training-time RTC: for each sample, draw prefix length d ~ U[0, max_delay]
    and condition on the clean prefix while masking its loss. 0 disables."""

    general_embodiment_train_ratio: float = RLDXConfig.general_embodiment_train_ratio
    """Ratio to train general_embodiment encoder/decoder"""

    n_cog_tokens: int = RLDXConfig.n_cog_tokens
    """Number of meta query tokens to use (for MSATv1)."""

    action_horizon: int = RLDXConfig.action_horizon
    """Number of future action steps predicted per chunk. Must equal the length
    of every action modality's delta_indices in the modality_config — the
    dataloader and MSAT both rely on this width, so a mismatch is caught at
    assembly time rather than during forward."""

    image_max_area: int = ImageGeometryConfig.max_area
    """Area budget (in pixels) for the aspect-ratio-preserving resize step.
    Defaults to 256*256 = 65536. For a 480x640 input this produces 192x256; for
    a 256x256 input it is a no-op."""

    image_resize_m: int = ImageGeometryConfig.resize_m
    """Alignment multiple for both output dimensions after the aspect-area resize.
    Defaults to 32."""

    color_jitter_params: dict[str, float] | None = field(
        default_factory=default_color_jitter_params
    )
    """
    Parameters for color jitter augmentation on images.

    Expected keys: "brightness", "contrast", "saturation", "hue" (all float).

    When a dict is given, an ``A.ColorJitter`` transform is added to the train
    pipeline (see ``rldx/data/augmentations.py``); missing keys default to 0.0.
    Set to None to disable color jitter entirely (no transform is added).
    """

    conversation_image_first: bool = RLDXConfig.conversation_image_first
    """If True, use image first in conversation"""

    model_revision: str | None = RLDXConfig.model_revision
    """Git commit / branch / tag to pin when ``--base-model-path`` is a HF Hub
    repo id. None = HEAD (reproducibility hazard for a moving branch). Threads
    through ``snapshot_download`` in ``launch_train`` and every nested
    ``from_pretrained`` call inside the VTC backbone via
    ``transformers_loading_kwargs['revision']``."""

    def with_defaults(self) -> "ModelArgs":
        """The tune flags left out take RLDXConfig's values."""
        return replace(
            self,
            tune_llm=RLDXConfig.tune_llm if self.tune_llm is None else self.tune_llm,
            tune_visual=RLDXConfig.tune_visual if self.tune_visual is None else self.tune_visual,
            tune_top_llm_layers=(
                RLDXConfig.tune_top_llm_layers
                if self.tune_top_llm_layers is None
                else self.tune_top_llm_layers
            ),
            tune_diffusion_model=(
                RLDXConfig.tune_diffusion_model
                if self.tune_diffusion_model is None
                else self.tune_diffusion_model
            ),
            freeze_cog_tokens=(
                RLDXConfig.freeze_cog_tokens
                if self.freeze_cog_tokens is None
                else self.freeze_cog_tokens
            ),
        )


@dataclass
class CheckpointOverrides:
    """None keeps the base checkpoint's value, or the model default when training from scratch.

    TODO: augmentation is split between the two rules (random_crop_fraction and
    random_rotation_angle here, color_jitter_params in ModelArgs); give it one.
    """

    backbone_path: str | None = None
    """Path to backbone model (e.g., 'Qwen/Qwen3-VL-8B-Instruct' or a VTC checkpoint).
    If None, defaults to 'Qwen/Qwen3-VL-8B-Instruct' for from-scratch training,
    or uses the checkpoint's saved value."""

    max_num_embodiments: int | None = None
    """Projector width -- how many embodiment slots the model carries.

    ``None`` leaves the model config's own default alone. Set it (from the
    experiment spec, or on the command line) whenever a run needs more slots
    than the default: the value then differs from the default, so it lands in
    the checkpoint's ``config.json`` and evaluation rebuilds the right width.
    Widening the model default instead leaves the field out of ``config.json``
    entirely, and the checkpoint stops being loadable by anything that defaults
    differently."""

    backbone_select_layer: int | None = None
    """Override ``RLDXConfig.select_layer`` — the LLM layer below which the
    backbone is truncated (layers above this index are dropped at build
    time). None = keep the checkpoint-loaded / RLDXConfig default (18).
    Mostly used for layer-ablation experiments."""

    use_percentiles: bool | None = None
    """Override state/action normalization: True = q01/q99, False = min/max. None = the
    RLDXConfig default for from-scratch, the checkpoint's config.json value for fine-tuning."""

    random_crop_fraction: float | None = None
    """Optional fractional crop applied after the aspect-area resize.
    Training uses a random crop position; evaluation uses a center crop.
    The cropped region is resized back to the pre-crop shape so downstream
    stages always see a fixed output size. None = no-op (default)."""

    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""


@dataclass
class TrainArgs:
    """
    Unified configuration for pretraining and fine-tuning a Vision-Language-Action (VLA) model.

    The datasets come from data:paths (paths and one embodiment tag) or data:mix (a registered
    mix under a root).
    """

    # Model configuration --------------------------------------------------------------------------

    base_model_path: str | None = None
    """Path to the pretrained base model checkpoint (e.g., HuggingFace Hub or local directory)."""

    allow_dim_mismatch: bool = False
    """Allow the modality-config-derived state/action input width to exceed the
    model's max_state_dim / max_action_dim. When the concatenated state/action
    dim (sum of end-start over the modality config's keys) is larger than the
    projector input width:
      * False (default): fail fast at assembly time with an actionable error
        (the processor's pad step would otherwise crash with a negative pad
        size deep in collation).
      * True: grow max_state_dim / max_action_dim to the derived value. The
        affected projector tensors (state_encoder.layer1 for state;
        action_encoder.W1 + action_decoder.layer2 for action) then shape-
        mismatch the checkpoint and are RE-INITIALIZED for ALL embodiment
        slots, losing the pretrained warm-start for that projection. Pair with
        --new-param-warmup-steps to ease the new params in."""

    # ----------------------------------------------------------------------------------------------

    # Dataset config -------------------------------------------------------------------------------
    data: (
        Annotated[DataByPaths, tyro.conf.subcommand("paths")]
        | Annotated[DataByMix, tyro.conf.subcommand("mix")]
        | None
    ) = None
    """Training datasets: data:paths --data.paths … --data.embodiment-tag …, or
    data:mix --data.root … --data.name …."""

    training: Annotated[TrainingArgs, tyro.conf.OmitArgPrefixes] = field(
        default_factory=TrainingArgs
    )
    data_loading: Annotated[DataArgs, tyro.conf.OmitArgPrefixes] = field(default_factory=DataArgs)
    model: Annotated[ModelArgs, tyro.conf.OmitArgPrefixes] = field(default_factory=ModelArgs)
    overrides: Annotated[CheckpointOverrides, tyro.conf.OmitArgPrefixes] = field(
        default_factory=CheckpointOverrides
    )
    action_model: Annotated[ActionModelArgs, tyro.conf.OmitArgPrefixes] = field(
        default_factory=ActionModelArgs
    )

    wandb: Annotated[WandbConfig, tyro.conf.subcommand("on")] | None = None
    """Log metrics to Weights & Biases (log in first): wandb:on."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment.
    If None, use the pre-registered modality config in rldx/configs/data/embodiment_configs.py.
    """

    # ----------------------------------------------------------------------------------------------

    # Plugin experiments ---------------------------------------------------------------------------
    experiment: str | None = None
    """Name of a planned run a plugin registers (plugins load via ``RLDX_PLUGINS``).

    Explicit flags still win over the preset."""

    # ----------------------------------------------------------------------------------------------

    # Model Tuning Flags ---------------------------------------------------------------------------
    action_model_lora: Annotated[ActionModelLoraArgs, tyro.conf.subcommand("on")] | None = None
    """LoRA adapters on the action model (MSAT): action-model-lora:on."""

    backbone_lora: Annotated[BackboneLoraArgs, tyro.conf.subcommand("on")] | None = None
    """LoRA adapters on the backbone LLM: backbone-lora:on."""

    # ----------------------------------------------------------------------------------------------

    # Video Module (cf. VTC) ----------------------------------------------------------------
    # NOTE: there is intentionally no ``use_video`` CLI knob — the release
    # codebase always builds the VTC backbone (vanilla Qwen3 path was
    # dropped) and every released checkpoint is trained with video.
    # ``video_length`` / ``video_stride`` are the only video-shape parameters
    # left as CLI overrides.
    video_length: int = 4
    """Number of video frames to use as input."""

    video_stride: int = 2
    """Stride (in action step units) between consecutive video frames in the
    context window. Used to compute video frame deltas:
    [(i - (L-1)) * stride for i in range(L)]."""
    # ----------------------------------------------------------------------------------------------

    # Optional modules -----------------------------------------------------------------------------
    motion: Annotated[MotionArgs, tyro.conf.subcommand("on")] | None = None
    """Motion module in the vision encoder: motion:on."""

    memory: Annotated[MemoryArgs, tyro.conf.subcommand("on")] | None = None
    """Memory module over past timesteps: memory:on."""

    physics: Annotated[PhysicsArgs, tyro.conf.subcommand("on")] | None = None
    """Physics (tactile / torque) stream: physics:on --physics.keys … --physics.dims …."""
    # ----------------------------------------------------------------------------------------------

    # Image pipeline -------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    # Data Augmentation ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    # Training Configuration -----------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------

    def __post_init__(self):
        # A LoRA trains only its adapters, so its part takes no tune flag.
        model = self.model
        if self.backbone_lora is not None:
            flags = {
                "--tune-llm": model.tune_llm,
                "--tune-visual": model.tune_visual,
                "--tune-top-llm-layers": model.tune_top_llm_layers,
                "--freeze-cog-tokens": model.freeze_cog_tokens,
            }
            given = [flag for flag, value in flags.items() if value is not None]
            if given:
                raise ValueError(
                    f"backbone-lora:on trains only its adapters; leave out {', '.join(given)}"
                )
        if self.action_model_lora is not None and model.tune_diffusion_model is not None:
            raise ValueError(
                "action-model-lora:on trains only its adapters; leave out --tune-diffusion-model"
            )
