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

from dataclasses import MISSING, Field, asdict, dataclass, field, fields, is_dataclass
from types import MappingProxyType

from transformers import PretrainedConfig

from . import register_model_config
from .rldx_v1 import RLDXConfigFormatV1


FORMAT_VERSION = 2
HF_CONFIG_KEYS = frozenset(PretrainedConfig().to_dict()) | {
    "attn_implementation",
    "torch_dtype",
    "num_labels",
    "name_or_path",
    "_commit_hash",
}


def internal():
    """A field the released tree does not have. None means off: it is not saved, and a v2 file
    without the field loads None."""
    return field(default=None, metadata={"internal": True})


def internal_off(f: Field, value) -> bool:
    """Whether ``f`` is internal and ``value`` is off."""
    return f.metadata.get("internal", False) and value is None


def internal_on(config) -> list[str]:
    """The internal fields of ``config`` and its sub-configs that are on."""
    on = []
    for f in fields(config):
        value = getattr(config, f.name)
        if is_dataclass(value):
            on += [f"{f.name}.{name}" for name in internal_on(value)]
        elif f.metadata.get("internal") and not internal_off(f, value):
            on.append(f.name)
    return on


# MSAT kwargs a saved diffusion_model_cfg may leave out; RLDXActionModel fills them when it
# builds MSAT. A new config starts from these with the shorter max sequence length.
MSAT_DEFAULTS = MappingProxyType(
    {
        "attention_head_dim": 64,
        "depth_multi_stream": 4,
        "depth_single_stream": 8,
        "dropout": 0.2,
        "final_dropout": True,
        "num_attention_heads": 24,
        "output_dim": 1024,
        "positional_embeddings": "rope_sa_only",
        "rope_theta": 10000.0,
        "temb_type": "input_token",
        "use_swiglu": True,
        "action_model_max_seq_len": 1024,
        "pre_norm": "layer_norm",
        "qk_norm": "rms_norm",
    }
)


@dataclass(frozen=True)
class MemoryConfig:
    length: int = 4  # past timesteps kept (= context window)
    n_cog_tokens: int | None = 16  # cognition tokens routed through memory; None = all
    concat: bool = True  # False replaces the routed tokens instead of appending
    dropout_prob: float = 0.0
    stride: int = 16  # action steps between memory snapshots (= execution horizon)
    cfg: dict = field(
        default_factory=lambda: {
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-5,
            "use_causal_attn": True,
            "use_rope": True,
        }
    )

    def __post_init__(self):
        if self.dropout_prob > 0.0 and not self.concat:
            raise ValueError("memory dropout_prob > 0.0 requires concat=True")


@dataclass(frozen=True)
class MotionConfig:
    insert_layer: int = 9
    d_hid: int = 512
    window: tuple[int, ...] = (5, 9, 9)
    ext_chnls: tuple[int, ...] = (256,)
    int_chnls: tuple[int, ...] = (256, 256, 512)
    corr_func: str = "cosine"
    n_encoders: int = 1
    use_layerscale: bool = False
    layerscale_init: float = 1e-5
    use_layernorm: bool = True  # GroupNorm(1); v1 configs keep their BN default (rldx_v1.py)
    use_syncbn: bool = False
    injection_point: str = "vision_encoder"  # "vision_encoder" or "vl_input"
    pool_type: str = "avg"  # "avg" or "conv" (spatial pooling for vl_input)
    drop: bool = True  # drop motion tokens at the internal_projection layer
    gradient_check: bool = False  # log motion module gradient norms during training

    def __post_init__(self):
        # JSON has no tuples.
        for name in ("window", "ext_chnls", "int_chnls"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def vision_config(self) -> dict:
        return {"use_motion": True, **{f"motion_{k}": v for k, v in asdict(self).items()}}


@dataclass(frozen=True)
class PhysicsConfig:
    keys: list[str]  # e.g. ["tactile", "torque"]
    dims: list[int]  # per key, aligned with keys (e.g. [30, 7])
    loss_weight: float = 0.1
    allow_missing: bool = False  # samples without physics are zero-filled and masked
    delta_indices: list[int] | None = None  # set from the modality config; <=0 hist, >0 fut
    use_flow_matching: bool = True  # False: all-conditioning + MSE loss
    dropout_prob: float = 0.0  # flow matching drops history tokens only, MSE the whole sequence

    def __post_init__(self):
        if not self.keys or len(self.dims) != len(self.keys):
            raise ValueError(
                f"physics needs dims for every key; got keys={self.keys}, dims={self.dims}"
            )

    @property
    def dim(self) -> int:
        return sum(self.dims)


@dataclass(frozen=True)
class ActionModelLoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    # MSAT linears; p_qkv/p_proj apply only to an MSAT built with physics.
    target_modules: list[str] = field(
        default_factory=lambda: [
            "vl_qkv",
            "vl_proj",
            "sa_qkv",
            "sa_proj",
            "p_qkv",
            "p_proj",
            "linear1",
            "linear2",
        ]
    )

    def __post_init__(self):
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError(f"action-model-lora needs rank > 0, got {self.rank!r}")
        if type(self.alpha) is not int or self.alpha <= 0:
            raise ValueError(f"action-model-lora needs alpha > 0, got {self.alpha!r}")


@dataclass(frozen=True)
class BackboneLoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    num_layers: int = -1  # top-N LLM layers; -1 = all
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    def __post_init__(self):
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError(f"backbone-lora needs rank > 0, got {self.rank!r}")
        if type(self.alpha) is not int or self.alpha <= 0:
            raise ValueError(f"backbone-lora needs alpha > 0, got {self.alpha!r}")
        if type(self.num_layers) is not int or (self.num_layers <= 0 and self.num_layers != -1):
            raise ValueError(
                f"backbone-lora needs num_layers > 0 or -1 (all), got {self.num_layers!r}"
            )


FEATURE_CONFIGS = {
    "memory": MemoryConfig,
    "motion": MotionConfig,
    "physics": PhysicsConfig,
    "action_model_lora": ActionModelLoraConfig,
    "backbone_lora": BackboneLoraConfig,
}


def default_color_jitter_params() -> dict[str, float]:
    return {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}


@dataclass(frozen=True)
class ImageGeometryConfig:
    max_area: int = 65536  # aspect-preserving resize to this many pixels, 256 * 256 by default
    resize_m: int = 32  # both sides aligned to this multiple
    crop_fraction: float | None = None  # train: random crop, eval: center crop; None = no crop


@dataclass(frozen=True)
class ImageAugmentationConfig:
    rotation_angle: int | None = None
    color_jitter: dict[str, float] | None = field(default_factory=default_color_jitter_params)


IMAGE_CONFIGS = {"image": ImageGeometryConfig, "augmentation": ImageAugmentationConfig}


@dataclass
class RLDXConfig(PretrainedConfig):
    """Unified configuration for RLDX model with backbone and action model."""

    # Model identification
    model_type: str = "RLDX-1"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone configuration
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    # Only "vtc_qwen3_vl" is supported. Any other value raises at
    # model-construction time.
    backbone_model_type: str = "vtc_qwen3_vl"
    model_revision: str | None = None
    tune_top_llm_layers: int = 4  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 4096  # project_to_dim
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 18
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = True  # Enable BF16 loading
    freeze_cog_tokens: bool = False  # Freeze cog_emb to prevent VLM backprop

    image: ImageGeometryConfig = field(default_factory=ImageGeometryConfig)
    augmentation: ImageAugmentationConfig = field(default_factory=ImageAugmentationConfig)
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_relative_action: bool = True
    use_percentiles: bool = True  # True normalizes to q01/q99, False to min/max
    conversation_image_first: bool = False

    # Action head configuration parameters
    max_state_dim: int = 64  # Default from state_shape
    max_action_dim: int = 64  # Default from action_shape
    action_horizon: int = 16
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    general_embodiment_train_ratio: float = 0

    mask_padded_action_dims: bool = True

    # Global parameters from YAML
    add_pos_embed: bool = True
    attn_dropout: float = 0.0
    use_vlln: bool = False
    max_seq_len: int = 1024

    # MSAT configuration
    n_cog_tokens: int = 64
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {**MSAT_DEFAULTS, "action_model_max_seq_len": 512}
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    action_model_lora: ActionModelLoraConfig | None = None
    # A set LoRA trains only its adapters; the tune fields of its part do not apply.
    backbone_lora: BackboneLoraConfig | None = None

    # State Augmentation parameters
    state_dropout_prob: float = 0.3  # State dropout probability
    state_additive_noise_scale: float = 0.0  # Scale for additive Gaussian noise on state features

    # Real-Time Chunking (RTC, rldx/model/modules/action_model/rtc.py): if > 0,
    # each training sample freezes a clean prefix of length U[0, max_delay] and
    # takes no loss on it. How to run RTC is InferenceConfig.rtc.
    rtc_training_max_delay: int = 0

    memory: MemoryConfig | None = None

    motion: MotionConfig | None = None

    # Video input configuration
    # ``use_video`` is an architectural invariant: every supported
    # checkpoint embeds VTC video tokens. The field is kept on
    # ``RLDXConfig`` so it survives ``save_pretrained`` /
    # ``from_pretrained`` round-trips, but it is no longer a CLI knob —
    # see ``rldx/experiment/features/video.py``.
    use_video: bool = True
    video_length: int = 4
    video_stride: int = 2  # Action-step stride between video frames in context window

    physics: PhysicsConfig | None = None

    # Multi-embodiment parameters
    max_num_embodiments: int = 8

    def __init__(self, **kwargs):
        unknown = kwargs.keys() - self.__dataclass_fields__.keys() - HF_CONFIG_KEYS
        if unknown:
            raise TypeError(f"RLDXConfig got unknown fields {sorted(unknown)}")
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        self._fill_missing_defaults()

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        fields = cls.__dataclass_fields__.keys()
        hf = {k: v for k, v in config_dict.items() if k in HF_CONFIG_KEYS and k not in fields}
        saved = {k: v for k, v in config_dict.items() if k not in hf and k != "format_version"}
        fields = cls._fields(config_dict.get("format_version"), saved)
        for name, feature in FEATURE_CONFIGS.items():
            if fields[name] is not None:
                fields[name] = feature(**fields[name])
        for name, sub_config in IMAGE_CONFIGS.items():
            fields[name] = sub_config(**fields[name])
        return super().from_dict({**hf, **fields}, **kwargs)

    @classmethod
    def _fields(cls, version, saved: dict) -> dict:
        saved = _drop_removed_keys(saved)
        if version is None:
            return RLDXConfigFormatV1.to_v2(saved)
        if version != FORMAT_VERSION:
            raise ValueError(f"unknown format_version {version!r}")
        declared = cls.__dataclass_fields__
        required = {name for name, f in declared.items() if not f.metadata.get("internal")}
        missing, unknown = sorted(required - saved.keys()), sorted(saved.keys() - declared.keys())
        if missing or unknown:
            raise ValueError(
                f"format_version {version} config is missing {missing}, has unknown {unknown}"
            )
        return saved

    def to_dict(self):
        return {k: _plain(v) for k, v in super().to_dict().items()}

    def to_diff_dict(self):
        hf = {k: v for k, v in super().to_diff_dict().items() if k not in self.__dataclass_fields__}
        return {**hf, **_saved(self), "format_version": FORMAT_VERSION}

    def _fill_missing_defaults(self):
        """Set default values for any dataclass fields not yet on the instance."""
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif f.default_factory is not MISSING:
                    setattr(self, f.name, f.default_factory())

    def __getattr__(self, name: str):
        # Strict: no silent default fallback. Every declared dataclass
        # field is populated by `_fill_missing_defaults()` in __init__,
        # so reaching __getattr__ means the attribute was never declared
        # on this class — fail fast.
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# diffusion_model_cfg keys of removed features and the value that had them off.
REMOVED_MSAT_KEYS = MappingProxyType({"use_blockwise_causal_mask": False, "meta_q_dropout": 0.0})


def _drop_removed_keys(saved: dict) -> dict:
    """Drops the keys of removed features when they were off; refuses a config that used one."""
    saved = dict(saved)
    if saved.pop("extra_augmentation_config", None) is not None:
        raise ValueError("config uses removed extra_augmentation_config (mask-guided augmentation)")
    if "augmentation" in saved:
        augmentation = dict(saved["augmentation"])
        if augmentation.pop("mask_guided", None) is not None:
            raise ValueError("config uses removed augmentation.mask_guided")
        saved["augmentation"] = augmentation
    if "normalization_type" in saved:
        mode = saved.pop("normalization_type")
        use_percentiles = saved.get("use_percentiles", True)
        if mode != ("q99" if use_percentiles else "minmax"):
            raise ValueError(
                f"config normalizes with normalization_type={mode!r}, "
                f"use_percentiles={use_percentiles}; only q99 (use_percentiles=True) and "
                "minmax (use_percentiles=False) are supported"
            )
    if "diffusion_model_cfg" in saved:
        msat = dict(saved["diffusion_model_cfg"])
        for key, off in REMOVED_MSAT_KEYS.items():
            if msat.pop(key, off) != off:
                raise ValueError(f"config uses removed diffusion_model_cfg.{key}")
        saved["diffusion_model_cfg"] = msat
    return saved


def _plain(value):
    sub_configs = (*FEATURE_CONFIGS.values(), *IMAGE_CONFIGS.values())
    return asdict(value) if value is not None and type(value) in sub_configs else value


def _saved(config) -> dict:
    """The fields of ``config`` to write, with internal fields that are off left out."""
    saved = {}
    for f in fields(config):
        value = getattr(config, f.name)
        if not internal_off(f, value):
            saved[f.name] = _saved(value) if is_dataclass(value) else value
    return saved


register_model_config("RLDX-1", RLDXConfig)
