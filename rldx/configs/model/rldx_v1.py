# SPDX-License-Identifier: Apache-2.0
"""The v1 ``config.json`` format: what each left-out field loads as, as of 2026-10-16."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from types import MappingProxyType
from typing import Any


class RLDXConfigFormatV1:
    DEFAULTS: Mapping[str, Any] = MappingProxyType(
        {
            "model_type": "RLDX-1",
            "model_dtype": "bfloat16",
            "model_name": "Qwen/Qwen3-VL-8B-Instruct",
            "backbone_model_type": "vtc_qwen3_vl",
            "model_revision": None,
            "tune_top_llm_layers": 4,
            "backbone_embedding_dim": 4096,
            "tune_llm": False,
            "tune_visual": False,
            "select_layer": 18,
            "reproject_vision": False,
            "use_flash_attention": True,
            "load_bf16": True,
            "freeze_cog_tokens": False,
            "image_max_area": 65536,
            "image_resize_m": 32,
            "random_crop_fraction": None,
            "random_rotation_angle": None,
            "color_jitter_params": {
                "brightness": 0.3,
                "contrast": 0.4,
                "saturation": 0.5,
                "hue": 0.08,
            },
            "formalize_language": True,
            "apply_sincos_state_encoding": False,
            "use_relative_action": True,
            "use_percentiles": True,
            "conversation_image_first": False,
            "max_state_dim": 64,
            "max_action_dim": 64,
            "action_horizon": 16,
            "hidden_size": 1024,
            "input_embedding_dim": 1536,
            "general_embodiment_train_ratio": 0,
            # Checkpoints saved before the key existed were trained unmasked.
            "mask_padded_action_dims": False,
            "add_pos_embed": True,
            "attn_dropout": 0.0,
            "use_vlln": False,
            "max_seq_len": 1024,
            "n_cog_tokens": 64,
            "diffusion_model_cfg": {
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
                "action_model_max_seq_len": 512,
                "pre_norm": "layer_norm",
                "qk_norm": "rms_norm",
            },
            "num_inference_timesteps": 4,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "tune_projector": True,
            "tune_diffusion_model": True,
            "tune_vlln": True,
            "action_model_use_lora": False,
            "action_model_lora_rank": 16,
            "action_model_lora_alpha": 32,
            "action_model_lora_dropout": 0.0,
            "action_model_lora_target_modules": [
                "vl_qkv",
                "vl_proj",
                "sa_qkv",
                "sa_proj",
                "p_qkv",
                "p_proj",
                "linear1",
                "linear2",
            ],
            "backbone_use_lora": False,
            "backbone_lora_rank": 16,
            "backbone_lora_alpha": 32,
            "backbone_lora_dropout": 0.0,
            "backbone_lora_num_layers": -1,
            "backbone_lora_target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "state_dropout_prob": 0.3,
            "state_additive_noise_scale": 0.0,
            "rtc_training_max_delay": 0,
            "rtc_inference_mode": "none",
            "rtc_inference_delay": 0,
            "rtc_inference_exec_horizon": 0,
            "rtc_jacobian_steps_only": 3,
            "use_memory": False,
            "memory_length": 4,
            "memory_n_cog_tokens": None,
            "concat_memory": False,
            "memory_dropout_prob": 0.0,
            "memory_stride": 16,
            "memory_cfg": {
                "hidden_size": 4096,
                "intermediate_size": 16384,
                "num_hidden_layers": 2,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "max_position_embeddings": 32,
                "rms_norm_eps": 1e-05,
                "use_causal_attn": True,
                "use_rope": True,
            },
            "use_motion": False,
            "motion_insert_layer": 9,
            "motion_d_hid": 512,
            "motion_window": (5, 9, 9),
            "motion_ext_chnls": (256,),
            "motion_int_chnls": (256, 256, 512),
            "motion_corr_func": "cosine",
            "motion_n_encoders": 1,
            "motion_use_layerscale": False,
            "motion_layerscale_init": 1e-05,
            "motion_use_layernorm": False,
            "motion_use_syncbn": False,
            "motion_injection_point": "vision_encoder",
            "motion_pool_type": "avg",
            "motion_drop": True,
            "motion_gradient_check": False,
            "motion_int_mode": "lite",
            "use_video": True,
            "video_length": 4,
            "video_stride": 2,
            "use_physics": False,
            "physics_keys": [],
            "physics_dims": [],
            "physics_loss_weight": 0.1,
            "allow_missing_physics": False,
            "physics_delta_indices": None,
            "physics_use_flow_matching": True,
            "physics_dropout_prob": 0.0,
            # Checkpoints saved before the key existed had 36 projector slots.
            "max_num_embodiments": 36,
        }
    )
    DROPPED_IN_V2 = frozenset(
        {
            "motion_int_mode",
            # How to run RTC is a serving choice now (InferenceConfig.rtc).
            "rtc_inference_mode",
            "rtc_inference_delay",
            "rtc_inference_exec_horizon",
            "rtc_jacobian_steps_only",
        }
    )
    # v2 field -> (v1 on/off flag, {v2 sub-config field: v1 flat key})
    FEATURES: Mapping[str, tuple[str, Mapping[str, str]]] = MappingProxyType(
        {
            "memory": (
                "use_memory",
                {
                    "length": "memory_length",
                    "n_cog_tokens": "memory_n_cog_tokens",
                    "concat": "concat_memory",
                    "dropout_prob": "memory_dropout_prob",
                    "stride": "memory_stride",
                    "cfg": "memory_cfg",
                },
            ),
            "motion": (
                "use_motion",
                {
                    field: f"motion_{field}"
                    for field in (
                        "insert_layer",
                        "d_hid",
                        "window",
                        "ext_chnls",
                        "int_chnls",
                        "corr_func",
                        "n_encoders",
                        "use_layerscale",
                        "layerscale_init",
                        "use_layernorm",
                        "use_syncbn",
                        "injection_point",
                        "pool_type",
                        "drop",
                        "gradient_check",
                    )
                },
            ),
            "physics": (
                "use_physics",
                {
                    "keys": "physics_keys",
                    "dims": "physics_dims",
                    "loss_weight": "physics_loss_weight",
                    "allow_missing": "allow_missing_physics",
                    "delta_indices": "physics_delta_indices",
                    "use_flow_matching": "physics_use_flow_matching",
                    "dropout_prob": "physics_dropout_prob",
                },
            ),
            "action_model_lora": (
                "action_model_use_lora",
                {
                    field: f"action_model_lora_{field}"
                    for field in ("rank", "alpha", "dropout", "target_modules")
                },
            ),
            "backbone_lora": (
                "backbone_use_lora",
                {
                    field: f"backbone_lora_{field}"
                    for field in ("rank", "alpha", "dropout", "num_layers", "target_modules")
                },
            ),
        }
    )

    # v2 field -> {v2 sub-config field: v1 flat key}; always present, unlike FEATURES.
    GROUPS: Mapping[str, Mapping[str, str]] = MappingProxyType(
        {
            "image": {
                "max_area": "image_max_area",
                "resize_m": "image_resize_m",
                "crop_fraction": "random_crop_fraction",
            },
            "augmentation": {
                "rotation_angle": "random_rotation_angle",
                "color_jitter": "color_jitter_params",
            },
        }
    )

    # Copies of top-level fields (physics, attn_dropout) and keys MSAT never took.
    DIFFUSION_KEYS_DROPPED_IN_V2 = frozenset(
        {
            "use_physics",
            "physics_dim",
            "attn_dropout",
            "random_meta_q_mask_p",
            "set_triple_stream_for_mq",
            "set_triple_stream_for_state",
            "state_dim",
            "action_dim",
            "mq_dim",
            "state_mlp_ratio",
            "action_mlp_ratio",
            "mq_mlp_ratio",
        }
    )

    @classmethod
    def to_v2(cls, saved: Mapping[str, Any]) -> dict[str, Any]:
        known = {k: v for k, v in saved.items() if k in cls.DEFAULTS}
        # Some v1 configs record the motion extractor's norm; v1 loading builds only BatchNorm.
        extraction_norm = saved.get("motion_extraction_norm", "batch")
        if saved.get("use_motion") and extraction_norm != "batch":
            raise ValueError(
                f"This v1 checkpoint's motion extractor uses motion_extraction_norm="
                f"{extraction_norm!r}; loading v1 configs supports only 'batch' (BatchNorm)."
            )
        filled = {**copy.deepcopy(dict(cls.DEFAULTS)), **known}
        v2 = {k: v for k, v in filled.items() if k not in cls.DROPPED_IN_V2}
        v2["diffusion_model_cfg"] = {
            k: v
            for k, v in v2["diffusion_model_cfg"].items()
            if k not in cls.DIFFUSION_KEYS_DROPPED_IN_V2
        }
        for name, (flag, keys) in cls.FEATURES.items():
            values = {field: v2.pop(key) for field, key in keys.items()}
            v2[name] = values if v2.pop(flag) else None
        for name, keys in cls.GROUPS.items():
            v2[name] = {field: v2.pop(key) for field, key in keys.items()}
        return v2
