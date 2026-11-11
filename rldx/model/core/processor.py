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

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
import json
import logging
from pathlib import Path
import random
import re
from typing import Any
import warnings

import albumentations as A
import cv2
import numpy as np
from rldx.configs.data.embodiment_configs import ModalityConfig
from rldx.configs.model.rldx import PhysicsConfig, RLDXConfig
from rldx.data.aug_snapshot import AugSnapshotter
from rldx.data.augmentations import (
    apply_with_replay,
    build_image_transformations_albumentations,
    resize_preserve_aspect_area_then_crop,
)
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.interfaces import BaseProcessor
from rldx.data.state_action.layout import join, split
from rldx.data.state_action.state_action_processor import StateActionProcessor
from rldx.data.utils import parse_modality_configs, to_json_serializable, write_json_atomic
from rldx.dist import rank_zero_print as _print
from rldx.model.core.collator import PromptEncoder, RLDXDataCollator
from rldx.model.core.slot_registry import EMBODIMENT_TAG_TO_PROJECTOR_INDEX, PROJECTOR_GROUPINGS
from rldx.planner.input.tensor_image import TensorImageSpec
from rldx.telemetry.step_profiler import profile_section as _profile
import torch
import torchvision.transforms.v2 as transforms
from transformers.utils import cached_file


FORMAT_VERSION = 2
# Keys of removed features, ignored when a saved file still has them. active_embodiment_tags
# only chose which embodiments q999 normalization refused to run without q001/q999 stats.
REMOVED_KEYS = frozenset({"active_embodiment_tags"})


@dataclass(frozen=True)
class RLDXProcessorConfig:
    modality_configs: dict[str, dict[str, ModalityConfig]]


class RLDXProcessorFormatV1:
    # Settings these files carried a copy of. config.json holds them; the copies are not read.
    SETTINGS = frozenset(
        {
            "image_max_area",
            "image_resize_m",
            "random_crop_fraction",
            "random_rotation_angle",
            "color_jitter_params",
            "extra_augmentation_config",
            "model_name",
            "model_type",
            "formalize_language",
            "max_state_dim",
            "max_action_dim",
            "max_action_horizon",
            "normalization_type",
            "use_percentiles",
            "clip_outliers",
            "apply_sincos_state_encoding",
            "use_relative_action",
            "memory_length",
            "general_embodiment_train_ratio",
            "allow_missing_physics",
            "physics_keys",
            "physics_dims",
            "conversation_image_first",
        }
    )

    @classmethod
    def to_config(cls, saved: Mapping[str, Any]) -> RLDXProcessorConfig:
        unknown = saved.keys() - cls.SETTINGS - {"modality_configs", *REMOVED_KEYS}
        if unknown:
            raise ValueError(f"processor_config.json has unknown keys {sorted(unknown)}")
        return RLDXProcessorConfig(
            modality_configs=parse_modality_configs(saved["modality_configs"])
        )


warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")


def build_projector_mapping(grouping: str | None) -> dict[str, int] | None:
    """Build an optional experiment-specific tag -> projector mapping.

    ``None`` deliberately means "use the normal mapping, or preserve the
    checkpoint mapping" so ordinary fine-tuning remains unchanged.
    """
    if grouping is None:
        return None
    groups = PROJECTOR_GROUPINGS.get(grouping)
    if groups is None:
        raise ValueError(f"Unknown projector grouping {grouping!r}")

    mapping = dict(EMBODIMENT_TAG_TO_PROJECTOR_INDEX)
    mapping.update(groups)
    return mapping


# Checkpoints written by ``rldx/experiment/utils.py`` keep the processor files in
# a ``processor/`` subdirectory; ``save_pretrained`` writes them flat into its target.
PROCESSOR_SUBDIR = "processor"


def resolve_processor_files(
    pretrained_model_name_or_path: str | Path,
) -> tuple[str, str, str]:
    """Locate a checkpoint's processor files, honoring the ``processor/`` subdir.

    Returns the resolved ``(processor_config.json, statistics.json,
    embodiment_id.json)`` paths. All three are required: every producer writes
    them together — ``save_pretrained`` below, and upstream GR00T N1.7, which
    this loader derives from — so one missing means a damaged or hand-assembled
    tree. Tolerating an absent ``embodiment_id.json`` would fall back to the
    built-in ``EMBODIMENT_TAG_TO_PROJECTOR_INDEX``, which silently infers
    through the wrong projector slot for a checkpoint trained with a custom map.

    Probes ``processor/`` before the root, so a checkpoint resolves to the
    same processor whichever entry point loads it. The root is the layout
    ``save_pretrained`` itself writes, and the one ``RLDX-1-MT-DROID`` ships.

    ``cached_file`` serves local directories and Hub repo ids through one call,
    so the two cannot drift apart here — the earlier local/Hub split is what let
    the Hub branch miss the subdir convention entirely.

    The base is decided **once**, by where ``processor_config.json`` lives, and
    the other two files are read from that same base. Falling back per file would
    let a partially populated root — the
    ``ln -s processor/processor_config.json .`` workaround applied to only some
    of the three — silently pair a root config with subdir statistics.
    """

    def _locate(filename: str, subfolder: str) -> str | None:
        # ``_raise_exceptions_for_missing_entries=False`` returns None for a
        # missing file while still raising on gated repos and connection
        # errors. Catching OSError instead would swallow network failures:
        # ``requests``' exceptions subclass OSError.
        return cached_file(
            pretrained_model_name_or_path,
            filename,
            subfolder=subfolder,
            _raise_exceptions_for_missing_entries=False,
        )

    for subfolder in (PROCESSOR_SUBDIR, ""):
        config_file = _locate("processor_config.json", subfolder)
        if config_file is not None:
            break
    else:
        raise OSError(
            f"{pretrained_model_name_or_path} does not appear to have a file named "
            f"processor_config.json — looked in '{PROCESSOR_SUBDIR}/' and at the root."
        )

    base = (
        f"{pretrained_model_name_or_path}/{subfolder}"
        if subfolder
        else str(pretrained_model_name_or_path)
    )
    resolved = {
        name: _locate(name, subfolder) for name in ("statistics.json", "embodiment_id.json")
    }
    missing = [name for name, found in resolved.items() if found is None]
    if missing:
        raise OSError(
            f"{base} has processor_config.json but no {', '.join(missing)}. The three "
            "processor files are written together and must be read from one directory: "
            "resolving them from different ones would mix two processors, and "
            "substituting a default for a missing one would infer through the wrong "
            "projector slot without any error surface."
        )
    return config_file, resolved["statistics.json"], resolved["embodiment_id.json"]


def _physics_length(config: RLDXProcessorConfig, physics: PhysicsConfig) -> int:
    lengths = (
        len(modalities[key].delta_indices)
        for modalities in config.modality_configs.values()
        for key in physics.keys
        if key in modalities
    )
    return next((n for n in lengths if n), 0)


class RLDXProcessor(BaseProcessor):
    data_collator_class = RLDXDataCollator

    def __init__(
        self,
        config: RLDXProcessorConfig,
        rldx_config: RLDXConfig,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        embodiment_id_mapping: dict[str, int] | None = None,
        transformers_loading_kwargs: dict = {"trust_remote_code": True, "use_fast": True},
    ):
        self.config = config
        self.rldx_config = rldx_config
        physics = rldx_config.physics
        self._physics_t_len = (
            _physics_length(config, physics) if physics is not None and physics.allow_missing else 0
        )

        self.state_action_processor = StateActionProcessor(
            modality_configs=config.modality_configs,
            statistics=statistics,
            use_percentiles=rldx_config.use_percentiles,
            apply_sincos_state_encoding=rldx_config.apply_sincos_state_encoding,
            use_relative_action=rldx_config.use_relative_action,
        )

        # Env-gated PNG pair dumper (no-op unless RLDX_DUMP_AUG_* env vars set).
        self._snapshotter = AugSnapshotter()

        self.embodiment_id_mapping = embodiment_id_mapping or EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        # handle the case where the fine-tuning embodiment tag is not in the pre-trained embodiment tag mapping
        for k, v in EMBODIMENT_TAG_TO_PROJECTOR_INDEX.items():
            if k not in self.embodiment_id_mapping:
                self.embodiment_id_mapping[k] = v

        self.train_image_transform = build_image_transformations_albumentations(
            rldx_config.image, rldx_config.augmentation
        )
        # Per-input-shape cache for the eval fast path. Keys are (h_in, w_in),
        # values are (h_resize, w_resize, h_crop, w_crop) precomputed once
        # from the same formula AspectAreaResizeAndCrop uses internally.
        # For the deployed 640×480 camera + max_area=65536 / m=32 this resolves
        # to 192×256 (H×W) — i.e. 256×192 (W×H) — and that result is reused
        # forever, bypassing the 47 ms jitter from albumentations' format
        # conversion chain (see commits 95cadaf5 / 736ab811).
        self._eval_resize_cache: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        self._collator = self.data_collator_class(
            PromptEncoder.load(
                rldx_config.model_name,
                transformers_loading_kwargs,
                image_max_area=rldx_config.image.max_area,
                image_first=rldx_config.conversation_image_first,
            )
        )

        self.train()

    @property
    def collator(self):
        return self._collator

    @property
    def train_image_spec(self) -> TensorImageSpec:
        return self._collator.encoder.tensor_image_path

    @property
    def eval_image_spec(self) -> TensorImageSpec:
        return replace(self.train_image_spec, crop_fraction=self.rldx_config.image.crop_fraction)

    def train(self):
        super().train()
        self.state_action_processor.train()

    def eval(self):
        super().eval()
        self.state_action_processor.eval()

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """Set dataset statistics for normalization."""
        self.state_action_processor.set_statistics(statistics, override=override)

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ):
        """Undo action normalization and convert relative actions to absolute."""
        action_horizon = len(
            self.config.modality_configs[embodiment_tag.value]["action"].delta_indices
        )
        layout = self.state_action_processor.layout(embodiment_tag.value, "action")
        return self.state_action_processor.unapply_action(
            split(action[..., :action_horizon, :], layout), embodiment_tag.value, state=state
        )

    def __call__(
        self,
        messages: list[dict[str, Any]],
    ):
        assert len(messages) == 1
        content = messages[0]["content"]
        embodiment_tag = content.embodiment
        action_data = content.actions
        state_data = content.states

        # Use StateActionProcessor to handle relative conversion and normalization
        with _profile("processor.state_action_normalize"):
            normalized_states, normalized_actions = self.state_action_processor.apply(
                state=state_data,
                action=action_data,
                embodiment_tag=embodiment_tag.value,
            )

        if normalized_actions:
            action_layout = self.state_action_processor.layout(embodiment_tag.value, "action")
            normalized_actions = torch.from_numpy(join(normalized_actions, action_layout))
            action_dim = normalized_actions.shape[1]
            # Pad action to max_action_dim
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        normalized_actions.shape[0],
                        self.rldx_config.max_action_dim - normalized_actions.shape[1],
                    ),
                ],
                dim=-1,
            )  # (t, max_action_dim)
            # Pad action to max_action_horizon
            action_horizon = normalized_actions.shape[0]
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        self.rldx_config.action_horizon - normalized_actions.shape[0],
                        self.rldx_config.max_action_dim,
                    ),
                ],
                dim=0,
            )  # (max_action_horizon, max_action_dim)
            # Create action mask. ``action_horizon`` here is the chunk's
            # actual length (after pad-to-max_action_horizon). End-of-episode
            # samples carry ``valid_action_horizon`` < action_horizon in
            # metadata so the loss skips chunk positions that ran past the
            # episode boundary (action targets there were clamp- or
            # zero-padded by ``extract_step_data`` and are not real labels).
            valid_action_horizon = int(content.metadata.get("valid_action_horizon", action_horizon))
            valid_action_horizon = max(0, min(valid_action_horizon, action_horizon))
            action_mask = torch.ones_like(normalized_actions)
            action_mask[action_horizon:] = 0
            action_mask[:, action_dim:] = 0
            if valid_action_horizon < action_horizon:
                action_mask[valid_action_horizon:action_horizon] = 0

            # PAD lifts the boundary out of ``action_mask`` on purpose, but physics
            # targets past the episode end are clamp-repeats, not labels. Keep the
            # boundary here so the physics loss masks them under either policy.
            in_episode = int(
                content.metadata.get("in_episode_action_horizon", valid_action_horizon)
            )
            in_episode = max(0, min(in_episode, action_horizon))
            action_boundary_mask = action_mask.clone()
            if in_episode < action_horizon:
                action_boundary_mask[in_episode:action_horizon] = 0
        else:
            assert not self.training, "Action is required in training mode"
            normalized_actions = None
            action_mask = None
            action_boundary_mask = None
            action_dim = self.state_action_processor.get_action_dim(embodiment_tag.value)

        action_dim_mask = torch.zeros(
            self.rldx_config.action_horizon, self.rldx_config.max_action_dim
        )
        action_dim_mask[:, :action_dim] = 1

        state_layout = self.state_action_processor.layout(embodiment_tag.value, "state")
        normalized_states = torch.from_numpy(join(normalized_states, state_layout))
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(
                    normalized_states.shape[0],
                    self.rldx_config.max_state_dim - normalized_states.shape[1],
                ),
            ],
            dim=-1,
        )

        image_keys = self.config.modality_configs[embodiment_tag.value]["video"].modality_keys

        if self.rldx_config.formalize_language:
            language = content.text.lower()
            language = re.sub(r"[^\w\s]", "", language)
        else:
            language = content.text

        with _profile("processor.image_preproc_total"):
            vlm_inputs = self._get_vlm_inputs(
                image_keys=image_keys,
                images=content.images,
                image_transform=self.train_image_transform if self.training else None,
                language=language,
                memory_length=self.memory_length,
            )

        transformed_inputs = {}
        transformed_inputs.update(vlm_inputs)
        if normalized_states is not None:
            transformed_inputs["state"] = normalized_states.to(torch.get_default_dtype())
        if normalized_actions is not None:
            transformed_inputs["action"] = normalized_actions.to(torch.get_default_dtype())
        if action_mask is not None:
            transformed_inputs["action_mask"] = action_mask
            transformed_inputs["action_boundary_mask"] = action_boundary_mask
        transformed_inputs["action_dim_mask"] = action_dim_mask

        # Process physics signals (tactile, torque, etc.) if present
        physics_available = False
        physics = self.rldx_config.physics
        if physics is not None:
            physics_input = content.physics
            filtered_physics = {
                k: v
                for k, v in physics_input.items()
                if any(k == pk or k.startswith(pk + ".") for pk in physics.keys)
            }
            if not filtered_physics and not physics.allow_missing:
                raise ValueError(
                    f"Model requires physics keys {physics.keys} but the "
                    f"sample carries no matching physics data "
                    f"(got keys: {sorted(physics_input)}). A physics-less "
                    "sample cannot be processed for this checkpoint unless "
                    "allow_missing_physics is set."
                )
            if filtered_physics:
                normalized_physics = self.state_action_processor.apply_physics(
                    filtered_physics, embodiment_tag.value
                )
                # apply_physics always returns a dict (possibly empty when every
                # requested modality was dropped as all-zero). The previous
                # `None` sentinel was a symptom of the mid-loop early-return
                # that silently discarded sibling valid modalities.
                assert isinstance(normalized_physics, dict), (
                    f"apply_physics must return a dict; got {type(normalized_physics)!r}"
                )
                # Validate per-key dims and concat in physics_keys order. Any
                # missing modality (apply_physics dropped it as all-zero, or the
                # dataset never shipped it) invalidates the whole physics tensor:
                # concatenating a subset silently produces a tensor smaller than
                # sum(physics_dims), which then mismatches the model's
                # physics_cond_encoder input dim downstream. Require ALL keys —
                # a partial set is an error unless allow_missing_physics
                # zero-fills below.
                per_key_tensors = []
                partial_physics = False
                for pk, expected_dim in zip(physics.keys, physics.dims):
                    # Sub-keys for this physics key (e.g. "tactile" → "tactile.left", "tactile.right")
                    sub_keys = sorted(
                        k for k in normalized_physics if k == pk or k.startswith(pk + ".")
                    )
                    if not sub_keys:
                        if not physics.allow_missing:
                            raise ValueError(
                                f"Physics key '{pk}' produced no data: it is "
                                "missing from the sample or was dropped as "
                                "all-zero by apply_physics. Discarding the "
                                "whole physics tensor and inferring without "
                                "physics would silently change model behavior."
                            )
                        partial_physics = True
                        break
                    key_tensor = torch.cat(
                        [torch.from_numpy(normalized_physics[k]) for k in sub_keys], dim=-1
                    )
                    actual_dim = key_tensor.shape[-1]
                    assert actual_dim == expected_dim, (
                        f"Physics dim mismatch for '{pk}': data has {actual_dim} "
                        f"but --physics.dims specifies {expected_dim}. "
                        f"Sub-keys: {sub_keys}"
                    )
                    per_key_tensors.append(key_tensor)

                if per_key_tensors and not partial_physics:
                    physics_tensor = torch.cat(per_key_tensors, dim=-1)  # (T, sum(physics_dims))
                    transformed_inputs["physics"] = physics_tensor.to(torch.get_default_dtype())
                    physics_available = True

        # When allow_missing_physics=True, always produce physics + physics_mask
        if physics is not None and physics.allow_missing:
            if physics_available:
                transformed_inputs["physics_mask"] = np.array([1.0], dtype=np.float32)
            else:
                if filtered_physics:
                    # The sample DID carry physics but it degraded to the
                    # zero-fill path (all-zero window dropped by apply_physics,
                    # or a partial group set). Training-consistent behavior,
                    # but never silent: the caller believes conditioning is on.
                    logging.warning(
                        "[RLDXProcessor] physics supplied but zero-filled "
                        f"(mask=0): keys={sorted(filtered_physics)} were "
                        "dropped as all-zero or left the set partial"
                    )
                transformed_inputs["physics"] = torch.zeros(
                    self._physics_t_len, physics.dim, dtype=torch.get_default_dtype()
                )
                transformed_inputs["physics_mask"] = np.array([0.0], dtype=np.float32)

        # Training-only: with probability `general_embodiment_train_ratio`, route
        # this sample through the `general_embodiment` projector so that it keeps
        # receiving gradient even when the active mix does not carry
        # `GENERAL_EMBODIMENT` data. Must be gated on `self.training` — otherwise
        # inference on any checkpoint that shipped with a non-zero ratio (e.g.
        # `RLWRLD/RLDX-1-PT` which was saved with 0.03125) would
        # stochastically swap the embodiment id, producing wrong-projector
        # predictions on ~ratio% of requests.
        if (
            self.training
            and self.rldx_config.general_embodiment_train_ratio != 0
            and random.random() < self.rldx_config.general_embodiment_train_ratio
        ):
            transformed_inputs["embodiment_id"] = self.embodiment_id_mapping["general_embodiment"]
        else:
            transformed_inputs["embodiment_id"] = self.embodiment_id_mapping[embodiment_tag.value]

        return transformed_inputs

    def _eval_geometry(self, h_in: int, w_in: int) -> tuple[int, int, int, int]:
        cached = self._eval_resize_cache.get((h_in, w_in))
        if cached is None:
            (h_r, w_r), (h_c, w_c) = resize_preserve_aspect_area_then_crop(
                h_in,
                w_in,
                max_area=self.rldx_config.image.max_area,
                m=self.rldx_config.image.resize_m,
            )
            cached = self._eval_resize_cache[(h_in, w_in)] = (h_r, w_r, h_c, w_c)
            _print(
                f"[RLDXProcessor] eval fast path: ({h_in}, {w_in}) → "
                f"resize ({h_r}, {w_r}) → crop ({h_c}, {w_c})"
            )
        return cached

    def _eval_resize_frame(self, frame: np.ndarray, geometry) -> np.ndarray:
        h_r, w_r, h_c, w_c = geometry
        if frame.shape[:2] != (h_r, w_r):
            # cv2.resize takes (W, H); preserve byte-identity with
            # AspectAreaResizeAndCrop (INTER_AREA).
            frame = cv2.resize(frame, (w_r, h_r), interpolation=cv2.INTER_AREA)
        if (h_r, w_r) != (h_c, w_c):
            y0 = (h_r - h_c) // 2
            x0 = (w_r - w_c) // 2
            frame = frame[y0 : y0 + h_c, x0 : x0 + w_c]
        return frame

    def _eval_fast_resize_views(
        self,
        *,
        image_keys: list[str],
        images: dict[str, np.ndarray | list[np.ndarray]],
    ) -> dict[str, torch.Tensor]:
        """Eval resize: the deterministic first step of the training transform.

        For each view, resize every frame with ``cv2.INTER_AREA`` to the
        m-aligned target HW computed once via ``resize_preserve_aspect_area_then_crop``
        (the same formula the training pipeline's ``AspectAreaResizeAndCrop``
        applies). The target is cached per input shape so subsequent frames
        skip the geometry math.

        Returns the same shape contract as the training path:
        ``dict[view] -> (T, H, W, 3) uint8 torch.Tensor``.
        """
        out: dict[str, torch.Tensor] = {}
        for view in image_keys:
            assert view in images, f"{view} not in {images}"
            frames = images[view]
            h_in, w_in = frames[0].shape[:2]
            geometry = self._eval_geometry(h_in, w_in)
            if geometry == (h_in, w_in, h_in, w_in):
                with _profile("vlm.eval_fast.torch_phase"):
                    out[view] = torch.from_numpy(np.asarray(frames))
                continue
            with _profile("vlm.eval_fast.numpy_phase"):
                stacked = np.stack([self._eval_resize_frame(frame, geometry) for frame in frames])
            with _profile("vlm.eval_fast.torch_phase"):
                out[view] = torch.from_numpy(stacked)
        return out

    def _get_vlm_inputs(
        self,
        image_keys: list[str],
        images: dict[str, np.ndarray | list[np.ndarray]],
        image_transform: transforms.Compose | A.Compose,
        language: str,
        memory_length: int = 1,
    ):
        # Eval only resizes on the host; the center crop runs on the GPU with the
        # patchify (``eval_image_spec``).
        if self.training:
            temporal_stacked_images = {}

            # Albumentations-based pipeline with replay so all views share the
            # same stochastic params (rotation, jitter, crop origin, ...).
            replay = None
            with _profile("vlm.albumentations_apply_per_view_total"):
                for view in image_keys:
                    assert view in images, f"{view} not in {images}"
                    transformed_images, replay = apply_with_replay(
                        image_transform, images[view], replay
                    )
                    temporal_stacked_images[view] = torch.stack(transformed_images)  # (T, H, W, C)
        else:
            with _profile("vlm.eval_fast_resize_per_view_total"):
                temporal_stacked_images = self._eval_fast_resize_views(
                    image_keys=image_keys, images=images
                )

        self._snapshotter.maybe_dump(
            orig_by_view={v: images[v] for v in image_keys},
            xfm_by_view={v: list(temporal_stacked_images[v]) for v in image_keys},
        )

        for k, v in temporal_stacked_images.items():
            assert isinstance(k, str), f"{k} is not a string"
            assert isinstance(v, torch.Tensor), f"{v} is not a torch tensor"
            assert v.ndim == 4, f"{v} is not a 4D tensor"
            assert v.dtype == torch.uint8, f"{v} is not a uint8 tensor"
            assert v.shape[-1] == 3, f"{v} is not a 3 channel tensor"

        T = temporal_stacked_images[image_keys[0]].shape[0]

        if memory_length > 1 and self.training:
            # Memory mode: one vlm_content per timestep for per-timestep backbone processing.
            # For combined mode (video+memory), T will be memory_length * video_length
            video_length = T // memory_length
            return {
                "vlm_content_list": [
                    self._vlm_content(
                        [
                            temporal_stacked_images[view][t * video_length : (t + 1) * video_length]
                            for view in image_keys
                        ],
                        language,
                    )
                    for t in range(memory_length)
                ]
            }
        return {
            "vlm_content": self._vlm_content(
                [temporal_stacked_images[view] for view in image_keys], language
            )
        }

    def _vlm_content(self, blocks: list[torch.Tensor], language: str) -> dict:
        return {
            "prompt": language,
            "image_blocks": blocks,
            "image_first": self.rldx_config.conversation_image_first,
            "image_wise_encoding": True,
            "num_views": len(blocks),
            "num_frames": blocks[0].shape[0],
        }

    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        # dump modality configs to dict using the recursive function
        save_directory.mkdir(parents=True, exist_ok=True)
        main_config_file = Path(save_directory) / "processor_config.json"
        statistics_file = Path(save_directory) / "statistics.json"
        embodiment_id_file = Path(save_directory) / "embodiment_id.json"

        config = {
            "processor_class": self.__class__.__name__,
            "processor_kwargs": {
                "format_version": FORMAT_VERSION,
                **to_json_serializable(asdict(self.config)),
            },
        }
        # Atomic installs. The single-writer guarantee is the caller's -- see the
        # rank guard at the call site.
        write_json_atomic(main_config_file, config, indent=2)
        # Save statistics
        write_json_atomic(
            statistics_file,
            to_json_serializable(self.state_action_processor.statistics),
            indent=2,
        )
        # Save embodiment id mapping
        write_json_atomic(embodiment_id_file, self.embodiment_id_mapping, indent=2)
        return [main_config_file, statistics_file, embodiment_id_file]

    @staticmethod
    def load_config(pretrained_model_name_or_path: str | Path) -> RLDXProcessorConfig:
        config_file, _, _ = resolve_processor_files(pretrained_model_name_or_path)
        saved = json.loads(Path(config_file).read_text())["processor_kwargs"]
        if "format_version" not in saved:
            return RLDXProcessorFormatV1.to_config(saved)
        version = saved["format_version"]
        if version != FORMAT_VERSION:
            raise ValueError(f"processor_config.json has unknown format_version {version!r}")
        expected = {"format_version", *(field.name for field in fields(RLDXProcessorConfig))}
        saved = {k: v for k, v in saved.items() if k not in REMOVED_KEYS}
        missing, unknown = sorted(expected - saved.keys()), sorted(saved.keys() - expected)
        if missing or unknown:
            raise ValueError(
                f"format_version {version} processor_config.json is missing {missing}, "
                f"has unknown {unknown}"
            )
        return RLDXProcessorConfig(
            modality_configs=parse_modality_configs(saved["modality_configs"])
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        rldx_config: RLDXConfig,
        embodiment_id_mapping: dict[str, int] | None = None,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
        **overrides,
    ):
        _, statistics_file, embodiment_id_file = resolve_processor_files(
            pretrained_model_name_or_path
        )
        config = cls.load_config(pretrained_model_name_or_path)
        if "modality_configs" in overrides:
            overrides["modality_configs"] = {
                **config.modality_configs,
                **overrides["modality_configs"],
            }
        return cls(
            replace(config, **overrides),
            rldx_config=rldx_config,
            statistics=json.loads(Path(statistics_file).read_text()),
            embodiment_id_mapping=(
                embodiment_id_mapping
                if embodiment_id_mapping is not None
                else json.loads(Path(embodiment_id_file).read_text())
            ),
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    @property
    def memory_length(self) -> int:
        memory = self.rldx_config.memory
        return 1 if memory is None else memory.length

    def get_modality_configs(self) -> dict[str, dict[str, ModalityConfig]]:
        return self.config.modality_configs
