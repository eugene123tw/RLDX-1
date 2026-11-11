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

import copy
import importlib.util
import json
from pathlib import Path

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


MODALITY_CONFIGS = {
    ##### Pre-registered posttrain configurations #####
    "unitree_g1": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(30)),
            modality_keys=[
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
                "base_height_command",
                "navigate_command",
            ],
            action_configs=[
                # left_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # left_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # waist
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # base_height_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # navigate_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "libero_panda": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "wrist_image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "oxe_widowx": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image_0"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "pad",
                "gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 8)),
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
            mean_std_embedding_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "oxe_google": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "x",
                "y",
                "z",
                "rx",
                "ry",
                "rz",
                "rw",
                "gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 8)),
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
            mean_std_embedding_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
}


def register_modality_config(
    config: dict, embodiment_tag: EmbodimentTag = EmbodimentTag.GENERAL_EMBODIMENT
):
    assert embodiment_tag.value not in MODALITY_CONFIGS, (
        f"Embodiment tag {embodiment_tag} already registered"
    )
    MODALITY_CONFIGS[embodiment_tag.value] = config


_LOADED_FILES: dict[Path, dict[str, dict[str, ModalityConfig]]] = {}


def load_modality_config_file(path: str | Path) -> dict[str, dict[str, ModalityConfig]]:
    """Run a ``.py`` modality config and return the tags it registered; a file runs once."""
    path = Path(path).resolve()
    if path in _LOADED_FILES:
        return _LOADED_FILES[path]
    if not path.is_file():
        raise FileNotFoundError(f"Modality config path does not exist: {path}")
    if path.suffix != ".py":
        raise ValueError(f"Unsupported modality config format: {path.suffix}. Use .py")
    before = copy.deepcopy(MODALITY_CONFIGS)
    spec = importlib.util.spec_from_file_location(f"_modality_config_{path.stem}", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    _LOADED_FILES[path] = {
        tag: configs for tag, configs in MODALITY_CONFIGS.items() if before.get(tag) != configs
    }
    return _LOADED_FILES[path]


def modality_configs_for(
    embodiment_tag: EmbodimentTag, path: str | Path | None = None
) -> dict[str, ModalityConfig]:
    """One embodiment's configs: from a ``.json`` of ModalityConfig fields, a ``.py``, or the registry."""
    if path is not None and Path(path).suffix == ".json":
        fields = json.loads(Path(path).read_text())
        return {modality: ModalityConfig(**value) for modality, value in fields.items()}
    if path is not None:
        load_modality_config_file(path)
    if embodiment_tag.value not in MODALITY_CONFIGS:
        raise ValueError(
            f"No modality config for embodiment tag '{embodiment_tag.name}' "
            f"(value='{embodiment_tag.value}'). Available tags: {sorted(MODALITY_CONFIGS)}. "
            "Pass --modality-config-path (.py or .json)."
        )
    return MODALITY_CONFIGS[embodiment_tag.value]
