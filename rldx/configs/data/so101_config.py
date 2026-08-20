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

"""
SO-101 (LeRobot ``so_follower``) Modality Configuration

Matches the joint convention used by ``rldx/eval/real_robot/SO100/eval_so100.py``:
a single 6-DOF arm+gripper, state/action vectors ordered as
``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]``,
sliced into ``single_arm`` (5) + ``gripper`` (1) joint groups.

Cameras and modality key names here must match ``meta/modality.json`` written
by ``run_scripts/data/convert_lerobot_v3_to_v2.py`` for the target dataset.
"""

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


so101 = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "overhead"],  # remapped from observation.images.{arm,overhead}
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
            "gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            # SO-101 joint targets are raw absolute joint/servo positions, not
            # a 6DOF end-effector pose, so both groups use NON_EEF.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],  # Synthesized from meta/tasks.jsonl by LeRobotEpisodeLoader
    ),
}


register_modality_config(so101, EmbodimentTag.GENERAL_EMBODIMENT)
