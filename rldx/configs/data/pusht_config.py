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
Push-T Modality Configuration

Push-T is a simple 2D planar object manipulation task with:
- Single top-down RGB camera (96x96x3)
- 2D end-effector state (x, y position)
- 2D actions (dx, dy delta movement)

No gripper, no tactile, no language annotations.
Features derived from lerobot/pusht dataset structure.
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


pusht = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["image"],  # Top-down RGB camera (remapped from observation.image)
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "agent_pos",  # 2D agent position: [x, y]
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # Action history context (16 steps back)
        modality_keys=[
            "agent_pos",  # 2D absolute target position: [x, y]
        ],
        action_configs=[
            # Push-T action is an abstract 2D target position, not a 6DOF
            # end-effector pose. Use NON_EEF so it is treated as a generic
            # vector (the EEF path expects xyz + rot6d = 9D).
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
}


register_modality_config(pusht, EmbodimentTag.GENERAL_EMBODIMENT)
