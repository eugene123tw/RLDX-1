"""ALLEX mid-training modality config — stereo ego video, chunk=40, hist=1.

Same action / state / torque / language layout as
`midtrain_allex_data_config.py`, but video consumes both ego cameras
(left + right) instead of mono left. Use with a stereo-rig dataset and
`--action-horizon 40`.
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


MIDTRAIN_ALLEX_STEREO_MODALITY_CONFIGS = {
    "allex": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["camera_ego_left", "camera_ego_right"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm_joints",
                "left_hand_joints",
                "neck_joints",
                "right_arm_joints",
                "right_hand_joints",
                "waist_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "left_arm_joints",
                "left_hand_joints",
                "neck_joints",
                "right_arm_joints",
                "right_hand_joints",
                "waist_joints",
            ],
            action_configs=[
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
            modality_keys=["annotation.human.task_description"],
        ),
        "torque": ModalityConfig(  # 48 dim
            # hist=1 (current timestep only), fut=40 (== action_horizon)
            delta_indices=list(range(0, 41)),
            modality_keys=[
                "left_arm_effort",
                "left_hand_effort",
                "neck_effort",
                "right_arm_effort",
                "right_hand_effort",
                "waist_effort",
            ],
        ),
    },
}


for name, modality_config in MIDTRAIN_ALLEX_STEREO_MODALITY_CONFIGS.items():
    register_modality_config(modality_config, EmbodimentTag.GENERAL_EMBODIMENT)
