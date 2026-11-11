from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


openarm_inspire = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["camera_ego_left", "camera_ego_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "neck_joints",
            "left_arm_joints",
            "right_arm_joints",
            "left_hand_joints",
            "right_hand_joints",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "neck_joints",
            "left_arm_joints",
            "right_arm_joints",
            "left_hand_joints",
            "right_hand_joints",
        ],
        action_configs=[
            # neck_joints (2 DOF)
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_arm_joints (7 DOF)
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm_joints (7 DOF)
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand_joints (6 DOF)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand_joints (6 DOF)
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
}


register_modality_config(openarm_inspire, EmbodimentTag.GENERAL_EMBODIMENT)
