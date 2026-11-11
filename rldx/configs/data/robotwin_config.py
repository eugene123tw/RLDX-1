from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


robotwin_franka = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head_camera", "left_camera", "right_camera", "third_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_endpose",
            "left_gripper",
            "right_endpose",
            "right_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
        action_configs=[
            # left_arm
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_gripper
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_gripper
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


register_modality_config(robotwin_franka, EmbodimentTag.GENERAL_EMBODIMENT)
