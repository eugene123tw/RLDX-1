from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


robocasa_panda_omron = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_view", "right_view", "wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "end_effector_position_absolute",
            "end_effector_rotation_absolute",
            "gripper_qpos",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "end_effector_position",
            "end_effector_rotation",
            "gripper_close",
            "base_motion",
            "control_mode",
        ],
        action_configs=[
            # end_effector_position
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # end_effector_rotation
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper_close
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # base_motion
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # control_mode
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

register_modality_config(robocasa_panda_omron, EmbodimentTag.GENERAL_EMBODIMENT)
