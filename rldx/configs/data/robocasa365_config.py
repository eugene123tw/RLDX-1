from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


robocasa365_panda_omron = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "end_effector_position_relative",
            "end_effector_rotation_relative",
            "gripper_qpos",
            "base_position",
            "base_rotation",
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
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(robocasa365_panda_omron, EmbodimentTag.GENERAL_EMBODIMENT)
