import copy

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


tactile_torque_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "primary",
            "wrist",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["end_effector_position", "end_effector_rotation", "gripper_position"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["end_effector_position", "end_effector_rotation", "gripper_close"],
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
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
    "tactile": ModalityConfig(
        delta_indices=list(range(0, 17)),
        modality_keys=[
            "left",
        ],
    ),
    "torque": ModalityConfig(
        delta_indices=list(range(0, 17)),
        modality_keys=[
            "torque",
        ],
    ),
}


register_modality_config(tactile_torque_config, EmbodimentTag.GENERAL_EMBODIMENT)
register_modality_config(copy.deepcopy(tactile_torque_config), EmbodimentTag.OXE_DROID)
