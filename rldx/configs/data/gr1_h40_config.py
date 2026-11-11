"""GR-1 Tabletop modality config for --action-horizon 40.

Matches gr1_config.py except for the action chunk length. Loaded independently
to avoid registering the same embodiment tags twice.
"""

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


fourier_gr1_arms_waist = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "left_hand",
            "right_arm",
            "right_hand",
            "waist",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=[
            "left_arm",
            "left_hand",
            "right_arm",
            "right_hand",
            "waist",
        ],
        action_configs=[
            # left_arm
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand
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
            # right_hand
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # waist
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.coarse_action"],
    ),
}

register_modality_config(fourier_gr1_arms_waist, EmbodimentTag.GENERAL_EMBODIMENT)
register_modality_config(copy.deepcopy(fourier_gr1_arms_waist), EmbodimentTag.NEW_EMBODIMENT)
