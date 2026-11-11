# SPDX-License-Identifier: Apache-2.0
"""Which projector column each normalisation tag trains.

Nothing at inference reads this: a checkpoint ships its own
``embodiment_id.json`` and that wins.
"""

from rldx import plugins


EMBODIMENT_TAG_TO_PROJECTOR_INDEX: dict[str, int] = {
    "general_embodiment": 0,
    "fractal20220817_data": 1,
    "bridge_orig": 2,
    "droid": 3,
    "new_embodiment": 7,
}


PROJECTOR_GROUPINGS: dict[str, dict[str, int]] = {}
for _plugin in plugins.modules("slots"):
    _plugin.register(EMBODIMENT_TAG_TO_PROJECTOR_INDEX, PROJECTOR_GROUPINGS)
