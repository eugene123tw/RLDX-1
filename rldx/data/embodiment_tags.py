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
#
# This file has been modified from the original NVIDIA Isaac GR00T N1.7.
# Original source: https://github.com/NVIDIA/Isaac-GR00T

"""
Embodiment tags are used to identify the robot embodiment in the data.

Naming convention:
<dataset>_<robot_name>

If using multiple datasets, e.g. sim GR1 and real GR1, we can drop the dataset name and use only the robot name.
"""

from enum import Enum

from rldx import plugins


_tags: dict[str, str] = {
    "GENERAL_EMBODIMENT": "general_embodiment",
    "NEW_EMBODIMENT": "new_embodiment",
    "OXE_FRACTAL": "fractal20220817_data",
    "OXE_BRIDGE_ORIG": "bridge_orig",
    "OXE_DROID": "droid",
}

for _plugin in plugins.modules("tags"):
    _plugin.register(_tags)

EmbodimentTag = Enum("EmbodimentTag", _tags)
EmbodimentTag.__doc__ = __doc__


def get_embodimenttag_by_name(name: str):
    """
    Returns the EmbodimentTag enum member that matches the given value string.
    Returns None if no match is found.
    """
    for tag in EmbodimentTag:
        if tag.value == name:
            return tag
    return None
