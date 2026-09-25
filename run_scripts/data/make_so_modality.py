#!/usr/bin/env python
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
Auto-generate ``meta/modality.json`` for an SO-100 / SO-101 LeRobot dataset.

RLDX-1's loader needs ``meta/modality.json`` to know how to slice the flat 6-DOF
state/action vectors into joint groups and which video features to expose. That
file is not part of any raw Hub dataset, so this script writes it from the
dataset's own ``meta/info.json`` -- making setup a single command:

    python run_scripts/data/make_so_modality.py --dataset-path /path/to/dataset

Works on both v2.1 and v3.0 datasets (it only reads ``info.json``). After this,
``meta/stats.json`` is generated automatically at train time (or run
``rldx/data/stats.py``), and no v3->v2 conversion is required -- the loader reads
v3.0 natively.

SO-100 / SO-101 convention (6-DOF arm + gripper):
    state/action = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
                    wrist_roll, gripper] -> single_arm[0:5] + gripper[5:6]
"""

import argparse
import json
from pathlib import Path

VIDEO_DTYPES = {"video", "image"}
IMAGE_PREFIX = "observation.images."


def build_modality(info: dict) -> dict:
    features = info.get("features", {})

    def _dim(key: str) -> int:
        assert key in features, f"'{key}' missing from meta/info.json features"
        shape = features[key]["shape"]
        return int(shape[0])

    state_dim, action_dim = _dim("observation.state"), _dim("action")
    if state_dim != 6 or action_dim != 6:
        raise ValueError(
            f"Expected 6-DOF SO-100/SO-101 state and action, got state={state_dim}, "
            f"action={action_dim}. This helper only covers the SO-100/SO-101 "
            "single_arm[0:5]+gripper[5:6] layout; write modality.json manually for others."
        )

    joint_groups = {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}}

    video = {}
    for key, feat in features.items():
        if feat.get("dtype") in VIDEO_DTYPES and key.startswith(IMAGE_PREFIX):
            short_name = key[len(IMAGE_PREFIX) :]
            video[short_name] = {"original_key": key}
    if not video:
        raise ValueError(
            "No video features (observation.images.*) found in meta/info.json; "
            f"available features: {list(features.keys())}"
        )

    return {"state": joint_groups, "action": dict(joint_groups), "video": video}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", required=True, help="Dataset root containing meta/info.json")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing modality.json")
    args = parser.parse_args()

    meta_dir = Path(args.dataset_path) / "meta"
    info_path = meta_dir / "info.json"
    assert info_path.exists(), f"{info_path} does not exist"
    with open(info_path) as f:
        info = json.load(f)

    modality = build_modality(info)

    out_path = meta_dir / "modality.json"
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} already exists; pass --force to overwrite")
    with open(out_path, "w") as f:
        json.dump(modality, f, indent=4)

    cams = ", ".join(modality["video"].keys())
    print(f"Wrote {out_path}")
    print(f"  robot_type: {info.get('robot_type', 'unknown')} | cameras: {cams}")
    print("  state/action: single_arm[0:5] + gripper[5:6]")


if __name__ == "__main__":
    main()
