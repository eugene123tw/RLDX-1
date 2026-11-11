#!/usr/bin/env bash
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
set -euxo pipefail
export NO_ALBUMENTATIONS_UPDATE=1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set paths relative to script location
PROJECT_REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LIBERO_REPO="$PROJECT_REPO/external_dependencies/LIBERO"
LIBERO_UV_ENV="$SCRIPT_DIR/libero_uv"

# --force so a re-run starts from a pristine checkout at the pinned commit.
# We carry no patches against LIBERO -- our copy is byte-identical to upstream.
git submodule update --init --force "$LIBERO_REPO"

# python -m pip install cmake==3.18.4
rm -rf $LIBERO_UV_ENV
mkdir -p $LIBERO_UV_ENV
uv venv $LIBERO_UV_ENV/.venv --python 3.10
source $LIBERO_UV_ENV/.venv/bin/activate
# uv pip install gymnasium==1.2.0 # -> 2.9.1 -> 
uv pip install --requirements $LIBERO_REPO/requirements.txt
uv pip install -e $LIBERO_REPO --config-settings editable_mode=compat
uv pip install --editable $PROJECT_REPO --no-deps
uv pip install torch==2.5.1 torchvision==0.20.1 pydantic av tianshou==0.5.1 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
uv pip install transformers==4.57.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
# Pin mujoco: LIBERO's requirements leave it unpinned, so robosuite 1.4.0 gets
# the newest release and env creation dies -- mj_fullM's signature changed in
# 3.10.0 (2026-06-22). Measured on this island: 3.9.0 passes, 3.10.0 is the
# first that does not; 3.3.1 matches the pin upstream added on main.
uv pip install numpy==1.26.4 mujoco==3.3.1

uv pip install --editable "$PROJECT_REPO" --no-deps

# A config of our own, so the asset paths are this checkout's rather than
# whatever the global ~/.libero/config.yaml points at. eval_libero.sh reads it.
export LIBERO_CONFIG_PATH="$LIBERO_UV_ENV/.venv/libero_config"
echo "y\n" | python -c "from rldx.eval.sim.LIBERO.libero_env import register_libero_envs"
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from rldx.eval.sim.LIBERO.libero_env import register_libero_envs
register_libero_envs()
import gymnasium as gym
env = gym.make("libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate")
env.reset()
env.close()
print("Env OK:", type(env))
PY