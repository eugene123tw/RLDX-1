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
# This file has been modified from the original NVIDIA Isaac GR00T.
# Original source: https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/eval/sim/robocasa/setup_RoboCasa.sh
set -euxo pipefail

# Where this script lives (put it inside your repo)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ROBOCASA365_REPO="$PROJECT_REPO/external_dependencies/robocasa365"
UV_ENV="$SCRIPT_DIR/robocasa365_uv"

# --force so a re-run starts from a pristine checkout at the pinned commit.
# We carry no patches against robocasa365.
git submodule update --init --force "$ROBOCASA365_REPO"

# Build helpers
# python -m pip install cmake==3.18.4
rm -rf "$UV_ENV"
mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
source "$UV_ENV/.venv/bin/activate"
uv pip install setuptools wheel

# Core deps
uv pip install torch==2.5.1 torchvision==0.20.1
# Linux-only: preinstall flash-attn to avoid compiling inside other wheels
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-1}
if [[ "$(uname -s)" == "Linux" && "$INSTALL_FLASH_ATTN" == "1" ]]; then
  uv pip install --no-build-isolation flash-attn==2.7.4.post1 || echo "flash-attn install skipped/failed; continuing"
fi

# Sim stack. robocasa's README insists on robosuite master rather than a
# release, so we pin the commit master resolved to on 2026-08-26 instead of
# tracking the branch. This freezes current behaviour -- it is NOT a claim that
# this is the commit any published number was produced against; the branch
# floated, so that is unrecoverable.
#
# This stack stays on that commit (__version__ 1.5.2) while its sibling
# run_scripts/eval/robocasa_kitchen/ moved to robosuite==1.5.1. That asymmetry is
# deliberate, and reversing it breaks this venv two ways:
#
#   1. robocasa365's robocasa/__init__.py asserts robosuite >= 1.5.2 (and
#      mujoco == 3.3.1, numpy == 2.2.5). 1.5.1 dies at import.
#   2. It does not need the downgrade. 1.5.2 deleted
#      MujocoXMLObject._get_geoms and stopped dispatching through ``self``,
#      which is what killed the Kitchen fork's physics-property override. This
#      fork interposes MujocoXMLObjectRobocasa, which re-adds _get_geoms and
#      keeps the 1.5.1-era set_scale() that still calls self._get_geoms(obj).
#      MJCFObject always passes a non-None scale, so set_scale always runs and
#      MJCFObject._get_geoms is still reached. Upstream added that class for
#      exactly this reason.
#
# Point 2 is load-bearing, undocumented upstream, and invisible to any version
# check -- this stack stays ">= 1.5.2" either way. So do NOT "refresh" this SHA
# to a newer master without first confirming MJCFObject._get_geoms still runs
# (build one and check its geoms carry robocasa's density, not robosuite's
# default). If upstream consolidates set_scale the way it already consolidated
# _get_geoms, this stack breaks silently while still reporting a fine version.
uv pip install "git+https://github.com/ARISE-Initiative/robosuite.git@5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
uv pip install -e "$ROBOCASA365_REPO" --config-settings editable_mode=compat
# Policy client transport (rldx/serving/server.py); versions match uv.lock.
uv pip install msgpack==1.1.0 pyzmq==27.0.1

# Make your project importable in this venv without re-resolving deps
uv pip install --editable "$PROJECT_REPO" --no-deps

# Assets and env/macros setup for RoboCasa365. Both upstream scripts call
# ``input()`` unconditionally, so both need the answer piped in or they die on
# ``EOFError`` non-interactively. setup_macros.py only prompts once
# ``macros_private.py`` exists, i.e. from the second run onwards -- and
# ``submodule update --force`` does not remove it, because it is untracked.
printf 'y\n' | python "$ROBOCASA365_REPO/robocasa/scripts/download_kitchen_assets.py"
printf 'y\n' | python "$ROBOCASA365_REPO/robocasa/scripts/setup_macros.py"

# Sanity import & env construction
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym, robocasa, robosuite
print("Imports OK:", robosuite.__version__)
env = gym.make("robocasa/PickPlaceCounterToCabinet", split="pretrain", seed=0)
_ = env.reset()
env.close()
print("Env OK:", type(env))
PY

# This sim venv runs only the gym/mujoco client (the model is served by
# the main RLDX venv over zmq), so flash-attn is never loaded. Disable
# it so transformers falls back to SDPA and avoid an ABI mismatch
# between the sim venv's torch and the prebuilt flash-attn wheel.
SITE="$UV_ENV/.venv/lib/python3.10/site-packages"
mv "$SITE/flash_attn" "$SITE/flash_attn__disabled" 2>/dev/null || true
mv "$SITE"/flash_attn_2_cuda*.so "$SITE/flash_attn_2_cuda__disabled.so" 2>/dev/null || true
mv "$SITE"/flash_attn_cuda*.so "$SITE/flash_attn_cuda__disabled.so" 2>/dev/null || true
