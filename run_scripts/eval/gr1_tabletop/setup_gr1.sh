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
# Original source: https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh
set -euxo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ROBOCASA_GR1_TABLETOP_TASKS_REPO="$PROJECT_REPO/external_dependencies/robocasa-gr1-tabletop-tasks"
PATCH_DIR="$SCRIPT_DIR/patches"
UV_ENV="$SCRIPT_DIR/robocasa_uv"

# Optional: if you want to avoid hardlink warnings with uv cache
# export UV_LINK_MODE=copy

# --force gives a pristine checkout at the pinned commit, so the patches below
# can be applied unconditionally and re-running this script stays idempotent.
# Do NOT reintroduce an ``apply --check`` guard here: git apply exits 0 on a
# patch it silently skipped, which reported success without patching anything.
git submodule update --init --force "$ROBOCASA_GR1_TABLETOP_TASKS_REPO"
for patch in "$PATCH_DIR"/*.patch; do
    [ -f "$patch" ] || continue
    git -C "$ROBOCASA_GR1_TABLETOP_TASKS_REPO" apply "$patch"
    echo "Applied: $(basename "$patch")"
done

# Ensure build tools (mirrors your LIBERO flow)
# python -m pip install cmake==3.18.4

# Fresh venv
rm -rf "$UV_ENV"
mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
source "$UV_ENV/.venv/bin/activate"

# Make sure the venv has a build backend
uv pip install setuptools wheel

# Heavy deps first
uv pip install torch==2.5.1 torchvision==0.20.1

# Preinstall flash-attn to avoid builds inside other installs.
# Guard it to Linux only (flash-attn not supported on macOS).
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-1}
if [[ "$(uname -s)" == "Linux" && "$INSTALL_FLASH_ATTN" == "1" ]]; then
  uv pip install --no-build-isolation flash-attn==2.7.4.post1 || echo "flash-attn install skipped/failed; continuing"
else
  echo "Skipping flash-attn (non-Linux or INSTALL_FLASH_ATTN=0)"
fi

# Core sim deps: robosuite first, then this repo editable.
# README: https://github.com/robocasa/robocasa-gr1-tabletop-tasks
# Deliberate deviation from that README, which says to clone robosuite master.
# This stack pins the 1.5.1 release instead. Kept as-is: it is already a pin
# (so it cannot drift the way master did) and it is what this venv has been
# built against, so changing it would be an unvalidated behaviour change.
uv pip install robosuite==1.5.1

# The repo’s requirements.txt only contains "-e .", so just install editable.
uv pip install -e "$ROBOCASA_GR1_TABLETOP_TASKS_REPO" --config-settings editable_mode=compat

# Optional: your eval stack uses gymnasium
uv pip install gymnasium==0.29.1 pydantic av==15.0.0 zmq transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8

# Make your project importable without re-resolving deps
uv pip install --editable "$PROJECT_REPO" --no-deps

uv pip install pandas idna

# Optional: lower robosuite simulation timestep to 0.005 for headless stability
# python - <<'PY'
# import importlib, re
# try:
#     rs = importlib.import_module("robosuite.macros_private")
#     path = rs.__file__
# except Exception:
#     rs = importlib.import_module("robosuite.macros")
#     path = rs.__file__
# txt = open(path, "r", encoding="utf-8").read()
# new = re.sub(r"(SIMULATION_TIMESTEP\s*=\s*)([0-9.]+)", r"\g<1>0.005", txt)
# if txt != new:
#     open(path, "w", encoding="utf-8").write(new)
#     print(f"Updated SIMULATION_TIMESTEP in {path}")
# else:
#     print("No SIMULATION_TIMESTEP change needed")
# PY

# Assets (per README)
python "$ROBOCASA_GR1_TABLETOP_TASKS_REPO/robocasa/scripts/download_tabletop_assets.py" -y

# Sanity import
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym, robocasa, robosuite
import robocasa.utils.gym_utils.gymnasium_groot
print("Imports OK:", robosuite.__version__)
env = gym.make("gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env", enable_render=True)
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