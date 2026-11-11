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
ROBOCASA_REPO="$PROJECT_REPO/external_dependencies/robocasa"
PATCH_DIR="$SCRIPT_DIR/patches"
UV_ENV="$SCRIPT_DIR/robocasa_uv"

# --force gives a pristine checkout at the pinned commit, so the patches below
# can be applied unconditionally and re-running this script stays idempotent.
# Do NOT reintroduce an ``apply --check`` guard here: git apply exits 0 on a
# patch it silently skipped, which reported success without patching anything.
git submodule update --init --force "$ROBOCASA_REPO"
for patch in "$PATCH_DIR"/*.patch; do
    [ -f "$patch" ] || continue
    git -C "$ROBOCASA_REPO" apply "$patch"
    echo "Applied: $(basename "$patch")"
done

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

# Sim stack. robosuite is pinned to the 1.5.1 release, not master as robocasa's
# README asks for.
#
# 1.5.2 deleted MujocoXMLObject._get_geoms and rerouted both of its call sites
# -- _get_object_subtree() and set_scale() -- through module-level helpers that
# do not dispatch through ``self``. RoboCasa only *defines* that override;
# robosuite is what calls it. So on 1.5.2 it never runs and the physics values
# robocasa's ObjCat registry passes are silently dropped.
#
# Net effect, measured on an A100 against this venv: every shipped asset XML
# already bakes MJCFObject's constructor defaults (all 914 assets / 29414
# geoms carry solref="0.001 1" density="100" friction="0.95 0.3 0.1"), and
# ObjCat differs from those in solref alone -- it passes (0.001, 2). So objects
# do NOT fall back to mujoco defaults; they keep dampratio 1 where robocasa
# wants 2, i.e. critically damped contacts instead of overdamped. That is the
# "object bounces off the plate or wedges instead of settling" behaviour that
# makes PnP unsolvable from reset. Nothing raises.
#
# Before bumping robosuite, build an MJCFObject with solref=(0.001, 2) and
# check its geoms report "0.001 2" rather than the asset's "0.001 1". The
# version string is NOT a proxy -- master reported 1.5.1 for five months after
# the hook died (4062cb4b, 2025-07-18 .. 2025-12-22).
uv pip install robosuite==1.5.1
uv pip install -e "$ROBOCASA_REPO" --config-settings editable_mode=compat
uv pip install gymnasium==0.29.1 pydantic av==15.0.0 zmq transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8
uv pip install pandas

# Make your project importable in this venv without re-resolving deps
uv pip install --editable "$PROJECT_REPO" --no-deps

# Stable headless timestep (optional but recommended)
# python - <<'PY'
# import importlib, re
# try:
#     rs = importlib.import_module("robosuite.macros_private"); path = rs.__file__
# except Exception:
#     rs = importlib.import_module("robosuite.macros"); path = rs.__file__
# txt = open(path, "r", encoding="utf-8").read()
# new = re.sub(r"(SIMULATION_TIMESTEP\s*=\s*)([0-9.]+)", r"\g<1>0.005", txt)
# if txt != new:
#     open(path, "w", encoding="utf-8").write(new); print(f"Updated SIMULATION_TIMESTEP in {path}")
# else:
#     print("No SIMULATION_TIMESTEP change needed")
# PY

# Assets for RoboCasa (kitchen). The upstream script ignores ``-y`` and calls
# ``input()`` unconditionally, so pipe the answer in or it dies on ``EOFError``
# non-interactively.
#
# It also re-downloads all ~5 GB on every run, so skip it when the assets are
# there. A bundle is complete once its folder exists and its zip is gone (the zip
# is removed only after extraction); fixtures/ is partly tracked, so look for a mesh.
kitchen_assets_present() {
  local a="$ROBOCASA_REPO/robocasa/models/assets"
  [ -d "$a/textures" ] && [ -d "$a/objects/objaverse" ] && [ -d "$a/generative_textures" ] \
    && [ -n "$(find "$a/fixtures" -name '*.obj' -print -quit)" ] \
    && ! compgen -G "$a/*.zip" >/dev/null && ! compgen -G "$a/objects/*.zip" >/dev/null
}
if kitchen_assets_present; then
  echo "Kitchen assets present; skipping download"
else
  printf 'y\n' | python "$ROBOCASA_REPO/robocasa/scripts/download_kitchen_assets.py"
  # Upstream exits 0 even after three failed download attempts.
  kitchen_assets_present || { echo "Kitchen asset download incomplete" >&2; exit 1; }
fi

# Sanity import & env construction
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym, robocasa, robosuite
import robocasa.utils.gym_utils.gymnasium_groot
print("Imports OK:", robosuite.__version__)
env = gym.make("robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env", enable_render=True)
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