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

from dataclasses import dataclass
import os
from typing import Annotated

from rldx.configs.execution import PRESETS, RuntimeProfileName, ServingLimits
from rldx.configs.model.inference import GuidedRtcConfig, InferenceConfig, TrainedRtcConfig
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.serving.policy import load_policy
from rldx.serving.server import DEFAULT_MODEL_SERVER_PORT, serve
from torch._inductor.runtime.cache_dir_utils import default_cache_dir
import tyro


@dataclass
class ServerArgs:
    """Configuration for running the RLDX-1 inference server."""

    model_path: str
    """HF hub id or path of the checkpoint to serve."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.GENERAL_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    sample_timestep_from_beta_dist: bool = False
    """Whether to sample timestep from beta distribution. If False, sample uniformly."""

    denoising_timesteps: list[float] | None = None
    """Fixed denoising timesteps for inference (e.g., [0.0, 0.1, 0.3, 0.6]).
    If provided, overrides uniform spacing and sample_timestep_from_beta_dist."""

    mask_padded_action_dims: bool | None = None
    """Hold padded action dims at 0 during denoising. None (default) believes
    the checkpoint's config; a config.json without the key predates masking
    and loads as False, reproducing its pre-masking numerics exactly."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    deactivate_memory: bool = False
    """If True, load a memory-trained model as a non-memory model (RLDX),
    skipping memory module weights. Useful for evaluating memory-dropout models
    without memory at inference time."""

    # ── Real-Time Chunking ─────────────────────────────────────────────────
    rtc: (
        None
        | Annotated[TrainedRtcConfig, tyro.conf.subcommand("trained")]
        | Annotated[GuidedRtcConfig, tyro.conf.subcommand("guided")]
    ) = None
    """Off unless given: rtc:trained needs a checkpoint trained with
    --rtc-training-max-delay >= the delay; rtc:guided runs on any checkpoint."""

    # ── Inference runtime ──────────────────────────────────────────────────
    runtime_profile: RuntimeProfileName = "compile"
    """Inference runtime preset measured per device and recipe (rldx.configs.execution.PRESETS)."""

    limits: Annotated[ServingLimits, tyro.conf.subcommand("on")] | None = None
    """Largest request the server accepts; also bounds nvfp4 GEMM tuning: limits:on."""


def main(config: ServerArgs):
    print("Starting RLDX-1 inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # Servers on one node would otherwise race on a shared autotune cache and read half-written files.
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{default_cache_dir()}_{config.port}")
    print(f"  Inductor cache: {os.environ['TORCHINDUCTOR_CACHE_DIR']}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")
    policy = load_policy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        inference_config=InferenceConfig(
            deactivate_memory=config.deactivate_memory,
            mask_padded_action_dims=config.mask_padded_action_dims,
            rtc=config.rtc,
            sample_timestep_from_beta_dist=config.sample_timestep_from_beta_dist,
            denoising_timesteps=(
                None if config.denoising_timesteps is None else tuple(config.denoising_timesteps)
            ),
        ),
        runtime_profile=PRESETS[config.runtime_profile],
        serving_limits=config.limits,
    )
    serve(policy, config.host, config.port)


def parse_cli(args: list[str] | None = None) -> ServerArgs:
    return tyro.cli(ServerArgs, args=args, config=(tyro.conf.CascadeSubcommandArgs,))


if __name__ == "__main__":
    main(parse_cli())
