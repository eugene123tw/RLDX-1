# SPDX-License-Identifier: Apache-2.0
"""The steps RLDXPolicy.from_checkpoint runs to load a checkpoint for serving."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import torch

from rldx.configs.execution import InferenceRuntimeProfile, ServingLimits
from rldx.configs.model.inference import InferenceConfig
from rldx.configs.model.rldx import RLDXConfig
from rldx.model.core.collator import PromptEncoder
from rldx.model.core.rldx import RLDX
from rldx.model.lora import fold_lora
from rldx.planner.input.tensor_image import TensorImageSpec
from rldx.runtime.inference import InferenceRuntime


INFERENCE_PLANNER_MEMO_SIZE = 128


def require_video_keys(modality_configs: dict) -> None:
    if "video" not in modality_configs or not modality_configs["video"].modality_keys:
        raise ValueError("tensor image path requires at least one video modality key")


def resolve_model_dir(model_path: str) -> Path:
    model_dir = Path(model_path)
    if not model_dir.exists():
        model_dir = Path(snapshot_download(model_path))
    return model_dir


def gemm_rows(
    limits: ServingLimits, encoder: PromptEncoder, modality_configs: dict, model: Any
) -> int:
    vision, text = encoder.max_sample_tokens(modality_configs["video"], limits.instruction_tokens)
    msat = model.action_model.max_sample_tokens(len(modality_configs["state"].delta_indices))
    return limits.batch * max(vision, text + model.backbone.input_spec.n_cog_tokens, msat)


def build_inference_runtime(
    model: Any,
    guided: bool,
    *,
    dtype: torch.dtype,
    profile: InferenceRuntimeProfile | None = None,
    gemm_rows: int | None = None,
    tensor_image_path: TensorImageSpec,
) -> InferenceRuntime:
    # Guided RTC differentiates inside get_action, which inference tensors forbid.
    return InferenceRuntime(
        model,
        dtype=dtype,
        profile=profile,
        gemm_rows=gemm_rows,
        inference_mode=not guided,
        tensor_image_path=tensor_image_path,
    )


def load_model(
    model_dir: Path,
    config: RLDXConfig,
    device: int | str,
    inference_config: InferenceConfig,
    *,
    dtype: torch.dtype,
) -> Any:
    if torch.device(device).type == "xpu":
        # Triton refuses to pick a driver when CUDA is visible too; compile regions target XPU.
        os.environ.setdefault("TRITON_DEFAULT_BACKEND", "intel")
    model = RLDX.from_pretrained(
        model_dir,
        config=config,
        inference=inference_config,
        device_map=device,
        torch_dtype=dtype,
    )
    fold_lora(model)
    model.eval()
    model.requires_grad_(False)
    return model
