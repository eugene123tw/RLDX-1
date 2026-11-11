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

"""Image augmentation pipeline for RLDX.

Pipeline (applied to every camera view):

  1. AspectAreaResizeAndCrop
       Aspect-ratio-preserving resize so the total pixel area is at most
       ``image.max_area``, then center-crop both spatial dims down to
       multiples of ``image.resize_m``.
       Deterministic — used by both train and eval.

  2. (optional) Fractional crop + resize-back
       Crop an ``image.crop_fraction`` sub-region and resize it back to the
       post-step-1 shape, so downstream stages always see a fixed output size.
       - Training: random position (FractionalRandomCropAndResize).
       - Eval: center position, on the GPU with the patchify
         (``rldx.planner.input.tensor_image.center_crop_and_resize``).
       Skipped entirely when ``image.crop_fraction`` is None (no-op).

  3. (train only, optional) Rotate / ColorJitter
       Standard photometric & mild geometric augmentation.

The builder :func:`build_image_transformations_albumentations` returns the
train transform. Eval runs step 1 alone through the processor's cv2 resize.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import albumentations as A
import cv2
import numpy as np
import torch


if TYPE_CHECKING:
    from rldx.configs.model.rldx import ImageAugmentationConfig, ImageGeometryConfig


# ---------------------------------------------------------------------------
# Geometry helper (pure function)
# ---------------------------------------------------------------------------


def resize_preserve_aspect_area_then_crop(
    h: int, w: int, max_area: int = 256**2, m: int = 32
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute aspect-ratio-preserving resize + m-aligned crop sizes.

    The resize never upscales. It chooses the largest scale such that the
    resulting area is at most ``max_area`` **and** the shorter side is a
    multiple of ``m``. The longer side then follows from the aspect ratio,
    and is finally cropped (not resized) to a multiple of ``m``.

    Args:
        h: Input height.
        w: Input width.
        max_area: Maximum pixel area budget for the resized image.
        m: Alignment multiple for the output dimensions.

    Returns:
        ``((h_r, w_r), (h_c, w_c))`` where the first pair is the resize
        target (aspect ratio preserved) and the second pair is the final
        crop (both dims multiples of ``m``).
    """
    smax = min(1.0, math.sqrt(max_area / (h * w)))
    short, long_ = (h, w) if h <= w else (w, h)

    # largest ``m``-multiple short side that respects the area budget
    short_r = max(m, int((short * smax) // m) * m)
    s = short_r / short

    # aspect-preserving long side (floor so resized area <= max_area)
    long_r = int(long_ * s)

    h_r, w_r = (short_r, long_r) if h <= w else (long_r, short_r)
    h_c = h_r - (h_r % m)
    w_c = w_r - (w_r % m)
    return (h_r, w_r), (h_c, w_c)


# ---------------------------------------------------------------------------
# Replay helper (consistent stochastic transforms across multiple views)
# ---------------------------------------------------------------------------


def apply_with_replay(transform, images, replay=None):
    """Apply an albumentations transform to multiple images with replay.

    When ``transform`` is a :class:`A.ReplayCompose`, the first image produces
    replay data that is reused for subsequent images so that random params
    (rotation, jitter, crop origin, ...) are consistent across all views.

    Args:
        transform: ``A.Compose`` or ``A.ReplayCompose``.
        images: Iterable of PIL images (or numpy arrays).
        replay: Optional replay blob from an earlier call. When None, the
            first image creates fresh replay data.

    Returns:
        ``(tensors, replay)`` — a list of ``uint8`` tensors ``(H, W, C)`` and
        the replay blob (or None for plain ``A.Compose``).
    """
    transformed_tensors = []
    current_replay = replay
    has_replay = hasattr(transform, "replay")

    for img in images:
        img_arr = np.array(img)

        if has_replay:
            if current_replay is None:
                augmented = transform(image=img_arr)
                current_replay = augmented["replay"]
            else:
                augmented = transform.replay(
                    image=img_arr,
                    saved_augmentations=current_replay,
                )
            img_array = augmented["image"]
        else:
            augmented = transform(image=img_arr)
            img_array = augmented["image"]

        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        elif img_array.dtype != np.uint8:
            raise ValueError(f"Unexpected data type: {img_array.dtype}")

        transformed_tensors.append(torch.from_numpy(img_array))

    return transformed_tensors, current_replay


# ---------------------------------------------------------------------------
# Step 1 — aspect-area resize + m-aligned crop
# ---------------------------------------------------------------------------


class AspectAreaResizeAndCrop(A.DualTransform):
    """Resize preserving aspect ratio (area-constrained) then center-crop.

    Step 1.1: Resize so total area <= ``max_area``, short side aligned to
              ``m``, aspect ratio preserved.
    Step 1.2: Center-crop both dims to multiples of ``m`` (removes at most
              ``m-1`` pixels per dim).

    This is the first (and only required) stage of the image pipeline.
    """

    _logged_shapes: set = set()

    def __init__(
        self,
        max_area: int = 65536,
        m: int = 32,
        interpolation: int = cv2.INTER_AREA,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.max_area = max_area
        self.m = m
        self.interpolation = interpolation

    def apply(self, img, resize_hw=(0, 0), crop_coords=(0, 0, 0, 0), **params):
        h_r, w_r = resize_hw
        resized = cv2.resize(img, (w_r, h_r), interpolation=self.interpolation)
        x_min, y_min, x_max, y_max = crop_coords
        return resized[y_min:y_max, x_min:x_max]

    def apply_with_params(self, params, *, image=None, **kwargs):
        if image is not None:
            params = {**params, **self._geometry(*image.shape[:2])}
        return super().apply_with_params(params, image=image, **kwargs)

    def _geometry(self, h, w):
        (h_r, w_r), (h_c, w_c) = resize_preserve_aspect_area_then_crop(
            h, w, max_area=self.max_area, m=self.m
        )
        y_min = (h_r - h_c) // 2
        x_min = (w_r - w_c) // 2
        return {
            "resize_hw": (h_r, w_r),
            "crop_coords": (x_min, y_min, x_min + w_c, y_min + h_c),
        }

    def get_params_dependent_on_data(self, params, data):
        h, w = params["shape"][:2]
        geometry = self._geometry(h, w)
        h_r, w_r = geometry["resize_hw"]
        x_min, y_min, x_max, y_max = geometry["crop_coords"]
        h_c, w_c = y_max - y_min, x_max - x_min
        shape_key = (h, w)
        if shape_key not in AspectAreaResizeAndCrop._logged_shapes:
            AspectAreaResizeAndCrop._logged_shapes.add(shape_key)
            print(
                f"[AspectAreaResizeAndCrop] ({h}, {w}) → resize ({h_r}, {w_r}) → crop ({h_c}, {w_c})"
            )
        return geometry

    def get_transform_init_args_names(self):
        return ("max_area", "m", "interpolation")


# ---------------------------------------------------------------------------
# Step 2 — fractional crop + resize back to pre-crop shape
# ---------------------------------------------------------------------------


class _FractionalCropAndResizeBase(A.DualTransform):
    """Crop a fraction of the input then resize back to the pre-crop shape.

    Subclasses choose the crop origin.
    Both the crop fraction and the output shape are parameterised so the
    downstream stages always see a consistent spatial size.
    """

    def __init__(
        self,
        crop_fraction: float = 0.95,
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be in (0.0, 1.0]")
        self.crop_fraction = crop_fraction
        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int] = (0, 0, 0, 0),
        out_hw: tuple[int, int] = (0, 0),
        **params,
    ) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        cropped = img[y_min:y_max, x_min:x_max]
        h_out, w_out = out_hw
        return cv2.resize(cropped, (w_out, h_out), interpolation=self.interpolation)

    def _origin(self, max_y: int, max_x: int) -> tuple[int, int]:
        raise NotImplementedError

    def apply_with_params(self, params, *, image=None, **kwargs):
        if image is not None and params.get("out_hw") != image.shape[:2]:
            h, w = image.shape[:2]
            ch = max(1, int(h * self.crop_fraction))
            cw = max(1, int(w * self.crop_fraction))
            max_y, max_x = h - ch, w - cw
            y_ratio, x_ratio = params["origin_ratio"]
            y_min, x_min = self._replay_origin(y_ratio, x_ratio, max_y, max_x)
            params = {
                **params,
                "crop_coords": (x_min, y_min, x_min + cw, y_min + ch),
                "out_hw": (h, w),
            }
        return super().apply_with_params(params, image=image, **kwargs)

    def get_params_dependent_on_data(self, params, data):
        h, w = params["shape"][:2]
        ch = max(1, int(h * self.crop_fraction))
        cw = max(1, int(w * self.crop_fraction))
        y_min, x_min = self._origin(h - ch, w - cw)
        return {
            "crop_coords": (x_min, y_min, x_min + cw, y_min + ch),
            "out_hw": (h, w),
            "origin_ratio": (
                y_min / (h - ch) if h > ch else 0.0,
                x_min / (w - cw) if w > cw else 0.0,
            ),
        }

    def _replay_origin(self, y_ratio, x_ratio, max_y, max_x):
        return round(y_ratio * max_y), round(x_ratio * max_x)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction", "interpolation")


class FractionalRandomCropAndResize(_FractionalCropAndResizeBase):
    """Random-position fractional crop, then resize back to the pre-crop (H, W)."""

    def _origin(self, max_y: int, max_x: int) -> tuple[int, int]:
        y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        return y, x


# ---------------------------------------------------------------------------
# Builder — constructs the train transform for the processor
# ---------------------------------------------------------------------------


def build_image_transformations_albumentations(
    image: ImageGeometryConfig, augmentation: ImageAugmentationConfig
) -> A.ReplayCompose:
    """Build the train transform: the aspect-area resize and crop of ``image``, then the
    stochastic crop, rotation and color jitter.

    Returns an albumentations ``ReplayCompose`` that takes ``image=np.ndarray``.
    """
    train_list: list = []
    train_list.append(
        AspectAreaResizeAndCrop(
            max_area=image.max_area, m=image.resize_m, interpolation=cv2.INTER_AREA
        )
    )

    if image.crop_fraction is not None:
        if not 0.0 < image.crop_fraction <= 1.0:
            raise ValueError(
                f"random_crop_fraction must be in (0.0, 1.0], got {image.crop_fraction!r}"
            )
        train_list.append(
            FractionalRandomCropAndResize(
                crop_fraction=image.crop_fraction, interpolation=cv2.INTER_LINEAR
            )
        )

    if augmentation.rotation_angle is not None and augmentation.rotation_angle != 0:
        train_list.append(A.Rotate(limit=augmentation.rotation_angle, p=1.0))

    if augmentation.color_jitter is not None:
        train_list.append(
            A.ColorJitter(
                brightness=augmentation.color_jitter.get("brightness", 0.0),
                contrast=augmentation.color_jitter.get("contrast", 0.0),
                saturation=augmentation.color_jitter.get("saturation", 0.0),
                hue=augmentation.color_jitter.get("hue", 0.0),
                p=1.0,
            )
        )

    return A.ReplayCompose(train_list, p=1.0)
