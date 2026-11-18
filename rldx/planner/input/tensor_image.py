from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TensorImageSpec:
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    rescale_factor: float
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    crop_fraction: float | None = None

    @classmethod
    def from_image_processor(cls, processor, crop_fraction: float | None = None):
        disabled = [
            name
            for name in ("do_rescale", "do_normalize", "do_convert_rgb")
            if not getattr(processor, name, False)
        ]
        if disabled:
            raise ValueError(f"tensor image path requires enabled {', '.join(disabled)}")
        return cls(
            patch_size=int(processor.patch_size),
            temporal_patch_size=int(processor.temporal_patch_size),
            merge_size=int(processor.merge_size),
            rescale_factor=float(processor.rescale_factor),
            image_mean=tuple(processor.image_mean),
            image_std=tuple(processor.image_std),
            crop_fraction=crop_fraction,
        )

    def grid(self, count: int, height: int, width: int) -> torch.Tensor:
        row = [1, height // self.patch_size, width // self.patch_size]
        return torch.tensor([row] * count, dtype=torch.long)

    def lookup(self, device: torch.device | str) -> torch.Tensor:
        values = np.arange(256, dtype=np.float64) * self.rescale_factor
        scaled = values.astype(np.float32)
        channels = [
            (scaled - np.float32(mean)) / np.float32(std)
            for mean, std in zip(self.image_mean, self.image_std)
        ]
        return torch.from_numpy(np.stack(channels)).to(device)

    def patchify(self, images: torch.Tensor, lookup: torch.Tensor | None = None) -> torch.Tensor:
        if lookup is None:
            lookup = self.lookup(images.device)
        if self.crop_fraction is not None:
            images = center_crop_and_resize(images, self.crop_fraction)
        count, height, width, channels = images.shape
        normalized = lookup[
            torch.arange(channels, device=images.device)[None, None, None, :], images.long()
        ]
        patch = self.patch_size
        temporal = self.temporal_patch_size
        merge = self.merge_size
        rows, cols = height // (patch * merge), width // (patch * merge)
        # Split merge windows, then patches, keeping tensors at rank <= 8 (OpenVINO GPU limit).
        # Output order matches the previous single rank-10 permute and the HF Qwen3-VL processor:
        # windows row-major, then (merge_h, merge_w) patches, each flattened as (C, T, ph, pw).
        # (count, H, W, C) -> (count, rows, merge*ph, cols, merge*pw*C)
        windows = normalized.reshape(count, rows, merge * patch, cols, merge * patch * channels)
        # -> (count*rows*cols, merge_h, ph, merge_w, pw, C)
        windows = windows.permute(0, 1, 3, 2, 4).reshape(
            count * rows * cols, merge, patch, merge, patch, channels
        )
        # -> (n_patches, C, 1, ph, pw), then repeat the still image over the temporal patch.
        patches = windows.permute(0, 1, 3, 5, 2, 4).reshape(-1, channels, 1, patch, patch)
        patches = patches.expand(-1, -1, temporal, -1, -1)
        return patches.reshape(
            count * rows * cols * merge * merge, channels * temporal * patch * patch
        )

    def pixel_values(
        self,
        lookup: torch.Tensor,
        image_groups: list[torch.Tensor],
        image_order: tuple[tuple[int, ...], ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ordered_pixels = [None] * sum(len(indices) for indices in image_order)
        for images, indices in zip(image_groups, image_order):
            patches = self.patchify(images, lookup)
            patches = patches.reshape(len(indices), -1, patches.shape[-1])
            for index, pixels in zip(indices, patches):
                ordered_pixels[index] = pixels
        return torch.cat(ordered_pixels).to(dtype)


def center_crop_and_resize(images: torch.Tensor, fraction: float) -> torch.Tensor:
    height, width = images.shape[1:3]
    crop_height, crop_width = max(1, int(height * fraction)), max(1, int(width * fraction))
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    window = images[:, top : top + crop_height, left : left + crop_width].permute(0, 3, 1, 2)
    resized = F.interpolate(
        window.float(), size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
