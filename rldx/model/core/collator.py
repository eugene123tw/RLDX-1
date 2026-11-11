from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property, partial
import re
from typing import Any, Callable, Dict

import numpy as np
from rldx.configs.data.embodiment_configs import ModalityConfig
from rldx.planner.input.tensor_image import TensorImageSpec
from rldx.planner.lru import lru_get
from rldx.telemetry.step_profiler import profile_section as _profile
import torch
from transformers import AutoProcessor, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import BatchEncoding


def build_processor(model_name: str, transformers_loading_kwargs: dict) -> ProcessorMixin:
    """Load a processor from a local path or HF Hub id.

    `AutoProcessor.from_pretrained` already accepts both local paths (absolute
    or relative) and HF repo ids, so no private-filesystem fallback is needed
    here. If you need to redirect a repo id to a local cache, pass the local
    path directly as `model_name`.
    """
    return AutoProcessor.from_pretrained(model_name, **transformers_loading_kwargs)


# The host processor's floor (qwen-vl-utils IMAGE_MIN_TOKEN_NUM).
MIN_MERGED_PATCHES = 4
_PROMPT_MARK = "\x00prompt\x00"


@dataclass(frozen=True)
class PromptFrame:
    encode: Callable[[str], list[int]]
    left: list[int]
    right: list[int]
    before: str
    after: str

    @classmethod
    def from_template(
        cls,
        processor: ProcessorMixin,
        rows: tuple[tuple[int, ...], ...],
        image_first: bool,
        merge_size: int,
    ) -> "PromptFrame":
        images = [{"type": "image"} for _ in rows]
        text = {"type": "text", "text": _PROMPT_MARK}
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": [*images, text] if image_first else [text, *images]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for row in rows:
            pads = "<|placeholder|>" * (int(np.prod(row)) // merge_size**2)
            rendered = rendered.replace(processor.image_token, pads, 1)
        rendered = rendered.replace("<|placeholder|>", processor.image_token)
        head, tail = rendered.split(_PROMPT_MARK)
        tokenizer = processor.tokenizer
        special = re.compile(
            "|".join(map(re.escape, sorted(tokenizer.added_tokens_encoder, key=len, reverse=True)))
        )
        cut = max((match.end() for match in special.finditer(head)), default=0)
        stop = next((match.start() for match in special.finditer(tail)), len(tail))
        return cls(
            encode=partial(tokenizer.encode, add_special_tokens=False),
            left=tokenizer.encode(head[:cut], add_special_tokens=False),
            right=tokenizer.encode(tail[stop:], add_special_tokens=False),
            before=head[cut:],
            after=tail[:stop],
        )

    def __call__(self, prompt: str) -> list[int]:
        return self.left + self.encode(self.before + prompt + self.after) + self.right

    @cached_property
    def fixed_tokens(self) -> int:
        return len(self(""))


class PromptEncoder:
    tensor_text_cache_size = 128
    tensor_text_frame_cache_size = 16

    def __init__(
        self,
        processor: ProcessorMixin,
        *,
        image_max_area: int,
        image_first: bool,
        instruction_limit: int | None = None,
        text_bucket: int | None = None,
    ):
        self.processor = processor
        self.tensor_image_path = TensorImageSpec.from_image_processor(processor.image_processor)
        self.image_max_area = image_max_area
        self.image_first = image_first
        self.instruction_limit = instruction_limit
        self.text_bucket = text_bucket

    @classmethod
    def load(
        cls,
        model_name: str,
        transformers_loading_kwargs: dict,
        *,
        image_max_area: int,
        image_first: bool,
    ) -> "PromptEncoder":
        processor = build_processor(model_name, {**transformers_loading_kwargs, "use_fast": False})
        # Left padding for Flash Attention.
        processor.tokenizer.padding_side = "left"
        return cls(processor, image_max_area=image_max_area, image_first=image_first)

    def encode(self, values: list[dict]) -> dict:
        spec = self.tensor_image_path
        with _profile("collate.vlm.gather_text_images"):
            blocks_by_sample = [value["image_blocks"] for value in values]
            prompts = tuple(value["prompt"] for value in values)
            image_first = tuple(value["image_first"] for value in values)
            groups: dict[tuple[int, int, int], list[tuple[tuple[int, ...], torch.Tensor]]] = {}
            grid_rows = []
            counts = []
            offset = 0
            for blocks in blocks_by_sample:
                frames = blocks[0].shape[0]
                views = len(blocks)
                count = frames * views
                counts.append(count)
                grid_rows.extend([None] * count)
                for view, block in enumerate(blocks):
                    self._check_image_shape(*block.shape[1:3])
                    indices = tuple(offset + frame * views + view for frame in range(frames))
                    groups.setdefault(tuple(block.shape[1:]), []).append((indices, block))
                    row = [
                        1,
                        block.shape[1] // spec.patch_size,
                        block.shape[2] // spec.patch_size,
                    ]
                    for index in indices:
                        grid_rows[index] = row
                offset += count
            raw_images = [torch.cat([block for _, block in group]) for group in groups.values()]
            image_order = tuple(
                tuple(index for indices, _ in group for index in indices)
                for group in groups.values()
            )
        with _profile("collate.vlm.qwen_process_vision_info"):
            grid = torch.tensor(grid_rows, dtype=torch.long)

        rows = grid.tolist()
        bounds = [sum(counts[:index]) for index in range(len(counts) + 1)]
        sample_rows = tuple(
            tuple(map(tuple, rows[start:stop])) for start, stop in zip(bounds, bounds[1:])
        )
        template = repr(self.processor.chat_template)
        with _profile("collate.vlm.hf_processor"):
            tokens = lru_get(
                self.__dict__.setdefault("_tensor_text_cache", OrderedDict()),
                (prompts, sample_rows, image_first, template),
                self.tensor_text_cache_size,
                lambda: self._tokenize_prompts(prompts, sample_rows, image_first),
            )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "image_grid_thw": grid,
            "raw_images": raw_images,
            "raw_image_order": image_order,
        }

    def _check_image_shape(self, height: int, width: int) -> None:
        unit = self.tensor_image_path.patch_size * self.tensor_image_path.merge_size
        if height % unit or width % unit or height * width < MIN_MERGED_PATCHES * unit**2:
            raise ValueError(
                f"image {height}x{width} must be a multiple of {unit} and hold at least "
                f"{MIN_MERGED_PATCHES} merged patches"
            )

    def _tokenize_prompts(self, prompts, sample_rows, image_first) -> BatchEncoding:
        input_ids = []
        for prompt, rows, first in zip(prompts, sample_rows, image_first):
            frame = self.prompt_frame(rows, first)
            ids = frame(prompt)
            instruction = len(ids) - frame.fixed_tokens
            if self.instruction_limit is not None and instruction > self.instruction_limit:
                raise ValueError(
                    f"instruction of {instruction} tokens exceeds the serving limit "
                    f"{self.instruction_limit}"
                )
            input_ids.append(ids)
        return self.processor.tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=self._padded(max(map(len, input_ids))),
            return_tensors="pt",
        )

    def _padded(self, tokens: int) -> int:
        if self.text_bucket is None:
            return tokens
        return -(-tokens // self.text_bucket) * self.text_bucket

    def prompt_frame(self, rows: tuple[tuple[int, ...], ...], image_first: bool) -> PromptFrame:
        return lru_get(
            self.__dict__.setdefault("_tensor_text_frames", OrderedDict()),
            (rows, image_first, repr(self.processor.chat_template)),
            self.tensor_text_frame_cache_size,
            lambda: PromptFrame.from_template(
                self.processor, rows, image_first, self.tensor_image_path.merge_size
            ),
        )

    def max_sample_tokens(self, video: ModalityConfig, instruction_tokens: int) -> tuple[int, int]:
        patches, frame = self._largest_frame(video)
        return patches, self._padded(frame.fixed_tokens + instruction_tokens)

    def warm_up_instructions(self, video: ModalityConfig, instruction_tokens: int) -> list[int]:
        if self.text_bucket is None:
            return sorted({1, min(2, instruction_tokens)})
        fixed = self._largest_frame(video)[1].fixed_tokens
        first_per_width: dict[int, int] = {}
        for tokens in range(1, instruction_tokens + 1):
            first_per_width.setdefault(self._padded(fixed + tokens), tokens)
        return list(first_per_width.values())

    def _largest_frame(self, video: ModalityConfig) -> tuple[int, PromptFrame]:
        merge = self.tensor_image_path.merge_size**2
        patches = self.image_max_area // self.tensor_image_path.patch_size**2 // merge * merge
        images = len(video.modality_keys) * len(video.delta_indices)
        frame = self.prompt_frame(((1, 1, patches),) * images, self.image_first)
        return images * patches, frame


class RLDXDataCollator:
    def __init__(self, encoder: PromptEncoder):
        self.encoder = encoder

    def __call__(self, features: list[Dict[str, Any]]) -> BatchFeature:
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                vlm_inputs = self.encoder.encode(values)
                for k, v in vlm_inputs.items():
                    batch[k] = v

                batch["image_wise_encoding"] = torch.tensor(
                    [v["image_wise_encoding"] for v in values]
                )
                batch["num_views"] = torch.tensor([v["num_views"] for v in values])
                if "num_frames" in values[0]:
                    batch["num_frames"] = torch.tensor([v["num_frames"] for v in values])

            elif key == "vlm_content_list":
                # Memory mode: each sample has K vlm_contents
                # Flatten B samples × K timesteps into B*K backbone inputs
                # Order: [sample0_t0, sample0_t1, ..., sample0_tK-1, sample1_t0, ...]
                all_vlm_contents = []
                for sample_contents in values:
                    all_vlm_contents.extend(sample_contents)

                vlm_inputs = self.encoder.encode(all_vlm_contents)
                for k, v in vlm_inputs.items():
                    batch[k] = v

                batch["image_wise_encoding"] = torch.tensor([1] * len(all_vlm_contents))
                batch["num_views"] = torch.tensor(
                    [
                        content["num_views"]
                        for sample_contents in values
                        for content in sample_contents
                    ]
                )
                if "num_frames" in all_vlm_contents[0]:
                    batch["num_frames"] = torch.tensor(
                        [
                            content["num_frames"]
                            for sample_contents in values
                            for content in sample_contents
                        ]
                    )

            elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
                raise Exception("Not implemented")

            else:
                # state, state_mask, action and action_mask - stack to form batch dimension
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    def __str__(self):
        return f"RLDXDataCollator(encoder={type(self.encoder).__name__})"
