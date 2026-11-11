from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Qwen3VLInputSpec:
    """Immutable values needed to prepare a Qwen3-VL model input."""

    n_cog_tokens: int
    spatial_merge_size: int
    vision_max_seqlen: int
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    compression_pattern_token_id: int
    motion_layout: Literal["disabled", "uniform"]
