"""Execution policy shared by models and runtimes."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, get_args


ExecutionBackend = Literal["eager", "compile"]
QuantGroup = Literal["text", "vision", "msat_vl", "msat_sa"]
QuantFormat = Literal["fp8_weight", "int8_weight", "fp8_w8a8", "nvfp4"]


@dataclass(frozen=True)
class QuantizationRecipe:
    formats: Mapping[QuantGroup, QuantFormat] = field(default_factory=dict)
    overrides: Mapping[str, QuantFormat] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for group, format_name in self.formats.items():
            if group not in get_args(QuantGroup):
                raise ValueError(f"unsupported quantization group: {group!r}")
            if format_name not in get_args(QuantFormat):
                raise ValueError(f"unsupported quantization format: {format_name!r}")
        for site, format_name in self.overrides.items():
            group = site.partition(".")[0]
            if group not in self.formats:
                raise ValueError(f"override {site!r} needs its group {group!r} in formats")
            if format_name not in get_args(QuantFormat):
                raise ValueError(f"unsupported quantization format: {format_name!r}")
        object.__setattr__(self, "formats", MappingProxyType(dict(self.formats)))
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))

    def format_for(self, group: QuantGroup, name: str) -> QuantFormat | None:
        return self.overrides.get(f"{group}.{name}", self.formats.get(group))

    def groups_with(self, format_name: QuantFormat) -> set[QuantGroup]:
        groups = {group for group, value in self.formats.items() if value == format_name}
        sites = {site for site, value in self.overrides.items() if value == format_name}
        return groups | {site.partition(".")[0] for site in sites}


@dataclass(frozen=True)
class ServingLimits:
    batch: int  # largest request batch
    instruction_tokens: int  # largest instruction, in tokens

    def __post_init__(self) -> None:
        if self.batch < 1 or self.instruction_tokens < 1:
            raise ValueError(f"invalid serving limits: {self}")


# cuBLASLt picks its sm_90 GEMMs only from this alignment up; training stays bitwise at it
DEFAULT_HIDDEN_ALIGNMENT = 8


@dataclass(frozen=True)
class TrainingRuntimeProfile:
    backend: ExecutionBackend = "compile"
    hidden_alignment: int = DEFAULT_HIDDEN_ALIGNMENT

    def __post_init__(self) -> None:
        if self.backend not in get_args(ExecutionBackend):
            raise ValueError(f"unsupported execution backend: {self.backend!r}")
        if self.hidden_alignment < 1:
            raise ValueError(f"invalid hidden alignment: {self.hidden_alignment!r}")


@dataclass(frozen=True)
class InferenceRuntimeProfile:
    backend: ExecutionBackend = "compile"
    hidden_alignment: int = DEFAULT_HIDDEN_ALIGNMENT
    quantization: QuantizationRecipe = field(default_factory=QuantizationRecipe)
    nvfp4_backends: Mapping[QuantGroup, str] = field(default_factory=dict)
    nvfp4_library_quantize: frozenset[QuantGroup] = frozenset()
    text_bucket: int | None = None

    def __post_init__(self) -> None:
        if self.backend not in get_args(ExecutionBackend):
            raise ValueError(f"unsupported execution backend: {self.backend!r}")
        if self.hidden_alignment < 1:
            raise ValueError(f"invalid hidden alignment: {self.hidden_alignment!r}")
        if self.text_bucket is not None and self.backend != "compile":
            raise ValueError("text_bucket fixes shapes for CUDA graphs, which need torch.compile")
        if self.text_bucket is not None and self.text_bucket < 1:
            raise ValueError(f"invalid text_bucket: {self.text_bucket!r}")
        nvfp4_groups = self.quantization.groups_with("nvfp4")
        if self.nvfp4_backends.keys() != nvfp4_groups:
            raise ValueError(
                f"nvfp4 backends {dict(self.nvfp4_backends)} must name exactly the nvfp4 groups "
                f"{sorted(nvfp4_groups)}"
            )
        if not self.nvfp4_library_quantize <= nvfp4_groups:
            raise ValueError(
                f"library quantize groups {sorted(self.nvfp4_library_quantize)} are not nvfp4 "
                f"groups {sorted(nvfp4_groups)}"
            )
        object.__setattr__(self, "nvfp4_backends", MappingProxyType(dict(self.nvfp4_backends)))


RuntimeProfile = TrainingRuntimeProfile | InferenceRuntimeProfile


PRESETS: Mapping[str, InferenceRuntimeProfile] = MappingProxyType(
    {
        "compile": InferenceRuntimeProfile(),
        "eager": InferenceRuntimeProfile(backend="eager"),
        "any:bf16": InferenceRuntimeProfile(
            hidden_alignment=64,
            text_bucket=16,
        ),
        "any:fp8": InferenceRuntimeProfile(
            hidden_alignment=64,
            quantization=QuantizationRecipe(
                {
                    "text": "fp8_w8a8",
                    "vision": "fp8_w8a8",
                    "msat_vl": "fp8_w8a8",
                    "msat_sa": "fp8_w8a8",
                }
            ),
            text_bucket=16,
        ),
        "thor:nvfp4-mix": InferenceRuntimeProfile(
            hidden_alignment=64,
            quantization=QuantizationRecipe(
                {"text": "fp8_w8a8", "vision": "nvfp4", "msat_vl": "nvfp4", "msat_sa": "fp8_w8a8"},
                overrides={"text.gate_up_proj": "nvfp4"},
            ),
            nvfp4_backends={"text": "cutlass", "vision": "cutlass", "msat_vl": "cutlass"},
            nvfp4_library_quantize=frozenset({"vision"}),
            text_bucket=16,
        ),
        "thor:nvfp4": InferenceRuntimeProfile(
            hidden_alignment=64,
            quantization=QuantizationRecipe(
                {"text": "nvfp4", "vision": "fp8_w8a8", "msat_vl": "nvfp4", "msat_sa": "fp8_w8a8"}
            ),
            nvfp4_backends={"text": "cutlass", "msat_vl": "cutlass"},
            text_bucket=16,
        ),
        "gb10:nvfp4-mix": InferenceRuntimeProfile(
            hidden_alignment=64,
            quantization=QuantizationRecipe(
                {"text": "fp8_w8a8", "vision": "nvfp4", "msat_vl": "nvfp4", "msat_sa": "fp8_w8a8"},
                overrides={"text.gate_up_proj": "nvfp4"},
            ),
            nvfp4_backends={"text": "b12x", "vision": "cutlass", "msat_vl": "cutlass"},
            nvfp4_library_quantize=frozenset({"vision"}),
            text_bucket=16,
        ),
        "gb10:nvfp4": InferenceRuntimeProfile(
            hidden_alignment=64,
            quantization=QuantizationRecipe(
                {"text": "nvfp4", "vision": "nvfp4", "msat_vl": "nvfp4", "msat_sa": "fp8_w8a8"}
            ),
            nvfp4_backends={"text": "b12x", "vision": "cutlass", "msat_vl": "cutlass"},
            nvfp4_library_quantize=frozenset({"vision"}),
            text_bucket=16,
        ),
    }
)
RuntimeProfileName = Literal[tuple(PRESETS)]
