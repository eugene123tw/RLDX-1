import math
from types import MappingProxyType
from typing import Iterable, Mapping

from torch import nn

from rldx.configs.execution import QuantGroup, QuantizationRecipe
from rldx.dist import rank_zero_print
from rldx.model.quantization.formats import FP4_MAX, FP8_MAX
from rldx.model.quantization.linear import GemmSelection, QuantizedLinear, planned_kernel
from rldx.model.quantization.sites import LinearSite


def quantize_linears(
    sites: Iterable[LinearSite],
    recipe: QuantizationRecipe,
    selections: Mapping[QuantGroup, GemmSelection] = MappingProxyType({}),
    library_groups: frozenset[QuantGroup] = frozenset(),
) -> None:
    if not recipe.formats:
        return
    pending = []
    seen = set()
    for site in sites:
        format_name = recipe.format_for(site.group, site.name)
        seen.add(f"{site.group}.{site.name}")
        if format_name is None:
            continue
        linear = getattr(site.owner, site.name, None)
        if isinstance(linear, QuantizedLinear):
            continue
        label = f"quantization site {site.group}.{site.name}"
        if not isinstance(linear, nn.Linear):
            raise ValueError(f"{label} is not a Linear: {type(linear).__name__}")
        selection = selections.get(site.group, GemmSelection())
        planned_kernel(format_name, linear, label, selection)
        bounded = format_name == "nvfp4" and site.input_bound is not None
        scale = _static_scale(site, label) if bounded else None
        pending.append((site, linear, format_name, label, selection, scale))
    unmatched = recipe.overrides.keys() - seen
    if unmatched:
        raise ValueError(f"overrides {sorted(unmatched)} match no quantization site")
    counts: dict[QuantGroup, int] = {group: 0 for group in recipe.formats}
    scales: dict[float, int] = {}
    dynamic = 0
    for site, linear, format_name, label, selection, scale in pending:
        library_quantize = site.group in library_groups and not site.elementwise_input
        layer = QuantizedLinear(linear, format_name, label, selection, library_quantize, scale)
        setattr(site.owner, site.name, layer)
        counts[site.group] += 1
        if scale is not None:
            scales[scale] = scales.get(scale, 0) + 1
        elif format_name == "nvfp4":
            dynamic += 1
    for group, count in counts.items():
        rank_zero_print(f"[quantization] {group}: {count} linears replaced")
    if scales or dynamic:
        rank_zero_print(
            f"[quantization] nvfp4 activation scales: static (scale: sites) {scales}, "
            f"dynamic {dynamic} sites"
        )


def _static_scale(site: LinearSite, label: str) -> float:
    bound = site.input_bound()
    if bound <= 0:
        raise ValueError(f"{label} input bound must be positive, got {bound}")
    return 2.0 ** math.floor(math.log2(FP8_MAX * FP4_MAX / bound))
