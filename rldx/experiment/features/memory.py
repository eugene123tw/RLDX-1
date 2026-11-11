"""Memory feature — HAMLET memory-augmented cognition tokens."""

from __future__ import annotations

from rldx.dist import rank_zero_print as _print
from rldx.experiment.features.context import AssemblyContext


class MemoryFeature:
    name = "memory"
    requires = frozenset()

    @staticmethod
    def is_active(cli) -> bool:
        return cli.memory is not None

    @staticmethod
    def apply(ctx: AssemblyContext) -> None:
        memory = ctx.cli.memory.to_config(ctx.model.memory)
        ctx.model.memory = memory

        anchors = {-(memory.length - 1 - i) * memory.stride for i in range(memory.length)}
        for emb_key in ctx.modality_configs:
            if "video" in ctx.modality_configs[emb_key]:
                ctx.add_anchors(emb_key, "video", anchors)

        _print(
            f"[i] memory: length {memory.length}, stride {memory.stride}, "
            f"video_anchors: {sorted(anchors)}"
        )
