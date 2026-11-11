"""motion module feature — Motion-Aware Spatio-Temporal Summarization (requires video)."""

from __future__ import annotations

from rldx.dist import rank_zero_print as _print
from rldx.experiment.features.context import AssemblyContext


class MossFeature:
    name = "motion"
    requires = frozenset({"video"})  # T-E: enforced by _check_dependencies

    @staticmethod
    def is_active(cli) -> bool:
        return cli.motion is not None

    @staticmethod
    def apply(ctx: AssemblyContext) -> None:
        motion = ctx.cli.motion.to_config(ctx.model.motion)
        ctx.model.motion = motion
        _print(
            f"[i] motion: insert_layer {motion.insert_layer}, "
            f"injection_point {motion.injection_point}, pool_type {motion.pool_type}, "
            f"drop {motion.drop}"
        )
