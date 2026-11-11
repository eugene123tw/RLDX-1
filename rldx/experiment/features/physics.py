"""Physics feature — tactile/torque conditioning stream in the action head."""

from __future__ import annotations

from dataclasses import replace

from rldx.dist import rank_zero_print as _print
from rldx.experiment.features.context import AssemblyContext


def _physics_delta_indices(keys: list[str], modality_configs) -> list[int]:
    for modalities in modality_configs.values():
        for key in keys:
            if key in modalities and modalities[key].delta_indices:
                return list(modalities[key].delta_indices)
    return []


class PhysicsFeature:
    name = "physics"
    requires = frozenset()

    @staticmethod
    def is_active(cli) -> bool:
        return cli.physics is not None

    @staticmethod
    def apply(ctx: AssemblyContext) -> None:
        modality_configs = ctx.modality_configs
        physics = ctx.cli.physics.to_config(ctx.model.physics)
        physics_keys = physics.keys
        # Without a physics modality, delta_indices carries over from the checkpoint,
        # and use_flow_matching always does.
        delta = _physics_delta_indices(physics_keys, modality_configs)
        if delta:
            physics = replace(physics, delta_indices=delta)
        # Read-only validation: shared delta_indices across physics modalities
        for emb_key in modality_configs:
            ref_delta = None
            has_any_physics = any(pk in modality_configs[emb_key] for pk in physics_keys)
            if not has_any_physics:
                if not physics.allow_missing:
                    raise ValueError(
                        f"Embodiment '{emb_key}' has no physics modality keys "
                        f"{physics_keys} in its modality config. "
                        f"Use --physics.allow-missing to allow this."
                    )
                continue
            for pk in physics_keys:
                if pk in modality_configs[emb_key]:
                    di = modality_configs[emb_key][pk].delta_indices
                    if ref_delta is None:
                        ref_delta = di
                    else:
                        assert di == ref_delta, (
                            f"All physics modalities must share the same "
                            f"delta_indices. Got mismatch for '{pk}': {di} vs "
                            f"reference: {ref_delta}"
                        )

        ctx.model.physics = physics
        _print(
            f"[i] physics: keys {physics.keys}, dims {physics.dims} "
            f"(total {sum(physics.dims)}), loss_weight {physics.loss_weight}, "
            f"allow_missing {physics.allow_missing}, dropout_prob {physics.dropout_prob}, "
            f"delta_indices {physics.delta_indices}"
        )
