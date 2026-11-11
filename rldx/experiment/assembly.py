"""Pure-function assembly of CLI + environment inputs into a RunConfig.

I/O (HF download, YAML reading, file existence) and the trainer kickoff stay
in the launcher; this module only mutates config dataclasses.
"""

import copy
from dataclasses import dataclass, replace
from typing import Optional

from rldx.configs.base_config import Config, RunRecord, get_config
from rldx.configs.data.data_config import DataConfig, SingleDatasetConfig
from rldx.configs.model.rldx import RLDXConfig
from rldx.configs.training.cli import TrainArgs
from rldx.configs.training.runtime import TrainingConfig
from rldx.dist import rank_zero_print as _print
from rldx.experiment.features import FEATURES, AssemblyContext, _check_dependencies
from rldx.experiment.utils import compare_model_configs, snapshot_model_config


# ============================================================================
# Assembly inputs — explicit contract
# ============================================================================


@dataclass(frozen=True)
class AssemblyInputs:
    """All data `assemble_run_config` needs. Caller owns I/O (HF download, YAML read)."""

    cli: TrainArgs
    datasets: list[SingleDatasetConfig]
    base_model_path: Optional[str] = None
    """Resolved local path to the base checkpoint (``None`` for from-scratch).
    Used by `_apply_training_overrides` to set
    ``run_config.training.start_from_checkpoint``."""
    base_run: Optional[RunRecord] = None
    """The run that produced the base checkpoint, read only to identify it; a
    finetune never starts its settings from it."""
    checkpoint_model: Optional[RLDXConfig] = None
    """The base checkpoint's ``config.json``, parsed. A finetune starts its model
    config here whether or not the checkpoint ships a yaml. None = from-scratch."""
    modality_jsons: Optional[dict] = None
    """``{dataset_path: parsed meta/modality.json}`` pre-read at the I/O
    boundary (launch_train). Used by _derive_modality_dims together with
    run_config.data.modality_configs (the configs the run actually trains with)
    to reconcile the concatenated state/action width against
    max_state_dim/max_action_dim. None = skip the dim-mismatch check."""


# ============================================================================
# Per-block appliers (each mutates run_config in place; returns None)
# ============================================================================


# CLI fields whose model config field has another name or sits in a sub-config.
_MODEL_FIELD = {
    "backbone_path": "model_name",
    "backbone_select_layer": "select_layer",
    "image_max_area": "image.max_area",
    "image_resize_m": "image.resize_m",
    "random_crop_fraction": "image.crop_fraction",
    "random_rotation_angle": "augmentation.rotation_angle",
    "color_jitter_params": "augmentation.color_jitter",
}


def _set_model_field(model: RLDXConfig, cli_name: str, value) -> None:
    sub_config, _, name = _MODEL_FIELD.get(cli_name, cli_name).rpartition(".")
    if sub_config:
        setattr(model, sub_config, replace(getattr(model, sub_config), **{name: value}))
    else:
        setattr(model, name, value)


def _apply_cli_model_overrides(cli: TrainArgs, run_config: Config) -> None:
    model = run_config.model
    model.model_type = "RLDX-1"
    for name, value in vars(cli.model.with_defaults()).items():
        _set_model_field(model, name, value)
    model.action_model_lora = (
        None
        if cli.action_model_lora is None
        else cli.action_model_lora.to_config(model.action_model_lora)
    )
    model.backbone_lora = (
        None if cli.backbone_lora is None else cli.backbone_lora.to_config(model.backbone_lora)
    )

    given = {name: value for name, value in vars(cli.overrides).items() if value is not None}
    for name, value in given.items():
        _set_model_field(model, name, value)

    # The backbone keeps the LLM layers up to the deepest select_layer.
    select = model.select_layer
    kept = max(select) if isinstance(select, (list, tuple)) else select
    if model.tune_top_llm_layers > kept:
        raise ValueError(
            f"--tune-top-llm-layers {model.tune_top_llm_layers} exceeds the {kept} LLM layers "
            f"select_layer {select} keeps"
        )
    if model.backbone_lora is not None and model.backbone_lora.num_layers > kept:
        raise ValueError(
            f"--backbone-lora.num-layers {model.backbone_lora.num_layers} exceeds the {kept} "
            f"LLM layers select_layer {select} keeps"
        )

    _print(f"[i] n_cog_tokens: {model.n_cog_tokens}")
    _print(f"[i] action_horizon: {model.action_horizon}")

    # Build the model with exactly the CLI's features, even when the checkpoint has them.
    # Video is not offloadable: `_validate_features_match_ckpt` rejects dropping it.
    for name in _OFFLOADABLE_FEATURES:
        if getattr(cli, name) is None:
            setattr(model, name, None)

    model.attn_dropout = cli.action_model.attn_dropout


# Features that can be offloaded from a pretrained checkpoint (model arch
# stays intact apart from the dropped stream, and strict-false state_dict load
# ignores the unused weight prefix). ``use_video`` is intentionally NOT here:
# the release codebase fixes video on (vanilla Qwen3 dropped, ``--use-video``
# CLI knob removed), so video is now an architectural invariant — any
# checkpoint trained with ``use_video=False`` is unsupported and rejected
# separately at the bottom of this function.
# Each name is both the model config field and the CLI group (`<name>:on`).
_OFFLOADABLE_FEATURES = ("memory", "motion", "physics")


def _note_boundary_policy_differs_from_ckpt(cli, base_run: Optional[RunRecord]) -> None:
    """Record when a run starts from a checkpoint trained under another policy.

    ``--base-model-path`` contributes weights only, so differing here is
    ordinary; a run launched with the wrong flag just looks identical otherwise.
    The case that must not be silent is resume, which
    ``assert_resume_matches_boundary_policy`` refuses.
    """
    if base_run is None:
        return  # from-scratch, nothing to compare against.

    ckpt_policy = base_run.episode_boundary_policy
    cli_policy = cli.data_loading.episode_boundary_policy
    if ckpt_policy is None:
        _print(
            "[i] base checkpoint does not record episode_boundary_policy; "
            f"this run uses '{cli_policy.value}'"
        )
    elif ckpt_policy != cli_policy:
        _print(
            f"[i] base checkpoint records episode_boundary_policy='{ckpt_policy.value}'; "
            f"this run uses '{cli_policy.value}'"
        )


def _validate_features_match_ckpt(cli, ckpt_model: Optional[RLDXConfig]) -> None:
    """Reconcile checkpoint feature flags with CLI flags.

    Video is an architectural invariant in the release codebase — a
    pre-release ``use_video=False`` checkpoint is rejected with a clear
    error, since loading non-VTC weights into the VTC backbone would
    silently mismatch shapes deeper in the load. A LoRA checkpoint keeps its
    base weights under the adapters, so the run must continue the same
    adapters. Memory / motion module /
    Physics are soft-offloadable: when the CLI omits the flag the stream is
    dropped from the build and its weights are discarded at load. Going the
    other way (ckpt off, CLI on) is always allowed.
    """
    if ckpt_model is None:
        return  # from-scratch, no checkpoint to compare against.

    # Fatal: a checkpoint without VTC weights cannot be loaded into the
    # release-codebase VTC backbone. Default ``use_video=True`` here
    # matches the ``RLDXConfig`` default, so older configs that lack
    # the field are treated as VTC-shaped rather than rejected.
    if not ckpt_model.use_video:
        raise ValueError(
            "Checkpoint was trained with use_video=False (vanilla Qwen3 "
            "backbone). The release codebase only supports the VTC backbone "
            "(vtc_qwen3_vl) — there is no CLI knob to drop video tokens. "
            "Use a base checkpoint trained with VTC (every released "
            "RLWRLD/RLDX-1-* checkpoint qualifies)."
        )

    # A LoRA checkpoint keeps its base weights under the adapters.
    saved = ckpt_model.backbone_lora
    if saved is not None and (
        cli.backbone_lora is None or cli.backbone_lora.to_config(saved) != saved
    ):
        raise ValueError(f"Checkpoint was trained with backbone-lora:on {saved}; pass the same.")
    saved = ckpt_model.action_model_lora
    if saved is not None and (
        cli.action_model_lora is None or cli.action_model_lora.to_config(saved) != saved
    ):
        raise ValueError(
            f"Checkpoint was trained with action-model-lora:on {saved}; pass the same."
        )

    # Soft: memory / motion / physics off → CLI wins. Log so the user isn't
    # surprised when the pretrained weights for those streams get dropped.
    offloaded = [
        name
        for name in _OFFLOADABLE_FEATURES
        if getattr(ckpt_model, name) is not None and getattr(cli, name) is None
    ]
    if offloaded:
        lines = [
            "[feature offload] Checkpoint has features enabled that the CLI disables. "
            "These streams will be dropped for this run (ckpt weights ignored):"
        ]
        for name in offloaded:
            lines.append(f"  - ckpt {name} on, CLI {name}:on not given → disabled")
        _print("\n".join(lines))


def _validate_action_horizon_matches_modality(
    action_horizon: int,
    modality_configs: dict,
    used_embodiment_tags: set[str] | None = None,
) -> None:
    """Fail fast if a used embodiment's action.delta_indices length disagrees
    with action_horizon. Embodiments not in `used_embodiment_tags` are skipped;
    pass None to check every entry.
    """
    for emb_key, mods in modality_configs.items():
        if used_embodiment_tags is not None and emb_key not in used_embodiment_tags:
            continue
        if "action" not in mods:
            continue
        actual = len(mods["action"].delta_indices)
        if actual != action_horizon:
            raise ValueError(
                f"action_horizon={action_horizon} disagrees with modality_config "
                f"for embodiment '{emb_key}': action.delta_indices has length {actual}. "
                f"Fix either --action-horizon or the modality config so they match."
            )


# Tags a user's own robot trains under; the base only has a mapping for its dims if an
# earlier run (e.g. the user's mid-train) trained this robot under the same tag.
_SHARED_SLOT_TAGS = ("general_embodiment", "new_embodiment")


def _warn_frozen_projector_on_shared_slots(
    tune_projector: bool, used_embodiment_tags: set[str]
) -> None:
    if tune_projector:
        return
    frozen = sorted(set(_SHARED_SLOT_TAGS) & used_embodiment_tags)
    if frozen:
        _print(
            f"[w] --no-tune-projector freezes the state/action encoders and decoder of "
            f"{frozen}. Unless the base checkpoint already trained this robot (same "
            f"state/action layout) under that tag, dims it never trained stay at their "
            f"random init and this embodiment will not be learned properly. Drop "
            f"--no-tune-projector in that case."
        )


def _derive_modality_dims(
    datasets: list[SingleDatasetConfig],
    modality_configs: dict,
    modality_jsons: dict[str, dict],
) -> tuple[int, int]:
    """Largest concatenated (state_dim, action_dim) across ``datasets``.

    Uses ``modality_configs`` — the configs the run will actually train with
    (``run_config.data.modality_configs``), not the module-global registry, so
    the derived width matches the keys ``RLDXProcessor`` concatenates even on
    the checkpoint-YAML path — and the per-dataset ``modality_jsons`` slices
    pre-read at the I/O boundary. The dim is the sum of ``end - start`` over the
    config's state/action ``modality_keys``.

    Fail-fast over silent under-validation: an unregistered embodiment, a
    dataset whose modality.json was not provided, a configured state/action key
    absent from it, or an invalid ``end <= start`` slice all raise with
    dataset/key context. Returning 0 in those cases would let a malformed or
    larger dataset silently lower the max so the assembly guard passes, only for
    the processor pad step to crash mid-training. A modality the config does not
    declare contributes 0 (nothing to concatenate) and is not an error.
    """
    max_state, max_action = 0, 0
    for spec in datasets:
        emb = spec.embodiment_tag
        mods = modality_configs.get(emb)
        if mods is None:
            raise ValueError(
                f"embodiment_tag {emb!r} is not registered in the run's modality "
                f"configs (datasets={spec.dataset_paths}). Load the matching "
                f"--modality-config-path or correct --data.embodiment-tag."
            )
        for path in spec.dataset_paths:
            meta = modality_jsons.get(path)
            if meta is None:
                raise ValueError(
                    f"modality.json for dataset {path!r} was not provided; cannot "
                    f"determine its state/action width."
                )
            for modality in ("state", "action"):
                if modality not in mods:
                    continue
                total = 0
                for key in mods[modality].modality_keys:
                    grp = meta.get(modality, {}).get(key)
                    if not isinstance(grp, dict) or "start" not in grp or "end" not in grp:
                        raise ValueError(
                            f"modality config {modality} key {key!r} is missing or has "
                            f"no start/end slice in {path}/meta/modality.json; cannot "
                            f"determine the concatenated {modality} width."
                        )
                    start, end = int(grp["start"]), int(grp["end"])
                    if end <= start:
                        raise ValueError(
                            f"invalid slice for {modality} key {key!r} in "
                            f"{path}/meta/modality.json: end={end} <= start={start}."
                        )
                    total += end - start
                if modality == "state":
                    max_state = max(max_state, total)
                else:
                    max_action = max(max_action, total)
    return max_state, max_action


def _resolve_dim_mismatch(
    allow_dim_mismatch: bool,
    model_config,
    derived_state_dim: int,
    derived_action_dim: int,
) -> None:
    """Reconcile the modality-config-derived state/action input width with the
    model's ``max_state_dim`` / ``max_action_dim``.

    ``max_state_dim`` / ``max_action_dim`` are the state/action projector input
    widths: the processor zero-pads the concatenated input up to them, and they
    are baked into the projector weights at build time. The derived dim (sum of
    end-start over the modality config's keys; ``0`` = could not resolve from
    modality.json → skip) is the true unpadded width the projectors must hold.

    When a derived dim exceeds its ``max_*_dim``:
      * ``allow_dim_mismatch=False`` (default) → raise ValueError. Without this
        the processor's pad step (``max_dim - dim``) goes negative and crashes
        with an opaque ``negative dimension`` error deep in collation.
      * ``allow_dim_mismatch=True`` → grow ``max_*_dim`` to the derived value.
        The affected projector tensors then shape-mismatch the checkpoint and
        are re-initialized for ALL embodiment slots (the projector weight is a
        single ``[num_categories, dim, ...]`` tensor), losing the pretrained
        warm-start for that projection. Pair with ``--new-param-warmup-steps``.
    """
    for name, label, derived, projector in (
        (
            "max_state_dim",
            "state",
            derived_state_dim,
            "state_encoder.layer1",
        ),
        (
            "max_action_dim",
            "action",
            derived_action_dim,
            "action_encoder.W1 + action_decoder.layer2",
        ),
    ):
        current = getattr(model_config, name)
        if derived <= current:
            continue
        if not allow_dim_mismatch:
            raise ValueError(
                f"modality config produces {label} dim={derived} > {name}={current}. "
                f"The {label} projector input width is fixed at build time, so the "
                f"processor cannot pad to a larger dim (it would crash with a negative "
                f"pad size). Pass --allow-dim-mismatch to grow {name} to {derived} and "
                f"re-initialize {projector} for ALL embodiment slots (PT warm-start for "
                f"that projection is lost; pair with --new-param-warmup-steps), or fix "
                f"the modality config so the concatenated {label} dim fits within {current}."
            )
        _print(
            f"[allow-dim-mismatch] {name}: {current} -> {derived} (modality-config "
            f"derived {label} dim). {projector} will be RE-INITIALIZED for ALL "
            f"embodiment slots — PT warm-start for that projection is lost. "
            f"Consider --new-param-warmup-steps to ease the new params in."
        )
        setattr(model_config, name, derived)


# ============================================================================
# Orchestrator
# ============================================================================


def _load_base_config(inputs: AssemblyInputs) -> Config:
    """Defaults, with the base checkpoint's model config when there is one."""
    run_config = get_config("RLDX-1")
    if inputs.checkpoint_model is not None:
        run_config.model = copy.deepcopy(inputs.checkpoint_model)

    run_config.data = DataConfig(
        **vars(inputs.cli.data_loading), download_cache=False, datasets=list(inputs.datasets)
    )
    run_config.load_config_path = None
    return run_config


def _apply_training_overrides(
    cli: TrainArgs,
    run_config: Config,
    base_model_path: Optional[str],
) -> None:
    """Build run_config.training from the CLI; the trainer reads it, not the model."""
    _print("\n2. Overwriting with CLI 'training' configs...")
    run_config.training = TrainingConfig(
        **vars(cli.training),
        wandb=cli.wandb,
        start_from_checkpoint=base_model_path if cli.base_model_path else None,
    )


def assemble_run_config(inputs: AssemblyInputs) -> Config:
    """Compose the run_config from defaults, the base checkpoint's model config and the CLI.

    Pure: no file I/O, no HF download, no Trainer start. Callers must have
    already resolved base_model_path and downloaded the model.

    Flow:
      1. Load base Config (defaults + checkpoint config.json)
      2. Apply non-feature CLI model overrides (always)
      3. Check FeatureDependency across active features (T-E)
      4. Apply each active feature's mutation via the Registry
      5. Finalize per-modality delta_indices as anchor×stride cross-product (T-A/T-B)
      6. Compare vs checkpoint snapshot if provided
    """
    _print("\n1. Overwriting with CLI 'model' configs...")
    run_config = _load_base_config(inputs)
    _validate_features_match_ckpt(inputs.cli, inputs.checkpoint_model)
    _note_boundary_policy_differs_from_ckpt(inputs.cli, inputs.base_run)
    _apply_cli_model_overrides(inputs.cli, run_config)
    # modality_jsons is None for callers that don't exercise the dim check;
    # launch_train always supplies it.
    if inputs.modality_jsons is not None:
        derived_state_dim, derived_action_dim = _derive_modality_dims(
            inputs.datasets,
            run_config.data.modality_configs,
            inputs.modality_jsons,
        )
        _resolve_dim_mismatch(
            inputs.cli.allow_dim_mismatch,
            run_config.model,
            derived_state_dim,
            derived_action_dim,
        )

    ctx = AssemblyContext(cli=inputs.cli, run_config=run_config)
    active = tuple(F for F in FEATURES if F.is_active(inputs.cli))
    _check_dependencies(active)
    for F in active:
        F.apply(ctx)
    ctx.finalize()

    _validate_action_horizon_matches_modality(
        inputs.cli.model.action_horizon,
        run_config.data.modality_configs,
        used_embodiment_tags={ds.embodiment_tag for ds in inputs.datasets},
    )
    _warn_frozen_projector_on_shared_slots(
        run_config.model.tune_projector,
        used_embodiment_tags={ds.embodiment_tag for ds in inputs.datasets},
    )

    if inputs.checkpoint_model is not None:
        compare_model_configs(snapshot_model_config(inputs.checkpoint_model), run_config.model)

    _apply_training_overrides(inputs.cli, run_config, inputs.base_model_path)

    return run_config
