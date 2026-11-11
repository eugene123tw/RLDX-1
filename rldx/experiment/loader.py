import json
from pathlib import Path

from safetensors import safe_open
from termcolor import colored

from rldx.configs.base_config import Config
from rldx.data.utils import parse_modality_configs
from rldx.dist import rank_zero_print as _print
from rldx.model.core.processor import RLDXProcessor, RLDXProcessorConfig, build_projector_mapping
from rldx.model.core.rldx import RLDX
from rldx.model.lora import to_base_layer_keys
from rldx.model.modules.action_model.padded_swiglu import canonicalize_padded_swiglu_checkpoint


def _load_non_complex_tensors(checkpoint_file: str) -> dict:
    """Older checkpoints saved the complex RoPE ``freqs_cis`` buffers; the model recomputes them."""
    tensors = {}
    with safe_open(checkpoint_file, framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if not t.is_complex():
                tensors[key] = t.contiguous()
    return tensors


def _load_checkpoint_state_dict(ckpt_path: Path, read_tensors) -> dict:
    """Load checkpoint state dict from sharded or single safetensors file."""
    if (ckpt_path / "model.safetensors.index.json").exists():
        with open(ckpt_path / "model.safetensors.index.json") as _f:
            index = json.load(_f)
        shard_files = set(index["weight_map"].values())
        full_state_dict = {}
        for shard_file in sorted(shard_files):
            full_state_dict.update(read_tensors(str(ckpt_path / shard_file)))
        return full_state_dict
    elif (ckpt_path / "model.safetensors").exists():
        return read_tensors(str(ckpt_path / "model.safetensors"))
    else:
        raise FileNotFoundError(f"No model weights found in {ckpt_path}")


def _load_state_dict_with_shape_filter(model, state_dict: dict):
    """Load state dict into model, gracefully handling shape mismatches for architecture changes."""
    state_dict = canonicalize_padded_swiglu_checkpoint(model, state_dict)
    model_state = model.state_dict()
    filtered = {}
    shape_mismatched = []

    for key, value in state_dict.items():
        if key in model_state:
            if value.shape == model_state[key].shape:
                filtered[key] = value
            else:
                shape_mismatched.append(
                    f"  {key}: checkpoint={list(value.shape)} vs model={list(model_state[key].shape)}"
                )

    missing, _ = model.load_state_dict(filtered, strict=False)
    return missing, shape_mismatched


def _loading_kwargs(config: Config) -> dict:
    training = config.training
    kwargs = {
        "trust_remote_code": training.transformers_trust_remote_code,
        "local_files_only": training.transformers_local_files_only,
    }
    if config.model.model_revision is not None:
        kwargs["revision"] = config.model.model_revision
    if training.transformers_cache_dir is not None:
        kwargs["cache_dir"] = training.transformers_cache_dir
    if training.transformers_access_token is not None:
        kwargs["token"] = training.transformers_access_token
    return kwargs


def load_processor(config: Config) -> RLDXProcessor:
    data = dict(
        modality_configs=parse_modality_configs(config.data.modality_configs),
    )
    shared = dict(
        rldx_config=config.model,
        embodiment_id_mapping=build_projector_mapping(config.data.projector_grouping),
        transformers_loading_kwargs=_loading_kwargs(config),
    )
    if config.training.start_from_checkpoint is not None:
        return RLDXProcessor.from_pretrained(
            config.training.start_from_checkpoint, **data, **shared
        )
    return RLDXProcessor(RLDXProcessorConfig(**data), **shared)


def load_model(config: Config) -> tuple[RLDX, set[str]]:
    loading_kwargs = _loading_kwargs(config)
    skip_weight_loading = config.training.skip_weight_loading
    if config.training.start_from_checkpoint is not None and not skip_weight_loading:
        ckpt_path = Path(config.training.start_from_checkpoint)
        _print(f"\n[i] Loading checkpoint from {ckpt_path}")

        # 1. Create model from runtime config (skip backbone pretrained weights)
        model = RLDX(
            config.model,
            transformers_loading_kwargs={**loading_kwargs, "skip_pretrained_weights": True},
        )

        # 2. Load checkpoint weights once
        state_dict = _load_checkpoint_state_dict(ckpt_path, _load_non_complex_tensors)

        # Remap older physics key prefixes onto the current PhysicsHead
        # layout (e.g. ``physics_encoder`` → ``physics.physics_cond_encoder``).
        from rldx.model.modules.action_model.physics_head import remap_physics_keys

        state_dict = remap_physics_keys(state_dict)
        state_dict = to_base_layer_keys(model, state_dict)

        # 3. Apply weights with shape filtering
        missing_keys, shape_mismatched = _load_state_dict_with_shape_filter(model, state_dict)

        if shape_mismatched:
            _print(
                colored(
                    f"[i] Shape-mismatched keys (re-initialized, {len(shape_mismatched)}):",
                    "yellow",
                )
            )
            for msg in shape_mismatched:
                _print(colored(msg, "yellow"))

        if missing_keys:
            _print(
                colored(f"[i] Missing keys (newly added modules, {len(missing_keys)}):", "yellow")
            )
            for k in missing_keys:
                _print(colored(f"  {k}", "yellow"))

        _print(
            colored(
                f"\n[i] Checkpoint loaded: {len(missing_keys)} missing, "
                f"{len(shape_mismatched)} shape-mismatched",
                "green",
            )
        )

        unloaded = set(missing_keys)

    else:
        # No checkpoint, or --skip-weight-loading: the backbone still loads
        # its pretrained weights from HuggingFace, everything else is fresh.
        if skip_weight_loading and config.training.start_from_checkpoint is not None:
            _print(
                colored(
                    "[w] --skip-weight-loading: architecture only, "
                    f"weights from {config.training.start_from_checkpoint} NOT loaded",
                    "yellow",
                )
            )
        model = RLDX(config.model, transformers_loading_kwargs=loading_kwargs)
        unloaded = set()

    _print(colored(f"\nModel Config: {model.config}", "yellow"))
    # Print parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise ValueError(
            "the model config trains no parameter; check the tune flags, LoRAs and tune_projector"
        )
    _print(f"\n[i] Total parameters: {total_params:,}")
    _print(
        f"[i] Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )
    _print("Model: ", model)

    return model, unloaded
