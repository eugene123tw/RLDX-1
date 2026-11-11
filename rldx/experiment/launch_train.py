import json
import logging
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional

from huggingface_hub import snapshot_download
from termcolor import colored
import tyro

from rldx import plugins
from rldx.configs.base_config import read_run
from rldx.configs.data.data_config import SingleDatasetConfig
from rldx.configs.data.embodiment_configs import load_modality_config_file
from rldx.configs.model.rldx import RLDXConfig
from rldx.configs.training.cli import TrainArgs
from rldx.dist import rank_zero_print as _print
from rldx.experiment.assembly import AssemblyInputs, assemble_run_config
from rldx.experiment.experiment import run
from rldx.experiment.utils import assert_world_size_matches


def _resolve_base_model(
    base_path_or_id: Optional[str], revision: Optional[str] = None
) -> Optional[str]:
    """Return a local path for the base checkpoint, downloading via HF hub if needed.

    Args:
        base_path_or_id: HF Hub repo id or local path. ``None`` → no base model.
        revision: HF git commit / branch / tag to pin.

    Returns ``None`` when no base model was requested.
    """
    if not base_path_or_id:
        return None
    if "/" in base_path_or_id and not Path(base_path_or_id).exists():
        return snapshot_download(
            repo_id=base_path_or_id,
            repo_type="model",
            local_dir=None,
            revision=revision,
        )
    return base_path_or_id


def _datasets(cli_config: TrainArgs) -> list[SingleDatasetConfig]:
    if cli_config.data is None:
        raise ValueError(
            "No training data: give data:paths --data.paths … --data.embodiment-tag …, "
            "or data:mix --data.root … --data.name …."
        )
    return cli_config.data.specs()


def _checkpoint_model(base_model_path: Optional[str]) -> Optional[RLDXConfig]:
    if base_model_path is None:
        return None
    return RLDXConfig.from_dict(json.loads((Path(base_model_path) / "config.json").read_text()))


def _read_dataset_modality_jsons(datasets: list[SingleDatasetConfig]) -> dict[str, dict]:
    """Read each dataset's ``meta/modality.json`` at the launch-time I/O
    boundary and return ``{dataset_path: parsed_json}``.

    Kept here (not in ``assemble_run_config``, which is pure) so the derived-dim
    check in assembly can run against the modality configs the run will actually
    train with (``run_config.data.modality_configs``) using these pre-read
    slices. Fail-fast: a dataset missing ``meta/modality.json`` raises, since
    its state/action width can't be validated against the projector widths.
    """
    out: dict[str, dict] = {}
    for spec in datasets:
        for path in spec.dataset_paths:
            if path in out:
                continue
            modality_json = Path(path) / "meta" / "modality.json"
            if not modality_json.exists():
                raise FileNotFoundError(
                    f"{modality_json} not found — required to validate the dataset's "
                    f"state/action width against max_state_dim / max_action_dim."
                )
            out[path] = json.loads(modality_json.read_text())
    return out


def resolve_cli_config(argv: list[str]) -> tuple[TrainArgs, ModuleType | None]:
    """Parse ``argv`` over the preset ``--experiment`` names, if a plugin defines it.

    Also returns that plugin: only it knows the experiment, so only it checks the run.
    """
    owner, default = None, TrainArgs()
    # Two-pass parse so the preset seeds defaults that tyro parses over;
    # anything typed on the command line still wins.
    for p in plugins.modules("presets"):
        if (preset := p.default_config(argv)) is not None:
            owner, default = p, preset
            break
    cli_config = tyro.cli(
        TrainArgs,
        default=default,
        args=argv,
        description=__doc__,
        config=(tyro.conf.CascadeSubcommandArgs,),
    )
    if cli_config.experiment != default.experiment:
        raise ValueError(
            f"No plugin defines experiment {cli_config.experiment!r} "
            f"(enabled via {plugins.ENV}: {plugins.enabled()})"
        )
    # Re-check the resolved config, not the spec: a flag can move the run.
    if owner is not None:
        owner.check(cli_config)
    return cli_config, owner


if __name__ == "__main__":
    if "LOG_LEVEL" not in os.environ:
        os.environ["LOG_LEVEL"] = "INFO"
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level))

    cli_config, preset_plugin = resolve_cli_config(sys.argv[1:])
    assert_world_size_matches(cli_config)

    datasets = _datasets(cli_config)

    # All rank workers should register for the modality config
    if cli_config.modality_config_path is not None:
        load_modality_config_file(cli_config.modality_config_path)

    # I/O boundary: HF download + YAML read. Isolated here so assemble_run_config
    # remains pure.
    base_model_path = _resolve_base_model(
        cli_config.base_model_path, revision=cli_config.model.model_revision
    )
    base_run = read_run(base_model_path) if base_model_path is not None else None
    if preset_plugin is not None:
        preset_plugin.check_base(cli_config, base_run)
    # Read modality.json slices here (I/O boundary) so assemble can reconcile
    # the derived state/action width against max_state_dim / max_action_dim
    # using the modality configs the run actually trains with.
    modality_jsons = _read_dataset_modality_jsons(datasets)

    # Pure assembly (model + features + training overrides)
    run_config = assemble_run_config(
        AssemblyInputs(
            cli=cli_config,
            datasets=datasets,
            base_model_path=base_model_path,
            base_run=base_run,
            checkpoint_model=_checkpoint_model(base_model_path),
            modality_jsons=modality_jsons,
        )
    )

    _print(colored(f"[i] {cli_config=}", "yellow"))
    run(run_config)
