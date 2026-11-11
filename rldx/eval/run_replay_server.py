# SPDX-License-Identifier: Apache-2.0
"""Serve a recorded LeRobot dataset through the policy server protocol."""

from dataclasses import dataclass

from rldx.configs.data.embodiment_configs import modality_configs_for
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.eval.replay_policy import ReplayPolicy
from rldx.serving.server import DEFAULT_MODEL_SERVER_PORT, serve
import tyro


@dataclass
class ReplayServerArgs:
    dataset_path: str
    """LeRobot dataset whose recorded actions answer the queries."""

    execution_horizon: int
    """Actions the robot executes from each chunk; the replay advances by it."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.GENERAL_EMBODIMENT

    modality_config_path: str | None = None
    """A .py config that registers the embodiment's modality config, or a JSON file of it;
    None uses the registered config."""

    host: str = "0.0.0.0"
    port: int = DEFAULT_MODEL_SERVER_PORT


def main(config: ReplayServerArgs) -> None:
    policy = ReplayPolicy(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs_for(config.embodiment_tag, config.modality_config_path),
        exec_horizon=config.execution_horizon,
        embodiment_tag=config.embodiment_tag,
    )
    serve(policy, config.host, config.port)


if __name__ == "__main__":
    main(tyro.cli(ReplayServerArgs))
