from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

import numpy as np


def _flat_action(action: dict) -> dict:
    return {f"action.{key}": value for key, value in action.items()}


LIBERO_ACTION_KEYS = {
    "eef_pos_delta": ("action.x", "action.y", "action.z"),
    "eef_rot_delta": ("action.roll", "action.pitch", "action.yaw"),
    "gripper_close": ("action.gripper",),
}


def _libero_action(action: dict) -> dict:
    return {
        **{
            f"action.{axis}": action["eef_pos_delta"][..., i : i + 1]
            for i, axis in enumerate(("x", "y", "z"))
        },
        **{
            f"action.{axis}": action["eef_rot_delta"][..., i : i + 1]
            for i, axis in enumerate(("roll", "pitch", "yaw"))
        },
        "action.gripper": 1.0 - action["gripper_close"],
    }


@dataclass(frozen=True)
class SimFormat:
    language: str
    cameras: Mapping[str, str] = field(default_factory=dict)
    states: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    action: Callable[[dict], dict] = _flat_action
    env_actions: Mapping[str, tuple[str, ...]] | None = None


LIBERO = SimFormat(
    language="annotation.human.action.task_description",
    cameras={"front_view": "video.image", "left_wrist_view": "video.wrist_image"},
    states={
        "eef_pos_absolute": ("state.x", "state.y", "state.z"),
        "eef_rot_absolute": ("state.roll", "state.pitch", "state.yaw"),
        "gripper_close": ("state.gripper",),
    },
    action=_libero_action,
    env_actions=LIBERO_ACTION_KEYS,
)
GR1 = SimFormat(
    language="annotation.human.coarse_action",
    cameras={"ego_view": "video.ego_view_bg_crop_pad_res256_freq20"},
)
ROBOCASA_KITCHEN = SimFormat(
    language="annotation.human.action.task_description",
    cameras={
        "left_view": "video.res256_image_side_0",
        "right_view": "video.res256_image_side_1",
        "wrist_view": "video.res256_image_wrist_0",
    },
)
ROBOCASA365 = SimFormat(language="annotation.human.task_description")

SIM_FORMATS = {
    "libero_sim": LIBERO,
    "gr1": GR1,
    "gr1_unified": GR1,
    "robocasa_panda_omron": ROBOCASA_KITCHEN,
    "robocasa": ROBOCASA365,
}


@dataclass(frozen=True)
class SimTranslation:
    video: Mapping[str, str]
    state: Mapping[str, tuple[str, ...]]
    language: tuple[str, str]
    env_action: Callable[[dict], dict]

    def observation(self, env_observation: dict) -> dict:
        policy_key, env_key = self.language
        return {
            "video": {key: env_observation[source] for key, source in self.video.items()},
            "state": {key: _state(env_observation, sources) for key, sources in self.state.items()},
            "language": {policy_key: [[str(text)] for text in env_observation[env_key]]},
        }


def parse(
    env_name: str,
    modality_configs: Mapping,
    observation_keys: Collection[str],
    action_keys: Collection[str],
) -> SimTranslation:
    family = env_name.split("/")[0]
    if family not in SIM_FORMATS:
        raise ValueError(f"no sim format for env family {family!r}; known: {sorted(SIM_FORMATS)}")
    sim = SIM_FORMATS[family]
    language_keys = modality_configs["language"].modality_keys
    if len(language_keys) != 1:
        raise ValueError(f"sim rollout feeds one instruction; the policy reads {language_keys}")
    (language_key,) = language_keys
    translation = SimTranslation(
        video={
            key: sim.cameras.get(key, f"video.{key}")
            for key in modality_configs["video"].modality_keys
        },
        state={
            key: sim.states.get(key, (f"state.{key}",))
            for key in modality_configs["state"].modality_keys
        },
        language=(language_key, sim.language),
        env_action=sim.action,
    )
    wanted = [
        *translation.video.values(),
        *(source for sources in translation.state.values() for source in sources),
        sim.language,
    ]
    missing = [key for key in wanted if key not in observation_keys]
    if missing:
        raise ValueError(f"{env_name} observations lack {missing} that the policy needs")
    policy_action_keys = modality_configs["action"].modality_keys
    env_actions = sim.env_actions or {key: (f"action.{key}",) for key in policy_action_keys}
    if set(env_actions) != set(policy_action_keys):
        raise ValueError(
            f"{env_name} turns policy actions {sorted(env_actions)} into env actions; "
            f"the policy predicts {sorted(policy_action_keys)}"
        )
    unknown = [key for keys in env_actions.values() for key in keys if key not in action_keys]
    if unknown:
        raise ValueError(f"{env_name} takes no {unknown} actions")
    return translation


def _state(env_observation: dict, sources: tuple[str, ...]) -> np.ndarray:
    if len(sources) == 1:
        return env_observation[sources[0]]
    return np.concatenate([env_observation[source] for source in sources], axis=-1)
