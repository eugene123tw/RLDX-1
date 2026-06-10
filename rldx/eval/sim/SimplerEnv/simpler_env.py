# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original NVIDIA Isaac GR00T N1.7.
# Original source: https://github.com/NVIDIA/Isaac-GR00T

import cv2
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transforms3d import euler as te, quaternions as tq


class GoogleFractalEnv(gym.Env):
    def __init__(self, env_name: str, image_size: tuple[int, int]):
        env = simpler_env.make(env_name)
        env._max_episode_steps = 10000
        self.env = env
        obs_low = env.observation_space["agent"]["eef_pos"].low
        obs_high = env.observation_space["agent"]["eef_pos"].high
        self.observation_space = gym.spaces.Dict(
            {
                "video.image": gym.spaces.Box(
                    low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
                ),
                "state.end_effector_position": gym.spaces.Box(
                    low=obs_low[0:3], high=obs_high[0:3], shape=(3,)
                ),
                "state.end_effector_rotation": gym.spaces.Box(
                    low=obs_low[3:7], high=obs_high[3:7], shape=(4,)
                ),
                "state.gripper_position": gym.spaces.Box(
                    low=obs_low[7], high=obs_high[7], shape=(1,)
                ),
                "annotation.human.action.task_description": gym.spaces.Text(max_length=512),
            }
        )
        action_low = env.action_space.low
        action_high = env.action_space.high
        self.action_space = gym.spaces.Dict(
            {
                "action.end_effector_position": gym.spaces.Box(
                    low=action_low[0:3], high=action_high[0:3], shape=(3,)
                ),
                "action.end_effector_rotation": gym.spaces.Box(
                    low=action_low[3:6], high=action_high[3:6], shape=(3,)
                ),
                "action.gripper_close": gym.spaces.Box(
                    low=action_low[6], high=action_high[6], shape=(1,)
                ),
            }
        )
        self.image_size = image_size
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.sticky_gripper_action = 0.0
        self.gripper_action_repeat = 0
        self.sticky_gripper_num_repeat = 15

    def reset(self, seed=None, options=None):
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.sticky_gripper_action = 0.0
        self.gripper_action_repeat = 0
        observation, info = self.env.reset()
        observation = self._process_observation(observation)
        info["success"] = False
        return observation, info

    def step(self, action):
        # Packed action format emitted by RLDXSimPolicyWrapper for the
        # OXE-Fractal-schema policy: end_effector_position (3D delta xyz),
        # end_effector_rotation (3D axis-angle delta), gripper_close (1D).
        pos = action["action.end_effector_position"]
        rot = action["action.end_effector_rotation"]
        action_vector = np.concatenate(
            [
                pos[..., 0:1],
                pos[..., 1:2],
                pos[..., 2:3],
                rot[..., 0:1],
                rot[..., 1:2],
                rot[..., 2:3],
                self._postprocess_gripper(action["action.gripper_close"]),
            ],
            axis=0,
        )
        observation, reward, done, truncated, info = self.env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = done
        return observation, reward, done, truncated, info

    def _process_observation(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs)
        proprio = obs["agent"]["eef_pos"]
        # SIMPLER stores quaternion as (w, x, y, z); the OXE Fractal schema
        # expects (x, y, z, w).
        quat_xyzw = np.roll(proprio[3:7], -1)
        gripper_closedness = 1 - proprio[7]
        return {
            "video.image": cv2.resize(img, (self.image_size[1], self.image_size[0])),
            "state.end_effector_position": np.asarray(proprio[0:3], dtype=np.float32),
            "state.end_effector_rotation": np.asarray(quat_xyzw, dtype=np.float32),
            "state.gripper_position": np.asarray([gripper_closedness], dtype=np.float32),
            "annotation.human.action.task_description": self.env.unwrapped.get_language_instruction(),
        }

    def _postprocess_gripper(self, current_gripper_action: float) -> float:
        current_gripper_action = (current_gripper_action * 2) - 1  # [0,1] -> [-1,1]
        relative_gripper_action = -current_gripper_action
        if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0
        return relative_gripper_action


class WidowXBridgeEnv(gym.Env):
    def __init__(self, env_name: str, image_size: tuple[int, int]):
        env = simpler_env.make(env_name)
        env._max_episode_steps = 10000
        self.env = env
        obs_low = env.observation_space["agent"]["eef_pos"].low
        obs_high = env.observation_space["agent"]["eef_pos"].high
        self.observation_space = gym.spaces.Dict(
            {
                "video.image_0": gym.spaces.Box(
                    low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
                ),
                "state.end_effector_position": gym.spaces.Box(
                    low=obs_low[0:3], high=obs_high[0:3], shape=(3,)
                ),
                "state.end_effector_rotation": gym.spaces.Box(
                    low=-np.pi, high=np.pi, shape=(3,)
                ),
                "state.gripper_position": gym.spaces.Box(
                    low=obs_low[7], high=obs_high[7], shape=(1,)
                ),
                "annotation.human.action.task_description": gym.spaces.Text(max_length=512),
            }
        )
        action_low = env.action_space.low
        action_high = env.action_space.high
        self.action_space = gym.spaces.Dict(
            {
                "action.end_effector_position": gym.spaces.Box(
                    low=action_low[0:3], high=action_high[0:3], shape=(3,)
                ),
                "action.end_effector_rotation": gym.spaces.Box(
                    low=action_low[3:6], high=action_high[3:6], shape=(3,)
                ),
                "action.gripper_close": gym.spaces.Box(
                    low=action_low[6], high=action_high[6], shape=(1,)
                ),
            }
        )
        self.image_size = image_size
        # Bridge orientation adjustment
        self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset()
        observation = self._process_observation(observation)
        info["success"] = False
        return observation, info

    def step(self, action):
        # Packed action format emitted by RLDXSimPolicyWrapper for the
        # OXE-bridge_orig-schema policy: end_effector_position (3D delta xyz),
        # end_effector_rotation (3D euler delta), gripper_close (1D).
        pos = action["action.end_effector_position"]
        rot = action["action.end_effector_rotation"]
        action_vector = np.concatenate(
            [
                pos[..., 0:1],
                pos[..., 1:2],
                pos[..., 2:3],
                rot[..., 0:1],
                rot[..., 1:2],
                rot[..., 2:3],
                self._postprocess_gripper(action["action.gripper_close"]),
            ],
            axis=0,
        )
        observation, reward, done, truncated, info = self.env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = done
        return observation, reward, done, truncated, info

    def _process_observation(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs)
        proprio = obs["agent"]["eef_pos"]
        rm_bridge = tq.quat2mat(proprio[3:7])
        rpy_bridge_converted = te.mat2euler(rm_bridge @ self.default_rot.T)
        return {
            "video.image_0": cv2.resize(img, (self.image_size[1], self.image_size[0])),
            "state.end_effector_position": np.asarray(proprio[0:3], dtype=np.float32),
            "state.end_effector_rotation": np.asarray(rpy_bridge_converted, dtype=np.float32),
            "state.gripper_position": np.asarray([proprio[7]], dtype=np.float32),
            "annotation.human.action.task_description": self.env.unwrapped.get_language_instruction(),
        }

    def _postprocess_gripper(self, action):
        # trained with [0, 1], 0 close, 1 open -> convert to SimplerEnv [-1, 1]
        return 2.0 * (action > 0.5) - 1.0


def register_simpler_envs():
    # Google/Fractal
    for env_name in [
        "google_robot_pick_coke_can",
        "google_robot_pick_object",
        "google_robot_move_near",
        "google_robot_open_drawer",
        "google_robot_close_drawer",
        "google_robot_place_in_closed_drawer",
    ]:
        register(
            id=f"simpler_env_google/{env_name}",
            entry_point="rldx.eval.sim.SimplerEnv.simpler_env:GoogleFractalEnv",
            kwargs={"env_name": env_name, "image_size": (256, 320)},
        )

    # WidowX/Bridge
    for env_name in [
        "widowx_spoon_on_towel",
        "widowx_carrot_on_plate",
        "widowx_stack_cube",
        "widowx_put_eggplant_in_basket",
        "widowx_put_eggplant_in_sink",
        "widowx_open_drawer",
        "widowx_close_drawer",
    ]:
        register(
            id=f"simpler_env_widowx/{env_name}",
            entry_point="rldx.eval.sim.SimplerEnv.simpler_env:WidowXBridgeEnv",
            kwargs={"env_name": env_name, "image_size": (256, 256)},
        )
