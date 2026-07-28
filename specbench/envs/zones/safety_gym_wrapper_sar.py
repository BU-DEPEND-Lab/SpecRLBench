from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from gymnasium.spaces import Box

from specbench.envs.zones.safety_gym_wrapper_ma import SafetyGymWrapperMA
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
    agent_inside_building_idx,
)


class SafetyGymWrapperMASAR(SafetyGymWrapperMA):
    """SAR wrapper: extends MA setup; overrides step/reset for rescue propositions."""

    sb3 = False
    action_dim = 2

    def __init__(self, env: Any, wall_sensor=True, sb3=False):
        super().__init__(env, wall_sensor=wall_sensor)
        self.sb3 = sb3
        self.prev_casualty_visible = False
        self.prev_entered_building = False

        # SAR proposition vocabulary (differs from LTL MA zone props).
        self.atomic_propositions = set()
        obs_space = env.observation_space
        if callable(obs_space):
            obs_space = obs_space(None)
        for key in obs_space.spaces.keys():
            if "zones" in key.split('_'):
                color = key.split('_')[0]
                self.colors.add(color)
                for i in range(self.num_agents * 2):
                    self.atomic_propositions.add(color + '_' + str(i))

        if self.sb3:
            act_space = env.action_space
            if callable(act_space):
                act_space = Box(low=-1.0, high=1.0, shape=(self.num_agents * self.action_dim,))
            if isinstance(act_space, spaces.Box):
                self.action_space = act_space
            else:
                raise TypeError(f"Expected Box action space for SB3, got {type(act_space)}")

    def step(self, action: ActType):
        if self.sb3:
            action = self.dictify_action(action)
        obs, reward, cost, terminated, truncated, info = gymnasium.Wrapper.step(self, action)

        if 'wall_sensor' in info["agent_0"]:
            for i, agent in enumerate(self.env.unwrapped.possible_agents):
                obs[agent][f'wall_sensor_{i}'] = info[agent]['wall_sensor']

        self.env.unwrapped.task.original_obs = obs

        info['propositions'] = []
        info['casualty_visible'] = False
        for i, a in enumerate(self.env.unwrapped.possible_agents):
            agent_info: dict = info[a]
            for k, v in agent_info.items():
                if isinstance(v, (int, float)) and v != 0 and "cost_sum" not in k:
                    info['propositions'].append(f"{k}_{i}")

            task = self.env.unwrapped.task
            inside = agent_inside_building_idx(task, i) is not None
            entered = bool(getattr(task, '_buildings_entered', set()))
            if (
                f'entrapped_casualtys_lidar_{i}' in obs[a]
                and not (inside or entered)
            ):
                obs[a][f'entrapped_casualtys_lidar_{i}'] = np.zeros(
                    obs[a][f'entrapped_casualtys_lidar_{i}'].size,
                )

            if (
                f'surface_casualtys_lidar_{i}' in obs[a]
                and max(obs[a][f'surface_casualtys_lidar_{i}']) != 0.0
                and not self.prev_casualty_visible
            ):
                info['casualty_visible'] = True
                self.prev_casualty_visible = True

        mission_complete = all(self.env.unwrapped.task.goal_achieved)

        if self.sb3:
            obs = self.flatten_obs(obs)
            reward = float(np.mean(list(reward.values())))
            truncated = any(list(truncated.values()))
            terminated = any(list(terminated.values())) or mission_complete
        elif mission_complete:
            terminated = {a: True for a in self.env.unwrapped.possible_agents}

        return obs, reward, terminated, truncated, info

    def reset(
          self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        if seed is not None:
            self._layout_seed = seed
        elif hasattr(self, "_layout_seed"):
            self._layout_seed = (self._layout_seed + 1) % 100
            seed = self._layout_seed
        obs, info = gymnasium.Wrapper.reset(self, seed=seed, options=options)
        info['propositions'] = []
        info['casualty_visible'] = False
        self.prev_casualty_visible = False
        self.prev_entered_building = False
        for i, a in enumerate(self.env.unwrapped.possible_agents):
            obs[a][f'wall_sensor_{i}'] = np.array([0, 0, 0, 0])
        self.env.unwrapped.task.original_obs = obs
        if self.sb3:
            obs = self.flatten_obs(obs)
        return obs, info

    def flatten_obs(self, obs):
        return {
            k: v
            for agent_obs in obs.values()
            for k, v in agent_obs.items()
        }

    def dictify_action(self, action) -> dict:
        return {
            f"agent_{i}": action[i * self.action_dim:(i + 1) * self.action_dim]
            for i in range(self.num_agents)
        }
