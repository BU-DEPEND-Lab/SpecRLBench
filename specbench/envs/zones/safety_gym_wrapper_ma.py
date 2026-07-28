from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from gymnasium.spaces import Box

from specbench.utils.ltl.logic import Assignment


class SafetyGymWrapperMA(gymnasium.Wrapper):
    """
    A wrapper from safety gymnasium LTL environments to the gymnasium API.
    """

    def __init__(self, env: Any, wall_sensor=True):
        super().__init__(env)
        self.render_parameters.camera_name = 'track'
        self.render_parameters.width = 256
        self.render_parameters.height = 256
        self.num_lidar_bins = env.unwrapped.task.lidar_conf.num_bins

        obs_space = env.observation_space
        if callable(obs_space):
            obs_space = obs_space(None)
        obs_keys = obs_space.spaces.keys()
        self.colors = set()
        self.atomic_propositions = set()
        self.num_agents = env.num_agents
        for key in obs_keys:
            if "zones" in key.split('_'):
                color = key.split('_')[0]
                self.colors.add(color)
                for i in range(self.num_agents):
                    self.atomic_propositions.add(color + '_' + str(i))

        obs_space = env.observation_space
        if callable(obs_space):
            obs_space = obs_space(None)
        if isinstance(obs_space, spaces.Dict):
            self.observation_space = obs_space
        else:
            self.observation_space = spaces.Dict(obs_space)

        if wall_sensor:
            for i, a in enumerate(self.unwrapped.possible_agents):
                self.observation_space[f'wall_sensor_{i}'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)
        self.last_dist = None

    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        if 'wall_sensor' in info["agent_0"]:
            for i, agent in enumerate(self.unwrapped.possible_agents):
                obs[agent][f'wall_sensor_{i}'] = info[agent]['wall_sensor']

        self.env.unwrapped.task.original_obs = obs

        if 'cost_ltl_walls' in info["agent_0"]:
            for i, a in enumerate(self.unwrapped.possible_agents):
                terminated[a] = terminated[a] or \
                    info[a]['cost_ltl_walls'] > 0 or \
                    info[a]['cost_collision'] > 0

        info['propositions'] = []
        for i, a in enumerate(self.unwrapped.possible_agents):
            zone_info = info[a]
            active_props = [c + '_' + str(i) for c in self.colors if zone_info[f'cost_zones_{c}'] > 0]
            info['propositions'].extend(active_props)

        return obs, reward, terminated, truncated, info

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        info['propositions'] = []
        for i, a in enumerate(self.unwrapped.possible_agents):
            obs[a][f'wall_sensor_{i}'] = np.array([0, 0, 0, 0])
        self.env.unwrapped.task.original_obs = obs
        return obs, info

    def get_propositions(self) -> list[str]:
        return sorted(self.atomic_propositions)

    def get_possible_assignments(self) -> list[Assignment]:
        assignments = []
        agent_props = {}
        for prop in self.atomic_propositions:
            agent_idx = prop[-1]
            agent_props.setdefault(agent_idx, set()).add(prop)
        per_agent_assignments = []
        for props in agent_props.values():
            per_agent_assignments.append(Assignment.zero_or_one_propositions(props))
        import itertools
        for combo in itertools.product(*per_agent_assignments):
            merged = Assignment()
            for a in combo:
                merged.update(a)
            assignments.append(merged)
        assert len(assignments) == (len(self.colors) + 1) ** self.num_agents, \
            f"Expected {(len(self.colors) + 1) ** self.num_agents} assignments, got {len(assignments)}"
        return assignments

    def get_all_possible_assignments(self) -> list[Assignment]:
        return Assignment.all_possible_assignments(tuple(self.get_propositions()))
