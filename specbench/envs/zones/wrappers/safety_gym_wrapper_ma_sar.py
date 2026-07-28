from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from gymnasium.spaces import Box

from specbench.utils.ltl.logic import Assignment
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
    agent_has_entrapped_at_building,
    agent_inside_building_idx,
)


class SafetyGymWrapperMASAR(gymnasium.Wrapper):
    """
    A wrapper from safety gymnasium LTL environments to the gymnasium API.
    """
    sb3 = False
    action_dim = 2

    def __init__(self, env: Any, wall_sensor=True, sb3=False):
        super().__init__(env)
        self.unwrapped.render_parameters.camera_name = 'track'
        self.unwrapped.render_parameters.width = 256
        self.unwrapped.render_parameters.height = 256
        self.num_lidar_bins = env.unwrapped.task.lidar_conf.num_bins
        self.sb3 = sb3
        self.prev_casualty_visible = False
        self.prev_entered_building = False

        # Robustly handle both property and method for observation_space
        obs_space = env.observation_space
        if callable(obs_space):
            obs_space = obs_space(None)
        obs_keys = obs_space.spaces.keys()
        self.colors = set()
        self.atomic_propositions = set()
        self.num_agents = env.unwrapped.num_agents
        for key in obs_keys:
            if "zones" in key.split('_'):
                color = key.split('_')[0]
                self.colors.add(color)
                for i in range(self.num_agents * 2):
                    self.atomic_propositions.add(color + '_' + str(i))

        obs_space = env.observation_space
        if callable(obs_space):
            obs_space = obs_space(None)
        if isinstance(obs_space, spaces.Dict):
            self.observation_space = obs_space
        else:
            self.observation_space = spaces.Dict(obs_space)

        if self.sb3:
            act_space = env.action_space
            if callable(act_space):
                act_space = Box(low=-1.0, high=1.0, shape=(self.num_agents * self.action_dim,))
            if isinstance(act_space, spaces.Box):
                self.action_space = act_space
            else:
                raise TypeError(f"Expected Box action space for SB3, got {type(act_space)}")
        if wall_sensor:
            for i, a in enumerate(self.env.unwrapped.possible_agents):
                self.observation_space[f'wall_sensor_{i}'] = Box(
                    low=0.0, high=1.0, shape=(4,), dtype=np.float64,
                )

    def step(self, action: ActType):
        if self.sb3:
            action = self.dictify_action(action)
        obs, reward, cost, terminated, truncated, info = super().step(action)

        # Update env boundary wall sensor info
        if 'wall_sensor' in info["agent_0"]:
            for i, agent in enumerate(self.env.unwrapped.possible_agents):
                obs[agent][f'wall_sensor_{i}'] = info[agent]['wall_sensor']

        self.env.unwrapped.task.original_obs = obs

        # TODO: may need to have separate termination for each agent,
        # one agent may violate its own subgoal such that the whole spec cannot be satisfied
        # (the episode should terminate), but it does not necessarily mean the other agent's
        # action is not valid.

        info['propositions'] = []
        info['casualty_visible'] = False
        for i, a in enumerate(self.env.unwrapped.possible_agents):
            agent_info: dict = info[a]
            active_props = {}
            for k, v in agent_info.items():
                if isinstance(v, (int, float)) and v != 0 and "cost_sum" not in k:
                    active_props[f"{k}_{i}"] = v

            info['propositions'].extend(active_props.keys())

            # Mask entrapped lidar until agent is inside or team has sticky-entered a building.
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

            # Surface casualty visibility logic
            if (
                f'surface_casualtys_lidar_{i}' in obs[a]
                and max(obs[a][f'surface_casualtys_lidar_{i}']) != 0.0
                and not self.prev_casualty_visible
            ):
                info['casualty_visible'] = True
                self.prev_casualty_visible = True
                # reward[a] += 1.0

        # Collaborative SAR: end episode only when the full team mission is complete
        mission_complete = all(self.env.unwrapped.task.goal_achieved)

        # SB3-specific logic for type matching
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
        obs, info = super().reset(seed=seed, options=options)
        info['propositions'] = []
        info['casualty_visible'] = False
        self.prev_casualty_visible = False
        self.prev_entered_building = False
        for i, a in enumerate(self.env.unwrapped.possible_agents):
            obs[a][f'wall_sensor_{i}'] = np.array([0, 0, 0, 0])
        self.env.unwrapped.task.original_obs = obs
        if self.sb3:
            obs = self.flatten_obs(obs)
        # print(f"seed={seed}")
        return obs, info

    def get_propositions(self) -> list[str]:
        return sorted(self.atomic_propositions)

    def get_possible_assignments(self) -> list[Assignment]:
        # For multi-agent: allow at most one proposition per agent to be true, but allow
        # different agents' props to be true simultaneously
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
