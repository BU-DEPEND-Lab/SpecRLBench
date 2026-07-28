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

        # Robustly handle both property and method for observation_space
        obs_space = env.observation_space
        if callable(obs_space):
            # If it's a method, call with None (or agent name if needed)
            obs_space = obs_space(None)
        obs_keys = obs_space.spaces.keys()
        # print(f"env.observation_space: {env.observation_space}")
        # obs_keys = env.observation_space["agent_0"].spaces.keys()
        # print(f"DEBUG: obs_keys = {obs_keys}")
        self.colors = set()
        self.atomic_propositions = set()
        self.num_agents = env.num_agents
        # self.num_agents = 1
        for key in obs_keys:
            # if key.endswith('zones_lidar'):
            if "zones" in key.split('_'):
                color = key.split('_')[0]
                self.colors.add(color)
                for i in range(self.num_agents):
                    self.atomic_propositions.add(color + '_' + str(i))
        # print(f"DEBUG: self.colors = {self.colors}")
        # print(f"DEBUG: self.atomic_propositions = {self.atomic_propositions}")

        # Robustly handle both property and method for observation_space
        obs_space = env.observation_space
        if callable(obs_space):
            obs_space = obs_space(None)
        # If it's already a Dict, use it directly; otherwise, wrap if needed
        if isinstance(obs_space, spaces.Dict):
            self.observation_space = obs_space
        else:
            self.observation_space = spaces.Dict(obs_space)
        # self.observation_space = spaces.Dict(env.observation_space["agent_0"])  # copy the observation space

        if wall_sensor:
            for i, a in enumerate(self.unwrapped.possible_agents):
                self.observation_space[f'wall_sensor_{i}'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)
            # self.observation_space['wall_sensor'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)
            # self.observation_space['wall_sensor1'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)
        # print(f"DEBUG: self.observation_space = {self.observation_space}")
        self.last_dist = None

    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        # print(f"DEBUG: info = {info}")
        # print(f"DEBUG: terminated = {terminated}, truncated = {truncated}")

        # print(f"DEBUG: obs = {obs}")
        # print(f"DEBUG: info = {info}")
        # update env boundary wall sensor info
        if 'wall_sensor' in info["agent_0"]:
            # obs["agent_0"]['wall_sensor']  = info["agent_0"]['wall_sensor']
            # obs["agent_1"]['wall_sensor1'] = info["agent_1"]['wall_sensor']
            # for i, a in enumerate(self.env.agents):
            #     suffix = '' if i == 0 else str(i)
            #     obs[a][f'wall_sensor{suffix}'] = info[a]['wall_sensor']
            for i, agent in enumerate(self.unwrapped.possible_agents):
                obs[agent][f'wall_sensor_{i}'] = info[agent]['wall_sensor']
            # print(f"DEBUG: obs wrapper = {obs}")

        # self.task.original_obs = obs
        self.env.unwrapped.task.original_obs = obs
        # print(f"DEBUG: original_obs SafetyGymWrapper step = {obs}")

        # update termination based on wall collision
        # TODO: may need to have seperate termination for each agent, 
        # one agent may violate its own subgoal such that the whole spec cannot be satisfied 
        # (the episode should terminate), but it does not necessarily mean the other agent's action is not valid. 
        if 'cost_ltl_walls' in info["agent_0"]:

            # terminated["agent_0"] = terminated["agent_0"] or \
            #     info["agent_0"]['cost_ltl_walls'] > 0 or \
            #     info["agent_0"]['cost_collision'] > 0
            # terminated["agent_1"] = terminated["agent_1"] or \
            #     info["agent_1"]['cost_ltl_walls'] > 0 or \
            #     info["agent_1"]['cost_collision'] > 0
            for i, a in enumerate(self.unwrapped.possible_agents):
                terminated[a] = terminated[a] or \
                    info[a]['cost_ltl_walls'] > 0 or \
                    info[a]['cost_collision'] > 0

            # if any(terminated.values()):
            #     print(f"DEBUG: collision detected!")
                # info['violation'] = True

        info['propositions'] = []
        for i, a in enumerate(self.unwrapped.possible_agents):
            # suffix = '' if i == 0 else f"_{i}"
            zone_info = info[a]
            active_props = [c + '_' + str(i) for c in self.colors if zone_info[f'cost_zones_{c}'] > 0]
            info['propositions'].extend(active_props)

        # zone_info = info["agent_0"]
        # active_props = [c + '_0' for c in self.colors if zone_info[f'cost_zones_{c}'] > 0]
        # info['propositions'].extend(active_props)

        # zone_info = info["agent_1"]
        # active_props = [c + '_1' for c in self.colors if zone_info[f'cost_zones_{c}'] > 0]
        # info['propositions'].extend(active_props)

        return obs, reward, terminated, truncated, info

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        info['propositions'] = []
        # obs["agent_0"]['wall_sensor'] = np.array([0, 0, 0, 0])
        # obs["agent_1"]['wall_sensor1'] = np.array([0, 0, 0, 0])
        for i, a in enumerate(self.unwrapped.possible_agents):
            obs[a][f'wall_sensor_{i}'] = np.array([0, 0, 0, 0])
        # Ensure original_obs is set at reset
        self.env.unwrapped.task.original_obs = obs
        # print(f"DEBUG: original_obs SafetyGymWrapper reset= {obs}")
        return obs, info

    def get_propositions(self) -> list[str]:
        return sorted(self.atomic_propositions)

    def get_possible_assignments(self) -> list[Assignment]:
        # For multi-agent: allow at most one proposition per agent to be true, but allow different agents' props to be true simultaneously
        assignments = []
        agent_props = {}
        # Group props by agent index (e.g., color0, color1)
        for prop in self.atomic_propositions:
            agent_idx = prop[-1]
            agent_props.setdefault(agent_idx, set()).add(prop)
        # For each agent, get zero_or_one assignments
        per_agent_assignments = []
        for props in agent_props.values():
            per_agent_assignments.append(Assignment.zero_or_one_propositions(props))
        # Cartesian product of per-agent assignments
        import itertools
        for combo in itertools.product(*per_agent_assignments):
            merged = Assignment()
            for a in combo:
                merged.update(a)
            assignments.append(merged)
        assert len(assignments) == (len(self.colors) + 1) ** self.num_agents, \
            f"Expected {(len(self.colors) + 1) ** self.num_agents} assignments, got {len(assignments)}"
        # print(f"DEBUG: possible assignments: {[a.get_true_propositions() for a in assignments]}")
        return assignments

    def get_all_possible_assignments(self) -> list[Assignment]:
        return Assignment.all_possible_assignments(tuple(self.get_propositions()))
