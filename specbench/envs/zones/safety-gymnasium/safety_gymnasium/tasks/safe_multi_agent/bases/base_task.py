# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
# ==============================================================================
"""Base task."""

from __future__ import annotations

import abc
import os
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import gymnasium
import mujoco
import numpy as np
import yaml

import safety_gymnasium
from safety_gymnasium.tasks.safe_multi_agent.bases.underlying import Underlying
from safety_gymnasium.tasks.safe_multi_agent.utils.common_utils import ResamplingError
from safety_gymnasium.tasks.safe_multi_agent.utils.task_utils import theta2vec


@dataclass
class LidarConf:
    r"""Lidar observation parameters.

    Attributes:
        num_bins (int): Bins (around a full circle) for lidar sensing.
        max_dist (float): Maximum distance for lidar sensitivity (if None, exponential distance).
        exp_gain (float): Scaling factor for distance in exponential distance lidar.
        type (str): 'pseudo', 'natural', 'pseudo_occluded', see self._obs_lidar().
        alias (bool): Lidar bins alias into each other.
    """

    num_bins: int = 16
    max_dist: float = 3
    exp_gain: float = 1.0
    type: str = 'pseudo'
    alias: bool = True


@dataclass
class CompassConf:
    r"""Compass observation parameters.

    Attributes:
        shape (int): 2 for XY unit vector, 3 for XYZ unit vector.
    """

    shape: int = 2


@dataclass
class RewardConf:
    r"""Reward options.

    Attributes:
        reward_orientation (bool): Reward for being upright.
        reward_orientation_scale (float): Scale for uprightness reward.
        reward_orientation_body (str): What body to get orientation from.
        reward_exception (float): Reward when encountering a mujoco exception.
        reward_clip (float): Clip reward, last resort against physics errors causing magnitude spikes.
    """

    reward_orientation: bool = False
    reward_orientation_scale: float = 0.002
    reward_orientation_body: str = 'agent'
    reward_exception: float = -10.0
    reward_clip: float = 10


@dataclass
class CostConf:
    r"""Cost options.

    Attributes:
        constrain_indicator (bool): If true, all costs are either 1 or 0 for a given step.
    """

    constrain_indicator: bool = True


@dataclass
class MechanismConf:
    r"""Mechanism options.

    Starting position distribution.

    Attributes:
        randomize_layout (bool): If false, set the random seed before layout to constant.
        continue_goal (bool): If true, draw a new goal after achievement.
        terminate_resample_failure (bool): If true, end episode when resampling fails,
        otherwise, raise a python exception.
    """

    randomize_layout: bool = True
    continue_goal: bool = True
    terminate_resample_failure: bool = True


@dataclass
class ObservationInfo:
    r"""Observation information generated in running.

    Attributes:
        obs_space_dict (:class:`gymnasium.spaces.Dict`): Observation space dictionary.
    """

    obs_space_dict: gymnasium.spaces.Dict = None


class BaseTask(Underlying):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    r"""Base task class for defining some common characteristic and mechanism.

    Methods:

    - :meth:`dist_goal`: Return the distance from the agent to the goal XY position.
    - :meth:`calculate_cost`: Determine costs depending on the agent and obstacles, actually all
      cost calculation is done in different :meth:`safety_gymnasium.bases.base_obstacle.BaseObject.cal_cost`
      which implemented in different types of object, We just combine all results of them here.
    - :meth:`build_observation_space`: Build observation space, combine agent specific observation space
      and task specific observation space together.
    - :meth:`_build_placements_dict`: Build placement dictionary for all types of object.
    - :meth:`toggle_observation_space`: Toggle observation space.
    - :meth:`_build_world_config`: Create a world_config from all separate configs of different types of object.
    - :meth:`_build_static_geoms_config`: Build static geoms config from yaml files.
    - :meth:`build_goal_position`: Build goal position, it will be called when the task is initialized or
      when the goal is achieved.
    - :meth:`_placements_dict_from_object`: Build placement dictionary for a specific type of object.
    - :meth:`obs`: Combine and return all separate observations of different types of object.
    - :meth:`_obs_lidar`: Return lidar observation, unify natural lidar and pseudo lidar in API.
    - :meth:`_obs_lidar_natural`: Return natural lidar observation.
    - :meth:`_obs_lidar_pseudo`: Return pseudo lidar observation.
    - :meth:`_obs_compass`: Return compass observation.
    - :meth:`_obs_vision`: Return vision observation, that is RGB image captured by camera
      fixed in front of agent.
    - :meth:`_ego_xy`: Return the egocentric XY vector to a position from the agent.
    - :meth:`calculate_reward`: Calculate reward, it will be called in every timestep, and it is
      implemented in different task.
    - :meth:`specific_reset`: Reset task specific parameters, it will be called in every reset.
    - :meth:`specific_step`: Step task specific parameters, it will be called in every timestep.
    - :meth:`update_world`: Update world, it will be called when ``env.reset()`` or :meth:`goal_achieved` == True.

    Attributes:

    - :attr:`num_steps` (int): Maximum number of environment steps in an episode.
    - :attr:`lidar_conf` (:class:`LidarConf`): Lidar observation parameters.
    - :attr:`reward_conf` (:class:`RewardConf`): Reward options.
    - :attr:`cost_conf` (:class:`CostConf`): Cost options.
    - :attr:`mechanism_conf` (:class:`MechanismConf`): Mechanism options.
    - :attr:`action_space` (gymnasium.spaces.Box): Action space.
    - :attr:`observation_space` (gymnasium.spaces.Dict): Observation space.
    - :attr:`obs_info` (:class:`ObservationInfo`): Observation information generated in running.
    - :attr:`_is_load_static_geoms` (bool): Whether to load static geoms in current task which is mean
      some geoms that has no randomness.
    - :attr:`goal_achieved` (bool): Determine whether the goal is achieved, it will be called in every timestep
      and it is implemented in different task.
    """

    def __init__(self, config: dict) -> None:  # pylint: disable-next=too-many-statements
        """Initialize the task.

        Args:
            config (dict): Configuration dictionary, used to pre-config some attributes
              according to tasks via :meth:`safety_gymnasium.register`.
        """
        config = dict(config)
        max_episode_steps = int(config.pop('max_episode_steps', 1000))
        super().__init__(config=config)

        self.num_steps = max_episode_steps

        self.lidar_conf = LidarConf()
        self.compass_conf = CompassConf()
        self.reward_conf = RewardConf()
        self.cost_conf = CostConf()
        self.mechanism_conf = MechanismConf()

        self.action_space = self.agent.action_space
        self.observation_space = None
        self.original_obs = None
        self.obs_info = ObservationInfo()

        self._is_load_static_geoms = False  # Whether to load static geoms in current task.
        self.static_geoms_names: dict
        self.static_geoms_contact_cost: float = None
        self.contact_other_cost: float = None

    def dist_goal(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(self.goal.pos)  # pylint: disable=no-member

    def calculate_cost(self, reset=False) -> dict:
        """Determine costs depending on the agent and obstacles."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        # cost = {'agent_0': {}, 'agent_1': {}}
        cost = {agent: {} for agent in self.agent.possible_agents}

        # Calculate constraint violations
        for obstacle in self._obstacles:
            if obstacle.name == 'gremlins' and reset: 
                cost = {f"agent_{i}": {"cost_collision": 0} for i in range(self.agent.agent_num)}
                continue
                # gremlins positions are not updated yet in the reset step
            obj_cost = obstacle.cal_cost()
            # print(f"DEBUG: obj_cost = {obj_cost}")
            # if 'agent_0' in obj_cost:
            #     cost['agent_0'].update(obj_cost['agent_0'])
            # if 'agent_1' in obj_cost:
            #     cost['agent_1'].update(obj_cost['agent_1'])
            for agent in obj_cost:
                cost[agent].update(obj_cost[agent])

        if self.contact_other_cost:
            for contact in self.data.contact[: self.data.ncon]:
                geom_ids = [contact.geom1, contact.geom2]
                geom_names = sorted([self.model.geom(g).name for g in geom_ids])
                if any(n in self.agent.body_info[1].geom_names for n in geom_names) and any(
                    n in self.agent.body_info[0].geom_names for n in geom_names
                ):
                    cost['agent_0']['cost_contact_other'] = self.contact_other_cost
                    cost['agent_1']['cost_contact_other'] = self.contact_other_cost

        if self._is_load_static_geoms and self.static_geoms_contact_cost:
            cost['cost_static_geoms_contact'] = 0.0
            for contact in self.data.contact[: self.data.ncon]:
                geom_ids = [contact.geom1, contact.geom2]
                geom_names = sorted([self.model.geom(g).name for g in geom_ids])
                if any(n in self.static_geoms_names for n in geom_names) and any(
                    n in self.agent.body_info.geom_names for n in geom_names
                ):
                    # pylint: disable-next=no-member
                    cost['cost_static_geoms_contact'] += self.static_geoms_contact_cost

        # Sum all costs into single total cost
        # print(f"cost = {cost}")
        for agent_cost in cost.values():
            agent_cost['cost_sum'] = sum(v for k, v in agent_cost.items() if k.startswith('cost_'))
        return cost

    # pylint: disable-next=too-many-branches
    def build_observation_space(self) -> gymnasium.spaces.Dict:
        """Construct observation space.  Happens only once during __init__ in Builder."""
        obs_space_dict = OrderedDict()  # See self.obs()

        sensor_dict = self.agent.build_sensor_observation_space()
        # print(f"DEBUG: sensor_dict = {sensor_dict}")
        # agent0_sensor_dict = {}
        # agent1_sensor_dict = {}

        obs_space_dict.update(sensor_dict)

        for obstacle in self._obstacles:
            if obstacle.is_lidar_observed:
                for i in range(self.agent.agent_num):
                    name = f"{obstacle.name}_lidar_{i}"
                    obs_space_dict[name] = gymnasium.spaces.Box(
                        0.0,
                        1.0,
                        (self.lidar_conf.num_bins,),
                        dtype=np.float64,
                    )
            if (
                hasattr(obstacle, 'is_lidar_ids_observed')
                and obstacle.is_lidar_ids_observed
                and self.lidar_conf.type == 'pseudo_occluded'
            ):
                high = max(int(obstacle.num) - 1, 0)
                for i in range(self.agent.agent_num):
                    name = f"{obstacle.name}_lidar_ids_{i}"
                    obs_space_dict[name] = gymnasium.spaces.Box(
                        -1, high, (self.lidar_conf.num_bins,), dtype=np.int32,
                    )
            if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                for i in range(self.agent.agent_num):
                    name = f"{obstacle.name}_comp_{i}"
                    # obstacle.name + '_' + 'comp'
                    obs_space_dict[name] = gymnasium.spaces.Box(
                        -1.0,
                        1.0,
                        (self.compass_conf.shape,),
                        dtype=np.float64,
                    )

        if self.observe_vision:
            width, height = self.vision_env_conf.vision_size
            rows, cols = height, width
            self.vision_env_conf.vision_size = (rows, cols)
            for i in range(self.agent.agent_num):
                name = f"vision_{i}"
                obs_space_dict[name] = gymnasium.spaces.Box(
                    0,
                    255,
                    (*self.vision_env_conf.vision_size, 3),
                    dtype=np.uint8,
                )

        # # lidar observation of other agents
        # agent0_sensor_dict['other_agents_lidar'] = gymnasium.spaces.Box(
        #     0.0, 1.0, (self.lidar_conf.num_bins,), dtype=np.float64,)
        # agent1_sensor_dict['other_agents_lidar1'] = gymnasium.spaces.Box(
        #     0.0, 1.0, (self.lidar_conf.num_bins,), dtype=np.float64,)

        # for name, space in sensor_dict.items():
        #     if name.endswith('1') and name[-2] != '_':
        #         agent1_sensor_dict[name] = space
        #     else:
        #         agent0_sensor_dict[name] = space
        # obs_space_dict.update(agent0_sensor_dict)

        # for obstacle in self._obstacles:
        #     if obstacle.is_lidar_observed:
        #         name = obstacle.name + '_' + 'lidar'
        #         obs_space_dict[name] = gymnasium.spaces.Box(
        #             0.0,
        #             1.0,
        #             (self.lidar_conf.num_bins,),
        #             dtype=np.float64,
        #         )
        #     if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
        #         name = obstacle.name + '_' + 'comp'
        #         obs_space_dict[name] = gymnasium.spaces.Box(
        #             -1.0,
        #             1.0,
        #             (self.compass_conf.shape,),
        #             dtype=np.float64,
        #         )

        # if self.observe_vision:
        #     width, height = self.vision_env_conf.vision_size
        #     rows, cols = height, width
        #     self.vision_env_conf.vision_size = (rows, cols)
        #     obs_space_dict['vision_0'] = gymnasium.spaces.Box(
        #         0,
        #         255,
        #         (*self.vision_env_conf.vision_size, 3),
        #         dtype=np.uint8,
        #     )

        # obs_space_dict.update(agent1_sensor_dict)
        # for obstacle in self._obstacles:
        #     if obstacle.is_lidar_observed:
        #         name = obstacle.name + '_' + 'lidar1'
        #         obs_space_dict[name] = gymnasium.spaces.Box(
        #             0.0,
        #             1.0,
        #             (self.lidar_conf.num_bins,),
        #             dtype=np.float64,
        #         )
        #     if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
        #         name = obstacle.name + '_' + 'comp1'
        #         obs_space_dict[name] = gymnasium.spaces.Box(
        #             -1.0,
        #             1.0,
        #             (self.compass_conf.shape,),
        #             dtype=np.float64,
        #         )

        # if self.observe_vision:
        #     width, height = self.vision_env_conf.vision_size
        #     rows, cols = height, width
        #     self.vision_env_conf.vision_size = (rows, cols)
        #     obs_space_dict['vision_1'] = gymnasium.spaces.Box(
        #         0,
        #         255,
        #         (*self.vision_env_conf.vision_size, 3),
        #         dtype=np.uint8,
        #     )

        self.obs_info.obs_space_dict = gymnasium.spaces.Dict(obs_space_dict)

        if self.observation_flatten:
            self.observation_space = gymnasium.spaces.utils.flatten_space(
                self.obs_info.obs_space_dict,
            )
        else:
            self.observation_space = self.obs_info.obs_space_dict
        # print(f"DEBUG: Assigned observation_space type: {type(self.observation_space)}, value: {self.observation_space}")

    def _build_placements_dict(self) -> None:
        """Build a dict of placements.

        Happens only once.
        """
        placements = {}

        placements.update(self._placements_dict_from_object('agent'))
        # for i in range(self.agent.agent_num):
        #     placements.update(self._placements_dict_from_object(f'agent_{i}'))
        for obstacle in self._obstacles:
            placements.update(self._placements_dict_from_object(obstacle.name))

        self.placements_conf.placements = placements

    def toggle_observation_space(self) -> None:
        """Toggle observation space."""
        self.observation_flatten = not self.observation_flatten
        self.build_observation_space()

    def _build_world_config(self, layout: dict) -> dict:  # pylint: disable=too-many-branches
        """Create a world_config from our own config."""
        world_config = {
            'floor_type': self.floor_conf.type,
            'floor_size': self.floor_conf.size,
            'agent_base': self.agent.base,
            'agent_xy': layout['agent'],
        }
        if self.agent.rot is None:
            world_config['agent_rot'] = self.random_generator.generate_rots(self.agent.agent_num)
        else:
            world_config['agent_rot'] = float(self.agent.rot)

        # process world config via different objects.
        world_config.update(
            {
                'geoms': {},
                'free_geoms': {},
                'mocaps': {},
            },
        )
        for obstacle in self._obstacles:
            # print(f"obstacle: {obstacle}")
            num = obstacle.num if hasattr(obstacle, 'num') else 1
            if obstacle.name == 'agent':
                num = self.agent.agent_num
            obstacle.process_config(world_config, layout, self.random_generator.generate_rots(num))
        if self._is_load_static_geoms:
            self._build_static_geoms_config(world_config['geoms'])

        return world_config

    def _build_static_geoms_config(self, geoms_config: dict) -> None:
        """Load static geoms from .yaml file.

        Static geoms are geoms which won't be considered when calculate reward and cost in general.
        And have no randomness.
        Some tasks may generate cost when contacting static geoms.
        """
        env_info = self.__class__.__name__.split('Level')
        config_name = env_info[0].lower()
        level = int(env_info[1])

        # load all config of meshes in specific environment from .yaml file
        base_dir = os.path.dirname(safety_gymnasium.__file__)
        with open(os.path.join(base_dir, f'configs/{config_name}.yaml'), encoding='utf-8') as file:
            meshes_config = yaml.load(file, Loader=yaml.FullLoader)  # noqa: S506

        self.static_geoms_names = set()
        for idx in range(level + 1):
            for group in meshes_config[idx].values():
                geoms_config.update(group)
                for item in group.values():
                    # if 'geoms' in item:
                    #     for geom in item['geoms']:
                    #         self.static_geoms_names.add(geom['name'])
                    # else:
                    #     self.static_geoms_names.add(item['name'])
                    self.static_geoms_names.add(item['name'])

    def build_goal_position(self) -> None:
        """Build a new goal position, maybe with resampling due to hazards."""
        # Resample until goal is compatible with layout
        if 'goal' in self.world_info.layout:
            del self.world_info.layout['goal']
        for _ in range(10000):  # Retries
            if self.random_generator.sample_goal_position():
                break
        else:
            raise ResamplingError('Failed to generate goal')
        # Move goal geom to new layout position
        if self.goal_achieved[0]:
            self.world_info.world_config_dict['geoms']['goal_red']['pos'][:2] = (
                self.world_info.layout['goal_red']
            )
            self._set_goal('goal_red', self.world_info.layout['goal_red'])
        if self.goal_achieved[1]:
            self.world_info.world_config_dict['geoms']['goal_blue']['pos'][:2] = (
                self.world_info.layout['goal_blue']
            )
            self._set_goal('goal_blue', self.world_info.layout['goal_blue'])
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def _placements_dict_from_object(self, object_name: dict) -> dict:
        """Get the placements dict subset just for a given object name."""
        placements_dict = {}
        attrs = dir(self)
        # print(f"DEBUG: attrs = {attrs}")
        assert hasattr(self, object_name), f'object {object_name} does not exist, but you use it!'
        data_obj = getattr(self, object_name)

        if hasattr(data_obj, 'num'):  # Objects with multiplicity
            object_fmt = object_name[:-1] + '{i}'
            object_num = getattr(data_obj, 'num', None)
            object_locations = getattr(data_obj, 'locations', [])
            object_placements = getattr(data_obj, 'placements', None)
            object_keepout = data_obj.keepout
        else:  # Unique objects
            object_fmt = object_name
            object_num = 1
            object_locations = getattr(data_obj, 'locations', [])
            object_placements = getattr(data_obj, 'placements', None)
            object_keepout = data_obj.keepout
        for i in range(object_num):
            if i < len(object_locations):
                x, y = object_locations[i]  # pylint: disable=invalid-name
                k = object_keepout + 1e-9  # Epsilon to account for numerical issues
                placements = [(x - k, y - k, x + k, y + k)]
            else:
                placements = object_placements
            placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
        # print(f"DEBUG: placements_dict = {placements_dict}")
        return placements_dict

    def obs(self) -> dict | np.ndarray:
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor's data correct
        obs = {}

        obs.update(self.agent.obs_sensor())

        # observation of other agents
        # this does not work
        # # TODO: harded-coded for now, should iterate over all other agents
        # obs['other_agents_lidar'] = self._obs_lidar([self.agent.pos_1], 0)
        # obs['other_agents_lidar1'] = self._obs_lidar([self.agent.pos_0], 0)

        # observations of obstacles
        for obstacle in self._obstacles:
            # print(f"obstacle.name: {obstacle.name}, obstacle.pos: {obstacle.pos}, obstacle.group: {obstacle.group}")
            if obstacle.is_lidar_observed:
                # this is used to represent other agents
                # print(f"obstacle.name: {obstacle.name}")
                if 'gremlins' in obstacle.name:
                    for i in range(self.agent.agent_num):
                        name = f"{obstacle.name}_lidar_{i}"
                        # print(f"DEBUG: obstacle.pos before processing: {obstacle.pos}")
                        poses = obstacle.pos.copy() # need to exclude self agent
                        del poses[i]
                        obs[name] = self._obs_lidar_new(i, poses, obstacle.group)
                # if False:
                    # print(f"lidar: {obstacle.pos}")
                    # obs[obstacle.name + '_lidar'] = self._obs_lidar(obstacle.pos[1], obstacle.group)
                    # # print(f"agent0: {obs[obstacle.name + '_lidar']}")
                    # obs[obstacle.name + '_lidar1'] = self._obs_lidar1(obstacle.pos[0], obstacle.group)
                    # print(f"agent1: {obs[obstacle.name + '_lidar1']}")
                else:
                    for i in range(self.agent.agent_num):
                        name = f"{obstacle.name}_lidar_{i}"
                        obs[name] = self._obs_lidar_new(i, obstacle.pos, obstacle.group)
                    # obs[obstacle.name + '_lidar'] = self._obs_lidar(obstacle.pos, obstacle.group)
                    # obs[obstacle.name + '_lidar1'] = self._obs_lidar1(obstacle.pos, obstacle.group)
                    # if 'gremlins' in obstacle.name:
                    #     print(f"agent0: {obs[obstacle.name + '_lidar']}")
                    #     print(f"agent1: {obs[obstacle.name + '_lidar1']}")
                # print(f"obs[{obstacle.name} + '_lidar']: {obs[obstacle.name + '_lidar']}")
                # TODO: mocap is added to represent other agents
    
    def process_obs(self, obs: dict) -> dict:
        processed_obs = {}
        for k, v in obs.items():
            m = re.search(r'(\d+)$', k)
            # print(k, m)
            agent_idx = int(m.group(1)) if m else 0
            if f"agent_{agent_idx}" not in processed_obs:
                processed_obs[f"agent_{agent_idx}"] = {}
            processed_obs[f"agent_{agent_idx}"][k] = v
        return processed_obs

    def _obstacle_instance_names(self, obstacle) -> list[str]:
        """MuJoCo body names for each lidar-observable instance in an obstacle."""
        if obstacle.type == 'mocaps':
            return [f'{obstacle.name[:-1]}{i}obj' for i in range(obstacle.num)]
        if hasattr(obstacle, 'num'):
            return [f'{obstacle.name[:-1]}{i}' for i in range(obstacle.num)]
        return [obstacle.name]

    def _obstacle_geom_id_map(self, obstacle) -> dict[int, int]:
        """Map MuJoCo geom id to instance index within an obstacle category."""
        if not hasattr(self, '_obstacle_geom_id_cache'):
            self._obstacle_geom_id_cache = {}
        cache_key = obstacle.name
        if cache_key not in self._obstacle_geom_id_cache:
            id_map: dict[int, int] = {}
            for row, name in enumerate(self._obstacle_instance_names(obstacle)):
                try:
                    geom_id = self.model.geom(name).id
                    id_map[geom_id] = row
                except KeyError:
                    body_id = self.model.body(name).id
                    for geom_id in range(self.model.ngeom):
                        if self.model.geom_bodyid[geom_id] == body_id:
                            id_map[geom_id] = row
            self._obstacle_geom_id_cache[cache_key] = id_map
        return self._obstacle_geom_id_cache[cache_key]

    def _obstacle_geom_ids(self, obstacle) -> set[int]:
        """Geom ids belonging to one lidar obstacle category (all instances)."""
        return set(self._obstacle_geom_id_map(obstacle).keys())

    def _obstacle_geom_id_for_instance(self, obstacle, instance_row: int) -> int | None:
        """Return one geom id for an instance index within an obstacle category."""
        for geom_id, row in self._obstacle_geom_id_map(obstacle).items():
            if row == instance_row:
                return geom_id
        return None

    def _lidar_occluder_group_mask(self) -> np.ndarray:
        """MuJoCo group mask: all groups used by lidar-observed obstacles."""
        if not hasattr(self, '_lidar_occluder_grp_cache'):
            groups = {
                int(obstacle.group)
                for obstacle in self._obstacles
                if obstacle.is_lidar_observed
            }
            # print(f"DEBUG: obstacles = {[obstacle.name for obstacle in self._obstacles if obstacle.is_lidar_observed and obstacle.exterior]}")
            # print(f"DEBUG: groups = {groups}")
            # pylint: disable-next=no-member
            ngroup = int(mujoco.mjNGROUP)
            self._lidar_occluder_grp_cache = np.asarray(
                [i in groups for i in range(ngroup)],
                dtype='uint8',
            )
        return self._lidar_occluder_grp_cache

    def _lidar_observable_geom_ids(self) -> set[int]:
        """Geom ids for instances on obstacles with is_lidar_observed=True."""
        if not hasattr(self, '_lidar_observable_geom_id_cache'):
            geom_ids: set[int] = set()
            for obstacle in self._obstacles:
                if obstacle.is_lidar_observed:
                    geom_ids.update(self._obstacle_geom_ids(obstacle))
            self._lidar_observable_geom_id_cache = geom_ids
        return self._lidar_observable_geom_id_cache

    def _get_lidar_suppressed_geom_ids(self) -> set[int]:
        """Geom ids skipped as occluders and hidden in render (task may update each step)."""
        return getattr(self, '_lidar_suppressed_geom_ids', set())

    def _lidar_ray_first_observable_geom(
        self,
        agent_idx: int,
        origin: np.ndarray,
        direction: np.ndarray,
        max_dist: float,
    ) -> tuple[int | None, float]:
        """First lidar-observable geom surface along a ray, skipping other geoms."""
        observable = self._lidar_observable_geom_ids()
        pos = np.asarray(origin, dtype='float64')
        vec = np.asarray(direction, dtype='float64')
        body = self.model.body(f'agent_{agent_idx}').id
        total_dist = 0.0
        for _ in range(64):
            if total_dist > max_dist:
                return None, -1.0
            geom_id = np.array([0], dtype='int32')
            dist = mujoco.mj_ray(  # pylint: disable=no-member
                self.model,
                self.data,
                pos,
                vec,
                self._lidar_occluder_group_mask(),
                1,
                body,
                geom_id,
            )
            if dist < 0:
                return None, -1.0
            hit_geom = int(geom_id[0])
            # print(f"DEBUG: geom_id = {geom_id}")
            hit_dist = total_dist + dist
            if hit_geom in self._get_lidar_suppressed_geom_ids():
                advance = dist + 1e-5
                pos = pos + vec * advance
                total_dist += advance
                continue
            if hit_geom in observable:
                return hit_geom, hit_dist
            advance = dist + 1e-5
            pos = pos + vec * advance
            total_dist += advance
        return None, -1.0

    def _lidar_line_of_sight(
        self,
        agent_idx: int,
        target_pos: np.ndarray,
        obstacle,
        instance_row: int,
    ) -> bool:
        """True when the first geom surface along agent→target belongs to this instance."""
        agent_pos = np.asarray(self.agent.get_agent_pos(agent_idx), dtype='float64')
        target = np.asarray(target_pos, dtype='float64')
        if target.shape == (2,):
            target = np.concatenate([target, [0.0]])
        vec = target - agent_pos
        dist_target = float(np.linalg.norm(vec))
        if dist_target < 1e-9:
            return True
        vec /= dist_target
        hit_geom, _hit_dist = self._lidar_ray_first_observable_geom(
            agent_idx,
            agent_pos,
            vec,
            dist_target + 1e-4,
        )
        if hit_geom is None:
            return False
        geom_map = self._obstacle_geom_id_map(obstacle)
        # print(f"DEBUG: hit_geom.name = {hit_geom.name}")
        return hit_geom in geom_map and geom_map[hit_geom] == instance_row

    def _accumulate_pseudo_lidar_reading(
        self,
        vals: np.ndarray,
        agent_idx: int,
        pos: np.ndarray,
        ids: np.ndarray | None = None,
        instance_id: int | None = None,
    ) -> None:
        pos = np.asarray(pos)
        if pos.shape == (3,):
            pos = pos[:2]
        z = complex(*self._ego_xy_new(agent_idx, pos))
        dist = np.abs(z)
        angle = np.angle(z) % (np.pi * 2)
        bin_size = (np.pi * 2) / self.lidar_conf.num_bins
        bin = int(angle / bin_size)
        bin_angle = bin_size * bin
        if self.lidar_conf.max_dist is None:
            sensor = np.exp(-self.lidar_conf.exp_gain * dist)
        else:
            sensor = max(0, self.lidar_conf.max_dist - dist) / self.lidar_conf.max_dist

        def _write(b: int, value: float) -> None:
            if value > vals[b]:
                vals[b] = value
                if ids is not None and instance_id is not None:
                    ids[b] = instance_id

        _write(bin, sensor)
        if self.lidar_conf.alias:
            alias = (angle - bin_angle) / bin_size
            _write((bin + 1) % self.lidar_conf.num_bins, alias * sensor)
            _write((bin - 1) % self.lidar_conf.num_bins, (1 - alias) * sensor)

    def _natural_lidar_sensor(self, dist: float) -> float:
        """Distance-to-sensor encoding shared by natural lidar variants."""
        if dist < 0:
            return 0.0
        return float(np.exp(-dist))

    def _obs_lidar_new(
        self,
        agent_idx: int,
        positions: np.ndarray | list,
        group: int,
        obstacle=None,
    ) -> np.ndarray:
        """Calculate and return a lidar observation.

        See sub methods for implementation.
        """
        if self.lidar_conf.type == 'pseudo':
            return self._obs_lidar_pseudo_new(agent_idx, positions)

        if self.lidar_conf.type == 'pseudo_occluded':
            if obstacle is None:
                raise ValueError('pseudo_occluded lidar requires obstacle')
            return self._obs_lidar_pseudo_occluded_new(agent_idx, obstacle)

        if self.lidar_conf.type == 'natural':
            return self._obs_lidar_natural_new(agent_idx, group)

        raise ValueError(f'Invalid lidar_type {self.lidar_conf.type}')

    def _obs_lidar(self, positions: np.ndarray | list, group: int) -> np.ndarray:
        """Calculate and return a lidar observation.

        See sub methods for implementation.
        """
        # print(f"self.lidar_conf.type: {self.lidar_conf.type}")
        if self.lidar_conf.type == 'pseudo':
            return self._obs_lidar_pseudo(positions)

        if self.lidar_conf.type == 'natural':
            return self._obs_lidar_natural(group)

        raise ValueError(f'Invalid lidar_type {self.lidar_conf.type}')

    def _obs_lidar1(self, positions: np.ndarray | list, group: int) -> np.ndarray:
        """Calculate and return a lidar observation.

        See sub methods for implementation.
        """
        if self.lidar_conf.type == 'pseudo':
            return self._obs_lidar_pseudo1(positions)

        if self.lidar_conf.type == 'natural':
            return self._obs_lidar_natural(group)

        raise ValueError(f'Invalid lidar_type {self.lidar_conf.type}')

    def _obs_lidar_natural_new(self, agent_idx: int, group: int) -> np.ndarray:
        """Natural lidar casts rays based on the ego-frame of the agent.

        Rays are circularly projected from the agent body origin around the agent z axis.
        """
        body = self.model.body(f'agent_{agent_idx}').id
        # pylint: disable-next=no-member
        grp = np.asarray([i == group for i in range(int(mujoco.mjNGROUP))], dtype='uint8')
        pos = np.asarray(self.agent.get_agent_pos(agent_idx), dtype='float64')
        mat_t = self.agent.get_agent_mat(agent_idx)
        obs = np.zeros(self.lidar_conf.num_bins)
        for i in range(self.lidar_conf.num_bins):
            theta = (i / self.lidar_conf.num_bins) * np.pi * 2
            vec = np.matmul(mat_t, theta2vec(theta))  # Rotate from ego to world frame
            vec = np.asarray(vec, dtype='float64')
            geom_id = np.array([0], dtype='int32')
            dist = mujoco.mj_ray(  # pylint: disable=no-member
                self.model,
                self.data,
                pos,
                vec,
                grp,
                1,
                body,
                geom_id,
            )
            if dist >= 0:
                obs[i] = self._natural_lidar_sensor(dist)
        return obs

    def _obs_lidar_pseudo_occluded_new(
        self,
        agent_idx: int,
        obstacle,
        return_ids: bool = False,
        skip_instance_rows: frozenset[int] | None = None,
    ):
        """Pseudo lidar with alias, gated by geom-surface line of sight per instance."""
        vals = np.zeros(self.lidar_conf.num_bins)
        ids = (
            np.full(self.lidar_conf.num_bins, -1, dtype=np.int32)
            if return_ids else None
        )
        skip_rows = skip_instance_rows or frozenset()
        skip_self = 'gremlins' in obstacle.name
        for row in range(obstacle.num):
            if row in skip_rows:
                continue
            if skip_self and row == agent_idx:
                continue
            if self._obstacle_geom_id_for_instance(obstacle, row) is None:
                continue
            pos = obstacle.pos[row]
            if not self._lidar_line_of_sight(agent_idx, pos, obstacle, row):
                continue
            self._accumulate_pseudo_lidar_reading(
                vals, agent_idx, pos, ids=ids, instance_id=row,
            )
        if return_ids:
            return vals, ids
        return vals

    def _obs_lidar_natural(self, group: int) -> np.ndarray:
        """Natural lidar casts rays based on the ego-frame of the agent.

        Rays are circularly projected from the agent body origin around the agent z axis.
        """
        body = self.model.body('agent').id
        # pylint: disable-next=no-member
        grp = np.asarray([i == group for i in range(int(mujoco.mjNGROUP))], dtype='uint8')
        pos = np.asarray(self.agent.pos, dtype='float64')
        mat_t = self.agent.mat
        obs = np.zeros(self.lidar_conf.num_bins)
        for i in range(self.lidar_conf.num_bins):
            theta = (i / self.lidar_conf.num_bins) * np.pi * 2
            vec = np.matmul(mat_t, theta2vec(theta))  # Rotate from ego to world frame
            vec = np.asarray(vec, dtype='float64')
            geom_id = np.array([0], dtype='int32')
            dist = mujoco.mj_ray(  # pylint: disable=no-member
                self.model,
                self.data,
                pos,
                vec,
                grp,
                1,
                body,
                geom_id,
            )
            if dist >= 0:
                obs[i] = self._natural_lidar_sensor(dist)
        return obs

    def _obs_lidar_pseudo_new(self, agent_idx: int, positions: np.ndarray) -> np.ndarray:
        obs = np.zeros(self.lidar_conf.num_bins)
        # Empty list → np.array([], ndmin=2) has shape (1, 0); iterating yields Bad pos [].
        if positions is None or len(positions) == 0:
            return obs
        positions = np.array(positions, ndmin=2)
        if positions.size == 0:
            return obs
        for pos in positions:
            self._accumulate_pseudo_lidar_reading(obs, agent_idx, pos)
        return obs

    def _obs_lidar_pseudo(self, positions: np.ndarray) -> np.ndarray:
        """Return an agent-centric lidar observation of a list of positions.

        Lidar is a set of bins around the agent (divided evenly in a circle).
        The detection directions are exclusive and exhaustive for a full 360 view.
        Each bin reads 0 if there are no objects in that direction.
        If there are multiple objects, the distance to the closest one is used.
        Otherwise the bin reads the fraction of the distance towards the agent.

        E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
        and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
        (The reading can be thought of as "closeness" or inverse distance)

        This encoding has some desirable properties:
            - bins read 0 when empty
            - bins smoothly increase as objects get close
            - maximum reading is 1.0 (where the object overlaps the agent)
            - close objects occlude far objects
            - constant size observation with variable numbers of objects
        """
        positions = np.array(positions, ndmin=2)
        obs = np.zeros(self.lidar_conf.num_bins)
        for pos in positions:
            # print(f"object pos: {pos}")
            pos = np.asarray(pos)
            if pos.shape == (3,):
                pos = pos[:2]  # Truncate Z coordinate
            # pylint: disable-next=invalid-name
            z = complex(*self._ego_xy(pos))  # X, Y as real, imaginary components
            dist = np.abs(z)
            angle = np.angle(z) % (np.pi * 2)
            bin_size = (np.pi * 2) / self.lidar_conf.num_bins
            bin = int(angle / bin_size)  # pylint: disable=redefined-builtin
            bin_angle = bin_size * bin
            if self.lidar_conf.max_dist is None:
                sensor = np.exp(-self.lidar_conf.exp_gain * dist)
            else:
                sensor = max(0, self.lidar_conf.max_dist - dist) / self.lidar_conf.max_dist
            obs[bin] = max(obs[bin], sensor)
            # Aliasing
            if self.lidar_conf.alias:
                alias = (angle - bin_angle) / bin_size
                assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin}'
                bin_plus = (bin + 1) % self.lidar_conf.num_bins
                bin_minus = (bin - 1) % self.lidar_conf.num_bins
                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
        return obs

    def _obs_lidar_pseudo1(self, positions: np.ndarray) -> np.ndarray:
        """Return an agent-centric lidar observation of a list of positions.

        Lidar is a set of bins around the agent (divided evenly in a circle).
        The detection directions are exclusive and exhaustive for a full 360 view.
        Each bin reads 0 if there are no objects in that direction.
        If there are multiple objects, the distance to the closest one is used.
        Otherwise the bin reads the fraction of the distance towards the agent.

        E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
        and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
        (The reading can be thought of as "closeness" or inverse distance)

        This encoding has some desirable properties:
            - bins read 0 when empty
            - bins smoothly increase as objects get close
            - maximum reading is 1.0 (where the object overlaps the agent)
            - close objects occlude far objects
            - constant size observation with variable numbers of objects
        """
        positions = np.array(positions, ndmin=2)
        obs = np.zeros(self.lidar_conf.num_bins)
        for pos in positions:
            pos = np.asarray(pos)
            if pos.shape == (3,):
                pos = pos[:2]  # Truncate Z coordinate
            # pylint: disable-next=invalid-name
            z = complex(*self._ego_xy1(pos))  # X, Y as real, imaginary components
            dist = np.abs(z)
            angle = np.angle(z) % (np.pi * 2)
            bin_size = (np.pi * 2) / self.lidar_conf.num_bins
            bin = int(angle / bin_size)  # pylint: disable=redefined-builtin
            bin_angle = bin_size * bin
            if self.lidar_conf.max_dist is None:
                sensor = np.exp(-self.lidar_conf.exp_gain * dist)
            else:
                sensor = max(0, self.lidar_conf.max_dist - dist) / self.lidar_conf.max_dist
            obs[bin] = max(obs[bin], sensor)
            # Aliasing
            if self.lidar_conf.alias:
                alias = (angle - bin_angle) / bin_size
                assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin}'
                bin_plus = (bin + 1) % self.lidar_conf.num_bins
                bin_minus = (bin - 1) % self.lidar_conf.num_bins
                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
        return obs

    def _obs_compass(self, pos: np.ndarray) -> np.ndarray:
        """Return an agent-centric compass observation of a list of positions.

        Compass is a normalized (unit-length) egocentric XY vector,
        from the agent to the object.

        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        # Get ego vector in world frame
        vec = pos - self.agent.pos
        # Rotate into frame
        vec = np.matmul(vec, self.agent.mat)
        # Truncate
        vec = vec[: self.compass_conf.shape]
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        assert vec.shape == (self.compass_conf.shape,), f'Bad vec {vec}'
        return vec

    def _obs_compass_new(self, agent_idx: int, pos: np.ndarray) -> np.ndarray:
        """Egocentric unit compass vector from agent_idx to pos (XY)."""
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.concatenate([pos, [0.0]])
        agent_3vec = self.agent.get_agent_pos(agent_idx)
        agent_mat = self.agent.get_agent_mat(agent_idx)
        vec = pos - agent_3vec
        vec = np.matmul(vec, agent_mat)[: self.compass_conf.shape]
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec

    def _obs_vision(self, camera_name='vision') -> np.ndarray:
        """Return pixels from the agent camera.

        Note:
            This is a 3D array of shape (rows, cols, channels).
            The channels are RGB, in that order.
            If you are on a headless machine, you may need to checkout this:
            URL: `issue <https://github.com/PKU-Alignment/safety-gymnasium/issues/27>`_
        """
        rows, cols = self.vision_env_conf.vision_size
        width, height = cols, rows
        return self.render(
            width,
            height,
            mode='rgb_array',
            camera_name=camera_name,
            cost={'agent_0': {}, 'agent_1': {}},
        )

    def _ego_xy_new(self, agent_idx: int, pos: np.ndarray) -> np.ndarray:
        """Return the egocentric XY vector to a position from the agent."""
        assert pos.shape == (2,), f'Bad pos {pos}'
        agent_3vec = self.agent.get_agent_pos(agent_idx)
        agent_mat = self.agent.get_agent_mat(agent_idx)
        pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        world_3vec = pos_3vec - agent_3vec
        return np.matmul(world_3vec, agent_mat)[:2]  # only take XY coordinates

    def _ego_xy(self, pos: np.ndarray) -> np.ndarray:
        """Return the egocentric XY vector to a position from the agent."""
        assert pos.shape == (2,), f'Bad pos {pos}'
        agent_3vec = self.agent.pos_0
        agent_mat = self.agent.mat_0
        pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        world_3vec = pos_3vec - agent_3vec
        return np.matmul(world_3vec, agent_mat)[:2]  # only take XY coordinates

    def _ego_xy1(self, pos: np.ndarray) -> np.ndarray:
        """Return the egocentric XY vector to a position from the agent."""
        assert pos.shape == (2,), f'Bad pos {pos}'
        agent_3vec = self.agent.pos_1
        agent_mat = self.agent.mat_1
        pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        world_3vec = pos_3vec - agent_3vec
        return np.matmul(world_3vec, agent_mat)[:2]  # only take XY coordinates

    @abc.abstractmethod
    def calculate_reward(self) -> float:
        """Determine reward depending on the agent and tasks."""

    @abc.abstractmethod
    def specific_reset(self) -> None:
        """Set positions and orientations of agent and obstacles."""

    @abc.abstractmethod
    def specific_step(self) -> None:
        """Each task can define a specific step function.

        It will be called when :meth:`safety_gymnasium.builder.Builder.step()` is called using env.step().
        For example, you can do specific data modification.
        """

    @abc.abstractmethod
    def update_world(self) -> None:
        """Update one task specific goal."""

    @property
    @abc.abstractmethod
    def goal_achieved(self) -> bool:
        """Check if task specific goal is achieved."""
