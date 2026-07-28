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
"""Multi Goal with a SAR environment."""

from collections import OrderedDict

import gymnasium
import mujoco
import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask
from safety_gymnasium.tasks.safe_multi_agent.world import World
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import LtlWalls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.buildings import Buildings
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.casualtys import Casualtys
from safety_gymnasium.tasks.safe_multi_agent.assets.mocaps.gremlins import Gremlins
from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal_sar.sar_config_loader import (
    apply_sar_recipe,
)
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
    agent_inside_building_idx,
    border_placements,
    building_count,
    building_geom,
    clear_building_pinned_locations,
    clamp_building_placement_keepout,
    mission_goal_achieved,
    sync_building_dependents_into_layout,
)


class MultiGoalSARLevel0(BaseTask):
    """Multi-agent zone navigation with optional ring-placed interior walls."""

    # Level identity (stock L0). Shared recipe lives in configs/multi_goal_sar.yaml.
    wall_count = 0
    reward_goal = 1.0
    surface_casualties_enabled: bool = True
    entrapped_casualties_enabled: bool = False
    building_num: int = 0

    # Placeholders so BaseTask._parse can accept CustomizedSAR overrides.
    wall_ring_radius = 2.0
    wall_margin = 1.0
    wall_base_half_sizes = [0.1, 0.3, 0.2]
    walls_keepout = 0.4
    walls_half_size_randomization = True
    building_keepout = 0.4
    building_border_side_length = 4.5
    building_margin = 0.8
    casualty_size = 0.05
    casualty_touch_offset = 0.15
    casualty_keepout = 0.2
    entrapped_casualty_keepout = 0.0
    agent_keepout = 0.25
    agent_placements = [(-0.67, -0.67, 0.67, 0.67)]
    gremlin_size = 0.175
    gremlin_dist_threshold = 0.175
    gremlin_keepout = 0.0
    building_perimeter_wall_height = 0.75
    building_perimeter_wall_collision_threshold = 8.0

    def __init__(self, config) -> None:
        self._cached_wall_half_sizes = None
        self._cached_building_rots = None
        config = dict(config)
        skip_keys = frozenset(config.pop('_sar_skip_constant_keys', ()))
        super().__init__(config=config)

        apply_sar_recipe(self, skip_keys=skip_keys)
        # Stock levels keep fixed lidar resolution (customized_defaults is CustomizedSAR-only).
        if 'lidar_conf.num_bins' not in skip_keys and 'num_bins' not in skip_keys:
            self.lidar_conf.num_bins = 16

        self.last_dist_casualty = None
        self._buildings_entered: set[int] = set()
        self._lidar_suppressed_geom_ids: set[int] = set()

        # Spawn agents in a specified area
        self._build_agent(
            self.agent_name,
            keepout=self.agent_keepout,
            placements=self.agent_placements,
        )
        surface_casualtys_int = int(self.agent_num * self.surface_casualties_enabled)
        # One surface casualty for solo training; otherwise one per agent.
        self.casualty_num = self.agent_num
        self._add_geoms(
            LtlWalls(contype=1),
        )

        if surface_casualtys_int > 0:
            self._add_geoms(
                Casualtys(
                    category="surface",
                    size=self.casualty_size,
                    num=surface_casualtys_int,
                    keepout=self.casualty_keepout,
                ),
            )

        # Gremlin count is always agent_num (not user-configurable).
        self._add_mocaps(
            Gremlins(
                num=self.agent_num,
                size=self.gremlin_size,
                dist_threshold=self.gremlin_dist_threshold,
                keepout=self.gremlin_keepout,
            )
        )

    def _dist_to_casualty(self, agent_idx: int) -> float:
        if not hasattr(self, 'surface_casualtys'):
            return 0.0
        casualty_pos = self.surface_casualtys.pos[0]
        return self.agent.dist_xy(agent_idx, casualty_pos)

    def _dist_to_casualtys(self, agent_idx: int) -> list[float]:
        if hasattr(self, 'surface_casualtys'):
            casualty_poses = (self.surface_casualtys.pos[i] for i in range(self.casualty_num))
            return [self.agent.dist_xy(agent_idx, pos) for pos in casualty_poses]
        elif hasattr(self, 'entrapped_casualtys'):
            casualty_poses = (self.entrapped_casualtys.pos[i] for i in range(self.casualty_num))
            return [self.agent.dist_xy(agent_idx, pos) for pos in casualty_poses]
        return []

    def _casualtys_rescued(self) -> list[float]:
            if hasattr(self, 'surface_casualtys'):
                return self.surface_casualtys.rescued
            elif hasattr(self, 'entrapped_casualtys'):
                return self.entrapped_casualtys.rescued
            return []            

    def build_observation_space(self) -> gymnasium.spaces.Dict:
        super().build_observation_space()
        buildings = building_geom(self)
        if buildings is not None:
            obs_space_dict = OrderedDict(self.obs_info.obs_space_dict.spaces)
            obs_space_dict[f'{buildings.color_name}_buildings_visited'] = gymnasium.spaces.Box(
                0.0,
                1.0,
                (buildings.num,),
                dtype=np.float64,
            )
            self.obs_info.obs_space_dict = gymnasium.spaces.Dict(obs_space_dict)
        if self.observation_flatten:
            self.observation_space = gymnasium.spaces.utils.flatten_space(
                self.obs_info.obs_space_dict,
            )
        else:
            self.observation_space = self.obs_info.obs_space_dict
        return self.observation_space

    def calculate_reward(self):
        """Distance delta toward visible casualty and touch bonus."""
        rewards = {}
        touch_threshold = 0.0
        if hasattr(self, 'surface_casualtys'):
            touch_threshold = self.surface_casualtys.size + self.casualty_touch_offset
        if hasattr(self, 'entrapped_casualtys'):
            touch_threshold = self.entrapped_casualtys.size + self.casualty_touch_offset

        for i in range(self.agent_num):
            a = f'agent_{i}'
            reward = 0

            # Distance-based reward shaping
            dists = self._dist_to_casualtys(i)
            if not dists:
                rewards[a] = 0
                self.last_dist_casualty[i] = 0.0
                continue
            min_dist = min(dists)
            min_casualty_rescued = self._casualtys_rescued()[dists.index(min_dist)]
            if min_dist <= touch_threshold and not min_casualty_rescued:
                reward += self.reward_goal
            self.last_dist_casualty[i] = min_dist

            rewards[a] = reward
        return rewards

    def specific_reset(self):
        """Reset SAR-specific episode state after layout resample."""
        if hasattr(self, 'surface_casualtys'):
            self.surface_casualtys.rescued = [False] * self.surface_casualtys.num
        if hasattr(self, 'entrapped_casualtys'):
            self.entrapped_casualtys.rescued = [False] * self.entrapped_casualtys.num
        self.last_dist_casualty = [self._dist_to_casualty(i) for i in range(self.agent_num)]
        self._buildings_entered = set()
        self._lidar_suppressed_geom_ids = set()
        buildings = building_geom(self)
        if buildings is not None:
            buildings.prev_contact = [False] * buildings.num
        self._sync_entered_building_state()

    def specific_step(self):
        self._sync_entered_building_state()
        self._sync_rescued_casualty_state()

    def _casualty_geoms(self):
        geoms = []
        if hasattr(self, 'surface_casualtys'):
            geoms.append(self.surface_casualtys)
        if hasattr(self, 'entrapped_casualtys'):
            geoms.append(self.entrapped_casualtys)
        return geoms

    def _sync_rescued_casualty_state(self) -> None:
        """Hide rescued casualties: alpha=0 + lidar suppress (mirror buildings)."""
        if not hasattr(self, 'model') or self.model is None:
            return
        suppressed = set(getattr(self, '_lidar_suppressed_geom_ids', set()))
        for geom in self._casualty_geoms():
            rescued = getattr(geom, 'rescued', None)
            if rescued is None:
                continue
            for row, is_rescued in enumerate(rescued):
                geom_id = self._obstacle_geom_id_for_instance(geom, row)
                if geom_id is None:
                    continue
                if is_rescued:
                    suppressed.add(geom_id)
                    self.model.geom_rgba[geom_id][-1] = 0.0
                else:
                    suppressed.discard(geom_id)
                    self.model.geom_rgba[geom_id][-1] = float(geom.alpha)
        self._lidar_suppressed_geom_ids = suppressed

    def _rescued_casualty_rows(self, obstacle) -> frozenset[int]:
        rescued = getattr(obstacle, 'rescued', None)
        if rescued is None:
            return frozenset()
        return frozenset(i for i, flag in enumerate(rescued) if flag)

    def _sync_entered_building_state(self) -> None:
        """Sticky-hide entered building shells for the rest of the episode."""
        buildings = building_geom(self)
        if buildings is None or not hasattr(self, 'model') or self.model is None:
            # Still refresh casualty hide when no buildings
            self._sync_rescued_casualty_state()
            return

        for agent_idx in range(self.agent_num):
            inside_idx = agent_inside_building_idx(self, agent_idx)
            if inside_idx is not None:
                self._buildings_entered.add(inside_idx)

        suppressed: set[int] = set()
        for row in self._buildings_entered:
            geom_id = self._obstacle_geom_id_for_instance(buildings, row)
            if geom_id is not None:
                suppressed.add(geom_id)

        self._lidar_suppressed_geom_ids = suppressed

        for row in range(buildings.num):
            geom_id = self._obstacle_geom_id_for_instance(buildings, row)
            if geom_id is None:
                continue
            if row in self._buildings_entered:
                self.model.geom_rgba[geom_id][-1] = 0.0
            else:
                self.model.geom_rgba[geom_id][-1] = buildings.alpha

        # Merge rescued casualty suppression after buildings
        self._sync_rescued_casualty_state()

    def update_world(self):
        pass

    def _prepare_layout(self) -> None:
        has_buildings = building_geom(self) is not None
        if has_buildings:
            clear_building_pinned_locations(self)
            clamp_building_placement_keepout(self, self.building_margin)
        if has_buildings or self.placements_conf.placements is None:
            self._build_placements_dict()
            self.random_generator.set_placements_info(
                self.placements_conf.placements,
                self.placements_conf.extents,
                self.placements_conf.margin,
            )
        if self.random_generator.agent_num is None:
            self.random_generator.agent_num = self.agent.agent_num
        self.world_info.layout = self.random_generator.build_layout()
        if has_buildings:
            sync_building_dependents_into_layout(self, self.world_info.layout)

    def _fast_resample_layout(self) -> None:
        self._prepare_layout()
        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)
        self._apply_layout_from_config()

    def _build(self):
        self._prepare_layout()
        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)
        if self.world is None:
            self.world = World(self.agent, self._obstacles, self.world_info.world_config_dict)
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_info.world_config_dict, state=False)
            if self.viewer:
                self._update_viewer(self.model, self.data)

    def _replace_geom(self, geom) -> None:
        """Update _geoms like _add_geoms but without duplicate registration checks."""
        self._geoms[geom.name] = geom
        setattr(self, geom.name, geom)
        geom.set_agent(self.agent)

    def _replace_border_buildings(self, num=None) -> None:
        self._replace_geom(Buildings(
            color=list(Buildings.COLORS)[0],
            size=self.building_keepout * 0.75,
            num=self.agent_num if num is None else num,
            keepout=self.building_keepout,
            placements=border_placements(
                self.building_border_side_length,
                self.building_margin,
            ),
        ))

    def _replace_building_perimeter_walls(self) -> None:
        factor = self.building_keepout * 0.75
        for i in range(building_count(self)):
            self._replace_geom(LtlWalls(
                name=f'building{i}_ltl_walls',
                locate_factor=factor,
                size=factor,
                height=self.building_perimeter_wall_height,
                collision_threshold=self.building_perimeter_wall_collision_threshold,
            ))

    def try_lidar_ids(self, obstacle, obs, i, skip_instance_rows=None):
        """pseudo_occluded lidar with per-instance line-of-sight (walls block view)."""
        skip_rows = skip_instance_rows or frozenset()
        is_occluded = getattr(obstacle, 'is_occluded', True)
        if (
            hasattr(obstacle, 'is_lidar_ids_observed')
            and obstacle.is_lidar_ids_observed
            and self.lidar_conf.type == 'pseudo_occluded'
        ):
            lidar, lidar_ids = self._obs_lidar_pseudo_occluded_new(
                i, obstacle, return_ids=True, skip_instance_rows=skip_rows,
            )
            obs[f"{obstacle.name}_lidar_{i}"] = lidar
            obs[f"{obstacle.name}_lidar_ids_{i}"] = lidar_ids
        elif not is_occluded:
            positions = [
                obstacle.pos[row]
                for row in range(obstacle.num)
                if row not in skip_rows
            ]
            obs[f"{obstacle.name}_lidar_{i}"] = self._obs_lidar_pseudo_new(i, positions)
        else:
            obs[f"{obstacle.name}_lidar_{i}"] = self._obs_lidar_pseudo_occluded_new(
                i, obstacle, skip_instance_rows=skip_rows,
            )

    def obs(self) -> dict | np.ndarray:
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor's data correct
        self._sync_entered_building_state()
        obs = {}

        obs.update(self.agent.obs_sensor())

        # observations of obstacles
        for obstacle in self._obstacles:
            if obstacle.is_lidar_observed:
                if 'gremlins' in obstacle.name:
                    for i in range(self.agent_num):
                        name = f"{obstacle.name}_lidar_{i}"
                        poses = obstacle.pos.copy()
                        del poses[i]
                        obs[name] = self._obs_lidar_new(
                            i, poses, obstacle.group, obstacle=obstacle,
                        )
                elif obstacle.name.endswith('_buildings'):
                    skip_rows = frozenset(self._buildings_entered)
                    for i in range(self.agent_num):
                        self.try_lidar_ids(obstacle, obs, i, skip_instance_rows=skip_rows)
                elif 'casualtys' in obstacle.name:
                    skip_rows = self._rescued_casualty_rows(obstacle)
                    for i in range(self.agent_num):
                        self.try_lidar_ids(obstacle, obs, i, skip_instance_rows=skip_rows)
                else:
                    for i in range(self.agent_num):
                        self.try_lidar_ids(obstacle, obs, i)

            if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                obs[obstacle.name + '_comp'] = self._obs_compass(obstacle.pos)

        buildings = building_geom(self)
        if buildings is not None:
            visited = np.zeros(buildings.num, dtype=np.float64)
            for row in self._buildings_entered:
                visited[row] = 1.0
            obs[f'{buildings.color_name}_buildings_visited'] = visited

        if self.observe_vision:
            for i in range(self.agent_num):
                name = f'vision_{i}'
                obs[name] = self._obs_vision(camera_name=name)
        if self.observation_flatten:
            obs = gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict, obs)
        return obs

    @property
    def goal_achieved(self):
        return mission_goal_achieved(self)
