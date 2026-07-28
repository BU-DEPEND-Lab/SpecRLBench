# Copyright 2024 anonymous-elephant. All Rights Reserved.
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

from dataclasses import dataclass
import re

import numpy as np
from safety_gymnasium.tasks.safe_multi_agent.assets.color import COLOR
from safety_gymnasium.tasks.safe_multi_agent.assets.group import GROUP
from safety_gymnasium.tasks.safe_multi_agent.bases.base_object import Geom


@dataclass
class LtlWalls(Geom):  # pylint: disable=too-many-instance-attributes
    """
    The walls at the boundary of any LTL tasks.
    """

    name: str = 'ltl_walls'
    num: int = 4
    locate_factor: float = 3.5
    theta: float = 0
    collision_threshold: float = 3.3
    size: float = 3.5
    height: float = 0.25
    placements: list = None
    locations: list = None 
    keepout: float = 0.0
    rots: list = None
    color: np.array = COLOR['wall']
    alpha: float = 0.9
    group: np.array = GROUP['wall']
    is_lidar_observed: bool = False
    d_x: float = 0.0
    d_y: float = 0.0
    h_index: int = None
    contype: int = 0

    def __post_init__(self) -> None:
        # Per-agent boundary contact flags (not per wall segment).
        self.prev_contact: list[bool] = []
        if self.name.startswith('building') and self.name.endswith('_ltl_walls'):
            self.color = COLOR['terracotta']
            self.h_index = int(re.search(r"\d+", self.name).group())
            if self.rots is not None:
                self.theta = self.rots[self.h_index]
        assert self.num in (2, 4)
        assert (
            self.locate_factor >= 0
        ), 'For cost calculation, the locate_factor must be greater than or equal to zero.'

        if self.locations is not None:
            self.d_x, self.d_y = self.locations[0], self.locations[1]

        self.sync_site((self.d_x, self.d_y), self.theta)

    def sync_site(self, center_xy, rot) -> None:
        """Set wall segment centers around a building center with rotation."""
        self.d_x, self.d_y = float(center_xy[0]), float(center_xy[1])
        self.theta = float(rot)
        self.locations = [
            (self.locate_factor + self.d_x, self.d_y),
            (-self.locate_factor + self.d_x, self.d_y),
            (self.d_x, self.locate_factor + self.d_y),
            (self.d_x, -self.locate_factor + self.d_y),
        ]
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        self.locations = [
            (
                (x - self.d_x) * cos_t - (y - self.d_y) * sin_t + self.d_x,
                (x - self.d_x) * sin_t + (y - self.d_y) * cos_t + self.d_y,
            )
            for x, y in self.locations
        ]
        self.index = 0

    def index_tick(self):
        """Count index."""
        self.index += 1
        self.index %= self.num

    def process_config(self, config, layout, rots):
        self.index = 0
        return super().process_config(config, layout, rots)

    def get_config(self, xy_pos, rot):  # pylint: disable=unused-argument
        """To facilitate get specific config for this object."""
        xy_pos = np.asarray(xy_pos, dtype=float)
        if self.name.startswith('building') and self.name.endswith('_ltl_walls'):
            rot = float(np.arctan2(xy_pos[1] - self.d_y, xy_pos[0] - self.d_x))
        else:
            rot = [np.arctan2(y - self.d_y, x - self.d_x) for x, y in self.locations][self.index]
        wall_size = np.array([0.025, self.size, self.height])
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, self.height],
            'rot': rot,
            'size': wall_size,
            'type': 'box',
            'contype': self.contype,
            'conaffinity': 0,
            'group': self.group,
            'rgba': self.color * np.array([1, 1, 1, self.alpha]),
        }
        self.index_tick()
        return body

    def cal_cost(self):
        cost = {}
        agent_num = self.agent.agent_num
        if len(self.prev_contact) < agent_num:
            self.prev_contact.extend([False] * (agent_num - len(self.prev_contact)))
        for i in range(agent_num):
            pos = self.agent.get_agent_pos(i)
            cond = pos[0] >= self.collision_threshold or \
                pos[0] <= -self.collision_threshold or \
                pos[1] >= self.collision_threshold or \
                pos[1] <= -self.collision_threshold
            cost[f'agent_{i}'] = {
                f'wall_sensor': self.wall_sensor(pos[0], pos[1]),
                f'cost_ltl_walls': cond * 1,
            }
            self.prev_contact[i] = cond
        return cost

    def wall_sensor(self, x, y):
        return np.array([
            self.calculate_wall_distance(pos, threshold)
            for pos, threshold in
            [(x, self.collision_threshold), (y, -self.collision_threshold), (x, -self.collision_threshold),
             (y, self.collision_threshold)]
        ])

    @staticmethod
    def calculate_wall_distance(pos: float, threshold: float, gain: float = 1):
        return np.exp(-gain * np.abs(pos - threshold))

    @property
    def pos(self):
        """Helper to get list of Sigwalls positions."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'{self.name[:-1]}{i}').xpos.copy() for i in range(self.num)]
