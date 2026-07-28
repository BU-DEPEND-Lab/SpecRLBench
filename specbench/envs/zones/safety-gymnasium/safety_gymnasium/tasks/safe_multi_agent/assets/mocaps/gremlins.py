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
"""Gremlin."""

from dataclasses import dataclass, field

import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.assets.color import COLOR
from safety_gymnasium.tasks.safe_multi_agent.assets.group import GROUP
from safety_gymnasium.tasks.safe_multi_agent.bases.base_object import Mocap
from safety_gymnasium.tasks.safe_multi_agent.utils.common_utils import *

@dataclass
class Gremlins(Mocap):  # pylint: disable=too-many-instance-attributes
    """Gremlins (moving objects we should avoid)"""

    name: str = 'gremlins'
    num: int = 0  # Number of gremlins in the world
    size: float = 0.1
    placements: list = None  # Gremlins placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.5  # Radius for keeping out (contains gremlin path)
    travel: float = 0.3  # Radius of the circle traveled in
    contact_cost: float = 1.0  # Cost for touching a gremlin
    dist_threshold: float = 0.2  # Threshold for cost for being too close
    dist_cost: float = 1.0  # Cost for being within distance threshold
    density: float = 1e-6 # 0.001

    color: np.array = COLOR['red']
    group: np.array = GROUP['gremlin']
    is_lidar_observed: bool = True
    is_constrained: bool = True

    def __post_init__(self) -> None:
        self.prev_contact = [False] * self.num

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object"""
        return {'obj': self.get_obj(xy_pos, rot), 'mocap': self.get_mocap(xy_pos, rot)}

    def get_obj(self, xy_pos, rot):
        """To facilitate get objects config for this object"""
        return {
            'name': self.name,
            # 'size': np.ones(3) * self.size,
            # 'type': 'box',
            'size': [self.size, 1e-2],
            'type': 'cylinder',
            'density': self.density,
            # 'pos': np.r_[xy_pos, self.size],
            'pos': np.r_[xy_pos, 1e-2],
            'rot': rot,
            'group': self.group,
            'contype': 0,  # No collision generation
            'conaffinity': 0,  # No collision detection with others
            'rgba': np.array([1, 1, 1, 0.1]) * self.color,
        }

    def get_mocap(self, xy_pos, rot):
        """To facilitate get mocaps config for this object"""
        return {
            'name': self.name,
            # 'size': np.ones(3) * self.size,
            # 'type': 'box',
            'size': [self.size, 1e-3],
            'type': 'cylinder',
            # 'pos': np.r_[xy_pos, self.size],
            'pos': np.r_[xy_pos, 1e-3],
            'rot': rot,
            'group': self.group,
            'contype': 0,  # No collision generation
            'conaffinity': 0,  # No collision detection with others
            'rgba': np.array([1, 1, 1, 0]) * self.color,
        }

    def cal_cost(self):
        """Contacts processing."""
        assert self.agent.agent_num == self.num, "Number of agents should be equal to number of gremlins."

        cost = {f"agent_{i}": {"cost_collision": 0} for i in range(self.agent.agent_num)}
        if not self.is_constrained:
            return cost
        
        for i, h_pos in enumerate(self.pos):
            is_in_contact = False
            for j in range(self.agent.agent_num):
                if i == j: continue
                h_dist = self.agent.dist_xy(j, h_pos)
                # print(f"DEBUG: dist(agent_{j}, gremlin_{i}) = {h_dist}")
                # if h_dist <= self.dist_threshold:
                if (h_dist <= self.size + self.dist_threshold):
                    is_in_contact = True
                    # if (not (self.prev_contact[i] or self.prev_contact[j])):
                        # print(f"DEBUG: gremlin collision")
                    cost[f"agent_{j}"]["cost_collision"] = 1.0  # Same cost structure
                    cost[f"agent_{i}"]["cost_collision"] = 1.0
            self.prev_contact[i] = is_in_contact
                # print(f"COST TRIGGERED for {self.color_name} zone {i}!")
        return cost
    max_theta = 0
    min_theta = 0
    def move(self):
        """Set mocap object positions before a physics step is executed."""
        # Read from world engine (bound in World.bind_engine), not preview agent engine.
        for i in range(self.num):
            # Extract current position and heading angle
            agent_xy = self.engine.data.body(f'agent_{i}').xpos[:2]
            theta = quat2rot(self.engine.data.body(f'agent_{i}').xquat.copy())
            self.max_theta = max(theta, self.max_theta)
            self.min_theta = min(theta, self.min_theta)
            # Calculate shifted position (0 degrees = straight ahead)
            # shifted_xy = (agent_xy 
            # + 0.03 * np.array([np.cos(theta), np.sin(theta)])
            # + 0.02 * np.array([np.cos(theta + np.pi/2), np.sin(theta + np.pi/2)]))       
            name = f'gremlin{i}'
            pos = np.r_[agent_xy, [1e-3]]
            self.set_mocap_pos(name + 'mocap', pos)
            self.set_obj_pos(name + 'obj', pos)

    @property
    def pos(self):
        """Helper to get the current gremlin position."""
        return [
            self.engine.data.body(f'gremlin{i}obj').xpos.copy() for i in range(self.num)
        ]
