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

import numpy as np
from safety_gymnasium.tasks.safe_multi_agent.assets.group import GROUP
from safety_gymnasium.tasks.safe_multi_agent.bases.base_object import Geom


class Casualtys(Geom):  # pylint: disable=too-many-instance-attributes
    """Casualties. (Improper spelling, but makes workflow more standardized)"""

    CATEGORIES = {
        # "green": np.array([0, 1, 0, 1]),
        # "red": np.array([1, 0, 0, 1]),
        # "yellow": np.array([1, 1, 0, 1]),
        # "light_gray": np.array([0.75, 0.75, 0.75, 1.0]),

        # -----NOTE: THESE TWO KEYS MUST BE LEFT AT END OF DICTIONARY-----
        "surface": np.array([1, 1, 0, 1]),
        "entrapped": np.array([1, 0, 0, 1]),
        # -----NOTE: THESE TWO KEYS MUST BE LEFT AT END OF DICTIONARY-----
    }

    def __init__(self, category: str, size: float, num: int, locations=None, placements=None, keepout=0.55):
        self.color_name = category
        self.name = f'{category}_casualtys'
        self.num = num
        self.size: float = size
        self.placements: list = placements  # Placements list for hazards (defaults to full extents)
        self.locations: list = locations if locations else []  # Fixed locations to override placements
        self.keepout: float = keepout  # Radius of hazard keepout for placement
        self.alpha: float = 1.0
        self.exterior: bool = (keepout > 0)
        
        # if self.color_name not in self.COLORS:
        #     self.color = self.COLORS['black']
        # else:
        self.color: np.array = self.CATEGORIES[self.color_name]
        self.group: int = GROUP['casualty']
        self.is_lidar_observed: bool = True
        self.is_lidar_ids_observed: bool = True
        self.is_constrained: bool = True
        self.is_meshed: bool = False
        self.rescued = [False] * num

    def calculate_group(self) -> int:
        # return GROUP['goal']
        max_predefined_group = max(GROUP.values())
        return max_predefined_group + sorted(self.CATEGORIES.keys()).index(self.color_name) + 1

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        # Return a flat geom config (single geom), compatible with World.build
        # print(f"DEBUG: size={self.size}")
        geom = {
            'name': self.name,
            'pos': np.r_[xy_pos, self.size*2],
            'rot': rot,
            'size': [self.size, self.size],
            'type': 'capsule',
            'contype': 0,
            'conaffinity': 0,   
            'group': self.group,
            'rgba': self.color * np.array([1.0, 1.0, 1.0, self.alpha]),
        }
        if self.is_meshed:
            geom.update(
                {
                    'type': 'mesh',
                    'mesh': 'bush',
                    'material': 'bush',
                    'euler': [np.pi / 2, 0, 0],
                },
            )
        return geom

    def cal_cost(self):
        cost = {agent: {f'cost_casualtys_{self.color_name}': 0} 
                for agent in self.agent.possible_agents}

        for body_idx, h_pos in enumerate(self.pos):
            # Skip casualties that were already reached
            if self.rescued[body_idx]:
                continue

            for agent_idx in range(self.agent.agent_num):
                agent_h_dist = self.agent.dist_xy(agent_idx, h_pos)

                if agent_h_dist <= self.size + 0.15 and not self.rescued[body_idx]:
                    self.rescued[body_idx] = True
                    cost[f'agent_{agent_idx}'][f'cost_casualtys_{self.color_name}'] = 1.0
                    break

        return cost
    
    @property
    def pos(self):
        """Helper to get the hazards positions from layout."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'{self.name[:-1]}{i}').xpos.copy() for i in range(self.num)]
