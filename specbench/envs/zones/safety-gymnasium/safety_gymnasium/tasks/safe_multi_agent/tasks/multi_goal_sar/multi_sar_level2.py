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

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import LtlWalls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.buildings import Buildings
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.casualtys import Casualtys
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import border_placements
from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal_sar.multi_sar_level1 import MultiGoalSARLevel1


class MultiGoalSARLevel2(MultiGoalSARLevel1):
    """Multi-agent zone navigation with optional ring-placed interior walls."""

    wall_count = 10
    building_num = 0
    surface_casualties_per_agent = 0
    entrapped_casualties_per_agent = 1

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # Omitted building_num → one building per agent. Explicit 0 → no buildings.
        if config.get('building_num') is None:
            self.building_num = self.agent_num

        geoms = []
        if self.building_num > 0:
            for i in range(self.building_num):
                self._add_geoms(LtlWalls(name=f'building{i}_ltl_walls'))
            geoms.append(
                Buildings(
                    color=list(Buildings.COLORS)[0],
                    size=self.building_keepout * 0.75,
                    num=int(self.building_num),
                    keepout=self.building_keepout,
                    placements=border_placements(
                        self.building_border_side_length,
                        self.building_margin,
                    ),
                ),
            )
            entrapped_num = int(self.agent_num * self.entrapped_casualties_per_agent)
            if entrapped_num > 0:
                geoms.append(
                    Casualtys(
                        num=int(entrapped_num),
                        category="entrapped",
                        size=self.casualty_size,
                        keepout=self.entrapped_casualty_keepout,
                    ),
                )
        if geoms:
            self._add_geoms(*geoms)

    def calculate_reward(self):
        return super().calculate_reward()

    def specific_reset(self):
        return super().specific_reset()

    def specific_step(self):
        return super().specific_step()

    def update_world(self):
        pass

    def _build(self):
        if self.building_num > 0:
            self._replace_border_buildings(num=self.building_num)
            entrapped_num = int(self.agent_num * self.entrapped_casualties_per_agent)
            if entrapped_num > 0:
                self._replace_geom(Casualtys(
                    category="entrapped",
                    size=self.casualty_size,
                    num=int(entrapped_num),
                    keepout=self.entrapped_casualty_keepout,
                ))
            self._replace_building_perimeter_walls()
        return super()._build()
