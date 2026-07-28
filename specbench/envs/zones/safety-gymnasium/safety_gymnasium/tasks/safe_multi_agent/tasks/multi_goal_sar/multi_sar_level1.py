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

import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import Walls
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import ring_placements, size_randomization
from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal_sar.multi_sar_level0 import MultiGoalSARLevel0


class MultiGoalSARLevel1(MultiGoalSARLevel0):
    """Multi-agent zone navigation with optional ring-placed interior walls."""

    wall_count = 10

    def __init__(self, config) -> None:
        super().__init__(config=config)
        if self.wall_count > 0:
            self._add_geoms(Walls(num=int(self.wall_count)))

    def specific_reset(self):
        return super().specific_reset()

    def specific_step(self):
        return super().specific_step()

    def update_world(self):
        pass

    def _build(self):
        if self.wall_count <= 0:
            return super()._build()

        if self.walls_half_size_randomization:
            self._cached_wall_half_sizes = size_randomization(
                self.wall_base_half_sizes,
                self.wall_count,
                margins=(np.array(self.wall_base_half_sizes) / 2).tolist(),
                random_generator=self.random_generator,
            ) if self._cached_wall_half_sizes is None else self._cached_wall_half_sizes
        else:
            if self._cached_wall_half_sizes is None:
                self._cached_wall_half_sizes = [
                    list(self.wall_base_half_sizes) for _ in range(self.wall_count)
                ]

        self._replace_geom(Walls(
            num=int(self.wall_count),
            placements=ring_placements(
                self.wall_ring_radius, self.wall_count, margin=self.wall_margin,
            ),
            half_sizes=self._cached_wall_half_sizes,
            keepout=self.walls_keepout,
        ))
        return super()._build()
