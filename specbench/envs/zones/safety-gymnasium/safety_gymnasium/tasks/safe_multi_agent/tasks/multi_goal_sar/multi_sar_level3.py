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
"""Multi Goal SAR level 3: walls + buildings with surface and entrapped casualties."""

from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal_sar.multi_sar_level2 import (
    MultiGoalSARLevel2,
)


class MultiGoalSARLevel3(MultiGoalSARLevel2):
    """L2 setup with both surface and entrapped casualties enabled."""

    wall_count = 10
    surface_casualties_enabled = True
    entrapped_casualties_enabled = True
