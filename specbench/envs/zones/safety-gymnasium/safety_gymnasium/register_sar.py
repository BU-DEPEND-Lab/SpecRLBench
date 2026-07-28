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
"""SAR environment and geom registration (isolated from stock safety_gymnasium)."""

from __future__ import annotations

from typing import Callable

_SAR_GEOMS_REGISTERED = False


def _register_sar_geoms() -> None:
    """Append SAR-only geom types to GEOMS_REGISTER registries."""
    global _SAR_GEOMS_REGISTERED  # pylint: disable=global-statement
    if _SAR_GEOMS_REGISTERED:
        return

    from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.buildings import Buildings
    from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.casualtys import Casualtys
    from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import (
        GEOMS_REGISTER as MA_GEOMS_REGISTER,
    )
    from safety_gymnasium.assets.geoms import GEOMS_REGISTER as SA_GEOMS_REGISTER

    for registry in (MA_GEOMS_REGISTER, SA_GEOMS_REGISTER):
        for geom_cls in (Buildings, Casualtys):
            if geom_cls not in registry:
                registry.append(geom_cls)

    _SAR_GEOMS_REGISTERED = True


def register_sar_envs(combine_multi: Callable) -> None:
    """Register SAR multi-agent env IDs and SAR geom types."""
    sar_robots = ['Point']

    sar_ma_tasks = {
        'LTL0MASAR1': {'agent_num': 1},
        'LTL1MASAR1': {'agent_num': 1},
        'LTL2MASAR1': {'agent_num': 1},
    }
    combine_multi(sar_ma_tasks, sar_robots, max_episode_steps=1000)

    multi_goal_sar_tasks = {
        'LTL0MASAR2': {'agent_num': 2},
        'LTL3MASAR1': {'agent_num': 1},
        'LTL1MASAR2': {'agent_num': 2},
        'LTL2MASAR2': {'agent_num': 2},
        'LTL3MASAR2': {'agent_num': 2},
    }
    combine_multi(multi_goal_sar_tasks, sar_robots, max_episode_steps=2500)

    _register_sar_geoms()
