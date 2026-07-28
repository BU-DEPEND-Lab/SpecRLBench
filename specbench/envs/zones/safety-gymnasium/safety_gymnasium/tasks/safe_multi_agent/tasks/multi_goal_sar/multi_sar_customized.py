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
"""Customized multi-agent SAR task with easy YAML-backed knobs."""

from __future__ import annotations

from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal_sar.multi_sar_level3 import (
    MultiGoalSARLevel3,
)
from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal_sar.sar_config_loader import (
    merge_customized_defaults,
    pop_post_init_lidar_overrides,
)

# Keys that CustomizedSAR may override; must_be_constant keys are never skipped.
_EASY_AND_CONFIGURABLE_KEYS = frozenset({
    'agent_num',
    'building_num',
    'wall_count',
    'surface_casualties_per_agent',
    'entrapped_casualties_per_agent',
    'reward_goal',
    'wall_ring_radius',
    'wall_margin',
    'wall_base_half_sizes',
    'walls_keepout',
    'walls_half_size_randomization',
    'building_keepout',
    'building_border_side_length',
    'building_margin',
    'casualty_size',
    'casualty_touch_offset',
    'casualty_keepout',
    'entrapped_casualty_keepout',
    'gremlin_size',
    'gremlin_dist_threshold',
    'gremlin_keepout',
    'building_perimeter_wall_height',
    'building_perimeter_wall_collision_threshold',
    'lidar_conf.num_bins',
    'num_bins',
})


class CustomizedSAR(MultiGoalSARLevel3):
    """Full SAR feature set with user-facing easy knobs from env config."""

    def __init__(self, config) -> None:
        raw = dict(config)
        merged = merge_customized_defaults(raw)
        lidar_overrides = pop_post_init_lidar_overrides(merged)

        skip_keys = frozenset(
            key for key in raw
            if key in _EASY_AND_CONFIGURABLE_KEYS or key.startswith('lidar_conf.')
        )
        if lidar_overrides:
            skip_keys = skip_keys | frozenset({'lidar_conf.num_bins', 'num_bins'})
        merged['_sar_skip_constant_keys'] = tuple(skip_keys)

        super().__init__(config=merged)

        for key, value in lidar_overrides.items():
            setattr(self.lidar_conf, key, value)

        # Gremlin count is always agent_num — never honor a user override.
        if hasattr(self, 'gremlins'):
            self.gremlins.num = self.agent_num
