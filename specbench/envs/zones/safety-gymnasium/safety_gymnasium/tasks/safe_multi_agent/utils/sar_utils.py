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
"""Utils for wall placement on rings and arcs."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING


def ring_locations(radius: float, n: int) -> list[tuple[float, float]]:
    """Fixed (x, y) centers evenly spaced on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float(radius * np.cos(theta)), float(radius * np.sin(theta))) for theta in angles]

def ring_placements(
    radius: float,
    n: int,
    margin: float | None = None,
    keepout: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """Axis-aligned sampling boxes centered on ring points (for random placement).

    Sampler shrinks each box by ``keepout`` on all sides; need ``margin > keepout``
  or ``draw_placement`` asserts with no valid rectangles.
    """
    if margin is None:
        margin = keepout + 0.05
    margin = max(margin, keepout + 1e-3)
    boxes = []
    for x, y in ring_locations(radius, n):
        boxes.append((x - margin, y - margin, x + margin, y + margin))
    return boxes

def border_placements(side_length, margin):
    """
    Generates 4 non-overlapping border boxes around a central square of side length N.
    The center of the system is at (0, 0).
    Output format: (x_min, y_min, x_max, y_max)
    """
    # Inner boundaries (the edges of the central square)
    half_n = side_length / 2.0
    outer = half_n + margin

    boxes = [
        (-outer,  half_n,  outer,  outer),        
        ( half_n, -half_n,  outer,  half_n),
        (-outer, -outer,  outer, -half_n),
        (-outer, -half_n, -half_n,  half_n)
    ]
    return boxes

def border_placement_keepout(margin: float, keepout: float) -> float:
    """Clamp keepout so border strips (thickness ``margin``) stay sampleable."""
    return min(keepout, margin / 2.0 - 1e-3)

def size_randomization(
    base_half_sizes: list,
    n: int,
    margin: float | None = None,
    margins: list[float] = [0, 0, 0],
    random_generator: RandomGenerator | None = None,
) -> list[list[float, float, float]]:
    """Axis-aligned sampling boxes centered on ring points (for random placement).
    Sampler shrinks each box by ``keepout`` on all sides; need ``margin > keepout``
  or ``draw_placement`` asserts with no valid rectangles.
    """
    if margin != None: margins = [margin, margin, margin] if margins == [0, 0, 0] else margins
    x, y, z = base_half_sizes
    x_margin, y_margin, z_margin = margins

    assert any(np.array(base_half_sizes)-np.array(margins) > 0), "Margins should ensure non-negative values" 
    
    return np.array([random_generator.uniform(
        x-x_margin, x+x_margin, n),
        random_generator.uniform(
        y-y_margin, y+y_margin, n),
        random_generator.uniform(
        z-z_margin, z+z_margin, n)]).transpose()

if TYPE_CHECKING:
    from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask
    from safety_gymnasium.tasks.safe_multi_agent.utils.random_generator import RandomGenerator


def is_building_ltl_wall(name: str) -> bool:
    """True for per-building perimeter walls, not the arena ``ltl_walls``."""
    return name.startswith('building') and name.endswith('_ltl_walls')


def building_geom(task: BaseTask):
    """Return the task's Buildings geom, if present."""
    for name in task._geoms:
        if name.endswith('_buildings'):
            return getattr(task, name)
    return None


def building_count(task: BaseTask) -> int:
    """Resolved building count (0 means no buildings; omit config → agent_num in L2+)."""
    return task.building_num


def building_prefix_from_geom(buildings) -> str:
    """Layout key prefix for building instances (e.g. ``terracotta_building``)."""
    return buildings.name[:-1]


def agent_inside_building_idx(task: BaseTask, agent_idx: int) -> int | None:
    """Return the building instance index the agent is inside, or None."""
    buildings = building_geom(task)
    if buildings is None:
        return None
    agent_xy = task.agent.get_agent_pos(agent_idx)[:2]
    for b_idx, b_pos in enumerate(buildings.pos):
        if np.max(np.abs(agent_xy - np.asarray(b_pos[:2]))) <= buildings.size:
            return b_idx
    return None


def agents_inside_building_indices(task: BaseTask) -> list[int | None]:
    """Per-agent building index when inside a shell, else None."""
    return [agent_inside_building_idx(task, i) for i in range(task.agent_num)]


def agent_has_entrapped_at_building(task: BaseTask, agent_idx: int) -> bool:
    """True when agent is inside a building that hosts an entrapped casualty."""
    inside_idx = agent_inside_building_idx(task, agent_idx)
    if inside_idx is None or not hasattr(task, 'entrapped_casualtys'):
        return False
    return inside_idx < task.entrapped_casualtys.num


def clear_building_pinned_locations(task: BaseTask) -> None:
    """Clear pinned building, entrapped, and perimeter-wall locations before resample."""
    buildings = building_geom(task)
    if buildings is None:
        return
    buildings.locations = []
    if hasattr(task, 'entrapped_casualtys'):
        task.entrapped_casualtys.locations = []
    for name in task._geoms:
        if is_building_ltl_wall(name):
            getattr(task, name).locations = []


def clamp_building_placement_keepout(task: BaseTask, margin: float) -> None:
    """Clamp building keepout so border strips stay sampleable."""
    buildings = building_geom(task)
    if buildings is None or not buildings.placements:
        return
    buildings.keepout = border_placement_keepout(margin, buildings.keepout)


def sync_building_dependents_into_layout(task: BaseTask, layout: dict) -> None:
    """Pin entrapped casualties and perimeter wall segments to building centers."""
    buildings = building_geom(task)
    if buildings is None or buildings.num <= 0:
        return

    building_prefix = building_prefix_from_geom(buildings)
    task._cached_building_rots = task.random_generator.generate_rots(building_count(task))
    buildings.rots = list(task._cached_building_rots)

    if hasattr(task, 'entrapped_casualtys'):
        for i in range(task.entrapped_casualtys.num):
            layout[f'entrapped_casualty{i}'] = layout[f'{building_prefix}{i}'].copy()

    for name in task._geoms:
        if not is_building_ltl_wall(name):
            continue
        wall_idx = int(name[len('building'):name.index('_ltl_walls')])
        wall = getattr(task, name)
        center_xy = layout[f'{building_prefix}{wall_idx}']
        rot = task._cached_building_rots[wall_idx]
        wall.sync_site(center_xy, rot)
        for seg_idx, loc in enumerate(wall.locations):
            layout[f'building{wall_idx}_ltl_wall{seg_idx}'] = np.asarray(loc, dtype=float)


_CASUALTY_GEOM_NAMES = ('surface_casualtys', 'entrapped_casualtys')


def all_casualties_rescued(task: BaseTask) -> bool:
    """Return True when every surface and entrapped casualty (if present) is rescued."""
    found_any = False
    for attr in _CASUALTY_GEOM_NAMES:
        if not hasattr(task, attr):
            continue
        geom = getattr(task, attr)
        if geom.num <= 0:
            continue
        found_any = True
        if not all(geom.rescued):
            return False
    return found_any


def mission_goal_achieved(task: BaseTask) -> tuple[bool, ...]:
    """Shared goal_achieved tuple: same team mission flag for each agent."""
    mission_complete = all_casualties_rescued(task)
    return tuple(mission_complete for _ in range(task.agent_num))

