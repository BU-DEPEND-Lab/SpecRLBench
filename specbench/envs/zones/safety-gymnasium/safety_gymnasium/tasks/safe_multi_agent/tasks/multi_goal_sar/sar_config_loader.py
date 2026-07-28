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
"""Load and apply shared SAR YAML recipe from safety_gymnasium/configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import safety_gymnasium
import yaml

_CACHED_CONFIG: dict[str, Any] | None = None

# Registration-only keys that must not reach BaseTask._parse.
REGISTRATION_KEYS = frozenset({'env_id'})

# Top-level YAML keys not applied as shared recipe onto tasks.
_META_SECTIONS = frozenset({'levels', 'customized_defaults'})

# Entity blocks expanded into flat task attributes before apply.
_ENTITY_ROOTS = frozenset({'walls', 'buildings', 'casualtys', 'gremlins', 'reward'})

# Nested YAML blocks that map onto task / conf objects.
_NESTED_CONF_ROOTS = frozenset({
    'lidar_conf',
    'cost_conf',
    'mechanism_conf',
    'placements_conf',
    'render_conf',
})


def _sar_config_path() -> Path:
    return Path(safety_gymnasium.__file__).resolve().parent / 'configs' / 'multi_goal_sar.yaml'


def load_sar_config(*, force_reload: bool = False) -> dict[str, Any]:
    """Load and cache ``configs/multi_goal_sar.yaml``."""
    global _CACHED_CONFIG  # pylint: disable=global-statement
    if _CACHED_CONFIG is None or force_reload:
        with _sar_config_path().open(encoding='utf-8') as handle:
            _CACHED_CONFIG = yaml.safe_load(handle) or {}
    return _CACHED_CONFIG


def _as_tuple_placements(value: Any) -> list[tuple[float, float, float, float]]:
    return [tuple(float(x) for x in row) for row in value]


def _expand_recipe_mapping(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten domain-entity YAML into task attribute / conf mapping."""
    flat: dict[str, Any] = {}

    for key, value in cfg.items():
        if key in _META_SECTIONS or key in _ENTITY_ROOTS:
            continue
        if key == 'agent' and isinstance(value, dict):
            flat['agent'] = value
            continue
        flat[key] = value

    walls = cfg.get('walls') or {}
    if walls:
        flat['wall_ring_radius'] = walls.get('ring_radius')
        flat['wall_margin'] = walls.get('margin')
        flat['wall_base_half_sizes'] = walls.get('base_half_sizes')
        flat['walls_keepout'] = walls.get('keepout')
        flat['walls_half_size_randomization'] = walls.get('half_size_randomization')

    buildings = cfg.get('buildings') or {}
    if buildings:
        flat['building_keepout'] = buildings.get('keepout')
        flat['building_border_side_length'] = buildings.get('border_side_length')
        flat['building_margin'] = buildings.get('margin')
        perimeter = buildings.get('perimeter_walls')
        if perimeter:
            flat['building_perimeter_walls'] = perimeter

    casualtys = cfg.get('casualtys') or {}
    if casualtys:
        flat['casualty_size'] = casualtys.get('size')
        flat['casualty_touch_offset'] = casualtys.get('touch_offset')
        flat['casualty_keepout'] = casualtys.get('keepout_surface')
        flat['entrapped_casualty_keepout'] = casualtys.get('keepout_entrapped')

    gremlins = cfg.get('gremlins') or {}
    if gremlins:
        flat['gremlins'] = gremlins

    reward = cfg.get('reward') or {}
    if 'reward_goal' in reward:
        flat['reward_goal'] = reward['reward_goal']

    return {k: v for k, v in flat.items() if v is not None}


def _apply_mapping(task: Any, mapping: dict[str, Any], *, prefix: str = '') -> None:
    """Apply a nested dict onto ``task`` or nested conf objects."""
    for key, value in mapping.items():
        path = f'{prefix}.{key}' if prefix else key

        if key == 'agent' and isinstance(value, dict):
            if 'keepout' in value:
                task.agent_keepout = float(value['keepout'])
            if 'placements' in value:
                task.agent_placements = _as_tuple_placements(value['placements'])
            continue

        if key == 'gremlins' and isinstance(value, dict):
            task.gremlin_size = float(value['size'])
            task.gremlin_dist_threshold = float(value['dist_threshold'])
            task.gremlin_keepout = float(value['keepout'])
            continue

        if key == 'building_perimeter_walls' and isinstance(value, dict):
            task.building_perimeter_wall_height = float(value['height'])
            task.building_perimeter_wall_collision_threshold = float(
                value['collision_threshold'],
            )
            continue

        if key in _NESTED_CONF_ROOTS and isinstance(value, dict):
            conf_obj = getattr(task, key)
            for nested_key, nested_value in value.items():
                setattr(conf_obj, nested_key, nested_value)
            continue

        if isinstance(value, dict):
            _apply_mapping(task, value, prefix=path)
            continue

        setattr(task, key, value)


def _filter_mapping(
    mapping: dict[str, Any],
    skip: frozenset[str],
    prefix: str = '',
) -> dict[str, Any]:
    """Drop keys (and nested leaves) listed in ``skip``."""
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f'{prefix}.{key}' if prefix else key
        if path in skip or key in skip:
            continue
        if isinstance(value, dict) and key not in {
            'agent', 'gremlins', 'building_perimeter_walls',
        }:
            nested = _filter_mapping(value, skip, prefix=path)
            if nested:
                out[key] = nested
            continue
        if isinstance(value, dict):
            nested = _filter_mapping(value, skip, prefix=path)
            if nested:
                out[key] = nested
            continue
        out[key] = value
    return out


def apply_sar_recipe(
    task: Any,
    skip_keys: frozenset[str] | None = None,
) -> None:
    """Set shared SAR recipe from ``configs/multi_goal_sar.yaml`` onto ``task``.

    ``skip_keys`` preserves explicit CustomizedSAR overrides already applied via
    ``BaseTask._parse`` (top-level names or dotted ``lidar_conf.num_bins``).
    """
    skip = skip_keys or frozenset()
    recipe = _expand_recipe_mapping(load_sar_config())
    filtered = _filter_mapping(recipe, skip)
    _apply_mapping(task, filtered)


def apply_sar_constants(
    task: Any,
    sections: tuple[str, ...] = (),  # noqa: ARG001 — legacy arg, ignored
    skip_keys: frozenset[str] | None = None,
) -> None:
    """Backward-compatible alias for ``apply_sar_recipe``."""
    apply_sar_recipe(task, skip_keys=skip_keys)


def merge_customized_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Merge YAML ``customized_defaults`` into a CustomizedSAR env config.

    Leaves registration keys stripped for BaseTask._parse. Nested
    ``lidar_conf.num_bins`` is returned under dotted key so callers can apply
    it after ``lidar_conf`` exists.
    """
    merged = dict(config)
    for key in REGISTRATION_KEYS:
        merged.pop(key, None)

    defaults = dict(load_sar_config().get('customized_defaults') or {})
    lidar_defaults = dict(defaults.pop('lidar_conf', None) or {})

    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value

    if 'num_bins' in lidar_defaults and 'lidar_conf.num_bins' not in merged:
        if not (
            isinstance(merged.get('lidar_conf'), dict)
            and 'num_bins' in merged['lidar_conf']
        ):
            merged['lidar_conf.num_bins'] = lidar_defaults['num_bins']

    if merged.get('building_num') is None:
        merged.pop('building_num', None)

    nested_lidar = merged.pop('lidar_conf', None)
    if isinstance(nested_lidar, dict):
        for nested_key, nested_value in nested_lidar.items():
            dotted = f'lidar_conf.{nested_key}'
            if dotted not in merged:
                merged[dotted] = nested_value

    return merged


def merge_easy_sar_config(config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for ``merge_customized_defaults``."""
    return merge_customized_defaults(config)


def pop_post_init_lidar_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Pull ``lidar_conf.*`` keys that must be applied after BaseTask creates confs."""
    overrides = {}
    for key in list(config.keys()):
        if key.startswith('lidar_conf.'):
            overrides[key.split('.', 1)[1]] = config.pop(key)
    return overrides


def get_level_yaml_defaults(level_key: str) -> dict[str, Any]:
    """Return documented level identity from YAML ``levels`` section."""
    levels = load_sar_config().get('levels') or {}
    return dict(levels.get(level_key) or {})
