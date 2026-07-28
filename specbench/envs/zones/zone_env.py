"""Canonical zone/LTL env factory for SpecRLBench (and RISE-Training via import).

Handles Point|Car|Ant, MA, SAR, WC, AC suffixes. Training-only ``sb3`` flattening is
accepted here so all consumers share one construction path.
"""

from __future__ import annotations

import gymnasium as gym


def _strip_cost_suffix(env_name: str) -> str:
    if "AC" in env_name:
        return env_name.replace("AC", "")
    if "WC" in env_name:
        return env_name.replace("WC", "")
    return env_name


def make_zone_env(env_name: str, render_mode=None, sb3: bool = False):
    """Construct zone/LTL env with the correct SafetyGymWrapper stack."""
    if env_name.startswith("Letter"):
        return gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    if env_name.startswith("Panda"):
        return gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    if env_name.startswith("Point") or env_name.startswith("Car") or env_name.startswith("Ant"):
        from specbench.envs.zones.safety_gym_wrapper import SafetyGymWrapper
        from specbench.envs.zones.safety_gym_wrapper_ma import SafetyGymWrapperMA
        from specbench.envs.zones.safety_gym_wrapper_sar import SafetyGymWrapperMASAR
        from specbench.envs.zones.safety_gym_wrapper_sar_wc import SafetyGymWrapperMASARWC
        from specbench.envs.zones.safety_gym_wrapper_sar_ac import SafetyGymWrapperMASARAC
        import safety_gymnasium

        base = _strip_cost_suffix(env_name)
        env = safety_gymnasium.make(
            base, disable_env_checker=True, render_mode=render_mode
        )
        if "SAR" in env_name:
            if "AC" in env_name:
                return SafetyGymWrapperMASARAC(env, sb3=sb3)
            if "WC" in env_name:
                return SafetyGymWrapperMASARWC(env, sb3=sb3)
            return SafetyGymWrapperMASAR(env, sb3=sb3)
        if "MA" in env_name:
            return SafetyGymWrapperMA(env)
        return SafetyGymWrapper(env)

    try:
        import safety_gymnasium

        return safety_gymnasium.make(
            env_name, disable_env_checker=True, render_mode=render_mode
        )
    except Exception as exc:
        raise ValueError(f"Unknown environment name: {env_name}") from exc
