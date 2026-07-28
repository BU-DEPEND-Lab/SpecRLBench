"""Zone/LTL env factory for SpecRLBench (benchmark-only, no SafePO/CMDP)."""

from __future__ import annotations

import gymnasium as gym


def make_zone_env(env_name: str, render_mode=None, sb3: bool = False):
    """Construct zone/LTL env with the correct SafetyGymWrapper stack.

    Handles Point|Car|Ant, MA, SAR, WC, and AC suffixes.
    """
    if env_name.startswith("Letter"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Panda"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Point") or env_name.startswith("Car") or env_name.startswith("Ant"):
        from specbench.envs.zones.wrappers import (
            SafetyGymWrapper,
            SafetyGymWrapperMA,
            SafetyGymWrapperMASAR,
            SafetyGymWrapperMASARWC,
            SafetyGymWrapperMASARAC,
        )
        import safety_gymnasium

        if "AC" in env_name:
            base = env_name.replace("AC", "")
        elif "WC" in env_name:
            base = env_name.replace("WC", "")
        else:
            base = env_name
        env = safety_gymnasium.make(
            base, disable_env_checker=True, render_mode=render_mode
        )
        if "SAR" in env_name:
            if "AC" in env_name:
                env = SafetyGymWrapperMASARAC(env, sb3=sb3)
            elif "WC" in env_name:
                env = SafetyGymWrapperMASARWC(env, sb3=sb3)
            else:
                env = SafetyGymWrapperMASAR(env, sb3=sb3)
        elif "MA" in env_name:
            env = SafetyGymWrapperMA(env)
        else:
            env = SafetyGymWrapper(env)
    else:
        try:
            import safety_gymnasium

            env = safety_gymnasium.make(
                env_name, disable_env_checker=True, render_mode=render_mode
            )
        except Exception:
            raise ValueError(f"Unknown environment name: {env_name}")
    return env
