# Compatibility shim for SafePO / OmniSafe imports.
# Re-export SafeMAEnv from the velocity multi-agent task module.
from safety_gymnasium.tasks.safe_multi_agent.tasks.velocity.safe_mujoco_multi import (  # noqa: F401
    SafeMAEnv,
)

__all__ = ["SafeMAEnv"]
