# Ensure environments are registered on import
from .envs.letter_world import letter_env
import safety_gymnasium
try:
    import panda_gym
except ImportError:
    panda_gym = None