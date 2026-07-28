import copy

from gymnasium import make as gymnasium_make
from gymnasium import register as gymnasium_register
from safety_gymnasium.utils.registration import make, register


def register_helper(env_config, multi_agent=False):
    """Register a environment to both Safety-Gymnasium and Gymnasium registry.

    Args:
        env_config: Registration dict including ``env_id`` and ``max_episode_steps``.
        multi_agent: If True, use the safe_multi_agent Builder (required for SAR).
    """
    env_name, dash, version = env_config['env_id'].partition('-')
    config = {'config': env_config, 'task_id': env_config['env_id']}

    entry_point = (
        'safety_gymnasium.tasks.safe_multi_agent.builder:Builder'
        if multi_agent
        else 'safety_gymnasium.builder:Builder'
    )

    register(
        id=env_config['env_id'],
        entry_point=entry_point,
        kwargs=config,
        max_episode_steps=env_config["max_episode_steps"],
        disable_env_checker=multi_agent,
    )
    gymnasium_register(
        id=f'{env_name}Gymnasium{dash}{version}',
        entry_point='safety_gymnasium.wrappers.gymnasium_conversion:make_gymnasium_environment',
        kwargs={'env_id': f'{env_name}Gymnasium{dash}{version}', **config},
        max_episode_steps=env_config["max_episode_steps"],
    )
