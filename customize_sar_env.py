"""Smoke / customize entrypoint for CustomizedSAR (multi-agent SAR).

Edit the knobs below (defaults from configs/multi_goal_sar.yaml customized_defaults),
then run:
  python SpecRLBench/customize_sar_env.py
"""

import safety_gymnasium
import specbench  # noqa: F401  # registers SpecRLBench env helpers

env_config = {
    'env_id': 'CustomizedSAR-v0',

    # agent type, choose from 'Point', 'Car', and 'Ant'
    'agent_name': 'Point',

    # maximum number of steps per episode
    'max_episode_steps': 1000,

    # === user-facing knobs (defaults from customized_defaults in multi_goal_sar.yaml) ===
    'agent_num': 1,
    'building_num': 0,  # 0 → no buildings/casualties; omit or None → agent_num
    'wall_count': 20,
    'surface_casualties_per_agent': 2,
    'entrapped_casualties_per_agent': 0,
    'reward_goal': 1.0,
    'lidar_conf.num_bins': 16,

    # optional advanced overrides (entity fields from configs/multi_goal_sar.yaml)
    # 'building_keepout': 0.4,
    # 'wall_ring_radius': 2.0,
}


from specbench.envs.zones.safety_gym_register import register_helper

register_helper(env_config=env_config)

env = safety_gymnasium.make(env_config['env_id'], render_mode='human')

obs, info = env.reset(seed=0)

for _ in range(200):
    action = {
        agent: env.action_space(agent).sample()
        for agent in env.unwrapped.possible_agents
    }
    env.step(action)  # MA Builder returns obs, reward, cost, terminated, truncated, info
    env.render()

env.close()
