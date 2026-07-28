"""SAR env smoke test (uses the same make_zone_env as LTL benchmarks and RISE)."""

import specbench
from specbench.envs.zones.zone_env import make_zone_env

seed = 0
env_names = [
    'PointLTL0MASAR1-v0',
    'PointLTL0MASAR2-v0',
    'PointLTL2MASAR2-v0',
]

for env_name in env_names:
    print(f"=" * 40)
    env = make_zone_env(env_name, render_mode=None)
    obs, info = env.reset(seed=seed)
    for _ in range(2):
        try:
            action = env.action_space.sample()
        except Exception:
            action = {a: env.action_space(a).sample() for a in env.possible_agents}
        obs, reward, terminated, truncated, info = env.step(action)
    print(f"checked env: {env_name}")
