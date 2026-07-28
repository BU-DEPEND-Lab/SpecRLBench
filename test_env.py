import gymnasium as gym
import specbench


def make_env(env_name, render_mode=None):
    if env_name.startswith("Letter"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Panda"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Point") or env_name.startswith("Car") or env_name.startswith("Ant"):
        from specbench.envs.zones.safety_gym_wrapper_ma import SafetyGymWrapperMA
        from specbench.envs.zones.safety_gym_wrapper import SafetyGymWrapper
        import safety_gymnasium

        env = safety_gymnasium.make(env_name, disable_env_checker=True, render_mode=render_mode)
        env = SafetyGymWrapperMA(env) if "MA" in env_name else SafetyGymWrapper(env)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
    return env


seed = 0
env_names = [

    # letter envs (full/partial observability)
    'LetterLTL0-v0',
    'LetterLTL0-v0.partial',

    # zone envs, multi-agent
    'PointLTL0MA3-v0',

    # zone envs, single-agent (full/partial observability, overlap/no overlap, static/dynamic zones)
    'PointLTL0-v0',
    'PointLTL0-v0.partial',
    'PointLTL0-v0.overlap',
    'PointLTL0-v0.partial_overlap',

    'PointLTL1-v0',
    'PointLTL1-v0.partial',
    'PointLTL1-v0.overlap',
    'PointLTL1-v0.partial_overlap',

    'PointLTL2-v0',
    'PointLTL2-v0.partial',
    'PointLTL2-v0.overlap',
    'PointLTL2-v0.partial_overlap',

    'CarLTL0-v0',
    'CarLTL0-v0.partial',
    'CarLTL0-v0.overlap',
    'CarLTL0-v0.partial_overlap',

    'CarLTL1-v0',
    'CarLTL1-v0.partial',
    'CarLTL1-v0.overlap',
    'CarLTL1-v0.partial_overlap',

    'AntLTL0-v0',
    'AntLTL0-v0.partial',
    'AntLTL0-v0.overlap',
    'AntLTL0-v0.partial_overlap',

    'AntLTL1-v0',
    'AntLTL1-v0.partial',
    'AntLTL1-v0.overlap',
    'AntLTL1-v0.partial_overlap',

    'AntLTL2-v0',
    'AntLTL2-v0.partial',
    'AntLTL2-v0.overlap',
    'AntLTL2-v0.partial_overlap',

    'PointLTL0Vision-v0',
    'PointLTL0Vision-v0.overlap',

    'PointLTL1Vision-v0',
    'PointLTL1Vision-v0.overlap',

    'PointLTL2Vision-v0',
    'PointLTL2Vision-v0.overlap',

    'CarLTL0Vision-v0',
    'CarLTL0Vision-v0.overlap',

    'CarLTL1Vision-v0',
    'CarLTL1Vision-v0.overlap',

    'CarLTL2Vision-v0',
    'CarLTL2Vision-v0.overlap',

    'AntLTL0Vision-v0',
    'AntLTL0Vision-v0.overlap',

    'AntLTL1Vision-v0',
    'AntLTL1Vision-v0.overlap',

    'AntLTL2Vision-v0',
    'AntLTL2Vision-v0.overlap',

    # Arm envs (full/partial observability, grippers-only/grippers and arm)
    'PandaLTLReach0Joints-v0',
    'PandaLTLReach0Joints-v0.partial',

    'PandaLTLReach1Joints-v0',
    'PandaLTLReach1Joints-v0.partial',

    # SAR smoke (RISE fork)
    'PointLTL0MASAR2-v0',
]

for env_name in env_names:
    print(f"="*40)
    env = make_env(env_name, render_mode=None)
    obs, info = env.reset(seed=seed)
    for i in range(2):
        try:
            action = env.action_space.sample()
        except Exception:
            action = {a: env.action_space(a).sample() for a in env.possible_agents}
        obs, reward, terminated, truncated, info = env.step(action)
    print(f"checked env: {env_name}")
