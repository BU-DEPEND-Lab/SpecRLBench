from specbench.envs.zones.safety_gym_wrapper_ma_sar import SafetyGymWrapperMASAR


class SafetyGymWrapperMASARWC(SafetyGymWrapperMASAR):
    """Wall-cost (WC) SAR wrapper: terminate + ``info['cost']`` on ``cost_walls``.

    Parent ``SafetyGymWrapperMASAR`` with ``sb3=True`` already collapses
    ``terminated`` to a bool. Must set that bool (not ``terminated[agent]``).
    """

    _cost_key = "cost_walls"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["cost"] = 0
        hit = False
        for a in self.env.unwrapped.possible_agents:
            agent_info = info.get(a) if isinstance(info, dict) else None
            if not isinstance(agent_info, dict):
                continue
            if float(agent_info.get(self._cost_key, 0) or 0) > 0:
                info["cost"] += 1
                hit = True
                if isinstance(terminated, dict):
                    terminated[a] = True
        if hit and not isinstance(terminated, dict):
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["cost"] = 0
        return obs, info
