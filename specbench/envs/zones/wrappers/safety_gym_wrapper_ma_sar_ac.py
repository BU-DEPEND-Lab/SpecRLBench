from specbench.envs.zones.wrappers.safety_gym_wrapper_ma_sar_wc import SafetyGymWrapperMASARWC


class SafetyGymWrapperMASARAC(SafetyGymWrapperMASARWC):
    """Agent-collision (AC) SAR wrapper: terminate + cost on ``cost_collision``.

    Same sb3 bool / dict termination handling as WC; only the cost channel differs.
    """

    _cost_key = "cost_collision"
