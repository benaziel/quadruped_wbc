from dataclasses import dataclass
import numpy as np
from wbc import WBC


@dataclass
class GaitParams:
    period: float
    duty_factor: list[float]  # per leg duty factor (how long each leg is in stance)
    phase_offset: list[float]  # bounded btn [0, 1]


class GaitScheduler:
    def __init__(self, params):
        self.params = params

    def leg_phase(self, foot_idx, t):
        return ((t / self.params.period) + self.params.phase_offset[foot_idx]) % 1.0

    def in_stance(self, foot_idx, t):
        return self.leg_phase(foot_idx, t) < self.params.duty_factor[foot_idx]

    def swing_phase(self, foot_idx, t):
        phi = self.leg_phase(foot_idx, t)
        beta = self.params.duty_factor[foot_idx]
        s = (phi - beta) / (1 - beta)  # renormalize s.t. lower bound is 0

        # clip negative (stance) to 0 bc it's unused
        return float(np.clip(s, 0.0, 1.0))
