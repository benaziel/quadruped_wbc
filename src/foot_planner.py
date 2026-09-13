import numpy as np


class FootPlanner:
    def __init__(self, schedule, step_height=0.06, k=0.05):
        self.schedule = schedule
        self.step_height = step_height
        self.k = k  # velocity feedback (i'll  tune the gain later)

    def compute_swing_p_des(self, state, foot_idx, v_des, swing_phase):
        period = self.schedule.params.period
        beta = self.schedule.params.duty_factor[
            foot_idx
        ]  # fraction of cycle the leg spends on the ground

        T_stance = period * beta
        T_swing = period - T_stance

        # thigh is directly above the natural foot stance pos
        thigh_pos = state.data.xpos[state.thigh_body_ids[foot_idx]].copy()
        v_body = state.data.qvel[:3]

        # where foot should touch down
        p_foothold = (
            thigh_pos[:2]
            + (T_stance / 2) * v_body[:2]
            + self.k * (v_body[:2] - v_des[:2])
        )
        z_des = self.step_height * np.sin(np.pi * swing_phase)

        vz_des = self.step_height * np.pi / T_swing * np.cos(np.pi * swing_phase)

        return np.r_[p_foothold, z_des], np.r_[0, 0, vz_des]
