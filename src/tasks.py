import mujoco
import numpy as np


class Task:
    def __init__(self, w):
        self.w = w  # cost weight on each task

    def compute(self, state):
        raise NotImplementedError


class BaseHeightTask(Task):
    def __init__(self, z_des, kp, kd, w):
        super().__init__(w)
        self.z_des = z_des
        self.kp = kp
        self.kd = kd

    def compute(self, state):
        J = np.zeros((1, state.nv))
        J[0, 2] = 1.0
        a_des = (
            self.kp * (self.z_des - state.data.qpos[2]) + self.kd * -state.data.qvel[2]
        )
        e = np.array([a_des])
        return J, e


class BaseOrientationTask(Task):
    def __init__(self, kp, kd, w):
        super().__init__(w)
        self.kp = kp
        self.kd = kd

    def compute(self, state):
        J = np.zeros((3, state.nv))
        J[:, 3:6] = np.eye(3)

        res = np.zeros(3)
        mujoco.mju_subQuat(res, np.array([1.0, 0.0, 0.0, 0.0]), state.data.qpos[3:7])
        e = self.kp * res + self.kd * -state.data.qvel[3:6]
        return J, e


class BaseLinearVelocityTask(Task):
    def __init__(self, v_des, kp, w):
        super().__init__(w)
        self.v_des = v_des
        self.kp = kp

    def compute(self, state):
        J = np.zeros((3, state.nv))
        J[:, 0:3] = np.eye(3)

        e = self.kp * (self.v_des - state.data.qvel[0:3])
        return J, e


class PostureTask(Task):
    def __init__(self, q_des, kp, kd, w):
        super().__init__(w)
        self.q_des = q_des
        self.kp = kp
        self.kd = kd

    def compute(self, state):
        n_tau = state.nv - 6
        J = np.c_[np.zeros((n_tau, 6)), np.eye(n_tau)]
        e = (
            self.kp * (self.q_des - state.data.qpos[7:])
            + self.kd * -state.data.qvel[6:]
        )

        return J, e


class SwingFootTask(Task):
    def __init__(self, foot_idx, p_des, kp, kd, w, v_des=None):
        super().__init__(w)
        self.foot_idx = foot_idx
        self.p_des = p_des
        self.v_des = v_des if v_des is not None else np.zeros(3)
        self.kp = kp
        self.kd = kd

    def compute(self, state):
        i = self.foot_idx
        J = state.J_feet[i]
        v_foot = J @ state.data.qvel  # measured foot vel
        a_des = self.kp * (self.p_des - state.p_feet[i]) + self.kd * (
            self.v_des - v_foot
        )
        e = a_des - state.Jdot_feet[i] @ state.data.qvel
        return J, e
