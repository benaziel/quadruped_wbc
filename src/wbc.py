import numpy as np
from scipy.linalg import block_diag
import scipy.sparse as sp
import einops

import mujoco
import osqp


class WBC:
    def __init__(self, model, mu=0.6):
        self.model = model

        self.nv = model.nv
        self.n_lambda = 12  # 3 * 4 feet
        self.n_tau = 12  # actuated joints, i don't think we actuate the torso

        self.foot_body_ids = [
            model.body(name).id for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ]

        self.hip_body_ids = [
            model.body(name).id for name in ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        ]

        self.thigh_body_ids = [
            model.body(name).id
            for name in ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        ]

        self.base_body_id = model.body("base_link").id

        self.prev_jacobians = {}
        self.prev_time = {}

        self.mu = mu
        self.tau_min = model.actuator_ctrlrange[:, 0]
        self.tau_max = model.actuator_ctrlrange[:, 1]

        self.S_T = np.zeros((self.nv, self.n_tau))
        for ctrl_idx in range(self.model.nu):
            joint_id = self.model.actuator_trnid[ctrl_idx, 0]
            dof_idx = self.model.jnt_dofadr[joint_id]
            self.S_T[dof_idx, ctrl_idx] = 1.0

    def get_jacobian(self, data, body_ids):
        jacobians = np.zeros((len(body_ids), 3, self.nv))
        for i, body_id in enumerate(body_ids):
            jacp = np.zeros((3, self.nv))
            jacr = np.zeros((3, self.nv))
            mujoco.mj_jac(
                self.model,
                data,
                jacp,
                jacr,
                data.xpos[body_id],
                body_id,
            )

            jacobians[i] = jacp

        return jacobians

    def get_jacobian_dot(self, data, body_ids):
        dt = data.time - self.prev_time.get(body_ids, 0.0)

        # not sure if this is valid but idk what to do on the first call otherwise
        J_prev = self.prev_jacobians.get(body_ids, None)
        if J_prev is None or dt < 1e-6:
            self.prev_jacobians[body_ids] = self.get_jacobian(data, body_ids)
            self.prev_time[body_ids] = data.time
            return np.zeros((len(body_ids), 3, self.nv))

        J = self.get_jacobian(data, body_ids)
        Jdot = (J - self.prev_jacobians[body_ids]) / dt

        self.prev_jacobians[body_ids] = J
        self.prev_time[body_ids] = data.time
        return Jdot

    def get_dynamics(self, data):
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, data.qM)

        bias = data.qfrc_bias
        return M, bias

    def friction_cone(self, mu):
        return np.array(
            [[1, 0, -mu], [-1, 0, -mu], [0, 1, -mu], [0, -1, -mu], [0, 0, -1]]
        )

    def compute_base_height_task(self, data, z_des, kp, kd):
        # i want to regulate height so i'm implicitly setting v_des to 0. i think that's fine(?)
        return kp * (z_des - data.qpos[2]) + kd * (-data.qvel[2])

    def compute_base_orientation_task(self, data, kp, kd):
        res = np.zeros(3)
        q_curr = data.qpos[3:7]
        mujoco.mju_subQuat(res, [1, 0, 0, 0], q_curr)
        return kp * res + kd * (-data.qvel[3:6])

    def trot_contact_mask(self, t, period=0.5):
        phase = (t % period) / period  # goes from 0-1
        if phase < 0.5:
            return [1, 0, 0, 1]
        else:
            return [0, 1, 1, 0]

    def compute_swing_p_des(
        self, data, foot_idx, v_des, swing_phase, step_height=0.06, k=0.05, period=0.5
    ):
        # thigh is directly above the natural foot stance position (hip has a y-offset to the thigh)
        thigh_pos = data.xpos[self.thigh_body_ids[foot_idx]].copy()
        v_body = data.qvel[:3]
        T_stance = 0.25  # period * duty_cycle = 0.5 * 0.5

        xy = thigh_pos[:2] + (T_stance / 2) * v_body[:2] + k * (v_body[:2] - v_des[:2])
        z = step_height * np.sin(np.pi * swing_phase)

        # velocity feedforward: time-derivative of the z trajectory
        T_swing = period / 2
        vz = step_height * np.pi / T_swing * np.cos(np.pi * swing_phase)

        return np.array([xy[0], xy[1], z]), np.array([0.0, 0.0, vz])

    def compute_qp(self, data, contact_mask, tasks):

        jacobians = self.get_jacobian(data, tuple(self.foot_body_ids))
        # stupid bug lol. can't hash lists so i can't use foot_body_ids as dict keys unless i turn it into a tuple

        J_c_T = einops.rearrange(jacobians, "n x_dim nv -> nv (n x_dim)")
        M, bias = self.get_dynamics(data)

        # dynamics
        A = np.hstack([M, -J_c_T, -self.S_T])
        b = -bias

        # friction cone inequality constraints
        blocks = [
            self.friction_cone(self.mu) if contact_mask[i] else np.zeros((5, 3))
            for i in range(4)
        ]
        F = block_diag(*blocks)
        C = np.hstack([np.zeros((20, self.nv)), F, np.zeros((20, self.n_tau))])

        # actuator constraints
        T = np.hstack(
            [np.zeros((self.n_tau, self.nv + self.n_lambda)), np.eye(self.n_tau)]
        )

        # stacking everything together
        P = np.vstack([A, C, T])
        l = np.concat([b, -np.inf * np.ones(20), self.tau_min])
        u = np.concat([b, np.zeros(20), self.tau_max])

        # J_task should be the row of the base jacobian corresponding to z
        # maps qddot to zddot
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jac(
            self.model,
            data,
            jacp,
            jacr,
            data.xpos[self.base_body_id],
            1,
        )

        Q = np.zeros((self.nv + self.n_lambda + self.n_tau,) * 2)
        q = np.zeros(self.nv + self.n_lambda + self.n_tau)

        # regularization so the QP stays well conditioned
        Q[self.nv : self.nv + self.n_lambda, self.nv : self.nv + self.n_lambda] += (
            1e-3 * np.eye(self.n_lambda)
        )
        Q[self.nv + self.n_lambda :, self.nv + self.n_lambda :] += 1e-6 * np.eye(
            self.n_tau
        )

        for task in tasks:
            if task["type"] == "base_height":
                J = np.zeros((1, self.nv))
                J[0, 2] = 1.0
                Jdot = np.zeros((1, self.nv))

                ddot_des = task["kp"] * (task["z_des"] - data.qpos[2]) + task["kd"] * (
                    -data.qvel[2]
                )
                e = np.array([ddot_des])

            elif task["type"] == "base_orientation":
                J = np.zeros((3, self.nv))
                J[:, 3:6] = np.eye(3)
                Jdot = np.zeros((3, self.nv))

                res = np.zeros(3)
                mujoco.mju_subQuat(res, np.array([1.0, 0.0, 0.0, 0.0]), data.qpos[3:7])

                ddot_des = task["kp"] * res + task["kd"] * (-data.qvel[3:6])
                e = ddot_des

            elif task["type"] == "swing_foot":
                foot_body_id = self.foot_body_ids[task["foot_idx"]]
                jacp = np.zeros((3, self.nv))
                mujoco.mj_jac(
                    self.model, data, jacp, None, data.xpos[foot_body_id], foot_body_id
                )
                J = jacp

                body_ids = (foot_body_id,)
                Jdot = self.get_jacobian_dot(data, body_ids)[0]

                p_foot = data.xpos[foot_body_id].copy()
                v_foot = J @ data.qvel
                v_des_task = task.get("v_des", np.zeros(3))
                ddot_des = task["kp"] * (task["p_des"] - p_foot) + task["kd"] * (
                    v_des_task - v_foot
                )
                e = ddot_des - Jdot @ data.qvel

            elif task["type"] == "base_linear_vel":
                J = np.zeros((3, self.nv))
                J[:, 0:3] = np.eye(3)
                Jdot = np.zeros((3, self.nv))

                ddot_des = task["kp"] * (task["v_des"] - data.qvel[0:3])
                e = ddot_des

            elif task["type"] == "posture":
                J = np.hstack([np.zeros((self.n_tau, 6)), np.eye(self.n_tau)])
                e = task["kp"] * (task["q_des"] - data.qpos[7:]) + task["kd"] * (
                    -data.qvel[6:]
                )

            else:
                continue

            Q[: self.nv, : self.nv] += task["w"] * J.T @ J
            q[: self.nv] += -task["w"] * J.T @ e

        swing_rows = []
        for i in range(4):
            if not contact_mask[i]:
                row = np.zeros((3, self.nv + self.n_lambda + self.n_tau))
                row[:, self.nv + 3 * i : self.nv + 3 * (i + 1)] = np.eye(3)
                swing_rows.append(row)

        if swing_rows:
            swing_block = np.vstack(swing_rows)
            n_swing = swing_block.shape[0]
            P = np.vstack([P, swing_block])
            l = np.concatenate([l, np.zeros(n_swing)])
            u = np.concatenate([u, np.zeros(n_swing)])

        prob = osqp.OSQP()
        prob.setup(
            sp.triu(sp.csc_matrix(Q)).tocsc(), q, sp.csc_matrix(P), l, u, verbose=False
        )
        res = prob.solve()
        if res.info.status not in ("solved", "solved_inaccurate"):
            print(f"OSQP: {res.info.status}")
        return res.x
