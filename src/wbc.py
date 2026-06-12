import numpy as np
from scipy.linalg import block_diag
import scipy.sparse as sp

import mujoco
import osqp


class WBC:
    def __init__(self, model, mu=0.6):
        self.model = model

        self.nv = model.nv
        self.n_lambda = 12  # 3 * 4 feet
        self.n_tau = 12  # actuated joints, i don't think we actuate the torso
        self.n = self.nv + self.n_lambda + self.n_tau

        self.foot_body_ids = [
            model.body(name).id for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ]
        self.thigh_body_ids = [
            model.body(name).id
            for name in ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        ]
        self.base_body_id = model.body("base_link").id

        self.mu = mu
        self.tau_min = model.actuator_ctrlrange[:, 0]
        self.tau_max = model.actuator_ctrlrange[:, 1]

        # selection matrix maps torques into joint-space forces
        self.S_T = np.zeros((self.nv, self.n_tau))
        for ctrl_idx in range(self.model.nu):
            joint_id = self.model.actuator_trnid[ctrl_idx, 0]
            dof_idx = self.model.jnt_dofadr[joint_id]
            self.S_T[dof_idx, ctrl_idx] = 1.0

        self.prob = None
        self.Q_pat = None
        self.P_pat = None
        self.prev_J = None
        self.prev_time = None

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

    def update_foot_kinematics(self, data):
        J = self.get_jacobian(data, tuple(self.foot_body_ids))

        if self.prev_J is None:
            Jdot = np.zeros_like(J)
        else:
            dt = data.time - self.prev_time
            Jdot = (J - self.prev_J) / dt if dt > 1e-9 else np.zeros_like(J)
        self.prev_J = J.copy()
        self.prev_time = data.time
        return J, Jdot

    def get_dynamics(self, data):
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, data.qM)

        bias = data.qfrc_bias
        return M, bias

    def friction_cone(self, mu):
        return np.array([
            [1, 0, -mu],
            [-1, 0, -mu],
            [0, 1, -mu],
            [0, -1, -mu],
            [0, 0, -1],
        ])

    def trot_contact_mask(self, t, period=0.5):
        phase = (t % period) / period  # goes from 0-1
        if phase < 0.5:
            return [1, 0, 0, 1]
        else:
            return [0, 1, 1, 0]

    def swing_phase(self, foot_idx, t, period=0.5):
        phase = (t % period) / period
        if foot_idx in (0, 3):
            return float(np.clip((phase - 0.5) / 0.5, 0.0, 1.0))
        return float(np.clip(phase / 0.5, 0.0, 1.0))

    def feet_in_contact(self, data):
        # measured contact state from the engine, per foot
        measured = [False] * 4
        for k in range(data.ncon):
            con = data.contact[k]
            for geom in (con.geom1, con.geom2):
                body = self.model.geom_bodyid[geom]
                if body in self.foot_body_ids:
                    measured[self.foot_body_ids.index(body)] = True
        return measured

    def compute_swing_p_des(
        self, data, foot_idx, v_des, swing_phase, step_height=0.06, k=0.05, period=0.5
    ):
        # thigh is directly above the natural foot stance position (hip has a y-offset to the thigh)
        thigh_pos = data.xpos[self.thigh_body_ids[foot_idx]].copy()
        v_body = data.qvel[:3]
        T_stance = period / 2

        xy = thigh_pos[:2] + (T_stance / 2) * v_body[:2] + k * (v_body[:2] - v_des[:2])
        z = step_height * np.sin(np.pi * swing_phase)

        # velocity feedforward: time-derivative of the z trajectory
        T_swing = period / 2
        vz = step_height * np.pi / T_swing * np.cos(np.pi * swing_phase)

        return np.array([xy[0], xy[1], z]), np.array([0.0, 0.0, vz])

    def compute_qp(self, data, contact_mask, tasks, J_feet, Jdot_feet):
        J_c_T = J_feet.reshape(self.n_lambda, self.nv).T
        M, bias = self.get_dynamics(data)

        # dynamics
        A = np.hstack([M, -J_c_T, -self.S_T])
        b = -bias

        # friction cone inequality constraints
        blocks = [self.friction_cone(self.mu) for _ in range(4)]
        F = block_diag(*blocks)
        n_cone = F.shape[0]
        C = np.hstack([np.zeros((n_cone, self.nv)), F, np.zeros((n_cone, self.n_tau))])

        # actuator constraints
        T = np.hstack([
            np.zeros((self.n_tau, self.nv + self.n_lambda)),
            np.eye(self.n_tau),
        ])

        # swing force rows (equality lambda_i = 0 for swing feet)
        Sw = np.hstack([
            np.zeros((self.n_lambda, self.nv)),
            np.eye(self.n_lambda),
            np.zeros((self.n_lambda, self.n_tau)),
        ])
        l_sw = np.empty(self.n_lambda)
        u_sw = np.empty(self.n_lambda)
        for i in range(4):
            rows = slice(3 * i, 3 * (i + 1))
            if contact_mask[i]:
                l_sw[rows], u_sw[rows] = -np.inf, np.inf
            else:
                l_sw[rows], u_sw[rows] = 0.0, 0.0

        # stacking everything together
        P = np.vstack([A, C, T, Sw])
        l = np.concat([b, -np.inf * np.ones(n_cone), self.tau_min, l_sw])
        u = np.concat([b, np.zeros(n_cone), self.tau_max, u_sw])

        Q = np.zeros((self.n, self.n))
        q = np.zeros(self.n)

        # regularization so the QP stays well conditioned
        Q[: self.nv, : self.nv] += 1e-6 * np.eye(self.nv)
        Q[self.nv : self.nv + self.n_lambda, self.nv : self.nv + self.n_lambda] += (
            1e-6 * np.eye(self.n_lambda)
        )
        Q[self.nv + self.n_lambda :, self.nv + self.n_lambda :] += 1e-6 * np.eye(
            self.n_tau
        )

        for task in tasks:
            if task["type"] == "base_height":
                J = np.zeros((1, self.nv))
                J[0, 2] = 1.0

                ddot_des = task["kp"] * (task["z_des"] - data.qpos[2]) + task["kd"] * (
                    -data.qvel[2]
                )
                e = np.array([ddot_des])

            elif task["type"] == "base_orientation":
                # mju_subQuat returns the error in the same local frame (free-joint angular dofs)
                J = np.zeros((3, self.nv))
                J[:, 3:6] = np.eye(3)

                res = np.zeros(3)
                mujoco.mju_subQuat(res, np.array([1.0, 0.0, 0.0, 0.0]), data.qpos[3:7])
                e = task["kp"] * res + task["kd"] * (-data.qvel[3:6])

            elif task["type"] == "swing_foot":
                i = task["foot_idx"]
                J = J_feet[i]

                p_foot = data.xpos[self.foot_body_ids[i]].copy()
                v_foot = J @ data.qvel
                v_des_task = task.get("v_des", np.zeros(3))
                ddot_des = task["kp"] * (task["p_des"] - p_foot) + task["kd"] * (
                    v_des_task - v_foot
                )
                e = ddot_des - Jdot_feet[i] @ data.qvel

            elif task["type"] == "base_linear_vel":
                J = np.zeros((3, self.nv))
                J[:, 0:3] = np.eye(3)
                e = task["kp"] * (task["v_des"] - data.qvel[0:3])

            elif task["type"] == "posture":
                J = np.hstack([np.zeros((self.n_tau, 6)), np.eye(self.n_tau)])
                e = task["kp"] * (task["q_des"] - data.qpos[7:]) + task["kd"] * (
                    -data.qvel[6:]
                )

            else:
                continue

            Q[: self.nv, : self.nv] += task["w"] * J.T @ J
            q[: self.nv] += -task["w"] * J.T @ e

        if self.prob is None:
            self._setup_problem(Q, q, P, l, u)
        else:
            self.prob.update(
                Px=Q[self.Q_rows, self.Q_cols],
                Ax=P[self.P_rows, self.P_cols],
                q=q,
                l=l,
                u=u,
            )

        res = self.prob.solve()
        if res.info.status not in ("solved", "solved_inaccurate"):
            print(f"OSQP: {res.info.status}")
            return None
        return res

    def _setup_problem(self, Q, q, P, l, u):
        # sparsity templates for osqp

        # constraint matirx template
        P_t = np.zeros_like(P)
        P_t[: self.nv, : self.nv] = 1.0
        P_t[: self.nv, self.nv : self.nv + self.n_lambda] = 1.0
        P_t[: self.nv, self.nv + self.n_lambda :] = self.S_T != 0
        P_t[self.nv :] = P[self.nv :] != 0

        # csc stores values col by col
        self.P_pat = sp.csc_matrix(P_t)
        self.P_rows = self.P_pat.indices.copy()
        self.P_cols = np.repeat(np.arange(self.n), np.diff(self.P_pat.indptr))

        # cost template
        Q_t = np.zeros_like(Q)
        Q_t[: self.nv, : self.nv] = 1.0
        np.fill_diagonal(Q_t, 1.0)
        self.Q_pat = sp.triu(sp.csc_matrix(Q_t)).tocsc()
        self.Q_rows = self.Q_pat.indices.copy()
        self.Q_cols = np.repeat(np.arange(self.n), np.diff(self.Q_pat.indptr))

        # first step vals
        Q_csc = self.Q_pat.copy()
        Q_csc.data = Q[self.Q_rows, self.Q_cols]
        P_csc = self.P_pat.copy()
        P_csc.data = P[self.P_rows, self.P_cols]

        self.prob = osqp.OSQP()
        self.prob.setup(Q_csc, q, P_csc, l, u, verbose=False)
