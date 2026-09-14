import numpy as np
import osqp
from scipy.linalg import block_diag
import scipy.sparse as sp


class WBC:
    def __init__(self, model, mu=0.6):
        self.model = model

        self.nv = model.nv
        self.n_lambda = 12  # 3 * 4 feet
        self.n_tau = 12  # actuated joints, i don't think we actuate the torso
        self.n = self.nv + self.n_lambda + self.n_tau

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

    def friction_cone(self, mu):
        return np.array([
            [1, 0, -mu],
            [-1, 0, -mu],
            [0, 1, -mu],
            [0, -1, -mu],
            [0, 0, -1],
        ])

    def compute_qp(self, state, contact_mask, tasks):
        J_c_T = state.J_feet.reshape(self.n_lambda, self.nv).T
        M, bias = state.M, state.bias

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
            J, e = task.compute(state)
            Q[: self.nv, : self.nv] += task.w * J.T @ J
            q[: self.nv] += -task.w * J.T @ e

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
