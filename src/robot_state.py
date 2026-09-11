import mujoco
import numpy as np

EPS = np.finfo(np.float64).eps


class RobotState:
    def __init__(self, model):
        self.model = model

        self.nv = model.nv

        self.foot_body_ids = [
            model.body(name).id for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ]
        self.thigh_body_ids = [
            model.body(name).id
            for name in ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        ]
        self.base_body_id = model.body("base_link").id

        # t, J_feet for finite diff cache
        self.J_feet_prev = None
        self.t_prev = None

    def update(self, data):
        self.data = data
        self.J_feet = self.get_jacobian(data, self.foot_body_ids, include_rot=False)
        self.Jdot_feet = self.get_Jdot_feet(data)
        self.M, self.bias = self.get_dynamics(data)

        self.p_feet = data.xpos[self.foot_body_ids].copy()
        self.feet_in_contact = self.get_feet_in_contact(data)

    def get_jacobian(self, data, body_ids, include_rot=True):
        dofs = 6 if include_rot else 3

        J = np.zeros((len(body_ids), dofs, self.nv))
        for i, body_id in enumerate(body_ids):
            J_pos = np.zeros((3, self.nv))
            J_rot = np.zeros((3, self.nv)) if include_rot else None

            mujoco.mj_jac(self.model, data, J_pos, J_rot, data.xpos[body_id], body_id)
            J[i] = np.r_[J_pos, J_rot] if include_rot else J_pos

        return J

    def get_Jdot_feet(self, data):
        J = self.J_feet

        if self.J_feet_prev is None:
            Jdot = np.zeros_like(J)
        else:
            dt = data.time - self.t_prev
            Jdot = (J - self.J_feet_prev) / dt if dt > EPS else np.zeros_like(J)

        self.J_feet_prev = J.copy()
        self.t_prev = data.time
        return Jdot

    def get_dynamics(self, data):
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, data.qM)
        bias = data.qfrc_bias
        return M, bias

    def get_feet_in_contact(self, data):
        measured = [0, 0, 0, 0]
        for k in range(data.ncon):
            con = data.contact[k]
            for geom in (con.geom1, con.geom2):
                body = self.model.geom_bodyid[geom]
                if body in self.foot_body_ids:
                    measured[self.foot_body_ids.index(body)] = True
        return measured
