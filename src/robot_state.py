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
        self.J_feet = self.get_J_feet(data)
        self.Jdot_feet = self.get_Jdot_feet(
            data,
        )
        self.M, self.bias = self.get_dynamics(data)

        self.p_feet = data.xpos[self.foot_body_ids].copy()
        self.feet_in_contact = self.get_feet_in_contact(data)

    def get_J_feet(data):
        raise NotImplementedError

    def get_Jdot_feet(data):
        raise NotImplementedError

    def get_dynamics(data):
        raise NotImplementedError

    def get_feet_in_contact(data):
        raise NotImplementedError
