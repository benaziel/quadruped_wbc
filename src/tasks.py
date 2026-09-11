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
        raise NotImplementedError
