import numpy as np  # noqa: I001
import mujoco
import mujoco.viewer

from src.robot_state import RobotState
from src.gait import GaitParams, GaitScheduler
from src.foot_planner import FootPlanner
from src.tasks import (
    BaseHeightTask,
    BaseOrientationTask,
    BaseLinearVelocityTask,
    PostureTask,
    SwingFootTask,
)
from src.wbc import WBC


model = mujoco.MjModel.from_xml_path("assets/go2/scene.xml")
data = mujoco.MjData(model)

trot = GaitParams(
    period=0.25, duty_factor=[0.5, 0.5, 0.5, 0.5], phase_offset=[0.0, 0.5, 0.5, 0.0]
)
pronk = GaitParams(
    period=0.25, duty_factor=[0.5, 0.5, 0.5, 0.5], phase_offset=[0.0, 0.0, 0.0, 0.0]
)

z_des = 0.51
v_des = np.array([0.5, 0.0, 0.0])
q_nominal = np.array([0.0, 0.9, -1.8] * 4)

step_count = 0
fail_count = 0

state = RobotState(model)
sched = GaitScheduler(pronk)
planner = FootPlanner(sched)
wbc = WBC(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    trunk_id = model.body("base_link").id
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = trunk_id
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -25
    viewer.cam.azimuth = 55

    while viewer.is_running():
        # gait scheduler
        t = data.time
        state.update(data)

        contact_mask = []
        for i in range(4):
            commanded = sched.in_stance(i, t)
            swing_phase = sched.swing_phase(i, t)

            # a foot commanded to swing that touches down in the descent half gets bumped to stance
            early_touchdown = (
                (not commanded) and state.feet_in_contact[i] and swing_phase > 0.5
            )
            contact_mask.append(commanded or early_touchdown)

        tasks = [
            BaseHeightTask(z_des=z_des, kp=500, kd=50, w=1.0),
            BaseOrientationTask(kp=500, kd=50, w=1.0),
            BaseLinearVelocityTask(v_des=v_des, kp=10, w=10.0),
            PostureTask(q_des=q_nominal, kp=200, kd=20, w=0.001),
        ]

        for i in range(4):
            if not contact_mask[i]:
                swing_phase = sched.swing_phase(i, t)
                p_des, v_des_foot = planner.compute_swing_p_des(
                    state, i, v_des, swing_phase
                )

                tasks.append(
                    SwingFootTask(
                        foot_idx=i, p_des=p_des, v_des=v_des_foot, kp=400, kd=40, w=10.0
                    )
                )

        result = wbc.compute_qp(state, contact_mask, tasks)
        if result is None:
            fail_count += 1
        else:
            lam = result.x[wbc.nv : wbc.nv + wbc.n_lambda]
            tau = result.x[wbc.nv + wbc.n_lambda :].copy()
            print(
                f"lambda_max={np.abs(lam).max():.1f}  tau_max={np.abs(tau).max():.4f}"
            )
            data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.sync()
        step_count += 1
