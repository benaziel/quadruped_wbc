import numpy as np
import mujoco
import mujoco.viewer
import imageio

from src.wbc import WBC

SAVING = False
VIDEO_PATH = "go2_trot.mp4"

z_des = 0.3
v_des = np.array([0.5, 0.0, 0.0])
period = 0.25
q_nominal = np.array([0.0, 0.9, -1.8] * 4)

model = mujoco.MjModel.from_xml_path("assets/go2/scene.xml")
data = mujoco.MjData(model)
wbc = WBC(model)

if SAVING:
    fps = 60
    video_writer = imageio.get_writer(VIDEO_PATH, fps=fps)
    renderer = mujoco.Renderer(model, height=480, width=640)
    offscreen_cam = mujoco.MjvCamera()
    offscreen_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    offscreen_cam.trackbodyid = model.body("base_link").id
    offscreen_cam.distance = 2.0
    offscreen_cam.elevation = -25
    offscreen_cam.azimuth = 55
    frames_recorded = 0

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
fail_count = 0
step_count = 0

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

        commanded = wbc.trot_contact_mask(t, period)
        measured = wbc.feet_in_contact(data)
        contact_mask = []
        for i in range(4):
            swing_phase = wbc.swing_phase(i, t, period)

            # a foot commanded to swing that touches down in the descent half gets bumped to stance
            early_touchdown = (not commanded[i]) and measured[i] and swing_phase > 0.5
            contact_mask.append(bool(commanded[i]) or early_touchdown)

        J_feet, Jdot_feet = wbc.update_foot_kinematics(data)

        tasks = [
            {"type": "base_height", "z_des": z_des, "kp": 500, "kd": 50, "w": 1.0},
            {"type": "base_orientation", "kp": 500, "kd": 50, "w": 1.0},
            {"type": "base_linear_vel", "v_des": v_des, "kp": 10, "w": 10.0},
            {
                "type": "posture",
                "q_des": q_nominal,
                "kp": 200,
                "kd": 20,
                "w": 0.001,
            },
        ]

        # swing traj generator
        for i in range(4):
            if not contact_mask[i]:
                swing_phase = wbc.swing_phase(i, t, period)
                p_des, v_des_foot = wbc.compute_swing_p_des(
                    data, i, v_des, swing_phase, period=period
                )

                tasks.append({
                    "type": "swing_foot",
                    "foot_idx": i,
                    "p_des": p_des,
                    "v_des": v_des_foot,
                    "kp": 400,
                    "kd": 40,
                    "w": 10.0,
                })

        result = wbc.compute_qp(data, contact_mask, tasks, J_feet, Jdot_feet)
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

        if SAVING and frames_recorded < data.time * fps:
            renderer.update_scene(data, camera=offscreen_cam)
            frame = renderer.render()
            video_writer.append_data(frame)
            frames_recorded += 1

if SAVING:
    video_writer.close()
    renderer.close()
    print(f"Saved as {VIDEO_PATH} successfully!")

# print(model.actuator_gaintype)
# print(model.actuator_biastype)
