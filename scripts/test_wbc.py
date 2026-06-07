import numpy as np
import mujoco
import mujoco.viewer
import imageio

from src.wbc import WBC

SAVING = True
z_des = 0.3

model = mujoco.MjModel.from_xml_path("assets/go2/scene.xml")
data = mujoco.MjData(model)
wbc = WBC(model)

v_des = np.array([0.25, 0.0, 0.0])
period = 0.5

if SAVING:
    fps = 60
    video_writer = imageio.get_writer("go2_trot_2.mp4", fps=fps)
    renderer = mujoco.Renderer(model, height=480, width=640)
    offscreen_cam = mujoco.MjvCamera()
    offscreen_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    offscreen_cam.trackbodyid = model.body("base_link").id
    offscreen_cam.distance = 1.5  # 1.25
    offscreen_cam.elevation = -15  # -25
    offscreen_cam.azimuth = -25  # 55
    frames_recorded = 0

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
        phase = (t % period) / period
        contact_mask = wbc.trot_contact_mask(t, period)
        q_des = np.array([0.0, 0.9, -1.8] * 4)

        # swing traj generator
        for i, in_contact in enumerate(contact_mask):
            if not in_contact:
                if i in (0, 3):
                    swing_phase = np.clip((phase - 0.5) / 0.5, 0.0, 1.0)
                else:
                    swing_phase = np.clip(phase / 0.5, 0.0, 1.0)
                s = np.sin(np.pi * swing_phase)
                q_des[3 * i + 1] = 0.9 + 0.3 * s  # thigh joint
                q_des[3 * i + 2] = -1.8 - 0.4 * s  # calf joint

        tasks = [
            {"type": "base_height", "z_des": z_des, "kp": 500, "kd": 50, "w": 1.0},
            {"type": "base_orientation", "kp": 500, "kd": 50, "w": 1.0},
            {"type": "base_linear_vel", "v_des": v_des, "kp": 10, "w": 10.0},
            {
                "type": "posture",
                "q_des": q_des,
                "kp": 200,
                "kd": 20,
                "w": 0.001,
            },
        ]

        for i, in_contact in enumerate(contact_mask):
            if not in_contact:
                if i in (0, 3):
                    swing_phase = np.clip((phase - 0.5) / 0.5, 0.0, 1.0)
                else:
                    swing_phase = np.clip(phase / 0.5, 0.0, 1.0)

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

        result = wbc.compute_qp(data, contact_mask, tasks)
        if result is not None:
            lam = result[wbc.nv : wbc.nv + wbc.n_lambda]
            tau = result[wbc.nv + wbc.n_lambda :]
            print(
                f"lambda_max={np.abs(lam).max():.1f}  tau_max={np.abs(tau).max():.4f}"
            )

            data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.sync()

        if SAVING:
            if frames_recorded < data.time * fps:
                renderer.update_scene(data, camera=offscreen_cam)
                frame = renderer.render()
                video_writer.append_data(frame)
                frames_recorded += 1

if SAVING:
    print("Saving video file...")
    video_writer.close()
    print("Saved as go2_trot.mp4 successfully!")

# print(model.actuator_gaintype)
# print(model.actuator_biastype)
