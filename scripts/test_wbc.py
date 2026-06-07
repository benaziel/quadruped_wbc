import numpy as np
import mujoco
import mujoco.viewer

from src.wbc import WBC

z_des = 0.27

model = mujoco.MjModel.from_xml_path("assets/go2/scene.xml")
data = mujoco.MjData(model)
wbc = WBC(model)

v_des = np.array([0.5, 0.0, 0.0])
period = 0.5

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = data.time
        phase = (t % period) / period
        contact_mask = wbc.trot_contact_mask(t, period)

        q_des = np.array([0.0, 0.9, -1.8] * 4)
        for i, in_contact in enumerate(contact_mask):
            if not in_contact:
                if i in (0, 3):
                    swing_phase = np.clip((phase - 0.5) / 0.5, 0.0, 1.0)
                else:
                    swing_phase = np.clip(phase / 0.5, 0.0, 1.0)
                s = np.sin(np.pi * swing_phase)
                q_des[3 * i + 1] = 0.9 + 0.3 * s
                q_des[3 * i + 2] = -1.8 - 0.4 * s

        tasks = [
            {"type": "base_height", "z_des": z_des, "kp": 500, "kd": 50, "w": 1.0},
            {"type": "base_orientation", "kp": 500, "kd": 50, "w": 1.0},
            {"type": "base_linear_vel", "v_des": v_des, "kp": 5, "w": 0.1},
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

                tasks.append(
                    {
                        "type": "swing_foot",
                        "foot_idx": i,
                        "p_des": p_des,
                        "v_des": v_des_foot,
                        "kp": 400,
                        "kd": 40,
                        "w": 10.0,
                    }
                )

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

# print(model.actuator_gaintype)
# print(model.actuator_biastype)
