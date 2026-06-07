observations:
[0:12]   joint positions   (FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf)
[12:24]  joint velocities  (same order)
[24:36]  joint accels      (same order)
[36:40]  imu quaternion    (w, x, y, z)
[40:43]  imu gyro          (wx, wy, wz)
[43:46]  imu acc           (ax, ay, az)
[46:50]  foot contacts     (FL, FR, RL, RR)


actions:
[0:12]   actuator commands (FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf)

qpos[0:3] -- base position (x, y, z)
qpos[3:7] -- base quaternion
qpos[7:] -- joint angles in order FL, FR, RL, RR
qvel[0:3] -- base linear velocity
qvel[3:6] -- base angular velocity
qvel[6:] -- joint velocities same order
xpos[1] -- base_link position in world frame
xpos[5] -- FL foot position
xpos[9] -- FR foot
xpos[13] -- RL foot
xpos[17] -- RR foot
cfrc_ext -- external forces on each body, shape (18, 6), each row is [torque_x, torque_y, torque_z, force_x, force_y, force_z]

cfrc_ext[5], cfrc_ext[9], cfrc_ext[13], cfrc_ext[17] for foot contact forces instead of binary contacts.