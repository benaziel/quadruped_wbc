# Go2 Quadruped Whole-Body Control in MuJoCo

Multi-task hierarchical optimization loop using a master gait clock to schedule contact transitions while OSQP solves for instantaneous joint torques.


At a high level, there's open-loop clock scheduling tracking a diagonal trot gait with sinusoidal leg-swing clearance curves and a Raibert-based foot placement heuristic. Further down there's a multi-objective QP tracking base position, spatial orientation, forward velocity, and null-space joint posture.

The decision vector is

$$
x = \begin{bmatrix} \ddot{q} \\\\ \lambda \\\\ \tau \end{bmatrix} \in \mathbb{R}^{42}.
$$

where $\ddot{q} \in \mathbb{R}^{18}$ is generalized acceleration (base contributes six and the twelve actuated joints contribute the rest), $\lambda \in \mathbb{R}^{12}$ represents contact forces (four feet, each with a 3D reaction force), and $\tau \in \mathbb{R}^{12}$ represents joint torques for the twelve actuated joints.

The three blocks are coupled through the floating-base EoM:
$$
M(q) \ddot{q} + C(q, \dot{q}) = S^\top \tau + J_c^\top \lambda
$$

which enters the QP as an equality constraint (alongside the friction cone inequality constraints and actuator limit box constraints).

Because a WBC is a snapshot optimizer, i.e., it's got a horizon of one timestep, it has no recursive feasibility. It has no problems commanding massive forces/torques, so managing these floating-base dynamics requires a more long-term strategy. I'm also just relying on Raibert (which is effectively a geometric hack) to tell the swing legs where to land, and while it works for steady-state trotting at a single tuned speed, it's probably worthwhile to add a high-level MPC layer that looks ahead over a finite horizon to optimize a low-dimensional model of the robot's centroidal dynamics.

`src.wbc.py` houses the `WBC` class, which computes mass matrices, handles Cartesian spatial tracking errors using MuJoCo's quaternion math, maps operational space objectives to generalized coordinates, and executes the optimization pipeline.

`scripts/test_wbc.py` configures target commands, updates swing-phase timelines, and pass control outputs into MuJoCo.

