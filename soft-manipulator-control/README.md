# Computed-Torque Control of an Underactuated Soft Robotic Manipulator

Robot Dynamics & Control coursework — a 2-DOF flexible-joint manipulator that presses against a moving wall and draws a sinusoidal trajectory, combining trajectory tracking with compliant physical interaction.

## The problem

The system is underactuated: each joint's motor connects to its link through a passive spring, so motor torque doesn't map directly to link motion. Controlling this well — and proving the controller is actually stable — requires more than a standard rigid-body approach.

## Approach

- Applied **singular perturbation reduction**: in the limit of high joint stiffness, the flexible-joint dynamics collapse to a rigid-body model augmented with motor inertia, which is what the controller is designed against (with the full flexible-joint model used for verification).
- Designed a **computed-torque controller** achieving exact feedback linearisation, tuned to critical damping (ζ = 1, ωₙ = 10 rad/s).
- Proved **global asymptotic stability** for the reduced model via a Lyapunov function and LaSalle's Invariance Principle, and bounded the tracking-error perturbation introduced by the (ignored) spring dynamics in the full model.
- Used the gripper's inherent spring compliance as an implicit impedance/interaction-control strategy: the trajectory is commanded slightly *past* the wall surface, and the spring deflection absorbs contact force elastically (similar in spirit to series elastic actuation).

## Results

- Tight end-effector trajectory tracking with no overshoot (critically damped response, as designed).
- Sustained wall contact throughout a 30-second, 15-cycle run, with spring deflections staying small (as the singular-perturbation assumption requires).
- Identified and analysed the manipulator's two kinematic singularities (fully extended / fully folded) and proposed damped-least-squares Jacobian inversion as a practical avoidance strategy.
- A follow-on parameter study showed that increasing the weaker joint's spring stiffness (matching it to the stronger joint) would reduce spring deflection ~3x and improve the model's accuracy, at the cost of needing higher control gains.

## Stack

Python, Drake (simulation), MeshCat (visualization), computed-torque/feedback-linearisation control, Lyapunov stability analysis.
