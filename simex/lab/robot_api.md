# Public RobotAPI

Policy globals are `robot` and `tb = Toolbox(robot)`. Source and target expose the
same interface. Positions are world metres; quaternions are `[w,x,y,z]`.

- `robot.observe(camera='overhead')`: RGB uint8 `(480,640,3)`, depth metres
  `(480,640)`, intrinsics `(3,3)`, camera_to_world `(4,4)`. Cameras: overhead, wrist.
- `robot.state()`: measured joints, gripper width, TCP pose, simulator time.
- `robot.ik(position, quaternion)`: public kinematic solution; no motion.
- `robot.move_joints(joints, duration=1.0)`: seven arm joint targets.
- `robot.move_ee(position, quaternion, duration=1.0)`: Cartesian pose target.
- `robot.gripper(width, force=40.0, duration=0.5)`: opening in metres, 0..0.08.
- `robot.wait(seconds)`: advance time, 0..15 seconds per call.

Motion durations are 0.01..15 seconds. Inspect returned feedback; a command
returning does not establish a grasp or task success. Policies may import numpy,
scipy, math, collections, itertools, statistics. They cannot access files,
network, simulator internals, object ground truth, contacts or scoring code.
The host separately scores each declared predicate. Complete placement includes
opening, withdrawal, separation and stable full containment after release.

Skill documentation must name interfaces, preconditions, postconditions, stopping
conditions, observed failure modes, bounded recovery, evidence IDs and validation
scope. Distinguish attempted action, transient lift, maintained hold, transport,
and complete released placement. Keep notes and execution archives out of the
transferred capability (`toolbox.py` and `skill.md` only).
