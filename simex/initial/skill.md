# Robot API and initial capability notes

Your script receives `robot` (RobotAPI) and `tb` (Toolbox) as global variables.
Use NumPy, math and scipy freely. Do not read files, use network/process APIs,
import physics engines or access object truth. All object localization must use RGBD.
There is no attachment, teleport, weld or object-pose control API.

The runtime accepts imports from numpy, scipy, math, collections, itertools and
statistics only. No getattr/setattr, globals/locals/vars, eval/exec/compile, dunder
attribute access or numerical file I/O. Use dictionary `.get` for optional fields.
Do not import toolbox or use an `if __name__` block: robot and tb already exist.
Call `tb.open_gripper()` / `tb.close_gripper()` or `robot.gripper(width)`.

Coordinates: world +Z up, meters, radians; robot base at world origin on table.
Robot is Franka Panda with 7 arm joints and a parallel gripper. Quaternions are
scalar-first `[w,x,y,z]`. TCP is the center between fingertips, with hand +Z
pointing out through the fingers; `[0,1,0,0]` points hand +Z downward. The finger
opening axis is hand local Y (verify via camera feedback and measured motions).

Available methods:
- `robot.observe('overhead'|'wrist')` returns RGB uint8 HxWx3, metric depth HxW,
  `intrinsics` 3x3, `camera_to_world` 4x4. Camera coordinates are OpenCV:
  +X right, +Y down, +Z forward.
- `robot.state()` returns joints, joint velocities, EE pose, gripper width, sim time.
- `robot.ik(position, quaternion)` returns success, joints, error; no motion.
- `robot.move_joints(joints, duration=1.0)` interpolates physical joint targets.
- `robot.move_ee(position, quaternion, duration=1.0)` uses IK then physical movement.
- `robot.gripper(width, force=40.0, duration=0.5)` commands total opening, 0..0.08m.
  `force` is the maximum actuator force per finger in newtons, in both engines;
  it is a limit, not a measurement or guarantee of object contact force.
- `robot.wait(seconds)` advances simulation; use it for settling, not `time.sleep`.
- `tb.locate('green'|'red'|'tote')` estimates visible pointcloud geometry by color.

Thin flat plates have raised rims and rest on the table. The box has a floor and
walls. Initial helpers have no validated grasp or transport strategy. Find and
verify strategies experimentally. Motion commands can report failure; observe
actual outcomes and check `success`. Task time limit is 120 simulated seconds.

Policy generation is a single text-only response. Output exact blocks:
===PLAN START===
plan of at most 200 words
===PLAN END===
===SCRIPT START===
complete executable Python; `robot` and `tb` already exist
===SCRIPT END===

Do not call tools during policy generation. Do not query simulator state or encode
known object coordinates from training scenes. Persistent knowledge must remain
in toolbox.py and this skill.md, not in a retained task-specific script.
