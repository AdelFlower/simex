# Robot API and initial capabilities

Your script receives `robot` and `tb`. Perceive the scene through RGBD only.
No object truth, attachments, simulator imports, I/O, introspection, dunder access
or eval/exec. Imports: numpy, scipy, math, collections, itertools, statistics.
Do not import toolbox or add an `if __name__` block.

World meters, +Z up, radians, quaternion `[w,x,y,z]`. TCP between fingertips;
hand +Z points through fingers;
`[0,1,0,0]` points down; opening axis is hand local Y. Reset starts open and empty.

Public robot API:
- `observe('overhead'|'wrist')`: rgb uint8 H×W×3, metric depth H×W, intrinsics 3×3,
  camera_to_world 4×4. Camera axes: right, down, forward.
- `state()`: joints, joint velocities, ee_position, ee_quaternion, gripper_width,
  sim_time. `ik(position, quaternion)` returns success/joints/error without motion.
- `move_joints(joints, duration=1.0)`, `move_ee(position, quaternion, duration=1.0)`.
- `gripper(width, force=40., duration=.5)`: total opening 0..0.08 m; force is a
  per-finger limit in newtons, not measured contact force. `wait(seconds)` settles.

Use `tb.` for helpers below; full signatures/defaults are in toolbox.py:
- `points(observation, mask)` -> world N×3; `color_mask(observation, color)` ->
  largest green/red/tote component. `locate(color)` -> visible median/bounds.
- `outline_points(observation, mask)` includes occluder edges.
  `fit_plane(points)` estimates a plane; `fit_circle(points, radius_range, plane)`
  rejects straight edges. Caller supplies radius bounds and optionally a surface
  plane for an occluded outline. Check rms, inlier_ratio and coverage (angular
  span / 2π); short arcs are uncertain. Degeneracy returns success=False.
- `measure(color)` gives bounds/count/pixel_m/image_edge. Sampling is not accuracy.
- `inspect(color, ...)` obtains a closer downward wrist view. Requires open empty
  hand, retains orientation, checks IK/tracking/budgets, never commands gripper.
  Optional when existing RGBD supports contact; failed inspection can yield useful views.
- `visibility(observation, world_points)` -> free before surface,
  occupied near it, unknown behind/outside valid depth; samples are not volumes.

Motion:
- `move_checked(position, quaternion, ..., accept_position_tolerance=None)`
  pursues 3 mm / 0.05 rad tracking around an absolute target; max_correction bounds
  total offset and guard approves every command.
  None keeps strict acceptance; wider acceptance allows residual after correction
  stops. Check status (reached/accepted/failed), reason, termination_reason and
  attempts with measured errors/API flags.
  Ordinary API tracking failure can have acceptable TCP. Unreachable, joint
  clipping, invalid pose and guard rejection remain failures. Position acceptance
  never relaxes orientation. TCP success proves neither clearance nor grasping.
- `move_path(positions, quaternion, **tracking)` copies waypoints and stops on
  failure; returns success/completed/results. `move_segment(position, quaternion,
  mode='free', **tracking)` freezes endpoints with 20 mm steps, 0.6 s per waypoint.
  Defaults: 3 mm desired, 15 mm accepted,
  0.05 rad orientation, two corrections, 25 mm total offset. Contact/holding modes
  always disable compensation. Final alignment must tighten acceptance explicitly.
- `open_gripper(width=.08, force=40., duration=.5)` and
  `close_gripper(width=0., force=40., duration=.5)` expose physical parameters.

This resettable simulation permits bounded contact trials from imperfect geometry.
Unknown rays, occlusion and coarse pixels are not global vetoes. Unknown is neither
collision nor certified free; keep observed obstacles and use measured feedback.
Use bounded feedback in free transit, never for contact/holding/blockage.
Stop/replan on actual obstruction or failed tracking.

Initial helpers have no validated grasp/transport/release strategy. Discover these
experimentally. Closing/width alone does not prove holding; observe whether the
object follows a lift. Held is intermediate when placement is required.
Document preconditions, postconditions and recovery.
Independent evaluation decides success; confidence gates must not redefine it.
Log actions/outcomes. Budget 120 simulated seconds including corrections.

Generate one text-only response, no tools. Exact blocks:
===PLAN START===
plan of at most 200 words
===PLAN END===
===SCRIPT START===
complete executable Python; robot and tb already exist
===SCRIPT END===

No training-scene coordinates or retained policies. Persist skills only in
toolbox.py/skill.md; generate each trial policy afresh.
