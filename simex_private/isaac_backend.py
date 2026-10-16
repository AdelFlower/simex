"""Trusted Isaac Sim 4.5 backend. Import only from the simulator worker.

Nothing here is exposed to generated policies. Plate poses and target physics
are available exclusively through ``metrics`` to the trusted evaluator.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import numpy as np

from simex.tasks import JOINT_LIMITS, validate_scene


def _rotation(quaternion):
    from scipy.spatial.transform import Rotation
    q = np.asarray(quaternion, dtype=float)
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def _quaternion(rotation):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(rotation).as_quat()
    return q[[3, 0, 1, 2]]


class IsaacBackend:
    """Contact-only Panda manipulation with optical-frame RGBD cameras."""

    TCP_OFFSET = np.array([0.0, 0.0, 0.1034])
    LIMITS = JOINT_LIMITS

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # SimulationApp MUST be initialized before importing any omni/Isaac API.
        from isaacsim import SimulationApp
        self.app = SimulationApp({
            "headless": self.config.get("headless", True),
            "width": 320, "height": 240,
            "renderer": "RayTracedLighting",
            "anti_aliasing": 0,
            "multi_gpu": False,
            "disable_viewport_updates": True,
            "sync_loads": True,
        })
        import carb
        carb.settings.get_settings().set("/rtx/post/dlss/execMode", 0)
        carb.settings.get_settings().set("/rtx/sceneDb/ambientLightIntensity", 0.5)
        self.world = None
        self.on_step = None
        self.steps = 0
        self.dt = 1 / 240
        self._next_record = 0.25
        self._plates = {}
        self._cameras = {}
        self._initial = {}
        self._maximum = {}
        self._in_callback = False
        self._closed = False

    def _cube(self, path, position, size, color, parent=None):
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
        cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        collision.CreateContactOffsetAttr(.001)
        collision.CreateRestOffsetAttr(0.0)
        return cube.GetPrim()

    def _material(self, path, friction):
        from pxr import UsdShade, UsdPhysics
        material = UsdShade.Material.Define(self.stage, path)
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(float(friction))
        physics.CreateDynamicFrictionAttr(float(friction) * 0.85)
        physics.CreateRestitutionAttr(0.0)
        return material

    @staticmethod
    def _bind_material(prim, material):
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    def reset(self, scene: dict) -> dict:
        scene = validate_scene(scene)
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import create_new_stage
        from isaacsim.robot.manipulators.examples.franka import Franka
        from isaacsim.robot.manipulators.examples.franka.kinematics_solver import KinematicsSolver
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics

        self.scene = copy.deepcopy(scene)
        self.dt = float(scene.get("physics", {}).get("dt", 1 / 240))
        self.on_step_saved = self.on_step
        self.on_step = None
        self._cameras.clear()
        self._plates.clear()
        self._initial.clear()
        self._maximum.clear()
        if self.world is not None:
            self.world.stop()
            self.world.clear()
            World.clear_instance()
        create_new_stage()
        self.world = World(stage_units_in_meters=1.0, physics_dt=self.dt, rendering_dt=1 / 30)
        self.stage = self.world.stage
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        self.world.get_physics_context().enable_ccd(True)
        self.world.get_physics_context().set_solver_type("TGS")
        table = scene.get("table", {})
        self._cube("/World/Table", table.get("center", [0.4, 0, -0.025]),
                   table.get("size", [1.2, 1.1, 0.05]), [0.48, 0.43, 0.36])
        floor_material = self._material("/World/TableMaterial", 0.7)
        self._bind_material(self.stage.GetPrimAtPath("/World/Table"), floor_material)

        robot_cfg = scene.get("robot", {})
        usd_path = self.config.get("usd_path", self.config.get("asset_path", os.environ.get(
            "SIMEX_FRANKA_USD", "/home/kemove/DexGarmentLab/Assets/Robots/franka/franka.usd")))
        if not Path(usd_path).is_file():
            raise FileNotFoundError(f"Local Franka USD missing: {usd_path}")
        self.robot = self.world.scene.add(Franka(
            prim_path="/World/Panda", name="panda", usd_path=usd_path,
            position=np.asarray(robot_cfg.get("base_position", [0, 0, 0]), dtype=float),
            end_effector_prim_name="panda_hand",
            gripper_open_position=np.array([0.04, 0.04]),
            gripper_closed_position=np.zeros(2),
        ))
        self._hand_path = "/World/Panda/panda_hand"
        for plate in scene.get("plates", []):
            name = plate["name"]
            path = f"/World/Plates/{name}"
            body = UsdGeom.Xform.Define(self.stage, path)
            body.AddTranslateOp().Set(Gf.Vec3d(*map(float, plate["position"])))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            physx = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
            physx.CreateEnableCCDAttr(True)
            physx.CreateSolverPositionIterationCountAttr(16)
            physx.CreateSolverVelocityIterationCountAttr(4)
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(float(plate.get("mass", 0.15)))
            radius, thickness = float(plate.get("radius", .07)), float(plate.get("thickness", .006))
            color = plate.get("color", [.1, .75, .15])
            cylinder = UsdGeom.Cylinder.Define(self.stage, path + "/disk")
            cylinder.CreateAxisAttr("Z")
            cylinder.CreateRadiusAttr(radius)
            cylinder.CreateHeightAttr(thickness)
            cylinder.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
            UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
            collision = PhysxSchema.PhysxCollisionAPI.Apply(cylinder.GetPrim())
            collision.CreateContactOffsetAttr(.001)
            collision.CreateRestOffsetAttr(0.0)
            # Convex segments of the raised rim are all children of ONE rigid body.
            rim_height = float(plate.get("rim_height", .012))
            rim_thickness = .004
            for segment in range(24):
                angle = 2 * math.pi * segment / 24
                prim = self._cube(
                    f"{path}/rim_{segment}",
                    [(radius - rim_thickness / 2) * math.cos(angle),
                     (radius - rim_thickness / 2) * math.sin(angle),
                     thickness / 2 + rim_height / 2],
                    [rim_thickness, 2 * radius * math.sin(math.pi / 24) + .001, rim_height], color)
                xform = UsdGeom.Xformable(prim)
                xform.AddRotateZOp().Set(math.degrees(angle))
                ops = xform.GetOrderedXformOps()
                xform.SetXformOpOrder([ops[0], ops[2], ops[1]])
            self._bind_material(body.GetPrim(), self._material(
                f"/World/Materials/{name}", plate.get("friction", .6)))
            self._plates[name] = self.world.scene.add(SingleRigidPrim(prim_path=path, name=name))

        tote = scene.get("tote", {})
        tote_position = tote.get("position", [.4, -.26, 0])
        sx, sy, sz = tote.get("inner_size", [.26, .22, .10])
        wall = float(tote.get("wall_thickness", .008))
        tote_root = UsdGeom.Xform.Define(self.stage, "/World/Tote")
        tote_root.AddTranslateOp().Set(Gf.Vec3d(*map(float, tote_position)))
        tote_root.AddRotateZOp().Set(math.degrees(float(tote.get("yaw", 0))))
        self._cube("/World/Tote/floor", [0, 0, wall / 2], [sx + 2*wall, sy + 2*wall, wall], [.2, .3, .48])
        for sign in [-1, 1]:
            self._cube(f"/World/Tote/x{sign + 1}", [sign*(sx+wall)/2, 0, (sz+wall)/2],
                       [wall, sy + 2*wall, sz + wall], [.25, .36, .58])
            self._cube(f"/World/Tote/y{sign + 1}", [0, sign*(sy+wall)/2, (sz+wall)/2],
                       [sx, wall, sz + wall], [.25, .36, .58])
        self._bind_material(tote_root.GetPrim(), floor_material)
        light = UsdLux.DomeLight.Define(self.stage, "/World/Light")
        light.CreateIntensityAttr(800)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        home = np.asarray(robot_cfg.get("home", [0, -.5, 0, -2.2, 0, 1.7, .785398]), dtype=float)
        self.world.reset()
        self._arm_indices = np.array([self.robot.get_dof_index(f"panda_joint{i}") for i in range(1, 8)])
        self._finger_indices = np.array([self.robot.get_dof_index(f"panda_finger_joint{i}") for i in [1, 2]])
        self.robot._articulation_view.set_max_efforts(
            np.full((1, 2), 40.0), joint_indices=self._finger_indices)
        self.robot.set_joint_positions(home, joint_indices=self._arm_indices)
        self.robot.set_joint_positions(np.array([.04, .04]), joint_indices=self._finger_indices)
        self._joint_target = self.robot.get_joint_positions().copy()
        self._command(self._joint_target)
        self.kinematics = KinematicsSolver(self.robot, end_effector_frame_name="panda_hand")
        # Lula may finish its native warning with ANSI reset but no newline.
        # End that line before the worker writes its framed JSON response.
        print("", flush=True)
        base, orientation = self.robot.get_world_pose()
        self.kinematics.get_kinematics_solver().set_robot_base_pose(base, orientation)
        self._setup_cameras(scene.get("cameras", {}))
        self.steps = 0
        self._next_record = .25
        self._step(max(1, round(.3 / self.dt)), record=False)
        # Renderer warm-up produces actual RGBD before reset returns.
        for _ in range(5):
            self.world.render()
        for name, body in self._plates.items():
            self._initial[name] = float(body.get_world_pose()[0][2])
            self._maximum[name] = self._initial[name]
        self.steps = 0
        self._next_record = .25
        self.on_step = self.on_step_saved
        return self.state()

    def _setup_cameras(self, cfg):
        from isaacsim.sensors.camera import Camera
        self._camera_width, self._camera_height = int(cfg.get("width", 320)), int(cfg.get("height", 240))
        resolution = (self._camera_width, self._camera_height)
        offset = np.asarray(cfg.get('offset', [0, 0, 0]), dtype=float)
        overhead = Camera("/World/Overhead", name="overhead", resolution=resolution, frequency=None)
        # Optical camera axes: +x world right, +y down, +z towards the table.
        overhead.set_world_pose(np.array([.45, 0., 1.1]) + offset, np.array([0., 1., 0., 0.]), camera_axes="ros")
        wrist = Camera(self._hand_path + "/WristCamera", name="wrist", resolution=resolution, frequency=None)
        wrist.set_local_pose(np.array([.04, 0, .04]) + offset, np.array([1., 0, 0, 0]), camera_axes="ros")
        for name, camera in [("overhead", overhead), ("wrist", wrist)]:
            camera.initialize()
            fovy = 60.0 if name == "overhead" else 65.0
            focal_pixels = self._camera_height / (2 * math.tan(math.radians(fovy / 2)))
            # Camera API uses USD lens units / 10; millimetre-sized values below
            # the renderer's lens minimum silently clamp and corrupt calibration.
            camera.set_focal_length(3.6 * focal_pixels / self._camera_width)
            camera.set_horizontal_aperture(3.6)
            camera.set_vertical_aperture(3.6 * self._camera_height / self._camera_width)
            camera.set_clipping_range(.01, 5.0)
            camera.add_distance_to_image_plane_to_frame()
            self._cameras[name] = camera

    def _command(self, positions):
        from isaacsim.core.utils.types import ArticulationAction
        self.robot.apply_action(ArticulationAction(joint_positions=np.asarray(positions)))

    def _step(self, count, record=True):
        # 30 Hz rendering, 240 Hz physics. Observation never advances simulation.
        stride = max(1, round(1 / (30 * self.dt)))
        for _ in range(count):
            self.world.step(render=(self.steps % stride == 0))
            self.steps += 1
            for name, body in self._plates.items():
                height = float(body.get_world_pose()[0][2])
                self._maximum[name] = max(self._maximum.get(name, height), height)
            if record and self.on_step is not None and self.steps * self.dt + 1e-9 >= self._next_record:
                self._next_record += .25
                if not self._in_callback:
                    self._in_callback = True
                    try:
                        self.world.render()
                        self.on_step()
                    finally:
                        self._in_callback = False

    def state(self) -> dict:
        hand_position, hand_quaternion = self.robot.end_effector.get_world_pose()
        hand_rotation = _rotation(hand_quaternion)
        joints = np.asarray(self.robot.get_joint_positions())
        velocities = np.asarray(self.robot.get_joint_velocities())
        return {
            "joints": joints[self._arm_indices].tolist(),
            "joint_velocities": velocities[self._arm_indices].tolist(),
            "ee_position": (hand_position + hand_rotation @ self.TCP_OFFSET).tolist(),
            "ee_quaternion": _quaternion(hand_rotation).tolist(),
            "gripper_width": float(joints[self._finger_indices].sum()),
            "sim_time": self.steps * self.dt,
        }

    def observe(self, camera="overhead") -> dict:
        if camera not in self._cameras:
            raise ValueError("camera must be 'overhead' or 'wrist'")
        cam = self._cameras[camera]
        # Render-only updates keep image poses synchronized without physics steps.
        self.world.render()
        rgba, depth = cam.get_rgba(), cam.get_depth()
        if rgba is None or depth is None or rgba.size == 0 or depth.size == 0:
            raise RuntimeError(f"Camera {camera} has no RGBD; renderer has not initialized")
        position, quaternion = cam.get_world_pose(camera_axes="ros")
        transform = np.eye(4)
        transform[:3, :3] = _rotation(quaternion)
        transform[:3, 3] = position
        return {"rgb": np.asarray(rgba[:, :, :3], dtype=np.uint8).copy(),
                "depth": np.asarray(depth, dtype=np.float32).copy(),
                "intrinsics": np.asarray(cam.get_intrinsics_matrix(), dtype=np.float64),
                "camera_to_world": transform, "camera": camera}

    def ik(self, position, quaternion) -> dict:
        position, quaternion = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
        with np.errstate(over='ignore', invalid='ignore'):
            norm = np.linalg.norm(quaternion)
        if (position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(quaternion).all() or not np.isfinite(norm) or norm < 1e-8):
            raise ValueError("IK requires finite xyz and a wxyz quaternion with finite norm >= 1e-8")
        quaternion /= norm
        rotation = _rotation(quaternion)
        hand_target = position - rotation @ self.TCP_OFFSET
        action, success = self.kinematics.compute_inverse_kinematics(
            hand_target, quaternion, position_tolerance=.002, orientation_tolerance=.025)
        joints = np.asarray(action.joint_positions, dtype=float)
        actual_position, actual_rotation = self.kinematics.get_kinematics_solver().compute_forward_kinematics("panda_hand", joints)
        error = float(np.linalg.norm(actual_position + actual_rotation @ self.TCP_OFFSET - position))
        orientation_error = float(np.arccos(np.clip((np.trace(rotation.T @ actual_rotation) - 1) / 2, -1, 1)))
        return {"success": bool(success and error < .004 and orientation_error < .04),
                "joints": joints.tolist(), "error": error, "orientation_error": orientation_error}

    def move_joints(self, joints, duration=1.0) -> dict:
        goal = np.asarray(joints, dtype=float)
        if goal.shape != (7,) or not np.isfinite(goal).all():
            raise ValueError("move_joints requires seven finite joint angles")
        goal = np.clip(goal, self.LIMITS[:, 0], self.LIMITS[:, 1])
        count = max(1, math.ceil(self._duration(duration) / self.dt))
        start = self.robot.get_joint_positions()[self._arm_indices].copy()
        for step in range(count):
            t = (step + 1) / count
            blend = t*t*(3 - 2*t)
            self._joint_target[self._arm_indices] = start + blend*(goal-start)
            self._command(self._joint_target)
            self._step(1)
        self._step(max(1, round(.15 / self.dt)))
        result = self.state()
        result["success"] = bool(np.max(np.abs(np.asarray(result["joints"])-goal)) < .08)
        return result

    def move_ee(self, position, quaternion, duration=1.0) -> dict:
        solution = self.ik(position, quaternion)
        if not solution["success"]:
            return {"success": False, "reason": "ik_failed", "ik": solution}
        return self.move_joints(solution["joints"], duration)

    def gripper(self, width, force=40.0, duration=.5) -> dict:
        """Command total opening in meters, with a force cap in N per finger."""
        width, force = float(width), float(force)
        if not math.isfinite(width) or not math.isfinite(force) or force <= 0:
            raise ValueError("gripper width/force must be finite and force must be positive")
        width = float(np.clip(width, 0, .08))
        self.robot._articulation_view.set_max_efforts(
            np.full((1, 2), np.clip(force, .1, 100)), joint_indices=self._finger_indices)
        self._joint_target[self._finger_indices] = width / 2
        self._command(self._joint_target)
        self._step(max(1, math.ceil(self._duration(duration)/self.dt)))
        result = self.state()
        result["success"] = True  # Contact can prevent reaching requested width.
        return result

    @staticmethod
    def _duration(duration):
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0 or duration > 60:
            raise ValueError("duration must be finite and in (0, 60] seconds")
        return duration

    def wait(self, seconds) -> dict:
        if float(seconds) == 0:
            return self.state()
        self._step(max(1, math.ceil(self._duration(seconds)/self.dt)))
        return self.state()

    def metrics(self) -> dict:
        objects = {}
        for name, body in self._plates.items():
            position, quaternion = body.get_world_pose()
            objects[name] = {"position": position.tolist(), "quaternion": quaternion.tolist(),
                             "max_height": self._maximum[name], "initial_height": self._initial[name],
                             "linear_velocity": body.get_linear_velocity().tolist()}
        tote = self.scene.get("tote", {})
        return {"objects": objects,
                "tote": {"position": tote.get("position", [.4, -.26, 0]),
                         "inner_size": tote.get("inner_size", [.26, .22, .10]),
                         "yaw": tote.get("yaw", 0)},
                "sim_time": self.steps * self.dt, "steps": self.steps}

    def close(self):
        if not self._closed:
            self._closed = True
            self.on_step = None
            self._cameras.clear()
            self.app.close()
