"""Physical MuJoCo Panda implementation of the SimEX robot boundary.

Only reset/metrics know scene objects. Motion, cameras and IK depend on robot
state alone. The unmodified, pinned Menagerie assets are composed at reset.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from simex.tasks import validate_scene


HOME = [0.0, -0.5, 0.0, -2.2, 0.0, 1.7, 0.785398]


def _vec(values) -> str:
    return " ".join(str(float(value)) for value in values)


def _finite(value, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return array


class MujocoBackend:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.asset_root = Path(self.config.get("asset_root", Path(__file__).resolve().parents[2] / "assets/panda")).resolve()
        self.model = None
        self.data = None
        self.renderer = None
        self.on_step: Callable[[], None] | None = None
        self._next_callback = 0.25
        self._steps = 0
        self._hooks = None
        self._command = None

    def _scene_xml(self, scene: dict) -> str:
        path = self.asset_root / "panda.xml"
        if not path.exists():
            raise FileNotFoundError(f"Missing pinned Panda assets: {path}. Run python scripts/fetch_panda.py")
        root = ET.parse(path).getroot()
        root.find("compiler").set("meshdir", str(self.asset_root / "assets"))
        keyframe = root.find("keyframe")
        if keyframe is not None:
            root.remove(keyframe)
        # Public ordering matches the Isaac robot, regardless of upstream names.
        for element in root.iter():
            for attribute, value in list(element.attrib.items()):
                if value in {f"joint{i}" for i in range(1, 8)}:
                    element.set(attribute, "panda_" + value)
        # Keep the original coupled tendon, but use a realistic position servo
        # rather than upstream's soft 100 N/m demo spring. The public force
        # argument is a per-finger force cap. The split tendon transmits half
        # its force to each finger; reset/gripper convert the public limit.
        gripper_actuator = root.find("actuator/general[@name='actuator8']")
        gripper_actuator.set("gainprm", _vec([2000 * .04 / 255, 0, 0]))
        gripper_actuator.set("biasprm", "0 -2000 -20")
        physics = scene.get("physics", {})
        if "gripper_stiffness_scale" in physics or "gripper_damping_scale" in physics:
            stiffness = 2000 * physics.get("gripper_stiffness_scale", 1.)
            damping = 20 * physics.get("gripper_damping_scale", 1.)
            gripper_actuator.set("gainprm", _vec([stiffness * .04 / 255, 0, 0]))
            gripper_actuator.set("biasprm", _vec([0, -stiffness, -damping]))
        option = root.find("option")
        option.set("timestep", str(scene.get("physics", {}).get("dt", 1 / 240)))
        option.set("gravity", "0 0 -9.81")
        option.set("cone", "elliptic")
        option.set("iterations", "100")
        option.set("noslip_iterations", "5")
        # Thin flat objects need more than a single cylinder/box support point.
        # Without multi-contact CCD the 6 mm disk rocks and penetrates the table.
        ET.SubElement(option, "flag", multiccd="enable")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth=str(self.width), offheight=str(self.height))
        ET.SubElement(visual, "map", znear="0.005", zfar="10")
        ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7", specular="0.15 0.15 0.15")
        world = root.find("worldbody")
        world.find("body[@name='link0']").set("pos", _vec(scene.get("robot", {}).get("base_position", [0, 0, 0])))
        hand = root.find(".//body[@name='hand']")
        ET.SubElement(hand, "site", name="simex_tcp", pos="0 0 0.1034", size="0.005", rgba="0 0 0 0")
        camera_offset = np.asarray(scene.get("cameras", {}).get("offset", [0, 0, 0]), dtype=float)
        # Camera +Z points backwards in MuJoCo. This quaternion points forwards
        # along the hand's +Z, with the lens clear of the fingers.
        ET.SubElement(hand, "camera", name="wrist", pos=_vec(np.array([0.04, 0, 0.04]) + camera_offset),
                      quat="0 1 0 0", fovy="65")
        ET.SubElement(world, "camera", name="overhead", pos=_vec(np.array([0.45, 0, 1.1]) + camera_offset),
                      quat="1 0 0 0", fovy="60")
        ET.SubElement(world, "light", pos="0.1 -0.5 1.8", dir="0.1 0.3 -1", diffuse="0.8 0.8 0.8")
        table = scene.get("table", {})
        ET.SubElement(world, "geom", name="table", type="box", pos=_vec(table.get("center", [.4, 0, -.025])),
                      size=_vec(np.asarray(table.get("size", [1.2, 1.1, .05])) / 2), rgba="0.56 0.49 0.39 1",
                      friction="0.7 0.005 0.0001")
        # Keep the robot mounted to the same top surface as the task fixtures.
        for plate in scene.get("plates", []):
            name = plate["name"]
            radius = float(plate.get("radius", .07))
            thickness = float(plate.get("thickness", .006))
            rim_height = float(plate.get("rim_height", .012))
            mass = float(plate.get("mass", .15))
            color = list(plate.get("color", [.1, .75, .15]))[:3] + [1]
            friction = _vec([float(plate.get("friction", .6)), .005, .0001])
            body = ET.SubElement(world, "body", name=f"plate_{name}", pos=_vec(plate["position"]))
            ET.SubElement(body, "freejoint", name=f"plate_free_{name}")
            common = dict(rgba=_vec(color), friction=friction, condim="4", solref="0.01 1")
            ET.SubElement(body, "geom", name=f"plate_disk_{name}", type="cylinder", size=_vec([radius, thickness / 2]),
                          mass=str(mass * .75), **common)
            segments, rim_width = 24, min(.004, radius / 8)
            for index in range(segments):
                angle = 2 * math.pi * index / segments
                center_radius = radius - rim_width / 2
                ET.SubElement(body, "geom", type="box",
                              pos=_vec([center_radius * math.cos(angle), center_radius * math.sin(angle), (thickness + rim_height) / 2]),
                              size=_vec([rim_width / 2, center_radius * math.tan(math.pi / segments), rim_height / 2]),
                              euler=_vec([0, 0, angle]), mass=str(mass * .25 / segments), **common)
        tote = scene.get("tote", {})
        tx, ty, tz = tote.get("position", [.4, -.26, 0])
        length, width, height = tote.get("inner_size", [.26, .22, .1])
        wall = float(tote.get("wall_thickness", .008))
        body = ET.SubElement(world, "body", name="tote", pos=_vec([tx, ty, tz]), euler=_vec([0, 0, tote.get("yaw", 0)]))
        pieces = [("floor", [0, 0, wall / 2], [length / 2 + wall, width / 2 + wall, wall / 2]),
                  ("left", [-(length + wall) / 2, 0, wall + height / 2], [wall / 2, width / 2 + wall, height / 2]),
                  ("right", [(length + wall) / 2, 0, wall + height / 2], [wall / 2, width / 2 + wall, height / 2]),
                  ("front", [0, -(width + wall) / 2, wall + height / 2], [length / 2, wall / 2, height / 2]),
                  ("back", [0, (width + wall) / 2, wall + height / 2], [length / 2, wall / 2, height / 2])]
        for name, position, size in pieces:
            ET.SubElement(body, "geom", name=f"tote_{name}", type="box", pos=_vec(position), size=_vec(size),
                          rgba="0.25 0.35 0.52 1", friction="0.6 0.005 0.0001")
        return ET.tostring(root, encoding="unicode")

    def reset(self, scene: dict) -> dict:
        scene = validate_scene(scene, calibrated=True)
        self.close()
        self.scene = copy.deepcopy(scene)
        self.width = int(scene.get("cameras", {}).get("width", 320))
        self.height = int(scene.get("cameras", {}).get("height", 240))
        xml = self._scene_xml(scene)
        if self.config.get('source_model'):
            from simex.source_model import Hooks, validate_xml, SourceModelError
            self._hooks = Hooks(self.config['source_model'])
            xml = validate_xml(xml, self._hooks.call('build_xml', scene, xml))
            try:
                self.model = mujoco.MjModel.from_xml_string(xml)
            except Exception as exc:
                raise SourceModelError(f'Source model does not compile: {exc}') from exc
            if self.model.nu != 8 or not np.isfinite(self.model.body_mass).all():
                raise SourceModelError('Invalid source robot interface or inertia')
        else:
            self.model = mujoco.MjModel.from_xml_string(xml)
        # MuJoCo combines equal-priority geom friction by max. Scale both
        # participants, so every physical sliding contact responds linearly,
        # including plate/table and plate/fingertip contacts below scale 1.
        self.model.geom_friction[:, 0] *= scene['physics']['contact_friction_scale']
        self.model.actuator_forcerange[7] = [-80, 80]  # 40 N per finger.
        self.data = mujoco.MjData(self.model)
        self._joint_ids = np.array([self.model.joint(f"panda_joint{i}").id for i in range(1, 8)])
        self._qpos_indices = self.model.jnt_qposadr[self._joint_ids]
        self._dof_indices = self.model.jnt_dofadr[self._joint_ids]
        self._limits = self.model.jnt_range[self._joint_ids].copy()
        self._finger_indices = [self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        self._tcp_id = self.model.site("simex_tcp").id
        self._ik_data = mujoco.MjData(self.model)
        self._object_ids = {plate["name"]: self.model.body(f"plate_{plate['name']}").id for plate in scene.get("plates", [])}
        home = _finite(scene.get("robot", {}).get("home", HOME), 7, "home")
        self.data.qpos[self._qpos_indices] = np.clip(home, self._limits[:, 0], self._limits[:, 1])
        self.data.qpos[self._finger_indices] = .04
        self.data.ctrl[:7] = self.data.qpos[self._qpos_indices]
        self.data.ctrl[7] = 255
        mujoco.mj_forward(self.model, self.data)
        for _ in range(math.ceil(.3 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        # Kinematics must describe the just-integrated state, also at reset.
        mujoco.mj_forward(self.model, self.data)
        self.data.time = 0.0
        self._steps = 0
        self._next_callback = .25
        self._initial_heights = {name: float(self.data.xpos[body_id, 2]) for name, body_id in self._object_ids.items()}
        self._max_heights = dict(self._initial_heights)
        return self.state()

    def _ready(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("reset(scene) must be called before using the simulator")

    def state(self) -> dict:
        self._ready()
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[self._tcp_id])
        return {"joints": self.data.qpos[self._qpos_indices].tolist(),
                "joint_velocities": self.data.qvel[self._dof_indices].tolist(),
                "ee_position": self.data.site_xpos[self._tcp_id].tolist(),
                "ee_quaternion": quaternion.tolist(),
                "gripper_width": float(np.sum(self.data.qpos[self._finger_indices])),
                "sim_time": float(self.data.time)}

    def observe(self, camera: str = "overhead") -> dict:
        self._ready()
        if camera not in ("overhead", "wrist"):
            raise ValueError("camera must be 'overhead' or 'wrist'")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        depth = self.renderer.render().copy().astype(np.float32)
        self.renderer.disable_depth_rendering()
        camera_id = self.model.camera(camera).id
        focal = self.height / (2 * math.tan(math.radians(self.model.cam_fovy[camera_id]) / 2))
        intrinsics = np.array([[focal, 0, (self.width - 1) / 2], [0, focal, (self.height - 1) / 2], [0, 0, 1]], dtype=float)
        transform = np.eye(4)
        transform[:3, :3] = self.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
        transform[:3, 3] = self.data.cam_xpos[camera_id]
        return {"rgb": rgb, "depth": depth, "intrinsics": intrinsics,
                "camera_to_world": transform, "camera": camera}

    def ik(self, position: list, quaternion: list) -> dict:
        self._ready()
        target = _finite(position, 3, "position")
        target_quat = _finite(quaternion, 4, "quaternion")
        with np.errstate(over='ignore', invalid='ignore'):
            norm = np.linalg.norm(target_quat)
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError("quaternion norm must be finite and at least 1e-8")
        target_quat /= norm
        target_mat = np.empty(9)
        mujoco.mju_quat2Mat(target_mat, target_quat)
        target_mat = target_mat.reshape(3, 3)
        start = self.data.qpos[self._qpos_indices].copy()
        lower, upper = self._limits[:, 0] + 1e-6, self._limits[:, 1] - 1e-6
        jac_position = np.zeros((3, self.model.nv))
        jac_rotation = np.zeros((3, self.model.nv))
        # The solver uses kinematics/Jacobians only. It never simulates contacts
        # or reads plate/tote states, and never modifies the rollout state.
        best = (float("inf"), start.copy(), float("inf"), float("inf"))
        seeds = [start, np.asarray(HOME), np.array([0, -.1, 0, -1.6, 0, 1.5, .785398])]
        for seed in seeds:
            joints = np.clip(seed.copy(), lower, upper)
            for _ in range(180):
                self._ik_data.qpos[self._qpos_indices] = joints
                mujoco.mj_kinematics(self.model, self._ik_data)
                mujoco.mj_comPos(self.model, self._ik_data)
                current_mat = self._ik_data.site_xmat[self._tcp_id].reshape(3, 3)
                position_error = target - self._ik_data.site_xpos[self._tcp_id]
                # Log SO(3) gives a well-defined error even near 180 degrees.
                difference = target_mat @ current_mat.T
                delta_quat = np.empty(4)
                mujoco.mju_mat2Quat(delta_quat, difference.ravel())
                if delta_quat[0] < 0:
                    delta_quat *= -1
                rotation_error = np.zeros(3)
                mujoco.mju_quat2Vel(rotation_error, delta_quat, 1.0)
                pos_norm, rot_norm = np.linalg.norm(position_error), np.linalg.norm(rotation_error)
                error = float(pos_norm + .15 * rot_norm)
                if error < best[0]:
                    best = (error, joints.copy(), float(pos_norm), float(rot_norm))
                if pos_norm < .002 and rot_norm < .02:
                    return {"success": True, "joints": joints.tolist(), "error": float(pos_norm), "orientation_error": float(rot_norm)}
                mujoco.mj_jacSite(self.model, self._ik_data, jac_position, jac_rotation, self._tcp_id)
                jacobian = np.vstack([jac_position[:, self._dof_indices], .35 * jac_rotation[:, self._dof_indices]])
                error_vector = np.r_[position_error, .35 * rotation_error]
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .002**2 * np.eye(6), error_vector)
                delta *= min(1.0, .15 / (np.max(np.abs(delta)) + 1e-12))
                joints = np.clip(joints + delta, lower, upper)
        return {"success": False, "joints": best[1].tolist(), "error": best[2], "orientation_error": best[3]}

    def _step(self) -> None:
        if self._hooks and self._command:
            from simex.source_model import SourceModelError
            from simex.errors import PolicyRequestError
            if self.data.time + self.model.opt.timestep > self.config.get('max_sim_time', 120) + 1e-8:
                raise PolicyRequestError('Simulation time budget exhausted')
            values = self._hooks.call('control_step', self._command, self.state(),
                                     float(self.data.time - self._command_start),
                                     float(self.model.opt.timestep), self.data.ctrl.tolist())
            try:
                control = np.asarray(values, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SourceModelError('Control hook returned nonnumeric motor targets') from exc
            limits = self.model.actuator_ctrlrange
            if (control.shape != (8,) or not np.isfinite(control).all()
                    or np.any(control < limits[:, 0]) or np.any(control > limits[:, 1])):
                raise SourceModelError('Control hook must return 8 finite in-range motor targets')
            self.data.ctrl[:] = control
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._steps += 1
        if not np.all(np.isfinite(self.data.qpos)):
            if self._hooks:
                from simex.source_model import SourceModelError
                raise SourceModelError('Source model produced non-finite simulator state')
            raise RuntimeError("Non-finite simulator state")
        for name, body_id in self._object_ids.items():
            self._max_heights[name] = max(self._max_heights[name], float(self.data.xpos[body_id, 2]))
        if self.data.time + 1e-9 >= self._next_callback:
            self._next_callback += .25
            if self.on_step is not None:
                self.on_step()

    @staticmethod
    def _duration(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0 or value > 60:
            raise ValueError("duration must be finite and in (0, 60] seconds")
        return value

    def move_joints(self, joints: list, duration: float = 1.0) -> dict:
        self._ready()
        requested = _finite(joints, 7, "joints")
        target = np.clip(requested, self._limits[:, 0], self._limits[:, 1])
        start = self.data.qpos[self._qpos_indices].copy()
        duration = self._model_duration('move_joints', duration, joints=target.tolist())
        steps = max(1, math.ceil(self._duration(duration) / self.model.opt.timestep))
        for index in range(1, steps + 1):
            phase = index / steps
            blend = phase**3 * (10 - 15 * phase + 6 * phase**2)
            self.data.ctrl[:7] = start + blend * (target - start)
            self._step()
        # Source alignment can model an additive arm settling interval without
        # changing the commanded trajectory or the gripper/wait durations.
        settle = self.scene['physics'].get('arm_settle_s', 0.)
        for _ in range(math.ceil(settle / self.model.opt.timestep)):
            self._step()
        error = float(np.max(np.abs(self.data.qpos[self._qpos_indices] - target)))
        return {"success": error < .08, "joint_error": error, "clamped": bool(np.any(target != requested)), **self.state()}

    def move_ee(self, position: list, quaternion: list, duration: float = 1.0) -> dict:
        result = self.ik(position, quaternion)
        if not result["success"]:
            return {"success": False, "reason": "unreachable", "ik": result, **self.state()}
        motion = self.move_joints(result["joints"], duration)
        motion["ik"] = result
        return motion

    def gripper(self, width: float, force: float = 40, duration: float = .5) -> dict:
        """Command total opening in meters, with a force cap in N per finger."""
        self._ready()
        width, force = float(width), float(force)
        if not math.isfinite(width) or not math.isfinite(force) or force <= 0:
            raise ValueError("gripper width/force must be finite and force must be positive")
        width = float(np.clip(width, 0, .08))
        force = float(np.clip(force, .1, 100))
        self.model.actuator_forcerange[7] = [-2 * force, 2 * force]
        start = float(self.data.ctrl[7])
        target = width / .08 * 255
        duration = self._model_duration('gripper', duration, width=width, force=force)
        steps = max(1, math.ceil(self._duration(duration) / self.model.opt.timestep))
        for index in range(1, steps + 1):
            self.data.ctrl[7] = start + (target - start) * index / steps
            self._step()
        return {"success": True, **self.state()}

    def wait(self, seconds: float) -> dict:
        self._ready()
        seconds = self._model_duration('wait', seconds)
        if float(seconds) == 0:
            return self.state()
        for _ in range(math.ceil(self._duration(seconds) / self.model.opt.timestep)):
            self._step()
        return self.state()

    def _model_duration(self, method, duration, **arguments):
        if not self._hooks:
            return duration
        from simex.source_model import SourceModelError
        command = {'method': method, 'duration': float(duration), **arguments}
        value = self._hooks.call('command_duration', command)
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not (0 <= value <= 60)
                or (method != 'wait' and value < .001)):
            raise SourceModelError('Invalid model command duration')
        self._command = {**command, 'duration': value}
        self._command_start = float(self.data.time)
        return value

    def metrics(self) -> dict:
        self._ready()
        objects = {}
        for name, body_id in self._object_ids.items():
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            objects[name] = {"position": self.data.xpos[body_id].tolist(), "quaternion": self.data.xquat[body_id].tolist(),
                             "max_height": self._max_heights[name], "initial_height": self._initial_heights[name],
                             "linear_velocity": velocity[3:].tolist()}
        tote = self.scene.get("tote", {})
        if self._hooks:
            from simex.source_geometry import measured_geometry
            objects, tote = measured_geometry(self.model, self.data, objects, self._object_ids)
        return {"objects": objects, "tote": {"position": tote.get("position", [.4, -.26, 0]),
                 "inner_size": tote.get("inner_size", [.26, .22, .1]), "yaw": tote.get("yaw", 0),
                 **({'bottom_thickness': tote['bottom_thickness']} if 'bottom_thickness' in tote else {})},
                "sim_time": float(self.data.time), "steps": self._steps,
                "contact_parameters": self.contact_parameters()}

    def contact_parameters(self) -> dict:
        """Private calibration diagnostics; never part of RobotAPI state.

        Report model-combined coefficients as well as observed contacts. These
        make ineffective calibrations detectable even when a policy never grips.
        The pinned assets have equal geom priorities and no explicit pairs.
        """
        self._ready()
        collision = (self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)
        fingers = [self.model.body(name).id for name in ('left_finger', 'right_finger')]
        finger_geoms = np.flatnonzero(np.isin(self.model.geom_bodyid, fingers) & collision)
        table_id = self.model.geom('table').id
        table = float(self.model.geom_friction[table_id, 0])
        per_plate = {}
        for name, body_id in self._object_ids.items():
            plate_id = self.model.geom(f'plate_disk_{name}').id
            plate = float(self.model.geom_friction[plate_id, 0])
            finger_pairs = np.maximum(self.model.geom_friction[finger_geoms, 0], plate)
            observed = {'table': set(), 'finger': set()}
            for contact in self.data.contact:
                ids = (int(contact.geom1), int(contact.geom2))
                if not any(self.model.geom_bodyid[i] == body_id for i in ids):
                    continue
                if table_id in ids:
                    observed['table'].add(float(contact.friction[0]))
                if any(i in finger_geoms for i in ids):
                    observed['finger'].add(float(contact.friction[0]))
            per_plate[name] = {
                'table_sliding': max(table, plate),
                'finger_sliding_range': [float(finger_pairs.min()), float(finger_pairs.max())],
                'observed_sliding': {key: sorted(value) for key, value in observed.items()},
            }
        return {'friction_scale': self.scene['physics']['contact_friction_scale'],
                'friction_rule': 'global_geom_sliding_scale_with_max_combination',
                'plates': per_plate}

    def close(self) -> None:
        if self._hooks is not None:
            self._hooks.close()
            self._hooks = None
        self._command = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.data = None
        self.model = None
