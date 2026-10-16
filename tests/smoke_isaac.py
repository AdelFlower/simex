"""Explicit hardware smoke: scripts/isaac_python.sh tests/smoke_isaac.py.

This file is deliberately not collected by pytest: it requires Isaac and RTX.
Writes evidence into runs/isaac_smoke; never edits policy assets.
"""
from pathlib import Path
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from simex_private.isaac_backend import IsaacBackend


def main():
    scene = {
        "task_id": "hardware_smoke", "seed": 0,
        "robot": {"base_position": [0, 0, 0], "home": [0, -.5, 0, -2.2, 0, 1.7, .785398]},
        "table": {"center": [.4, 0, -.025], "size": [1.2, 1.1, .05]},
        "plates": [{"name": "green", "position": [.55, .16, .006], "radius": .07,
                    "thickness": .006, "rim_height": .012, "mass": .15,
                    "color": [.1, .75, .15], "friction": .6}],
        "tote": {"position": [.4, -.26, 0], "inner_size": [.26, .22, .1], "wall_thickness": .008, "yaw": 0},
        "targets": ["green"], "physics": {"dt": 1 / 240}, "cameras": {"width": 320, "height": 240},
    }
    folder = Path("runs/isaac_smoke") / f"session_{time.time_ns()}"
    folder.mkdir(parents=True, exist_ok=True)
    backend = IsaacBackend()
    evidence = {}
    try:
        evidence["reset"] = backend.reset(scene)
        print("SIMEX_SMOKE reset", evidence["reset"], flush=True)
        for camera in ["overhead", "wrist"]:
            observation = backend.observe(camera)
            assert observation["rgb"].shape == (240, 320, 3)
            assert observation["depth"].shape == (240, 320)
            assert np.isfinite(observation["depth"]).sum() > 1000
            assert observation["rgb"].std() > 1
            np.savez_compressed(folder / f"{camera}.npz", **observation)
            try:
                from PIL import Image
                Image.fromarray(observation["rgb"]).save(folder / f"{camera}.png")
            except ImportError:
                pass
            evidence[camera] = {"finite_depth_pixels": int(np.isfinite(observation["depth"]).sum()),
                                "rgb_std": float(observation["rgb"].std()),
                                "intrinsics": observation["intrinsics"].tolist(),
                                "camera_to_world": observation["camera_to_world"].tolist()}
            if camera == "overhead":
                rgb = observation["rgb"].astype(float)
                mask = (rgb[:, :, 1] > 1.25*rgb[:, :, 0]) & (rgb[:, :, 1] > 1.15*rgb[:, :, 2]) & np.isfinite(observation["depth"])
                rows, columns = np.where(mask)
                assert len(rows) > 20, "No green plate detected in physical RGBD"
                pixels = np.stack([columns, rows, np.ones_like(rows)], axis=0)
                xyz_camera = np.linalg.inv(observation["intrinsics"]) @ pixels * observation["depth"][mask]
                xyz_world = observation["camera_to_world"][:3, :3] @ xyz_camera + observation["camera_to_world"][:3, 3:4]
                estimate = np.median(xyz_world, axis=1)
                evidence["green_backprojection"] = estimate.tolist()
                assert np.linalg.norm(estimate[:2] - [.55, .16]) < .025, estimate
            print("SIMEX_SMOKE camera", camera, evidence[camera], flush=True)
        state = backend.state()
        solution = backend.ik(state["ee_position"], state["ee_quaternion"])
        assert solution["success"], solution
        evidence["ik"] = solution
        joints = np.asarray(state["joints"])
        joints[0] += .08
        evidence["move_joints"] = backend.move_joints(joints, .8)
        assert abs(evidence["move_joints"]["joints"][0] - joints[0]) < .08
        state = backend.state()
        target = np.asarray(state["ee_position"]) + [0, 0, .03]
        evidence["move_ee"] = backend.move_ee(target, state["ee_quaternion"], .8)
        assert np.linalg.norm(np.asarray(backend.state()["ee_position"])-target) < .015
        evidence["gripper"] = backend.gripper(.03, duration=.4)
        assert evidence["gripper"]["gripper_width"] < .055
        efforts = backend.robot._articulation_view.get_max_efforts(joint_indices=backend._finger_indices)
        np.testing.assert_allclose(efforts, [[40., 40.]])
        evidence["finger_force_limits_N"] = np.asarray(efforts).tolist()
        before_wait = backend.state()["sim_time"]
        backend.wait(0)
        assert backend.state()["sim_time"] == before_wait
        backend.gripper(.08, duration=.4)
        evidence["metrics"] = backend.metrics()
        assert abs(evidence["metrics"]["objects"]["green"]["position"][2] - .003) < .015
        callback_times = []
        def record():
            before = backend.state()["sim_time"]
            backend.observe("wrist")
            assert backend.state()["sim_time"] == before
            callback_times.append(before)
        backend.on_step = record
        backend.wait(.55)
        assert len(callback_times) >= 2
        evidence["callback_times"] = callback_times
        evidence["second_reset"] = backend.reset(scene)
        assert backend.state()["sim_time"] == 0
        assert backend.observe("overhead")["rgb"].shape == (240, 320, 3)
        assert np.linalg.norm(np.asarray(backend.metrics()["objects"]["green"]["position"])[:2]-[.55,.16]) < .002
        evidence["status"] = "passed"
        print("SIMEX_SMOKE passed", flush=True)
    except BaseException as exc:
        import traceback
        evidence["status"] = "failed"
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
        print(evidence["traceback"], file=sys.stderr, flush=True)
        raise
    finally:
        (folder / "result.json").write_text(json.dumps(evidence, indent=2))
        backend.close()
        print('Evidence:', folder, flush=True)


if __name__ == "__main__":
    main()
