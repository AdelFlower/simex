"""Real-engine contract checks; run with MUJOCO_GL=egl python -m unittest discover -s tests."""
import unittest

import numpy as np

from simex.backends.mujoco_backend import MujocoBackend


class MujocoContractTests(unittest.TestCase):
    def setUp(self):
        self.backend = MujocoBackend()
        self.scene = {"seed": 23, "plates": [{"name": "green", "position": [.55, .16, .006]}],
                      "cameras": {"width": 320, "height": 240}}
        self.backend.reset(self.scene)

    def tearDown(self):
        self.backend.close()

    def test_real_contact_support_and_reset_determinism(self):
        initial = self.backend.metrics()["objects"]["green"]["position"]
        self.backend.wait(.5)
        settled = self.backend.metrics()["objects"]["green"]["position"]
        self.assertAlmostEqual(settled[2], .003, delta=.0005)
        self.assertLess(np.linalg.norm(np.array(settled[:2]) - [.55, .16]), .001)
        # The only equality is the upstream coupling between the two fingers;
        # there are no welds or attachments between robot and task objects.
        self.assertEqual(self.backend.model.neq, 1)
        self.backend.reset(self.scene)
        np.testing.assert_allclose(self.backend.metrics()["objects"]["green"]["position"], initial, atol=1e-12)
        self.assertEqual(self.backend.metrics()["steps"], 0)

    def test_rgbd_calibration_and_wrist(self):
        observation = self.backend.observe("overhead")
        rgb, depth = observation["rgb"], observation["depth"]
        self.assertEqual(rgb.shape, (240, 320, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(depth.dtype, np.float32)
        self.assertTrue(np.isfinite(depth).all())
        self.assertTrue((depth > 0).all())
        # A color-only detector locates visible plate pixels. Back-projecting
        # them checks both metric depth and the OpenCV/world axis convention.
        green = (rgb[:, :, 1].astype(float) > 1.5 * rgb[:, :, 0]) & (rgb[:, :, 1].astype(float) > 1.5 * rgb[:, :, 2])
        rows, columns = np.nonzero(green)
        self.assertGreater(len(rows), 20)
        pixels = np.stack([columns, rows, np.ones_like(rows)])
        points_camera = (np.linalg.inv(observation["intrinsics"]) @ pixels) * depth[rows, columns]
        transform = observation["camera_to_world"]
        points_world = transform[:3, :3] @ points_camera + transform[:3, 3:4]
        np.testing.assert_allclose(np.median(points_world[:2], axis=1), [.55, .16], atol=.015)
        self.assertTrue(-.002 < np.median(points_world[2]) < .025)
        wrist = self.backend.observe("wrist")
        self.assertEqual(wrist["rgb"].shape, rgb.shape)
        rotation = wrist["camera_to_world"][:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertLess(rotation[2, 2], -.9)

    def test_robot_ik_is_read_only_and_reports_unreachable(self):
        before = self.backend.data.qpos.copy()
        result = self.backend.ik([.55, .16, .2], [0, 1, 0, 0])
        self.assertTrue(result["success"])
        self.assertLess(result["error"], .002)
        np.testing.assert_array_equal(self.backend.data.qpos, before)
        self.assertFalse(self.backend.ik([3, 0, 0], [1, 0, 0, 0])["success"])
        self.assertEqual(self.backend.metrics()["steps"], 0)
        self.assertNotIn("objects", self.backend.state())

    def test_motion_advances_physics_and_callbacks(self):
        observed = []
        self.backend.on_step = lambda: observed.append(self.backend.state()["sim_time"])
        initial = self.backend.state()
        target = initial["joints"].copy()
        target[0] += .1
        result = self.backend.move_joints(target, 1.)
        self.assertTrue(result["success"])
        self.assertGreater(result["joints"][0], initial["joints"][0] + .07)
        self.assertEqual(self.backend.metrics()["steps"], 240)
        np.testing.assert_allclose(observed, [.25, .5, .75, 1], atol=1e-9)
        self.backend.gripper(0, duration=.5)
        self.backend.wait(.5)
        self.assertLess(self.backend.state()["gripper_width"], .003)
        self.assertEqual(self.backend.metrics()["steps"], 480)


if __name__ == "__main__":
    unittest.main()
