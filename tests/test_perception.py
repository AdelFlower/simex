import copy
import unittest

import numpy as np

from simex.backends.mujoco_backend import MujocoBackend
from simex.perception import estimates_from_rgbd, source_task_from_observation
from simex.tasks import TASK_IDS, make_task


class PerceptionTests(unittest.TestCase):
    def test_real_rgbd_centers_for_all_tasks(self):
        backend = MujocoBackend()
        try:
            for task_id in TASK_IDS:
                with self.subTest(task=task_id):
                    task = make_task(task_id, 72)
                    backend.reset(task["scene"])
                    # The estimator receives only the actual public observation.
                    estimate = estimates_from_rgbd(backend.observe("overhead"))
                    for plate in task["scene"]["plates"]:
                        xy = estimate["objects"][plate["name"]]["center"][:2]
                        np.testing.assert_allclose(xy, plate["position"][:2], atol=.02)
                    np.testing.assert_allclose(estimate["tote"]["center"][:2], task["scene"]["tote"]["position"][:2], atol=.02)
        finally:
            backend.close()

    def test_reconstruction_copies_only_visible_xy(self):
        nominal = make_task("far_plate", 0)
        evidence = {"seed": 92001, "scene": {"physics": {"dt": 4}},
                    "initial_visual_estimates": {"objects": {
                        "green": {"center": [.61, .12, .09], "mass": 300},
                        "red": {"center": [.30, .28, .01]}},
                        "tote": {"center": [.42, -.30, .12], "yaw": 2}}}
        frozen = copy.deepcopy(evidence)
        result = source_task_from_observation("far_plate", evidence)
        expected = copy.deepcopy(nominal)
        expected["scene"]["plates"][0]["position"][:2] = [.61, .12]
        expected["scene"]["plates"][1]["position"][:2] = [.30, .28]
        expected["scene"]["tote"]["position"][:2] = [.42, -.30]
        self.assertEqual(result, expected)
        self.assertEqual(evidence, frozen)

    def test_invalid_and_missing_detections_keep_nominal(self):
        nominal = make_task("shifted_single", 0)
        for value in (None, {}, {"initial_visual_estimates": None},
                      {"initial_visual_estimates": {"objects": {"green": {"center": [99, 0, 0]},
                                                               "red": {"center": [float("nan"), 0, 0]}},
                                                    "tote": {"center": [0, 0]}}}):
            self.assertEqual(source_task_from_observation("shifted_single", value), nominal)

    def test_invalid_depth_and_robot_trim_are_excluded(self):
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb[:, :] = [20, 90, 200]
        observation = {"rgb": rgb, "depth": np.full((20, 20), .5),
                       "intrinsics": np.array([[20, 0, 10], [0, 20, 10], [0, 0, 1.]]),
                       "camera_to_world": np.array([[1, 0, 0, .4], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1.]])}
        self.assertNotIn("tote", estimates_from_rgbd(observation))
        observation["depth"][:] = np.nan
        self.assertEqual(estimates_from_rgbd(observation)["objects"], {})


if __name__ == "__main__":
    unittest.main()
