"""Outcome tests cover whole-object containment and history requirements."""
import copy
import math
import unittest

from simex.scoring import score
from simex.tasks import make_task


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.task = make_task("far_plate", 0)
        self.tote = self.task["scene"]["tote"]
        center = self.tote["position"]
        self.metrics = {"tote": copy.deepcopy(self.tote), "objects": {
            "green": {"position": [center[0], center[1], .011], "quaternion": [1, 0, 0, 0],
                      "initial_height": .003, "max_height": .13, "linear_velocity": [0, 0, 0]},
            "red": {"position": [.32, .26, .003], "quaternion": [1, 0, 0, 0],
                    "initial_height": .003, "max_height": .003, "linear_velocity": [0, 0, 0]}}}

    def test_lifted_plate_resting_fully_in_tote_succeeds(self):
        self.assertTrue(score(self.task, self.metrics))

    def test_partial_entry_and_wall_balance_do_not_succeed(self):
        self.metrics["objects"]["green"]["position"][0] += .10
        self.assertFalse(score(self.task, self.metrics))

    def test_delivery_without_required_lift_fails(self):
        self.metrics["objects"]["green"]["max_height"] = .04
        self.assertFalse(score(self.task, self.metrics))

    def test_distractor_partial_entry_fails(self):
        center = self.tote["position"]
        # Red center lies outside, but its disk intersects the tote interior.
        self.metrics["objects"]["red"]["position"] = [center[0] + .16, center[1], .03]
        self.assertFalse(score(self.task, self.metrics))

    def test_plate_above_rim_or_still_falling_fails(self):
        original = copy.deepcopy(self.metrics)
        self.metrics["objects"]["green"]["position"][2] = .11
        self.assertFalse(score(self.task, self.metrics))
        self.metrics = original
        self.metrics["objects"]["green"]["linear_velocity"] = [0, 0, -.2]
        self.assertFalse(score(self.task, self.metrics))

    def test_tote_yaw_is_used_for_containment(self):
        center = self.tote["position"]
        self.metrics["tote"]["yaw"] = math.pi / 2
        # This offset fits the long axis only after the tote is rotated.
        self.metrics["objects"]["green"]["position"][:2] = [center[0], center[1] + .055]
        self.assertTrue(score(self.task, self.metrics))
        self.metrics["tote"]["yaw"] = 0
        self.assertFalse(score(self.task, self.metrics))


if __name__ == "__main__":
    unittest.main()
