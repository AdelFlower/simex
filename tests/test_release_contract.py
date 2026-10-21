import copy

import pytest

from simex.release import release_evidence
from simex.scoring import score
from simex.tasks import make_task, validate_predicate, with_placement_contract


def evidence():
    task = make_task("far_plate", 1, "released_placement_v1")
    green = {"position": [.39, -.26, .012], "quaternion": [1., 0., 0., 0.],
             "linear_velocity": [0., 0., 0.], "initial_height": .003, "max_height": .15}
    red = {**green, "position": [.32, .26, .003]}
    robot = {"sim_time": 3., "ee_position": [.39, -.26, .3], "gripper_width": .08}
    metrics = {"objects": {"green": green, "red": red}, "tote": task["scene"]["tote"]}
    actions = [
        {"method": "gripper", "arguments": {"width": 0.},
         "before": {"gripper_width": .08},
         "after": {"gripper_width": .01, "sim_time": 1., "ee_position": [.39, -.26, .04]}},
        {"method": "gripper", "arguments": {"width": .08},
         "before": {"gripper_width": .01},
         "after": {"gripper_width": .08, "sim_time": 2., "ee_position": [.39, -.26, .04]}}]
    samples = [{"sim_time": t, "robot": {**robot, "sim_time": t},
                "objects": copy.deepcopy(metrics["objects"])} for t in [2.5, 2.75, 3.]]
    return task, metrics, robot, actions, samples


def evaluate(parts):
    task, metrics, robot, actions, samples = parts
    metrics["release_evidence"] = release_evidence(task["scene"], metrics, robot, actions, samples)
    return score(task, metrics), metrics["release_evidence"]["green"]


def test_geometry_success_does_not_imply_release():
    parts = evidence()
    task, metrics, _, actions, _ = parts
    assert score(make_task("far_plate", 1), metrics)
    assert not score(task, metrics)  # Missing release evidence is not success.
    actions.pop()
    passed, detail = evaluate(parts)
    assert not passed
    assert detail["reason"] == "no_opening_after_measured_close"


def test_release_requires_measured_opening_withdrawal_separation_and_stability():
    passed, detail = evaluate(evidence())
    assert passed
    assert detail["released"] is True
    assert detail["settle_samples"] == 3
    assert detail["settle_span_s"] == .5
    assert detail["withdrawal_m"] == pytest.approx(.26)


@pytest.mark.parametrize("change", [
    "still_closed", "no_withdrawal", "still_near_hand", "later_close", "close_not_executed",
    "finger_contact", "sample_gap", "short_window", "missing_velocity", "missing_orientation",
    "moving", "drifting", "outside_during_settle", "stale_trace",
])
def test_incomplete_or_unreleased_placements_fail_closed(change):
    parts = evidence()
    task, metrics, robot, actions, samples = parts
    if change == "still_closed":
        robot["gripper_width"] = .01
    elif change == "no_withdrawal":
        robot["ee_position"] = actions[-1]["after"]["ee_position"]
    elif change == "still_near_hand":
        for sample in samples:
            sample["robot"]["ee_position"] = [.39, -.26, .06]
    elif change == "later_close":
        actions.append(copy.deepcopy(actions[0]))
    elif change == "close_not_executed":
        actions[0]["after"]["gripper_width"] = .08
    elif change == "finger_contact":
        metrics["contact_parameters"] = {"plates": {"green": {"observed_sliding": {"finger": [1.]}}}}
    elif change == "sample_gap":
        samples.pop(1)
    elif change == "short_window":
        samples[0]["sim_time"] = 2.6
    elif change == "missing_velocity":
        samples[1]["objects"]["green"].pop("linear_velocity")
    elif change == "missing_orientation":
        samples[1]["objects"]["green"].pop("quaternion")
    elif change == "moving":
        samples[1]["objects"]["green"]["linear_velocity"] = [0., 0., .2]
    elif change == "drifting":
        samples[1]["objects"]["green"]["position"][0] += .01
    elif change == "outside_during_settle":
        samples[1]["objects"]["green"]["position"][0] += .3
    elif change == "stale_trace":
        robot["sim_time"] = 4.
    assert not evaluate(parts)[0]


def test_contract_is_explicit_idempotent_and_preserves_legacy_tasks():
    legacy = make_task("far_plate", 1)
    original = copy.deepcopy(legacy)
    strict = with_placement_contract(legacy, "released_placement_v1")
    assert legacy == original
    assert strict["scene"] == legacy["scene"]
    assert with_placement_contract(strict, "released_placement_v1") == strict
    assert strict["predicate"]["all"][:-1] == legacy["predicate"]["all"]
    validate_predicate(strict["predicate"], ["green", "red"])
    with pytest.raises(ValueError):
        validate_predicate({"kind": "released", "object": "green", "min": 0.}, ["green"])
    with pytest.raises(ValueError):
        with_placement_contract(legacy, "unknown")
