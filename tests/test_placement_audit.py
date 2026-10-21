import json

import pytest

from scripts.audit_placement import audit_rollout, main
from simex.tasks import make_task


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_rollout(directory, *, opening=True, high=False):
    scene = make_task("far_plate", 1)["scene"]
    green = {"position": [.39, -.26, .20 if high else .012],
             "quaternion": [1., 0., 0., 0.], "initial_height": .003,
             "max_height": .20, "linear_velocity": [0., 0., 0.]}
    red = {**green, "position": [.32, .26, .003]}
    actions = [{"method": "gripper", "arguments": {"width": 0.},
                "before": {"gripper_width": .08}, "after": {"gripper_width": .01}}]
    if opening:
        actions.append({"method": "gripper", "arguments": {"width": .08},
                        "before": {"gripper_width": .01},
                        "after": {"gripper_width": .08, "sim_time": 2.,
                                  "ee_position": [.39, -.26, .04]}})
    save(directory / "result.json", {"success": True, "status": "success",
         "public_evidence": {"actions": actions}})
    save(directory / "manifest.json", {"engine": "mujoco", "task": "test"})
    save(directory / "execution.json", {"actual_started": True})
    save(directory / "private/scene.json", scene)
    save(directory / "private/metrics.json", {"objects": {"green": green, "red": red},
         "tote": scene["tote"]})
    samples = [{"sim_time": t, "robot": {"ee_position": [.39, -.26, z], "gripper_width": .08},
                "objects": {"green": green, "red": red}} for t, z in [(2.25, .1), (3., .3)]]
    (directory / "private/object_trace.jsonl").write_text(
        "\n".join(json.dumps(s) for s in samples))
    return actions


def test_audit_separates_successful_lift_from_full_placement_and_release(tmp_path):
    fixture_rollout(tmp_path, opening=False, high=True)
    result = audit_rollout(tmp_path)
    assert result["recorded_task_success"] is True
    assert result["full_placement_predicate_passed"] is False
    assert result["opening_after_close"] is result["after_opening"] is None
    assert result["contact_release_certified"] is False
    assert result["max_object_lift_m"] == pytest.approx(.197)


def test_audit_measures_opening_withdrawal_and_stability_without_certifying_contact(tmp_path):
    fixture_rollout(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = audit_rollout(tmp_path)
    assert result["full_placement_predicate_passed"] is True
    assert result["opening_after_close"]["width_after_m"] == .08
    interval = result["after_opening"]
    assert interval["first_sample_delay_s"] == .25
    assert interval["observed_duration_after_opening_s"] == 1.
    assert interval["sample_span_s"] == .75
    assert interval["tcp_displacement_from_opening_m"] == pytest.approx(.26)
    assert interval["final_tcp_to_object_center_m"] == pytest.approx(.288)
    assert interval["object_displacement_from_first_post_open_sample_m"] == 0.
    assert interval["final_window"]["requested_span_observed"] is False
    assert interval["final_window"]["canonical_predicate_passed_all_samples"] is True
    assert result["contact_release_certified"] is False
    assert result["errors"] == []
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_later_close_supersedes_opening_and_missing_trace_remains_unknown(tmp_path):
    actions = fixture_rollout(tmp_path)
    actions.append({"method": "gripper", "arguments": {"width": .01},
                    "before": {"gripper_width": .08}, "after": {"gripper_width": .01}})
    save(tmp_path / "result.json", {"success": True, "status": "success",
         "public_evidence": {"actions": actions}})
    assert audit_rollout(tmp_path)["opening_after_close"] is None
    fixture_rollout(tmp_path)
    (tmp_path / "private/object_trace.jsonl").unlink()
    result = audit_rollout(tmp_path)
    assert result["full_placement_predicate_passed"] is True
    assert result["after_opening"] is None
    assert result["errors"] == ["private/object_trace.jsonl: FileNotFoundError"]


def test_nonfinite_opening_endpoint_does_not_become_measurement(tmp_path):
    actions = fixture_rollout(tmp_path)
    actions[-1]["after"] = {"sim_time": float("nan"), "gripper_width": float("nan"),
                            "ee_position": [0., 0., float("inf")]}
    save(tmp_path / "result.json", {"public_evidence": {"actions": actions}})
    result = audit_rollout(tmp_path)
    assert result["opening_after_close"]["end_time_s"] is None
    assert result["after_opening"] is None
    json.dumps(result, allow_nan=False)


def test_run_audit_excludes_engine_preflight_trials(tmp_path, monkeypatch, capsys):
    fixture_rollout(tmp_path / "discovery/trial")
    fixture_rollout(tmp_path / "diagnostics/isaac")
    monkeypatch.setattr("sys.argv", ["audit_placement.py", "--run", str(tmp_path)])
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["rollouts"]) == 1
    assert "/discovery/trial" in report["rollouts"][0]["rollout"]


@pytest.mark.parametrize("contacts,expected", [([1.], True), ([], False), (None, None)])
def test_private_contact_measurement_distinguishes_contact_absence_and_unavailable(tmp_path, contacts, expected):
    fixture_rollout(tmp_path)
    p = tmp_path / "private/metrics.json"
    metrics = json.loads(p.read_text())
    metrics["contact_parameters"] = {"plates": {"green": {"observed_sliding": {"finger": contacts}}}}
    save(p, metrics)
    assert audit_rollout(tmp_path)["final_finger_contact_observed"] is expected


@pytest.mark.parametrize("final_motion", [False, True])
def test_final_window_separates_initial_settling_from_persistent_motion(tmp_path, final_motion):
    fixture_rollout(tmp_path)
    path = tmp_path / "private/object_trace.jsonl"
    template = json.loads(path.read_text().splitlines()[-1])
    samples = []
    for i in range(33):
        sample = json.loads(json.dumps(template))
        sample["sim_time"] = 2.25 + i / 4
        green = sample["objects"]["green"]
        if i < 8 or (final_motion and i == 32):
            green["linear_velocity"] = [0., 0., .6]
            green["position"][2] += .05
        samples.append(sample)
    path.write_text("\n".join(json.dumps(s) for s in samples))
    result = audit_rollout(tmp_path)
    interval = result["after_opening"]
    assert interval["max_object_speed_m_s"] == .6
    window = interval["final_window"]
    assert window["requested_span_observed"] is True
    assert window["samples"] == 21
    assert window["canonical_predicate_passed_all_samples"] is not final_motion
    assert window["max_object_speed_m_s"] == (.6 if final_motion else 0.)
    assert window["max_object_displacement_from_window_start_m"] == pytest.approx(.05 if final_motion else 0.)
    assert result["contact_release_certified"] is False


def test_final_window_does_not_infer_missing_trace_velocity_from_final_metrics(tmp_path):
    fixture_rollout(tmp_path)
    path = tmp_path / "private/object_trace.jsonl"
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    samples[0]["objects"]["green"].pop("linear_velocity")
    path.write_text("\n".join(json.dumps(s) for s in samples))
    window = audit_rollout(tmp_path)["after_opening"]["final_window"]
    assert window["max_object_speed_m_s"] is None
    assert window["canonical_predicate_passed_all_samples"] is None
