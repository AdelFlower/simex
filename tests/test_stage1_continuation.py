"""Deterministic provenance fixtures, not physical manipulation results."""
import copy
import json
from pathlib import Path

import pytest

from simex import research
from simex.storage import atomic_json


class NoCalls:
    count = 0

    def ask(self, *args, **kwargs):
        raise AssertionError("Import must not generate policies or research decisions")


def recorded_trial(directory, task, *, opened=True, inside=True):
    directory.mkdir(parents=True)
    center = task["scene"]["tote"]["position"]
    green = {"position": [center[0], center[1], .012] if inside else [.6, .2, .003],
             "quaternion": [1., 0., 0., 0.], "linear_velocity": [0., 0., 0.],
             "initial_height": .003, "max_height": .15}
    red = {**green, "position": [.32, .26, .003]}
    robot = {"sim_time": 3., "ee_position": [center[0], center[1], .3], "gripper_width": .08}
    actions = [{"method": "gripper", "arguments": {"width": 0.},
                "before": {"gripper_width": .08}, "after": {**robot, "sim_time": 1., "gripper_width": .01}}]
    if opened:
        actions.append({"method": "gripper", "arguments": {"width": .08},
                        "before": {"gripper_width": .01},
                        "after": {**robot, "sim_time": 2., "ee_position": [center[0], center[1], .04]}})
    actions.append({"method": "wait", "arguments": {"seconds": 1.}, "after": robot})
    metrics = {"objects": {"green": green, "red": red}, "tote": task["scene"]["tote"],
               "contact_parameters": {"plates": {"green": {"observed_sliding": {"finger": [] if opened else [1.]}}}}}
    from simex.scoring import score
    passed = score(task, metrics)
    result = {"status": "success" if passed else "failed", "success": passed, "actual_started": True,
              "public_evidence": {"actions": actions, "keyframes": [], "stdout": ""},
              "output_dir": str(directory), "script_path": str(directory / "policy.py")}
    for name, data in {
        "result.json": result, "execution.json": {"actual_started": True},
        "manifest.json": {"engine": "mujoco", "fixed_script": False},
        "research_request.json": {"engine": "mujoco", "task": task},
        "private/metrics.json": metrics, "private/scene.json": task["scene"],
    }.items():
        atomic_json(directory / name, data)
    (directory / "policy.py").write_text("pass\n")
    samples = [{"sim_time": t, "robot": {**robot, "sim_time": t},
                "objects": metrics["objects"]} for t in [2.5, 2.75, 3.]]
    (directory / "private/object_trace.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
    return {"task": task["name"], "seed": task["scene"]["seed"], "status": result["status"],
            "success": passed, "actual_started": True, "output_dir": str(directory),
            "video": None, "phase_images": []}


@pytest.fixture
def parent(tmp_path):
    root = tmp_path / "parent"
    cfg = {"population": 2, "cycles": 1, "selection_trials": 1, "seed": 41,
           "max_agent_calls": 100, "adapt_trials": 1, "eval_trials": 1}
    cfg = research._initialize(root, cfg)
    tasks = json.loads((root / "selection_tasks.json").read_text())
    auxiliary = json.loads((root / "auxiliary_selection_tasks.json").read_text())
    members = []
    for index in range(2):
        frozen = root / "discovery" / f"member_{index:02d}" / "frozen"
        frozen.mkdir(parents=True)
        (frozen / "toolbox.py").write_text(f"MEMBER = {index}\nclass Toolbox:\n    def __init__(self, robot):\n        self.robot = robot\n")
        (frozen / "skill.md").write_text("Fixture learned library")
        atomic_json(frozen / "frozen.json", {})
        records = [recorded_trial(root / "selection" / f"member_{index:02d}" / f"{j:03d}", task,
                                 opened=j < (1 if index == 0 else 2), inside=index == 0 or j < 2)
                   for j, task in enumerate(tasks)]
        aux = [recorded_trial(root / "auxiliary_selection" / f"member_{index:02d}" / f"{j:03d}", task)
               for j, task in enumerate(auxiliary)]
        members.append({"member": index, "toolbox": str(frozen), "summaries": [],
                        "selection_results": records, "auxiliary_results": aux,
                        "successes": sum(r["success"] for r in records),
                        "auxiliary_successes": sum(r["success"] for r in aux),
                        "selection_valid_executions": len(records) + len(aux), "capability_bank_size": 2})
    atomic_json(root / "discovery.json", {"winner": 0, "process_complete": True,
                                         "members": members, "agent_count": 2})
    for i in range(2):
        atomic_json(root / "agent_calls" / f"{i:05d}_policy" / "request.json", {"role": "policy"})
    child_cfg = {**cfg, "stage1_parent": str(root), "placement_contract": "released_placement_v1",
                 "max_agent_calls": 98}
    return root, tmp_path / "child", child_cfg


def test_continuation_regrades_all_source_trials_without_reexecution_or_parent_mutation(parent):
    source, child, cfg = parent
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file() and p.name != ".research.lock"}
    result = research.discover(object(), NoCalls(), child, cfg)
    assert result["winner"] == 1
    assert [m["successes"] for m in result["members"]] == [1, 2]
    assert [m["legacy_predicate_successes"] for m in result["members"]] == [3, 2]
    assert result["inherited_stage1"]["parent_agent_budget_used"] == 2
    assert result["inherited_stage1"]["continuation_model_budget"] == 98
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file() and p.name != ".research.lock"}
    assert not (child / "selection").exists()
    assert (child / "frozen/stage1/toolbox.py").read_bytes() == (source / "discovery/member_01/frozen/toolbox.py").read_bytes()
    assert research.discover(object(), NoCalls(), child, cfg) == result
    report = research.report(child)
    assert report["stage1"]["full_task_successes"] == 2
    assert report["stage1"]["released_placement_verified"] is True
    assert report["agent_budget_total_used"] == 2
    assert report["target_completed"] == 0
    from scripts.verify_run import verify
    assert verify(child)["integrity_ok"] is True
    (child / "frozen/stage1/skill.md").write_text("Changed after inheritance")
    checked = verify(child)
    assert checked["integrity_ok"] is False
    assert any("Selected continuation capability" in error for error in checked["errors"])


@pytest.mark.parametrize("violation", ["target_executed", "budget", "changed_seed", "missing_trace"])
def test_continuation_refuses_target_feedback_budget_expansion_or_incomplete_evidence(parent, violation):
    source, child, cfg = parent
    cfg = copy.deepcopy(cfg)
    if violation == "target_executed":
        atomic_json(source / "adaptation/far_plate/trial_00/target/execution.json", {"actual_started": True})
    elif violation == "budget":
        cfg["max_agent_calls"] = 99
    elif violation == "changed_seed":
        cfg["seed"] += 1
    elif violation == "missing_trace":
        (source / "selection/member_00/000/private/object_trace.jsonl").unlink()
    with pytest.raises((research.ResearchError, FileNotFoundError)):
        research.discover(object(), NoCalls(), child, cfg)
    assert not (child / "discovery.json").exists()


def test_contract_cannot_change_on_resume(parent):
    _, child, cfg = parent
    research.discover(object(), NoCalls(), child, cfg)
    with pytest.raises(research.ResearchError, match="Placement contract changed"):
        research.discover(object(), NoCalls(), child, {**cfg, "placement_contract": "legacy_geometry"})


def test_release_thresholds_are_frozen_in_manifest(parent, monkeypatch):
    _, child, cfg = parent
    research.discover(object(), NoCalls(), child, cfg)
    from simex.release import CONTRACT_SPEC
    monkeypatch.setitem(CONTRACT_SPEC, "minimum_withdrawal_m", .01)
    with pytest.raises(research.ResearchError, match="Release thresholds changed"):
        research.discover(object(), NoCalls(), child, cfg)
