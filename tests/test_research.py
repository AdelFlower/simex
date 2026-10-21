"""Protocol tests use deterministic stand-ins, never report simulated research success."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from simex import research


TASKS = ["far", "near", "offset"]
INITIAL_CODE = "INITIAL = 1\nclass Toolbox:\n    def __init__(self, robot):\n        self.robot = robot\n"
STAGE1_CODE = INITIAL_CODE.replace("INITIAL", "STAGE1")
SMALL = {"population": 2, "cycles": 1, "probes": 1, "focused": 1,
         "revisions": 1, "replay": 1, "adapt_trials": 2, "repairs": 2,
         "screen_tasks": 1, "align_attempts": 1, "eval_trials": 2,
         "selection_trials": 1, "seed": 41}


def task(task_id, seed):
    return {"name": task_id, "description": f"Move a plate: {task_id}",
            "scene": {"task_id": task_id, "seed": seed, "private_target_marker": "SECRET"}}


class FakeAgent:
    def __init__(self):
        self.count = 0
        self.calls = []

    def ask(self, role, prompt, images=None):
        self.count += 1
        payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
        self.calls.append((role, payload, images))
        if role in ("discover_probe", "discover_focus"):
            return {"weakness": "edge grip", "tasks": [{"name": "probe", "description": "Lift green",
                    "base_task_id": "far", "seed": 100, "scene_overrides": {},
                    "predicate": {"all": [{"kind": "lift", "object": "green", "min": 0.05}]}}]}
        if role == "discover_revision":
            return {"toolbox.py": STAGE1_CODE, "skill.md": "Source capability notes\n",
                    "hypothesis": "Approach rim"}
        if role == "discover_select":
            return {"accept": True, "reason": "Plausible reusable change"}
        if role == "align":
            return {"aligned": False, "source_config": {"friction_scale": 1.2},
                    "reason": "Observed grip slip"}
        if role == "repair_hypotheses":
            return {"hypotheses": ["Grip rim", "Slow the lift"]}
        if role == "repair":
            return {"toolbox.py": INITIAL_CODE + f"REPAIR = {payload['hypothesis']!r}\n",
                    "skill.md": "Unverified repair\n"}
        if role == "screen_tasks":
            return {"tasks": [{"base_task_id": payload["task_id"], "seed": seed}
                              for seed in payload["suggested_source_seeds"]]}
        if role == "repair_select":
            # Deliberately reject repairs despite their higher simulated score.
            return {"selected": "current", "reason": "Target evidence contradicts simulated success"}
        raise AssertionError(role)


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.target_results = []

    def rollout(self, engine, task, toolbox, out, script=None, source_config=None):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        source = (Path(toolbox) / "toolbox.py").read_text()
        generated = script is None
        code = script if script is not None else f"# Fresh policy {len(self.calls)}\nprint('run')\n"
        (out / "policy.py").write_text(code)
        call = {"engine": engine, "task": task, "toolbox": source, "out": out,
                "script": code, "generated": generated, "source_config": source_config}
        self.calls.append(call)
        success = "REPAIR" in source and engine == "mujoco"
        status = self.target_results.pop(0) if engine == "isaac" and self.target_results else "success" if success else "failed"
        return {"success": success, "status": status,
                "steps": 5, "actual_started": status != "infrastructure_error",
                "script_path": str(out / "policy.py"), "output_dir": str(out),
                "private_metrics": {"secret": "SHOULD_NEVER_REACH_AGENT"},
                "public_evidence": {"stdout": "robot observation", "keyframes": [],
                                    "private_metrics": "SHOULD_NEVER_REACH_AGENT"}}


@pytest.fixture
def harness(tmp_path):
    package = tmp_path / "package"
    initial = package / "initial"
    initial.mkdir(parents=True)
    (initial / "toolbox.py").write_text(INITIAL_CODE)
    (initial / "skill.md").write_text("Initial motion primitives\n")
    run = tmp_path / "run"
    with patch.object(research, "__file__", str(package / "research.py")), \
         patch.object(research, "TASK_IDS", TASKS), \
         patch.object(research, "make_task", task), \
         patch.object(research, "_source_reconstruction", lambda task_id, evidence: task(task_id, 0)), \
         patch.object(research, "source_task_from_spec",
                      lambda spec: task(spec["base_task_id"], spec["seed"])):
        yield FakeRunner(), FakeAgent(), run


def test_complete_protocol_budgets_fixed_replay_and_independence(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    assert not any(call["engine"] == "isaac" for call in runner.calls)
    adapted = research.adapt(runner, agent, run, SMALL)
    evaluated = research.evaluate(runner, agent, run, SMALL)
    target_calls = [call for call in runner.calls if call["engine"] == "isaac"]
    assert len(target_calls) == 3 * 2 + 3 * 3 * 2
    assert len(evaluated["records"]) == 18
    assert all(call["generated"] for call in target_calls)
    # Each task resets the model and capability library to the same Stage 1 state.
    for task_id in TASKS:
        first = next(call for call in target_calls if f"adaptation/{task_id}/trial_00" in str(call["out"]))
        assert first["toolbox"] == STAGE1_CODE
        assert adapted["tasks"][task_id]["trials"][0]["selected"] == "current"
        initial_alignment = next(call for call in runner.calls if
                                 f"adaptation/{task_id}/trial_00/alignment/initial" in str(call["out"]))
        assert initial_alignment["source_config"] == {}
        assert initial_alignment["task"]["scene"]["seed"] == 0
        for trial in range(SMALL["adapt_trials"]):
            prefix = f"adaptation/{task_id}/trial_{trial:02d}"
            target = next(call for call in target_calls if prefix in str(call["out"]))
            replays = [call for call in runner.calls if prefix in str(call["out"]) and
                       ("alignment" in str(call["out"]) or call["out"].name == "replay")]
            assert len(replays) == 5  # Initial + one calibration + current/two candidate screens.
            assert all(not call["generated"] and call["script"] == target["script"] for call in replays)
    # Seed matching across all methods and fresh independent policy generation.
    for task_id in TASKS:
        evaluation = [call for call in target_calls if "evaluation" in str(call["out"]) and
                      call["task"]["name"] == task_id]
        assert len({call["task"]["scene"]["seed"] for call in evaluation}) == 2
        assert len({call["script"] for call in evaluation}) == 6
    assert all("SHOULD_NEVER_REACH_AGENT" not in json.dumps(payload)
               for _, payload, _ in agent.calls)
    report = research.report(run)
    assert report["complete"] and report["target_completed"] == 24
    assert all(row["success_rate"] == 0 for row in report["evaluation"])
    assert (run / "report.md").exists()
    # Completed resumes neither execute trials nor query the model again.
    calls, queries = len(runner.calls), agent.count
    research.discover(runner, agent, run, SMALL)
    research.adapt(runner, agent, run, SMALL)
    research.evaluate(runner, agent, run, SMALL)
    assert (len(runner.calls), agent.count) == (calls, queries)


def test_alignment_receives_library_actually_used_in_fixed_replay(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    ask = agent.ask
    seen_libraries = []

    def inspect_alignment(role, prompt, images=None):
        payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
        if role == "align":
            target = next(call for call in runner.calls
                          if call["engine"] == "isaac" and call["script"] == payload["script"])
            replay = runner.calls[-1]
            assert replay["engine"] == "mujoco"
            assert replay["script"] == target["script"]
            # The current library may be a prior repair, not the Stage 1 baseline.
            assert payload["toolbox"]["toolbox.py"] == target["toolbox"] == replay["toolbox"]
            assert set(payload["toolbox"]) == {"toolbox.py", "skill.md"}
            assert payload["toolbox"]["skill.md"]
            assert "SHOULD_NEVER_REACH_AGENT" not in json.dumps(payload)
            seen_libraries.append(payload["toolbox"]["toolbox.py"])
        if role == "repair_select":
            return {"selected": "candidate_00", "reason": "Exercise next-trial library provenance"}
        return ask(role, prompt, images)

    agent.ask = inspect_alignment
    research.adapt(runner, agent, run, SMALL)
    assert len(seen_libraries) == len(TASKS) * SMALL["adapt_trials"]
    assert STAGE1_CODE in seen_libraries
    assert any("REPAIR" in library for library in seen_libraries)


def test_calibration_history_survives_resume_without_replaying_trials(harness):
    runner, agent, run = harness
    config = {**SMALL, "align_attempts": 2}
    research.discover(runner, agent, run, config)
    ask = agent.ask
    stopped = False

    def interrupt_second_assessment(role, prompt, images=None):
        nonlocal stopped
        payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
        if role == "align" and len(payload["alignment_history"]) == 2 and not stopped:
            history = payload["alignment_history"]
            assert history[0]["source_config"] == {}
            assert history[1]["source_config"] == {"friction_scale": 1.2}
            assert history[1]["proposal"]["reason"] == "Observed grip slip"
            assert history[1]["source"]["status"] == "failed"
            assert "SHOULD_NEVER_REACH_AGENT" not in json.dumps(history)
            stopped = True
            raise RuntimeError("test interruption before second calibration decision")
        return ask(role, prompt, images)

    agent.ask = interrupt_second_assessment
    with pytest.raises(RuntimeError, match="test interruption"):
        research.adapt(runner, agent, run, config)
    completed_paths = [call["out"] for call in runner.calls]
    result = research.adapt(runner, agent, run, config)
    # Every recorded physics call, including those before interruption, ran once.
    paths = [call["out"] for call in runner.calls]
    assert len(paths) == len(set(paths))
    assert all(paths.count(path) == 1 for path in completed_paths)
    assert sum(call["engine"] == "isaac" for call in runner.calls) == 6
    record = research._json(run / "adaptation/far/trial_00/alignment/history.json")
    assert record["status"] == "update_limit_reached_unconfirmed"
    assert record["updates_applied"] == 2
    assert len(record["history"]) == 3
    # The final replay has not been assessed by the align agent.
    assert record["last_assessment"]["assessed_step"] == 1
    assert record["last_assessment"]["response"]["aligned"] is False
    assert "SHOULD_NEVER_REACH_AGENT" not in json.dumps(record)
    for role in ("repair_hypotheses", "repair", "repair_select"):
        payloads = [payload for name, payload, _ in agent.calls if name == role]
        assert payloads
        assert all(p["alignment"]["status"] == "update_limit_reached_unconfirmed" for p in payloads)
        assert all(len(p["alignment"]["history"]) == 3 for p in payloads)
    for task_id in TASKS:
        summary = result["tasks"][task_id]["trials"][0]["alignment"]
        assert summary["history"][0]["source_config"] == {}
        assert summary["history"][1]["proposal"]["reason"] == "Observed grip slip"
    later = [payload for role, payload, _ in agent.calls if role == "align" and payload["task_history"]]
    assert later and all(p["task_history"][0]["alignment"]["updates_applied"] == 2 for p in later)


@pytest.mark.parametrize("mode", ["confirmed", "no_budget", "no_script"])
def test_calibration_confirmation_requires_assessment_of_current_replay(harness, mode):
    runner, agent, run = harness
    config = {**SMALL, "align_attempts": 0 if mode == "no_budget" else 2}
    research.discover(runner, agent, run, config)
    ask = agent.ask

    def assess(role, prompt, images=None):
        payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
        if role == "align" and len(payload["alignment_history"]) == 2:
            return {"aligned": True, "source_config": {"mass_scale": 2.0}, "reason": "Observed behavior matches"}
        return ask(role, prompt, images)

    agent.ask = assess
    if mode == "no_script":
        runner.target_results = ["invalid_policy"] * 6
    research.adapt(runner, agent, run, config)
    record = research._json(run / "adaptation/far/trial_00/alignment/history.json")
    if mode == "confirmed":
        assert record["status"] == "agent_confirmed_aligned"
        assert record["updates_applied"] == 1
        assert len(record["history"]) == 2
        assert record["last_assessment"]["assessed_step"] == 1
        assert record["history"][-1]["source_config"] == {"friction_scale": 1.2}
    else:
        assert record["updates_applied"] == 0
        assert record["last_assessment"] is None
        assert record["status"] == ("fixed_policy_unavailable" if mode == "no_script" else "update_limit_reached_unconfirmed")
        assert len(record["history"]) == (0 if mode == "no_script" else 1)


@pytest.mark.parametrize("failure", ["timeout", "invalid_policy", "generation_error", "interrupted"])
def test_target_failure_consumes_slot_but_pre_execution_infrastructure_can_resume(harness, failure):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    toolbox = run / "frozen" / "stage1"
    runner.target_results = ["infrastructure_error", failure]
    output = run / "target_test"
    with pytest.raises(research.ResearchError, match="Infrastructure failure"):
        research._target(runner, run, "adapt/far/00", toolbox, output)
    ledger = research._json(run / "target_ledger.json")
    assert ledger["slots"]["adapt/far/00"]["state"] == "pending_infrastructure"
    assert research._target(runner, run, "adapt/far/00", toolbox, output)["status"] == failure
    calls = len(runner.calls)
    assert research._target(runner, run, "adapt/far/00", toolbox, output)["status"] == failure
    assert len(runner.calls) == calls
    assert research._json(run / "target_ledger.json")["slots"]["adapt/far/00"]["state"] == "complete"


def test_durable_result_recovers_reservation_without_reexecution(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    toolbox = run / "frozen" / "stage1"
    output = run / "recover"
    ledger = research._json(run / "target_ledger.json")
    ledger["slots"]["adapt/far/00"].update(
        state="running", toolbox_sha256=research.toolbox_hash(toolbox))
    research._save(run / "target_ledger.json", ledger)
    saved_result = {"status": "interrupted", "success": False, "actual_started": True, "steps": 12}
    research._save(output / "result.json", saved_result)
    count = len(runner.calls)
    assert research._target(runner, run, "adapt/far/00", toolbox, output) == saved_result
    assert len(runner.calls) == count
    assert research._json(run / "target_ledger.json")["slots"]["adapt/far/00"]["state"] == "complete"


def test_unresolved_target_is_not_silently_reexecuted(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    ledger = research._json(run / "target_ledger.json")
    ledger["slots"]["adapt/far/00"]["state"] = "running"
    research._save(run / "target_ledger.json", ledger)
    calls = len(runner.calls)
    with pytest.raises(research.ResearchError, match="refusing automatic rerun"):
        research._target(runner, run, "adapt/far/00", run / "frozen" / "stage1", run / "lost")
    assert len(runner.calls) == calls


def test_resume_does_not_verify_frozen_artifact_or_schedule_hashes(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    (run / "frozen" / "initial" / "skill.md").write_text("Updated initial usage notes\n")
    (run / "frozen" / "stage1" / "skill.md").write_text("Updated source usage notes\n")
    selection = research._json(run / "selection_tasks.json")
    selection[0]["description"] = "Updated selection instructions"
    research._save(run / "selection_tasks.json", selection)
    ledger = research._json(run / "target_ledger.json")
    ledger["slots"]["adapt/far/00"]["seed"] += 1
    research._save(run / "target_ledger.json", ledger)
    manifest = research._json(run / "manifest.json")
    manifest["provenance"]["code_sha256"] = "legacy implementation hash"
    research._save(run / "manifest.json", manifest)
    assert research._initialize(run, SMALL) == manifest["config"]
    calls, queries = len(runner.calls), agent.count
    research.discover(runner, agent, run, SMALL)
    assert (len(runner.calls), agent.count) == (calls, queries)


def test_missing_artifacts_and_changed_budgets_still_stop_resume(harness):
    _, _, run = harness
    research._initialize(run, SMALL)
    with pytest.raises(research.ResearchError, match="budgets/seeds"):
        research._initialize(run, {**SMALL, "cycles": SMALL["cycles"] + 1})
    (run / "target_ledger.json").unlink()
    with pytest.raises(research.ResearchError, match="ledger is missing"):
        research._initialize(run, SMALL)
    (run / "frozen/initial/skill.md").unlink()
    with pytest.raises(research.ResearchError, match="artifact missing"):
        research._check_frozen(run / "frozen/initial")


def test_completed_rollouts_ignore_hash_metadata_without_reexecution(harness):
    runner, _, run = harness
    research._initialize(run, SMALL)
    toolbox = run / "frozen/initial"
    target_dir = run / "target"
    research._target(runner, run, "adapt/far/00", toolbox, target_dir)
    result = research._json(target_dir / "result.json")
    result["steps"] += 1
    research._save(target_dir / "result.json", result)
    ledger = research._json(run / "target_ledger.json")
    ledger["slots"]["adapt/far/00"].update(toolbox_sha256="outdated", result_sha256="outdated")
    research._save(run / "target_ledger.json", ledger)
    count = len(runner.calls)
    assert research._target(runner, run, "adapt/far/00", toolbox, target_dir) == result
    assert len(runner.calls) == count
    source_dir = run / "source"
    source_task = task("far", 42)
    expected = research._run(runner, "mujoco", source_task, toolbox, source_dir)
    request = research._json(source_dir / "research_request.json")
    request.update(toolbox_sha256="outdated", script_sha256="outdated")
    research._save(source_dir / "research_request.json", request)
    count = len(runner.calls)
    assert research._run(runner, "mujoco", source_task, toolbox, source_dir) == expected
    assert len(runner.calls) == count
    assert research._json(source_dir / "research_request.json") == request


def test_default_schedule_has_exactly_sixty_target_slots(harness):
    _, _, run = harness
    research._initialize(run, {})
    slots = list(research._json(run / "target_ledger.json")["slots"].values())
    assert len(slots) == 60
    assert sum(slot["phase"] == "adapt" for slot in slots) == 15
    assert sum(slot["phase"] == "evaluate" for slot in slots) == 45


def test_model_and_runtime_provenance_are_immutable_on_resume(harness):
    _, _, run = harness
    config = {**SMALL, "model": "model-a", "codex_version": "codex-test-1", "max_sim_time": 120}
    research._initialize(run, config)
    for changes in ({"model": "model-b"}, {"codex_version": "codex-test-2"}, {"max_sim_time": 30}):
        with pytest.raises(research.ResearchError, match="provenance"):
            research._initialize(run, {**config, **changes})


def test_exhausted_agent_does_not_reserve_target_slot(harness):
    runner, agent, run = harness
    research._initialize(run, SMALL)
    agent.config = {"max_agent_calls": 0}
    runner.agent = agent
    with pytest.raises(research.ResearchError, match="budget exhausted"):
        research._target(runner, run, "adapt/far/00", run / "frozen" / "initial", run / "target")
    assert research._json(run / "target_ledger.json")["slots"]["adapt/far/00"]["state"] == "pending"
    assert not runner.calls


def test_invalid_agent_task_stops_without_marking_discovery_complete(harness):
    runner, agent, run = harness
    agent.ask = lambda **kwargs: {"tasks": ["malformed"]}
    with pytest.raises(research.ResearchError, match="Invalid generated source task"):
        research.discover(runner, agent, run, SMALL)
    assert not (run / "discovery.json").exists()
    assert not runner.calls


def test_real_task_builder_accepts_prompted_plate_override_schema():
    response = {"tasks": [{"name": "lift_green", "description": "Lift green by five centimeters",
        "base_task_id": "far_plate", "seed": 101,
        "scene_overrides": {"plates": [{"name": "green", "position": [0.51, 0.19, 0.004],
                            "radius": .07, "mass": .15, "friction": .6},
                            {"name": "red", "position": [.32, .26, .004]}],
                            "tote": {"position": [.39, -.26, 0], "yaw": 0}},
        "predicate": {"all": [{"kind": "lift", "object": "green", "min": .05}]}}]}
    built = research._tasks_from_response(response, 1)[0]
    assert built["scene"]["plates"][0]["position"] == [.51, .19, .004]
    assert built["scene"]["plates"][1]["color"] == [.8, .12, .1]
    assert built["scene"]["plates"][0]["rim_height"] > 0


def test_report_separates_budget_completion_from_valid_physics(harness):
    _, _, run = harness
    research._initialize(run, SMALL)
    ledger = research._json(run / "target_ledger.json")
    for slot in ledger["slots"].values():
        slot.update(state="complete", status="invalid_policy", actual_started=False, success=False)
    research._save(run / "target_ledger.json", ledger)
    source = run / "source_error"
    research._save(source / "research_request.json", {"engine": "mujoco"})
    research._save(source / "result.json", {"status": "invalid_policy", "actual_started": False})
    research._save(source / "research_attempts" / "000.json", {"status": "infrastructure_error"})
    research._save(source / "research_attempts" / "001.json", {"status": "invalid_policy"})
    result = research.report(run)
    assert result["protocol_complete"]
    assert not result["complete"] and not result["engineering_evidence_complete"]
    assert result["source_valid_physics_rollouts"] == 0
    assert result["source_rollout_statuses"] == {"invalid_policy": 1}
    assert result["source_attempt_statuses"]["infrastructure_error"] == 1


@pytest.mark.parametrize("code", [
    "x = 1\n", INITIAL_CODE + "import os\n", INITIAL_CODE + "x = np.load('truth.npy')\n",
])
def test_toolbox_runtime_contract_is_validated_before_physics(tmp_path, code):
    with pytest.raises(research.ResearchError, match="Invalid proposed toolbox"):
        research._write_toolbox(tmp_path / "candidate", {"toolbox.py": code, "skill.md": "notes"})


@pytest.mark.parametrize("signature,decorator", [
    ("", ""),
    ("self", ""),
    ("self, **kwargs", ""),
    ("self, *, robot", ""),
    ("self, robot, required", ""),
    ("self, robot, required, /", ""),
    ("self, robot, *, required", ""),
    ("self, *args, required", ""),
    ("robot, required", "staticmethod"),
    ("cls", "classmethod"),
])
def test_toolbox_rejects_constructor_argument_mismatch(tmp_path, signature, decorator):
    prefix = f"    @{decorator}\n" if decorator else ""
    code = f"class Toolbox:\n{prefix}    def __init__({signature}):\n        pass\n"
    destination = tmp_path / "candidate"
    with pytest.raises(research.InvalidToolbox, match=r"must accept Toolbox\(robot\)"):
        research._write_toolbox(destination, {"toolbox.py": code, "skill.md": "notes"})
    assert not destination.exists()


@pytest.mark.parametrize("signature,decorator", [
    ("self, robot", ""),
    ("self, arm", ""),
    ("self, robot=None", ""),
    ("self, robot, option=None", ""),
    ("self, robot, /", ""),
    ("self, /, robot", ""),
    ("self, robot, *, option=None", ""),
    ("self, *args", ""),
    ("*args, **kwargs", ""),
    ("robot", "staticmethod"),
    ("cls, robot", "classmethod"),
])
def test_toolbox_accepts_compatible_constructor_arguments(tmp_path, signature, decorator):
    prefix = f"    @{decorator}\n" if decorator else ""
    code = f"class Toolbox:\n{prefix}    def __init__({signature}):\n        pass\n"
    destination = tmp_path / "candidate"
    research._write_toolbox(destination, {"toolbox.py": code, "skill.md": "notes"})
    # Independently verify that accepted signatures match the runtime call.
    namespace = {}
    exec(code, namespace)
    namespace["Toolbox"](object())
    assert (destination / "toolbox.py").read_text() == code


def test_toolbox_constructor_validation_does_not_execute_candidate(tmp_path):
    code = "raise RuntimeError('must not execute during validation')\n" + INITIAL_CODE
    research._write_toolbox(tmp_path / "candidate", {"toolbox.py": code, "skill.md": "notes"})


def test_toolbox_checks_last_constructor_definition(tmp_path):
    code = INITIAL_CODE + "    def __init__(self):\n        pass\n"
    with pytest.raises(research.InvalidToolbox, match=r"must accept Toolbox\(robot\)"):
        research._write_toolbox(tmp_path / "candidate", {"toolbox.py": code, "skill.md": "notes"})


@pytest.mark.parametrize("status,started,expected", [
    ("invalid_policy", False, None), ("invalid_policy", True, None),
    ("generation_error", False, None), ("failed", False, None),
    ("failed", True, "print('real execution')\n"),
    ("interrupted", True, "print('real execution')\n"),
])
def test_only_actually_executed_policies_can_be_fixed_replayed(tmp_path, status, started, expected):
    script = tmp_path / "policy.py"
    script.write_text("print('real execution')\n")
    assert research._executed_script({"status": status, "actual_started": started,
                                     "script_path": str(script)}) == expected


@pytest.mark.parametrize("member_zero_valid", [0, 1])
def test_source_selection_rejects_zero_execution_and_prefers_more_valid_trials(harness, member_zero_valid):
    runner, agent, run = harness
    original = runner.rollout
    def rollout(**kwargs):
        result = original(**kwargs)
        path = kwargs["out"]
        if "selection" in path.parts:
            zero = "member_00" in path.parts
            valid = not zero or int(path.name) < member_zero_valid
            result.update(success=False, actual_started=valid,
                          status="failed" if valid else "invalid_policy",
                          steps=1 if zero and valid else 0 if zero else 25)
        return result
    runner.rollout = rollout
    result = research.discover(runner, agent, run, SMALL)
    assert result["winner"] == 1
    assert result["members"][0]["valid_executions"] == member_zero_valid
    assert result["members"][1]["valid_executions"] == 3
    assert result["members"][1]["mean_valid_steps"] == 25


def test_source_policy_preparation_parallel_bounded_and_resume_safe(harness):
    runner, _, run = harness
    research._initialize(run, SMALL)
    runner.config = {"source_policy_workers": 99}
    barrier = threading.Barrier(3)
    lock = threading.Lock()
    counts = {"active": 0, "peak": 0, "prepared": 0}
    prepared_calls = []
    def prepare(task, toolbox, out):
        with lock:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
            prepared_calls.append(int(out.name))
        if int(out.name) < 3:
            barrier.wait(timeout=3)
        research._save(out / "policy_generation.json", {"status": "ready"})
        with lock:
            counts["active"] -= 1
            counts["prepared"] += 1
        return {"status": "ready"}
    runner.prepare_policy = prepare
    original = runner.rollout
    def rollout(**kwargs):
        assert counts["prepared"] == 5 and counts["active"] == 0
        return original(**kwargs)
    runner.rollout = rollout
    tasks = [task("far", index) for index in range(5)]
    toolbox = run / "frozen" / "initial"
    output = run / "suite"
    research._suite(runner, tasks, toolbox, output)
    assert counts["peak"] == 3 and len(prepared_calls) == 5
    assert [call["task"]["scene"]["seed"] for call in runner.calls] == list(range(5))
    research._suite(runner, tasks, toolbox, output)
    assert len(prepared_calls) == 5 and len(runner.calls) == 5
    research._suite(runner, tasks[:1], toolbox, run / "fixed", script="print('same')\n")
    assert len(prepared_calls) == 5 and runner.calls[-1]["generated"] is False


def test_preparation_exception_surfaces_before_any_physics(harness):
    runner, _, run = harness
    research._initialize(run, SMALL)
    runner.config = {"source_policy_workers": 3}
    def prepare(task, toolbox, out):
        raise ValueError("generation checkpoint signature changed")
    runner.prepare_policy = prepare
    with pytest.raises(research.ResearchError, match="preparation failed"):
        research._suite(runner, [task("far", 0)], run / "frozen" / "initial", run / "source")
    assert not runner.calls


def test_prepared_policy_can_execute_when_last_model_call_used_budget(harness):
    runner, agent, run = harness
    research._initialize(run, SMALL)
    runner.agent = agent
    agent.count = 1
    agent.config = {"max_agent_calls": 1}
    output = run / "source"
    research._save(output / "policy_generation.json", {"status": "ready"})
    result = research._run(runner, "mujoco", task("far", 0), run / "frozen" / "initial", output)
    assert result["status"] == "failed" and len(runner.calls) == 1


@pytest.mark.parametrize("value", [
    {"friction_scale": -1}, {"camera_offset": [0, 0, 1]},
    {"friction_scale": .029999}, {"friction_scale": 2.01},
    {"tcp_offset": [0, 0, 0]}, {"mass_scale": float("nan")},
])
def test_source_calibration_rejects_unbounded_or_unsupported_parameters(value):
    with pytest.raises(research.ResearchError):
        research._calibration(value)


def test_low_friction_calibration_reaches_source_screening_only(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    ask = agent.ask

    def low_friction(role, prompt, images=None):
        if role == "align":
            assert "friction_scale [0.03,2]" in prompt
            return {"aligned": False, "source_config": {"friction_scale": .1},
                    "reason": "Test contact-slip hypothesis from public replay evidence"}
        return ask(role, prompt, images)

    agent.ask = low_friction
    offset = len(runner.calls)
    research.adapt(runner, agent, run, SMALL)
    calls = runner.calls[offset:]
    screening = [call for call in calls if "screen" in call["out"].parts]
    assert screening and all(call["source_config"]["friction_scale"] == .1 for call in screening)
    proposals = [payload for role, payload, _ in agent.calls if role == "screen_tasks"]
    assert proposals and all(payload["source_config"] == {"friction_scale": .1}
                             for payload in proposals)
    target = [call for call in calls if call["engine"] == "isaac"]
    assert target and all(not call.get("source_config") for call in target)


def test_actuator_calibration_profile_is_frozen_and_source_only(harness):
    runner, agent, run = harness
    config = {**SMALL, "source_calibration_profile": "actuator_v1"}
    research.discover(runner, agent, run, config)
    with pytest.raises(research.ResearchError, match="calibration profile changed"):
        research._initialize(run, SMALL)
    ask = agent.ask
    parameters = {"arm_settle_s": .15, "gripper_stiffness_scale": .5,
                  "gripper_damping_scale": 2.}

    def calibrate(role, prompt, images=None):
        if role == "align":
            assert "arm_settle_s [0,0.5]" in prompt
            assert "per-finger force cap" in prompt
            return {"aligned": False, "source_config": parameters,
                    "reason": "Test distinct arm timing and finger response hypotheses"}
        return ask(role, prompt, images)

    agent.ask = calibrate
    offset = len(runner.calls)
    research.adapt(runner, agent, run, config)
    calls = runner.calls[offset:]
    screening = [call for call in calls if "screen" in call["out"].parts]
    assert screening and all(call['source_config'] == parameters for call in screening)
    target = [call for call in calls if call['engine'] == 'isaac']
    assert target and all(not call.get('source_config') for call in target)
    offset = len(runner.calls)
    research.evaluate(runner, agent, run, config)
    assert runner.calls[offset:]
    assert all(not call.get('source_config') for call in runner.calls[offset:])


def test_basic_calibration_run_cannot_enable_actuator_profile_on_resume(harness):
    _, _, run = harness
    research._initialize(run, SMALL)
    with pytest.raises(research.ResearchError, match="calibration profile changed"):
        research._initialize(run, {**SMALL, 'source_calibration_profile': 'actuator_v1'})


@pytest.mark.parametrize('mode', ['legacy', 'interactive'])
def test_discovery_knows_complete_release_contract_without_replacing_subtask_predicates(harness, monkeypatch, mode):
    from simex.tasks import TASK_IDS, make_task, source_task_from_spec
    runner, agent, run = harness
    monkeypatch.setattr(research, 'TASK_IDS', TASK_IDS)
    monkeypatch.setattr(research, 'make_task', make_task)
    monkeypatch.setattr(research, 'source_task_from_spec', lambda spec: source_task_from_spec(
        {**spec, 'base_task_id': {'far': 'far_plate', 'near': 'far_to_near', 'offset': 'shifted_single'}.get(
            spec.get('base_task_id'), spec.get('base_task_id', 'far_plate'))}))
    ask = agent.ask

    def finish_coding(role, prompt, images=None):
        if role == 'discover_code':
            agent.count += 1
            payload = json.loads(prompt.split('INPUT JSON:\n', 1)[1])
            agent.calls.append((role, payload, images))
            return {'action': 'finish', 'reason': 'Contract propagation test only'}
        return ask(role, prompt, images)

    agent.ask = finish_coding
    config = {**SMALL, 'placement_contract': 'released_placement_v1', 'discovery_mode': mode}
    research.discover(runner, agent, run, config)
    expected_roles = {'discover_probe', 'discover_focus',
                      'discover_code' if mode == 'interactive' else 'discover_revision'}
    for role in expected_roles:
        requests = [payload for kind, payload, _ in agent.calls if kind == role]
        assert requests
        for payload in requests:
            reference = payload['complete_placement_reference']
            assert reference['contract'] == 'released_placement_v1'
            assert {'kind': 'released', 'object': 'green'} in reference['predicate']['all']
            assert 'not a fixed curriculum' in reference['scope']
            assert 'at least 5 cm' in reference['description']
    probes = [call for call in runner.calls if call['out'].parent.name == 'probes']
    assert probes
    assert all(call['task']['predicate'] == {'all': [{'kind': 'lift', 'object': 'green', 'min': .05}]}
               for call in probes)
    assert all(call['engine'] == 'mujoco' for call in runner.calls)


@pytest.mark.parametrize('full_success,auxiliary_success,expected', [
    (None, 1, 1), (0, 1, 0), (None, None, 0),
])
def test_selection_ranks_full_then_common_auxiliary_and_never_fast_failure(
        harness, full_success, auxiliary_success, expected):
    runner, agent, run = harness
    original = runner.rollout
    def rollout(**kwargs):
        result = original(**kwargs)
        parts = kwargs['out'].parts
        member = 0 if 'member_00' in parts else 1
        passed = ('selection' in parts and member == full_success or
                  'auxiliary_selection' in parts and member == auxiliary_success)
        result.update(success=passed, status='success' if passed else 'failed',
                      steps=500 if member == 0 else 1)
        return result
    runner.rollout = rollout
    result = research.discover(runner, agent, run, SMALL)
    assert result['winner'] == expected
    assert result['process_complete']
    assert result['full_task_verified'] is (full_success is not None)
    report = research.report(run)
    assert report['stage1']['full_task_predicate_verified'] is (full_success is not None)
    assert report['stage1']['full_task_verification_scope'] == 'frozen_task_predicates'
    assert report['stage1']['released_placement_verified'] is None
    text = (run / 'report.md').read_text()
    assert '得分不等同于完成释放放置' in text
    assert '已出现独立完整入箱成功' not in text
    common = research._json(run / 'auxiliary_selection_tasks.json')
    assert [(t['predicate']['all'][0]['kind'], t['predicate']['all'][0]['min']) for t in common] == [
        ('lift', .05), ('lift', .05), ('displaced', .03), ('displaced', .03)]
    auxiliary_calls = [call for call in runner.calls if 'auxiliary_selection' in call['out'].parts]
    assert [call['task'] for call in auxiliary_calls[:4]] == [call['task'] for call in auxiliary_calls[4:]]
    assert all(call['generated'] for call in auxiliary_calls)
    last_discovery = max(i for i, call in enumerate(runner.calls) if 'discovery' in call['out'].parts)
    assert all(i > last_discovery for i, call in enumerate(runner.calls) if 'auxiliary_selection' in call['out'].parts)
    assert all('selection_lift_' not in json.dumps(payload) for _, payload, _ in agent.calls)


def test_completed_discovery_does_not_verify_auxiliary_schedule_hash(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    path = run / 'auxiliary_selection_tasks.json'
    tasks = research._json(path)
    tasks[0]['predicate']['all'][0]['min'] = .001
    research._save(path, tasks)
    count = len(runner.calls), agent.count
    assert research.discover(runner, agent, run, SMALL)["process_complete"]
    assert (len(runner.calls), agent.count) == count


def test_frozen_peer_ideas_survive_focus_and_legacy_revision_without_later_round_access(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, {**SMALL, "cycles": 2})
    expected = [research._json(run / f"discovery/member_{member:02d}/cycle_{cycle:02d}/peer_summaries.json")
                for cycle in range(2) for member in range(2)]
    assert expected[:2] == [[], []]
    for member, summaries in enumerate(expected[2:]):
        assert summaries[0]["member"] == 1 - member
        assert [s["cycle"] for s in summaries[0]["summaries"]] == [0]
        assert "toolbox.py" not in json.dumps(summaries)
    for role in ("discover_probe", "discover_focus", "discover_revision"):
        seen = [payload["peer_experiment_summaries"] for name, payload, _ in agent.calls if name == role]
        assert seen == expected


def test_stage1_report_shows_all_nine_trials_and_keeps_stage1_call_count(harness):
    runner, agent, run = harness
    original = runner.rollout
    def rollout(**kwargs):
        result = original(**kwargs)
        result['public_evidence']['video'] = str(kwargs['out'] / 'public.mp4')
        return result
    runner.rollout = rollout
    discovery = research.discover(runner, agent, run, {**SMALL, 'selection_trials': 3})
    saved_count = discovery['agent_count']
    agent.count += 100
    result = research.report(run)['stage1']
    assert result['process_complete'] and not result['full_task_verified']
    assert result['agent_calls'] == saved_count
    assert len(result['members']) == 2
    assert all(len(member['selection_results']) == 9 for member in result['members'])
    assert all(record['video'].endswith('public.mp4') for member in result['members']
               for record in member['selection_results'])
    assert '尚未验证完整入箱能力' in (run / 'report.md').read_text()


def test_old_protocol_manifest_requires_new_run(harness):
    _, _, run = harness
    research._initialize(run, SMALL)
    manifest = research._json(run / 'manifest.json')
    manifest['format_version'] = 2
    research._save(run / 'manifest.json', manifest)
    with pytest.raises(research.ResearchError, match='create a new run'):
        research._initialize(run, SMALL)
