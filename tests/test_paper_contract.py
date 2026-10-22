"""Paper-level contracts, using deterministic stand-ins rather than research trials."""

import json
from pathlib import Path

from simex import research
from simex.agent import Agent
from test_research import INITIAL_CODE, SMALL, STAGE1_CODE, TASKS, harness


def test_fifth_repair_is_frozen_without_sixth_target_and_evaluation_has_no_feedback(harness):
    runner, agent, run = harness
    config = {**SMALL, "adapt_trials": 5, "eval_trials": 5}
    research.discover(runner, agent, run, config)
    # Stage 1's capability-bank replay is another fresh CaP task, unlike the
    # exact executed-script replay required specifically for Stage 2 alignment.
    assert all(call["generated"] for call in runner.calls)
    original = agent.ask

    def ask(**kwargs):
        answer = original(**kwargs)
        payload = json.loads(kwargs["prompt"].split("INPUT JSON:\n", 1)[1])
        if kwargs["role"] == "repair":
            trial = len(payload["task_history"])
            return {"toolbox.py": INITIAL_CODE + f"TRIAL_REPAIR = {trial}\n",
                    "skill.md": f"Repair from adaptation trial {trial}\n"}
        if kwargs["role"] == "repair_select":
            return {"selected": "candidate_00", "reason": "Agent-selected repair"}
        return answer

    agent.ask = ask
    research.adapt(runner, agent, run, config)
    target_calls = [call for call in runner.calls if call["engine"] == "isaac"]
    assert len(target_calls) == 15
    for task_id in TASKS:
        calls = [call for call in target_calls if f"adaptation/{task_id}/" in str(call["out"])]
        assert len(calls) == 5
        assert calls[0]["toolbox"] == STAGE1_CODE
        assert "TRIAL_REPAIR = 3" in calls[-1]["toolbox"]
        frozen = run / "frozen" / "adapted" / task_id
        assert "TRIAL_REPAIR = 4" in (frozen / "toolbox.py").read_text()
        assert research.toolbox_hash(frozen) == research._json(frozen / "frozen.json")["sha256"]

    # Evaluation can generate policies, but it must never call an optimizer or
    # expose the final scores to adaptation before choosing its frozen library.
    decisions_before_evaluation = len(agent.calls)
    research.evaluate(runner, agent, run, config)
    assert len(agent.calls) == decisions_before_evaluation
    target_calls = [call for call in runner.calls if call["engine"] == "isaac"]
    assert len(target_calls) == 60
    evaluation = [call for call in target_calls if "evaluation/" in str(call["out"])]
    assert len(evaluation) == 45
    assert all(call["generated"] for call in evaluation)
    for call in evaluation:
        method = call["out"].relative_to(run / "evaluation").parts[0]
        if method == "simex":
            assert "TRIAL_REPAIR = 4" in call["toolbox"]
        elif method == "stage1_transfer":
            assert call["toolbox"] == STAGE1_CODE
        else:
            assert call["toolbox"] == INITIAL_CODE
    for task_id in TASKS:
        methods = {}
        for call in evaluation:
            if call["task"]["name"] == task_id:
                method = call["out"].parts[-3]
                methods.setdefault(method, []).append(call["task"]["scene"]["seed"])
        assert set(methods) == set(research.METHODS)
        assert methods["direct_cap"] == methods["stage1_transfer"] == methods["simex"]


def test_one_shot_policy_context_contains_only_task_text_and_current_capabilities(tmp_path):
    toolbox = tmp_path / "toolbox"
    toolbox.mkdir()
    (toolbox / "toolbox.py").write_text(INITIAL_CODE)
    (toolbox / "skill.md").write_text("CURRENT_CAPABILITY_NOTES\n")
    agent = Agent.__new__(Agent)
    captured = []

    def call(role, prompt, *args, **kwargs):
        captured.append((role, prompt))
        return "===PLAN START===\nObserve.\n===PLAN END===\n===SCRIPT START===\nprint('run')\n===SCRIPT END==="

    agent._call = call
    task = {"name": "Transfer", "description": "Move the green plate into the box.",
            "scene": {"seed": "PRIVATE_SEED", "position": "PRIVATE_POSE"},
            "predicate": {"private_rule": "PRIVATE_EVALUATOR"},
            "previous_policy": "PRIVATE_PRIOR_SCRIPT"}
    agent.policy(task, toolbox, tmp_path / "trial")
    assert len(captured) == 1
    role, prompt = captured[0]
    assert role == "policy"
    assert task["description"] in prompt
    assert "CURRENT_CAPABILITY_NOTES" in prompt
    assert INITIAL_CODE in prompt
    assert "PRIVATE_" not in prompt


def test_repair_selector_gets_source_task_and_fixed_script_without_hidden_target_scene(harness, monkeypatch):
    runner, agent, run = harness

    def source_task(task_id, seed, predicate=None):
        return {"name": task_id, "description": f"Move the green plate: {task_id}",
                "scene": {"task_id": task_id, "seed": seed, "public_source": True},
                "predicate": predicate or {"all": [{"kind": "inside", "object": "green"}]}}

    # Target tasks retain the harness's SECRET field. Reconstructed and generated
    # source scenes intentionally have a distinguishable public source marker.
    monkeypatch.setattr(research, "_source_reconstruction",
                        lambda task_id, evidence: source_task(task_id, 0))
    monkeypatch.setattr(research, "source_task_from_spec",
                        lambda spec: source_task(spec["base_task_id"], spec["seed"], spec.get("predicate")))
    research.discover(runner, agent, run, SMALL)
    research.adapt(runner, agent, run, SMALL)
    decisions = [payload for role, payload, _ in agent.calls if role == "repair_select"]
    targets = [call for call in runner.calls if call["engine"] == "isaac"]
    assert len(decisions) == len(targets) == len(TASKS) * SMALL["adapt_trials"]
    for payload, target in zip(decisions, targets):
        assert target["task"]["scene"]["private_target_marker"] == "SECRET"
        assert payload["source_task"]["scene"]["public_source"] is True
        assert payload["source_task"]["scene"]["seed"] == 0
        assert payload["fixed_script"] == target["script"]
        assert len(payload["generalization_tasks"]) == SMALL["screen_tasks"]
        assert all(task["scene"]["public_source"] for task in payload["generalization_tasks"])
        assert all(task["predicate"] == payload["source_task"]["predicate"]
                   for task in payload["generalization_tasks"])
        assert payload["toolboxes"]["current"]["toolbox.py"] == STAGE1_CODE
        assert set(payload["toolboxes"]) == set(payload["available_choices"])
        assert "SECRET" not in json.dumps(payload)
        assert "SHOULD_NEVER_REACH_AGENT" not in json.dumps(payload)
