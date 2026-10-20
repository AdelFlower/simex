"""Regression coverage for rejection feedback, candidate failures and public evidence."""
import json
from pathlib import Path

import pytest
from PIL import Image

from simex import research
from simex.evidence import summarize_public_evidence
from test_research import FakeAgent, INITIAL_CODE, SMALL, STAGE1_CODE, TASKS, harness


def test_rejected_revision_feedback_changes_next_request_and_keeps_baseline(harness):
    runner, agent, run = harness
    config = {**SMALL, "revisions": 2, "population": 1}
    original = agent.ask
    def ask(**kwargs):
        answer = original(**kwargs)
        if kwargs["role"] == "discover_select":
            return {"accept": False, "reason": "The measured approach misses the rim"}
        return answer
    agent.ask = ask
    research.discover(runner, agent, run, config)
    revisions = [payload for role, payload, _ in agent.calls if role == "discover_revision"]
    assert len(revisions) == 2
    first, second = revisions
    assert first["prior_attempts"] == []
    history = second["prior_attempts"][0]
    assert history["accepted"] is False
    assert history["reason"] == "The measured approach misses the rim"
    assert history["candidate_toolbox"]["toolbox.py"] == STAGE1_CODE
    assert history["results"][0]["status"] == "failed"
    assert first["toolbox"] == second["toolbox"]
    assert first["toolbox"]["toolbox.py"] == INITIAL_CODE
    checkpoints = sorted((run / "discovery/member_00/cycle_00").glob("revision_*/proposal.json"))
    assert len({json.loads(path.read_text())["request_sha256"] for path in checkpoints}) == 2


@pytest.mark.parametrize("invalid_code", [
    "class Toolbox(:\n",
    "class Toolbox:\n    def __init__(self):\n        pass\n",
    "return 1\n" + INITIAL_CODE,
])
def test_invalid_revision_consumes_budget_and_resume_reaches_next_candidate(harness, invalid_code):
    runner, agent, run = harness
    config = {**SMALL, "revisions": 2, "population": 1}
    original = agent.ask
    attempts = []
    fail_once = True
    def ask(**kwargs):
        nonlocal fail_once
        if kwargs["role"] == "discover_revision":
            payload = json.loads(kwargs["prompt"].split("INPUT JSON:\n", 1)[1])
            if not payload["prior_attempts"]:
                attempts.append("invalid")
                return {"toolbox.py": invalid_code, "skill.md": "incomplete"}
            if fail_once:
                fail_once = False
                raise RuntimeError("simulated interruption after invalid candidate")
            attempts.append("valid")
            assert payload["prior_attempts"][0]["validation"]["status"] == "invalid_candidate"
        return original(**kwargs)
    agent.ask = ask
    with pytest.raises(RuntimeError, match="simulated interruption"):
        research.discover(runner, agent, run, config)
    research.discover(runner, agent, run, config)
    assert attempts == ["invalid", "valid"]
    revision = run / "discovery/member_00/cycle_00/revision_00"
    assert json.loads((revision / "validation.json").read_text())["status"] == "invalid_candidate"
    assert not (revision / "tests").exists()
    assert not (revision / "toolbox").exists()
    count = len(runner.calls), agent.count
    research.discover(runner, agent, run, config)
    assert (len(runner.calls), agent.count) == count


def test_stage2_all_invalid_candidates_retain_current_without_resampling(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    original = agent.ask
    def ask(**kwargs):
        answer = original(**kwargs)
        if kwargs["role"] == "repair":
            return {"toolbox.py": "import os\n", "skill.md": "Invalid candidate"}
        return answer
    agent.ask = ask
    result = research.adapt(runner, agent, run, SMALL)
    selections = [payload for role, payload, _ in agent.calls if role == "repair_select"]
    assert len(selections) == len(TASKS) * SMALL["adapt_trials"]
    for payload in selections:
        assert payload["available_choices"] == ["current"]
        assert set(payload["screen"]) == {"current"}
        assert set(payload["hypotheses"]) == {"candidate_00", "candidate_01"}
        assert all(c["validation"]["status"] == "invalid_candidate" for c in payload["candidates"].values())
    assert all(t["selected"] == "current" for value in result["tasks"].values() for t in value["trials"])
    assert len([1 for role, _, _ in agent.calls if role == "repair"]) == 12
    assert not any("candidate_" in str(call["out"]) for call in runner.calls)
    count = len(runner.calls), agent.count
    research.adapt(runner, agent, run, SMALL)
    assert (len(runner.calls), agent.count) == count


def test_stage2_rejection_history_is_task_local_and_public(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    research.adapt(runner, agent, run, SMALL)
    requests = [payload for role, payload, _ in agent.calls if role == "repair_hypotheses"]
    for index, task in enumerate(TASKS):
        first, second = requests[index * 2:index * 2 + 2]
        assert first["task_history"] == []
        assert len(second["task_history"]) == 1
        previous = second["task_history"][0]
        assert previous["selected"] == "current"
        assert previous["reason"] == "Target evidence contradicts simulated success"
        assert previous["candidates"]["candidate_00"]["hypothesis"] == "Grip rim"
        assert "+REPAIR" in previous["candidates"]["candidate_00"]["changes"]["toolbox.py"]
        assert previous["screen"]["candidate_00"]["fixed_replay"]["success"] is True
        assert "SHOULD_NEVER_REACH_AGENT" not in json.dumps(second)
        assert "SECRET" not in json.dumps(second)


def test_invalid_candidate_does_not_hide_immutable_artifact_change(tmp_path):
    proposal = {"toolbox.py": INITIAL_CODE, "skill.md": "notes"}
    research._candidate(proposal, tmp_path)
    with pytest.raises(research.ResearchError, match="Immutable"):
        research._candidate({**proposal, "toolbox.py": STAGE1_CODE}, tmp_path)


@pytest.mark.parametrize("invalid_code", [
    "return 1\n",
    "break\n",
    "def helper(value, value):\n    pass\n",
    "def helper():\n    nonlocal missing\n",
])
def test_candidate_rejects_compile_errors_before_writing_toolbox(tmp_path, invalid_code):
    candidate, outcome = research._candidate(
        {"toolbox.py": invalid_code + INITIAL_CODE, "skill.md": "notes"}, tmp_path)
    assert candidate is None
    assert outcome["status"] == "invalid_candidate"
    assert "Invalid proposed toolbox" in outcome["reason"]
    assert json.loads((tmp_path / "validation.json").read_text()) == outcome
    assert not (tmp_path / "toolbox").exists()


@pytest.mark.parametrize("before,after,body", [
    ("old", "new", "-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n"),
    ("old\n", "new", "-old\n+new\n\\ No newline at end of file\n"),
    ("old", "new\n", "-old\n\\ No newline at end of file\n+new\n"),
    ("old\n", "new\n", "-old\n+new\n"),
    ("same", "same\n", "-same\n\\ No newline at end of file\n+same\n"),
    ("same\n", "same", "-same\n+same\n\\ No newline at end of file\n"),
    ("", "new", "+new\n\\ No newline at end of file\n"),
    ("old", "", "-old\n\\ No newline at end of file\n"),
    ("old\nlast", "new\nlast", "-old\n+new\n last\n\\ No newline at end of file\n"),
    ("same", "same", ""),
])
def test_changes_preserves_diff_lines_and_final_newline_information(tmp_path, before, after, body):
    source = tmp_path / "skill.md"
    source.write_text(before)
    proposal = {"skill.md": after}
    diff = research._changes(tmp_path, proposal)["skill.md"]
    if body:
        assert diff.startswith("--- current/skill.md\n+++ candidate/skill.md\n@@ ")
        assert diff.split("\n", 3)[3] == body
    else:
        assert diff == ""
    assert source.read_text() == before
    assert proposal == {"skill.md": after}


def _rollout_with_images(out):
    public = out / "public"
    public.mkdir(parents=True)
    frames = []
    for i in range(8):
        path = public / f"keyframe_{i:05d}.png"
        Image.new("RGB", (32, 24), (20 * i, 30, 60)).save(path)
        frames.append(str(path))
    states = [{"sim_time": i / 4, "ee_position": [0.4, 0.2, i / 10], "gripper_width": 0.08 - i / 100}
              for i in range(8)]
    (public / "robot_trace.jsonl").write_text("\n".join(json.dumps(s) for s in states))
    return {"status": "failed", "success": False, "output_dir": str(out),
            "public_evidence": {"keyframes": frames, "read_images": [],
                "robot_trace_path": str(public / "robot_trace.jsonl"),
                "private_metrics": "SECRET", "trace_tail": states[-2:],
                "actions": [{"method": "gripper", "kwargs": {"width": 0.02},
                             "start": states[1], "end": states[4]}]}}


def test_every_screen_rollout_gets_image_and_full_motion_summary(tmp_path):
    payload = {"screen": {}}
    for candidate in ("current", "candidate_00", "candidate_01"):
        prefix = tmp_path / "screen" / candidate
        payload["screen"][candidate] = {
            "fixed_replay": research._public(_rollout_with_images(prefix / "replay")),
            "generalization": [research._public(_rollout_with_images(prefix / "generalization" / str(i)))
                               for i in range(3)]}
    attached = research._images(payload)
    assert len(attached) == 12
    assert len([p for p in attached if "/replay/" in p]) == 3
    for group in payload["screen"].values():
        for record in [group["fixed_replay"], *group["generalization"]]:
            evidence = record["evidence"]
            assert evidence["motion_summary"]["samples"] == 8
            assert evidence["motion_summary"]["initial"]["sim_time"] == 0
            assert evidence["motion_summary"]["final"]["sim_time"] == 1.75
            assert len(evidence["actions"]) == 1
            assert len(evidence["phase_images"]) == 1
    assert "SECRET" not in json.dumps(payload)


@pytest.mark.parametrize("use_symlink", [False, True])
def test_public_summary_never_reads_private_trace_or_image(tmp_path, use_symlink):
    out = tmp_path / "trial"
    (out / "public").mkdir(parents=True)
    private = out / "private"
    private.mkdir()
    trace = private / "metrics.json"
    trace.write_text("NOT JSON AND MUST NOT BE READ")
    image = private / "truth.png"
    Image.new("RGB", (20, 20), "red").save(image)
    if use_symlink:
        (out / "public/trace.jsonl").symlink_to(trace)
        (out / "public/frame.png").symlink_to(image)
        trace, image = out / "public/trace.jsonl", out / "public/frame.png"
    result = summarize_public_evidence({"robot_trace_path": str(trace), "keyframes": [str(image)],
        "read_images": [str(image)], "action_images": [{"path": str(image), "sim_time": 0}]}, str(out))
    assert result["motion_summary"]["samples"] == 0
    assert "robot_trace" not in result
    assert result["keyframes"] == result["read_images"] == result["action_images"] == []
    assert result["phase_images"] == []
    assert not (out / "public/research_contact_sheet.jpg").exists()


def test_video_contact_sheet_covers_motion_phases(tmp_path):
    import imageio.v2 as imageio
    import numpy as np
    result = _rollout_with_images(tmp_path / "trial")
    public = tmp_path / "trial/public"
    video = public / "overhead.mp4"
    with imageio.get_writer(str(video), fps=4, macro_block_size=1) as writer:
        for i in range(8):
            writer.append_data(np.full((32, 48, 3), i * 25, dtype=np.uint8))
    result["public_evidence"]["video"] = str(video)
    evidence = research._public(result)["evidence"]
    labels = [phase["phase"] for phase in evidence["visual_phases"]]
    assert labels == ["initial", "before first close", "after first close", "highest TCP after close", "final"]
    assert evidence["visual_phases"][-1]["sim_time"] == 1.75
    assert Path(evidence["phase_images"][0]).exists()


def test_worker_action_schema_and_boundaries_are_preserved(tmp_path):
    result = _rollout_with_images(tmp_path / "trial")
    evidence = result["public_evidence"]
    previous = evidence["actions"][0]
    evidence["actions"] = [{"method": previous["method"], "arguments": previous["kwargs"],
                            "before": previous["start"], "after": previous["end"]}]
    evidence["action_images"] = [
        {"path": path, "sim_time": i / 4, "method": "gripper", "action": i // 2, "phase": "before"}
        for i, path in enumerate(evidence["keyframes"])]
    public = research._public(result)["evidence"]
    assert public["actions"][0]["kwargs"] == {"width": 0.02}
    assert public["actions"][0]["end"]["sim_time"] == 1
    assert "after first close" in [p["phase"] for p in public["visual_phases"]]


def test_internal_diagnostics_do_not_enter_optimizer_evidence(tmp_path):
    result = _rollout_with_images(tmp_path / "trial")
    result["status"] = "interrupted"
    evidence = result["public_evidence"]
    evidence.update(reason="backend failure PRIVATE_ASSET_PATH", stdout="PRIVATE_ASSET_PATH",
                    recording_errors=["PRIVATE_ASSET_PATH"])
    evidence["actions"][0]["error"] = "PRIVATE_ASSET_PATH"
    public = research._public(result)
    assert "PRIVATE_ASSET_PATH" not in json.dumps(public)
    assert public["evidence"]["actions"][0]["error"]


def test_selecting_invalid_repair_keeps_current_and_finishes_budget(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    original = agent.ask
    def ask(**kwargs):
        answer = original(**kwargs)
        if kwargs["role"] == "repair":
            return {"toolbox.py": "class Toolbox(:\n", "skill.md": "invalid"}
        if kwargs["role"] == "repair_select":
            return {"selected": "candidate_00", "reason": "Invalid choice"}
        return answer
    agent.ask = ask
    adapted = research.adapt(runner, agent, run, SMALL)
    assert all(t["selected"] == "current" for value in adapted["tasks"].values() for t in value["trials"])
    assert len(list((run / "adaptation").glob("*/trial_*/selection_validation.json"))) == 6


def test_mixed_validity_repairs_keep_ids_and_share_accepted_baseline(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    original = agent.ask
    def ask(**kwargs):
        answer = original(**kwargs)
        if kwargs["role"] == "repair":
            payload = agent.calls[-1][1]
            assert payload["toolbox"]["toolbox.py"] == STAGE1_CODE
            if payload["hypothesis"] == "Grip rim":
                return {"toolbox.py": "class Toolbox(:\n", "skill.md": "invalid"}
        return answer
    agent.ask = ask
    research.adapt(runner, agent, run, SMALL)
    for role, payload, _ in agent.calls:
        if role == "repair_select":
            assert payload["available_choices"] == ["current", "candidate_01"]
            assert payload["hypotheses"]["candidate_01"] == "Slow the lift"
            assert set(payload["screen"]) == {"current", "candidate_01"}
