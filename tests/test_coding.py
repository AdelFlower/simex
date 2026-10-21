"""Interactive coding contracts with fake models and rollouts only."""
import copy
import json
import threading
from pathlib import Path

import pytest

from simex import research
from simex.agent import Agent, AgentBudgetExhausted, AgentResponseError, AgentTransportError
from simex.coding import CodingSession, read_history, session_history
from simex.research_context import MAX_PROMPT_CHARS, ResearchContext
from simex.runner import Runner
from test_research import FakeAgent, FakeRunner, INITIAL_CODE, SMALL, harness, task


def edit(code=INITIAL_CODE + "REPAIR = 1\n", notes="Capability usage"):
    return {"action": "edit", "files": {"toolbox.py": code, "skill.md": notes},
            "hypothesis": "Exercise the revised manipulation helper"}


def decide(payload, accept=True):
    return {"action": "decide", "draft_id": payload["working_draft"],
            "test_id": payload["last_feedback"]["test_id"], "accept": accept,
            "reason": "The observed execution supports this decision.",
            "accept_unresolved_history": True,
            "history_reason": "Historical failures remain uncertain; retain their original evidence."}


FINISH = {"action": "finish", "reason": "No further justified changes in this cycle."}


def test_read_coverage_tracks_delivered_bytes_duplicates_and_empty_pages():
    def read(start, end):
        return {"response": {"action": "read"}, "feedback": {"status": "read",
            "reads": [{"id": "text_source", "offset": start, "end": end}]}}
    history = [read(0, 4000), read(4000, 7500), read(0, 5000), read(7500, 7500),
               {"response_error": {"reason": "interrupted"}, "feedback": {"status": "invalid_response"}}]
    result = read_history(history)
    assert result["consecutive_reads"] == result["read_operations"] == 4
    assert result["recent"] == [{"id": "text_source", "ranges": [[0, 7500]], "requests": 4,
                                "repeated_reads": 1, "empty_pages": 1, "omitted_ranges": 0}]
    # A malformed operation cannot crash context construction or erase coverage.
    assert read_history(history + [{"response": []}]) == result
    edited = read_history(history + [{"response": {"action": "edit"}}])
    assert edited["consecutive_reads"] == 0 and edited["recent"] == result["recent"]


def make_adapter(monkeypatch, tmp_path):
    monkeypatch.setattr("simex.agent.shutil.which", lambda command: "/usr/bin/" + command)
    monkeypatch.setattr("simex.agent.Path.home", lambda: tmp_path)
    return Agent(tmp_path / "run", {"reasoning_effort": "high"})


class CodingAgent(FakeAgent):
    def __init__(self, actions):
        super().__init__()
        self.actions = list(actions)
        self.config = {"max_agent_calls": 2000}
        self.coding_inputs = []
        self.requests = []

    def ask(self, role, prompt, images=None):
        if role != "discover_code":
            assert role not in ("discover_revision", "discover_select")
            return super().ask(role, prompt, images)
        if self.count >= self.config["max_agent_calls"]:
            raise AgentBudgetExhausted("Configured model-call budget exhausted")
        self.count += 1
        payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
        self.coding_inputs.append(payload)
        self.requests.append({"role": role, "prompt": prompt, "images": images})
        assert len(prompt) <= MAX_PROMPT_CHARS - 1000
        assert len(images or []) <= 4
        assert "SHOULD_NEVER_REACH_AGENT" not in prompt
        assert self.actions, "Unexpected extra coding model call"
        answer = self.actions.pop(0)
        if isinstance(answer, AgentTransportError):
            self.count -= 1
        if isinstance(answer, BaseException):
            raise answer
        return answer(payload) if callable(answer) else copy.deepcopy(answer)


def build_session(tmp_path, actions, *, max_turns=40, max_tests=10, code=INITIAL_CODE,
                  skills="Capability usage", task_count=1):
    root = tmp_path / "run"
    initial = root / "initial"
    initial.mkdir(parents=True, exist_ok=True)
    (initial / "toolbox.py").write_text(code)
    (initial / "skill.md").write_text(skills)
    runner, agent = FakeRunner(), CodingAgent(actions)
    tasks = [task("far", index) for index in range(task_count)]
    baseline = [research._run(runner, "mujoco", value, initial, root / "before" / str(i))
                for i, value in enumerate(tasks)]
    directory = root / "discovery/member_00/cycle_00/coding"
    cfg = {"coding_max_turns": max_turns, "coding_max_tests": max_tests}
    args = (runner, agent, directory, cfg,
            {"current": str(initial), "current_results": baseline, "bank": []},
            {"weakness": "Unverified grasp"}, tasks, task_count, [], 0)
    return CodingSession(*args), args


def test_invalid_draft_is_fixed_using_feedback_before_testing(tmp_path):
    def correct(payload):
        error = payload["last_feedback"]["validation"]
        assert error["status"] == "invalid_candidate"
        assert error["file"] == "toolbox.py" and error["line"] == 1
        assert "return 1" in error["excerpt"]
        assert payload["remaining_tests"] == 10
        assert payload["toolbox"]["toolbox.py"].startswith("return 1")
        return edit()
    session, _ = build_session(tmp_path, [edit("return 1\n" + INITIAL_CODE), correct,
                                         {"action": "test"}, decide, FINISH])
    state = session.run()
    assert state["turns_used"] == 5 and len(state["tests"]) == 1
    assert state["status"] == "complete" and state["end_reason"] == "model_finished"
    assert "REPAIR" in Path(state["current"], "toolbox.py").read_text()
    assert len(state["bank"]) == 1
    assert state["draft_id"] == state["current_id"]
    assert len(session.runner.calls) == 2
    stats = research._json(session.directory / "summary.json")
    assert stats["validation_failures"] == 1 and stats["accepted"] == 1


def test_rejected_draft_retained_reset_and_new_baseline(tmp_path):
    def next_edit(payload):
        assert payload["last_feedback"]["accepted"] is False
        assert payload["accepted_draft"] == "initial"
        assert "REPAIR = 1" in payload["toolbox"]["toolbox.py"]
        assert payload["coding_history"]["evidence_id"].startswith("text_")
        return {"action": "edit", "edits": [{"file": "toolbox.py", "old": "REPAIR = 1", "new": "REPAIR = 2"}],
                "hypothesis": "Use rejection feedback"}
    def from_accepted(payload):
        assert payload["working_draft"] == payload["accepted_draft"]
        assert "REPAIR = 2" in payload["toolbox"]["toolbox.py"]
        return edit(INITIAL_CODE + "REPAIR = 3\n")
    session, _ = build_session(tmp_path, [
        edit(), {"action": "test"}, lambda p: decide(p, False), next_edit,
        {"action": "test"}, decide, {"action": "reset"}, from_accepted,
        {"action": "test"}, decide, FINISH])
    state = session.run()
    assert len(state["accepted"]) == 2
    assert state["tests"][2]["baseline_id"] == state["tests"][1]["draft_id"]
    assert len(session.runner.calls) == 4
    assert all(call["generated"] for call in session.runner.calls)


def test_repeated_test_and_decision_reuse_without_extra_physics(tmp_path):
    def repeat_decision(payload):
        decision = dict(payload["last_feedback"])
        return {"action": "decide", "draft_id": payload["working_draft"],
                "test_id": decision["test_id"], "accept": False, "reason": "Cannot revise saved decision"}
    session, _ = build_session(tmp_path, [
        edit(), {"action": "test"}, {"action": "test"}, decide, repeat_decision,
        {"action": "test"}, FINISH])
    state = session.run()
    assert len(state["tests"]) == len(state["accepted"]) == 1
    assert state["tests"][0]["decision"]["accepted"] is True
    assert len(session.runner.calls) == 2


def test_edits_after_test_cannot_be_accepted_without_another_test(tmp_path):
    def stale_decision(payload):
        return {"action": "decide", "draft_id": payload["working_draft"],
                "test_id": "test_000", "accept": True, "reason": "Try the stale test"}
    session, _ = build_session(tmp_path, [
        edit(), {"action": "test"}, edit(INITIAL_CODE + "REPAIR = 2\n"),
        stale_decision, FINISH])
    state = session.run()
    assert state["current_id"] == "initial" and not state["accepted"]
    history = session_history(session.directory)
    assert history[3]["feedback"]["status"] == "operation_error"


def test_latest_candidate_evidence_survives_reads_and_subsequent_edits(tmp_path):
    def check(payload):
        latest = payload["latest_test"]
        assert latest["test_id"] == "test_000"
        assert latest["draft_id"] == "draft_000"
        assert latest["results"][0]["success"] is True
        assert latest["baseline_results"][0]["success"] is False
        assert payload["results"][0]["success"] is False
        return latest

    def read(payload):
        check(payload)
        assert "results" not in payload["last_feedback"]  # no duplicate payload
        return {"action": "read", "evidence_requests": [
            {"id": payload["coding_history"]["evidence_id"], "offset": 0, "limit": 100}]}

    def change(payload):
        check(payload)
        assert payload["last_feedback"]["status"] == "read"
        return edit(INITIAL_CODE + "REPAIR = 2\n")

    def finish(payload):
        assert check(payload)["draft_id"] != payload["working_draft"]
        return FINISH

    session, _ = build_session(tmp_path, [edit(), {"action": "test"}, read, change, finish])
    state = session.run()
    assert state["current_id"] == "initial"
    assert len(session.runner.calls) == 2  # reading and editing do not rerun trials


def test_invalid_operations_and_json_spend_turns_not_tests(tmp_path):
    session, _ = build_session(tmp_path, [
        AgentResponseError("Invalid JSON", "{broken"), {"action": "shell", "cmd": "echo no"},
        {"action": "edit", "files": {"other.py": "bad"}, "hypothesis": "bad path"},
        edit("return 1\n" + INITIAL_CODE), {"action": "test"}, FINISH])
    state = session.run()
    assert state["turns_used"] == 6 and not state["tests"]
    assert session_history(session.directory)[0]["feedback"]["output"] == "{broken"
    assert len(session.runner.calls) == 1


def test_failed_replacement_preserves_draft_and_is_reported(tmp_path):
    session, _ = build_session(tmp_path, [
        {"action": "edit", "edits": [{"file": "toolbox.py", "old": "NOT PRESENT", "new": "pass"}],
         "hypothesis": "Invalid replacement"}, FINISH])
    state = session.run()
    assert len(state["drafts"]) == 1 and state["draft_id"] == "initial"
    assert "match exactly once" in session_history(session.directory)[0]["feedback"]["reason"]


@pytest.mark.parametrize("max_turns", [1, 40])
def test_interaction_limit_retains_only_accepted_version(tmp_path, max_turns):
    session, args = build_session(tmp_path, [edit()] + [{"action": "reset"}] * (max_turns - 1),
                                  max_turns=max_turns)
    state = session.run()
    assert state["turns_used"] == session.agent.count == max_turns
    assert state["end_reason"] == "interaction_limit" and state["current_id"] == "initial"
    assert CodingSession(*args).run()["turns_used"] == max_turns
    assert session.agent.count == max_turns


def test_tenth_batch_limit_and_last_interaction_guard(tmp_path):
    actions = []
    for index in range(10):
        actions += [edit(INITIAL_CODE + f"REPAIR = {index}\n"), {"action": "test"}, decide]
    actions += [edit(INITIAL_CODE + "REPAIR = 11\n"), {"action": "test"}, FINISH]
    session, _ = build_session(tmp_path, actions)
    state = session.run()
    assert len(state["tests"]) == 10 and len(session.runner.calls) == 11
    assert "No test batches remain" in session_history(session.directory)[-2]["feedback"]["reason"]
    last, _ = build_session(tmp_path / "last", [edit(), {"action": "test"}], max_turns=2)
    assert not last.run()["tests"]
    assert "leave an interaction" in session_history(last.directory)[-1]["feedback"]["reason"]


def test_network_failures_are_exempt_and_retry_five_times(tmp_path):
    session, _ = build_session(tmp_path, [AgentTransportError("connection reset")] * 5 + [FINISH])
    state = session.run()
    assert state["turns_used"] == session.agent.count == 1
    record = research._json(session.directory / "turns/000.json")
    assert len(record["transport_errors"]) == 5
    assert len({req["prompt"] for req in session.agent.requests}) == 1


def test_network_failure_limit_can_resume_without_spending_coding_turn(tmp_path):
    session, args = build_session(tmp_path, [AgentTransportError("connection reset")] * 6 + [FINISH])
    with pytest.raises(AgentTransportError):
        session.run()
    assert research._json(session.directory / "summary.json")["end_reason"] == "network_failure"
    assert session.agent.count == 0
    state = CodingSession(*args).run()
    assert state["turns_used"] == session.agent.count == 1


def test_global_budget_stops_then_resumes_same_unrequested_turn(tmp_path):
    session, args = build_session(tmp_path, [FINISH])
    session.agent.config["max_agent_calls"] = 0
    with pytest.raises(AgentBudgetExhausted):
        session.run()
    assert research._json(session.directory / "summary.json")["end_reason"] == "model_budget_exhausted"
    assert research._json(session.directory / "turns/000.json")["charged"] is False
    session.agent.config["max_agent_calls"] = 1
    assert CodingSession(*args).run()["turns_used"] == 1


@pytest.mark.parametrize("checkpoint", ["response", "draft", "test", "commit"])
def test_resume_at_operation_boundaries_does_not_repeat_model_or_physics(tmp_path, monkeypatch, checkpoint):
    session, args = build_session(tmp_path, [edit(), {"action": "test"}, decide, FINISH])
    saved = research._save
    interrupted = False
    def save(path, value):
        nonlocal interrupted
        saved(path, value)
        matches = (
            checkpoint == "response" and path.name == "000.json" and "response" in value and "after" not in value
            or checkpoint == "draft" and path.name == "validation.json"
            or checkpoint == "test" and path.name == "test.json" and value["status"] == "complete"
            or checkpoint == "commit" and path.name == "002.json" and "after" in value
        )
        if matches and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
    monkeypatch.setattr(research, "_save", save)
    with pytest.raises(KeyboardInterrupt):
        session.run()
    assert interrupted
    if checkpoint == "commit":
        assert research._json(session.directory / "summary.json")["accepted"] == 1
    state = CodingSession(*args).run()
    assert len(session.runner.calls) == 2
    assert session.agent.count == state["turns_used"] == 4
    assert len(state["accepted"]) == len(state["tests"]) == 1


def test_partial_test_resumes_cached_trials_and_one_batch(tmp_path):
    session, args = build_session(tmp_path, [edit(), {"action": "test"}, decide, FINISH], task_count=2)
    rollout = session.runner.rollout
    stopped = False
    def partial(**kwargs):
        nonlocal stopped
        if "rollouts" in kwargs["out"].parts and kwargs["out"].name == "001" and not stopped:
            stopped = True
            raise KeyboardInterrupt()
        return rollout(**kwargs)
    session.runner.rollout = partial
    with pytest.raises(KeyboardInterrupt):
        session.run()
    interrupted = research._json(session.directory / "summary.json")
    assert interrupted["test_batches"] == 1 and interrupted["completed_test_batches"] == 0
    assert interrupted["coding_turns"] == 2
    state = CodingSession(*args).run()
    assert len(state["tests"]) == 1 and state["turns_used"] == 4
    assert len(session.runner.calls) == 4  # two baseline + two candidate rollouts


def test_incomplete_test_cannot_be_decided(tmp_path, monkeypatch):
    session, args = build_session(tmp_path, [edit(), {"action": "test"}, decide, FINISH], task_count=2)
    real_suite = research._suite
    monkeypatch.setattr(research, "_suite", lambda *a, **kw: real_suite(*a, **kw)[:1])
    with pytest.raises(research.ResearchError, match="Incomplete"):
        session.run()
    assert not session.state["accepted"]
    monkeypatch.setattr(research, "_suite", real_suite)
    assert len(CodingSession(*args).run()["accepted"]) == 1
    assert len(session.runner.calls) == 4


def test_large_toolbox_requires_edits_and_old_evidence_ids_survive(tmp_path):
    previous_id = []
    def read_source(payload):
        identifier = payload["toolbox"]["skill.md"]["evidence_id"]
        previous_id.append(identifier)
        assert payload["edit_mode"] == "edits_only"
        return {"action": "read", "evidence_requests": [{"id": identifier, "offset": 0, "limit": 100}]}
    def apply_edit(payload):
        assert payload["_coding_context"]["read_results"][0]["text"].startswith("Large notes")
        assert payload["read_history"]["recent"][0]["ranges"] == [[0, 100]]
        return {"action": "edit", "edits": [{"file": "skill.md", "old": "Large notes", "new": "Revised notes"}],
                "hypothesis": "Clarify the measured behavior"}
    session, _ = build_session(tmp_path, [read_source, apply_edit,
        lambda p: {"action": "read", "evidence_requests": [{"id": previous_id[0], "offset": 0, "limit": 50}]},
        FINISH], skills="Large notes\n" + "Existing capability documentation.\n" * 2100)
    state = session.run()
    context = ResearchContext.load(session.directory / "evidence.context")
    assert context.assets[previous_id[0]]["text"].startswith("Large notes")
    assert state["current_id"] == "initial"
    assert session.agent.coding_inputs[-1]["_coding_context"]["read_results"][0]["text"].startswith("Large notes")
    coverage = session.agent.coding_inputs[-1]["read_history"]
    assert coverage["consecutive_reads"] == 1
    assert coverage["recent"][0]["repeated_reads"] == 1


def test_medium_library_remains_visible_across_reads_but_requires_exact_edits(tmp_path):
    code = INITIAL_CODE + "THRESHOLD = 1\n#" + "p" * 40_000 + "\n"
    def read(payload):
        assert payload["toolbox"]["toolbox.py"] == code
        assert payload["edit_mode"] == "edits_only"
        return {"action": "read", "evidence_requests": [
            {"id": payload["accepted_toolbox"]["toolbox.py"]["evidence_id"], "limit": 100}]}
    def revise(payload):
        assert payload["toolbox"]["toolbox.py"] == code
        assert payload["toolbox"]["skill.md"] == "Capability usage"
        return {"action": "edit", "edits": [
            {"file": "toolbox.py", "old": "THRESHOLD = 1", "new": "THRESHOLD = 2"}],
            "hypothesis": "Use exact visible source after an evidence read"}
    session, _ = build_session(tmp_path, [read, revise, FINISH], code=code)
    state = session.run()
    assert "THRESHOLD = 2" in Path(state["drafts"][state["draft_id"]]["path"], "toolbox.py").read_text()
    assert len(session.runner.calls) == 1


def test_discover_uses_coding_loop_and_preserves_adapt_and_evaluate(harness):
    runner, _, run = harness
    agent = CodingAgent([edit(), {"action": "test"}, decide, FINISH] * 2)
    cfg = {**SMALL, "discovery_mode": "interactive", "coding_max_turns": 40, "coding_max_tests": 10,
           "revisions": 999}
    research.discover(runner, agent, run, cfg)
    assert agent.actions == []
    assert all(role not in ("discover_select", "discover_revision") for role, _, _ in agent.calls)
    cycle = run / "discovery/member_00/cycle_00"
    assert research._json(cycle / "summary.json")["coding"]["accepted"] == 1
    assert len(research._json(cycle / "research_history.json")["coding_history"]) == 4
    research.adapt(runner, agent, run, cfg)
    research.evaluate(runner, agent, run, cfg)
    counts = len(runner.calls), agent.count
    research.discover(runner, agent, run, cfg)
    assert (len(runner.calls), agent.count) == counts
    report = research.report(run)
    assert report["coding"]["coding_turns"] == 8
    assert report["coding"]["test_batches"] == report["coding"]["accepted"] == 2
    assert all(item["end_reason"] == "model_finished" for item in report["coding"]["sessions"])


def test_coding_mode_and_limits_are_frozen_and_old_manifest_remains_legacy(harness):
    _, _, run = harness
    research._initialize(run, SMALL)
    manifest = research._json(run / "manifest.json")
    manifest["config"].pop("discovery_mode")
    research._save(run / "manifest.json", manifest)
    assert research._initialize(run, SMALL).get("discovery_mode", "legacy") == "legacy"
    with pytest.raises(research.ResearchError, match="Discovery mode"):
        research._initialize(run, {**SMALL, "discovery_mode": "interactive"})
    new_run = run / "new"
    cfg = {**SMALL, "discovery_mode": "interactive", "coding_max_turns": 40, "coding_max_tests": 10}
    research._initialize(new_run, cfg)
    for key in ("coding_max_turns", "coding_max_tests"):
        with pytest.raises(research.ResearchError, match="Coding budgets"):
            research._initialize(new_run, {**cfg, key: 2})
    research._initialize(new_run, {**cfg, "revisions": 100})


def test_parallel_policy_budget_exhaustion_leaves_unattempted_generations_uncached(tmp_path):
    class BudgetWriter:
        def __init__(self):
            self.count = 0
            self.config = {"max_agent_calls": 1}
            self.lock = threading.Lock()

        def policy(self, *_):
            with self.lock:
                if self.count >= self.config["max_agent_calls"]:
                    raise AgentBudgetExhausted("Configured model-call budget exhausted")
                self.count += 1
            return "print('ready')\n"
    writer = BudgetWriter()
    runner = Runner(writer, tmp_path, {"source_policy_workers": 3})
    initial = Path(research.__file__).parent / "initial"
    tasks = [task("far", i) for i in range(6)]
    with pytest.raises(AgentBudgetExhausted):
        research._suite(runner, tasks, initial, tmp_path / "suite")
    caches = list((tmp_path / "suite").glob("*/policy_generation.json"))
    assert len(caches) == writer.count == 1
    assert research._json(caches[0])["status"] == "ready"
    writer.config["max_agent_calls"] = 6
    runner.rollout = lambda **kwargs: {"status": "failed", "success": False, "actual_started": True}
    assert len(research._suite(runner, tasks, initial, tmp_path / "suite")) == 6
    assert writer.count == 6
    assert all(research._json(p)["status"] == "ready" for p in
               (tmp_path / "suite").glob("*/policy_generation.json"))


def test_complete_test_is_recovered_when_budget_is_exhausted_before_decision(tmp_path):
    session, args = build_session(tmp_path, [edit(), {"action": "test"}, decide, FINISH])
    session.agent.config["max_agent_calls"] = 2
    with pytest.raises(AgentBudgetExhausted):
        session.run()
    assert len(session.runner.calls) == 2
    assert research._json(session.directory / "summary.json")["completed_test_batches"] == 1
    session.agent.config["max_agent_calls"] = 4
    state = CodingSession(*args).run()
    assert state["current_id"] != "initial"
    assert len(session.runner.calls) == 2


def test_budget_interrupting_test_resumes_without_new_batch_or_repeating_policies(tmp_path):
    session, args = build_session(tmp_path, [edit(), {"action": "test"}, decide, FINISH], task_count=2)
    rollout = session.runner.rollout
    stopped = False
    def exhaust(**kwargs):
        nonlocal stopped
        if "rollouts" in kwargs["out"].parts and kwargs["out"].name == "001" and not stopped:
            stopped = True
            raise AgentBudgetExhausted("Configured model-call budget exhausted")
        return rollout(**kwargs)
    session.runner.rollout = exhaust
    with pytest.raises(AgentBudgetExhausted):
        session.run()
    summary = research._json(session.directory / "summary.json")
    assert summary["test_batches"] == 1 and summary["end_reason"] == "model_budget_exhausted"
    state = CodingSession(*args).run()
    assert len(session.runner.calls) == 4
    assert len(state["tests"]) == 1 and session.agent.count == 4


def test_replay_regression_cannot_be_overridden_by_coding_model(tmp_path):
    session, _ = build_session(tmp_path, [edit(INITIAL_CODE + "CHANGED = 1\n"),
                                         {"action": "test"}, decide, FINISH],
                               code=INITIAL_CODE + "REPAIR = 1\n", task_count=2)
    session.focused = 1
    session.replay = [{"task_sha256": "known-success"}]
    state = session.run()
    decision = state["tests"][0]["decision"]
    assert decision["agent_accept"] is True and decision["accepted"] is False
    assert "cannot override" in decision["gate_reason"]
    assert state["current_id"] == "initial"


def test_missing_decision_reason_is_correctable_in_next_interaction(tmp_path):
    session, _ = build_session(tmp_path, [
        edit(), {"action": "test"},
        lambda p: {**decide(p), "reason": ""},
        lambda p: {"action": "decide", "draft_id": p["working_draft"], "test_id": "test_000",
                   "accept": True, "reason": "Observed a valid improvement"}, FINISH])
    state = session.run()
    assert len(state["tests"]) == 1 and len(state["accepted"]) == 1
    assert session_history(session.directory)[2]["feedback"]["status"] == "operation_error"


def test_large_draft_rejects_whole_file_reconstruction(tmp_path):
    session, _ = build_session(tmp_path, [edit(), FINISH], skills="Long documentation.\n" * 2500)
    state = session.run()
    assert len(state["drafts"]) == 1
    assert "exact block edits" in session_history(session.directory)[0]["feedback"]["reason"]


def _ids(value, kind=None):
    if isinstance(value, dict):
        if "evidence_id" in value and (kind is None or value.get("kind") == kind):
            yield value["evidence_id"]
        for child in value.values():
            yield from _ids(child, kind)
    elif isinstance(value, list):
        for child in value:
            yield from _ids(child, kind)


def test_image_request_attaches_actual_public_image_after_restart(tmp_path, monkeypatch):
    from PIL import Image
    session, args = build_session(tmp_path, [], max_turns=3)
    public = tmp_path / "public.png"
    Image.new("RGB", (10, 10), (2, 40, 80)).save(public)
    payload = {"weakness": "Inspect", "phase_images": [str(public)]}
    bundle = research._json(session.directory / "initial.json")
    bundle["payload"] = payload
    research._save(session.directory / "initial.json", bundle)
    agent = session.agent
    agent.actions = [
        lambda p: {"action": "read", "evidence_requests": [{"id": next(_ids(p, "image"))}]},
        FINISH]
    session = CodingSession(*args)
    save = research._save
    once = False
    def interrupt(path, value):
        nonlocal once
        save(path, value)
        if path.name == "000.json" and "after" in value and not once:
            once = True
            raise KeyboardInterrupt()
    monkeypatch.setattr(research, "_save", interrupt)
    with pytest.raises(KeyboardInterrupt):
        session.run()
    CodingSession(*args).run()
    attached = agent.requests[-1]["images"]
    assert len(attached) == 1 and Path(attached[0]).read_bytes() == public.read_bytes()


def test_cross_cycle_context_exposes_full_operation_history(harness):
    runner, _, run = harness
    calls = []
    def start(payload):
        calls.append(payload)
        return edit(INITIAL_CODE + f"REPAIR = {len(calls)}\n")
    agent = CodingAgent([start, {"action": "test"}, decide, FINISH] * 2)
    cfg = {**SMALL, "population": 1, "cycles": 2, "discovery_mode": "interactive",
           "coding_max_turns": 40, "coding_max_tests": 10}
    research.discover(runner, agent, run, cfg)
    previous = calls[1]["recent_research_history"][0]["coding_history"]
    assert previous["count"] == 4
    context = ResearchContext.load(run / "discovery/member_00/cycle_01/coding/evidence.context")
    records = json.loads(context.assets[previous["evidence_id"]]["text"])
    assert [item["response"]["action"] for item in records] == ["edit", "test", "decide", "finish"]


def test_coding_retains_frozen_peer_ideas_but_not_peer_implementation(harness):
    runner, _, run = harness
    actions = []
    for member in range(2):
        actions += [edit(INITIAL_CODE + f"REPAIR = 'ONLY_MEMBER_{member}_IMPLEMENTATION'\n"),
                    {"action": "test"}, decide, FINISH]
    agent = CodingAgent(actions + [FINISH, FINISH])
    cfg = {**SMALL, "cycles": 2, "discovery_mode": "interactive",
           "coding_max_turns": 40, "coding_max_tests": 10}
    research.discover(runner, agent, run, cfg)
    for member, payload in enumerate(agent.coding_inputs[-2:]):
        peers = payload["peer_experiment_summaries"]
        frozen = research._json(run / f"discovery/member_{member:02d}/cycle_01/peer_summaries.json")
        assert peers == frozen
        assert peers[0]["member"] == 1 - member
        assert [s["cycle"] for s in peers[0]["summaries"]] == [0]
        assert peers[0]["summaries"][0]["accepted_hypotheses"]
        assert f"ONLY_MEMBER_{1-member}_IMPLEMENTATION" not in json.dumps(payload)
        assert f"ONLY_MEMBER_{member}_IMPLEMENTATION" in payload["toolbox"]["toolbox.py"]
    assert all(p["peer_experiment_summaries"] == [] for p in agent.coding_inputs[:-2])


def test_full_coding_history_stays_bounded_and_readable_with_large_evidence(tmp_path):
    from PIL import Image
    from simex.coding import INSTRUCTIONS
    images = []
    for i in range(12):
        image = tmp_path / f"frame_{i:02d}.png"
        Image.new("RGB", (8, 8), (i, 10, 20)).save(image)
        images.append(str(image))
    result = {"status": "failed", "success": False, "steps": 100,
              "execution": {"actual_started": True, "script": "print('diagnostic')\n" * 2000},
              "evidence": {"keyframes": images,
                           "robot_trace": [{"sim_time": i, "ee_position": [0.1, 0.2, 0.3]} for i in range(1000)],
                           "stdout": "Observed execution diagnostics\n" * 2000}}
    history = [{"turn": i, "response": {"action": "test" if i % 4 == 0 else "read"},
                "feedback": {"status": "complete", "results": [copy.deepcopy(result) for _ in range(6)]}}
               for i in range(40)]
    payload = {"toolbox": {"toolbox.py": INITIAL_CODE, "skill.md": "Large notes\n" * 5000},
               "coding_history": history, "last_feedback": history[-1]["feedback"],
               "remaining_interactions": 1, "remaining_tests": 0}
    context = ResearchContext(payload, images, instruction=INSTRUCTIONS)
    context.persist(tmp_path / "coding.context")
    prompt, attached = context.coding_prompt(INSTRUCTIONS)
    assert len(prompt) < 63_000 and len(attached) == 4
    ref = context.summary["coding_history"]
    assert ref["count"] == 40
    records = json.loads(context.assets[ref["evidence_id"]]["text"])
    assert len(records) == 40
    first = records[0]["feedback"]["results"][0]
    assert first["status"] == "failed" and first["success"] is False
    # All detailed traces and source pages remain registered, including the first operation.
    assert any("Observed execution diagnostics" in asset.get("text", "") for asset in context.assets.values())
    assert any('"sim_time":999' in asset.get("text", "") for asset in context.assets.values())
    replies, _ = context.read([{"id": ref["evidence_id"], "offset": 0, "limit": 16000}])
    assert replies[0]["text"] and len(json.dumps(replies)) < 18000


def test_adapter_response_survives_crash_before_coding_response_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace
    session, args = build_session(tmp_path, [])
    actual = make_adapter(monkeypatch, tmp_path)
    session.agent = actual
    args = (args[0], actual, *args[2:])
    calls = []
    def execute(command, **kwargs):
        calls.append(kwargs["input"])
        work = Path(command[command.index("--bind") + 1])
        (work / "response.txt").write_text(json.dumps(FINISH))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr("simex.agent.subprocess.run", execute)
    ask = actual.ask
    def crash_after_response(**kwargs):
        ask(**kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(actual, "ask", crash_after_response)
    with pytest.raises(KeyboardInterrupt):
        session.run()
    monkeypatch.setattr(actual, "ask", ask)
    state = CodingSession(*args).run()
    assert state["status"] == "complete" and state["turns_used"] == 1
    assert actual.count == len(calls) == 1


def test_unanswered_model_call_stays_charged_on_resume(tmp_path):
    session, args = build_session(tmp_path, [KeyboardInterrupt(), FINISH], max_turns=2)
    with pytest.raises(KeyboardInterrupt):
        session.run()
    state = CodingSession(*args).run()
    assert state["turns_used"] == session.agent.count == 2
    assert session_history(session.directory)[0]["feedback"]["status"] == "invalid_response"


def test_forbidden_source_diagnostic_identifies_actual_line(tmp_path):
    session, _ = build_session(tmp_path, [edit("import os\n" + INITIAL_CODE), FINISH])
    session.run()
    validation = session_history(session.directory)[0]["feedback"]["validation"]
    assert validation["line"] == 1 and "1: import os" in validation["excerpt"]


def test_checkpoint_references_results_without_duplicating_raw_rollouts(tmp_path):
    session, args = build_session(tmp_path, [edit(), {"action": "test"}, decide, FINISH])
    session.run()
    checkpoint = research._json(session.directory / "state.json")
    assert checkpoint["results_stored_separately"] is True
    assert "current_results" not in checkpoint
    assert all("results" not in item and "baseline_results" not in item for item in checkpoint["tests"])
    restored = CodingSession(*args).run()
    assert restored["current_results"][0]["success"] is True


def test_refunded_adapter_attempt_is_not_charged_by_interrupted_coding_request(tmp_path, monkeypatch):
    from types import SimpleNamespace
    session, args = build_session(tmp_path, [])
    actual = make_adapter(monkeypatch, tmp_path)
    session.agent = actual
    args = (args[0], actual, *args[2:])
    attempts = []
    def execute(command, **kwargs):
        attempts.append(kwargs["input"])
        if len(attempts) == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="connection reset")
        work = Path(command[command.index("--bind") + 1])
        (work / "response.txt").write_text(json.dumps(FINISH))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr("simex.agent.subprocess.run", execute)
    ask = actual.ask
    def interrupt_refund(**kwargs):
        try:
            return ask(**kwargs)
        except AgentTransportError:
            raise KeyboardInterrupt()
    monkeypatch.setattr(actual, "ask", interrupt_refund)
    with pytest.raises(KeyboardInterrupt):
        session.run()
    assert research._json(session.directory / "summary.json")["coding_turns"] == 0
    monkeypatch.setattr(actual, "ask", ask)
    state = CodingSession(*args).run()
    assert state["turns_used"] == actual.count == 1
    assert actual.attempt_count == 2


def test_recovery_does_not_execute_response_from_prohibited_adapter_tool(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from simex.agent import InvalidPolicy
    session, args = build_session(tmp_path, [], max_turns=2)
    actual = make_adapter(monkeypatch, tmp_path)
    session.agent = actual
    args = (args[0], actual, *args[2:])
    calls = []
    def execute(command, **kwargs):
        calls.append(kwargs["input"])
        work = Path(command[command.index("--bind") + 1])
        (work / "response.txt").write_text(json.dumps(edit() if len(calls) == 1 else FINISH))
        events = '{"item":{"type":"command_execution"}}\n' if len(calls) == 1 else ""
        return SimpleNamespace(returncode=0, stdout=events, stderr="")
    monkeypatch.setattr("simex.agent.subprocess.run", execute)
    with pytest.raises(InvalidPolicy):
        session.run()
    state = CodingSession(*args).run()
    assert state["turns_used"] == 2 and len(state["drafts"]) == 1


def test_partial_draft_write_is_completed_from_saved_response(tmp_path, monkeypatch):
    from simex import coding
    session, args = build_session(tmp_path, [edit(), {"action": "test"}, decide, FINISH])
    write = coding._immutable_text
    once = False
    def interrupted(path, text):
        nonlocal once
        write(path, text)
        if not once:
            once = True
            raise KeyboardInterrupt()
    monkeypatch.setattr(coding, "_immutable_text", interrupted)
    with pytest.raises(KeyboardInterrupt):
        session.run()
    state = CodingSession(*args).run()
    assert state["turns_used"] == session.agent.count == 4
    assert len(state["drafts"]) == 2 and len(state["accepted"]) == 1


def test_started_experiment_with_infrastructure_failure_is_never_repeated(tmp_path):
    session, args = build_session(tmp_path, [edit(), {"action": "test"}])
    rollout = session.runner.rollout
    def infrastructure(**kwargs):
        result = rollout(**kwargs)
        result.update(status="interrupted", success=False, actual_started=True, failure_kind="infrastructure")
        return result
    session.runner.rollout = infrastructure
    with pytest.raises(research.ResearchError, match="Infrastructure failure"):
        session.run()
    for _ in range(2):
        with pytest.raises(research.ResearchError, match="Infrastructure interrupted"):
            CodingSession(*args).run()
    assert len(session.runner.calls) == 2
    assert session.agent.count == 2
