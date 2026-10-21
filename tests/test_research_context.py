"""Research input bounds, lossless reads, isolation and interruption recovery."""
import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from simex import research
from simex.agent import AgentError, AgentTransportError
from simex.research_context import (ContextError, MAX_IMAGES, MAX_INTERACTIONS, MAX_PROMPT_CHARS,
    MAX_READ_ROUNDS, READ_CHARS, SUMMARY_CHARS, ResearchContext, apply_source_edits, dumps,
    prepare_edit_context)


def unpack(prompt):
    return json.loads(prompt.split("INPUT JSON:\n", 1)[1])


def large_evidence(tmp_path, count=12):
    images, rows = [], []
    for index in range(count):
        image = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (8, 8), (index, 20, 30)).save(image)
        images.append(str(image))
        actions = [{"method": "move_ee", "kwargs": {"position": [.4, .2, .1]},
                    "start": {"ee_position": [.4, .2, .1], "sim_time": i},
                    "end": {"ee_position": [.4, .2, .1], "sim_time": i+1}}
                   for i in range(80)]
        rows.append({"status": "failed", "success": False, "steps": 500,
            "evidence": {"stdout": f"trial {index}\n" + "MEASURE pose [0.4, 0.2, 0.1]\n"*1_000 + "STOP tracking error",
                "actions": actions, "action_summary": {"arm_commands": 80, "arm_motion_observed": True},
                "phase_images": [str(image)]},
            "execution": {"actual_started": True, "script": f"# trial {index}\n" + "print('measure')\n"*1_000}})
    toolbox = {"toolbox.py": "class Toolbox:\n" + "    # primitive\n"*2_000, "skill.md": "public API"}
    return {"toolbox": toolbox, "probe_toolbox": copy.deepcopy(toolbox), "results": rows}, images


def test_large_packet_keeps_every_outcome_and_bounds_every_read_round(tmp_path):
    payload, images = large_evidence(tmp_path)
    original = copy.deepcopy(payload)
    ctx = ResearchContext(payload, images)
    manifest = ctx.persist(tmp_path / "context")
    assert payload == original
    assert len(dumps(payload)) > 700_000
    assert manifest["summary_chars"] <= SUMMARY_CHARS
    assert len(ctx.summary["results"]) == 12
    assert all(r["success"] is False and r["evidence"]["action_summary"]["arm_commands"] == 80
               for r in ctx.summary["results"])
    prompt, attached = ctx.prompt("Diagnose failure.")
    assert len(prompt) < MAX_PROMPT_CHARS
    assert len(attached) == MAX_IMAGES
    assert len(ctx.image_ids) == 12  # all are accessible, not silently discarded
    assert "same_as" in ctx.summary["probe_toolbox"]["toolbox.py"]
    identifier = ctx.summary["results"][0]["execution"]["script"]["evidence_id"]
    for turn in range(1, MAX_READ_ROUNDS + 1):
        replies, _ = ctx.read([{"id": identifier, "limit": READ_CHARS}]*4)
        assert len(dumps(replies)) <= READ_CHARS
        prompt, _ = ctx.prompt("Diagnose failure.", round_index=turn, notes="x"*2_000,
                               replies=replies, reads_so_far=[{"id": identifier}]*16)
        assert len(prompt) < MAX_PROMPT_CHARS
    last = ctx.image_ids[-1]
    replies, requested = ctx.read([{"id": last}])
    prompt, attached = ctx.prompt("Inspect last trial.", replies=replies, image_ids=requested)
    assert len(attached) == 1 and Path(attached[0]).read_bytes() == Path(images[-1]).read_bytes()
    assert unpack(prompt)["_research_context"]["attached_images"][0]["id"] == last


def test_paged_text_is_exact_including_escaped_logs_and_unicode(tmp_path):
    original = ('"quoted"\\\n\t中文\n' * 3_000) + "UNIQUE_FAILURE_DETAIL"
    ctx = ResearchContext({"script": original})
    ctx.persist(tmp_path)
    identifier = ctx.summary["script"]["evidence_id"]
    text, offset = "", 0
    while offset < len(original):
        replies, _ = ctx.read([{"id": identifier, "offset": offset, "limit": READ_CHARS}])
        page = replies[0]
        assert page["end"] > offset and len(dumps(replies)) <= READ_CHARS
        offset = page["end"]
        text += page["text"]
    assert text == original
    replies, _ = ctx.read([{"id": identifier, "find": "UNIQUE_FAILURE_DETAIL"}])
    assert replies[0]["text"] == "UNIQUE_FAILURE_DETAIL"
    assert replies[0]["complete"] is False  # complete means the WHOLE document


@pytest.mark.parametrize("source_chars", [None, 63_000])
def test_whole_current_library_survives_every_turn_with_rejection_history(tmp_path, source_chars):
    payload, images = large_evidence(tmp_path)
    if source_chars is not None:
        payload["toolbox"]["toolbox.py"] = "#" * source_chars
    payload["prior_attempts"] = [{"revision": 0, "accepted": False, "reason": "tracking failed",
        "candidate_toolbox": payload["toolbox"], "baseline_results": payload["results"][:4],
        "results": payload["results"][:4]}]
    payload["recent_research_history"] = [{"cycle": i, "attempts": [{"revision": j,
        "accepted": False, "reason": "earlier failure", "results": payload["results"][:4]}
        for j in range(2)]} for i in range(2)]
    instruction = "Preserve the complete current implementation. " * 80
    ctx = ResearchContext(payload, images, pinned_keys=("toolbox",), instruction=instruction)
    ctx.persist(tmp_path / "pinned")
    prompt, _ = ctx.prompt(instruction)
    assert unpack(prompt)["toolbox"] == payload["toolbox"]
    assert len(unpack(prompt)["results"]) == 12
    assert ctx.summary["prior_attempts"][0]["accepted"] is False
    identifier = ctx.summary["results"][0]["execution"]["script"]["evidence_id"]
    budget = min(READ_CHARS, MAX_PROMPT_CHARS-len(prompt)-7_000)
    assert budget > 1_000
    replies, _ = ctx.read([{"id": identifier, "limit": 16_000}], budget=budget)
    assert len(dumps(replies)) <= budget
    for turn in range(1, MAX_READ_ROUNDS+1):
        followup, _ = ctx.prompt(instruction, round_index=turn, notes="notes"*400,
            replies=replies, reads_so_far=[{"id": identifier, "offset": 0, "end": 100}]*16)
        assert unpack(followup)["toolbox"] == payload["toolbox"]
        assert len(followup) < MAX_PROMPT_CHARS


@pytest.mark.parametrize("role", ["align", "repair_hypotheses"])
@pytest.mark.parametrize("source_chars", [1_000, 40_000, 70_000])
def test_readonly_analysis_keeps_bounded_code_without_edit_protocol(tmp_path, role, source_chars):
    payload, images = large_evidence(tmp_path)
    payload["toolbox"]["toolbox.py"] = "#" * source_chars
    instruction = "Inspect the current implementation and public observations."
    prepared, pinned, edits = prepare_edit_context(role, instruction, payload)
    assert prepared == instruction and edits is False
    context = ResearchContext(payload, images, pinned_keys=pinned, instruction=prepared)
    context.persist(tmp_path / "analysis")
    for round_index in (0, 1, MAX_READ_ROUNDS):
        prompt, attachments = context.prompt(prepared, round_index=round_index, notes="Findings" * 250)
        packet = unpack(prompt)
        assert len(prompt) < MAX_PROMPT_CHARS and len(attachments) <= MAX_IMAGES
        assert "Output override: use exact block edits" not in prompt
        if source_chars < 64_000:
            assert packet["toolbox"] == payload["toolbox"]
        else:
            assert read_all_text(context, packet["toolbox"]["toolbox.py"]["evidence_id"]) == payload["toolbox"]["toolbox.py"]


@pytest.mark.parametrize("role,answer", [
    ("align", {"aligned": False, "source_config": {}, "reason": "Different retention."}),
    ("repair_hypotheses", {"hypotheses": ["Contact geometry", "Closure control"]}),
])
def test_readonly_ask_preserves_code_across_evidence_reads_and_returns_decision(tmp_path, role, answer):
    payload, _ = large_evidence(tmp_path)
    payload["toolbox"]["toolbox.py"] += "#" * 10_000

    class Reader:
        count = 0

        def ask(self, prompt, images, **kwargs):
            self.count += 1
            packet = unpack(prompt)
            assert packet["toolbox"] == payload["toolbox"]
            assert "Output override: use exact block edits" not in prompt
            if self.count == 1:
                return {"evidence_requests": [{"id": packet["results"][0]["evidence"]["stdout"]["evidence_id"],
                                               "find": "STOP tracking error"}],
                        "notes": "Keep helper semantics visible while inspecting execution."}
            assert "STOP tracking error" in packet["_research_context"]["read_results"][0]["text"]
            return answer

    agent = Reader()
    assert research._ask(agent, tmp_path / "decision.json", role, "Return the requested decision.", payload) == answer
    assert agent.count == 2


@pytest.mark.parametrize("read_request", [
    {"id": "/etc/passwd"}, {"id": "../private/result.json"},
    {"id": "ID", "offset": -1}, {"id": "ID", "offset": True},
    {"id": "ID", "limit": 100_000}, {"id": "ID", "path": "/etc/passwd"},
    {"id": "ID", "find": ""}, {"id": "ID", "find": 10},
])
def test_broker_does_not_authorize_arbitrary_files_or_unbounded_reads(read_request):
    ctx = ResearchContext({"script": "bounded code\n"*1_000})
    identifier = ctx.summary["script"]["evidence_id"]
    read_request = {k: identifier if v == "ID" else v for k, v in read_request.items()}
    with pytest.raises(ContextError):
        ctx.read([read_request])


def test_private_fields_never_enter_public_packet_or_assets(tmp_path):
    evidence = {"status": "failed", "success": False, "private_metrics": "SECRET_TARGET",
                "public_evidence": {"stdout": "observed\n"*2_000,
                                    "private_metrics": "SECRET_TARGET"}}
    ctx = ResearchContext({"target": research._public(evidence)})
    ctx.persist(tmp_path)
    assert "SECRET_TARGET" not in dumps(ctx.summary)
    assert all("SECRET_TARGET" not in asset.get("text", "") for asset in ctx.assets.values())
    with pytest.raises(ContextError):
        ctx.read([{"id": str(tmp_path.parent / "private/state.json")}])


def public_rollout_with_raw_evidence(tmp_path):
    public = tmp_path / "trial" / "public"
    public.mkdir(parents=True)
    states = [{"sim_time": i, "ee_position": [.4, .2, .3], "gripper_width": .08,
               "joints": [0.] * 7, "joint_velocities": [.812345 if i == 10 else 0.] + [0.] * 6}
              for i in range(32)]
    trace = public / "robot_trace.jsonl"
    trace.write_text("\n".join(json.dumps({**state, "private_metrics": "SECRET_TARGET"}) for state in states))
    images = []
    for i in range(4):
        image = public / f"frame_{i}.png"
        Image.new("RGB", (8, 8), (i * 40, 20, 30)).save(image)
        images.append(str(image))
    return {"status": "failed", "success": False, "output_dir": str(public.parent),
        "public_evidence": {"status": "failed", "stdout": "", "robot_trace_path": str(trace),
            "trace_tail": states[-3:], "keyframes": images[:2], "read_images": [images[2]],
            "action_images": [{"path": images[3], "sim_time": 5, "action": 0,
                               "method": "move_ee", "phase": "before", "private_metrics": "SECRET_TARGET"}],
            "actions": []}}, states, images


def read_all_text(context, identifier):
    chunks, offset = [], 0
    while True:
        replies, _ = context.read([{"id": identifier, "offset": offset, "limit": READ_CHARS}])
        page = replies[0]
        assert len(dumps(replies)) <= READ_CHARS
        chunks.append(page["text"])
        if page["end"] == page["total_chars"]:
            return "".join(chunks)
        assert page["end"] > offset
        offset = page["end"]


@pytest.mark.parametrize("context_mode", ["research", "coding_initial", "coding_read", "coding_test"])
def test_dense_rollout_indexes_fit_beside_nearly_full_pinned_library(tmp_path, monkeypatch, context_mode):
    """Regression: v6 stopped after 12 trials before asking for any revision."""
    # Keep testing the original tight budget: extra capacity must not replace
    # lossless fallback handling when source eventually grows again.
    monkeypatch.setattr("simex.research_context.MAX_PROMPT_CHARS", 64_000)
    payload, images = large_evidence(tmp_path)
    payload["toolbox"] = {"toolbox.py": "# initial helper\n" * 1550,
                          "skill.md": "Robot API reference\n" * 60}
    for index, row in enumerate(payload["results"]):
        evidence = row["evidence"]
        evidence["action_summary"].update(
            close_commands=1, open_commands=0, min_tcp_height_m=.02,
            tcp_rise_after_close_m=.07, command_stage="closed_then_lifted",
            last_motion_rejection={"source": "policy_stdout", "reason": "FAIL " * 60},
            termination={"kind": "policy_returned", "reason": "STOP " * 60})
        evidence.update(
            keyframes=images, read_images=images, action_images=images,
            robot_trace=[{"sim_time": i, "ee_position": [.4, .2, .3]} for i in range(30)],
            trace_tail=[{"sim_time": i} for i in range(5)],
            recording_errors=[], evidence_errors=[])
        row["execution"]["script"] += f"\n# unique policy {index}\n"
    instruction = "Preserve current source and inspect the experiment evidence. " * 70
    interactive = context_mode.startswith("coding_")
    if interactive:
        from simex.coding import INSTRUCTIONS
        from simex.tasks import make_task
        instruction = INSTRUCTIONS + "\n\n" + research.TOOLBOX_CONTRACT
        # Exact near-limit library plus scene/task catalogs from the real protocol.
        payload["toolbox"] = {"toolbox.py": "#" * 29_700, "skill.md": "Robot API"}
        tasks = [make_task("far_plate", i) for i in range(12)]
        payload["tasks"] = tasks[:4]
        payload["probe_examples"] = [{"task": task, "result": result}
            for task, result in zip(tasks[4:], payload["results"][4:])]
        payload["results"] = payload["results"][:4]
        payload["source_scene_template"] = tasks[0]["scene"]
        payload["source_scene_templates"] = {f"scene_{i}": task["scene"] for i, task in enumerate(tasks[:3])}
        payload["read_history"] = {"read_operations": 24, "consecutive_reads": 24, "omitted_ids": 0,
            "recent": [{"id": "text_" + f"{i:032d}", "ranges": [[0, 8000]], "requests": 2,
                        "repeated_reads": 1, "empty_pages": 0, "omitted_ranges": 0} for i in range(12)]}
        if context_mode == "coding_read":
            payload["last_feedback"] = {"status": "read", "reads": [
                {"id": "text_" + str(i) * 32, "offset": 0, "end": 2000} for i in range(4)]}
            payload["coding_history"] = [{"turn": 0, "response": {"action": "read"},
                                          "feedback": payload["last_feedback"]}]
        elif context_mode == "coding_test":
            payload["last_feedback"] = {"status": "complete",
                "baseline_results": copy.deepcopy(payload["results"]),
                "results": copy.deepcopy(payload["results"])}
    context = ResearchContext(payload, images, pinned_keys=("toolbox",), instruction=instruction)
    context.persist(tmp_path / "dense_context")
    prompt, _ = (context.coding_prompt(instruction) if interactive else context.prompt(instruction))
    assert len(prompt) < 64_000
    assert unpack(prompt)["toolbox"] == payload["toolbox"]
    if interactive:
        assert unpack(prompt)["read_history"] == payload["read_history"]
    def outcomes(value):
        if isinstance(value, dict):
            if "status" in value and "success" in value:
                return [value]
            return [row for child in value.values() for row in outcomes(child)]
        if isinstance(value, list):
            return [row for child in value for row in outcomes(child)]
        return []
    rows, originals = outcomes(context.summary), outcomes(payload)
    assert len(rows) == len(originals) and all(row["success"] is False for row in rows)
    compacted = [row for row in rows if row.get("detail_overview")]
    assert compacted
    for row in compacted:
        assert row["evidence"]["action_summary"]["min_tcp_height_m"] == .02
        assert row["evidence"]["action_summary"]["close_commands"] == 1
        details = json.loads(read_all_text(context, row["detail_overview"]["evidence_id"]))
        assert details["evidence"]["action_summary"]["command_stage"] == "closed_then_lifted"
        catalog_id = details["evidence"]["keyframes"]["evidence_id"]
        catalog = json.loads(read_all_text(context, catalog_id))
        assert len(catalog) == 12
        assert all(entry["evidence_id"] in context.assets for entry in catalog)
        original = json.loads(read_all_text(context, details["full_result"]["evidence_id"]))
        assert original in originals


def test_public_raw_evidence_remains_readable_with_summaries_and_cached_results(tmp_path):
    result, states, originals = public_rollout_with_raw_evidence(tmp_path)
    original_result = copy.deepcopy(result)
    public_result = research._public(result)
    # Exercise both the new cache and migration from the older summary-only cache.
    assert research._public(result) == public_result
    cache_path = Path(result["output_dir"]) / "public/research_evidence.json"
    cache = json.loads(cache_path.read_text())
    cache["source_metadata"]["version"] = 2
    cache["summary"].pop("robot_trace")
    cache_path.write_text(json.dumps(cache))
    assert research._public(result) == public_result
    assert result == original_result
    evidence = public_result["evidence"]
    assert evidence["robot_trace"] == states
    assert evidence["trace_tail"] == states[-3:]
    assert evidence["motion_summary"]["initial"]["joint_velocities"] == [0.] * 7
    assert evidence["motion_summary"]["final"]["joint_velocities"] == [0.] * 7
    payload = {"target": public_result}
    defaults = research._images(payload)
    assert defaults == evidence["phase_images"]
    assert set(research._images(payload, include_all=True)) == set(defaults + originals)
    context = ResearchContext(payload, research._images(payload, include_all=True),
                              default_image_paths=defaults)
    context.persist(tmp_path / "context")
    packet = context.summary["target"]["evidence"]
    assert json.loads(read_all_text(context, packet["robot_trace"]["evidence_id"])) == states
    assert json.loads(read_all_text(context, packet["trace_tail"]["evidence_id"])) == states[-3:]
    for name, paths in (("keyframes", originals[:2]), ("read_images", originals[2:3]),
                        ("action_images", originals[3:])):
        catalog = json.loads(read_all_text(context, packet[name]["evidence_id"]))
        assert len(catalog) == len(paths)
        for entry, path in zip(catalog, paths):
            identifier = (entry["path"] if name == "action_images" else entry)["evidence_id"]
            replies, requested = context.read([{"id": identifier}])
            _, attached = context.prompt("Inspect the original frame.", replies=replies, image_ids=requested)
            assert len(attached) == 1
            assert Path(attached[0]).read_bytes() == Path(path).read_bytes()
    assert "SECRET_TARGET" not in dumps(context.summary)
    assert all("SECRET_TARGET" not in asset.get("text", "") for asset in context.assets.values())


def test_ask_reads_raw_trace_then_attaches_requested_original_frame(tmp_path):
    result, _, originals = public_rollout_with_raw_evidence(tmp_path)
    payload = {"target": research._public(result)}
    sheet = Path(payload["target"]["evidence"]["phase_images"][0]).read_bytes()
    original_image = Path(originals[2]).read_bytes()

    class Reader:
        count = 0

        def ask(self, prompt, images, **kwargs):
            self.count += 1
            assert len(prompt) < MAX_PROMPT_CHARS
            assert len(images) == 1
            packet = unpack(prompt)
            meta = packet["_research_context"]
            if self.count == 1:
                assert Path(images[0]).read_bytes() == sheet
                evidence = packet["target"]["evidence"]
                return {"evidence_requests": [
                    {"id": evidence["robot_trace"]["evidence_id"], "find": '"sim_time":10', "limit": 500},
                    {"id": evidence["read_images"]["evidence_id"]}]}
            if self.count == 2:
                assert Path(images[0]).read_bytes() == sheet
                assert "0.812345" in meta["read_results"][0]["text"]
                catalog = json.loads(meta["read_results"][1]["text"])
                self.image_id = catalog[0]["evidence_id"]
                return {"evidence_requests": [{"id": self.image_id}],
                        "notes": "The intermediate joint velocity is absent from the endpoint summary."}
            assert self.count == 3
            assert meta["attached_images"][0]["id"] == self.image_id
            assert Path(images[0]).read_bytes() == original_image
            return {"aligned": False, "reason": "Reviewed the intermediate sample and original frame."}

    agent = Reader()
    answer = research._ask(agent, tmp_path / "decision.json", "align", "Inspect the public evidence.", payload)
    assert answer["aligned"] is False
    assert agent.count == 3


def test_large_image_catalogs_and_traces_keep_the_overview_bounded(tmp_path):
    payload, images = large_evidence(tmp_path)
    states = [{"sim_time": i, "joints": [i / 100] * 7} for i in range(160)]
    for record in payload["results"]:
        record["evidence"].update(keyframes=images * 40, robot_trace=states)
    context = ResearchContext(payload, images, default_image_paths=images[:MAX_IMAGES])
    context.persist(tmp_path / "catalog_context")
    prompt, attached = context.prompt("Inspect public motion and frames.")
    assert len(prompt) < MAX_PROMPT_CHARS
    assert len(dumps(context.summary)) <= SUMMARY_CHARS
    assert len(attached) == MAX_IMAGES
    for record in context.summary["results"]:
        evidence = record["evidence"]
        assert evidence["keyframes"]["kind"] == "image_catalog"
        assert evidence["keyframes"]["count"] == 480
        assert evidence["robot_trace"]["count"] == len(states)
    evidence = context.summary["results"][-1]["evidence"]
    catalog = json.loads(read_all_text(context, evidence["keyframes"]["evidence_id"]))
    assert len(catalog) == 480
    assert all(entry["evidence_id"] in context.image_ids for entry in catalog)
    assert json.loads(read_all_text(context, evidence["robot_trace"]["evidence_id"])) == states


def test_interrupted_read_round_resumes_without_repeating_completed_turn(tmp_path):
    class Interactive:
        count = 0
        fail = True
        calls = []
        def ask(self, role, prompt, images=None):
            self.calls.append(prompt)
            payload = unpack(prompt)
            if not payload["_research_context"]["read_results"]:
                self.count += 1
                return {"evidence_requests": [{"id": payload["script"]["evidence_id"], "find": "BUG_LINE"}],
                        "notes": "Read the failure site."}
            if self.fail:
                raise AgentTransportError("connection interrupted")
            self.count += 1
            assert "BUG_LINE" in payload["_research_context"]["read_results"][0]["text"]
            return {"aligned": True, "reason": "Observed in the requested source."}

    agent = Interactive()
    target = tmp_path / "decision.json"
    payload = {"script": "# code\n"*1_000 + "BUG_LINE"}
    with pytest.raises(AgentTransportError):
        research._ask(agent, target, "align", "Diagnose.", payload)
    assert agent.count == 1  # Only the completed read is charged.
    assert len(agent.calls) == 7  # One read, then one connection attempt plus five retries.
    assert (tmp_path / "decision.context/turn_00.json").exists()
    assert not target.exists()
    turn_path = tmp_path / "decision.context/turn_00.json"
    turn = json.loads(turn_path.read_text())
    turn["request_sha256"] = "outdated hash"
    turn_path.write_text(json.dumps(turn))
    manifest_path = tmp_path / "decision.context/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_sha256"] = manifest["overview_sha256"] = "outdated hash"
    manifest_path.write_text(json.dumps(manifest))
    agent.fail = False
    answer = research._ask(agent, target, "align", "Diagnose.", payload)
    assert answer["aligned"] and agent.count == 2
    assert len(set(agent.calls[1:])) == 1
    assert research._ask(agent, target, "align", "Diagnose.", payload) == answer
    assert agent.count == 2
    assert research._ask(agent, target, "align", "Diagnose.",
                         {"script": payload["script"] + "changed"}) == answer
    assert agent.count == 2


def test_nontransport_errors_do_not_trigger_resampling(tmp_path):
    class Invalid:
        count = 0
        def ask(self, **kwargs):
            self.count += 1
            raise AgentError("Invalid JSON")
    agent = Invalid()
    with pytest.raises(AgentError, match="Invalid JSON"):
        research._ask(agent, tmp_path / "decision.json", "align", "Decide.", {})
    assert agent.count == 1


def test_read_round_limit_is_durable_and_never_runs_a_rollout(tmp_path):
    assert MAX_INTERACTIONS == MAX_READ_ROUNDS + 1 == 20
    class Endless:
        count = 0
        def ask(self, prompt, **kwargs):
            self.count += 1
            return {"evidence_requests": [{"id": unpack(prompt)["script"]["evidence_id"]}]}
    agent = Endless()
    payload = {"script": "code\n"*1_000}
    for _ in range(2):
        with pytest.raises(research.ResearchError, match="read budget exhausted"):
            research._ask(agent, tmp_path / "decision.json", "align", "Decide.", payload)
        assert agent.count == MAX_READ_ROUNDS + 1


def test_twentieth_interaction_can_decide_after_nineteen_bounded_reads(tmp_path):
    class Reader:
        count = 0
        def ask(self, prompt, **kwargs):
            value = unpack(prompt)
            assert len(prompt) <= MAX_PROMPT_CHARS - 1_000
            remaining = value["_research_context"]["read_rounds_remaining"]
            assert remaining == MAX_READ_ROUNDS - self.count
            self.count += 1
            if remaining == 0:
                return {"aligned": True}
            start = (self.count - 1) * 4
            return {"evidence_requests": [
                {"id": item["evidence_id"], "limit": 4000}
                for item in value["scripts"][start:start + 4]],
                "notes": "Detailed observations. " * 90}

    agent = Reader()
    payload = {"scripts": [f"# source {i}\n" + "x" * 6000 for i in range(76)]}
    result = research._ask(agent, tmp_path / "decision.json", "align", "Compare the evidence.", payload)
    assert result == {"aligned": True}
    assert agent.count == 20


def test_image_hash_change_does_not_repeat_completed_decision(tmp_path):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "red").save(frame)
    class Final:
        count = 0
        def ask(self, **kwargs):
            self.count += 1
            return {"aligned": True}
    agent = Final()
    payload = {"phase_images": [str(frame)]}
    path = tmp_path / "decision.json"
    research._ask(agent, path, "align", "Decide.", payload)
    Image.new("RGB", (8, 8), "blue").save(frame)
    assert research._ask(agent, path, "align", "Decide.", payload) == {"aligned": True}
    assert agent.count == 1


def test_large_library_uses_exact_edits_and_preserves_unseen_code(tmp_path):
    source = "class Toolbox:\n    def __init__(self, robot):\n        self.robot = robot\n"
    source += "# unrelated existing primitive\n" * 3_000 + "THRESHOLD = 1\n"
    current = {"toolbox.py": source, "skill.md": "Preserve public API"}
    class Editor:
        count = 0
        def ask(self, prompt, **kwargs):
            self.count += 1
            payload = unpack(prompt)
            if not payload["_research_context"]["read_results"]:
                return {"evidence_requests": [{"id": payload["toolbox"]["toolbox.py"]["evidence_id"],
                                               "find": "THRESHOLD", "limit": 100}]}
            assert "THRESHOLD = 1" in payload["_research_context"]["read_results"][0]["text"]
            return {"edits": [{"file": "toolbox.py", "old": "THRESHOLD = 1\n", "new": "THRESHOLD = 2\n"}],
                    "hypothesis": "Local correction"}
    agent = Editor()
    path = tmp_path / "proposal.json"
    proposal = research._ask(agent, path, "discover_revision", "Revise.", {"toolbox": current})
    assert proposal["toolbox.py"] == source.replace("THRESHOLD = 1", "THRESHOLD = 2")
    assert proposal["skill.md"] == current["skill.md"]
    candidate, validation = research._candidate(proposal, tmp_path / "candidate")
    assert candidate and validation["status"] == "valid"
    assert agent.count == 2
    assert research._ask(agent, path, "discover_revision", "Revise.", {"toolbox": current}) == proposal
    assert agent.count == 2


@pytest.mark.parametrize("edit", [
    {"file": "../private.json", "old": "one", "new": "two"},
    {"file": "toolbox.py", "old": "", "new": "two"},
    {"file": "toolbox.py", "old": "missing", "new": "two"},
    {"file": "toolbox.py", "old": "repeat", "new": "two"},
])
def test_invalid_source_edit_cannot_write_files_or_choose_an_ambiguous_match(edit):
    current = {"toolbox.py": "repeat repeat one", "skill.md": "notes"}
    with pytest.raises(ContextError):
        apply_source_edits({"edits": [edit]}, current)
    assert current["toolbox.py"] == "repeat repeat one"


def test_bad_block_edit_consumes_candidate_without_model_retry(tmp_path):
    class InvalidEditor:
        count = 0
        def ask(self, **kwargs):
            self.count += 1
            return {"edits": [{"file": "toolbox.py", "old": "missing", "new": "unsafe guess"}]}
    current = {"toolbox.py": "# unchanged\n"*4_000, "skill.md": "notes"}
    agent = InvalidEditor()
    proposal = research._ask(agent, tmp_path / "proposal.json", "repair", "Repair.", {"toolbox": current})
    candidate, validation = research._candidate(proposal, tmp_path / "candidate")
    assert candidate is None and "exactly once" in validation["reason"]
    assert agent.count == 1


def test_offline_audit_checks_evidence_presence_without_hash_verification(tmp_path):
    from scripts.verify_run import verify_contexts
    ctx = ResearchContext({"script": "# immutable\n"*1_000})
    ctx.persist(tmp_path / "decision.context")
    errors = []
    assert verify_contexts(tmp_path, errors) and not errors
    blob = next((tmp_path / "decision.context").glob("text_*.txt"))
    blob.write_text("changed")
    errors = []
    verify_contexts(tmp_path, errors)
    assert not errors
    blob.unlink()
    verify_contexts(tmp_path, errors)
    assert any("evidence blob missing" in error for error in errors)


def test_progress_and_termination_stay_visible_when_rollouts_are_compacted(tmp_path):
    payload, images = large_evidence(tmp_path, count=24)
    for record in payload['results']:
        record['evidence']['action_summary'].update(
            close_commands=1, open_commands=0, min_tcp_height_m=.016,
            tcp_rise_after_close_m=.08, command_stage='tcp_raised_after_close',
            last_motion_rejection={'action_index': 12, 'method': 'move_ee',
                                   'source': 'action_result', 'reason': 'unreachable'},
            termination={'status': 'failed', 'kind': 'policy_returned',
                         'source': 'policy_stdout', 'reason': 'STOP held view unavailable'})
        record['evidence']['stdout'] += '\n' + 'postmortem images logged\n'*100
    ctx = ResearchContext(payload, images)
    ctx.persist(tmp_path / 'progress_context')
    assert len(dumps(ctx.summary)) <= SUMMARY_CHARS
    assert any(r.get('brief_overview') for r in ctx.summary['results'])
    for original, compact in zip(payload['results'], ctx.summary['results']):
        assert compact['evidence']['action_summary'] == original['evidence']['action_summary']
        assert compact['evidence']['phase_images'][0]['evidence_id'] in ctx.image_ids
        if compact.get('brief_overview'):
            assert compact['evidence']['last_diagnostic_line'] == 'STOP tracking error'
    prompt, attached = ctx.prompt('Inspect the observed commands and independent outcomes.')
    assert len(prompt) < MAX_PROMPT_CHARS
    assert len(attached) == MAX_IMAGES
    assert 'does not establish object contact' in prompt


def test_closure_transition_survives_the_smallest_rollout_overview(tmp_path):
    payload, images = large_evidence(tmp_path, count=2)
    closure = {'action_index': 25, 'command_width_m': 0., 'command_force_n': 35.,
               'endpoint_width_m': .04, 'next_arm_delay_s': .2,
               'next_arm_start_width_m': .018,
               'next_arm_endpoint': {'action_index': 28, 'width_m': .005,
                                     'tcp_delta_m': [0., 0., .02]}}
    for row in payload['results']:
        row['evidence']['action_summary']['first_close_response'] = closure.copy()
    ctx = ResearchContext(payload, images)
    for row in ctx.summary['results']:
        ctx._compact_rollout(row)
    ctx.persist(tmp_path / 'closure_context')
    for row in ctx.summary['results']:
        compact = row['evidence']['action_summary']['first_close_response']
        assert compact == {key: closure[key] for key in
                           ('action_index', 'endpoint_width_m', 'next_arm_delay_s', 'next_arm_start_width_m')}
        detail = json.loads(read_all_text(ctx, row['detail_overview']['evidence_id']))
        assert detail['evidence']['action_summary']['first_close_response'] == closure
    prompt, _ = ctx.prompt('Compare the public action sequence.')
    assert len(prompt) < MAX_PROMPT_CHARS
    assert 'Command return does not establish' in prompt


def test_context_diagnostic_excerpts_skip_successful_recovery_records(tmp_path):
    payload, images = large_evidence(tmp_path, count=1)
    payload['results'][0]['evidence']['stdout'] += (
        "\nMOTION {'success': True, 'results': [{'status': 'failed'}, {'status': 'accepted'}]}")
    context = ResearchContext(payload, images)
    stdout = context.summary['results'][0]['evidence']['stdout']
    assert stdout['last_diagnostic_line'] == 'STOP tracking error'
