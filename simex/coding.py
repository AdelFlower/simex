"""Checkpointed Stage 1 coding operations, executed only by the trusted harness."""
from __future__ import annotations

import ast
import copy
import os
from pathlib import Path

from . import research as r
from .agent import Agent, AgentBudgetExhausted, AgentResponseError, AgentTransportError
from .research_context import (
    ContextError, MAX_FULL_FILE_CHARS, MAX_NOTES_CHARS, MAX_PINNED_CHARS, MAX_PROMPT_CHARS,
    READ_CHARS, ResearchContext, apply_source_edits, dumps,
)


INSTRUCTIONS = """
Improve the reusable capability library for the supplied weakness using public
source experiment evidence. Return exactly one JSON operation per interaction.
The harness performs these operations; do not use CLI tools or access host files.
Each operation may include "notes" (at most 2000 characters, replacing prior notes).

Operations:
{"action":"read","evidence_requests":[{"id":"text_...","offset":0,"limit":8000}]}
{"action":"edit","files":{"toolbox.py":"complete source","skill.md":"complete text"},"hypothesis":"..."}
{"action":"edit","edits":[{"file":"toolbox.py","old":"exact block","new":"replacement"}],"hypothesis":"..."}
{"action":"test"}
{"action":"decide","draft_id":"draft_...","test_id":"test_...","accept":true,"reason":"...",
 "accept_unresolved_history":false,"history_reason":""}
{"action":"reset"}
{"action":"finish","reason":"..."}

Read only listed evidence IDs: at most 4 reads, 16000 returned JSON characters and
4 attached images per interaction. For images use {"id":"image_..."}; text reads
may use "find":"literal substring". Excerpts are incomplete; read relevant source
before editing. Old IDs remain readable. Only the latest read pages are attached;
carry findings in notes. Full operation history is available via coding_history.
read_history lists ranges already delivered and repeated/empty reads. Use actual
returned end offsets; a requested limit does not mean that many characters arrived.
Prefer one or two targeted reads, using find for the relevant function or failure.
Notes must preserve concrete new observations and their evidence IDs, not repeat
the general rules. Once the evidence supports a testable mechanism, make a small
edit and test it. Do not exhaustively reread similar failures before trying a repair.
Reserve interactions for edit, test and decide; reading alone acquires no capability.

Accumulate executable capabilities, as well as guidance. When fresh policies
reimplement the same multi-step perception or manipulation routine inconsistently,
use the successful and failed executions to identify a shared procedure and
factor it into a reusable toolbox helper. A small parameter edit is useful when
it addresses the mechanism; adding a helper is appropriate when the missing
capability is a procedure. Repeated prose-only revisions are not a substitute
for an implemented procedure when the evidence supports one. Derive object
targets from runtime RGBD, preserve measured uncertainty and bounded recovery,
and document the helper's preconditions, postconditions and remaining work.
Do not store a whole successful task script, its fixture coordinates or its
particular scene as a capability. Test through fresh policies and verify that
they actually exercise the new helper before attributing outcomes to it.
peer_experiment_summaries are the same summaries frozen at this cycle's probe
boundary. Use their ideas for independent implementation and testing; a peer's
success is not this member's demonstrated capability. No peer implementation is available.

Edit only toolbox.py and skill.md. Full files are allowed only when
edit_mode=full_or_edits; otherwise use 1 to 16 exact replacements. Each nonempty
old block must match exactly once after previous replacements. Edits immediately
return static validation. Invalid source remains editable but cannot be tested.
Replacement errors preserve the prior draft. Correct errors using another edit.

The working draft and accepted toolbox are separate. Rejected drafts remain
editable; reset restores the accepted toolbox. Test runs the entire frozen focused
and replay task set with fresh one-shot policies. Repeating a test for the same
draft returns saved evidence. A test must leave a subsequent interaction for a
decision. Test results include executed scripts, action summaries and public
evidence. Read them before deciding; robot motion alone does not prove acquisition.
Top-level results describe the accepted version. latest_test persistently contains
the most recent tested draft's results, its baseline_results, regression and IDs,
even after intervening reads or edits. Check its draft_id against working_draft:
editing after that test leaves the new draft untested. Default images come from
latest_test.results when available; do not confuse baseline and candidate evidence.

Decide only about the current tested draft. An edit after testing requires another
test. Accept only a plausible reusable improvement supported by observed behavior
and code. Focused score decreases and currently passing replays becoming failures
cannot be accepted. Historical tasks failing in both versions require explicit
accept_unresolved_history=true and a substantive history_reason. With zero
successes, explain the hypothesis actually exercised and leave success unverified.
Different fresh policies confound causal attribution; a tie alone is not improvement.
If the relevant trials stop before arm motion, retain the accepted toolbox and
repair the executable exploration path. Decisions are final for their test.

Use remaining_interactions and remaining_tests to plan work. Static failures and
invalid operations spend an interaction, not a test. All model calls, including
policy generation, share the global budget; confirmed network failures are exempt.
Finish when further changes are not justified. The harness retains only the last
accepted version on finish or interaction exhaustion, never an untested draft.
""".strip()


def _immutable_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise r.ResearchError(f"Immutable coding draft changed: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validation(files):
    try:
        r._validate_toolbox(files)
        return {"status": "valid"}
    except r.InvalidToolbox as exc:
        cause = exc.__cause__ or exc
        line = getattr(cause, "lineno", None)
        if line is None:
            try:
                tree = ast.parse(files["toolbox.py"])
                line = next((node.lineno for node in tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "Toolbox"), 1)
            except SyntaxError:
                line = 1
        lines = files["toolbox.py"].splitlines()
        start = max(0, line - 3)
        return {"status": "invalid_candidate", "reason": str(exc), "file": "toolbox.py",
                "line": line, "column": getattr(cause, "offset", None),
                "excerpt": "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, min(len(lines), line + 2)))}


def statistics(state, directory=None):
    drafts = {key: value["validation"] for key, value in state["drafts"].items() if key != "initial"}
    tests = {value["id"]: value for value in state["tests"]}
    turns = state["turns_used"]
    if directory is not None:
        directory = Path(directory)
        # Include reserved work that has not reached its per-turn commit.
        for path in (directory / "drafts").glob("*/validation.json"):
            drafts.setdefault(path.parent.name, r._json(path))
        for path in (directory / "tests").glob("*/test.json"):
            tests.setdefault(path.parent.name, r._json(path))
        pending = r._json(directory / "turns" / f"{turns:03d}.json", {})
        if pending.get("charged") and "after" not in pending:
            turns += 1
    return {"status": state["status"], "end_reason": state.get("end_reason"),
            "coding_turns": turns, "drafts": len(drafts),
            "validation_failures": sum(value["status"] != "valid" for value in drafts.values()),
            "test_batches": len(tests),
            "completed_test_batches": sum(t["status"] == "complete" for t in tests.values()),
            "accepted": len(state["accepted"]),
            "accepted_draft": state["current_id"], "working_draft": state["draft_id"]}


def coding_report(run_dir):
    sessions = [{"directory": str(path.parent.relative_to(run_dir)), **r._json(path)}
                for path in sorted(Path(run_dir).glob("discovery/member_*/cycle_*/coding/summary.json"))]
    fields = ("coding_turns", "drafts", "validation_failures", "test_batches",
              "completed_test_batches", "accepted")
    return {**{key: sum(item[key] for item in sessions) for key in fields}, "sessions": sessions}


def session_history(directory):
    """Only model-visible responses and public feedback, never raw harness state."""
    history = []
    for path in sorted((Path(directory) / "turns").glob("*.json")):
        record = r._json(path)
        if "after" in record:
            history.append({"turn": record["turn"], "response": record.get("response"),
                            "response_error": record.get("response_error"),
                            "feedback": record["feedback"]})
    return history


def read_history(history, limit=12):
    """Bounded coverage memory across the otherwise stateless model calls."""
    seen, operations, consecutive = {}, 0, 0
    for turn in history:
        response = turn.get("response")
        action = response.get("action") if isinstance(response, dict) else None
        if action != "read":
            if action in ("edit", "test", "decide", "reset", "finish"):
                consecutive = 0
            continue
        operations += 1
        consecutive += 1
        for item in turn.get("feedback", {}).get("reads", []):
            identifier = item.get("id")
            start, end = item.get("offset"), item.get("end")
            if not isinstance(identifier, str) or type(start) is not int or type(end) is not int:
                continue
            record = seen.pop(identifier, {"id": identifier, "ranges": [], "requests": 0,
                                           "repeated_reads": 0, "empty_pages": 0})
            record["requests"] += 1
            if end <= start:
                record["empty_pages"] += 1
            else:
                record["repeated_reads"] += any(a <= start and end <= b for a, b in record["ranges"])
                merged = []
                for a, b in sorted(record["ranges"] + [[start, end]]):
                    if merged and a <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], b)
                    else:
                        merged.append([a, b])
                record["ranges"] = merged
            seen[identifier] = record
    recent = []
    for record in list(seen.values())[-limit:]:
        # Coverage remains exact; omitted disjoint ranges are explicitly counted.
        recent.append({**record, "ranges": record["ranges"][-4:],
                       "omitted_ranges": max(0, len(record["ranges"]) - 4)})
    return {"read_operations": operations, "consecutive_reads": consecutive,
            "omitted_ids": max(0, len(seen) - limit), "recent": recent}


class CodingSession:
    def __init__(self, runner, agent, directory, cfg, initial, payload, tasks, focused, replay, cycle):
        self.runner, self.agent = runner, agent
        self.directory = Path(directory)
        self.run_dir = self.directory.parents[3]
        self.max_turns, self.max_tests = cfg["coding_max_turns"], cfg["coding_max_tests"]
        bundle = r._json(self.directory / "initial.json")
        if bundle is None:
            current = str(Path(initial["current"]).resolve())
            state = {"version": 1, "status": "running", "end_reason": None, "turns_used": 0,
                     "current": current, "current_id": "initial", "draft_id": "initial",
                     "current_results": initial["current_results"], "bank": initial["bank"],
                     "drafts": {"initial": {"path": current, "validation": {"status": "valid"},
                                            "hypothesis": "Initial accepted toolbox"}},
                     "tests": [], "accepted": [], "notes": "", "read_results": [], "image_ids": None,
                     "last_feedback": {"status": "ready"}}
            bundle = {"state": state, "payload": payload, "tasks": tasks, "focused": focused,
                      "replay": replay, "cycle": cycle, "max_turns": self.max_turns,
                      "max_tests": self.max_tests}
            r._save(self.directory / "initial.json", bundle)
        if (bundle["max_turns"], bundle["max_tests"]) != (self.max_turns, self.max_tests):
            raise r.ResearchError("Coding budgets changed; create a new run")
        self.payload, self.tasks = bundle["payload"], bundle["tasks"]
        self.focused, self.replay, self.cycle = bundle["focused"], bundle["replay"], bundle["cycle"]
        self.baseline_results = bundle["state"]["current_results"]
        self.state = copy.deepcopy(bundle["state"])
        # The per-turn commit is authoritative. state.json is a reporting snapshot.
        for index in range(self.max_turns):
            record = r._json(self.directory / "turns" / f"{index:03d}.json")
            if record is None or "after" not in record:
                break
            self.state = record["after"]
        self.state = self._restore_state(self.state, self.baseline_results)
        if not any(item.get("kind") == "image" and item.get("attached")
                   for item in self.state["read_results"]):
            # Older checkpoints cached default baseline images after text reads.
            # Only explicitly requested images should override current defaults.
            self.state["image_ids"] = None
        self.context = None

    def _checkpoint_state(self, state):
        # Rollout evidence is durable in test.json. Do not duplicate it in forty
        # per-turn commits and the status snapshot.
        checkpoint = {key: value for key, value in state.items() if key not in ("current_results", "tests")}
        checkpoint["results_stored_separately"] = True
        checkpoint["tests"] = [{key: value for key, value in test.items()
                                if key not in ("results", "baseline_results")} for test in state["tests"]]
        return checkpoint

    def _restore_state(self, state, baseline):
        if not state.pop("results_stored_separately", False):
            return state
        tests = []
        for entry in state["tests"]:
            saved = r._json(self.directory / "tests" / entry["id"] / "test.json")
            if saved is None or saved.get("status") != "complete":
                raise r.ResearchError("Completed coding test checkpoint is missing")
            tests.append({**saved, **entry})
        state["tests"] = tests
        state["current_results"] = (baseline if state["current_id"] == "initial" else
            next(test["results"] for test in tests if test["draft_id"] == state["current_id"]))
        return state

    def _snapshot(self):
        r._save(self.directory / "state.json", self._checkpoint_state(self.state))
        r._save(self.directory / "summary.json", statistics(self.state, self.directory))
        r._save(self.directory.parent.parent / "capability_bank.json", self.state["bank"])

    def _context(self):
        state = self.state
        files = r._contents(Path(state["drafts"][state["draft_id"]]["path"]))
        source_chars = len(dumps(files))
        small = source_chars <= MAX_FULL_FILE_CHARS
        tests = [{"test_id": item["id"], "draft_id": item["draft_id"], "status": item["status"],
                  "decision": item.get("decision")} for item in state["tests"]]
        history = session_history(self.directory)
        latest_test = self._test_feedback(state["tests"][-1]) if state["tests"] else None
        feedback = state["last_feedback"]
        if latest_test and feedback.get("test_id") == latest_test["test_id"]:
            # Keep one durable copy of each result set; a read must not make the
            # latest experiment disappear into the paged operation history.
            feedback = {key: value for key, value in feedback.items()
                        if key not in ("results", "baseline_results")}
        payload = {**self.payload, "toolbox": files, "working_draft": state["draft_id"],
                   "accepted_draft": state["current_id"],
                   "accepted_toolbox": r._contents(Path(state["current"])),
                   "results": [r._source_evidence(item) for item in state["current_results"]],
                   "validation": state["drafts"][state["draft_id"]]["validation"],
                   "remaining_interactions": self.max_turns - state["turns_used"],
                   "remaining_tests": self.max_tests - len(state["tests"]),
                   "edit_mode": "full_or_edits" if small else "edits_only",
                   "tests": tests, "latest_test": latest_test, "last_feedback": feedback,
                   "read_history": read_history(history), "coding_history": history}
        instruction = INSTRUCTIONS + "\n\n" + r.TOOLBOX_CONTRACT
        default_images = (r._images(latest_test["results"]) if latest_test else [])
        default_images = (default_images or r._images(feedback.get("results", []))
                          or r._images(payload["results"])
                          or r._images(self.payload))
        context = ResearchContext(payload, r._images(payload, include_all=True),
            default_image_paths=default_images,
            pinned_keys=("toolbox",) if source_chars <= MAX_PINNED_CHARS else (), instruction=instruction)
        evidence_dir = self.directory / "evidence.context"
        if (evidence_dir / "manifest.json").exists():
            context.retain_assets(ResearchContext.load(evidence_dir))
        context.persist(evidence_dir)
        self.context = context
        return instruction

    def _request(self, record, path):
        """Persist the response before executing its operation."""
        if "response" in record or "response_error" in record:
            return
        if record.get("charged"):
            # Recover a completed adapter response after a crash at the call boundary.
            call_index = record.get("call_index")
            calls_dir = getattr(self.agent, "calls_dir", None)
            call_dir = Path(calls_dir) / f"{call_index:05d}_discover_code" if calls_dir and call_index else None
            result = r._json(call_dir / "result.json", {}) if call_dir else {}
            if result.get("budget_charged") is False:
                record["charged"] = False
            elif (call_dir and result.get("status") == "ok" and not result.get("prohibited_tools")
                  and (call_dir / "response.txt").is_file()):
                try:
                    record["response"] = Agent.parse_response("discover_code", (call_dir / "response.txt").read_text())
                except AgentResponseError as exc:
                    record["response_error"] = {"reason": str(exc), "output": exc.response}
            else:
                record["response_error"] = {
                    "reason": "The previous model request was interrupted without a saved operation; "
                              "the interaction remains charged. Inspect the current state and continue."}
            r._save(path, record)
            if "response" in record or "response_error" in record:
                return
        for attempt in range(r.MAX_TRANSPORT_RETRIES + 1):
            r._agent_capacity(self.agent)
            record["charged"] = True
            next_index = getattr(self.agent, "_call_index", None)
            record["call_index"] = next_index + 1 if isinstance(next_index, int) else None
            r._save(path, record)
            self._snapshot()
            try:
                record["response"] = self.agent.ask(**record["request"])
            except AgentBudgetExhausted:
                record["charged"] = False
                r._save(path, record)
                raise
            except AgentTransportError as exc:
                record["charged"] = False
                record.setdefault("transport_errors", []).append(
                    {"error": str(exc), "automatic_retry": attempt < r.MAX_TRANSPORT_RETRIES})
                r._save(path, record)
                if attempt == r.MAX_TRANSPORT_RETRIES:
                    raise
                continue
            except AgentResponseError as exc:
                record["response_error"] = {"reason": str(exc), "output": exc.response}
            except BaseException:
                # A charged, unanswered request must not acquire a fresh free turn
                # on restart. Its adapter logs can still supply a durable response.
                calls_dir = getattr(self.agent, "calls_dir", None)
                index = record["call_index"]
                if calls_dir and index:
                    result = r._json(Path(calls_dir) / f"{index:05d}_discover_code" / "result.json", {})
                    if result.get("budget_charged") is False:
                        record["charged"] = False
                r._save(path, record)
                raise
            r._save(path, record)
            return

    def _edit(self, response, state, turn):
        old_path = Path(state["drafts"][state["draft_id"]]["path"])
        old = r._contents(old_path)
        if ("files" in response) == ("edits" in response):
            raise ContextError("Edit requires either files or edits, not both")
        hypothesis = response.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ContextError("Edit requires a nonempty hypothesis")
        if "files" in response:
            files = response["files"]
            if len(dumps(old)) > MAX_FULL_FILE_CHARS:
                raise ContextError("This draft requires exact block edits; read source pages first")
            if (not isinstance(files, dict) or set(files) != set(r.CAPABILITY_FILES)
                    or any(not isinstance(value, str) for value in files.values())):
                raise ContextError("files must contain complete toolbox.py and skill.md strings")
        else:
            edited = apply_source_edits({"edits": response["edits"], "hypothesis": hypothesis}, old)
            files = {name: edited[name] for name in r.CAPABILITY_FILES}
        if files == old:
            return {"status": "unchanged", "draft_id": state["draft_id"],
                    "validation": state["drafts"][state["draft_id"]]["validation"]}
        draft_id = f"draft_{turn:03d}"
        directory = self.directory / "drafts" / draft_id
        for name in r.CAPABILITY_FILES:
            _immutable_text(directory / name, files[name])
        validation = _validation(files)
        r._save(directory / "validation.json", validation)
        state["drafts"][draft_id] = {"path": str(directory.resolve()), "validation": validation,
                                    "hypothesis": hypothesis}
        state["draft_id"] = draft_id
        return {"status": "edited", "draft_id": draft_id, "validation": validation,
                "changes": r._changes(old_path, files)}

    def _test(self, state):
        draft_id = state["draft_id"]
        previous = next((item for item in state["tests"] if item["draft_id"] == draft_id), None)
        if previous is not None:
            return self._test_feedback(previous)
        if draft_id == "initial":
            return {"status": "baseline", "draft_id": draft_id,
                    "results": [r._source_evidence(item) for item in state["current_results"]]}
        draft = state["drafts"][draft_id]
        if draft["validation"]["status"] != "valid":
            raise ContextError("Fix the current draft's validation errors before testing")
        if len(state["tests"]) >= self.max_tests:
            raise ContextError("No test batches remain; read, decide, reset or finish")
        if self.max_turns - state["turns_used"] < 2:
            raise ContextError("A new test must leave an interaction for its decision")
        test_id = f"test_{len(state['tests']):03d}"
        directory = self.directory / "tests" / test_id
        test_path = directory / "test.json"
        test = r._json(test_path)
        if test is None:
            test = {"id": test_id, "draft_id": draft_id, "status": "running",
                    "baseline_id": state["current_id"], "baseline_path": state["current"],
                    "baseline_results": state["current_results"], "draft_path": draft["path"]}
            r._save(test_path, test)  # reserve once, before generating policies or executing
            self._snapshot()
        if test["draft_id"] != draft_id or test["baseline_id"] != state["current_id"]:
            raise r.ResearchError("Pending coding test does not match the saved draft and baseline")
        if test["status"] != "complete":
            results = r._suite(self.runner, self.tasks, Path(draft["path"]), directory / "rollouts")
            regression = r._discovery_regression(test["baseline_results"], results, self.focused, self.replay)
            test.update(status="complete", results=results, regression=regression)
            r._save(test_path, test)
        state["tests"].append(test)
        return self._test_feedback(test)

    @staticmethod
    def _test_feedback(test):
        return {"status": test["status"], "test_id": test["id"], "draft_id": test["draft_id"],
                "baseline_id": test["baseline_id"], "regression": test["regression"],
                "baseline_results": [r._source_evidence(item) for item in test["baseline_results"]],
                "results": [r._source_evidence(item) for item in test["results"]],
                "decision": test.get("decision")}

    def _decide(self, response, state):
        draft_id = response.get("draft_id")
        if draft_id != state["draft_id"]:
            raise ContextError("Decide must reference the current working draft")
        test = next((item for item in state["tests"] if item["id"] == response.get("test_id")), None)
        if test is None or test["status"] != "complete" or test["draft_id"] != draft_id:
            raise ContextError("Decide requires a complete test of this exact draft; edits require a new test")
        if "decision" in test:
            return {"status": "decided", "test_id": test["id"], **test["decision"]}
        if test["baseline_id"] != state["current_id"]:
            raise ContextError("The test baseline is no longer the accepted toolbox")
        if type(response.get("accept")) is not bool or not isinstance(response.get("reason"), str) or not response["reason"].strip():
            raise ContextError("Decide requires a boolean accept and a nonempty reason")
        if "accept_unresolved_history" in response and type(response["accept_unresolved_history"]) is not bool:
            raise ContextError("accept_unresolved_history must be a boolean")
        if "history_reason" in response and not isinstance(response["history_reason"], str):
            raise ContextError("history_reason must be text")
        regression = r._discovery_regression(
            test["baseline_results"], test["results"], self.focused, self.replay)
        acceptance = r._discovery_acceptance(response, regression)
        test["decision"] = {**acceptance, "agent_accept": response["accept"], "reason": response["reason"]}
        if acceptance["accepted"]:
            state["current"], state["current_id"] = test["draft_path"], draft_id
            state["current_results"] = test["results"]
            state["accepted"].append(state["drafts"][draft_id]["hypothesis"])
            state["bank"] = r._remember_successes(
                state["bank"], self.tasks, test["results"], Path(state["current"]),
                self.cycle, f"accepted_{test['id']}")
        return {"status": "decided", "test_id": test["id"], "draft_id": draft_id, **test["decision"]}

    def _operate(self, response, state, turn):
        if not isinstance(response, dict):
            raise ContextError("Return a JSON operation object")
        action = response.get("action")
        allowed = {"read": {"evidence_requests"}, "edit": {"files", "edits", "hypothesis"},
                   "test": set(), "decide": {"draft_id", "test_id", "accept", "reason",
                                            "accept_unresolved_history", "history_reason"},
                   "reset": set(), "finish": {"reason"}}
        if not isinstance(action, str) or action not in allowed:
            raise ContextError("Unknown action; use read, edit, test, decide, reset or finish")
        if set(response) - (allowed[action] | {"action", "notes"}):
            raise ContextError("Unexpected fields for action " + action)
        if "notes" in response:
            if not isinstance(response["notes"], str):
                raise ContextError("notes must be text")
            state["notes"] = response["notes"][:MAX_NOTES_CHARS]
        if action == "read":
            instruction = INSTRUCTIONS + "\n\n" + r.TOOLBOX_CONTRACT
            base, _ = self.context.coding_prompt(instruction, notes=state["notes"])
            budget = min(READ_CHARS, MAX_PROMPT_CHARS - len(base) - 3_000)
            replies, images = self.context.read(response.get("evidence_requests"), budget=budget)
            state["read_results"] = replies
            state["image_ids"] = images or None
            return {"status": "read", "reads": [{key: item[key] for key in
                    ("id", "offset", "end", "total_chars", "complete", "page_limited", "found", "attached")
                    if key in item} for item in replies]}
        state["read_results"], state["image_ids"] = [], None
        if action == "edit":
            return self._edit(response, state, turn)
        if action == "test":
            return self._test(state)
        if action == "decide":
            return self._decide(response, state)
        if action == "reset":
            state["draft_id"] = state["current_id"]
            return {"status": "reset", "draft_id": state["draft_id"]}
        if not isinstance(response.get("reason"), str) or not response["reason"].strip():
            raise ContextError("finish requires a nonempty reason")
        state.update(status="complete", end_reason="model_finished")
        return {"status": "finished", "reason": response["reason"]}

    def run(self):
        try:
            if self.state["status"] != "complete":
                self.state.update(status="running", end_reason=None)
            self._snapshot()
            while self.state["status"] != "complete" and self.state["turns_used"] < self.max_turns:
                turn = self.state["turns_used"]
                path = self.directory / "turns" / f"{turn:03d}.json"
                record = r._json(path)
                if record is None:
                    instruction = self._context()
                    prompt, images = self.context.coding_prompt(
                        instruction, notes=self.state["notes"], replies=self.state["read_results"],
                        image_ids=self.state["image_ids"])
                    record = {"turn": turn, "charged": False,
                              "request": {"role": "discover_code", "prompt": prompt, "images": images}}
                    r._save(path, record)
                elif self.context is None:
                    self.context = ResearchContext.load(self.directory / "evidence.context")
                self._request(record, path)
                next_state = copy.deepcopy(self.state)
                if "response_error" in record:
                    feedback = {"status": "invalid_response", **record["response_error"]}
                else:
                    try:
                        feedback = self._operate(record["response"], next_state, turn)
                    except ContextError as exc:
                        feedback = {"status": "operation_error", "reason": str(exc)}
                next_state.update(turns_used=turn + 1, last_feedback=feedback)
                if next_state["turns_used"] == self.max_turns and next_state["status"] != "complete":
                    next_state.update(status="complete", end_reason="interaction_limit")
                record.update(feedback=feedback, after=self._checkpoint_state(next_state))
                r._save(path, record)  # the atomic commit; reporting may be rebuilt from it
                self.state = next_state
                self._snapshot()
                r._progress(self.run_dir, "coding_operation_complete", cycle=self.cycle + 1,
                            session=str(self.directory),
                            turn=turn + 1, action=record.get("response", {}).get("action")
                            if isinstance(record.get("response"), dict) else None,
                            outcome=feedback["status"], **statistics(self.state))
        except BaseException as exc:
            committed = r._json(self.directory / "turns" / f"{self.state['turns_used']:03d}.json", {})
            if "after" in committed:
                self.state = self._restore_state(committed["after"], self.baseline_results)
            reason = ("model_budget_exhausted" if isinstance(exc, AgentBudgetExhausted) else
                      "network_failure" if isinstance(exc, AgentTransportError) else
                      "interrupted" if isinstance(exc, KeyboardInterrupt) else "infrastructure_error")
            if self.state["status"] != "complete":
                self.state.update(status="interrupted", end_reason=reason)
            self._snapshot()
            raise
        return self.state


def run_coding_cycle(runner, agent, directory, cfg, *, current, current_results, bank,
                     payload, tasks, focused, replay, cycle):
    session = CodingSession(runner, agent, directory, cfg,
        {"current": str(current), "current_results": current_results, "bank": bank},
        payload, tasks, focused, replay, cycle)
    state = session.run()
    attempts = []
    for test in state["tests"]:
        draft = state["drafts"][test["draft_id"]]
        decision = test.get("decision", {})
        attempts.append({"revision": len(attempts), "hypothesis": draft["hypothesis"],
                         "accepted": decision.get("accepted", False),
                         "reason": decision.get("reason", "No decision; retained accepted toolbox"),
                         "gate_reason": decision.get("gate_reason"),
                         "history_reason": decision.get("history_reason"),
                         "demonstrated_improvement": decision.get("demonstrated_improvement", False),
                         "changes": r._changes(Path(test["baseline_path"]), r._contents(Path(draft["path"]))),
                         "results": [r._source_evidence(item) for item in test["results"]],
                         "regression": test["regression"]})
    return (Path(state["current"]), state["current_results"], state["bank"],
            state["accepted"], attempts, statistics(state))
