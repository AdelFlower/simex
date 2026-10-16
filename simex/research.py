"""Checkpointed, budgeted SimEX research loops.

Only public rollout evidence is placed in agent prompts.  Task truth, schedules,
and the target ledger belong to the trusted harness, outside its agent sandbox.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import difflib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .tasks import TASK_IDS, make_task, source_task_from_spec


DEFAULTS = {
    "population": 2, "cycles": 3, "probes": 8, "focused": 4,
    "revisions": 2, "replay": 2, "adapt_trials": 5, "repairs": 2,
    "screen_tasks": 3, "align_attempts": 3, "eval_trials": 5,
    "selection_trials": 3, "seed": 20261015,
}
METHODS = ("direct_cap", "stage1_transfer", "simex")
CAPABILITY_FILES = ("toolbox.py", "skill.md")
PROVENANCE_KEYS = ("model", "codex_version", "reasoning_effort", "agent_timeout",
                   "policy_timeout", "max_sim_time", "isaac_asset", "source_policy_workers")
TOOLBOX_CONTRACT = (
    "Runtime contract: toolbox.py must define class Toolbox with __init__(self,robot). "
    "The policy receives tb=Toolbox(robot); preserve existing public methods because "
    "fixed scripts will reuse them. Keep the full RobotAPI reference in skill.md. "
    "Only numpy, scipy, math, collections, itertools, statistics imports are allowed; "
    "do not import typing, dataclasses, time, os, pathlib or simulator modules. "
    "No filesystem/network/process access, eval/exec, introspection, dunder-attribute "
    "access, or numpy/scipy file I/O. Single-underscore numerical helpers are allowed. "
    "Use public self.robot API methods only; do not call self.robot._call/_stream. "
    "Use robot.gripper(width) for grippers, never a method named open. "
    "All object positions must come from RGBD during execution; do not bake source "
    "task poses into the library. Retain measured uncertainties in skill.md."
)
TASK_SPEC_HELP = (
    "Each task is {name,description,base_task_id,seed,scene_overrides,predicate}. "
    f"base_task_id is one of {list(TASK_IDS)}. Use small reachable scene variations. "
    "Use scene_overrides only for plates and tote initially; keep plate names/colors "
    "green and red because those are the initial perception classes. Example: "
    '{"plates":[{"name":"green","position":[0.55,0.17,0.004],"radius":0.07,'
    '"mass":0.15,"friction":0.6},{"name":"red","position":[0.32,0.26,0.004]}],'
    '"tote":{"position":[0.39,-0.26,0],"yaw":0.0}}. Plate bounds: x [0.15,0.75], '
    "abs(y)<=0.45, z [0,0.25], radius [0.03,0.12], mass [0.02,1], friction [0.1,2]. "
    "A predicate is "
    '{"all":[{"kind":"lift","object":"green","min":0.05},'
    '{"kind":"inside","object":"green"}]}; kinds lift,inside,outside,'
    "displaced (min meters). Design relevant, nontrivial predicates; never "
    "weaken a predicate after seeing an outcome. Task descriptions must tell "
    "the policy what to achieve without object truth poses. If overriding plates, "
    "supply complete plate objects, including names and all geometry/appearance "
    "fields from the source scene template; keep targets consistent with the plates."
)


class ResearchError(RuntimeError):
    """An experiment needs attention before safe continuation."""


class InvalidToolbox(ResearchError):
    """A consumed candidate proposal failed its public code contract."""


def _progress(run_dir: Path, event: str, **details):
    record = {"time_unix": time.time(), "event": event, **details}
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    label = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[SimEX] {event} {label}", flush=True)


def _agent_capacity(agent):
    """Do not allocate a scarce target slot after the model budget is exhausted."""
    if agent is None:
        return
    settings = getattr(agent, "config", {})
    count = getattr(agent, "count", None)
    limit = settings.get("max_agent_calls", 800) if isinstance(settings, dict) else 800
    if isinstance(count, int) and count >= int(limit):
        raise ResearchError("Model-call budget exhausted; no further rollout has been scheduled")


def _provenance(config: dict) -> dict:
    package = Path(__file__).resolve().parent
    code = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))}
    private = package.parent / "simex_private"
    if private.exists():
        code.update({"simex_private/" + str(path.relative_to(private)):
                     hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(private.rglob("*.py"))})
    dependencies = {}
    for dependency in ("mujoco", "numpy", "scipy"):
        try:
            dependencies[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            dependencies[dependency] = None
    return {"settings": {key: config.get(key) for key in PROVENANCE_KEYS},
            "code_sha256": _digest(code), "dependencies": dependencies}


def _json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _save(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def toolbox_hash(path: Path) -> str:
    return _digest({name: (Path(path) / name).read_text() for name in CAPABILITY_FILES})


def _contents(path: Path) -> dict:
    return {name: (Path(path) / name).read_text() for name in CAPABILITY_FILES}


def _snapshot(source: Path, destination: Path) -> Path:
    return _write_toolbox(destination, _contents(source))


def _write_toolbox(destination: Path, contents: dict) -> Path:
    for name in CAPABILITY_FILES:
        if not isinstance(contents.get(name), str):
            raise InvalidToolbox(f"Agent must return complete {name} text")
    try:
        from .policy_process import validate
        tree = validate(contents["toolbox.py"])
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Toolbox"]
        if not classes or not any(isinstance(node, ast.FunctionDef) and node.name == "__init__"
                                  for node in classes[0].body):
            raise ValueError("toolbox.py must define Toolbox.__init__(self, robot)")
    except (SyntaxError, ValueError) as exc:
        raise InvalidToolbox(f"Invalid proposed toolbox: {exc}") from exc
    destination.mkdir(parents=True, exist_ok=True)
    for name in CAPABILITY_FILES:
        target = destination / name
        if target.exists() and target.read_text() != contents[name]:
            raise ResearchError(f"Immutable experiment artifact changed: {target}")
        if not target.exists():
            target.write_text(contents[name])
    return destination


def _freeze(source: Path, destination: Path) -> Path:
    _snapshot(source, destination)
    record = {"sha256": toolbox_hash(destination)}
    previous = _json(destination / "frozen.json")
    if previous is not None and previous != record:
        raise ResearchError(f"Frozen toolbox modified: {destination}")
    _save(destination / "frozen.json", record)
    return destination


def _check_frozen(path: Path):
    record = _json(path / "frozen.json")
    if not record or toolbox_hash(path) != record["sha256"]:
        raise ResearchError(f"Frozen toolbox integrity check failed: {path}")


@contextlib.contextmanager
def _lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".research.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResearchError("Another research command is using this run") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _initialize(run_dir: Path, config: dict) -> dict:
    cfg = {**DEFAULTS, **config}
    for key in DEFAULTS:
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool):
            raise ResearchError(f"{key} must be an integer")
        minimum = 0 if key in ("replay", "revisions", "align_attempts") else 1
        if cfg[key] < minimum:
            raise ResearchError(f"{key} must be >= {minimum}")
    if cfg["adapt_trials"] > 5 or cfg["align_attempts"] > 3:
        raise ResearchError("The protocol caps target adaptation at 5 trials and alignment at 3 edits")
    manifest_path = run_dir / "manifest.json"
    manifest = _json(manifest_path)
    if manifest is not None:
        if any(manifest["config"][key] != cfg[key] for key in DEFAULTS):
            raise ResearchError("Research budgets/seeds cannot change when resuming a run")
        if manifest.get("provenance") != _provenance(cfg):
            raise ResearchError("Model, runtime settings, dependencies or implementation changed; "
                                "create a new run to preserve experiment provenance")
        _check_frozen(run_dir / "frozen" / "initial")
        ledger = _json(run_dir / "target_ledger.json")
        schedule = {key: {name: value for name, value in slot.items()
                         if name in ("phase", "method", "task_id", "trial", "seed")}
                    for key, slot in ledger["slots"].items()}
        if _digest(schedule) != manifest["target_schedule_sha256"]:
            raise ResearchError("Frozen target schedule was modified")
        return manifest["config"]
    initial = Path(__file__).parent / "initial"
    _freeze(initial, run_dir / "frozen" / "initial")
    selection = [make_task(task_id, cfg["seed"] + 10000 + i * 100 + j)
                 for i, task_id in enumerate(TASK_IDS)
                 for j in range(cfg["selection_trials"])]
    _save(run_dir / "selection_tasks.json", selection)
    slots = {}
    for i, task_id in enumerate(TASK_IDS):
        for trial in range(cfg["adapt_trials"]):
            key = f"adapt/{task_id}/{trial:02d}"
            slots[key] = {"phase": "adapt", "task_id": task_id, "trial": trial,
                          "seed": cfg["seed"] + 20000 + i * 1000 + trial,
                          "state": "pending"}
        for method in METHODS:
            for trial in range(cfg["eval_trials"]):
                key = f"evaluate/{method}/{task_id}/{trial:02d}"
                slots[key] = {"phase": "evaluate", "method": method,
                              "task_id": task_id, "trial": trial,
                              "seed": cfg["seed"] + 30000 + i * 1000 + trial,
                              "state": "pending"}
    _save(run_dir / "target_ledger.json", {"slots": slots})
    _save(manifest_path, {"format_version": 2, "config": cfg, "provenance": _provenance(cfg),
                          "selection_sha256": _digest(selection),
                          "target_schedule_sha256": _digest({k: {a: b for a, b in v.items()
                              if a != "state"} for k, v in slots.items()}),
                          "initial_sha256": toolbox_hash(run_dir / "frozen" / "initial")})
    return cfg


def _images(value: Any) -> list[str]:
    """Attach evidence for every rollout; one labelled sheet per rollout when available."""
    found = []
    def visit(item):
        if isinstance(item, dict):
            image_keys = ("phase_images",) if item.get("phase_images") else ("keyframes", "read_images")
            for key, child in item.items():
                if key in image_keys and isinstance(child, list):
                    for path in child:
                        if isinstance(path, str) and path not in found and Path(path).is_file():
                            found.append(path)
                elif key not in ("phase_images", "keyframes", "read_images") and isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return found


def _ask(agent, path: Path, role: str, instruction: str, payload: dict) -> dict:
    if role in ("discover_revision", "repair"):
        instruction += "\n\n" + TOOLBOX_CONTRACT
    image_paths = _images(payload)
    if image_paths:
        instruction += "\n\nAttached images, in order (each contact sheet labels its rollout and phases):\n" + \
            json.dumps([{"image": i + 1, "path": path} for i, path in enumerate(image_paths)])
    prompt = instruction + "\n\nINPUT JSON:\n" + json.dumps(payload, ensure_ascii=False, default=str)
    request = {"role": role, "prompt": prompt, "images": image_paths}
    previous = _json(path)
    if previous is not None:
        if previous["request_sha256"] != _digest(request):
            raise ResearchError(f"Checkpoint input changed: {path}")
        return previous["response"]
    _agent_capacity(agent)
    try:
        response = agent.ask(role=role, prompt=prompt, images=image_paths)
    except Exception as exc:
        _save(path.with_suffix(".error.json"), {"role": role, "error": str(exc),
              "exception": type(exc).__name__, "agent_count": getattr(agent, "count", None)})
        raise
    if not isinstance(response, dict):
        raise ResearchError(f"Agent role {role} returned a non-object")
    _save(path, {"request_sha256": _digest(request), "request": request,
                 "response": response, "agent_count": getattr(agent, "count", None)})
    return response


def _public(result: dict) -> dict:
    evidence = result.get("public_evidence", {})
    if isinstance(evidence, str) and Path(evidence).is_file():
        evidence = _json(Path(evidence), {})
    if not isinstance(evidence, dict):
        evidence = {}
    # Infrastructure diagnostics stay with the operator; backend traceback text
    # can name hidden target assets, files or implementation details.
    if result.get("status") in ("interrupted", "infrastructure_error"):
        evidence = {**evidence, "reason": "Execution interrupted by infrastructure failure.", "stdout": ""}
    if evidence.get("recording_errors"):
        evidence = {**evidence, "recording_errors": ["Public evidence capture failed."] * len(evidence["recording_errors"])}
    from .evidence import summarize_public_evidence
    evidence = summarize_public_evidence(evidence, result.get("output_dir"))
    allowed = ("status", "success", "reason", "trace_tail", "robot_trace_tail",
               "read_images", "keyframes", "stdout", "initial_visual_estimates",
               "phase_images", "visual_phases", "motion_summary", "actions",
               "video", "recording_errors", "evidence_errors")
    selected = {key: evidence[key] for key in allowed if key in evidence}
    if selected.get("phase_images"):
        # The sheet embeds chosen frames; unused path inventories add no model
        # evidence and can make multi-candidate prompts unnecessarily large.
        selected.pop("read_images", None)
        selected.pop("keyframes", None)
    if selected.get("motion_summary", {}).get("samples"):
        selected.pop("trace_tail", None)
        selected.pop("robot_trace_tail", None)
    return {"status": result.get("status"), "success": bool(result.get("success")),
            "steps": result.get("steps", 0), "evidence": selected}


def _candidate(proposal: dict, directory: Path) -> tuple[Path | None, dict]:
    """Invalid code spends this proposal slot; it never triggers resampling on resume."""
    try:
        toolbox = _write_toolbox(directory / "toolbox", proposal)
    except InvalidToolbox as exc:
        outcome = {"status": "invalid_candidate", "reason": str(exc)}
        _save(directory / "validation.json", outcome)
        return None, outcome
    outcome = {"status": "valid", "toolbox_sha256": toolbox_hash(toolbox)}
    _save(directory / "validation.json", outcome)
    return toolbox, outcome


def _changes(current: Path, proposal: dict) -> dict:
    return {name: "".join(difflib.unified_diff(
        (current / name).read_text().splitlines(keepends=True),
        proposal.get(name, "").splitlines(keepends=True),
        fromfile="current/" + name, tofile="candidate/" + name))
        for name in CAPABILITY_FILES if isinstance(proposal.get(name), str)}


def _outcome_summary(public_result: dict | None) -> dict | None:
    """History retains outcomes and diagnostics without reattaching all old images."""
    if public_result is None:
        return None
    evidence = public_result.get("evidence", {})
    return {"status": public_result.get("status"), "success": public_result.get("success"),
            "steps": public_result.get("steps"), "reason": evidence.get("reason", ""),
            "stdout": evidence.get("stdout", "")[-2000:],
            "motion_summary": evidence.get("motion_summary")}


def _run(runner, engine: str, task: dict, toolbox: Path, out: Path,
         script: str | None = None, source_config: dict | None = None) -> dict:
    request = {"engine": engine, "task": task, "toolbox_sha256": toolbox_hash(toolbox),
               "script_sha256": _digest(script), "source_config": source_config or {}}
    request_path = out / "research_request.json"
    old = _json(request_path)
    if old is not None and old != request:
        raise ResearchError(f"Rollout checkpoint input changed: {out}")
    _save(request_path, request)
    previous = _json(out / "result.json")
    if previous is not None and previous.get("status") != "infrastructure_error":
        if previous.get("failure_kind") == "infrastructure":
            raise ResearchError(f"Infrastructure interrupted this rollout; recover its outcome before continuing: {out}")
        return previous
    if script is None and not (out / "policy_generation.json").exists():
        _agent_capacity(getattr(runner, "agent", None))
    result = runner.rollout(engine=engine, task=task, toolbox=toolbox, out=out,
                            script=script, source_config=source_config)
    if not isinstance(result, dict):
        raise ResearchError("Runner did not return a result object")
    _save(out / "result.json", result)
    attempts = out / "research_attempts"
    attempt = len(list(attempts.glob("*.json")))
    _save(attempts / f"{attempt:03d}.json", result)
    if result.get("status") == "infrastructure_error" or result.get("failure_kind") == "infrastructure":
        raise ResearchError(f"Infrastructure failure; resume this run after repair: {out}")
    return result


def _target(runner, run_dir: Path, slot_key: str, toolbox: Path, out: Path) -> dict:
    ledger_path = run_dir / "target_ledger.json"
    ledger = _json(ledger_path)
    slot = ledger["slots"][slot_key]
    result_path = out / "result.json"
    existing = _json(result_path)
    if slot["state"] == "complete":
        if existing is None:
            raise ResearchError(f"Completed target result missing: {out}")
        if toolbox_hash(toolbox) != slot["toolbox_sha256"]:
            raise ResearchError("Target slot cannot be reused with a changed toolbox")
        if _digest(existing) != slot["result_sha256"]:
            raise ResearchError("Completed target result was modified")
        return existing
    if slot["state"] in ("running", "unresolved", "unresolved_infrastructure") and existing is None:
        slot["state"] = "unresolved"
        _save(ledger_path, ledger)
        raise ResearchError(
            f"Target slot {slot_key} has no durable outcome; refusing automatic rerun. "
            "Recover the original result.json or record an interrupted result with "
            "actual_started=true before resuming.")
    if existing is not None and existing.get("status") != "infrastructure_error":
        if slot.get("toolbox_sha256") not in (None, toolbox_hash(toolbox)):
            raise ResearchError("Interrupted target slot has a different toolbox")
        result = existing
    else:
        _agent_capacity(getattr(runner, "agent", None))
        slot.update(state="running", output_dir=str(out.resolve()),
                    toolbox_sha256=toolbox_hash(toolbox))
        _save(ledger_path, ledger)
        try:
            result = _run(runner, "isaac", make_task(slot["task_id"], slot["seed"]),
                          toolbox, out)
        except BaseException:
            existing = _json(result_path)
            if existing is not None and existing.get("failure_kind") == "infrastructure" and existing.get("actual_started"):
                slot.update(state="unresolved_infrastructure", status=existing.get("status"), actual_started=True,
                            result_sha256=_digest(existing))
            else:
                slot["state"] = ("pending_infrastructure" if existing is not None and
                                 existing.get("status") == "infrastructure_error" else "unresolved")
            _save(ledger_path, ledger)
            raise
    if result.get("failure_kind") == "infrastructure" and result.get("actual_started"):
        slot.update(state="unresolved_infrastructure", status=result.get("status"),
                    actual_started=True, toolbox_sha256=toolbox_hash(toolbox),
                    result_sha256=_digest(result))
        _save(ledger_path, ledger)
        raise ResearchError(
            f"Target slot {slot_key} was interrupted by infrastructure after execution started; "
            "it remains incomplete and cannot be rerun automatically. Preserve the evidence "
            "and recover the original outcome or explicitly resolve this interrupted slot.")
    slot.update(state="complete", success=bool(result.get("success")),
                status=result.get("status"), steps=result.get("steps", 0),
                actual_started=result.get("actual_started"),
                toolbox_sha256=toolbox_hash(toolbox), result_sha256=_digest(result))
    _save(ledger_path, ledger)
    return result


def _tasks_from_response(response: dict, count: int) -> list[dict]:
    specs = response.get("tasks")
    if not isinstance(specs, list) or len(specs) != count:
        raise ResearchError(f"Agent must return exactly {count} tasks")
    tasks = []
    for index, spec in enumerate(specs):
        try:
            if not isinstance(spec, dict):
                raise ValueError("task must be an object")
            tasks.append(source_task_from_spec(spec))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ResearchError(f"Invalid generated source task {index}: {exc}") from exc
    return tasks


def _suite(runner, tasks: list[dict], toolbox: Path, out: Path,
           source_config: dict | None = None, script: str | None = None) -> list[dict]:
    def progress(event, **details):
        root = getattr(runner, "run_dir", None)
        if root is not None:
            _progress(Path(root), event, suite=str(out), **details)
        else:
            print(f"[SimEX] {event} suite={out} {details}", flush=True)

    prepare = getattr(runner, "prepare_policy", None)
    if script is None and callable(prepare):
        # Only independent source policy generation runs concurrently; physics
        # remains serial. Runner signs and reuses each exact generation outcome.
        settings = getattr(runner, "config", {})
        workers = max(1, min(3, int(settings.get("source_policy_workers", 3))))
        pending = [(i, task) for i, task in enumerate(tasks)
                   if not (out / f"{i:03d}" / "result.json").exists()]
        if pending:
            progress("source_policy_preparation_started", count=len(pending), workers=workers)
            errors = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(prepare, task, toolbox, out / f"{i:03d}"): i
                           for i, task in pending}
                for future in as_completed(futures):
                    try:
                        prepared = future.result()
                        progress("source_policy_prepared", trial=futures[future] + 1,
                                 status=prepared.get("status"))
                    except Exception as exc:
                        errors.append(exc)
                        progress("source_policy_preparation_error", trial=futures[future] + 1,
                                 error=str(exc))
            if errors:
                raise ResearchError(f"Source policy preparation failed: {errors[0]}") from errors[0]
    results = []
    for index, task in enumerate(tasks):
        result = _run(runner, "mujoco", task, toolbox, out / f"{index:03d}", script, source_config)
        results.append(result)
        progress("source_rollout_complete", trial=index + 1, scheduled=len(tasks),
                 status=result.get("status"), success=bool(result.get("success")))
    return results


def _score(results: list[dict]) -> tuple[int, int]:
    return sum(bool(result.get("success")) for result in results), sum(
        int(result.get("steps") or 0) for result in results)


def discover(runner, agent, run_dir: Path, config: dict) -> dict:
    """Run independent source curricula and select without target access."""
    run_dir = Path(run_dir).resolve()
    with _lock(run_dir):
        cfg = _initialize(run_dir, config)
        done = _json(run_dir / "discovery.json")
        if done:
            _check_frozen(run_dir / "frozen" / "stage1")
            return done
        _progress(run_dir, "discovery_started", population=cfg["population"], cycles=cfg["cycles"])
        members = []
        for index in range(cfg["population"]):
            root = run_dir / "discovery" / f"member_{index:02d}"
            current = _snapshot(run_dir / "frozen" / "initial", root / "initial")
            bank: list[dict] = []
            summaries = []
            for cycle in range(cfg["cycles"]):
                _progress(run_dir, "source_cycle_started", member=index, cycle=cycle + 1)
                cycle_dir = root / f"cycle_{cycle:02d}"
                # Freeze exactly the available peer summaries at this boundary.
                peer_path = cycle_dir / "peer_summaries.json"
                peers = _json(peer_path)
                if peers is None:
                    peers = [{"member": prior["member"], "summaries": prior["summaries"]}
                             for prior in members]
                    _save(peer_path, peers)
                probes = _tasks_from_response(_ask(agent, cycle_dir / "probe_proposal.json",
                    "discover_probe", f"Generate exactly {cfg['probes']} broad source-only tasks "
                    "to expose missing reusable robot capabilities. Return {tasks:[...]}. " + TASK_SPEC_HELP,
                    {"member": index, "cycle": cycle, "toolbox": _contents(current),
                     "source_scene_template": make_task(TASK_IDS[0], 0)["scene"],
                     "own_history": summaries, "peer_experiment_summaries": peers}), cfg["probes"])
                _save(cycle_dir / "probe_tasks.json", probes)
                probe_results = _suite(runner, probes, current, cycle_dir / "probes")
                _progress(run_dir, "source_probe_complete", member=index, cycle=cycle + 1,
                          successes=_score(probe_results)[0], scheduled=len(probes))
                focus = _ask(agent, cycle_dir / "focus_proposal.json", "discover_focus",
                    f"Identify one systematic weakness and return {{weakness:str,tasks:[...]}} "
                    f"with exactly {cfg['focused']} focused source tasks. " + TASK_SPEC_HELP,
                    {"toolbox": _contents(current), "probe_tasks": probes,
                     "probe_results": [_public(result) for result in probe_results]})
                focused = _tasks_from_response(focus, cfg["focused"])
                replay = bank[-cfg["replay"]:] if cfg["replay"] else []
                _save(cycle_dir / "focused_tasks.json", focused)
                _save(cycle_dir / "replay_tasks.json", replay)
                current_results = _suite(runner, focused + replay, current, cycle_dir / "before")
                accepted = []
                revision_history = []
                for revision in range(cfg["revisions"]):
                    rev_dir = cycle_dir / f"revision_{revision:02d}"
                    proposal = _ask(agent, rev_dir / "proposal.json", "discover_revision",
                        "Propose reusable improvements for the focused weakness. Return complete "
                        '{"toolbox.py":str,"skill.md":str,"hypothesis":str}. Only these two files '
                        "persist as robot capabilities. Preserve API compatibility. Do not read "
                        "simulator truth or implement direct simulator control. Use prior_attempts "
                        "to learn from rejected code, outcomes and rejection reasons.",
                        {"weakness": focus.get("weakness"), "toolbox": _contents(current),
                         "tasks": focused + replay, "results": [_public(r) for r in current_results],
                         "prior_attempts": revision_history})
                    candidate, validation = _candidate(proposal, rev_dir)
                    attempt_record = {"revision": revision, "hypothesis": proposal.get("hypothesis"),
                                      "changes": _changes(current, proposal),
                                      "candidate_toolbox": {name: proposal.get(name) for name in CAPABILITY_FILES},
                                      "validation": validation}
                    if candidate is None:
                        attempt_record.update(accepted=False, reason=validation["reason"], results=[])
                        revision_history.append(attempt_record)
                        _save(rev_dir / "accepted.json", attempt_record)
                        _progress(run_dir, "source_revision_complete", member=index, cycle=cycle + 1,
                                  revision=revision + 1, accepted=False, status="invalid_candidate")
                        continue
                    candidate_results = _suite(runner, focused + replay, candidate, rev_dir / "tests")
                    no_regression = (_score(candidate_results[:len(focused)])[0] >=
                                     _score(current_results[:len(focused)])[0] and
                                     all(not old.get("success") or new.get("success")
                                         for old, new in zip(current_results[len(focused):],
                                                             candidate_results[len(focused):])))
                    decision = _ask(agent, rev_dir / "decision.json", "discover_select",
                        'Review source evidence and return {"accept":bool,"reason":str}. '
                        "Accept only a plausible reusable improvement with no focused-score "
                        "decrease and no failure of a previously passing capability replay.",
                        {"hypothesis": proposal.get("hypothesis"), "no_regression": no_regression,
                         "current": [_public(r) for r in current_results],
                         "candidate": [_public(r) for r in candidate_results]})
                    take = decision.get("accept") is True and no_regression
                    attempt_record.update(accepted=take, agent_accept=decision.get("accept"),
                                          no_regression=no_regression, reason=decision.get("reason"),
                                          baseline_results=[_public(r) for r in current_results],
                                          results=[_public(r) for r in candidate_results])
                    revision_history.append(attempt_record)
                    _save(rev_dir / "accepted.json", attempt_record)
                    if take:
                        current, current_results = candidate, candidate_results
                        accepted.append(proposal.get("hypothesis", ""))
                    _progress(run_dir, "source_revision_complete", member=index, cycle=cycle + 1,
                              revision=revision + 1, accepted=take,
                              focused_successes=_score(current_results[:len(focused)])[0])
                for task, result in zip(probes + focused + replay, probe_results + current_results):
                    if result.get("success") and _digest(task) not in {_digest(t) for t in bank}:
                        bank.append(task)
                summary = {"cycle": cycle, "weakness": focus.get("weakness"),
                           "probe_successes": _score(probe_results)[0],
                           "focused_successes": _score(current_results[:len(focused)])[0],
                           "accepted_hypotheses": accepted, "capability_bank_size": len(bank)}
                summaries.append(summary)
                _save(cycle_dir / "summary.json", summary)
                _save(root / "capability_bank.json", bank)
                _snapshot(current, cycle_dir / "final_toolbox")
            frozen = _freeze(current, root / "frozen")
            members.append({"member": index, "toolbox": str(frozen), "summaries": summaries})
        selection = _json(run_dir / "selection_tasks.json")
        manifest = _json(run_dir / "manifest.json")
        if _digest(selection) != manifest["selection_sha256"]:
            raise ResearchError("Frozen source selection tasks were modified")
        for member in members:
            _progress(run_dir, "source_selection_started", member=member["member"], scenes=len(selection))
            scores = _suite(runner, selection, Path(member["toolbox"]),
                            run_dir / "selection" / f"member_{member['member']:02d}")
            member["successes"], member["steps"] = _score(scores)
            valid = [result for result in scores if result.get("status") in ("success", "failed")
                     and result.get("actual_started") is True]
            member["valid_executions"] = len(valid)
            member["mean_valid_steps"] = sum(int(result.get("steps") or 0) for result in valid) / len(valid) \
                if valid else None
        eligible = [member for member in members if member["valid_executions"] > 0]
        if not eligible:
            raise ResearchError("No source selection trial completed a valid physics policy; "
                                "discovery is not complete and target adaptation must not start")
        winner = min(eligible, key=lambda item: (-item["successes"], -item["valid_executions"],
                                               item["mean_valid_steps"], item["member"]))
        destination = _freeze(Path(winner["toolbox"]), run_dir / "frozen" / "stage1")
        result = {"winner": winner["member"], "toolbox": str(destination), "members": members,
                  "selection_tasks": len(selection), "agent_count": getattr(agent, "count", None)}
        _save(run_dir / "discovery.json", result)
        _progress(run_dir, "discovery_complete", winner=winner["member"], successes=winner["successes"])
        return result


def _calibration(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ResearchError("source_config must be an object")
    bounds = {"mass_scale": (0.5, 2.0), "friction_scale": (0.3, 2.0),
              "controller_duration_scale": (0.5, 2.0)}
    if set(value) - (set(bounds) | {"camera_offset"}):
        raise ResearchError("Unsupported source calibration parameter")
    for key, (lo, hi) in bounds.items():
        if key in value and (not isinstance(value[key], (int, float)) or
                             isinstance(value[key], bool) or not lo <= value[key] <= hi):
            raise ResearchError(f"{key} must be in [{lo}, {hi}]")
    if "camera_offset" in value:
        offset = value["camera_offset"]
        if not isinstance(offset, list) or len(offset) != 3 or any(
            not isinstance(v, (float, int)) or isinstance(v, bool) or not -0.05 <= v <= 0.05
            for v in offset
        ):
            raise ResearchError("camera_offset must have three components within +/-0.05 m")
    return copy.deepcopy(value)


def _executed_script(result: dict) -> str | None:
    if result.get("actual_started") is not True or result.get("status") in (
        "invalid_policy", "generation_error", "infrastructure_error"
    ):
        return None
    path = result.get("script_path")
    if not path:
        return None
    if not Path(path).is_file():
        raise ResearchError(f"Executed policy artifact missing: {path}")
    return Path(path).read_text()


def _source_reconstruction(task_id: str, public_evidence: dict) -> dict:
    from .perception import source_task_from_observation
    return source_task_from_observation(task_id, public_evidence)


def adapt(runner, agent, run_dir: Path, config: dict) -> dict:
    """Adapt each task independently; target slots are never a repair score search."""
    run_dir = Path(run_dir).resolve()
    with _lock(run_dir):
        cfg = _initialize(run_dir, config)
        stage1 = run_dir / "frozen" / "stage1"
        _check_frozen(stage1)
        done = _json(run_dir / "adaptation.json")
        if done:
            for task_id in TASK_IDS:
                _check_frozen(run_dir / "frozen" / "adapted" / task_id)
            return done
        _progress(run_dir, "adaptation_started", tasks=len(TASK_IDS), trials_per_task=cfg["adapt_trials"])
        outcomes = {}
        for task_index, task_id in enumerate(TASK_IDS):
            root = run_dir / "adaptation" / task_id
            current = _snapshot(stage1, root / "initial")
            source_config = {}
            trial_summaries = []
            for trial in range(cfg["adapt_trials"]):
                _progress(run_dir, "adaptation_trial_started", task=task_id, trial=trial + 1)
                trial_dir = root / f"trial_{trial:02d}"
                target = _target(runner, run_dir, f"adapt/{task_id}/{trial:02d}",
                                 current, trial_dir / "target")
                fixed_script = _executed_script(target)
                # Reconstruct source geometry from initial public RGBD estimates;
                # never reuse the target's hidden seed or initial state.
                reconstruction_path = trial_dir / "source_reconstruction.json"
                nominal_task = _json(reconstruction_path)
                if nominal_task is None:
                    nominal_task = _source_reconstruction(task_id, _public(target)["evidence"])
                    _save(reconstruction_path, nominal_task)
                source = None
                if fixed_script is not None:
                    source = _run(runner, "mujoco", nominal_task, current,
                                  trial_dir / "alignment" / "initial", fixed_script, source_config)
                    for attempt in range(cfg["align_attempts"]):
                        align_dir = trial_dir / "alignment" / f"attempt_{attempt:02d}"
                        decision = _ask(agent, align_dir / "decision.json", "align",
                            "Compare target and fixed-policy source evidence. Return "
                            '{"aligned":bool,"source_config":{...},"reason":str}. '
                            "If aligned=true keep the model. Otherwise propose source calibration only: "
                            "mass_scale [0.5,2], friction_scale [0.3,2], controller_duration_scale "
                            "[0.5,2], camera_offset [dx,dy,dz] each +/-0.05 m. friction_scale uniformly "
                            "scales source sliding friction for the whole scene (including robot contact "
                            "geometries), so it changes the effective contact friction. Missing keys preserve "
                            "the current values. Never change this script or toolbox during alignment. "
                            "Use image and motion evidence; alignment is not success optimization.",
                            {"task": nominal_task["description"], "script": fixed_script,
                             "source_config": source_config, "target": _public(target),
                             "source": _public(source), "task_history": trial_summaries})
                        if decision.get("aligned") is True:
                            break
                        update = _calibration(decision.get("source_config", {}))
                        source_config = _calibration({**source_config, **update})
                        source = _run(runner, "mujoco", nominal_task, current,
                                      align_dir / "replay", fixed_script, source_config)
                else:
                    _save(trial_dir / "alignment" / "unavailable.json", {
                        "reason": "No policy was produced/executed; no fixed-script replay is possible."})
                _save(trial_dir / "source_config.json", source_config)
                hypotheses = _ask(agent, trial_dir / "hypotheses.json", "repair_hypotheses",
                    f"Propose exactly {cfg['repairs']} distinct reusable toolbox repair hypotheses "
                    'based on target and aligned source evidence. Return {"hypotheses":[str,...]}. '
                    "Keep alternatives independent; do not modify the simulator further. "
                    "Use task_history to avoid repeating rejected repairs without new evidence.",
                    {"task": nominal_task["description"], "toolbox": _contents(current),
                     "script": fixed_script, "target": _public(target),
                     "source": _public(source) if source else None, "source_config": source_config,
                     "task_history": trial_summaries})
                ideas = hypotheses.get("hypotheses")
                if (not isinstance(ideas, list) or len(ideas) != cfg["repairs"] or
                    any(not isinstance(idea, str) or not idea.strip() for idea in ideas) or
                    len(set(ideas)) != len(ideas)):
                    raise ResearchError("Repair hypotheses must be distinct nonempty strings")
                candidates = {"current": current}
                candidate_records = {}
                hypothesis_map = {}
                for candidate_index, hypothesis in enumerate(ideas):
                    candidate_id = f"candidate_{candidate_index:02d}"
                    candidate_dir = trial_dir / candidate_id
                    hypothesis_map[candidate_id] = hypothesis
                    proposal = _ask(agent, candidate_dir / "proposal.json", "repair",
                        'Implement only this hypothesis; return complete {"toolbox.py":str,"skill.md":str}. '
                        "Preserve API compatibility with the executed policy. Use only RobotAPI. "
                        "Document measured limits and uncertainty. No target simulator access.",
                        {"hypothesis": hypothesis, "toolbox": _contents(current),
                         "task": nominal_task["description"], "script": fixed_script,
                         "target": _public(target), "source": _public(source) if source else None,
                         "task_history": trial_summaries})
                    candidate, validation = _candidate(proposal, candidate_dir)
                    candidate_records[candidate_id] = {"hypothesis": hypothesis,
                        "changes": _changes(current, proposal), "validation": validation}
                    if candidate is not None:
                        candidates[candidate_id] = candidate
                screen_proposal = _ask(agent, trial_dir / "screen_proposal.json", "screen_tasks",
                    f"Generate exactly {cfg['screen_tasks']} source generalization scenarios "
                    "for this same task, varying placements and plausible physical conditions. "
                    "Return {tasks:[...]}. Preserve the supplied success predicate exactly; "
                    "do not weaken the task or invent target simulator parameters. These scenarios "
                    "will be frozen and shared by current and all repair candidates. " + TASK_SPEC_HELP,
                    {"task_id": task_id, "description": nominal_task["description"],
                     "source_scene_template": nominal_task["scene"],
                     "required_predicate": nominal_task.get("predicate"),
                     "suggested_source_seeds": [cfg["seed"] + 40000 + task_index * 1000 +
                         trial * 100 + j for j in range(cfg["screen_tasks"])]})
                generalization = _tasks_from_response(screen_proposal, cfg["screen_tasks"])
                if nominal_task.get("predicate") is not None and any(
                    task.get("predicate") != nominal_task["predicate"] for task in generalization
                ):
                    raise ResearchError("Screening scenarios changed the required task predicate")
                _save(trial_dir / "screen_tasks.json", generalization)
                evidence = {}
                for candidate_id, candidate in candidates.items():
                    screen_dir = trial_dir / "screen" / candidate_id
                    replay = (_run(runner, "mujoco", nominal_task, candidate, screen_dir / "replay",
                                   fixed_script, source_config) if fixed_script is not None else None)
                    results = _suite(runner, generalization, candidate, screen_dir / "generalization",
                                     source_config)
                    evidence[candidate_id] = {"fixed_replay": _public(replay) if replay else None,
                                              "generalization": [_public(r) for r in results]}
                selected = _ask(agent, trial_dir / "repair_selection.json", "repair_select",
                    'Choose a candidate or retain current using videos/images, traces and scores. '
                    'Return {"selected":"current"|"candidate_00"|...,"reason":str}. '
                    "Simulation scores are supporting evidence, not an automatic objective. Consider "
                    "target mismatch and robustness; you may reject a high-scoring implausible repair.",
                    {"hypotheses": hypothesis_map, "candidates": candidate_records,
                     "available_choices": list(candidates), "task_history": trial_summaries,
                     "target": _public(target), "source_config": source_config, "screen": evidence})
                choice = selected.get("selected")
                selection_error = None
                if choice not in candidates:
                    selection_error = f"Unavailable or invalid selected repair: {choice!r}; retained current."
                    _save(trial_dir / "selection_validation.json", {
                        "status": "invalid_selection", "reason": selection_error,
                        "requested": choice, "selected": "current"})
                    choice = "current"
                current = candidates[choice]
                _snapshot(current, trial_dir / "selected_toolbox")
                summary = {"trial": trial, "target_success": bool(target.get("success")),
                           "target_status": target.get("status"), "selected": choice,
                           "reason": selection_error or selected.get("reason"),
                           "agent_reason": selected.get("reason"), "source_config": source_config,
                           "fixed_script_sha256": _digest(fixed_script),
                           "toolbox_sha256": toolbox_hash(current),
                           "target": _outcome_summary(_public(target)),
                           "candidates": candidate_records,
                           "screen": {name: {
                               "fixed_replay": _outcome_summary(items["fixed_replay"]),
                               "generalization": [_outcome_summary(r) for r in items["generalization"]]}
                               for name, items in evidence.items()}}
                trial_summaries.append(summary)
                _save(trial_dir / "summary.json", summary)
                _progress(run_dir, "adaptation_trial_complete", task=task_id, trial=trial + 1,
                          target_status=target.get("status"), selected=choice)
            frozen = _freeze(current, run_dir / "frozen" / "adapted" / task_id)
            outcomes[task_id] = {"toolbox": str(frozen), "source_config": source_config,
                                 "trials": trial_summaries,
                                 "initial_stage1_sha256": toolbox_hash(stage1)}
        result = {"tasks": outcomes, "agent_count": getattr(agent, "count", None)}
        _save(run_dir / "adaptation.json", result)
        _progress(run_dir, "adaptation_complete", tasks=len(outcomes))
        return result


def evaluate(runner, agent, run_dir: Path, config: dict) -> dict:
    """Fresh policy scripts, frozen capabilities, matched scene seeds, no feedback."""
    run_dir = Path(run_dir).resolve()
    with _lock(run_dir):
        cfg = _initialize(run_dir, config)
        done = _json(run_dir / "evaluation.json")
        toolboxes = {task_id: {"direct_cap": run_dir / "frozen" / "initial",
                              "stage1_transfer": run_dir / "frozen" / "stage1",
                              "simex": run_dir / "frozen" / "adapted" / task_id}
                     for task_id in TASK_IDS}
        for methods in toolboxes.values():
            for path in methods.values():
                _check_frozen(path)
        if done:
            return done
        _progress(run_dir, "evaluation_started", scheduled=len(TASK_IDS) * len(METHODS) * cfg["eval_trials"])
        records = []
        for task_id in TASK_IDS:
            for trial in range(cfg["eval_trials"]):
                for method in METHODS:
                    toolbox = toolboxes[task_id][method]
                    _check_frozen(toolbox)
                    output = run_dir / "evaluation" / method / task_id / f"trial_{trial:02d}"
                    result = _target(runner, run_dir,
                                     f"evaluate/{method}/{task_id}/{trial:02d}", toolbox, output)
                    records.append({"task_id": task_id, "trial": trial, "method": method,
                                    "success": bool(result.get("success")),
                                    "status": result.get("status"), "steps": result.get("steps", 0),
                                    "output_dir": str(output), "toolbox_sha256": toolbox_hash(toolbox)})
                    _progress(run_dir, "evaluation_trial_complete", task=task_id, method=method,
                              trial=trial + 1, status=result.get("status"), completed=len(records))
        result = {"records": records, "agent_count": getattr(agent, "count", None)}
        _save(run_dir / "evaluation.json", result)
        _progress(run_dir, "evaluation_complete", trials=len(records))
        return result


def report(run_dir: Path) -> dict:
    """Write an honest partial-or-complete report without optimizer feedback."""
    run_dir = Path(run_dir).resolve()
    manifest = _json(run_dir / "manifest.json")
    if not manifest:
        raise ResearchError(f"No experiment manifest in {run_dir}")
    ledger = _json(run_dir / "target_ledger.json", {"slots": {}})
    slots = list(ledger["slots"].values())
    rows = []
    for method in METHODS:
        for task_id in TASK_IDS:
            relevant = [slot for slot in slots if slot["phase"] == "evaluate" and
                        slot["method"] == method and slot["task_id"] == task_id]
            completed = [slot for slot in relevant if slot["state"] == "complete"]
            successes = sum(bool(slot.get("success")) for slot in completed)
            rows.append({"method": method, "task_id": task_id, "scheduled": len(relevant),
                         "completed": len(completed), "successes": successes,
                         "success_rate": successes / len(relevant) if len(completed) == len(relevant)
                         and relevant else None})
    adaptations = _json(run_dir / "adaptation.json", {}).get("tasks", {})
    decisions = 0
    for path in run_dir.rglob("*.json"):
        record = _json(path)
        decisions += isinstance(record, dict) and "request_sha256" in record
    calls = list((run_dir / "agent_calls").glob("*/request.json"))
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    models = set()
    for path in calls:
        request = _json(path, {})
        if request.get("model"):
            models.add(request["model"])
        for entry in _json(path.parent / "result.json", {}).get("usage", []):
            if isinstance(entry, dict):
                for key in usage:
                    usage[key] += int(entry.get(key, 0) or 0)
    traces = {}
    mismatch_count = 0
    for task_id in TASK_IDS:
        task_trials = []
        for path in sorted((run_dir / "adaptation" / task_id).glob("trial_*/summary.json")):
            summary = _json(path)
            target = _json(path.parent / "target" / "result.json", {})
            baseline = _json(path.parent / "alignment" / "initial" / "result.json")
            mismatch = baseline is not None and bool(baseline.get("success")) != bool(target.get("success"))
            mismatch_count += mismatch
            task_trials.append({"trial": summary["trial"], "success": summary["target_success"],
                                "selected": summary["selected"],
                                "source_target_success_disagreement": mismatch,
                                "selection_record": str(path.parent / "repair_selection.json")})
        traces[task_id] = task_trials
    source_statuses = Counter()
    source_attempt_statuses = Counter()
    source_valid = 0
    for path in run_dir.rglob("research_request.json"):
        request = _json(path)
        if request.get("engine") != "mujoco":
            continue
        outcome = _json(path.parent / "result.json")
        if outcome is None:
            source_statuses["missing_result"] += 1
            continue
        source_statuses[outcome.get("status", "unknown")] += 1
        source_valid += outcome.get("status") in ("success", "failed", "completed") and bool(
            outcome.get("actual_started", True))
        attempts = list((path.parent / "research_attempts").glob("*.json"))
        for attempt in attempts:
            source_attempt_statuses[_json(attempt).get("status", "unknown")] += 1
        if not attempts:
            source_attempt_statuses[outcome.get("status", "unknown")] += 1
    target_statuses = Counter(slot.get("status", slot["state"]) for slot in slots)
    protocol_complete = bool(slots) and all(slot["state"] == "complete" for slot in slots)
    valid_evaluation_groups = all(any(
        slot["phase"] == "evaluate" and slot.get("method") == method and slot["task_id"] == task_id
        and slot.get("status") in ("success", "failed", "completed") and slot.get("actual_started")
        for slot in slots) for method in METHODS for task_id in TASK_IDS)
    engineering_evidence_complete = source_valid > 0 and valid_evaluation_groups and all(
        (run_dir / filename).exists() for filename in ("discovery.json", "adaptation.json", "evaluation.json"))
    result = {"config": manifest["config"], "target_scheduled": len(slots),
              "target_completed": sum(slot["state"] == "complete" for slot in slots),
              "target_unresolved": sum(slot["state"] in ("unresolved", "unresolved_infrastructure")
                                       for slot in slots),
              "target_infrastructure_interrupted": sum(slot["state"] == "unresolved_infrastructure"
                                                       for slot in slots),
              "evaluation": rows, "adaptation": adaptations,
              "protocol_complete": protocol_complete,
              "engineering_evidence_complete": engineering_evidence_complete,
              "complete": protocol_complete and engineering_evidence_complete,
              "source_rollout_statuses": dict(source_statuses),
              "source_attempt_statuses": dict(source_attempt_statuses),
              "source_valid_physics_rollouts": source_valid,
              "target_rollout_statuses": dict(target_statuses),
              "agent_calls": len(calls) if calls else decisions,
              "research_decisions": decisions, "token_usage": usage, "models": sorted(models),
              "adaptation_trace": traces, "source_target_success_disagreements": mismatch_count}
    result["agent_calls_note"] = "All recorded model calls; token totals only include provider-reported usage."
    _save(run_dir / "report.json", result)
    text = ["# SimEX 本地初步复现报告", "",
            f"目标仿真：已完成 {result['target_completed']} / {result['target_scheduled']} 个排定试验。",
            "结果状态：" + ("流程及物理执行证据完整。" if result["complete"] else
                           "预算已执行完，但有效物理执行证据不足，不能判定工程验收完整。"
                           if protocol_complete else "未完成；不把缺失试验计为实测失败。"), "",
            "| 方法 | 任务 | 成功/排定 | 已完成 | 成功率 |",
            "|---|---|---:|---:|---:|"]
    for row in rows:
        rate = "未完成" if row["success_rate"] is None else f"{row['success_rate']:.1%}"
        text.append(f"| {row['method']} | {row['task_id']} | {row['successes']}/{row['scheduled']} | "
                    f"{row['completed']} | {rate} |")
    text.extend(["", "本实验采用单臂 Franka Panda、三个改编搬盘任务、缩小的种群与迭代预算，"
                 "每轮候选及筛选场景数量见 manifest.json。研究运行次数为 1；模型由本机 Codex 配置决定。",
                 "本报告不将完整执行等同于方法改善，也不宣称复现论文的 94% 成功率。",
                 "", "每次目标记录、固定脚本源重放、校准配置、候选证据和代理选择理由保存在 adaptation/；"
                 "最终评估独立保存在 evaluation/，不再反馈给优化代理。",
                 "执行失败、超时及非法策略消耗排定试验。启动前基础设施失败允许修复后恢复；"
                 "执行中的基础设施中断锁定原试验、保持未完成，不作为普通失败计入成功率，也不自动重跑。"
                 "需要恢复原始有效结果，或另建独立研究运行，不能以重试筛选成功。"])
    text.extend(["", f"记录模型调用 {result['agent_calls']} 次；代理报告 token 用量：{usage}。",
                 f"源仿真最终状态计数：{dict(source_statuses)}；含重启前基础设施失败的尝试计数："
                 f"{dict(source_attempt_statuses)}。有效源物理执行 {source_valid} 次。",
                 f"目标试验状态计数：{dict(target_statuses)}。非法策略、生成失败和基础设施失败"
                 "单独列出，不能用它们证明抓取方法或模拟器链路已经有效运行。",
                 f"初始固定脚本重放与目标成功判定不同的试验：{mismatch_count} 次。"
                 "该统计只反映结果差异；接触和轨迹是否匹配需要查看图像及代理校准理由。"])
    for task_id, trace in traces.items():
        if trace:
            outcomes = " → ".join("成功" if item["success"] else "失败" for item in trace)
            selections = ", ".join(item["selected"] for item in trace)
            text.append(f"- {task_id} 目标试验结果：{outcomes}；各轮选择：{selections}。")
    (run_dir / "report.md").write_text("\n".join(text) + "\n")
    return result
