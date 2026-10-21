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
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .agent import AgentBudgetExhausted
from .contracts import CAPABILITY_GUIDANCE
from .errors import ResearchError
from .research_context import (ContextError, MAX_PROMPT_CHARS, MAX_READ_ROUNDS,
                               READ_CHARS, ResearchContext, apply_source_edits, prepare_edit_context)
from .tasks import (TASK_IDS, PLACEMENT_CONTRACTS, FRICTION_SCALE_BOUNDS, ACTUATOR_CALIBRATION_BOUNDS,
                    make_task, source_task_from_spec, with_placement_contract)


DEFAULTS = {
    "population": 2, "cycles": 3, "probes": 8, "focused": 4,
    "revisions": 2, "replay": 2, "adapt_trials": 5, "repairs": 2,
    "screen_tasks": 3, "align_attempts": 3, "eval_trials": 5,
    "selection_trials": 3, "seed": 20261015,
}
METHODS = ("direct_cap", "stage1_transfer", "simex")
MAX_TRANSPORT_RETRIES = 5
CAPABILITY_FILES = ("toolbox.py", "skill.md")
PROVENANCE_KEYS = ("model", "codex_version", "reasoning_effort", "agent_timeout",
                   "policy_timeout", "max_sim_time", "isaac_asset", "source_policy_workers",
                   "stage1_reasoning_effort", "stage2_reasoning_effort", "evaluation_reasoning_effort",
                   "stage1_agent_timeout", "stage2_agent_timeout", "evaluation_agent_timeout")
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
) + "\n\n" + CAPABILITY_GUIDANCE
TASK_SPEC_HELP = (
    "Each task is {name,description,base_task_id,seed,scene_overrides,predicate}. "
    f"base_task_id is one of {list(TASK_IDS)}. Use small reachable scene variations. "
    "Use scene_overrides only for plates and tote initially; keep plate names/colors "
    "green and red because those are the initial perception classes. Copy plate fields "
    "from the supplied current scene template before varying them. Plate bounds: x [0.15,0.75], "
    "abs(y)<=0.45, z [0,0.25], radius [0.03,0.12], mass [0.02,1], friction [0.1,2]. "
    "A predicate is "
    '{"all":[{"kind":"lift","object":"green","min":0.05},'
    '{"kind":"inside","object":"green"}]}; kinds lift,sustained_lift,inside,outside,'
    "displaced (min meters), and released (fixed versioned opening, withdrawal, "
    "separation and settled-containment checks; no min field). Preserve any supplied "
    "released predicate exactly. inside checks geometric containment and can pass "
    "while an object is still held; it does not establish release. Include released "
    "when a task claims complete released placement. Subtasks may use their own "
    "partial predicates, but their descriptions must state that limited objective. "
    "A subtask success is evidence only for its declared predicate. "
    "lift tests historical maximum height, so a transient excursion can pass. "
    'For sampled elevation throughout the final interval use {"kind":"sustained_lift",'
    '"object":"green","min":0.05,"seconds":1.0}; seconds is [0.5,5]. '
    "This requires elevated final state and complete sampled window coverage with "
    "no gap above 0.30 s; it does not certify grasp attachment. Allow time to observe "
    "the full interval and state this objective in the task description. Use it for "
    "an elevated endpoint subtask, not a task meant to finish with the object placed down. "
    "Design relevant, nontrivial predicates; never "
    "weaken a predicate after seeing an outcome. Task descriptions must tell "
    "the policy what to achieve without object truth poses. If overriding plates, "
    "supply complete plate objects, including names and all geometry/appearance "
    "fields from source_scene_template; inherit its current mass, friction and dimensions. "
    "Do not copy physical parameters from an example. For intentional mass, friction or "
    "geometry changes, also return scene_change_reasons: an object mapping the zero-based "
    "task index (as a string) to the experimental reason for those changes. The harness "
    "records the actual experimental variables and flags missing explanations without "
    "changing or resampling tasks. Keep targets consistent with plates."
)


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
        raise AgentBudgetExhausted("Model-call budget exhausted; no further rollout has been scheduled")


def _provenance(config: dict) -> dict:
    dependencies = {}
    for dependency in ("mujoco", "numpy", "scipy"):
        try:
            dependencies[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            dependencies[dependency] = None
    return {"settings": {key: config.get(key) for key in PROVENANCE_KEYS},
            "dependencies": dependencies}


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


def _validate_toolbox(contents: dict):
    """Check capability source without writing or executing it."""
    for name in CAPABILITY_FILES:
        if not isinstance(contents.get(name), str):
            raise InvalidToolbox(f"Agent must return complete {name} text")
    try:
        from .policy_process import validate
        tree = validate(contents["toolbox.py"])
        # Parsing alone misses errors such as return outside a function.
        # Compile without executing candidate code or inheriting harness flags.
        compile(tree, "toolbox.py", "exec", dont_inherit=True)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Toolbox"]
        constructor = next((node for node in reversed(classes[-1].body)
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name == "__init__"), None) if classes else None
        if not isinstance(constructor, ast.FunctionDef):
            raise ValueError("toolbox.py must define Toolbox.__init__(self, robot)")
        # Check argument binding without executing candidate code or defaults.
        # Instance/class methods receive self/cls plus robot; static methods only robot.
        supplied = 1 if any(isinstance(node, ast.Name) and node.id == "staticmethod"
                            for node in constructor.decorator_list) else 2
        args = constructor.args
        positional = len(args.posonlyargs) + len(args.args)
        required = positional - len(args.defaults)
        if (required > supplied or (positional < supplied and args.vararg is None)
                or any(default is None for default in args.kw_defaults)):
            raise ValueError("Toolbox.__init__ must accept Toolbox(robot) without additional required arguments")
    except (SyntaxError, ValueError) as exc:
        raise InvalidToolbox(f"Invalid proposed toolbox: {exc}") from exc


def _write_toolbox(destination: Path, contents: dict) -> Path:
    _validate_toolbox(contents)
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
    _save(destination / "frozen.json", record)
    return destination


def _check_frozen(path: Path):
    """Require the snapshot artifacts; recorded hashes are informational only."""
    for name in (*CAPABILITY_FILES, "frozen.json"):
        if not (path / name).is_file():
            raise ResearchError(f"Frozen toolbox artifact missing: {path / name}")


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
    calibration_profile = cfg.get("source_calibration_profile", "basic")
    if calibration_profile not in ("basic", "actuator_v1"):
        raise ResearchError("source_calibration_profile must be basic or actuator_v1")
    contract = cfg.setdefault("placement_contract", "legacy_geometry")
    if contract not in PLACEMENT_CONTRACTS:
        raise ResearchError("Unsupported placement contract")
    if cfg.get("stage1_parent"):
        cfg["stage1_parent"] = str(Path(cfg["stage1_parent"]).resolve())
    mode = cfg.setdefault("discovery_mode", "legacy")
    if mode not in ("legacy", "interactive"):
        raise ResearchError("discovery_mode must be legacy or interactive")
    if mode == "interactive":
        for key, default in (("coding_max_turns", 40), ("coding_max_tests", 10)):
            cfg.setdefault(key, default)
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ResearchError(f"{key} must be a positive integer")
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
        if manifest.get("format_version") != 3:
            raise ResearchError("Stage 1 protocol changed; create a new run to preserve experiment provenance")
        _check_selection_sets(run_dir, manifest)
        old = manifest["config"]
        if old.get("source_calibration_profile", "basic") != calibration_profile:
            raise ResearchError("Source calibration profile changed; create a new run")
        if old.get("placement_contract", "legacy_geometry") != contract:
            raise ResearchError("Placement contract changed; create a new run")
        if contract == "released_placement_v1":
            from .release import CONTRACT_SPEC
            if manifest.get("placement_contract_spec") != CONTRACT_SPEC:
                raise ResearchError("Release thresholds changed; create a new run")
        if old.get("stage1_parent") != cfg.get("stage1_parent"):
            raise ResearchError("Stage 1 provenance changed; create a new run")
        if old.get("discovery_mode", "legacy") != mode:
            raise ResearchError("Discovery mode changed; create a new run")
        if mode == "interactive" and any(old.get(key) != cfg[key] for key in
                                        ("coding_max_turns", "coding_max_tests")):
            raise ResearchError("Coding budgets cannot change when resuming a run")
        if any(old[key] != cfg[key] for key in DEFAULTS
               if not (mode == "interactive" and key == "revisions")):
            raise ResearchError("Research budgets/seeds cannot change when resuming a run")
        previous_provenance = manifest.get("provenance", {})
        if {key: previous_provenance.get(key) for key in ("settings", "dependencies")} != _provenance(cfg):
            raise ResearchError("Model, runtime settings or dependencies changed; "
                                "create a new run to preserve experiment provenance")
        _check_frozen(run_dir / "frozen" / "initial")
        ledger = _json(run_dir / "target_ledger.json")
        if not isinstance(ledger, dict) or not isinstance(ledger.get("slots"), dict):
            raise ResearchError("Target ledger is missing or invalid")
        return manifest["config"]
    initial = (Path(cfg["stage1_parent"]) / "frozen" / "initial"
               if cfg.get("stage1_parent") else Path(__file__).parent / "initial")
    _freeze(initial, run_dir / "frozen" / "initial")
    selection = [with_placement_contract(make_task(task_id, cfg["seed"] + 10000 + i * 100 + j), contract)
                 for i, task_id in enumerate(TASK_IDS)
                 for j in range(cfg["selection_trials"])]
    _save(run_dir / "selection_tasks.json", selection)
    auxiliary = _auxiliary_selection_tasks(cfg)
    _save(run_dir / "auxiliary_selection_tasks.json", auxiliary)
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
    from .release import CONTRACT_SPEC
    _save(manifest_path, {"format_version": 3, "config": cfg, "provenance": _provenance(cfg),
                          "placement_contract_spec": CONTRACT_SPEC if contract == "released_placement_v1" else None,
                          "selection_sha256": _digest(selection),
                          "auxiliary_selection_sha256": _digest(auxiliary),
                          "target_schedule_sha256": _digest({k: {a: b for a, b in v.items()
                              if a != "state"} for k, v in slots.items()}),
                          "initial_sha256": toolbox_hash(run_dir / "frozen" / "initial")})
    return cfg


def _auxiliary_selection_tasks(cfg: dict) -> list[dict]:
    """Common frozen tie-breakers, never supplied to discovery researchers."""
    tasks = []
    for kind, threshold, instruction in (
        ("lift", .05, "Lift the green plate by at least 5 cm"),
        ("displaced", .03, "Move the green plate at least 3 cm from its initial position"),
    ):
        for index in range(2):
            task = make_task(TASK_IDS[index], cfg["seed"] + 15000 + index)
            task["name"] = f"selection_{kind}_{index}"
            task["description"] = instruction + ". The red plate must remain outside the tote."
            task["predicate"] = {"all": [{"kind": kind, "object": "green", "min": threshold},
                                          {"kind": "outside", "object": "red"}]}
            tasks.append(task)
    return tasks


def _check_selection_sets(run_dir: Path, manifest: dict):
    for filename in ("selection_tasks.json", "auxiliary_selection_tasks.json"):
        values = _json(run_dir / filename)
        if not isinstance(values, list):
            raise ResearchError(f"Source selection tasks are missing or invalid: {filename}")


def _scene_context(cfg: dict | None = None) -> dict:
    templates = {task_id: make_task(task_id, 0)["scene"] for task_id in TASK_IDS}
    context = {"source_scene_template": templates[TASK_IDS[0]], "source_scene_templates": templates}
    if (cfg or {}).get("placement_contract") == "released_placement_v1":
        reference = with_placement_contract(make_task(TASK_IDS[0], 0), "released_placement_v1")
        context["complete_placement_reference"] = {
            "contract": "released_placement_v1", "description": reference["description"],
            "predicate": reference["predicate"],
            "scope": "Reference for the meaning of complete released placement, not a fixed "
                     "curriculum. Choose broad tasks, variations, sub-goals and predicates from "
                     "source evidence. Prerequisite or held-object tasks need not satisfy the "
                     "complete objective; label their scope explicitly and do not count their "
                     "success as full placement. Keep each task's predicate frozen during optimization.",
        }
    return context


def _record_scene_variations(response: dict, tasks: list[dict], path: Path):
    """Audit source experiments without adding hidden metadata to policy inputs."""
    reasons = response.get("scene_change_reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}
    def differences(before, after, prefix=""):
        # Plate order is not an experimental variable. Compare validated objects
        # by identity, including additions/removals, rather than list position.
        if prefix == "plates" and isinstance(before, list) and isinstance(after, list):
            return differences({item["name"]: item for item in before},
                               {item["name"]: item for item in after}, prefix)
        if isinstance(before, dict) and isinstance(after, dict):
            return [change for key in sorted(set(before) | set(after))
                    for change in differences(before.get(key), after.get(key), f"{prefix}.{key}".strip("."))]
        if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
            return [change for index, (left, right) in enumerate(zip(before, after))
                    for change in differences(left, right, f"{prefix}.{index}")]
        return [] if before == after else [{"field": prefix, "template": before, "experiment": after}]
    records = []
    physical_fields = {"mass", "friction", "radius", "thickness", "rim_height", "inner_size", "wall_thickness"}
    for index, (spec, task) in enumerate(zip(response["tasks"], tasks)):
        template = source_task_from_spec({"base_task_id": spec.get("base_task_id", TASK_IDS[0]),
                                          "seed": spec.get("seed", 0)})["scene"]
        changes = differences(template, task["scene"])
        physical = [change for change in changes if set(change["field"].split(".")) & physical_fields
                    or (change["field"].startswith("plates.") and change["field"].count(".") == 1)]
        reason = reasons.get(str(index))
        record = {"task_index": index, "task_sha256": _digest(task), "template_sha256": _digest(template),
                  "source_scene_template": template, "changes": changes,
                  "physical_changes": physical,
                  "experimental_variables": [change["field"] for change in physical],
                  "experimental_reason": reason if isinstance(reason, str) else None,
                  "declaration_status": ("not_needed" if not physical else
                      "explained" if isinstance(reason, str) and reason.strip() else "missing_reason")}
        records.append(record)
    previous = _json(path)
    def audit_fields(items):
        return [{key: value for key, value in item.items()
                 if key not in ("task_sha256", "template_sha256")} for item in items]
    if previous is not None and audit_fields(previous) != audit_fields(records):
        raise ResearchError(f"Frozen source scene variation record changed: {path}")
    _save(path, records)
    return [{key: value for key, value in record.items() if key != "source_scene_template"}
            for record in records]


def _images(value: Any, *, include_all: bool = False) -> list[str]:
    """Choose default sheets, or enumerate all approved images for on-demand reads."""
    found = []
    def visit(item):
        if isinstance(item, dict):
            image_keys = ("phase_images", "keyframes", "read_images")
            if not include_all:
                image_keys = ("phase_images",) if item.get("phase_images") else ("keyframes", "read_images")
            for key, child in item.items():
                if key in image_keys and isinstance(child, list):
                    for path in child:
                        if isinstance(path, str) and path not in found and Path(path).is_file():
                            found.append(path)
                elif key == "action_images" and include_all and isinstance(child, list):
                    for entry in child:
                        path = entry.get("path") if isinstance(entry, dict) else None
                        if isinstance(path, str) and path not in found and Path(path).is_file():
                            found.append(path)
                elif key not in ("phase_images", "keyframes", "read_images", "action_images") and isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return found


def _ask(agent, path: Path, role: str, instruction: str, payload: dict) -> dict:
    """Research decisions can request bounded public evidence; policies cannot.

    Each successful read turn is checkpointed separately. Network failures get
    up to five retries here, without spending a dialogue turn or model budget.
    """
    from .agent import AgentTransportError

    previous = _json(path)
    if previous is not None:
        return previous["response"]
    if role in ("discover_revision", "repair"):
        instruction += "\n\n" + TOOLBOX_CONTRACT
    instruction, pinned_keys, block_edits = prepare_edit_context(role, instruction, payload)
    context = ResearchContext(payload, _images(payload, include_all=True),
        default_image_paths=_images(payload), instruction=instruction, pinned_keys=pinned_keys)
    input_hash = _digest({"role": role, "instruction": instruction, "context": context.input_sha256})
    directory = path.with_suffix(".context")
    context.persist(directory)
    notes, replies, reads_so_far, image_ids = "", [], [], None
    initial_request = None
    try:
        for turn in range(MAX_READ_ROUNDS + 1):
            prompt, image_paths = context.prompt(instruction, round_index=turn, notes=notes,
                replies=replies, reads_so_far=reads_so_far, image_ids=image_ids)
            request = {"role": role, "prompt": prompt, "images": image_paths}
            if initial_request is None:
                initial_request = request
            turn_path = directory / f"turn_{turn:02d}.json"
            saved = _json(turn_path)
            if saved is not None:
                response = saved["response"]
            else:
                for attempt in range(MAX_TRANSPORT_RETRIES + 1):
                    _agent_capacity(agent)
                    print(f"[SimEX] research_request role={role} read_round={turn} "
                          f"prompt_chars={len(prompt)} images={len(image_paths)} "
                          f"raw_payload_chars={context.raw_chars}", flush=True)
                    try:
                        response = agent.ask(role=role, prompt=prompt, images=image_paths)
                        break
                    except AgentTransportError as exc:
                        errors_path = directory / f"turn_{turn:02d}.transport_errors.json"
                        errors = _json(errors_path, [])
                        errors.append({"error": str(exc), "agent_count": getattr(agent, "count", None),
                                       "time_unix": time.time(),
                                       "automatic_retry": attempt < MAX_TRANSPORT_RETRIES})
                        _save(errors_path, errors)
                        if attempt == MAX_TRANSPORT_RETRIES:
                            raise
                        print(f"[SimEX] research_transport_retry role={role} read_round={turn} "
                              f"retry={attempt + 1}/{MAX_TRANSPORT_RETRIES}; no rollout is repeated", flush=True)
                if not isinstance(response, dict):
                    raise ResearchError(f"Agent role {role} returned a non-object")
                _save(turn_path, {"request_sha256": _digest(request), "request": request,
                    "response": response, "agent_count": getattr(agent, "count", None)})
            if "evidence_requests" not in response:
                if block_edits:
                    try:
                        response = apply_source_edits(response, payload["toolbox"])
                    except ContextError as exc:
                        response = {"invalid_edits": str(exc), "source_edits": response,
                                    "hypothesis": response.get("hypothesis")}
                _save(path, {"input_sha256": input_hash, "request_sha256": _digest(initial_request),
                    "request": initial_request, "response": response,
                    "agent_count": getattr(agent, "count", None), "read_rounds": turn,
                    "context_directory": str(directory)})
                return response
            if turn == MAX_READ_ROUNDS:
                raise ResearchError("Research evidence read budget exhausted without a decision; checkpoint retained")
            if set(response) - {"evidence_requests", "notes"} or not isinstance(response.get("notes", ""), str):
                raise ResearchError("Evidence reads must be separate from the final decision and use text notes")
            notes = response.get("notes", "")
            try:
                # Include accumulated notes/indexes before allocating this page.
                # Leave room for new read entries, image labels and adapter text.
                base_prompt, _ = context.prompt(instruction, round_index=turn + 1,
                    notes=notes, reads_so_far=reads_so_far, image_ids=image_ids)
                read_budget = min(READ_CHARS, MAX_PROMPT_CHARS-len(base_prompt)-3_000)
                replies, requested_images = context.read(response["evidence_requests"], budget=read_budget)
                image_ids = requested_images or context.default_image_ids
                reads_so_far += [{key: item[key] for key in ("id", "offset", "end", "found", "attached")
                                  if key in item} for item in replies]
            except ContextError as exc:
                # A bad read consumes a dialogue turn, not a simulator trial.
                # Give a bounded public error so the agent can correct the ID.
                replies = [{"read_error": str(exc)}]
    except Exception as exc:
        _save(path.with_suffix(".error.json"), {"role": role, "error": str(exc),
              "exception": type(exc).__name__, "agent_count": getattr(agent, "count", None)})
        raise


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
    elif result.get("output_dir"):
        # Older results embedded only the final 16k characters, sometimes starting
        # in the middle of a long motion record. The policy's own captured stdout
        # is public evidence. Read only this fixed artifact, never model-supplied
        # paths or other files in the rollout directory.
        root = Path(result["output_dir"]).resolve()
        transcript = (root / "policy_stdout.log").resolve()
        if transcript == root / "policy_stdout.log" and transcript.is_file():
            evidence = {**evidence, "stdout": transcript.read_text()}
    if evidence.get("recording_errors"):
        evidence = {**evidence, "recording_errors": ["Public evidence capture failed."] * len(evidence["recording_errors"])}
    from .evidence import summarize_public_evidence
    evidence = summarize_public_evidence(evidence, result.get("output_dir"))
    allowed = ("status", "success", "reason", "trace_tail", "robot_trace_tail",
               "read_images", "keyframes", "stdout", "initial_visual_estimates",
               "phase_images", "action_images", "visual_phases", "motion_summary", "robot_trace",
               "actions", "action_summary",
               "video", "recording_errors", "evidence_errors")
    selected = {key: evidence[key] for key in allowed if key in evidence}
    # Keep raw public evidence available to the read broker. It replaces traces
    # and original image inventories with IDs instead of discarding them.
    return {"status": result.get("status"), "success": bool(result.get("success")),
            "steps": result.get("steps", 0), "evidence": selected}


def _candidate(proposal: dict, directory: Path) -> tuple[Path | None, dict]:
    """Invalid code spends this proposal slot; it never triggers resampling on resume."""
    try:
        if proposal.get("invalid_edits"):
            raise InvalidToolbox(proposal["invalid_edits"])
        toolbox = _write_toolbox(directory / "toolbox", proposal)
    except InvalidToolbox as exc:
        outcome = {"status": "invalid_candidate", "reason": str(exc)}
        _save(directory / "validation.json", outcome)
        return None, outcome
    outcome = {"status": "valid", "toolbox_sha256": toolbox_hash(toolbox)}
    _save(directory / "validation.json", outcome)
    return toolbox, outcome


def _changes(current: Path, proposal: dict) -> dict:
    changes = {}
    for name in CAPABILITY_FILES:
        if not isinstance(proposal.get(name), str):
            continue
        diff = difflib.unified_diff(
            (current / name).read_text().splitlines(keepends=True),
            proposal[name].splitlines(keepends=True),
            fromfile="current/" + name, tofile="candidate/" + name)
        # Keep separate diff lines and preserve missing final newlines explicitly.
        changes[name] = "".join(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in diff)
    return changes


def _outcome_summary(public_result: dict | None) -> dict | None:
    """History retains outcomes and diagnostics without reattaching all old images."""
    if public_result is None:
        return None
    evidence = public_result.get("evidence", {})
    return {"status": public_result.get("status"), "success": public_result.get("success"),
            "steps": public_result.get("steps"), "reason": evidence.get("reason", ""),
            "stdout": evidence.get("stdout", "")[-2000:],
            "motion_summary": evidence.get("motion_summary"),
            "action_summary": evidence.get("action_summary")}


def _started_without_outcome(out: Path, result: dict | None) -> bool:
    """An earlier startup failure cannot resolve a later execution handshake."""
    if result is not None and result.get("status") != "infrastructure_error":
        return False
    execution = _json(out / "execution.json", {})
    return execution.get("actual_started") is True or bool(result and result.get("actual_started"))


def _run(runner, engine: str, task: dict, toolbox: Path, out: Path,
         script: str | None = None, source_config: dict | None = None, source_model=None) -> dict:
    request = {"engine": engine, "task": task, "toolbox_sha256": toolbox_hash(toolbox),
               "script_sha256": _digest(script), "source_config": source_config or {}}
    if source_model is not None:
        from .source_model import check_model, model_hash
        check_model(source_model)
        request['source_model_sha256'] = model_hash(source_model)
    strict = source_model is not None or getattr(runner, 'config', {}).get('protocol') == 'supplied_skill_stage2_v1'
    request_path = out / "research_request.json"
    old = _json(request_path)
    if old is not None and (old != request if strict else any(old.get(key) != request[key]
                               for key in ("engine", "task", "source_config"))):
        raise ResearchError(f"Rollout checkpoint input changed: {out}")
    result_path = out / "result.json"
    previous = _json(result_path)
    if _started_without_outcome(out, previous):
        raise ResearchError(f"Rollout execution started without a durable outcome; "
                            f"refusing automatic rerun: {out}")
    if previous is not None and previous.get("status") != "infrastructure_error":
        if previous.get("failure_kind") == "infrastructure":
            raise ResearchError(f"Infrastructure interrupted this rollout; recover its outcome before continuing: {out}")
        if getattr(runner, 'config', {}).get('protocol') == 'supplied_skill_stage2_v1':
            from .stage2 import check_rollout
            check_rollout(out)
        return previous
    if script is None and not (out / "policy_generation.json").exists():
        _agent_capacity(getattr(runner, "agent", None))
    _save(request_path, request)
    if previous is not None:
        # Remove the old failure from the active result path before retrying.
        # An interrupted retry must not be mistaken for this earlier failure.
        archived = out / "research_attempts" / "prior_results"
        archived.mkdir(parents=True, exist_ok=True)
        index = max((int(path.stem) for path in archived.glob("*.json")
                     if path.stem.isdigit()), default=-1) + 1
        result_path.replace(archived / f"{index:03d}.json")
    extra = {'source_model': source_model} if source_model is not None else {}
    result = runner.rollout(engine=engine, task=task, toolbox=toolbox, out=out,
                            script=script, source_config=source_config, **extra)
    if not isinstance(result, dict):
        raise ResearchError("Runner did not return a result object")
    _save(result_path, result)
    if getattr(runner, 'config', {}).get('protocol') == 'supplied_skill_stage2_v1':
        from .stage2 import seal_rollout
        seal_rollout(out)
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
        return existing
    if _started_without_outcome(out, existing):
        slot.update(state="unresolved_infrastructure", status="interrupted", actual_started=True)
        slot.pop("result_sha256", None)
        _save(ledger_path, ledger)
        raise ResearchError(
            f"Target slot {slot_key} started without a durable outcome; refusing automatic rerun. "
            "Recover the original outcome before resuming.")
    if slot["state"] in ("running", "unresolved", "unresolved_infrastructure") and existing is None:
        slot["state"] = "unresolved"
        _save(ledger_path, ledger)
        raise ResearchError(
            f"Target slot {slot_key} has no durable outcome; refusing automatic rerun. "
            "Recover the original result.json or record an interrupted result with "
            "actual_started=true before resuming.")
    if existing is not None and existing.get("status") != "infrastructure_error":
        result = existing
    else:
        if not (out / "policy_generation.json").exists():
            _agent_capacity(getattr(runner, "agent", None))
        slot.update(state="running", output_dir=str(out.resolve()),
                    toolbox_sha256=toolbox_hash(toolbox))
        _save(ledger_path, ledger)
        try:
            cfg = _json(run_dir / "manifest.json", {}).get("config", {})
            task = with_placement_contract(make_task(slot["task_id"], slot["seed"]),
                                           cfg.get("placement_contract", "legacy_geometry"))
            if cfg.get('camera_resolution'):
                task['scene']['cameras'].update(zip(('width', 'height'), cfg['camera_resolution']))
            result = _run(runner, "isaac", task,
                          toolbox, out)
        except BaseException:
            existing = _json(result_path)
            if _started_without_outcome(out, existing):
                slot.update(state="unresolved_infrastructure", status="interrupted", actual_started=True)
                slot.pop("result_sha256", None)
            elif existing is not None and existing.get("failure_kind") == "infrastructure" and existing.get("actual_started"):
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
           source_config: dict | None = None, script: str | None = None, source_model=None) -> list[dict]:
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
                    except AgentBudgetExhausted:
                        for pending_future in futures:
                            pending_future.cancel()
                        raise
                    except Exception as exc:
                        errors.append(exc)
                        progress("source_policy_preparation_error", trial=futures[future] + 1,
                                 error=str(exc))
            if errors:
                raise ResearchError(f"Source policy preparation failed: {errors[0]}") from errors[0]
    results = []
    for index, task in enumerate(tasks):
        result = _run(runner, "mujoco", task, toolbox, out / f"{index:03d}", script, source_config, source_model)
        results.append(result)
        activity = result.get("action_summary", {})
        progress("source_rollout_complete", trial=index + 1, scheduled=len(tasks),
                 status=result.get("status"), success=bool(result.get("success")),
                 arm_commands=activity.get("arm_commands"),
                 gripper_commands=activity.get("gripper_commands"),
                 arm_motion_observed=activity.get("arm_motion_observed"))
    return results


def _score(results: list[dict]) -> tuple[int, int]:
    return sum(bool(result.get("success")) for result in results), sum(
        int(result.get("steps") or 0) for result in results)


def _source_evidence(result: dict) -> dict:
    """Executed CaP code is research evidence, never a persistent robot skill.

    Deliberately separate from _public: Stage 2's target boundary is unchanged.
    These artifacts are only passed to source discovery/repair researchers, not
    to the fresh one-shot policy writer.
    """
    public = _public(result)
    script = _executed_script(result)
    execution = {"actual_started": result.get("actual_started") is True,
                 "script": script, "script_sha256": _digest(script) if script is not None else None}
    output = result.get("output_dir")
    if output:
        response = Path(output) / "policy_response.txt"
        if response.is_file():
            match = re.search(r"===PLAN START===\s*(.*?)\s*===PLAN END===", response.read_text(), re.S)
            if match:
                execution["plan"] = match.group(1)
    public["execution"] = execution
    return public


def _predicate_signature(task: dict) -> str:
    """Group goal structures, independent of poses, thresholds and task names."""
    def shape(node):
        if not isinstance(node, dict):
            return None
        for operator in ("all", "any"):
            if operator in node:
                children = [shape(child) for child in node[operator]]
                return {operator: sorted(children, key=lambda value: json.dumps(value, sort_keys=True))}
        return {key: node[key] for key in ("kind", "object") if key in node}
    return json.dumps(shape(task.get("predicate")), sort_keys=True)


def _choose_replay(bank: list[dict], count: int, previous: list[list[str]]) -> list[dict]:
    """Deterministic, coverage-aware sampling; no goal-specific priority rules."""
    uses = Counter(key for selected in previous for key in selected)
    signatures = {record["task_sha256"]: record["predicate_signature"] for record in bank}
    coverage = Counter(signatures[key] for selected in previous for key in selected if key in signatures)
    remaining, chosen = list(bank), []
    while remaining and len(chosen) < count:
        entry = min(remaining, key=lambda record: (
            coverage[record["predicate_signature"]], uses[record["task_sha256"]],
            record["first_success"]["cycle"], record["task_sha256"]))
        chosen.append(copy.deepcopy(entry))
        remaining.remove(entry)
        coverage[entry["predicate_signature"]] += 1
    return chosen


def _remember_successes(bank: list[dict], tasks: list[dict], results: list[dict],
                        toolbox: Path, cycle: int, phase: str) -> list[dict]:
    """Preserve successes and latest observations, including pre-edit versions."""
    if len(tasks) != len(results):
        raise ResearchError("Task/result counts differ; refusing incomplete capability observations")
    records = copy.deepcopy(bank)
    by_task = {record["task_sha256"]: record for record in records}
    for index, (task, result) in enumerate(zip(tasks, results)):
        task_hash = _digest(task)
        passed = bool(result.get("success")) and result.get("actual_started") is True
        if not passed and task_hash not in by_task:
            continue
        evidence = _source_evidence(result)
        success = {"cycle": cycle, "phase": phase, "trial": index,
                   "toolbox_sha256": toolbox_hash(toolbox), "toolbox_path": str(toolbox.resolve()),
                   "script_sha256": evidence["execution"]["script_sha256"],
                   "script_path": result.get("script_path"), "output_dir": result.get("output_dir"),
                   "result_sha256": _digest(result),
                   "outcome": _outcome_summary(evidence)}
        if task_hash not in by_task:
            record = {"schema_version": 1, "task_sha256": task_hash, "task": copy.deepcopy(task),
                      "predicate_signature": _predicate_signature(task), "first_success": success}
            records.append(record)
            by_task[task_hash] = record
        if passed:
            by_task[task_hash]["last_success"] = success
        by_task[task_hash]["latest_observation"] = success
    return records


def _replay_evidence(records: list[dict]) -> list[dict]:
    """Historical scripts aid diagnosis; _suite still generates fresh policies."""
    evidence = []
    for record in records:
        origin = record["first_success"]
        result = _json(Path(origin["output_dir"]) / "result.json") if origin.get("output_dir") else None
        if result is None:
            raise ResearchError("Historical success evidence missing from capability bank")
        public = _source_evidence(result)
        evidence.append({"task_sha256": record["task_sha256"], "task": record["task"],
                         "successful_toolbox_sha256": origin["toolbox_sha256"],
                         "successful_toolbox": _contents(Path(origin["toolbox_path"])),
                         "historical_success": public})
    return evidence


def _discovery_regression(current: list[dict], candidate: list[dict], focused: int,
                          replay: list[dict]) -> dict:
    if type(focused) is not int or focused < 0 or not (
        len(current) == len(candidate) == focused + len(replay)
    ):
        raise ResearchError("Incomplete comparison: expected every focused and replay result")
    focused_ok = _score(candidate[:focused])[0] >= _score(current[:focused])[0]
    history = [{"task_sha256": record["task_sha256"], "historically_passed": True,
                "current_success": bool(old.get("success")),
                "candidate_success": bool(new.get("success")),
                "baseline_drift": not bool(old.get("success"))}
               for record, old, new in zip(replay, current[focused:], candidate[focused:])]
    paired_ok = all(not item["current_success"] or item["candidate_success"] for item in history)
    unresolved = [item["task_sha256"] for item in history
                  if not item["current_success"] and not item["candidate_success"]]
    # Fresh policy variation alone does not prove a code regression. Tied
    # historical failures require an explicit agent tradeoff, never a silent
    # reset of the previous success or an impossible all-history-perfect gate.
    return {"no_regression": focused_ok and paired_ok, "focused_non_decrease": focused_ok,
            "current_successes": _score(current)[0], "candidate_successes": _score(candidate)[0],
            "zero_success_comparison": _score(current)[0] == 0 and _score(candidate)[0] == 0,
            "paired_replays_preserved": paired_ok,
            "historical_replays_preserved": all(item["candidate_success"] for item in history),
            "unresolved_historical_tasks": unresolved, "historical_replays": history,
            "comparison": "fresh_one_shot_policies; historical scripts are diagnostic evidence only"}


def _discovery_acceptance(decision: dict, regression: dict) -> dict:
    unresolved = regression["unresolved_historical_tasks"]
    reason = decision.get("history_reason")
    acknowledged = decision.get("accept_unresolved_history") is True and isinstance(reason, str) and bool(reason.strip())
    gate_reason = None
    if not regression["no_regression"]:
        gate_reason = "Focused score decreased or a currently passing replay failed; historical acknowledgement cannot override this guard."
    elif unresolved and not acknowledged:
        gate_reason = ("Historical successes failed in both fresh-policy versions. To accept this uncertain tradeoff, "
                       "return accept_unresolved_history=true and a nonempty history_reason tied to those tasks; "
                       "otherwise retain the current toolbox. Historical successes remain in the bank.")
    elif regression.get("zero_success_comparison") and not (
            isinstance(decision.get("reason"), str) and decision["reason"].strip()):
        gate_reason = "Both versions have zero successes; reason must describe the hypothesis actually exercised and its observed outcome."
    return {"accepted": decision.get("accept") is True and gate_reason is None,
            "demonstrated_improvement": regression.get("candidate_successes", 0) > regression.get("current_successes", 0),
            "gate_reason": gate_reason, "unresolved_history_acknowledged": acknowledged,
            "history_reason": reason if isinstance(reason, str) else None}


def _history_attempts(attempts: list[dict]) -> list[dict]:
    """Bound cross-cycle evidence; same-cycle evidence remains complete."""
    return [{"revision": item["revision"], "hypothesis": item.get("hypothesis"),
             "accepted": item["accepted"], "reason": item.get("reason"),
             "demonstrated_improvement": item.get("demonstrated_improvement", False),
             "gate_reason": item.get("gate_reason"), "history_reason": item.get("history_reason"),
             "changes": {name: diff[:6000] for name, diff in item.get("changes", {}).items()},
             "changes_truncated": {name: len(diff) > 6000 for name, diff in item.get("changes", {}).items()},
             "results": [_outcome_summary(result) for result in item.get("results", [])],
             "regression": item.get("regression")}
            for item in attempts[-2:]]


def _discover_cycle(runner, agent, run_dir: Path, cfg: dict, index: int, cycle: int,
                    state: dict, peers: list[dict]) -> dict:
    root = run_dir / "discovery" / f"member_{index:02d}"
    current, bank = state["current"], state["bank"]
    summaries, histories = state["summaries"], state["histories"]
    probe_toolbox = current
    _progress(run_dir, "source_cycle_started", member=index, cycle=cycle + 1)
    cycle_dir = root / f"cycle_{cycle:02d}"
    peer_path = cycle_dir / "peer_summaries.json"
    saved_peers = _json(peer_path)
    if saved_peers is not None and saved_peers != peers:
        raise ResearchError("Frozen peer summaries changed at cycle boundary")
    _save(peer_path, peers)
    probe_response = _ask(agent, cycle_dir / "probe_proposal.json",
        "discover_probe", f"Generate exactly {cfg['probes']} broad source-only tasks "
        "to expose missing reusable robot capabilities. Return {tasks:[...]}. "
        "Match difficulty to demonstrated abilities. If acquisition has not been "
        "demonstrated, include focused prerequisite experiments with achievable "
        "object-motion predicates, not mostly complete delivery tasks. Choose the "
        "tasks yourself; no fixed curriculum is prescribed. " + TASK_SPEC_HELP,
        {"member": index, "cycle": cycle, "toolbox": _contents(current),
         **_scene_context(cfg),
         "own_history": summaries, "recent_research_history": histories[-2:],
         "peer_experiment_summaries": peers})
    probes = _tasks_from_response(probe_response, cfg["probes"])
    probe_variations = _record_scene_variations(probe_response, probes, cycle_dir / "probe_scene_variations.json")
    _save(cycle_dir / "probe_tasks.json", probes)
    probe_results = _suite(runner, probes, current, cycle_dir / "probes")
    probe_evidence = [_source_evidence(result) for result in probe_results]
    bank = _remember_successes(bank, probes, probe_results, probe_toolbox, cycle, "probe")
    _save(cycle_dir / "bank_after_probes.json", bank)
    _save(root / "capability_bank.json", bank)
    _progress(run_dir, "source_probe_complete", member=index, cycle=cycle + 1,
              successes=_score(probe_results)[0], scheduled=len(probes))
    focus = _ask(agent, cycle_dir / "focus_proposal.json", "discover_focus",
        f"Identify one systematic weakness and return {{weakness:str,tasks:[...]}} "
        f"with exactly {cfg['focused']} focused source tasks. Inspect executed code/plans "
        "alongside observations; separate policy-writing errors from reusable-tool failures. "
        "Check action_summary first: no arm commands means manipulation was not tested. "
        "Use close/open command counts, minimum TCP height, TCP rise after close, motion "
        "rejections, termination reason and phase images to distinguish observation-only "
        "stops, contact attempts, lift attempts and placement failures. These are action "
        "observations, never proof of grasp success; compare the independent predicate. "
        "Diagnose unnecessary precondition vetoes and observation recovery before "
        "attributing such outcomes to friction, payload or grasp mechanics. "
        "Successful one-shot scripts are diagnostic evidence, not persistent capabilities. "
        "Treat frozen peer summaries as hypotheses to check independently, not this member's "
        "demonstrated capabilities. " + TASK_SPEC_HELP,
        {"toolbox": _contents(current), **_scene_context(cfg), "probe_tasks": probes, "probe_results": probe_evidence,
         "probe_scene_variations": probe_variations,
         "peer_experiment_summaries": peers,
         "recent_research_history": histories[-2:]})
    focused = _tasks_from_response(focus, cfg["focused"])
    _record_scene_variations(focus, focused, cycle_dir / "focused_scene_variations.json")
    replay_path = cycle_dir / "replay_records.json"
    replay_records = _json(replay_path)
    if replay_records is None:
        replay_records = _choose_replay(bank, cfg["replay"], state["replay_history"])
        _save(replay_path, replay_records)
    replay = [record["task"] for record in replay_records]
    anchors = _replay_evidence(replay_records)
    _save(cycle_dir / "focused_tasks.json", focused)
    _save(cycle_dir / "replay_tasks.json", replay)
    current_results = _suite(runner, focused + replay, current, cycle_dir / "before")
    bank = _remember_successes(bank, focused + replay, current_results, current, cycle, "focused_baseline")
    _save(cycle_dir / "bank_after_baseline.json", bank)
    _save(root / "capability_bank.json", bank)
    accepted, revision_history = [], []
    coding_summary = None
    interactive = cfg.get("discovery_mode", "legacy") == "interactive"
    if interactive:
        from .coding import run_coding_cycle
        current, current_results, bank, accepted, revision_history, coding_summary = run_coding_cycle(
            runner, agent, cycle_dir / "coding", cfg, current=current, current_results=current_results,
            bank=bank, tasks=focused + replay, focused=len(focused), replay=replay_records, cycle=cycle,
            payload={"weakness": focus.get("weakness"), **_scene_context(cfg),
                     "tasks": focused + replay,
                     "probe_examples": [{"task": task, "result": result}
                                        for task, result in zip(probes, probe_evidence)],
                     "probe_toolbox": _contents(probe_toolbox),
                     "historical_replay_anchors": anchors,
                     "peer_experiment_summaries": peers,
                     "recent_research_history": histories[-2:]})
    for revision in range(0 if interactive else cfg["revisions"]):
        rev_dir = cycle_dir / f"revision_{revision:02d}"
        proposal = _ask(agent, rev_dir / "proposal.json", "discover_revision",
            "Propose reusable improvements for the focused weakness. Return complete "
            '{"toolbox.py":str,"skill.md":str,"hypothesis":str}. Only these two files '
            "persist as robot capabilities. Preserve API compatibility. Do not read "
            "simulator truth or implement direct simulator control. Study successful probe "
            "scripts as well as failed focused scripts: extract reusable mechanisms instead "
            "of encoding task poses or persisting one-shot scripts. Use prior_attempts and "
            "recent_research_history to learn from rejected code, outcomes and reasons. "
            "Historical replay tasks remain previously solved even if the freshly generated "
            "current policy fails; preserve those capabilities without claiming that fresh "
            "policy variation alone proves a toolbox regression. If policies stopped "
            "before motion, repair executable exploration and overly strict usage "
            "advice; adding another helper with an impossible caller guard is not "
            "a remedy. Simulation trials may proceed with bounded uncertainty as "
            "described in the exploration contract. Use action_summary and phase images "
            "to identify whether closing, lifting or releasing was actually exercised; "
            "robot motion alone does not demonstrate object acquisition.",
            {"weakness": focus.get("weakness"), "toolbox": _contents(current), **_scene_context(cfg),
             "tasks": focused + replay, "results": [_source_evidence(r) for r in current_results],
             "probe_examples": [{"task": task, "result": result}
                                for task, result in zip(probes, probe_evidence)],
             "probe_toolbox": _contents(probe_toolbox),
             "probe_toolbox_sha256": toolbox_hash(probe_toolbox),
             "historical_replay_anchors": anchors, "prior_attempts": revision_history,
             "peer_experiment_summaries": peers,
             "recent_research_history": histories[-2:]})
        candidate, validation = _candidate(proposal, rev_dir)
        attempt_record = {"revision": revision, "hypothesis": proposal.get("hypothesis"),
                          "baseline_toolbox_sha256": toolbox_hash(current),
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
        regression = _discovery_regression(current_results, candidate_results, len(focused), replay_records)
        no_regression = regression["no_regression"]
        decision = _ask(agent, rev_dir / "decision.json", "discover_select",
            'Review the task predicates, implementation diff, executed scripts/plans and source '
            'evidence; return {"accept":bool,"reason":str,"accept_unresolved_history":bool,"history_reason":str}. '
            "Accept only a plausible reusable improvement with no focused-score decrease and "
            "no current-pass to candidate-fail replay regression. If both versions fail a "
            "historically successful task, acceptance additionally requires accept_unresolved_history=true "
            "and a nonempty history_reason explaining the specific uncertain tradeoff. Otherwise "
            "retain current. A fresh current replay failure does not erase a "
            "historical success. Different one-shot scripts confound causal attribution; "
            "describe that uncertainty. A tie is not improvement by itself: cite a concrete "
            "reusable change and behavioral/code evidence. When both versions have zero "
            "successes, reason must state the hypothesis actually tested, cite the executed "
            "action/phase evidence and outcome, and explicitly leave capability success "
            "unverified. Additional helpers or a less restrictive guard alone are not "
            "demonstrated improvement. Policy stdout may implement a "
            "different check from the declared task predicate; compare both explicitly. "
            "For an exploration weakness, require evidence that the candidate's "
            "observation/manipulation path was actually exercised. If all relevant "
            "trials still stop before arm motion, retain current and request an "
            "executable remedy; do not describe unexecuted code as a learned capability.",
            {"hypothesis": proposal.get("hypothesis"), "no_regression": no_regression,
             "regression": regression, "tasks": focused + replay, "focused_count": len(focused),
             "historical_replay_anchors": anchors, "current_toolbox": _contents(current),
             "candidate_toolbox": _contents(candidate), "changes": attempt_record["changes"],
             "current": [_source_evidence(r) for r in current_results],
             "candidate": [_source_evidence(r) for r in candidate_results]})
        acceptance = _discovery_acceptance(decision, regression)
        take = acceptance["accepted"]
        attempt_record.update(accepted=take, agent_accept=decision.get("accept"),
                              no_regression=no_regression, regression=regression,
                              reason=decision.get("reason"),
                              gate_reason=acceptance["gate_reason"],
                              unresolved_history_acknowledged=acceptance["unresolved_history_acknowledged"],
                              history_reason=acceptance["history_reason"],
                              demonstrated_improvement=acceptance["demonstrated_improvement"],
                              baseline_results=[_source_evidence(r) for r in current_results],
                              results=[_source_evidence(r) for r in candidate_results])
        revision_history.append(attempt_record)
        _save(rev_dir / "accepted.json", attempt_record)
        if take:
            current, current_results = candidate, candidate_results
            accepted.append(proposal.get("hypothesis", ""))
            bank = _remember_successes(bank, focused + replay, current_results, current, cycle,
                                       f"accepted_revision_{revision:02d}")
            _save(rev_dir / "capability_bank.json", bank)
            _save(root / "capability_bank.json", bank)
        _progress(run_dir, "source_revision_complete", member=index, cycle=cycle + 1,
                  revision=revision + 1, accepted=take,
                  focused_successes=_score(current_results[:len(focused)])[0])
    summary = {"cycle": cycle, "weakness": focus.get("weakness"),
               "probe_successes": _score(probe_results)[0],
               "focused_successes": _score(current_results[:len(focused)])[0],
               "replay_successes": _score(current_results[len(focused):])[0],
               "replay_tasks": len(replay), "accepted_hypotheses": accepted,
               "capability_bank_size": len(bank),
               "current_version_confirmed_capabilities": sum(
                   record["latest_observation"]["toolbox_path"] == str(current.resolve()) and
                   bool(record["latest_observation"]["outcome"]["success"]) for record in bank)}
    history = {"cycle": cycle, "weakness": focus.get("weakness"),
               "attempts": _history_attempts(revision_history)}
    if coding_summary is not None:
        from .coding import session_history
        summary["coding"] = coding_summary
        history["coding"] = coding_summary
        history["coding_history"] = session_history(cycle_dir / "coding")
    _save(cycle_dir / "summary.json", summary)
    _save(cycle_dir / "research_history.json", history)
    _save(cycle_dir / "capability_bank.json", bank)
    _save(root / "capability_bank.json", bank)
    _snapshot(current, cycle_dir / "final_toolbox")
    return {"current": current, "bank": bank, "summaries": summaries + [summary],
            "histories": histories + [history], "replay_history": state["replay_history"] +
            [[record["task_sha256"] for record in replay_records]]}


def discover(runner, agent, run_dir: Path, config: dict) -> dict:
    """Independent curricula, interleaved at cycle boundaries, without target access."""
    if callable(getattr(agent, "set_phase", None)):
        agent.set_phase("stage1")
    run_dir = Path(run_dir).resolve()
    with _lock(run_dir):
        cfg = _initialize(run_dir, config)
        done = _json(run_dir / "discovery.json")
        if done:
            _check_frozen(run_dir / "frozen" / "stage1")
            return done
        if cfg.get("stage1_parent"):
            from .continuation import inherit_stage1
            return inherit_stage1(run_dir, cfg)
        _progress(run_dir, "discovery_started", population=cfg["population"], cycles=cfg["cycles"])
        states = [{"current": _snapshot(run_dir / "frozen" / "initial",
                                        run_dir / "discovery" / f"member_{index:02d}" / "initial"),
                   "bank": [], "summaries": [], "histories": [], "replay_history": []}
                  for index in range(cfg["population"])]
        for cycle in range(cfg["cycles"]):
            # All peers use the same previous-cycle boundary. Reconstructing in
            # this order on resume cannot expose a later member/cycle summary.
            boundary = [copy.deepcopy(state["summaries"]) for state in states]
            for index in range(cfg["population"]):
                peers = [{"member": peer, "summaries": summaries[-2:]}
                         for peer, summaries in enumerate(boundary) if peer != index and summaries]
                states[index] = _discover_cycle(runner, agent, run_dir, cfg, index, cycle, states[index], peers)
        members = []
        for index, state in enumerate(states):
            frozen = _freeze(state["current"], run_dir / "discovery" / f"member_{index:02d}" / "frozen")
            members.append({"member": index, "toolbox": str(frozen), "summaries": state["summaries"]})
        selection = _json(run_dir / "selection_tasks.json")
        auxiliary = _json(run_dir / "auxiliary_selection_tasks.json")
        _check_selection_sets(run_dir, _json(run_dir / "manifest.json"))
        for member in members:
            _progress(run_dir, "source_selection_started", member=member["member"], scenes=len(selection))
            scores = _suite(runner, selection, Path(member["toolbox"]),
                            run_dir / "selection" / f"member_{member['member']:02d}")
            auxiliary_scores = _suite(runner, auxiliary, Path(member["toolbox"]),
                                      run_dir / "auxiliary_selection" / f"member_{member['member']:02d}")
            member["successes"], member["steps"] = _score(scores)
            member["auxiliary_successes"] = _score(auxiliary_scores)[0]
            valid = [result for result in scores if result.get("status") in ("success", "failed")
                     and result.get("actual_started") is True]
            member["valid_executions"] = len(valid)
            member["selection_valid_executions"] = len(valid) + sum(
                result.get("status") in ("success", "failed") and result.get("actual_started") is True
                for result in auxiliary_scores)
            # Kept as a diagnostic only: fast failure is not a selection advantage.
            member["mean_valid_steps"] = sum(int(result.get("steps") or 0) for result in valid) / len(valid) \
                if valid else None
            member["selection_results"] = [_selection_record(task, result, trial)
                for trial, (task, result) in enumerate(zip(selection, scores))]
            member["auxiliary_results"] = [_selection_record(task, result, trial)
                for trial, (task, result) in enumerate(zip(auxiliary, auxiliary_scores))]
            member["capability_bank_size"] = len(states[member["member"]]["bank"])
        eligible = [member for member in members if member["valid_executions"] > 0]
        if not eligible:
            raise ResearchError("No source selection trial completed a valid physics policy; "
                                "discovery is not complete and target adaptation must not start")
        winner = min(eligible, key=lambda item: (-item["successes"], -item["auxiliary_successes"],
                                               -item["selection_valid_executions"], item["member"]))
        destination = _freeze(Path(winner["toolbox"]), run_dir / "frozen" / "stage1")
        result = {"winner": winner["member"], "toolbox": str(destination), "members": members,
                  "process_complete": True, "full_task_successes": winner["successes"],
                  "full_task_verified": winner["successes"] > 0,
                  "auxiliary_successes": winner["auxiliary_successes"],
                  "capability_bank_size": winner["capability_bank_size"],
                  "selection_tasks": len(selection), "auxiliary_selection_tasks": len(auxiliary),
                  "selection_rule": "full_successes, auxiliary_successes, valid_executions, member_index",
                  "agent_count": getattr(agent, "count", None),
                  "agent_count_note": "Cumulative budget-charged calls at Stage 1 completion; network failures excluded."}
        _save(run_dir / "discovery.json", result)
        _progress(run_dir, "discovery_complete", winner=winner["member"],
                  full_task_successes=winner["successes"], auxiliary_successes=winner["auxiliary_successes"],
                  full_task_verified=result["full_task_verified"])
        return result


def _selection_record(task: dict, result: dict, trial: int) -> dict:
    evidence = _public(result)["evidence"]
    return {"trial": trial, "task": task["name"], "task_sha256": _digest(task),
            "seed": task["scene"].get("seed"), "success": bool(result.get("success")),
            "status": result.get("status"), "actual_started": result.get("actual_started"),
            "output_dir": result.get("output_dir"), "video": evidence.get("video"),
            "phase_images": evidence.get("phase_images", []), "reason": evidence.get("reason")}


def _calibration(value: dict, profile: str = "basic") -> dict:
    if not isinstance(value, dict):
        raise ResearchError("source_config must be an object")
    bounds = {"mass_scale": (0.5, 2.0), "friction_scale": FRICTION_SCALE_BOUNDS,
              "controller_duration_scale": (0.5, 2.0)}
    if profile == "actuator_v1":
        bounds.update(ACTUATOR_CALIBRATION_BOUNDS)
    elif profile != "basic":
        raise ResearchError("Unsupported source calibration profile")
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


def _actuator_calibration_help(cfg: dict) -> str:
    if cfg.get("source_calibration_profile", "basic") != "actuator_v1":
        return ""
    return (
        "The actuator_v1 profile also permits source-only arm_settle_s [0,0.5]: "
        "append this many seconds holding the final arm target after each successful-IK "
        "move, before reporting its state; do not stretch the arm trajectory. "
        "gripper_stiffness_scale [0.25,1] and gripper_damping_scale [1,4] multiply "
        "the source finger position-servo stiffness and velocity damping independently "
        "(defaults 1); they preserve the requested width, linear target ramp, command "
        "duration and per-finger force cap. "
        "These parameters do not alter the arm gains, cameras, objects, target engine "
        "or policy API. arm_settle_s defaults to 0 and does not apply to gripper or wait. "
        "Use the public command-response evidence to justify a change and inspect its "
        "fixed-policy replay; an aperture or timing match alone does not establish "
        "contact equivalence or task-relevant alignment. "
    )


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


def _alignment_summary(alignment: dict) -> dict:
    """Carry calibration decisions and public outcomes across task iterations."""
    return {**{key: copy.deepcopy(value) for key, value in alignment.items() if key != "history"},
            "history": [{**{key: copy.deepcopy(value) for key, value in entry.items() if key != "source"},
                         "source": _outcome_summary(entry["source"])}
                        for entry in alignment["history"]]}


def adapt(runner, agent, run_dir: Path, config: dict) -> dict:
    """Adapt each task independently; target slots are never a repair score search."""
    if callable(getattr(agent, "set_phase", None)):
        agent.set_phase("stage2")
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
                    nominal_task = with_placement_contract(
                        nominal_task, cfg.get("placement_contract", "legacy_geometry"))
                    _save(reconstruction_path, nominal_task)
                source = None
                alignment = {"context_version": 1, "status": "fixed_policy_unavailable",
                             "updates_applied": 0, "max_updates": cfg["align_attempts"],
                             "last_assessment": None, "history": []}
                if fixed_script is not None:
                    source = _run(runner, "mujoco", nominal_task, current,
                                  trial_dir / "alignment" / "initial", fixed_script, source_config)
                    alignment["status"] = "update_limit_reached_unconfirmed"
                    alignment["history"].append({"step": 0,
                        "source_config": copy.deepcopy(source_config), "proposal": None,
                        "source": _public(source)})
                    for attempt in range(cfg["align_attempts"]):
                        align_dir = trial_dir / "alignment" / f"attempt_{attempt:02d}"
                        decision = _ask(agent, align_dir / "decision.json", "align",
                            "Compare target and fixed-policy source evidence. Return "
                            '{"aligned":bool,"source_config":{...},"reason":str}. '
                            "If aligned=true keep the model. Otherwise propose source calibration only: "
                            "mass_scale [0.5,2], "
                            f"friction_scale [{FRICTION_SCALE_BOUNDS[0]:g},{FRICTION_SCALE_BOUNDS[1]:g}], controller_duration_scale "
                            "[0.5,2], camera_offset [dx,dy,dz] each +/-0.05 m. "
                            "mass_scale multiplies plate masses only, not robot link masses or controller gains. "
                            "It changes object inertia and contact loads, not unloaded robot dynamics. "
                            "controller_duration_scale multiplies source move and gripper command durations; "
                            "wait durations, servo gains and the physics timestep remain unchanged. camera_offset translates "
                            "both source cameras in their mounting coordinates; returned camera calibration "
                            "remains consistent with those poses. friction_scale uniformly "
                            "scales source sliding friction for the whole scene (including robot contact "
                            "geometries), so it changes the effective contact friction. "
                            f"{_actuator_calibration_help(cfg)}"
                            "Missing keys preserve "
                            "the current values. Never change this script or toolbox during alignment. "
                            "toolbox contains the exact frozen implementation and skill notes used by "
                            "both rollouts; inspect its helper semantics when interpreting motion or "
                            "termination evidence. It is read-only evidence, not a repair proposal. "
                            "Use image and motion evidence; alignment is not success optimization. "
                            "First identify the outcome-relevant failure or behavior and its observed "
                            "phase. Judge whether the source reproduces the object response, triggering "
                            "observations and policy decisions relevant to that outcome. Identical "
                            "command counts, timing or intermediate tracking samples are not required "
                            "when the remaining differences do not change that behavior or undermine "
                            "the proposed repair tests. Explain their relevance in reason rather than "
                            "optimizing trace similarity without an outcome-related rationale. An "
                            "alignment judgment covers only the observed behavior: explicitly retain "
                            "uncertainty about unexercised contact or placement and other mismatches. "
                            "alignment_history contains this trial's prior calibration proposals and "
                            "their completed public replays. Compare observed effects before repeating "
                            "or reversing an update. Equal failure labels do not establish alignment.",
                            {"task": nominal_task["description"], "script": fixed_script,
                             "toolbox": _contents(current),
                             "source_config": source_config, "target": _public(target),
                             "source": _public(source), "task_history": trial_summaries,
                             "alignment_history": alignment["history"]})
                        alignment["last_assessment"] = {
                            "assessed_step": len(alignment["history"]) - 1,
                            "response": copy.deepcopy(decision)}
                        if decision.get("aligned") is True:
                            alignment["status"] = "agent_confirmed_aligned"
                            break
                        profile = cfg.get("source_calibration_profile", "basic")
                        update = _calibration(decision.get("source_config", {}), profile)
                        source_config = _calibration({**source_config, **update}, profile)
                        source = _run(runner, "mujoco", nominal_task, current,
                                      align_dir / "replay", fixed_script, source_config)
                        alignment["updates_applied"] += 1
                        alignment["history"].append({"step": alignment["updates_applied"],
                            "source_config": copy.deepcopy(source_config),
                            "proposal": copy.deepcopy(decision), "source": _public(source)})
                else:
                    _save(trial_dir / "alignment" / "unavailable.json", {
                        "reason": "No policy was produced/executed; no fixed-script replay is possible."})
                _save(trial_dir / "alignment" / "history.json", alignment)
                _save(trial_dir / "source_config.json", source_config)
                hypotheses = _ask(agent, trial_dir / "hypotheses.json", "repair_hypotheses",
                    f"Propose exactly {cfg['repairs']} distinct reusable toolbox repair hypotheses "
                    'based on target and current calibrated source evidence. Return {"hypotheses":[str,...]}. '
                    "Keep alternatives independent; do not modify the simulator further. "
                    "Use task_history to avoid repeating rejected repairs without new evidence. "
                    "Consult alignment status and history: exhausting the calibration budget does "
                    "not confirm behavioral alignment, and a final replay may not have been assessed.",
                    {"task": nominal_task["description"], "toolbox": _contents(current),
                     "script": fixed_script, "target": _public(target),
                     "source": _public(source) if source else None, "source_config": source_config,
                     "task_history": trial_summaries, "alignment": alignment})
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
                         "task_history": trial_summaries, "alignment": alignment})
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
                    "will be frozen and shared by current and all repair candidates. "
                    "The supplied source_config is applied after these scene overrides. With the "
                    "pinned MuJoCo source assets, plate/table sliding friction is max(plate_friction, "
                    "0.7) and plate/finger sliding friction is max(plate_friction,1.0), then both "
                    "are multiplied by friction_scale. Material-friction changes below those "
                    "contact floors do not vary effective contact friction. Choose effective "
                    "physical contrasts, such as mass or friction beyond the relevant floor, "
                    "and describe the actual variation rather than labeling unchanged contact "
                    "friction as a robustness test. " + TASK_SPEC_HELP,
                    {"task_id": task_id, "description": nominal_task["description"],
                     "source_scene_template": nominal_task["scene"],
                     "source_config": copy.deepcopy(source_config),
                     "required_predicate": nominal_task.get("predicate"),
                     "suggested_source_seeds": [cfg["seed"] + 40000 + task_index * 1000 +
                         trial * 100 + j for j in range(cfg["screen_tasks"])]})
                generalization = _tasks_from_response(screen_proposal, cfg["screen_tasks"])
                _record_scene_variations(screen_proposal, generalization, trial_dir / "screen_scene_variations.json")
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
                    "target mismatch and robustness; you may reject a high-scoring implausible repair. "
                    "Full-task screen success is useful evidence, not a mandatory selection gate. "
                    "A local correction may be selected when replay and generalization evidence "
                    "support its predicted removal of a specific target failure, later failures are "
                    "separately identified, and material regressions are addressed. Explain that "
                    "causal evidence and remaining uncertainty; extra arm motion or earlier stopping "
                    "alone is not an improvement. Failed task outcomes must remain recorded as failures. "
                    "Use the calibration history and its assessment status to judge remaining mismatch; "
                    "a completed calibration budget is not evidence that alignment succeeded.",
                    {"hypotheses": hypothesis_map, "candidates": candidate_records,
                     "available_choices": list(candidates), "task_history": trial_summaries,
                     "source_task": nominal_task, "fixed_script": fixed_script,
                     "generalization_tasks": generalization,
                     "toolboxes": {key: _contents(value) for key, value in candidates.items()},
                     "target": _public(target), "source_config": source_config, "screen": evidence,
                     "alignment": alignment})
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
                           "alignment": _alignment_summary(alignment),
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
    if callable(getattr(agent, "set_phase", None)):
        agent.set_phase("evaluation")
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


def _stage1_report(run_dir: Path, manifest: dict, calls: int) -> dict:
    discovery = _json(run_dir / "discovery.json", {})
    tasks = _json(run_dir / "selection_tasks.json", [])
    auxiliary = _json(run_dir / "auxiliary_selection_tasks.json", [])
    members = []
    lineage = discovery.get("inherited_stage1")
    for index in range(manifest.get("config", {}).get("population", DEFAULTS["population"])):
        member = {"member": index}
        inherited = next((row for row in discovery.get("members", []) if row["member"] == index), None) if lineage else None
        for label, suite, frozen in (("selection_results", "selection", tasks),
                                     ("auxiliary_results", "auxiliary_selection", auxiliary)):
            if inherited:
                member[label] = copy.deepcopy(inherited[label])
                continue
            records = []
            for trial, task in enumerate(frozen):
                out = run_dir / suite / f"member_{index:02d}" / f"{trial:03d}"
                outcome = _json(out / "result.json")
                records.append(_selection_record(task, outcome or {"status": "pending", "output_dir": str(out)}, trial))
            member[label] = records
        member["full_task_successes"] = sum(record["success"] for record in member["selection_results"])
        member["auxiliary_successes"] = sum(record["success"] for record in member["auxiliary_results"])
        bank = _json(run_dir / "discovery" / f"member_{index:02d}" / "capability_bank.json", [])
        member["capability_bank_size"] = inherited["capability_bank_size"] if inherited else len(bank)
        if inherited:
            member["legacy_predicate_successes"] = inherited["legacy_predicate_successes"]
        members.append(member)
    winner = discovery.get("winner")
    selected = next((member for member in members if member["member"] == winner), None)
    full = selected["full_task_successes"] if selected else 0
    strict = manifest["config"].get("placement_contract") == "released_placement_v1"
    return {"process_complete": bool(discovery), "winner": winner,
            "full_task_verified": bool(discovery) and full > 0, "full_task_successes": full,
            "full_task_predicate_verified": bool(discovery) and full > 0,
            "full_task_verification_scope": "released_placement_v1" if strict else "frozen_task_predicates",
            "released_placement_verified": bool(discovery) and full > 0 if strict else None,
            "auxiliary_successes": selected["auxiliary_successes"] if selected else None,
            "capability_bank_size": selected["capability_bank_size"] if selected else None,
            "selection_tasks_per_member": len(tasks), "auxiliary_tasks_per_member": len(auxiliary),
            "agent_calls": discovery.get("agent_count", calls),
            "agent_calls_note": "Cumulative budget-charged calls at Stage 1 completion; later phases excluded.",
            "members": members, "inherited_stage1": lineage}


def report(run_dir: Path) -> dict:
    """Write an honest partial-or-complete report without optimizer feedback."""
    run_dir = Path(run_dir).resolve()
    manifest = _json(run_dir / "manifest.json")
    if not manifest:
        raise ResearchError(f"No experiment manifest in {run_dir}")
    if manifest.get('config', {}).get('protocol') == 'supplied_skill_stage2_v1':
        from .stage2 import report as supplied_report
        return supplied_report(run_dir)
    from .coding import coding_report
    coding = coding_report(run_dir)
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
    decisions, evidence_turns = 0, 0
    for path in run_dir.rglob("*.json"):
        record = _json(path)
        if isinstance(record, dict) and "request_sha256" in record:
            if path.parent.suffix == ".context":
                evidence_turns += 1
            else:
                decisions += 1
    calls = list((run_dir / "agent_calls").glob("*/request.json"))
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    models = set()
    network_exempt = 0
    for path in calls:
        request = _json(path, {})
        if request.get("model"):
            models.add(request["model"])
        call_result = _json(path.parent / "result.json", {})
        network_exempt += call_result.get("budget_charged") is False
        for entry in call_result.get("usage", []):
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
                                "alignment_status": summary.get("alignment", {}).get("status", "not_recorded"),
                                "calibration_updates": summary.get("alignment", {}).get("updates_applied"),
                                "source_target_success_disagreement": mismatch,
                                "selection_record": str(path.parent / "repair_selection.json")})
        traces[task_id] = task_trials
    source_statuses = Counter()
    source_attempt_statuses = Counter()
    source_activity = Counter()
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
        from .evidence import action_summary
        activity = outcome.get('action_summary') or action_summary(outcome.get('public_evidence', {}))
        source_activity[activity['classification']] += 1
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
              "completion_scope": "Scheduled protocol and execution provenance, not capability or release verification.",
              "success_definition": "Frozen task predicates; gripper opening, withdrawal and contact release require separate audit.",
              "source_rollout_statuses": dict(source_statuses),
              "source_rollout_activity": dict(source_activity),
              "source_attempt_statuses": dict(source_attempt_statuses),
              "source_valid_physics_rollouts": source_valid,
              "target_rollout_statuses": dict(target_statuses),
              "agent_calls": len(calls) if calls else decisions + coding["coding_turns"],
              "agent_budget_used": len(calls) - network_exempt if calls else decisions + coding["coding_turns"],
              "network_failures_exempt": network_exempt,
              "research_decisions": decisions, "research_dialogue_turns": evidence_turns,
              "token_usage": usage, "models": sorted(models),
              "adaptation_trace": traces, "source_target_success_disagreements": mismatch_count,
              "coding": coding}
    result["stage1"] = _stage1_report(run_dir, manifest, result["agent_budget_used"])
    lineage = result["stage1"].get("inherited_stage1") or {}
    result["agent_budget_inherited"] = lineage.get("parent_agent_budget_used", 0)
    result["agent_budget_total_used"] = result["agent_budget_inherited"] + result["agent_budget_used"]
    strict = manifest["config"].get("placement_contract") == "released_placement_v1"
    if strict:
        result["success_definition"] = (
            "released_placement_v1: lift, full containment and distractor exclusion, plus measured opening, "
            "withdrawal, hand/object separation and a complete settled post-opening window.")
    result["agent_calls_note"] = ("All recorded attempts, including budget-exempt network failures; "
                                  "token totals include all provider-reported usage.")
    _save(run_dir / "report.json", result)
    stage1 = result["stage1"]
    text = ["# SimEX 本地初步复现报告", "",
            "Stage 1 流程：" + ("已完成。" if stage1["process_complete"] else "未完成。"),
            (f"Stage 1 所选成员：{stage1['winner']}；入箱判据通过 {stage1['full_task_successes']} / "
             f"{stage1['selection_tasks_per_member']}；共同基础能力选择任务成功 "
             f"{stage1['auxiliary_successes']} / {stage1['auxiliary_tasks_per_member']}。"
             if stage1["winner"] is not None else "Stage 1 尚未完成成员选拔；完整任务判据通过率待评估。"),
            f"截至 Stage 1 的累计模型预算用量：{stage1['agent_calls']}；所选成员历史成功任务数："
            f"{stage1['capability_bank_size']}。历史成功任务包含自生成子任务，不能等同于完整入箱。",
            ("Stage 1 已通过版本化释放判据；录像仍保留供人工复核。" if strict else
             "Stage 1 已出现入箱判据通过；释放与撤手仍需单独核对。")
            if stage1["full_task_predicate_verified"] else "尚未验证完整入箱能力。",
            ("本运行采用 released_placement_v1：开爪至少 6 cm、撤手至少 5 cm、TCP 与盘心分离"
             "至少盘子半径加 5 cm，最后半秒持续入箱、张爪、低速且位置漂移不超过 5 mm。"
             if strict else
             "原入箱判据检查抬升、盘子包围体入箱和末速度，不检查打开夹爪、撤手或脱离接触。"
             "以下得分不等同于完成释放放置；另用 scripts/audit_placement.py 和录像核对。"),
            "", "| 成员 | 类型 | 场景/种子 | 状态 | 判据通过 | 视频/记录 |",
            "|---:|---|---|---|---|---|"]
    if lineage:
        text[2:2] = [
            f"Stage 1 继承自 `{lineage['parent_run']}`：仅对既有选拔轨迹重新评分，没有重新训练或生成源策略。",
            f"原评分选择成员 {lineage['legacy_winner']}，释放判据选择成员 {lineage['release_winner']}。"
            f"继承的模型预算用量为 {lineage['parent_agent_budget_used']}，后续剩余预算为 "
            f"{lineage['continuation_model_budget']}。", ""]
    for member in stage1["members"]:
        for label, kind in (("selection_results", "入箱判据"), ("auxiliary_results", "基础能力")):
            for record in member[label]:
                video = record.get("video")
                artifact = f"[视频]({video})" if video else f"[记录]({record['output_dir']})"
                text.append(f"| {member['member']} | {kind} | {record['task']} / {record['seed']} | "
                            f"{record['status']} | "
                            f"{'待评估' if record['status'] == 'pending' else '是' if record['success'] else '否'} | {artifact} |")
    text.extend(["", f"目标仿真：已完成 {result['target_completed']} / {result['target_scheduled']} 个排定试验。",
            "结果状态：" + ("流程及物理执行证据完整。" if result["complete"] else
                           "预算已执行完，但有效物理执行证据不足，不能判定工程验收完整。"
                           if protocol_complete else "未完成；不把缺失试验计为实测失败。"), "",
            "| 方法 | 任务 | 判据通过/排定 | 已完成 | 判据通过率 |",
            "|---|---|---:|---:|---:|"])
    for row in rows:
        rate = "未完成" if row["success_rate"] is None else f"{row['success_rate']:.1%}"
        text.append(f"| {row['method']} | {row['task_id']} | {row['successes']}/{row['scheduled']} | "
                    f"{row['completed']} | {rate} |")
    text.extend(["", "本实验采用单臂 Franka Panda、三个改编搬盘任务、缩小的种群与迭代预算，"
                 "每轮候选及筛选场景数量见 manifest.json。研究运行次数为 1；"
                 f"记录模型为 {', '.join(sorted(models)) or manifest['config'].get('model', '未记录')}，"
                 "阶段推理等级和超时见 manifest.json。",
                 "本报告不将完整执行等同于方法改善，也不宣称复现论文的 94% 成功率。",
                 "", "每次目标记录、固定脚本源重放、校准配置、候选证据和代理选择理由保存在 adaptation/；"
                 "最终评估独立保存在 evaluation/，不再反馈给优化代理。",
                 "执行失败、超时及非法策略消耗排定试验。启动前基础设施失败允许修复后恢复；"
                 "执行中的基础设施中断锁定原试验、保持未完成，不作为普通失败计入成功率，也不自动重跑。"
                 "需要恢复原始有效结果，或另建独立研究运行，不能以重试筛选成功。"])
    text.extend(["", f"记录模型连接尝试 {result['agent_calls']} 次；研究调用预算用量 "
                 f"{result['agent_budget_used']} 次；网络失败豁免 {network_exempt} 次；"
                 f"代理报告 token 用量：{usage}。",
                 f"已保存研究决策 {decisions} 次、证据对话轮次 {evidence_turns} 次。"
                 "普通研究决策最多 20 轮交互；交互式编码使用独立的周期额度。"
                 "网络失败最多自动重连 5 次，不扣模型调用预算，"
                 "不占交互轮次，也不增加物理试验次数。",
                 f"源仿真最终状态计数：{dict(source_statuses)}；含重启前基础设施失败的尝试计数："
                 f"{dict(source_attempt_statuses)}。有效源物理执行 {source_valid} 次。",
                 f"源试验动作分类：{dict(source_activity)}。no_arm_commands 表示没有发出机械臂"
                 "运动指令；等待或仅开合夹爪不代表尝试了抓取。动作发生也不等于抓取成功。",
                 f"目标试验状态计数：{dict(target_statuses)}。非法策略、生成失败和基础设施失败"
                 "单独列出，不能用它们证明抓取方法或模拟器链路已经有效运行。",
                 f"初始固定脚本重放与目标成功判定不同的试验：{mismatch_count} 次。"
                 "该统计只反映结果差异；接触和轨迹是否匹配需要查看图像及代理校准理由。"])
    if coding["sessions"]:
        text.extend(["", f"交互式编码：交互 {coding['coding_turns']} 轮，草稿 {coding['drafts']} 份，"
                     f"检查失败 {coding['validation_failures']} 次，完整测试批次 "
                     f"{coding['completed_test_batches']} / {coding['test_batches']}，"
                     f"接受 {coding['accepted']} 次。"])
        for session in coding["sessions"]:
            text.append(f"- {session['directory']}：{session['status']}；"
                        f"结束原因 {session['end_reason'] or '尚未结束'}；"
                        f"保留版本 {session['accepted_draft']}。")
    for task_id, trace in traces.items():
        if trace:
            outcomes = " → ".join("成功" if item["success"] else "失败" for item in trace)
            selections = ", ".join(item["selected"] for item in trace)
            alignments = ", ".join(item["alignment_status"] for item in trace)
            text.append(f"- {task_id} 目标试验结果：{outcomes}；各轮选择：{selections}；"
                        f"对齐评估状态：{alignments}。")
    (run_dir / "report.md").write_text("\n".join(text) + "\n")
    return result
