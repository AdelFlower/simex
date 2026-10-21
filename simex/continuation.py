"""Continue learned Stage 1 under a corrected, target-blind placement contract."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from .errors import ResearchError
from .release import RELEASE_CONTRACT, release_evidence
from .scoring import score
from .tasks import with_placement_contract


def read(path):
    return json.loads(Path(path).read_text())


def regrade_saved_rollout(directory, task):
    """Score saved measurements only. Never execute or overwrite the old trial."""
    directory = Path(directory)
    result = read(directory / "result.json")
    manifest = read(directory / "manifest.json")
    execution = read(directory / "execution.json")
    request = read(directory / "research_request.json")
    if (manifest.get("engine") != "mujoco" or manifest.get("fixed_script") is not False
            or execution.get("actual_started") is not True or result.get("actual_started") is not True
            or result.get("status") not in ("success", "failed") or request.get("task") != task):
        raise ResearchError(f"Source regrading requires a completed original fresh-policy trial: {directory}")
    metrics = read(directory / "private/metrics.json")
    samples = [json.loads(line) for line in (directory / "private/object_trace.jsonl").read_text().splitlines()]
    evidence = result.get("public_evidence", {})
    actions = evidence.get("actions", [])
    robot = actions[-1].get("after", {}) if actions else {}
    metrics["release_evidence"] = release_evidence(task["scene"], metrics, robot, actions, samples)
    strict = with_placement_contract(task, RELEASE_CONTRACT)
    return bool(score(strict, metrics)), metrics["release_evidence"], result


def inherit_stage1(run_dir, cfg):
    """Select from all saved source trials with release scoring before any target policy.

    The parent remains immutable, including its legacy winner and scores. The
    child stores explicit lineage and regrading records, rather than pretending
    that source policies were regenerated under the corrected contract.
    """
    from . import research as r

    run_dir, parent = Path(run_dir).resolve(), Path(cfg["stage1_parent"]).resolve()
    if cfg.get("placement_contract") != RELEASE_CONTRACT:
        raise ResearchError("Stage 1 continuation requires released_placement_v1")
    if parent == run_dir or run_dir.is_relative_to(parent) or parent.is_relative_to(run_dir):
        raise ResearchError("Parent and continuation must be separate, non-nested runs")
    with r._lock(parent):
        manifest, previous = read(parent / "manifest.json"), read(parent / "discovery.json")
        if previous.get("inherited_stage1") or previous.get("process_complete") is not True:
            raise ResearchError("Continuation requires a completed original Stage 1")
        if any(slot.get("actual_started") is True for slot in read(parent / "target_ledger.json")["slots"].values()):
            raise ResearchError("Cannot reselect Stage 1 after a parent target policy executed")
        for path in list((parent / "adaptation").glob("*/trial_*/target/execution.json")) + list(
                (parent / "evaluation").glob("*/*/trial_*/execution.json")):
            if read(path).get("actual_started") is True:
                raise ResearchError("Cannot reselect Stage 1 after a parent target policy executed")
        parent_cfg = manifest["config"]
        source_keys = ("population", "cycles", "probes", "focused", "revisions", "replay",
                       "seed", "selection_trials", "discovery_mode", "coding_max_turns", "coding_max_tests")
        if any(cfg.get(key) != parent_cfg.get(key) for key in source_keys):
            raise ResearchError("Continuation must preserve the parent training protocol and complete selection set")
        calls = list((parent / "agent_calls").glob("*/request.json"))
        charged = sum(not (p.parent / "result.json").exists()
                      or read(p.parent / "result.json").get("budget_charged") is not False for p in calls)
        remaining = int(parent_cfg.get("max_agent_calls", 800)) - charged
        if int(cfg.get("max_agent_calls", 800)) > remaining:
            raise ResearchError("Continuation would exceed the parent's remaining model-call budget")
        source_tasks, target_tasks = read(parent / "selection_tasks.json"), read(run_dir / "selection_tasks.json")
        if target_tasks != [with_placement_contract(task, RELEASE_CONTRACT) for task in source_tasks]:
            raise ResearchError("Continuation changed source selection scenes or task identities")
        auxiliary = read(parent / "auxiliary_selection_tasks.json")
        if read(run_dir / "auxiliary_selection_tasks.json") != auxiliary:
            raise ResearchError("Continuation changed the auxiliary selection set")
        if len(previous.get("members", [])) != cfg["population"]:
            raise ResearchError("Parent source population is incomplete")
        members, private_records = [], []
        for original in previous["members"]:
            index = original["member"]
            source = parent / "discovery" / f"member_{index:02d}" / "frozen"
            r._check_frozen(source)
            frozen = r._freeze(source, run_dir / "discovery" / f"member_{index:02d}" / "frozen")
            records = []
            for trial, task in enumerate(source_tasks):
                directory = parent / "selection" / f"member_{index:02d}" / f"{trial:03d}"
                passed, detail, result = regrade_saved_rollout(directory, task)
                evidence = result.get("public_evidence", {})
                records.append({"trial": trial, "task": task["name"], "seed": task["scene"]["seed"],
                    "status": "success" if passed else "failed", "success": passed,
                    "actual_started": True, "output_dir": str(directory),
                    "video": evidence.get("video"), "phase_images": evidence.get("keyframes", []),
                    "legacy_success": bool(result["success"]), "legacy_status": result["status"],
                    "score_source": "saved_source_trace_regrade", "placement_contract": RELEASE_CONTRACT})
                private_records.append({"member": index, "trial": trial, "rollout": str(directory),
                                        "success": passed, "release_evidence": detail})
            auxiliary_records = copy.deepcopy(original["auxiliary_results"])
            if len(auxiliary_records) != len(auxiliary):
                raise ResearchError("Parent auxiliary selection is incomplete")
            for trial, task in enumerate(auxiliary):
                out = parent / "auxiliary_selection" / f"member_{index:02d}" / f"{trial:03d}"
                result, request = read(out / "result.json"), read(out / "research_request.json")
                if (request.get("task") != task or request.get("engine") != "mujoco"
                        or result.get("actual_started") is not True
                        or result.get("status") not in ("success", "failed")
                        or bool(result["success"]) != bool(auxiliary_records[trial]["success"])):
                    raise ResearchError("Parent auxiliary execution evidence is incomplete")
            member = copy.deepcopy(original)
            member.update(toolbox=str(frozen), selection_results=records,
                          auxiliary_results=auxiliary_records,
                          legacy_predicate_successes=original["successes"],
                          successes=sum(row["success"] for row in records))
            members.append(member)
        winner = min(members, key=lambda m: (-m["successes"], -m["auxiliary_successes"],
                                             -m["selection_valid_executions"], m["member"]))
        destination = r._freeze(Path(winner["toolbox"]), run_dir / "frozen" / "stage1")
        lineage = {"parent_run": str(parent), "parent_discovery": str(parent / "discovery.json"),
                   "imported_at": time.time(), "placement_contract": RELEASE_CONTRACT,
                   "parent_target_policy_executions": 0, "parent_agent_budget_used": charged,
                   "parent_total_model_budget": parent_cfg.get("max_agent_calls", 800),
                   "continuation_model_budget": cfg.get("max_agent_calls", 800),
                   "legacy_winner": previous["winner"], "release_winner": winner["member"],
                   "source_selection_physics_rollouts": len(source_tasks) * len(members),
                   "source_auxiliary_physics_rollouts": len(auxiliary) * len(members),
                   "selection_policy": "Regrade every existing frozen source selection trial; no resampling.",
                   "training_scope": "Parent curricula used their original predicates; only selection is regraded."}
        result = {"winner": winner["member"], "toolbox": str(destination), "members": members,
                  "process_complete": True, "full_task_successes": winner["successes"],
                  "full_task_verified": winner["successes"] > 0,
                  "auxiliary_successes": winner["auxiliary_successes"],
                  "capability_bank_size": winner["capability_bank_size"],
                  "selection_tasks": len(source_tasks), "auxiliary_selection_tasks": len(auxiliary),
                  "selection_rule": "release_successes, auxiliary_successes, valid_executions, member_index",
                  "agent_count": previous.get("agent_count"), "inherited_stage1": lineage}
        r._save(run_dir / "private" / "source_selection_regrade.json", private_records)
        r._save(run_dir / "stage1_import.json", lineage)
        r._save(run_dir / "discovery.json", result)
        r._progress(run_dir, "stage1_inherited", winner=winner["member"],
                    release_successes=winner["successes"], parent=str(parent))
        return result
