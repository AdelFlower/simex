#!/usr/bin/env python3
"""Read-only operator audit of placement, opening and withdrawal evidence.

Private truth stays in this offline report. This does not change training scores,
feed an agent, execute physics, or certify the absence of robot/object contacts.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simex.scoring import score


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value, size=3):
    return isinstance(value, (list, tuple)) and len(value) == size and all(finite(v) for v in value)


def final_window(samples, object_name, task, tote, seconds=5.):
    """Measure a final post-opening window without declaring release or stability."""
    window = [s for s in samples if s["sim_time"] >= samples[-1]["sim_time"] - seconds - 1e-6]
    span = window[-1]["sim_time"] - window[0]["sim_time"]
    objects = [s.get("objects", {}).get(object_name, {}) for s in window]
    positions = [obj.get("position") for obj in objects]
    velocities = [obj.get("linear_velocity") for obj in objects]
    outcome = {
        "requested_span_s": seconds, "samples": len(window), "span_s": span,
        "requested_span_observed": span >= seconds - 1e-6,
        "canonical_predicate_passed_all_samples": None,
        "max_object_speed_m_s": None,
        "max_object_displacement_from_window_start_m": None,
    }
    if all(vector(v) for v in velocities):
        outcome["max_object_speed_m_s"] = max(math.hypot(*v) for v in velocities)
    if all(vector(p) for p in positions):
        outcome["max_object_displacement_from_window_start_m"] = max(
            math.dist(positions[0], p) for p in positions)
    if task is not None:
        predicates = task["predicate"]["all"]
        required = {p["object"] for p in predicates}
        results = []
        for sample in window:
            observed = sample.get("objects", {})
            # The scorer has defaults for some fields. Missing trace measurements
            # must remain unknown in this audit, rather than becoming zero speed.
            if any(not isinstance(observed.get(name), dict)
                   or not vector(observed[name].get("position"))
                   or not vector(observed[name].get("quaternion"), 4)
                   or not vector(observed[name].get("linear_velocity"))
                   for name in required):
                break
            if not all(finite(observed[object_name].get(key))
                       for key in ("initial_height", "max_height")):
                break
            try:
                results.append(score(task, {"objects": observed, "tote": tote}))
            except (KeyError, TypeError, ValueError):
                break
        if len(results) == len(window):
            outcome["canonical_predicate_passed_all_samples"] = all(results)
    return outcome


def audit_rollout(directory, object_name="green"):
    directory = Path(directory)
    errors = []

    def read(name):
        try:
            value = json.loads((directory / name).read_text())
            if not isinstance(value, dict):
                raise ValueError("expected an object")
            return value
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {type(exc).__name__}")
            return {}

    result = read("result.json")
    manifest = read("manifest.json")
    scene = read("private/scene.json")
    metrics = read("private/metrics.json")
    evidence = result.get("public_evidence")
    if not isinstance(evidence, dict):
        evidence = read("public_evidence.json")
    output = {
        "rollout": str(directory.resolve()), "engine": manifest.get("engine"),
        "task": manifest.get("task"), "recorded_status": result.get("status"),
        "recorded_task_success": result.get("success"), "object": object_name,
        "full_placement_predicate_passed": None, "final_object_speed_m_s": None,
        "final_finger_contact_observed": None,
        "max_object_lift_m": None, "opening_after_close": None,
        "after_opening": None, "contact_release_certified": False,
        "review_note": "Opening and withdrawal are evidence, not contact-force measurements. "
                       "Review saved video; this audit does not replace the frozen task score.",
        "video": str(directory / "public/overhead.mp4"),
        "errors": errors,
    }
    contacts = metrics.get("contact_parameters", {}).get("plates", {}).get(
        object_name, {}).get("observed_sliding", {}).get("finger")
    if isinstance(contacts, list) and all(finite(value) for value in contacts):
        output["final_finger_contact_observed"] = bool(contacts)
    obj = metrics.get("objects", {}).get(object_name, {})
    placement_task = None
    if obj and scene:
        predicates = [{"kind": "lift", "object": object_name, "min": .05},
                      {"kind": "inside", "object": object_name}]
        if object_name != "red" and "red" in metrics.get("objects", {}):
            predicates.append({"kind": "outside", "object": "red"})
        placement_task = {"scene": scene, "predicate": {"all": predicates}}
        try:
            output["full_placement_predicate_passed"] = score(placement_task, metrics)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"full placement predicate: {type(exc).__name__}")
        if vector(obj.get("linear_velocity")):
            output["final_object_speed_m_s"] = math.hypot(*obj["linear_velocity"])
        if finite(obj.get("max_height")) and finite(obj.get("initial_height")):
            output["max_object_lift_m"] = obj["max_height"] - obj["initial_height"]

    close_seen, opening = False, None
    for index, action in enumerate(evidence.get("actions", [])):
        if not isinstance(action, dict) or action.get("method") != "gripper":
            continue
        before = action.get("before", action.get("start", {}))
        after = action.get("after", action.get("end", {}))
        arguments = action.get("arguments", action.get("kwargs", {}))
        if not all(isinstance(value, dict) for value in (before, after, arguments)):
            continue
        initial, target = before.get("gripper_width"), arguments.get("width")
        if not (finite(initial) and finite(target)):
            continue
        if target < initial - .002:
            close_seen, opening = True, None  # A later close supersedes a previous opening.
        elif close_seen and target > initial + .002:
            opening = {"action_index": index, "commanded_width_m": target,
                       "width_before_m": initial,
                       "width_after_m": after.get("gripper_width") if finite(after.get("gripper_width")) else None,
                       "end_time_s": after.get("sim_time") if finite(after.get("sim_time")) else None,
                       "tcp_at_opening_end_m": after.get("ee_position") if vector(after.get("ee_position")) else None}
    output["opening_after_close"] = opening

    samples = []
    try:
        for line in (directory / "private/object_trace.jsonl").read_text().splitlines():
            sample = json.loads(line)
            if not isinstance(sample, dict) or "error" in sample:
                errors.append("private trace contains an invalid or failed sample")
                continue
            if not finite(sample.get("sim_time")):
                errors.append("private trace sample has no finite time")
                continue
            samples.append(sample)
    except (OSError, ValueError) as exc:
        errors.append(f"private/object_trace.jsonl: {type(exc).__name__}")
    if opening and finite(opening["end_time_s"]):
        later = sorted((s for s in samples if s["sim_time"] >= opening["end_time_s"]),
                       key=lambda s: s["sim_time"])
        if later:
            first, last = later[0], later[-1]
            robot = last.get("robot", {})
            final_obj = last.get("objects", {}).get(object_name, {})
            final_tcp = robot.get("ee_position")
            positions = [s.get("objects", {}).get(object_name, {}).get("position") for s in later]
            positions = [p for p in positions if vector(p)]
            velocities = [s.get("objects", {}).get(object_name, {}).get("linear_velocity") for s in later]
            velocities = [v for v in velocities if vector(v)]
            interval = {"samples": len(later),
                        "first_sample_delay_s": first["sim_time"] - opening["end_time_s"],
                        "observed_duration_after_opening_s": last["sim_time"] - opening["end_time_s"],
                        "sample_span_s": last["sim_time"] - first["sim_time"],
                        "final_gripper_width_m": robot.get("gripper_width") if finite(robot.get("gripper_width")) else None,
                        "tcp_displacement_from_opening_m": None,
                        "final_tcp_to_object_center_m": None,
                        "object_displacement_from_first_post_open_sample_m": None,
                        "max_object_speed_m_s": max((math.hypot(*v) for v in velocities), default=None)}
            if vector(final_tcp) and vector(opening["tcp_at_opening_end_m"]):
                interval["tcp_displacement_from_opening_m"] = math.dist(
                    final_tcp, opening["tcp_at_opening_end_m"])
            if vector(final_tcp) and vector(final_obj.get("position")):
                interval["final_tcp_to_object_center_m"] = math.dist(final_tcp, final_obj["position"])
            if len(positions) == len(later):
                interval["object_displacement_from_first_post_open_sample_m"] = math.dist(
                    positions[0], positions[-1])
            else:
                errors.append("post-opening object positions are incomplete")
            interval["final_window"] = final_window(
                later, object_name, placement_task, metrics.get("tote"))
            output["after_opening"] = interval
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rollout", type=Path)
    source.add_argument("--run", type=Path)
    parser.add_argument("--object", default="green", dest="object_name")
    args = parser.parse_args()
    if args.rollout:
        directories = [args.rollout]
    else:
        stages = ("discovery", "selection", "auxiliary_selection", "adaptation", "evaluation")
        directories = sorted(p.parent for stage in stages for p in (args.run / stage).rglob("result.json")
                             if (p.parent / "manifest.json").is_file()
                             and (p.parent / "execution.json").is_file())
    audits = [audit_rollout(path, args.object_name) for path in directories]
    print(json.dumps({"purpose": "Private operator evidence; never model input.",
                      "rollouts": audits}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
