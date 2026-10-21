"""Trusted release evidence shared by live scoring and saved-rollout regrading."""
from __future__ import annotations

import math

from .scoring import score

RELEASE_CONTRACT = "released_placement_v1"
OPEN_WIDTH = .06
WITHDRAWAL = .05
CLEARANCE_MARGIN = .05
SETTLE_SECONDS = .5
MAX_SAMPLE_GAP = .30
MAX_SPEED = .05
MAX_DRIFT = .005
CONTRACT_SPEC = {
    "name": RELEASE_CONTRACT, "minimum_open_width_m": OPEN_WIDTH,
    "minimum_withdrawal_m": WITHDRAWAL, "tcp_clearance_beyond_object_radius_m": CLEARANCE_MARGIN,
    "settle_window_s": SETTLE_SECONDS, "maximum_sample_gap_s": MAX_SAMPLE_GAP,
    "maximum_speed_exclusive_m_s": MAX_SPEED, "maximum_drift_m": MAX_DRIFT,
    "minimum_settle_samples": 3, "known_finger_contact_vetoes": True,
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value, size=3):
    return isinstance(value, (list, tuple)) and len(value) == size and all(finite(v) for v in value)


def release_evidence(scene, metrics, robot, actions, samples):
    """Use measured actions and private samples; never infer release from a score.

    Missing evidence fails closed. Known finger contact vetoes release; engines
    without contact diagnostics must still establish open fingers, withdrawal,
    geometric separation and settled containment over the complete final window.
    """
    closed, opening = False, None
    for action in actions:
        if not isinstance(action, dict) or action.get("method") != "gripper":
            continue
        before, after, arguments = (action.get(key, {}) for key in ("before", "after", "arguments"))
        if not all(isinstance(v, dict) for v in (before, after, arguments)):
            continue
        initial, target, achieved = before.get("gripper_width"), arguments.get("width"), after.get("gripper_width")
        if not all(finite(v) for v in (initial, target, achieved)):
            continue
        if target < initial - .002:
            opening = None
            closed = achieved < initial - .002 and not action.get("error")
        elif closed and target >= OPEN_WIDTH and target > initial + .002 and not action.get("error"):
            if finite(after.get("sim_time")) and vector(after.get("ee_position")):
                opening = {"time": after["sim_time"], "tcp": after["ee_position"]}

    trace = sorted((s for s in samples if isinstance(s, dict) and finite(s.get("sim_time"))),
                   key=lambda s: s["sim_time"])
    output = {}
    for geometry in scene["plates"]:
        name = geometry["name"]
        geometry = {**geometry, 'radius': metrics.get('objects', {}).get(name, {}).get(
            'collision_radius', geometry['radius'])}
        item = {"contract": RELEASE_CONTRACT, "released": False, "reason": "no_opening_after_measured_close"}
        output[name] = item
        if opening is None:
            continue
        if not (isinstance(robot, dict) and finite(robot.get("sim_time"))
                and finite(robot.get("gripper_width")) and vector(robot.get("ee_position"))):
            item["reason"] = "missing_final_robot_measurement"
            continue
        item["final_gripper_width_m"] = robot["gripper_width"]
        item["withdrawal_m"] = math.dist(robot["ee_position"], opening["tcp"])
        if robot["gripper_width"] < OPEN_WIDTH or item["withdrawal_m"] < WITHDRAWAL:
            item["reason"] = "fingers_not_open_or_hand_not_withdrawn"
            continue
        contacts = metrics.get("contact_parameters", {}).get("plates", {}).get(
            name, {}).get("observed_sliding", {}).get("finger")
        if isinstance(contacts, list) and contacts:
            item["reason"] = "final_finger_contact"
            continue
        if not trace or not 0 <= robot["sim_time"] - trace[-1]["sim_time"] <= MAX_SAMPLE_GAP:
            item["reason"] = "missing_recent_object_trace"
            continue
        end = trace[-1]["sim_time"]
        window = [s for s in trace if s["sim_time"] >= end - SETTLE_SECONDS - 1e-6]
        times = [s["sim_time"] for s in window]
        if (len(window) < 3 or times[0] < opening["time"]
                or times[-1] - times[0] < SETTLE_SECONDS - 1e-6
                or any(not 0 < b - a <= MAX_SAMPLE_GAP for a, b in zip(times, times[1:]))):
            item["reason"] = "incomplete_post_opening_settle_window"
            continue
        positions, speeds, clearances = [], [], []
        for sample in window:
            obj, state = sample.get("objects", {}).get(name, {}), sample.get("robot", {})
            if (not vector(obj.get("position")) or not vector(obj.get("linear_velocity"))
                    or not vector(obj.get("quaternion"), 4) or not vector(state.get("ee_position"))
                    or not finite(state.get("gripper_width"))):
                break
            try:
                contained = score({"scene": scene, "predicate": {"kind": "inside", "object": name}},
                                  {"objects": sample["objects"], "tote": metrics["tote"]})
            except (KeyError, TypeError, ValueError):
                break
            if not contained or state["gripper_width"] < OPEN_WIDTH:
                break
            positions.append(obj["position"])
            speeds.append(math.hypot(*obj["linear_velocity"]))
            clearances.append(math.dist(state["ee_position"], obj["position"]))
        if len(positions) != len(window):
            item["reason"] = "unobserved_or_uncontained_settle_sample"
            continue
        item.update(settle_samples=len(window), settle_span_s=times[-1] - times[0],
                    max_speed_m_s=max(speeds),
                    max_drift_m=max(math.dist(positions[0], p) for p in positions),
                    min_tcp_object_distance_m=min(clearances))
        if (item["max_speed_m_s"] >= MAX_SPEED or item["max_drift_m"] > MAX_DRIFT
                or item["min_tcp_object_distance_m"] < geometry["radius"] + CLEARANCE_MARGIN):
            item["reason"] = "object_not_settled_or_not_separated"
            continue
        item.update(released=True, reason="opened_withdrawn_separated_and_settled_inside")
    return output
