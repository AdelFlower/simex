"""Build model-visible summaries from public robot/video records only.

A contact sheet is produced separately for every rollout. Stage names describe
robot motion events; they do not assert object contact, a successful grasp, or
any hidden simulator state.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

STATE_KEYS = ("sim_time", "joints", "joint_velocities", "ee_position", "ee_quaternion", "gripper_width")
MOTION_METHODS = ("move_joints", "move_ee", "gripper", "wait")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value, length):
    return isinstance(value, (list, tuple)) and len(value) == length and all(_number(v) for v in value)


def _endpoints(action):
    before = action.get('before', action.get('start', {}))
    after = action.get('after', action.get('end', {}))
    return (before if isinstance(before, dict) else {}, after if isinstance(after, dict) else {})


def public_action_result(value):
    """Keep public API diagnostics, excluding poses already in action endpoints."""
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in ('success', 'clamped') if isinstance(value.get(key), bool)}
    if isinstance(value.get('reason'), str):
        result['reason'] = value['reason'][:320]
    if _number(value.get('joint_error')):
        result['joint_error'] = value['joint_error']
    ik = value.get('ik')
    if isinstance(ik, dict) and isinstance(ik.get('success'), bool):
        result['ik'] = {'success': ik['success']}
    return result


def _gripper_direction(action):
    before, _ = _endpoints(action)
    kwargs = action.get('arguments', action.get('kwargs', {}))
    width = kwargs.get('width') if isinstance(kwargs, dict) else None
    initial = before.get('gripper_width')
    if not (_number(width) and _number(initial)):
        return 'unknown'
    return 'close' if width < initial - .002 else 'open' if width > initial + .002 else 'hold'


def _diagnostic_line(stdout, motion=False):
    if not isinstance(stdout, str):
        return None
    failure = re.compile(r'stop|fail|abort|unreachable|reject|exhaust|'
                         r'(?:tracking|motion|pose|joint)\s+error|\b[A-Za-z]*Error:|^\s*ERROR(?:\s|:)', re.I)
    movement = re.compile(r'mov|motion|track|segment|approach|pose|reach|joint|guard|correction', re.I)
    for line in reversed(stdout.splitlines()):
        if failure.search(line) and (not motion or movement.search(line)):
            return line.strip()[:320]
    return None


def _termination(evidence):
    status = evidence.get('status')
    reason = evidence.get('reason')
    result = {'status': status, 'kind': 'unknown', 'reason': None, 'source': None}
    if status in ('success', 'failed', 'completed'):
        result['kind'] = 'policy_returned'
    elif status in ('timeout', 'policy_error', 'invalid_policy', 'generation_error',
                    'interrupted', 'infrastructure_error'):
        result['kind'] = status
    if isinstance(reason, str) and reason.strip():
        result.update(reason=_diagnostic_line(reason) or reason.strip()[:320], source='execution_reason')
    else:
        line = _diagnostic_line(evidence.get('stdout'))
        if line:
            # A log line reports the policy's interpretation, not a certified
            # failure cause. The actual execution status remains separate.
            result.update(reason=line, source='policy_stdout')
        elif status in ('success', 'failed', 'completed'):
            reasons = {'success': 'Policy returned; independent task predicate passed.',
                       'failed': 'Policy returned; independent task predicate failed.',
                       'completed': 'Policy returned.'}
            result.update(reason=reasons[status], source='execution_status')
    return result


def action_summary(evidence, states=None):
    """Summarize public commands and robot motion, never object/grasp success.

    Rise is measured from a close command's endpoint until the next opening
    command starts. Withdrawal after release cannot count as carried-object
    motion. Trace samples improve the estimate; action endpoints are sufficient
    when no complete public trace is available.
    """
    termination = _termination(evidence)
    if 'actions' not in evidence:
        return {'classification': 'unavailable', 'arm_commands': None,
                'gripper_commands': None, 'arm_motion_observed': None,
                'command_stage': 'unavailable', 'close_commands': None, 'open_commands': None,
                'min_tcp_height_m': None, 'tcp_rise_after_close_m': None,
                'last_motion_rejection': None, 'termination': termination}
    actions = evidence['actions']
    counts = {name: 0 for name in MOTION_METHODS}
    directions = {'close': 0, 'open': 0, 'hold': 0, 'unknown': 0}
    travel, moved, opened_after_close = 0., False, False
    heights, close_intervals = [], []
    active_close, last_rejection = None, None
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        method = action.get('method')
        if method not in counts:
            continue
        counts[method] += 1
        before, after = _endpoints(action)
        for state in (before, after):
            p = state.get('ee_position')
            if _vector(p, 3):
                heights.append(float(p[2]))
        if method == 'gripper':
            direction = _gripper_direction(action)
            directions[direction] += 1
            if direction == 'open' and active_close is not None:
                opened_after_close = True
                active_close['end'] = before.get('sim_time')
                active_close['opened'] = True
                active_close = None
            elif direction == 'close' and active_close is None:
                position = after.get('ee_position')
                active_close = {'start': after.get('sim_time'), 'end': None, 'opened': False,
                                'height': float(position[2]) if _vector(position, 3) else None,
                                'heights': []}
                close_intervals.append(active_close)
        if active_close is not None:
            position = after.get('ee_position')
            if _vector(position, 3):
                active_close['heights'].append(float(position[2]))
        if method not in ('move_ee', 'move_joints'):
            continue
        response = action.get('result', {})
        if not isinstance(response, dict):
            response = {}
        if response.get('success') is False or response.get('clamped') or action.get('error'):
            reason = ('Backend action interrupted' if action.get('error') else
                      response.get('reason') or ('joint_limits_clamped' if response.get('clamped')
                                                else 'controller_reported_failure'))
            last_rejection = {'action_index': index, 'method': method, 'source': 'action_result',
                              'reason': str(reason)[:240]}
        a, b = before.get('ee_position'), after.get('ee_position')
        if _vector(a, 3) and _vector(b, 3):
            distance = math.dist(a, b)
            travel += distance
            moved |= distance > .001
        a, b = before.get('joints'), after.get('joints')
        if _vector(a, 7) and _vector(b, 7):
            moved |= any(abs(x - y) > .001 for x, y in zip(a, b))
        a, b = before.get('ee_quaternion'), after.get('ee_quaternion')
        if _vector(a, 4) and _vector(b, 4):
            norm = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
            if norm > 0 and math.isfinite(norm):
                angle = 2 * math.acos(min(1., abs(sum(x*y for x, y in zip(a, b))) / norm))
                moved |= angle > .005
    for state in states if states is not None else evidence.get('trace_tail', evidence.get('robot_trace_tail', [])):
        if not isinstance(state, dict) or not _vector(state.get('ee_position'), 3):
            continue
        height, time = float(state['ee_position'][2]), state.get('sim_time')
        heights.append(height)
        if _number(time):
            for interval in close_intervals:
                if (_number(interval['start']) and time >= interval['start'] and
                        (not interval['opened'] or (_number(interval['end']) and time <= interval['end']))):
                    interval['heights'].append(height)
    rises = [max(0., max(interval['heights'], default=interval['height']) - interval['height'])
             for interval in close_intervals if interval['height'] is not None]
    rise = max(rises) if rises else None
    if last_rejection is None:
        line = _diagnostic_line(evidence.get('stdout'), motion=True)
        if line:
            last_rejection = {'source': 'policy_stdout', 'reason': line}
    arm = counts['move_ee'] + counts['move_joints']
    stage = ('opened_after_close' if opened_after_close else
             'tcp_raised_after_close' if rise is not None and rise > .01 else
             'close_commanded' if directions['close'] else
             'arm_commanded_without_close' if arm else 'no_arm_commands')
    return {'arm_commands': arm, 'gripper_commands': counts['gripper'],
            'wait_commands': counts['wait'], 'arm_motion_observed': bool(moved),
            'tcp_endpoint_travel_m': travel,
            'classification': ('no_arm_commands' if not arm else
                               'arm_motion_observed' if moved else 'arm_commands_without_measured_motion'),
            'close_commands': directions['close'], 'open_commands': directions['open'],
            'hold_commands': directions['hold'], 'unknown_gripper_commands': directions['unknown'],
            'min_tcp_height_m': min(heights) if heights else None,
            'tcp_rise_after_close_m': rise, 'command_stage': stage,
            'last_motion_rejection': last_rejection, 'termination': termination}


def _public_file(value, directory: Path) -> Path | None:
    if not isinstance(value, str):
        return None
    path = Path(value).resolve()
    if path.is_relative_to(directory) and path.is_file():
        return path
    return None


def _state(value):
    return {key: value[key] for key in STATE_KEYS if key in value} if isinstance(value, dict) else {}


def _action_state(value):
    # Every action is retained, with the public pose/width/time needed to explain
    # its behavior. Raw joint and velocity samples remain in robot_trace.jsonl.
    state = _state(value)
    return {key: state[key] for key in ("sim_time", "ee_position", "ee_quaternion", "gripper_width")
            if key in state}


def _read_trace(path):
    if path is None:
        return []
    return [_state(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def _phases(states, actions):
    if not states:
        return []
    indices = [("initial", 0)]
    close = next((a for a in actions if a["method"] == "gripper" and
                  _gripper_direction(a) == 'close'), None)
    if close:
        def nearest(t):
            return min(range(len(states)), key=lambda i: abs(states[i].get("sim_time", 0) - t))
        before = nearest(close["start"].get("sim_time", 0))
        after = nearest(close["end"].get("sim_time", 0))
        indices += [("before first close", before), ("after first close", after)]
        opening = next((a for a in actions if a["method"] == "gripper" and
                        a["start"].get("sim_time", 0) >= close["end"].get("sim_time", 0) and
                        _gripper_direction(a) == 'open'), None)
        last = max(after, nearest(opening["start"].get("sim_time", 0))) if opening else len(states)-1
        high = max(range(after, last+1),
                   key=lambda i: states[i].get("ee_position", [0, 0, 0])[2])
        indices.append(("highest TCP after close", high))
        if opening:
            indices.append(("after opening", nearest(opening["end"].get("sim_time", 0))))
    else:
        indices.extend((f"timeline {i}/4", round(i * (len(states) - 1) / 4)) for i in range(1, 4))
    indices.append(("final", len(states) - 1))
    return [{"phase": label, "trace_index": index,
             "sim_time": states[index].get("sim_time", 0)} for label, index in indices]


def _motion_summary(states):
    result = {"samples": len(states)}
    if not states:
        return result
    result.update(initial=_state(states[0]), final=_state(states[-1]))
    for key in ("gripper_width", "sim_time"):
        values = [s[key] for s in states if isinstance(s.get(key), (int, float))]
        if values:
            result[key + "_range"] = [min(values), max(values)]
    positions = [s["ee_position"] for s in states if len(s.get("ee_position", [])) == 3]
    if positions:
        result["ee_position_bounds"] = [[min(p[i] for p in positions) for i in range(3)],
                                         [max(p[i] for p in positions) for i in range(3)]]
    return result


def _make_sheet(directory, video, images, phases, sample_count, label, action_images):
    frames = []
    # Exact action-boundary captures survive video encoder failures and avoid
    # presenting a nearby periodic video frame as the end of a short action.
    phase_frames = []
    if action_images:
        for phase in phases:
            nearest = min(action_images, key=lambda item: abs(item[0]["sim_time"] - phase["sim_time"]))
            if abs(nearest[0]["sim_time"] - phase["sim_time"]) <= .26:
                try:
                    with Image.open(nearest[1]) as im:
                        phase_frames.append((phase, im.convert("RGB")))
                except OSError:
                    break
            else:
                break
        if len(phase_frames) == len(phases):
            frames = phase_frames
    if not frames and video and phases and sample_count:
        import imageio.v2 as imageio
        reader = None
        try:
            reader = imageio.get_reader(str(video), format="ffmpeg")
            # Recorder captures video and robot state at the same sampling boundary.
            # Its final trace entry can precede an encoder failure; bound by video frames.
            count = reader.count_frames()
            for phase in phases:
                index = min(max(0, phase["trace_index"]), count - 1)
                frames.append((phase, Image.fromarray(reader.get_data(index))))
        except Exception:
            frames = []
        finally:
            if reader is not None:
                reader.close()
    if not frames:
        images = [path for _, path in action_images] or images
        chosen = images if len(images) <= 6 else [images[round(i * (len(images) - 1) / 5)] for i in range(6)]
        for index, path in enumerate(chosen):
            try:
                with Image.open(path) as im:
                    frames.append(({"phase": f"recorded image {index + 1}", "source": path.name}, im.convert("RGB")))
            except OSError:
                continue
    if not frames:
        return None, []
    width, height, title = 480, 360, 46
    cols = 3 if len(frames) > 2 else len(frames)
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * (height + title) + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), label, fill="black")
    selected = []
    for i, (phase, frame) in enumerate(frames):
        x, y = (i % cols) * width, 40 + (i // cols) * (height + title)
        description = phase["phase"]
        if "sim_time" in phase:
            description += f"; t={phase['sim_time']:.2f}s"
        if "source" in phase:
            description += "; " + phase["source"]
        draw.text((x + 6, y + 6), description, fill="black")
        sheet.paste(ImageOps.contain(frame.convert("RGB"), (width, height)), (x, y + title))
        selected.append(phase)
    path = directory / "research_contact_sheet.jpg"
    sheet.save(path, quality=88)
    return str(path), selected


def summarize_public_evidence(evidence: dict, output_dir: str | None) -> dict:
    """Read only files beneath this rollout's public/ directory, never private/.

    This function does not infer object success. Scores enter through the separate
    trusted evaluator result and remain distinct from observable robot evidence.
    """
    evidence = {**evidence, 'action_summary': action_summary(evidence)}
    if not output_dir:
        return evidence
    directory = (Path(output_dir) / "public").resolve()
    if not directory.is_dir():
        return evidence
    trace = _public_file(evidence.get("robot_trace_path", str(directory / "robot_trace.jsonl")), directory)
    video = _public_file(evidence.get("video"), directory)
    images = [_public_file(path, directory) for path in evidence.get("keyframes", []) + evidence.get("read_images", [])]
    images = list(dict.fromkeys(path for path in images if path is not None))
    # Keep legacy trace fallback but never forward a path escaping public/.
    evidence = {**evidence,
        "keyframes": [str(p) for p in images if str(p) in evidence.get("keyframes", [])],
        "read_images": [str(p) for p in images if str(p) in evidence.get("read_images", [])],
        "video": str(video) if video else None}
    action_images = []
    for entry in evidence.get("action_images", []):
        path = _public_file(entry.get("path"), directory) if isinstance(entry, dict) else None
        if path is not None and isinstance(entry.get("sim_time"), (int, float)):
            action_images.append((entry, path))
    evidence["action_images"] = [
        {**{key: entry[key] for key in ("action", "method", "phase", "sim_time") if key in entry},
         "path": str(path)}
        for entry, path in action_images]
    actions = []
    for action in evidence.get("actions", []):
        if not isinstance(action, dict) or action.get("method") not in MOTION_METHODS:
            continue
        actions.append({"method": action["method"], "kwargs": action.get("arguments", action.get("kwargs", {})),
                        "start": _action_state(action.get("before", action.get("start", {}))),
                        "end": _action_state(action.get("after", action.get("end", {}))),
                        **({"result": public_action_result(action["result"])} if "result" in action else {}),
                        **({"error": "Action failed; see the public execution status."} if action.get("error") else {})})
    sources = [p for p in [trace, video, *images, *(p for _, p in action_images)] if p is not None]
    source_metadata = {"version": 3,
        "files": [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in sources],
        "actions": actions, "diagnostics": {k: evidence.get(k) for k in ('status', 'reason', 'stdout')}}
    # Compare ordinary cache metadata, never stored content hashes.
    source_metadata = json.loads(json.dumps(source_metadata))
    cache = directory / "research_evidence.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("source_metadata") == source_metadata and all(
                Path(p).is_file() for p in cached["summary"].get("phase_images", [])):
            return {**evidence, **cached["summary"]}
    summary = {"actions": actions}
    try:
        states = _read_trace(trace)
        if trace is not None:
            # Preserve every approved robot-state sample for paged research reads.
            summary["robot_trace"] = states
        if not states:
            states = sorted([state for action in actions for state in (action["start"], action["end"])
                             if "sim_time" in state], key=lambda state: state["sim_time"])
        summary["motion_summary"] = _motion_summary(states)
        summary["action_summary"] = action_summary(evidence, states=states)
        phases = _phases(states, actions)
        label = "Rollout: " + "/".join(Path(output_dir).parts[-5:])
        sheet, selected = _make_sheet(directory, video, images, phases, len(states), label, action_images)
        summary.update(phase_images=[sheet] if sheet else [], visual_phases=selected)
        if not sheet:
            summary["evidence_errors"] = ["No readable public video or image was available for this rollout."]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        summary["evidence_errors"] = [f"Public evidence summary unavailable: {type(exc).__name__}: {exc}"]
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(json.dumps({"source_metadata": source_metadata, "summary": summary}))
    temporary.replace(cache)
    return {**evidence, **summary}
