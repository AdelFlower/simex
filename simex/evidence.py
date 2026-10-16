"""Build model-visible summaries from public robot/video records only.

A contact sheet is produced separately for every rollout. Stage names describe
robot motion events; they do not assert object contact, a successful grasp, or
any hidden simulator state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

STATE_KEYS = ("sim_time", "joints", "joint_velocities", "ee_position", "ee_quaternion", "gripper_width")
MOTION_METHODS = ("move_joints", "move_ee", "gripper", "wait")


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
                  a["kwargs"].get("width", a["end"].get("gripper_width", 0)) <
                  a["start"].get("gripper_width", 0) - .002), None)
    if close:
        def nearest(t):
            return min(range(len(states)), key=lambda i: abs(states[i].get("sim_time", 0) - t))
        before = nearest(close["start"].get("sim_time", 0))
        after = nearest(close["end"].get("sim_time", 0))
        indices += [("before first close", before), ("after first close", after)]
        high = max(range(after, len(states)),
                   key=lambda i: states[i].get("ee_position", [0, 0, 0])[2])
        indices.append(("highest TCP after close", high))
        opening = next((a for a in actions if a["method"] == "gripper" and
                        a["start"].get("sim_time", 0) >= close["end"].get("sim_time", 0) and
                        a["end"].get("gripper_width", 0) > a["start"].get("gripper_width", 0) + .002), None)
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
    actions = []
    for action in evidence.get("actions", []):
        if not isinstance(action, dict) or action.get("method") not in MOTION_METHODS:
            continue
        actions.append({"method": action["method"], "kwargs": action.get("arguments", action.get("kwargs", {})),
                        "start": _action_state(action.get("before", action.get("start", {}))),
                        "end": _action_state(action.get("after", action.get("end", {}))),
                        **({"error": "Action failed; see the public execution status."} if action.get("error") else {})})
    sources = [p for p in [trace, video, *images, *(p for _, p in action_images)] if p is not None]
    signature = hashlib.sha256(json.dumps({"files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources],
                                           "actions": actions}, sort_keys=True).encode()).hexdigest()
    cache = directory / "research_evidence.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("signature") == signature and all(Path(p).is_file() for p in cached["summary"].get("phase_images", [])):
            return {**evidence, **cached["summary"]}
    summary = {"actions": actions}
    try:
        states = _read_trace(trace)
        if not states:
            states = sorted([state for action in actions for state in (action["start"], action["end"])
                             if "sim_time" in state], key=lambda state: state["sim_time"])
        summary["motion_summary"] = _motion_summary(states)
        phases = _phases(states, actions)
        label = "Rollout: " + "/".join(Path(output_dir).parts[-5:])
        sheet, selected = _make_sheet(directory, video, images, phases, len(states), label, action_images)
        summary.update(phase_images=[sheet] if sheet else [], visual_phases=selected)
        if not sheet:
            summary["evidence_errors"] = ["No readable public video or image was available for this rollout."]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        summary["evidence_errors"] = [f"Public evidence summary unavailable: {type(exc).__name__}: {exc}"]
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(json.dumps({"signature": signature, "summary": summary}))
    temporary.replace(cache)
    return {**evidence, **summary}
