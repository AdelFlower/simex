#!/usr/bin/env python3
"""Plot completed fixed-policy replays from private operator diagnostics only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simex.storage import atomic_json


def read(path):
    return json.loads(path.read_text())


def close_window(actions):
    """Use measured action times; end the window before the next opening."""
    closed, completed_width = None, None
    for action in actions:
        if action.get("method") != "gripper":
            continue
        before, after = action.get("before", {}), action.get("after", {})
        width, target = before.get("gripper_width"), action.get("arguments", {}).get("width")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (width, target)):
            continue
        if closed is None and target < width - .002:
            closed = after["sim_time"]
            measured_width = after.get("gripper_width")
            if isinstance(measured_width, (int, float)) and math.isfinite(measured_width):
                completed_width = measured_width
        elif closed is not None and target > width + .002:
            return closed, before["sim_time"], completed_width
    return closed, None, completed_width


def plot(run, task, trial):
    root = run / "adaptation" / task / f"trial_{trial:02d}"
    paths = [root / "target", root / "alignment/initial"]
    if not all((p / "result.json").is_file() for p in paths):
        raise ValueError("The target and initial source replay must both be complete")
    paths.extend(p.parent for p in sorted((root / "alignment").glob("attempt_*/replay/result.json")))
    script = (paths[0] / "policy.py").read_bytes()
    library_hash = read(paths[0] / "research_request.json")["toolbox_sha256"]
    rows, closing_events, sources, series = [], [], [], []
    for index, path in enumerate(paths):
        result = read(path / "result.json")
        if not result.get("actual_started") or (path / "policy.py").read_bytes() != script:
            raise ValueError(f"Missing execution or different replay policy: {path}")
        trace_path = path / "private/object_trace.jsonl"
        samples = [json.loads(line) for line in trace_path.read_text().splitlines()]
        if not samples or any("error" in sample for sample in samples):
            raise ValueError(f"Missing or invalid private trace: {path}")
        evidence_path = path / "public_evidence.json"
        close, opening, close_width = close_window(read(evidence_path)["actions"])
        request = read(path / "research_request.json")
        if request["toolbox_sha256"] != library_hash:
            raise ValueError(f"Different recorded toolbox in calibration replay: {path}")
        cfg = request["source_config"]
        label = "Target (Isaac)" if index == 0 else (
            f"Source {index - 1}: duration {cfg.get('controller_duration_scale', 1):g}, "
            f"friction {cfg.get('friction_scale', 1):g}, mass {cfg.get('mass_scale', 1):g}")
        if cfg.get("camera_offset"):
            label += f", camera {cfg['camera_offset']}"
        baseline = samples[0]["objects"]["green"]["initial_height"]
        times, heights, after_times, widths = [], [], [], []
        if close is not None:
            closing_events.append({"rollout": str(path.relative_to(root)),
                                   "sim_time_s": close,
                                   "finger_width_mm": None if close_width is None else 1000 * close_width,
                                   "measurement_origin": "public_gripper_action_after"})
        if close_width is not None:
            after_times.append(0.)
            widths.append(1000 * close_width)
        for sample in samples:
            now = sample["sim_time"]
            height = 1000 * (sample["objects"]["green"]["position"][2] - baseline)
            width = 1000 * sample["robot"]["gripper_width"]
            if not all(math.isfinite(x) for x in (now, height, width)):
                raise ValueError(f"Non-finite trace value: {path}")
            if times and now < times[-1]:
                raise ValueError(f"Trace time moved backwards: {path}")
            times.append(now)
            heights.append(height)
            in_window = close is not None and now >= close and (opening is None or now <= opening)
            if in_window and (close_width is None or now > close + 1e-9):
                after_times.append(now - close)
                widths.append(width)
            rows.append({"rollout": str(path.relative_to(root)), "sim_time_s": now,
                         "time_after_first_close_s": now - close if close is not None else None,
                         "before_next_open": in_window, "plate_rise_mm": height,
                         "finger_width_mm": width})
        series.append((label, times, heights, after_times, widths, close_width))
        sources.append({"rollout": str(path), "source_config": cfg,
                        "first_close_completed_s": close, "next_open_started_s": opening,
                        "first_close_completed_width_m": close_width,
                        "public_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                        "task_success": result["success"], "samples": len(samples)})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    colors = ["#c43b36", "#8291a2", "#277da8", "#bf8e1a", "#3f8062", "#8856a7"]
    for index, (label, times, heights, after_times, widths, close_width) in enumerate(series):
        style = {"color": colors[index % len(colors)], "lw": 2.3 if index == 0 else 1.7,
                 "zorder": 5 if index == 0 else 2}
        axes[0].plot(times, heights, label=label, **style)
        axes[1].plot(after_times, widths, **style)
        if close_width is not None:
            axes[1].plot([0.], [1000 * close_width], marker="o", markersize=4,
                         color=style["color"], zorder=6)
    axes[0].axhline(50, ls="--", lw=1, color="#555555", alpha=.6)
    axes[0].text(.99, 51, "50 mm lift", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8, color="#555555")
    axes[0].set(xlabel="Simulation time (s)", ylabel="Sampled plate rise from initial height (mm)",
                title="Object motion", xlim=(0, max(max(s[1]) for s in series) * 1.025))
    axes[1].set(xlabel="Time after first closing completed (s)", ylabel="Finger opening (mm)",
                title="Closing completion to next opening")
    if not any(s[3] for s in series):
        axes[1].text(.5, .5, "No measured closing window", transform=axes[1].transAxes, ha="center")
    for ax in axes:
        ax.grid(alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    history_path = root / "alignment/history.json"
    status = read(history_path)["status"] if history_path.exists() else "in progress"
    fig.suptitle(f"{task} — adaptation trial {trial + 1:02d}: completed calibration replays", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .065),
               ncol=2, fontsize=8.5, frameon=False)
    successes = sum(s["task_success"] for s in sources)
    fig.text(.5, .038, f"Task successes: {successes}/{len(sources)}. Alignment status: {status}.",
             ha="center", fontsize=9)
    fig.text(.5, .013, "Private operator diagnostics; width alone does not certify contact. "
             "Circles: measured closing endpoints; lines: subsequent trace samples.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .22, 1, .94))
    base = run / "figures" / f"{task}_trial_{trial:02d}_retention"
    base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        temporary = base.with_suffix(f".{extension}.tmp")
        fig.savefig(temporary, format=extension, dpi=160)
        temporary.replace(base.with_suffix(f".{extension}"))
    plt.close(fig)
    temporary = base.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(base.with_suffix(".csv"))
    closing_path = base.with_name(base.name + "_closing_events.csv")
    temporary = closing_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("rollout", "sim_time_s", "finger_width_mm",
                                                   "measurement_origin"))
        writer.writeheader()
        writer.writerows(closing_events)
    temporary.replace(closing_path)
    atomic_json(base.with_suffix(".json"), {
        "scope": "operator-only private traces; no new physics or research-agent input",
        "generator": "scripts/plot_alignment.py", "trial": str(root), "alignment_status": status,
        "fixed_policy_sha256": hashlib.sha256(script).hexdigest(),
        "recorded_toolbox_sha256": library_hash, "sources": sources,
        "rows": len(rows), "closing_events": closing_events,
        "closing_events_csv": str(closing_path), "plot": str(base.with_suffix(".png"))})
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", choices=("far_plate", "far_to_near", "shifted_single"), required=True)
    parser.add_argument("--trial", type=int, required=True, help="Zero-based adaptation trial index")
    args = parser.parse_args()
    if args.trial < 0:
        parser.error("--trial must be nonnegative")
    print(plot(args.run, args.task, args.trial))
