#!/usr/bin/env python3
"""Render saved experiment results; never run policies or supply optimizer feedback.

Requires matplotlib in the Python environment used for this analysis command.
Refresh report.json with `simex report` before rendering.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    ledger = json.loads((args.run / "target_ledger.json").read_text())
    completed_slots = sum(s["state"] == "complete" for s in ledger["slots"].values())
    if completed_slots != report["target_completed"]:
        parser.error("report.json is stale; refresh it with simex report before plotting")
    output = args.output or args.run / "figures"
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    methods = ["direct_cap", "stage1_transfer", "simex"]
    names = ["Direct CaP", "Stage 1 transfer", "SimEX"]
    tasks = list(dict.fromkeys(row["task_id"] for row in report["evaluation"]))
    fig = plt.figure(figsize=(12, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    source_ax = fig.add_subplot(grid[0, 0])
    eval_ax = fig.add_subplot(grid[0, 1])
    adapt_ax = fig.add_subplot(grid[1, :])
    stage1 = report["stage1"]
    members = stage1.get("members", [])
    rows = []
    for index, member in enumerate(members):
        results = member.get("selection_results", [])
        scheduled = stage1["selection_tasks_per_member"]
        completed = len(results)
        successes = member["full_task_successes"]
        rate = successes / scheduled if scheduled and completed == scheduled else None
        selected = member["member"] == stage1.get("winner")
        if rate is not None:
            source_ax.bar(index, rate, color="#247b72" if selected else "#90aaa7", width=.55)
            source_ax.text(index, rate + .035, f"{successes}/{scheduled}", ha="center")
        else:
            source_ax.text(index, .08, f"Pending\n{completed}/{scheduled} done", ha="center")
        rows.append({"phase": "source_selection", "method": f"member_{member['member']:02d}",
                     "task": "pooled_full_task", "scheduled": scheduled, "completed": completed,
                     "successes": successes, "success_rate": rate})
    source_ax.set_xticks(range(len(members)), [
        f"Member {m['member']}" + (" (selected)" if m["member"] == stage1.get("winner") else "")
        for m in members])
    if not members:
        source_ax.text(.5, .5, "Source selection pending", ha="center", transform=source_ax.transAxes)
    source_ax.set_ylim(0, 1.12)
    source_ax.set_ylabel("Success fraction")
    source_ax.set_title("Frozen source selection")
    source_ax.spines[["top", "right"]].set_visible(False)

    by_group = {(r["method"], r["task_id"]): r for r in report["evaluation"]}
    for y, task in enumerate(tasks):
        for x, method in enumerate(methods):
            record = by_group[(method, task)]
            rate = record["success_rate"]
            color = "#e6e8eb" if rate is None else plt.get_cmap("Blues")(.10 + .80 * rate)
            eval_ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=2))
            if rate is None:
                label = f"Pending\n{record['completed']}/{record['scheduled']} done"
                if record["completed"]:
                    label += f"\n{record['successes']} successes"
            else:
                label = f"{record['successes']}/{record['scheduled']}\n{rate:.0%}"
            eval_ax.text(x + .5, y + .5, label, ha="center", va="center", fontsize=10,
                         color="white" if rate is not None and rate > .65 else "#263238")
            rows.append({"phase": "frozen_target_evaluation", "method": method, "task": task,
                         **{k: record[k] for k in ("scheduled", "completed", "successes", "success_rate")}})
    eval_ax.set(xlim=(0, 3), ylim=(len(tasks), 0), title="Frozen target evaluation (matched seeds)")
    eval_ax.set_xticks([.5, 1.5, 2.5], names)
    eval_ax.set_yticks([i + .5 for i in range(len(tasks))], tasks)
    eval_ax.tick_params(length=0)
    for spine in eval_ax.spines.values():
        spine.set_visible(False)

    slots = [s for s in ledger["slots"].values() if s["phase"] == "adapt"]
    count = report["config"]["adapt_trials"]
    for y, task in enumerate(tasks):
        ordered = sorted((s for s in slots if s["task_id"] == task), key=lambda s: s["trial"])
        for x, slot in enumerate(ordered):
            complete = slot["state"] == "complete"
            color = ("#247b72" if slot.get("success") else "#c96d61") if complete else "#e6e8eb"
            adapt_ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=3))
            label = ("Success" if slot.get("success") else slot.get("status", "Failed")) if complete else slot["state"]
            adapt_ax.text(x + .5, y + .5, label, ha="center", va="center", fontsize=10,
                          color="white" if complete else "#53616b")
    adapt_ax.set(xlim=(0, count), ylim=(len(tasks), 0), title="Adaptation target trials (outcome before that trial's repair)")
    adapt_ax.set_xticks([i + .5 for i in range(count)], [f"Trial {i + 1}" for i in range(count)])
    adapt_ax.set_yticks([i + .5 for i in range(len(tasks))], tasks)
    adapt_ax.tick_params(length=0)
    for spine in adapt_ax.spines.values():
        spine.set_visible(False)

    contract = report["config"].get("placement_contract", "legacy_geometry")
    inherited = bool(stage1.get("inherited_stage1"))
    scope = "Saved source selection regraded; no new selection rollouts." if inherited else "Source selection uses frozen task predicates."
    fig.suptitle(f"SimEX: MuJoCo to Isaac Sim | {contract}\n"
                 f"{report['target_completed']}/{report['target_scheduled']} target slots completed. "
                 "Pending cells are not success rates.", fontsize=13)
    fig.supxlabel(scope + " Results describe this local budget and seed schedule.", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(output / f"results.{extension}", dpi=160)
    plt.close(fig)
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["phase", "method", "task", "scheduled", "completed", "successes", "success_rate"])
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
