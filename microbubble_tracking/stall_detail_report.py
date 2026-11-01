"""汇总可选的细分诊断；只读运行结果，不把新复测解释为历史事件的现场。"""

import argparse
import csv
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def statistics(values):
    if not values:
        return {"count": 0}
    return dict(
        zip(
            ("min", "median", "p99", "max"),
            map(float, np.percentile(values, [0, 50, 99, 100])),
            strict=True,
        ),
        count=len(values),
    )


def summarize_run(run):
    """核对交接时间闭合；等待新帧的空闲时间与交接工作分别报告。"""
    report = json.loads((run / "paced_report.json").read_text())
    consumer = read_csv(run / "consumer_timing.csv")
    stages = read_csv(run / "consumer_stalls.csv")
    grouped = defaultdict(list)
    for row in stages:
        grouped[row["stage"]].append(float(row["wall_ms"]))
    summary = {
        "frames": {
            key: report[key]
            for key in (
                "offered_frames",
                "processed_frames",
                "replaced_frames",
                "expired_frames",
            )
        },
        "process_ms": report["metrics"]["process_ms"],
        "stages_ms": {name: statistics(values) for name, values in grouped.items()},
        "slow_stages": sorted(stages, key=lambda row: -float(row["wall_ms"]))[:40],
    }
    handoff_path = run / "consumer_handoff.csv"
    if not handoff_path.exists():
        return summary, []
    handoffs = defaultdict(list)
    for row in read_csv(handoff_path):
        handoffs[int(row["after_frame"])].append(row)
    all_work = defaultdict(list)
    busy_work = defaultdict(list)
    details = []
    for previous, following in pairwise(consumer):
        if previous["outcome"] != "processed":
            continue
        frame = int(previous["video_frame"])
        rows = handoffs[frame]
        start, end = float(previous["finished_s"]), float(following["taken_s"])
        if float(rows[0]["start_s"]) != start or float(rows[-1]["end_s"]) != end:
            raise ValueError(f"Incomplete handoff: {run.name} frame {frame}")
        if any(a["end_s"] != b["start_s"] for a, b in pairwise(rows)):
            raise ValueError(f"Handoff has a hole: {run.name} frame {frame}")
        totals = defaultdict(float)
        for row in rows:
            totals[row["stage"]] += float(row["wall_ms"])
        if abs(sum(totals.values()) - (end - start) * 1000) > 1e-6:
            raise ValueError("Handoff duration does not close")
        waiting = "wait_for_frame" in totals
        detail = dict(
            run=run.name,
            after_frame=frame,
            next_frame=int(following["video_frame"]),
            gap_ms=(end - start) * 1000,
            waited_for_frame=waiting,
            **totals,
        )
        details.append(detail)
        for key, value in totals.items():
            all_work[key].append(value)
            if not waiting:
                busy_work[key].append(value)
    summary["handoff"] = {
        "verified_intervals": len(details),
        "all_stages_ms": {
            name: statistics(values) for name, values in all_work.items()
        },
        "without_frame_wait_stages_ms": {
            name: statistics(values) for name, values in busy_work.items()
        },
        "without_frame_wait_total_ms": statistics(
            [row["gap_ms"] for row in details if not row["waited_for_frame"]]
        ),
        "longest_without_frame_wait": sorted(
            [row for row in details if not row["waited_for_frame"]],
            key=lambda row: -row["gap_ms"],
        )[:20],
    }
    return summary, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, details = {}, []
    for run in args.runs:
        summary, rows = summarize_run(run)
        summaries[run.name] = summary
        details.extend(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if details:
        keys = list(dict.fromkeys(key for row in details for key in row))
        with (args.output / "handoff_intervals.csv").open("x", newline="") as file:
            writer = csv.DictWriter(file, keys)
            writer.writeheader()
            writer.writerows(details)


if __name__ == "__main__":
    main()
