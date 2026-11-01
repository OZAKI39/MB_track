"""比较同一批完整帧的分项耗时，并核查优化前后的科研记录逐项一致。"""

import argparse
import csv
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

if __package__:
    from .online_replay import file_hash
    from .online_tracking import TIMING_FIELDS
else:
    from online_replay import file_hash
    from online_tracking import TIMING_FIELDS


STAGES = (
    ("input_ms", "Input", "#C8E0D8"),
    ("segment_ms", "Segmentation", "#829E63"),
    ("track_ms", "Tracking", "#C2CF87"),
    ("background_ms", "Background", "#F3F2B2"),
    ("records_ms", "Records", "#FCB895"),
    ("database_write_ms", "Database operations", "#F799A7"),
    ("commit_ms", "Commit wait", "#B75B66"),
    ("other_ms", "Other", "#BBBBBB"),
)


def read_timings(directory):
    with (directory / "stage_timing.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("No measured frames")
    values = {key: np.array([float(row[key]) for row in rows]) for key in TIMING_FIELDS}
    values["records_ms"] = sum(
        values[key]
        for key in ("frame_setup_ms", "pack_records_ms", "read_updates_ms", "return_ms")
    )
    if not np.allclose(
        sum(values[key] for key, _, _ in STAGES), values["process_total_ms"], atol=1e-8
    ):
        raise ValueError("Stage accounting does not add up to full processing time")
    summary = {
        key: {
            "mean_ms": float(np.mean(value)),
            "median_ms": float(np.median(value)),
            "p99_ms": float(np.percentile(value, 99)),
            "max_ms": float(np.max(value)),
        }
        for key, value in values.items()
    }
    return rows, summary


def compare_records(before, after):
    """逐行比较轮廓二进制、浮点值、编号、事件和时间；只排除实测耗时。"""
    with (
        closing(
            sqlite3.connect(
                (before / "tracking.sqlite").resolve().as_uri() + "?mode=ro", uri=True
            )
        ) as old,
        closing(
            sqlite3.connect(
                (after / "tracking.sqlite").resolve().as_uri() + "?mode=ro", uri=True
            )
        ) as new,
    ):
        checks = {}
        for table in (
            "frames",
            "observations",
            "tracks",
            "events",
            "event_updates",
            "track_updates",
            "input_rejections",
        ):
            columns = [
                row[1]
                for row in old.execute(f"PRAGMA table_info({table})")
                if row[1] != "processing_ms"
            ]
            query = f"SELECT {','.join(columns)} FROM {table} ORDER BY 1,2"
            old_rows, new_rows = old.execute(query), new.execute(query)
            count = 0
            while True:
                left, right = old_rows.fetchone(), new_rows.fetchone()
                if left != right:
                    raise ValueError(
                        f"Scientific records changed: {table}, row {count}"
                    )
                if left is None:
                    break
                count += 1
            checks[table] = {"identical": True, "rows": count}
        for name, db in (("before", old), ("after", new)):
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError(f"Database integrity failure: {name}")
    for name in (
        "background_initial.npy",
        "background_final.npy",
        "action_links.csv",
        "source_actions.csv",
        "source_video_frames.csv",
    ):
        if file_hash(before / name) != file_hash(after / name):
            raise ValueError(f"Scientific output changed: {name}")
        checks[name] = {"identical": True, "sha256": file_hash(after / name)}
    old_session = json.loads((before / "session.json").read_text())
    new_session = json.loads((after / "session.json").read_text())
    for key in (
        "experiment_id",
        "stream_id",
        "segmentation",
        "tracking",
        "nominal_fps",
        "max_gap_s",
        "source_time_resolution_s",
        "time_tolerance_s",
        "background_sha256",
    ):
        if old_session[key] != new_session[key]:
            raise ValueError(f"Scientific setting changed: {key}")
    return checks


def make_report(before_runs, after_runs, output, font):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if len(before_runs) != len(after_runs):
        raise ValueError("Supply a matching before/after pair for each recording")
    font_manager.fontManager.addfont(str(font))
    if font_manager.FontProperties(fname=str(font)).get_name() != "Arial":
        raise ValueError("Actual Arial is required")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    comparisons, summaries, labels = [], [], []
    with (output / "stages.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, ("run", "stage", "mean_ms", "median_ms", "p99_ms", "max_ms")
        )
        writer.writeheader()
        for before, after in zip(before_runs, after_runs):
            check = compare_records(before, after)
            before_rows, before_stats = read_timings(before)
            after_rows, after_stats = read_timings(after)
            if [(r["video_frame"], r["source_frame"]) for r in before_rows] != [
                (r["video_frame"], r["source_frame"]) for r in after_rows
            ]:
                raise ValueError("Performance comparison used different frames")
            comparisons.append(
                {
                    "before": str(before.resolve()),
                    "after": str(after.resolve()),
                    "checks": check,
                    "frames": len(before_rows),
                }
            )
            exp = Path(
                json.loads((before / "session.json").read_text())["provenance"]["video"]
            ).stem
            for path, stats, phase in (
                (before, before_stats, "Before"),
                (after, after_stats, "After"),
            ):
                summaries.append(stats)
                labels.append(f"{exp} / {phase}")
                for stage, values in stats.items():
                    writer.writerow({"run": path.name, "stage": stage, **values})
    (output / "verification.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    bottoms = np.zeros(len(labels))
    for key, label, color in STAGES:
        heights = np.array([stats[key]["mean_ms"] for stats in summaries])
        ax.bar(labels, heights, bottom=bottoms, label=label, color=color, width=0.58)
        bottoms += heights
    ax.axhline(1000 / 60, color="#444444", ls="--", lw=1.2, label="60 fps budget")
    for index, height in enumerate(bottoms):
        ax.text(index, height + 0.3, f"{height:.2f}", ha="center", va="bottom")
    ax.set_ylabel("Mean processing time (ms)")
    ax.set_ylim(0, max(bottoms) * 1.15)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15)
    for ext in ("png", "pdf"):
        fig.savefig(output / ("stage_breakdown." + ext), dpi=200, facecolor="white")
    plt.close(fig)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", nargs="+", type=Path, required=True)
    parser.add_argument("--after", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--font", type=Path, default=Path("/mnt/c/Windows/Fonts/arial.ttf")
    )
    args = parser.parse_args()
    make_report(args.before, args.after, args.output_dir, args.font)


if __name__ == "__main__":
    main()
