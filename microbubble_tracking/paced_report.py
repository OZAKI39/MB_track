"""读取真实速度回放记录，核对帧去向并绘制性能图；不改变追踪结果。"""

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return dict(
        zip(
            ("min", "median", "p95", "p99", "max"),
            map(float, np.percentile(values, (0, 50, 95, 99, 100))),
        )
    )


def summarize(directory):
    """逐帧检查发送记录、替换记录、处理日志与数据库，防止漏计尾帧。"""
    settings = json.loads((directory / "pacing_settings.json").read_text())
    state = json.loads((directory / "run_state.json").read_text())
    source = read_csv(directory / "source_timing.csv")
    consumed = read_csv(directory / "consumer_timing.csv")
    processed = [r for r in consumed if r["outcome"] == "processed"]
    memory = read_csv(directory / "memory.csv")
    offered_ids = {int(r["video_frame"]) for r in source}
    taken_ids = {int(r["video_frame"]) for r in consumed}
    replaced_ids = {
        int(r["replaced_video_frame"]) for r in source if r["replaced_video_frame"]
    }
    db = sqlite3.connect(
        f"{(directory / 'tracking.sqlite').resolve().as_uri()}?mode=ro", uri=True
    )
    try:
        db_frames = db.execute(
            "SELECT COALESCE(video_frame,source_frame),source_frame,source_time_s "
            "FROM frames ORDER BY 1"
        ).fetchall()
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        db.close()
    processed_ids = {int(r["video_frame"]) for r in processed}
    if {r[0] for r in db_frames} != processed_ids or integrity != "ok":
        raise ValueError("Timing log and committed tracking frames disagree")
    if len(offered_ids) != len(source) or len(taken_ids) != len(consumed):
        raise ValueError("Repeated video frame in timing log")
    if taken_ids & replaced_ids or not (taken_ids | replaced_ids) <= offered_ids:
        raise ValueError("Invalid frame replacement bookkeeping")
    abandoned_ids = offered_ids - taken_ids - replaced_ids
    if state["status"] == "completed" and abandoned_ids:
        raise ValueError("Completed replay left an unaccounted frame")
    source_by_id = {int(r["video_frame"]): r for r in source}
    if any(
        sf != int(source_by_id[vf]["source_frame"])
        or abs(ts - float(source_by_id[vf]["video_s"])) > 1e-12
        for vf, sf, ts in db_frames
    ):
        raise ValueError("Original source frame/time was not preserved")
    outcome_by_id = {int(r["video_frame"]): r["outcome"] for r in consumed}
    with (directory / "frame_outcomes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["video_frame", "source_frame", "video_s", "outcome"])
        for frame in sorted(offered_ids):
            row = source_by_id[frame]
            outcome = (
                "replaced"
                if frame in replaced_ids
                else outcome_by_id.get(frame, "abandoned_on_stop")
            )
            writer.writerow([frame, row["source_frame"], row["video_s"], outcome])
    fps = settings["nominal_fps"]
    duration = (
        (float(source[-1]["video_s"]) - float(source[0]["video_s"]) + 1 / fps)
        if source
        else 0
    )
    send_late = [
        (float(r["published_s"]) - float(r["deadline_s"])) * 1000 for r in source
    ]
    metrics = {
        key: statistics([float(r[key]) for r in processed])
        for key in ("queue_ms", "process_ms", "core_ms", "age_ms")
    }
    metrics["send_late_ms"] = statistics(send_late)
    metrics["decode_ms"] = statistics([float(r["decode_ms"]) for r in source])
    time_s = np.array([float(r["video_s"]) for r in processed])
    ages = np.array([float(r["age_ms"]) for r in processed])
    early_late = None
    if len(time_s) > 1 and time_s[-1] - time_s[0] >= 10:
        early_late = {
            "first_5s_median_ms": float(np.median(ages[time_s < time_s[0] + 5])),
            "last_5s_median_ms": float(np.median(ages[time_s > time_s[-1] - 5])),
        }
    expired = sum(r["outcome"] == "expired" for r in consumed)
    return {
        "status": state["status"],
        "synthetic_repeated_replay": settings.get("synthetic_replay", False),
        "error": state["error"],
        "offered_frames": len(source),
        "processed_frames": len(processed),
        "replaced_frames": len(replaced_ids),
        "expired_frames": expired,
        "failed_or_cancelled_frames": sum(
            r["outcome"] in ("error", "cancelled") for r in consumed
        ),
        "abandoned_frames": len(abandoned_ids),
        "nominal_duration_s": duration,
        "processed_fps": len(processed) / duration if duration else 0,
        "frame_budget_ms": 1000 / fps,
        "metrics": metrics,
        "age_early_late": early_late,
        "memory_sampled_mb": {
            key: statistics([float(r[key]) for r in memory if r[key]])
            for key in ("consumer_rss_mb", "source_rss_mb")
        },
        "max_pending_frames": 1,
        "all_offered_frames_accounted": True,
        "database_integrity": integrity,
        "all_frames_processed": (
            state["status"] == "completed" and len(source) == len(processed)
        ),
        "source_p99_within_frame_budget": bool(
            send_late and np.percentile(send_late, 99) < 1000 / fps
        ),
        "results_older_than_limit": sum(
            float(r["age_ms"]) > settings["max_age_ms"] for r in processed
        ),
        "scope": "1x software replay; no camera, recording load, Windows bridge or AFG commands",
        "timing_limit": "Latency is local replay timing, not exposure-to-result latency",
    }


def draw_run(directory, output, font):
    """输出英文 Arial 图；运行结束后绘图，不占用测速期间的资源。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if not font.exists():
        raise FileNotFoundError(f"Arial font is required: {font}")
    font_manager.fontManager.addfont(str(font))
    if font_manager.FontProperties(fname=str(font)).get_name() != "Arial":
        raise ValueError("Selected font is not Arial")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    rows = [
        r
        for r in read_csv(directory / "consumer_timing.csv")
        if r["outcome"] == "processed"
    ]
    source = read_csv(directory / "source_timing.csv")
    memory = read_csv(directory / "memory.csv")
    settings = json.loads((directory / "pacing_settings.json").read_text())
    times = [float(r["video_s"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), layout="constrained")
    ax = axes[0, 0]
    ax.plot(
        times,
        [float(r["process_ms"]) for r in rows],
        lw=0.8,
        color="#087F8C",
        label="Full processing",
    )
    ax.plot(
        times,
        [float(r["core_ms"]) for r in rows],
        lw=0.6,
        color="#B97828",
        alpha=0.8,
        label="Algorithm",
    )
    ax.axhline(
        1000 / settings["nominal_fps"],
        color="#555555",
        ls="--",
        lw=1,
        label="Frame budget",
    )
    ax.set(ylabel="Processing time (ms)")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[0, 1]
    ax.plot(
        times,
        [float(r["age_ms"]) for r in rows],
        lw=0.8,
        color="#8055A2",
        label="Result age",
    )
    ax.plot(
        [float(r["video_s"]) for r in source],
        [(float(r["published_s"]) - float(r["deadline_s"])) * 1000 for r in source],
        lw=0.8,
        color="#87A66B",
        label="Source lateness",
    )
    ax.set(ylabel="Time since scheduled arrival (ms)")
    ax.legend(frameon=False, fontsize=9)
    outcomes = read_csv(directory / "frame_outcomes.csv")
    ax = axes[1, 0]
    duration = float(source[-1]["video_s"]) + 1 / settings["nominal_fps"]
    bins = np.arange(0, np.ceil(duration) + 1)
    for outcome, color in (
        ("processed", "#087F8C"),
        ("replaced", "#B97828"),
        ("expired", "#8055A2"),
    ):
        counts, _ = np.histogram(
            [float(r["video_s"]) for r in outcomes if r["outcome"] == outcome], bins
        )
        ax.stairs(
            counts,
            bins,
            baseline=None,
            label=outcome.capitalize(),
            color=color,
            lw=1.4,
        )
    ax.set(ylabel="Frames per 1 s bin")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 1]
    for key, label, color in (
        ("consumer_rss_mb", "Processing", "#087F8C"),
        ("source_rss_mb", "Video source", "#B97828"),
    ):
        points = [r for r in memory if r[key]]
        ax.plot(
            [float(r["elapsed_s"]) for r in points],
            [float(r[key]) for r in points],
            lw=1.3,
            color=color,
            label=label,
        )
    ax.set(ylabel="Resident memory (MiB)")
    ax.legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.set_xlabel("Replay time (s)")
        ax.set_xlim(0, duration)
        ax.grid(axis="y", alpha=0.15)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        target = output / f"{directory.name}.{extension}"
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--font", type=Path, default=Path("/mnt/c/Windows/Fonts/arial.ttf")
    )
    args = parser.parse_args()
    for directory in args.runs:
        draw_run(directory, args.output_dir, args.font)


if __name__ == "__main__":
    main()
