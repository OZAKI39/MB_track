"""只读审计一次回放的替换帧；对齐送帧、分析、保存及内存清理记录。

覆盖决定没有独立时间戳，因此用发送端取得锁后的标记与新帧发布时间给出区间，
不把区间中点冒充真实覆盖时刻。只使用同一次 WSL 回放的 perf_counter 时间。
"""

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def indexed(rows, field):
    result = {int(row[field]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Repeated {field}")
    return result


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def interval_state(lower_s, upper_s, finished_s):
    if upper_s < finished_s:
        return "analysis_running"
    if lower_s > finished_s:
        return "between_frames"
    return "boundary_unresolved"


def draw_timeline(cases, output, font):
    """画出实际处理区间与被替换帧的等待区间；不生成假想处理数据。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not font.is_file():
        raise FileNotFoundError(f"Arial font is required: {font}")
    font_manager.fontManager.addfont(str(font))
    if font_manager.FontProperties(fname=str(font)).get_name() != "Arial":
        raise ValueError("The selected font must be Arial")
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(12, 10))
    labels = []
    for index, case in enumerate(cases):
        row = case["summary"]
        base = float(case["dropped_source"]["published_s"])
        previous, following = case["previous"], case["following"]
        for frame, color in ((previous, "#467DA6"), (following, "#87A96B")):
            start = (float(frame["process_start_s"]) - base) * 1000
            duration = float(frame["process_ms"])
            ax.broken_barh([(start, duration)], (index - 0.12, 0.28), facecolors=color)
        gap_start = (float(previous["finished_s"]) - base) * 1000
        gap_width = row["between_frames_ms"]
        ax.broken_barh(
            [(gap_start, gap_width)], (index - 0.12, 0.28), facecolors="#CCCCCC"
        )
        lower = (row["replace_lower_s"] - base) * 1000
        upper = (row["replace_upper_s"] - base) * 1000
        ax.broken_barh([(0, lower)], (index + 0.23, 0.16), facecolors="#E1C48C")
        ax.broken_barh(
            [(lower, upper - lower)], (index - 0.20, 0.62), facecolors="#B24D54"
        )
        for event in case["main_thread_gc"]:
            start = (float(event["start_s"]) - base) * 1000
            ax.broken_barh(
                [(start, float(event["wall_ms"]))],
                (index - 0.12, 0.28),
                facecolors="#E38657",
            )
        labels.append(
            f"Pass {row['pass_number']} | F {row['original_video_frame']} | "
            f"{row['original_video_s']:.3f} s"
        )
    ax.axvline(0, color="#333333", lw=0.9, linestyle="--")
    ax.set_yticks(range(len(cases)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time relative to publication of the dropped frame (ms)")
    ax.set_title("All 20 dropped frames — Exp. A")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            Patch(color="#467DA6", label="Previous analysis"),
            Patch(color="#87A96B", label="Next analysis"),
            Patch(color="#CCCCCC", label="Between frames"),
            Patch(color="#E1C48C", label="Pending frame"),
            Patch(color="#B24D54", label="Overwrite interval"),
            Patch(color="#E38657", label="Garbage collection"),
            Line2D([0], [0], color="#333333", linestyle="--", label="Frame published"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output / "drop_timeline.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "drop_timeline.pdf", bbox_inches="tight")
    plt.close(fig)


def audit(run, output, font):
    run = run.resolve()
    names = (
        "source_timing.csv",
        "consumer_timing.csv",
        "frame_outcomes.csv",
        "source_stalls.csv",
        "consumer_stalls.csv",
        "consumer_stalls.gc.csv",
        "writer_timing.csv",
        "consumer_stalls.meta.json",
        "run_state.json",
        "pacing_settings.json",
        "writer_report.json",
    )
    fingerprints = {
        name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in names
    }
    source = indexed(read_rows(run / "source_timing.csv"), "video_frame")
    consumer = indexed(read_rows(run / "consumer_timing.csv"), "video_frame")
    outcomes = indexed(read_rows(run / "frame_outcomes.csv"), "video_frame")
    source_stages = {
        (int(row["frame"]), row["stage"]): row
        for row in read_rows(run / "source_stalls.csv")
    }
    consumer_stages = read_rows(run / "consumer_stalls.csv")
    collections = read_rows(run / "consumer_stalls.gc.csv")
    saved = indexed(read_rows(run / "writer_timing.csv"), "source_frame")
    metadata = json.loads((run / "consumer_stalls.meta.json").read_text())
    settings = json.loads((run / "pacing_settings.json").read_text())
    writer_report = json.loads((run / "writer_report.json").read_text())
    origin = json.loads((run / "run_state.json").read_text())["origin_perf_s"]
    replaced_by = {
        int(row["replaced_video_frame"]): int(row["video_frame"])
        for row in source.values()
        if row["replaced_video_frame"]
    }
    rejected = {i for i, row in outcomes.items() if row["outcome"] == "replaced"}
    if set(replaced_by) != rejected or rejected & set(consumer):
        raise ValueError("Frame outcomes and actual replacement records disagree")
    if len(rejected) != 20 or set(source) != set(consumer) | rejected:
        raise ValueError("Expected the specified run with exactly 20 replaced frames")
    code = Path(__file__).resolve().parent
    for name in ("paced_source.py", "paced_replay.py", "stall_probe.py"):
        if (
            hashlib.sha256((code / name).read_bytes()).hexdigest()
            != settings["code_hashes"][name]
        ):
            raise ValueError(f"Recorded timing semantics need review: {name} changed")
    with sqlite3.connect(
        f"{(run / 'tracking.sqlite').as_uri()}?mode=ro", uri=True
    ) as db:
        database_frames = {
            row[0] for row in db.execute("SELECT source_frame FROM frames")
        }
        if database_frames != set(consumer) or database_frames != set(saved):
            raise ValueError(
                "Processed frame IDs and database/writer frame IDs disagree"
            )
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(integrity)
    cases = []
    for lost in sorted(rejected):
        replacement = replaced_by[lost]
        before = max(i for i in consumer if i < lost)
        after = min(i for i in consumer if i > lost)
        previous, following = consumer[before], consumer[after]
        # 此次 20 个事件都只替换一帧，替换者随即成为下一张被分析的帧。
        if after != replacement:
            raise ValueError("Consecutive replacements require a different timeline")
        lower = float(source_stages[replacement, "slot_lock"]["end_s"])
        upper = float(source[replacement]["published_s"])
        finished = float(previous["finished_s"])
        if (
            not float(source[lost]["published_s"])
            < lower
            <= upper
            < float(following["taken_s"])
        ):
            raise ValueError("Invalid overwrite interval")
        stages = [r for r in consumer_stages if int(r["frame"]) == before]
        gc = [
            r
            for r in collections
            if int(r["thread_id"]) == metadata["native_thread_id"]
            and float(r["start_s"]) < finished
            and float(r["end_s"]) > float(previous["process_start_s"])
        ]
        substeps = [k for k in previous if k.startswith("seg_") and k.endswith("_ms")]
        substeps += [
            "input_ms",
            "track_features_ms",
            "track_matching_ms",
            "background_ms",
            "pack_records_ms",
        ]
        biggest = max(substeps, key=lambda key: float(previous[key]))
        row = {
            "replay_frame": lost,
            "pass_number": int(source[lost]["replay_cycle"]) + 1,
            "original_video_frame": int(source[lost]["original_video_frame"]),
            "original_source_frame": int(source[lost]["original_source_frame"]),
            "original_video_s": float(source[lost]["original_video_s"]),
            "replay_video_s": float(source[lost]["video_s"]),
            "previous_replay_frame": before,
            "next_replay_frame": after,
            "frame_budget_ms": 1000 / settings["nominal_fps"],
            "previous_queue_ms": float(previous["queue_ms"]),
            "previous_process_ms": float(previous["process_ms"]),
            "next_process_ms": float(following["process_ms"]),
            "between_frames_ms": (float(following["taken_s"]) - finished) * 1000,
            "replacement_interval_ms": (upper - float(source[lost]["published_s"]))
            * 1000,
            "replace_lower_s": lower,
            "replace_upper_s": upper,
            "replacement_from_previous_finish_lower_ms": (lower - finished) * 1000,
            "replacement_from_previous_finish_upper_ms": (upper - finished) * 1000,
            "state_at_overwrite": interval_state(lower, upper, finished),
            "largest_previous_substep": biggest,
            "largest_previous_substep_ms": float(previous[biggest]),
            "previous_segment_ms": float(previous["segment_ms"]),
            "previous_track_ms": float(previous["track_ms"]),
            "previous_save_confirmation_ms": float(saved[before]["durable_ms"]),
            "next_save_confirmation_ms": float(saved[after]["durable_ms"]),
            "replacement_sender_lock_stage_ms": float(
                source_stages[replacement, "slot_lock"]["wall_ms"]
            ),
            "main_gc_ms": sum(float(event["wall_ms"]) for event in gc),
            "previous_main_major_faults": sum(
                int(stage["major_faults"]) for stage in stages
            ),
        }
        cases.append(
            {
                "summary": row,
                "previous": previous,
                "following": following,
                "dropped_source": source[lost],
                "replacement_source": source[replacement],
                "previous_stages": stages,
                "main_thread_gc": gc,
                "previous_save": saved[before],
                "next_save": saved[after],
                "neighboring_consumed_frames": [
                    consumer[i]
                    for i in sorted(consumer)
                    if before - 5 <= i <= after + 2
                ],
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "all_20_frames.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(cases[0]["summary"]))
        writer.writeheader()
        writer.writerows(case["summary"] for case in cases)
    write_json(output / "cases.json", cases)
    write_json(
        output / "verification.json",
        {
            "source_run": str(run),
            "input_sha256": fingerprints,
            "audit_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "origin_perf_s": origin,
            "clock": "One WSL host perf_counter; no cross-OS subtraction",
            "source_frames": len(source),
            "processed_frames": len(consumer),
            "replaced_frames": len(rejected),
            "database_frames": len(database_frames),
            "database_integrity": integrity,
            "all_submitted_saved": writer_report["all_submitted_saved"],
            "peak_save_pending": writer_report["peak_pending_records"],
            "state_counts": {
                state: sum(
                    case["summary"]["state_at_overwrite"] == state for case in cases
                )
                for state in (
                    "analysis_running",
                    "between_frames",
                    "boundary_unresolved",
                )
            },
        },
    )
    draw_timeline(cases, output, font)
    # 检查分析过程没有改写原始计时文件。
    if any(
        hashlib.sha256((run / name).read_bytes()).hexdigest() != digest
        for name, digest in fingerprints.items()
    ):
        raise RuntimeError("An input log changed during the audit")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--font", type=Path, default=Path("/mnt/c/Windows/Fonts/arial.ttf")
    )
    args = parser.parse_args()
    print(audit(args.run, args.output_dir, args.font).resolve())


if __name__ == "__main__":
    main()
