"""只读核对本轮回放；停顿统计仅覆盖正式送帧起点到最后分析帧完成。"""

import csv
import hashlib
import json
import platform
import sqlite3
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TABLES = (
    "frames", "observations", "tracks", "events", "track_updates",
    "event_updates", "input_rejections",
)


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def fingerprint(database):
    """包含轮廓原始字节、编号、事件及帧时间；仅排除机器运行耗时。"""
    result = {}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        result["integrity"] = db.execute("PRAGMA integrity_check").fetchone()[0]
        result["foreign_key_errors"] = db.execute("PRAGMA foreign_key_check").fetchall()
        for table in TABLES:
            columns = [row[1] for row in db.execute(f"PRAGMA table_info({table})")]
            columns = [name for name in columns if name != "processing_ms"]
            order = "source_frame, track_id" if table == "observations" else columns[0]
            digest = hashlib.sha256()
            count = 0
            for row in db.execute(
                f"SELECT {','.join(columns)} FROM {table} ORDER BY {order}"
            ):
                digest.update(repr(row).encode())
                digest.update(b"\n")
                count += 1
            result[table] = {"rows": count, "sha256": digest.hexdigest()}
    return result


def median_memory(rows, start, end, key):
    values = [float(row[key]) for row in rows if start <= float(row["elapsed_s"]) < end]
    return statistics.median(values) if values else None


def audit_run(name):
    folder = ROOT / name
    replay = folder / "replay"
    report = read_json(replay / "paced_report.json")
    state = read_json(replay / "run_state.json")
    consumer = read_csv(replay / "consumer_timing.csv")
    processed = [row for row in consumer if row["outcome"] == "processed"]
    end = max(float(row["finished_s"]) for row in processed)
    events = read_csv(folder / "gc_events.csv")
    during = [row for row in events if state["origin_perf_s"] <= float(row["start_s"]) < end]
    gc_report = read_json(folder / "gc_report.json")
    writer = read_json(replay / "writer_report.json")
    memory = read_csv(replay / "memory.csv")
    duration = report["nominal_duration_s"]
    stats = report["metrics"]["process_ms"]
    result = {
        "run": name,
        "seconds": duration,
        "offered": report["offered_frames"],
        "processed": report["processed_frames"],
        "replaced": report["replaced_frames"],
        "expired": report["expired_frames"],
        "failed_cancelled": report["failed_or_cancelled_frames"],
        "abandoned": report["abandoned_frames"],
        "process_median_ms": stats["median"],
        "process_p99_ms": stats["p99"],
        "process_max_ms": stats["max"],
        "first_frame_ms": float(processed[0]["process_ms"]),
        "automatic_gc_count": len(during),
        "automatic_gc_max_ms": max((float(row["wall_ms"]) for row in during), default=0),
        "automatic_gc_over_5ms": sum(float(row["wall_ms"]) > 5 for row in during),
        "automatic_full_gc_count": sum(row["generation"] == "2" for row in during),
        "rss_peak_mib": max(float(row["consumer_rss_mb"]) for row in memory),
        "save_peak_pending": writer["peak_pending_records"],
        "saved": writer["committed_frames"],
        "saved_all": writer["all_submitted_saved"],
        "save_worker_finished": writer["worker_finished"],
        "save_pending_at_end": writer["pending_records"],
        "accounted_all_inputs": report["all_offered_frames_accounted"],
        "status": report["status"],
        "gc_observer_overflow": gc_report["overflow_events"],
    }
    for key in ("consumer_rss_mb", "source_rss_mb"):
        early = median_memory(memory, 30, 90 if duration > 90 else 45, key)
        late = median_memory(memory, duration - 60 if duration > 90 else 45, duration, key)
        result[key + "_early_median"] = early
        result[key + "_late_median"] = late
        result[key + "_growth_mib"] = late - early if early is not None and late is not None else None
    science = fingerprint(replay / "tracking.sqlite")
    with sqlite3.connect(f"file:{replay / 'tracking.sqlite'}?mode=ro", uri=True) as db:
        stored_frames = [row[0] for row in db.execute("SELECT source_frame FROM frames ORDER BY source_frame")]
    result["consumer_matches_saved_frames"] = stored_frames == [int(row["source_frame"]) for row in processed]
    result["database_integrity"] = science["integrity"]
    result["foreign_keys_ok"] = not science["foreign_key_errors"]
    lifecycle_path = replay / "gc_lifecycle.json"
    if lifecycle_path.exists():
        lifecycle = read_json(lifecycle_path)
        before, after = lifecycle["before_stream"], lifecycle["after_save"]
        result.update(
            pre_gc_ms=before["wall_ms"],
            post_gc_ms=after["wall_ms"],
            gc_before_origin=before["finished_perf_s"] < state["origin_perf_s"],
            gc_after_last_frame=after["started_perf_s"] > end,
            gc_after_save=after["started_perf_s"] >= lifecycle["save_finished_perf_s"],
            gc_same_analysis_thread=before["pid"] == before["thread_id"] == after["pid"] == after["thread_id"],
            gc_two_completed=before["status"] == after["status"] == "completed",
        )
        result["gc_settings_unchanged"] = all(
            record["enabled_before"] and record["enabled_after"]
            and record["thresholds_before"] == record["thresholds_after"] == [700, 10, 10]
            for record in (before, after)
        ) and gc_report["stream_thresholds"] == gc_report["restored_thresholds"] == [700, 10, 10]
    worst = sorted(during, key=lambda row: float(row["wall_ms"]), reverse=True)[:5]
    missing = sorted(set(range(report["offered_frames"])) - {int(row["video_frame"]) for row in processed})
    source = read_csv(replay / "source_timing.csv") if missing else []
    gaps = []
    for frame in missing:
        nearby = [row for row in processed if frame - 3 <= int(row["video_frame"]) <= frame + 2]
        left = min(float(row["process_start_s"]) for row in nearby)
        right = max(float(row["finished_s"]) for row in nearby)
        gaps.append({
            "missing_video_frame": frame,
            "consumer_neighbors": nearby,
            "source_neighbors": [row for row in source if frame - 3 <= int(row["video_frame"]) <= frame + 2],
            "gc_overlapping_neighbors": [row for row in during if float(row["start_s"]) < right and float(row["end_s"]) > left],
        })
    detailed = {
        "science": science,
        "worst_automatic_gc": worst,
        "slowest_frames": sorted(processed, key=lambda row: float(row["process_ms"]), reverse=True)[:10],
        "missing_frames": gaps,
    }
    return result, detailed


def main():
    names = ["before_expA", "before_expB", "after_expA", "after_expB",
             "long_A_1", "long_B_1", "long_A_2", "long_B_2"]
    rows, details = [], {}
    for name in names:
        row, details[name] = audit_run(name)
        rows.append(row)
        print(name, row["processed"], "GC max", round(row["automatic_gc_max_ms"], 4), flush=True)
    comparisons = []
    for left, right in (("before_expA", "after_expA"), ("before_expB", "after_expB"),
                        ("long_A_1", "long_A_2"), ("long_B_1", "long_B_2")):
        tables = {table: details[left]["science"][table] == details[right]["science"][table] for table in TABLES}
        comparisons.append({"left": left, "right": right, "equal_tables": tables, "all_equal": all(tables.values())})
    complete_input = []
    previous = ROOT.parent / "analysis_label_preparation_20261030-170020"
    for label in ("A", "B"):
        left, right = previous / ("full_" + label), ROOT / ("science_" + label)
        expected, actual = fingerprint(left / "tracking.sqlite"), fingerprint(right / "tracking.sqlite")
        tables = {table: expected[table] == actual[table] for table in TABLES}
        files = {name: hashlib.sha256((left / name).read_bytes()).hexdigest() == hashlib.sha256((right / name).read_bytes()).hexdigest()
                 for name in ("background_initial.npy", "background_final.npy", "source_actions.csv", "source_video_frames.csv")}
        complete_input.append({
            "video": label, "before": str(left), "after": str(right),
            "purpose": "Complete input correctness comparison, not real-time performance",
            "equal_tables": tables, "equal_files": files,
            "all_equal": all(tables.values()) and all(files.values()),
            "actual_fingerprint": actual,
            "writer": read_json(right / "writer_report.json"),
            "gc_lifecycle": read_json(right / "gc_lifecycle.json"),
            "replay_report": read_json(right / "replay_report.json"),
        })
    cancel_path = ROOT / "cancel/replay"
    cancellation = {name: read_json(cancel_path / (name + ".json"))
                    for name in ("run_state", "summary", "writer_report", "gc_lifecycle")}
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "gc_window": "origin_perf_s <= GC start < last processed frame finished_s",
        "long_memory_windows_s": {"early": [30, 90], "late": [270, 330]},
        "rows": rows, "science_comparisons": comparisons, "details": details,
        "complete_input_comparisons": complete_input, "cancellation": cancellation,
    }
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
