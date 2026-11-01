"""Audit completed replay files without changing the recordings or databases."""

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np


OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[2]
MODULE = PROJECT / "customer_workspace/microbubble_tracking"
TABLES = (
    "frames", "observations", "tracks", "events", "event_updates",
    "track_updates", "input_rejections",
)


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    if not records:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def database_audit(path):
    """Compare stored scientific values, excluding wall-clock processing cost."""
    tables = {}
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        for table in TABLES:
            columns = [
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
                if row[1] != "processing_ms"
            ]
            order = "source_frame,track_id" if table == "observations" else columns[0]
            query = f"SELECT {','.join(columns)} FROM {table} ORDER BY {order}"
            digest = hashlib.sha256()
            count = 0
            for row in connection.execute(query):
                digest.update(repr(row).encode())
                digest.update(b"\n")
                count += 1
            tables[table] = {"rows": count, "sha256": digest.hexdigest()}
        frame_ids = [row[0] for row in connection.execute(
            "SELECT source_frame FROM frames ORDER BY source_frame"
        )]
    finally:
        connection.close()
    return tables, frame_ids


def summarize_run(name):
    directory = OUTPUT / name
    replay = directory / "replay"
    report = read_json(replay / "paced_report.json")
    writer = read_json(replay / "writer_report.json")
    state = read_json(replay / "run_state.json")
    settings = read_json(replay / "pacing_settings.json")
    gc_report = read_json(directory / "gc_report.json")
    consumer = read_csv(replay / "consumer_timing.csv")
    processed = [row for row in consumer if row["outcome"] == "processed"]
    outcomes = read_csv(replay / "frame_outcomes.csv")
    source = read_csv(replay / "source_timing.csv")
    events = read_csv(directory / "gc_events.csv")
    memory = read_csv(replay / "memory.csv")
    origin_s = state["origin_perf_s"]
    last_s = max(float(row["finished_s"]) for row in processed)
    active_events = [
        row for row in events if origin_s <= float(row["start_s"]) < last_s
    ]
    tables, saved_ids = database_audit(replay / "tracking.sqlite")
    processed_ids = [int(row["source_frame"]) for row in processed]
    assert saved_ids == processed_ids
    assert writer["submitted_frames"] == writer["committed_frames"] == len(processed)
    assert writer["pending_records"] == 0
    assert writer["all_submitted_saved"] and writer["worker_finished"]
    assert not writer["error"] and not report["error"]
    assert report["all_offered_frames_accounted"]
    assert len(outcomes) == report["offered_frames"] == len(source)
    assert len({row["source_frame"] for row in outcomes}) == len(outcomes)
    assert sorted(int(row["source_frame"]) for row in outcomes
                  if row["outcome"] == "processed") == saved_ids
    assert gc_report["original_thresholds"] == gc_report["restored_thresholds"]
    assert gc_report["original_enabled"] == gc_report["restored_enabled"] is True
    assert gc_report["stream_thresholds"] == [700, 10, 10]
    assert gc_report["overflow_events"] == gc_report["garbage_after"] == 0
    assert not any(settings[key] for key in (
        "measure_stages", "diagnose_stalls", "diagnose_details"
    ))
    assert settings["conversion_memory"] == "reuse"

    links = []
    overlap_by_frame = Counter()
    for event in active_events:
        for row in processed:
            overlap_s = min(float(event["end_s"]), float(row["finished_s"])) - max(
                float(event["start_s"]), float(row["process_start_s"])
            )
            if overlap_s > 0:
                overlap_by_frame[int(row["source_frame"])] += overlap_s * 1000
                links.append({
                    "run": name, "source_frame": row["source_frame"],
                    "generation": event["generation"],
                    "gc_wall_ms": event["wall_ms"],
                    "gc_thread_cpu_ms": event["thread_cpu_ms"],
                    "collected": event["collected"], "overlap_ms": overlap_s * 1000,
                    "frame_process_ms": row["process_ms"],
                    "trigger_stack": event["trigger_stack"],
                })

    gaps = [row for row in outcomes if row["outcome"] != "processed"]
    context = []
    for gap in gaps:
        gap_id = int(gap["source_frame"])
        context.append({
            "gap": gap,
            "nearby_consumer": [row for row in consumer
                                if abs(int(row["source_frame"]) - gap_id) <= 3],
            "nearby_source": [row for row in source
                              if abs(int(row["source_frame"]) - gap_id) <= 3],
        })
    duration_s = report["nominal_duration_s"]
    rss = [float(row["consumer_rss_mb"]) for row in memory]
    memory_stats = {"sampled_max_mib": max(rss)}
    if last_s - origin_s > 90:
        early = [float(row["consumer_rss_mb"]) for row in memory
                 if 30 <= float(row["elapsed_s"]) < 90]
        late = [float(row["consumer_rss_mb"]) for row in memory
                if float(row["elapsed_s"]) >= duration_s - 60]
        memory_stats.update({
            "median_30_to_90_s_mib": float(np.median(early)),
            "median_last_60_s_mib": float(np.median(late)),
            "growth_mib": float(np.median(late) - np.median(early)),
        })
    worst = max(processed, key=lambda row: float(row["process_ms"]))
    summary = {
        "status": report["status"], "counts": {
            key: report[key] for key in (
                "offered_frames", "processed_frames", "replaced_frames",
                "expired_frames", "failed_or_cancelled_frames", "abandoned_frames",
            )
        },
        "outcome_counts": dict(Counter(row["outcome"] for row in outcomes)),
        "metrics": report["metrics"],
        "process_ms": report["metrics"]["process_ms"],
        "gc_during_frames": {
            "counts": dict(Counter(row["generation"] for row in active_events)),
            "max_ms": max((float(row["wall_ms"]) for row in active_events), default=0),
            "collected": sum(int(row["collected"]) for row in active_events),
        },
        "gc_outside_frames": [row for row in events if row not in active_events],
        "memory": memory_stats,
        "all_saved": True, "settings_restored": True,
        "max_pending_records": writer["peak_pending_records"],
        "pending_at_end": writer["pending_records"],
        "scientific_tables": tables,
        "worst_frame": {
            "source_frame": int(worst["source_frame"]),
            "process_ms": float(worst["process_ms"]),
            "gc_overlap_ms": overlap_by_frame[int(worst["source_frame"])],
        },
        "gap_context": context,
    }
    return summary, links, saved_ids


def main():
    plan = read_json(OUTPUT / "plan.json")
    assert (OUTPUT / "memory_after.txt").is_file(), "Replays are still running"
    code_unchanged = {path: file_hash(MODULE / path) == digest
                      for path, digest in plan["code_sha256"].items()}
    inputs_unchanged = {path: file_hash(MODULE / path) == digest
                        for path, digest in plan["input_sha256"].items()}
    assert all(code_unchanged.values()) and all(inputs_unchanged.values())
    summaries = {}
    saved = {}
    links = []
    comparison = []
    previous = read_json(Path(plan["previous_result"]) / "verification.json")
    for name in plan["order"]:
        summary, run_links, frame_ids = summarize_run(name)
        summaries[name] = summary
        saved[name] = frame_ids
        links.extend(run_links)
        comparison.append({
            "run": name, "status": summary["status"], **summary["counts"],
            "process_median_ms": summary["process_ms"]["median"],
            "process_p99_ms": summary["process_ms"]["p99"],
            "process_max_ms": summary["process_ms"]["max"],
            "gc_max_ms": summary["gc_during_frames"]["max_ms"],
            "rss_peak_mib": summary["memory"]["sampled_max_mib"],
            "pending_peak": summary["max_pending_records"],
        })
        print(json.dumps({"run": name, **comparison[-1]}, ensure_ascii=False))
    same_frames = saved["normal"] == saved["precollect"]
    evidence = {
        "runs": summaries,
        "code_unchanged": code_unchanged, "inputs_unchanged": inputs_unchanged,
        "paired_processed_frames_identical": same_frames,
        "paired_scientific_tables_identical": (
            summaries["normal"]["scientific_tables"]
            == summaries["precollect"]["scientific_tables"]
        ) if same_frames else None,
        "matches_previous_complete_baseline": {
            name: summaries[name]["scientific_tables"]
            == previous["runs"]["long_normal"]["scientific_tables"]
            for name in ("normal", "precollect")
        },
        "comparison_limit": "Different processed input sequences cannot establish paired output equality.",
        "scope": "Offline 1x A replay only; no live camera, recording or AFG load.",
    }
    with (OUTPUT / "verification.json").open("x") as stream:
        json.dump(evidence, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    write_csv(OUTPUT / "comparison.csv", comparison)
    write_csv(OUTPUT / "gc_frame_links.csv", links)
    print("AUDIT_COMPLETE", same_frames, evidence["matches_previous_complete_baseline"])


if __name__ == "__main__":
    main()
