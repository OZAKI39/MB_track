"""Read-only checks of replay data and scientific equivalence for this diagnosis."""

import csv
import hashlib
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "gc_lifecycle_20261030-171433"
CODE = ROOT.parents[1] / "microbubble_tracking"
TABLES = ("frames", "observations", "tracks", "events", "track_updates",
          "event_updates", "input_rejections")


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(name, rows):
    with (ROOT / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(database):
    """Hash saved science rows, excluding machine-dependent processing time."""
    result = {}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        result["integrity"] = db.execute("PRAGMA integrity_check").fetchone()[0]
        result["foreign_key_errors"] = db.execute("PRAGMA foreign_key_check").fetchall()
        for table in TABLES:
            columns = [row[1] for row in db.execute(f"PRAGMA table_info({table})")
                       if row[1] != "processing_ms"]
            order = "source_frame, track_id" if table == "observations" else columns[0]
            digest = hashlib.sha256()
            count = 0
            for row in db.execute(f"SELECT {','.join(columns)} FROM {table} ORDER BY {order}"):
                digest.update(repr(row).encode())
                digest.update(b"\n")
                count += 1
            result[table] = {"rows": count, "sha256": digest.hexdigest()}
    return result


def main():
    rows, fingerprints = [], {}
    for name in ("native_A", "native_B", "host_A", "host_B"):
        folder = ROOT / name
        report = read_json(folder / "paced_report.json")
        writer = read_json(folder / "writer_report.json")
        state = read_json(folder / "run_state.json")
        timing = read_csv(folder / "consumer_timing.csv")
        processed = [row for row in timing if row["outcome"] == "processed"]
        end = max(float(row["finished_s"]) for row in processed)
        gc = [row for row in read_csv(folder / "consumer_stalls.gc.csv")
              if state["origin_perf_s"] <= float(row["start_s"]) < end]
        lifecycle = read_json(folder / "gc_lifecycle.json")
        science = fingerprint(folder / "tracking.sqlite")
        fingerprints[name] = science
        with sqlite3.connect(f"file:{folder / 'tracking.sqlite'}?mode=ro", uri=True) as db:
            frames = [row[0] for row in db.execute("SELECT source_frame FROM frames ORDER BY source_frame")]
        unchanged_gc = all(
            lifecycle[key]["enabled_before"] and lifecycle[key]["enabled_after"]
            and lifecycle[key]["thresholds_before"] == lifecycle[key]["thresholds_after"] == [700, 10, 10]
            for key in ("before_stream", "after_save")
        )
        checked = {
            "database_integrity_ok": science["integrity"] == "ok" and not science["foreign_key_errors"],
            "stored_frames_match": frames == [int(row["source_frame"]) for row in processed],
            "all_submitted_saved": writer["all_submitted_saved"] and writer["committed_frames"] == len(frames),
            "writer_finished": writer["worker_finished"] and writer["pending_records"] == 0,
            "gc_settings_unchanged": unchanged_gc,
            "gc_before_stream": lifecycle["before_stream"]["finished_perf_s"] < state["origin_perf_s"],
            "gc_after_save": lifecycle["after_save"]["started_perf_s"] >= lifecycle["save_finished_perf_s"] > end,
        }
        assert all(checked.values()), (name, checked)
        stats = report["metrics"]["process_ms"]
        rows.append({
            "run": name, "offered": report["offered_frames"], "processed": len(frames),
            "replaced": report["replaced_frames"], "seconds": report["nominal_duration_s"],
            "process_median_ms": stats["median"], "process_p99_ms": stats["p99"],
            "process_max_ms": stats["max"],
            "automatic_gc_max_ms": max((float(row["wall_ms"]) for row in gc), default=0),
            "save_peak_pending": writer["peak_pending_records"], **checked,
        })
    comparisons = []
    pairs = [(ROOT / "native_A", PREVIOUS / "after_expA/replay"),
             (ROOT / "native_B", PREVIOUS / "before_expB/replay")]
    reference = ROOT / "first_cold_1/tracking"
    for condition in ("cold", "analysis"):
        for repeat in range(1, 4):
            pairs.append((reference, ROOT / f"first_{condition}_{repeat}/tracking"))
    for left, right in pairs:
        a = fingerprint(left / "tracking.sqlite")
        b = fingerprint(right / "tracking.sqlite")
        tables = {name: a[name] == b[name] for name in TABLES}
        files = {name: hashlib.sha256((left / name).read_bytes()).hexdigest()
                 == hashlib.sha256((right / name).read_bytes()).hexdigest()
                 for name in ("background_initial.npy", "background_final.npy")}
        comparisons.append({"left": str(left), "right": str(right), "tables": tables, "backgrounds": files})
        assert all(tables.values()) and all(files.values()), comparisons[-1]
    code = {name: hashlib.sha256((CODE / name).read_bytes()).hexdigest() == digest
            for name, digest in read_json(ROOT / "plan.json")["code_before"].items()}
    assert all(code.values()), code
    historical = []
    for name, detail in read_json(PREVIOUS / "verification.json")["details"].items():
        if name.startswith("before_"):
            continue
        for gap in detail["missing_frames"]:
            frame = gap["missing_video_frame"]
            before = max((row for row in gap["consumer_neighbors"] if int(row["video_frame"]) < frame),
                         key=lambda row: int(row["video_frame"]))
            historical.append({"run": name, "missing_video_frame": frame,
                               "previous_processed_video_frame": before["video_frame"],
                               "previous_process_ms": before["process_ms"], "previous_core_ms": before["core_ms"],
                               "overlapping_gc_events": len(gap["gc_overlapping_neighbors"]),
                               "native_trace_available_for_original_event": False})
    write_csv("run_summary.csv", rows)
    write_csv("historical_gaps.csv", historical)
    (ROOT / "verification.json").write_text(json.dumps({
        "runs": rows, "scientific_comparisons": comparisons, "code_unchanged": code,
        "fingerprints": fingerprints,
        "limits": "Timing overhead is diagnostic; comparisons exclude processing_ms only. Replays are not hardware experiments.",
    }, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "offered": sum(row["offered"] for row in rows),
                      "processed_and_saved": sum(row["processed"] for row in rows),
                      "scientific_comparisons": len(comparisons), "historical_gaps": len(historical)}, indent=2))


if __name__ == "__main__":
    main()
