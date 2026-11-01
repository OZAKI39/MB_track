"""核对启动重复测试及完整 A/B 回放；只读取实验数据，写出核对摘要。"""

import csv
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "gc_lifecycle_20261030-171433"
TABLES = ("frames", "observations", "tracks", "events", "track_updates",
          "event_updates", "input_rejections")


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(folder):
    """包括原始轮廓、编号、事件和时间，只排除机器处理耗时。"""
    result = {}
    with sqlite3.connect(f"file:{folder / 'tracking.sqlite'}?mode=ro", uri=True) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
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
    for name in ("background_initial.npy", "background_final.npy"):
        result[name] = file_hash(folder / name)
    return result


def verify_lifecycle(folder):
    preparation = read_json(folder / "analysis_preparation.json")
    ready = read_json(folder / "analysis_ready.json")
    gc = read_json(folder / "gc_lifecycle.json")
    writer = read_json(folder / "writer_report.json")
    assert preparation["status"] == "completed" and preparation["scientific_state_unchanged"]
    assert ready["status"] == "ready"
    assert preparation["finished_perf_s"] <= gc["before_stream"]["started_perf_s"]
    assert gc["before_stream"]["finished_perf_s"] <= ready["ready_perf_s"]
    assert gc["after_save"]["started_perf_s"] >= gc["save_finished_perf_s"]
    assert preparation["pid"] == ready["pid"] == gc["before_stream"]["pid"]
    assert preparation["thread_id"] == ready["thread_id"] == gc["before_stream"]["thread_id"]
    for phase in ("before_stream", "after_save"):
        assert gc[phase]["enabled_before"] and gc[phase]["enabled_after"]
        assert gc[phase]["thresholds_before"] == gc[phase]["thresholds_after"] == [700, 10, 10]
    assert writer["worker_finished"] and writer["all_submitted_saved"]
    assert not writer["uncommitted"] and writer["pending_records"] == 0
    return preparation, ready, writer


def main():
    rows, science_checks, summaries = [], [], []
    for label in ("A", "B"):
        reference = fingerprint(ROOT / f"before_{label}_01/tracking")
        for phase in ("before", "after"):
            values = []
            for repeat in range(1, 11):
                name = f"{phase}_{label}_{repeat:02d}"
                folder = ROOT / name
                timing = read_csv(folder / "timings.csv")
                actual = fingerprint(folder / "tracking")
                assert actual == reference, name
                assert actual["frames"]["rows"] == len(timing) == 60
                assert [int(row["frame"]) for row in timing] == list(range(60))
                science_checks.append({"run": name, "equal_to": f"before_{label}_01", "equal": True})
                row = {"run": name, "video": label, "phase": phase,
                       "first_frame_ms": float(timing[0]["wall_ms"]), "preparation_ms": None}
                if phase == "after":
                    preparation, ready, _ = verify_lifecycle(folder / "tracking")
                    assert ready["ready_perf_s"] <= float(timing[0]["start_s"])
                    row["preparation_ms"] = preparation["duration_ms"]
                    row["ready_before_first_frame_ms"] = (
                        float(timing[0]["start_s"]) - ready["ready_perf_s"]
                    ) * 1000
                rows.append(row)
                values.append(row["first_frame_ms"])
            summaries.append({"video": label, "phase": phase, "starts": len(values),
                              "median_ms": statistics.median(values), "max_ms": max(values),
                              "over_60fps_budget": sum(value > 1000 / 60 for value in values)})
    full = []
    before_hashes = read_json(ROOT / "plan.json")["code_before"]
    for label in ("A", "B"):
        previous = PREVIOUS / f"science_{label}"
        metadata = read_json(previous / "session.json")
        assert all(before_hashes[name] == digest for name, digest in metadata["implementation_sha256"].items())
        expected = fingerprint(previous)
        for kind in ("science", "paced"):
            folder = ROOT / f"{kind}_{label}"
            preparation, ready, writer = verify_lifecycle(folder)
            actual = fingerprint(folder)
            assert actual == expected, f"{kind}_{label}"
            assert actual["frames"]["rows"] == writer["committed_frames"] == 1320
            current_meta = read_json(folder / "session.json")
            for key in ("video_sha256", "reference_sha256", "synchronization_hashes"):
                assert current_meta["provenance"][key] == metadata["provenance"][key]
            copies = {}
            for name in ("source_actions.csv", "source_video_frames.csv", "action_links.csv"):
                copies[name] = file_hash(folder / name) == file_hash(previous / name)
            assert all(copies.values()), copies
            checked = {"run": f"{kind}_{label}", "science_equal": True,
                       "frames": actual["frames"]["rows"], "source_files_equal": copies,
                       "preparation_ms": preparation["duration_ms"], "writer": writer}
            if kind == "paced":
                timing = read_csv(folder / "consumer_timing.csv")
                source = read_csv(folder / "source_timing.csv")
                origin = read_json(folder / "run_state.json")["origin_perf_s"]
                assert ready["ready_perf_s"] < origin <= float(timing[0]["process_start_s"])
                assert min(float(row["published_s"]) for row in source if row["published_s"]) >= origin
                report = read_json(folder / "paced_report.json")
                assert report["all_frames_processed"] and report["replaced_frames"] == 0
                assert report["all_offered_frames_accounted"]
                checked.update(first_frame_ms=float(timing[0]["process_ms"]),
                               first_video_frame=int(timing[0]["video_frame"]),
                               ready_before_origin_ms=(origin - ready["ready_perf_s"]) * 1000,
                               replay=report)
            full.append(checked)
    with (ROOT / "startup_samples.csv").open("w", newline="") as file:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"passed": True, "startup_summary": summaries, "startup_samples": rows,
              "prefix_science_checks": science_checks, "full_video_checks": full,
              "scope": "Fresh analysis processes; shared OS file cache, not reboot/cold-disk tests. Offline only."}
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print("40 prefixes and four complete A/B results are scientifically identical to their references.")


if __name__ == "__main__":
    main()
