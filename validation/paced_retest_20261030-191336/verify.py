"""汇总本次 A/B 原速回放，并只读核对既有完整结果。"""

import csv
import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "blur_reuse_20261030-185211"


def load(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def fingerprint(folder):
    """排除机器计算耗时；保留轮廓字节、编号、事件及真实输入时间。"""
    result = {}
    with sqlite3.connect(f"file:{folder / 'tracking.sqlite'}?mode=ro", uri=True) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
        for table in ("frames", "observations", "tracks", "events", "track_updates",
                      "event_updates", "input_rejections"):
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
    for name in ("background_initial.npy", "background_final.npy", "source_actions.csv",
                 "source_video_frames.csv", "action_links.csv"):
        result[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    return result


def main():
    summary = {"scope": "Original-speed offline replay; one complete run per video", "runs": []}
    for label in ("A", "B"):
        folder = ROOT / f"exp{label}"
        report = load(folder / "paced_report.json")
        writer = load(folder / "writer_report.json")
        timing = read_csv(folder / "consumer_timing.csv")
        outcomes = read_csv(folder / "frame_outcomes.csv")
        missing = [row for row in outcomes if row["outcome"] != "processed"]
        actual = fingerprint(folder)
        assert report["status"] == "completed" and report["all_offered_frames_accounted"]
        assert writer["all_submitted_saved"] and writer["worker_finished"]
        assert writer["pending_records"] == 0 and not writer["uncommitted"]
        assert writer["committed_frames"] == report["processed_frames"] == actual["frames"]["rows"]
        reference = PREVIOUS / f"after_{label}_1"
        equal = actual == fingerprint(reference) if not missing else None
        if not missing:
            assert equal, label
        record = {
            "video": label, "offered": report["offered_frames"],
            "processed": report["processed_frames"], "saved": writer["committed_frames"],
            "missing": missing, "first_frame_ms": float(timing[0]["process_ms"]),
            "process_ms": report["metrics"]["process_ms"],
            "age_ms": report["metrics"]["age_ms"],
            "memory_mib": report["memory_sampled_mb"],
            "save_queue_peak": writer["peak_pending_records"],
            "full_scientific_results_equal": equal,
            "reference": str(reference),
            "previous_gap_region": [row for row in timing if 1138 <= int(row["video_frame"]) <= 1150] if label == "A" else [],
        }
        if missing:
            indices = {int(row["video_frame"]) for row in missing}
            record["new_gap_windows"] = [row for row in timing if any(
                abs(int(row["video_frame"]) - index) <= 5 for index in indices
            )]
        summary["runs"].append(record)
    code = ROOT.parent.parent / "microbubble_tracking"
    before = load(ROOT / "plan.json")["source_before_sha256"]
    assert all(hashlib.sha256((code / name).read_bytes()).hexdigest() == value
               for name, value in before.items())
    summary["code_unchanged"] = True
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in summary["runs"]:
        print(json.dumps({key: value for key, value in row.items()
                          if key not in ("previous_gap_region", "new_gap_windows")}, indent=2))


if __name__ == "__main__":
    main()
