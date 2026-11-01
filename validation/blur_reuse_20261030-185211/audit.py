"""核对本轮实验记录；只读原始输出，生成汇总数据，不导入历史算法。"""

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent.parent / "microbubble_tracking"
TABLES = ("frames", "observations", "tracks", "events", "track_updates",
          "event_updates", "input_rejections")


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(folder):
    result = {}
    with sqlite3.connect(f"file:{folder / 'tracking.sqlite'}?mode=ro", uri=True) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
        for table in TABLES:
            columns = [r[1] for r in db.execute(f"PRAGMA table_info({table})")
                       if r[1] != "processing_ms"]
            order = "source_frame, track_id" if table == "observations" else columns[0]
            hasher, count = hashlib.sha256(), 0
            for row in db.execute(f"SELECT {','.join(columns)} FROM {table} ORDER BY {order}"):
                hasher.update(repr(row).encode())
                hasher.update(b"\n")
                count += 1
            result[table] = {"rows": count, "sha256": hasher.hexdigest()}
    for name in ("background_initial.npy", "background_final.npy", "source_actions.csv",
                 "source_video_frames.csv", "action_links.csv"):
        result[name] = digest(folder / name)
    return result


def stats(values):
    return dict(zip(("median", "p99", "max"), map(float, np.percentile(values, (50, 99, 100)))))


def main():
    report = {"replays": [], "comparisons": [], "stress": [], "kernel": {}}
    before_hashes = read_json(ROOT / "plan.json")["code_before"]
    for experiment in ("A", "B"):
        reference = fingerprint(ROOT / f"before_{experiment}_1")
        combined = {}
        for phase in ("before", "after"):
            times = []
            for repeat in range(1, 4):
                name = f"{phase}_{experiment}_{repeat}"
                folder = ROOT / name
                actual = fingerprint(folder)
                if phase == "before":
                    hashes = read_json(folder / "session.json")["implementation_sha256"]
                    assert all(before_hashes[n] == h for n, h in hashes.items())
                paced = read_json(folder / "paced_report.json")
                writer = read_json(folder / "writer_report.json")
                complete = paced["all_frames_processed"]
                if complete:
                    assert actual == reference, name
                else:
                    # 保留缺帧运行；不同输入序列的背景、轨迹不能冒充完整结果一致。
                    assert paced["all_offered_frames_accounted"]
                assert writer["all_submitted_saved"] and writer["pending_records"] == 0
                assert writer["worker_finished"] and not writer["uncommitted"]
                assert actual["frames"]["rows"] == writer["committed_frames"] == paced["processed_frames"]
                timing = read_csv(folder / "consumer_timing.csv")
                times.extend(timing)
                memory = read_csv(folder / "memory.csv")
                report["replays"].append({
                    "run": name, "scientific_results_equal": True if complete else None,
                    "comparison_note": "all frames equal" if complete else "input gap; not a full-video equivalence run",
                    "frames": actual["frames"]["rows"],
                    "replaced_frames": paced["replaced_frames"],
                    "expired_frames": paced["expired_frames"],
                    "first_frame_ms": float(timing[0]["process_ms"]),
                    "blur_ms": stats([float(r["seg_blur_ms"]) for r in timing]),
                    "process_ms": stats([float(r["process_ms"]) for r in timing]),
                    "rss_peak_mib": max(float(r["consumer_rss_mb"]) for r in memory),
                    "rss_last5_median_mib": float(np.median([float(r["consumer_rss_mb"]) for r in memory[-5:]])),
                    "save_queue_peak": writer["peak_pending_records"],
                })
                if phase == "after":
                    prep = read_json(folder / "analysis_preparation.json")
                    assert prep["buffer_bytes"]["blurred"] == 5120 * 1024
                    assert prep["scientific_state_unchanged"]
            combined[phase] = {
                "frames": len(times),
                "blur_ms": stats([float(r["seg_blur_ms"]) for r in times]),
                "process_ms": stats([float(r["process_ms"]) for r in times]),
            }
        report["comparisons"].append({"experiment": experiment, **combined})
        folder = ROOT / f"stress_{experiment}"
        paced = read_json(folder / "paced_report.json")
        writer = read_json(folder / "writer_report.json")
        assert paced["all_offered_frames_accounted"] and paced["offered_frames"] == 6600
        # 两段原始 PTS 略有差异；保留实际时长，不强改成整数 110 秒。
        assert abs(paced["nominal_duration_s"] - 110) < 0.001
        assert writer["all_submitted_saved"] and writer["pending_records"] == 0
        assert writer["worker_finished"] and not writer["uncommitted"]
        assert paced["database_integrity"] == "ok"
        assert writer["committed_frames"] == paced["processed_frames"]
        memory = read_csv(folder / "memory.csv")
        settled = [r for r in memory if float(r["elapsed_s"]) >= 22]
        last = [r for r in memory if float(r["elapsed_s"]) >= 88]
        first = [r for r in memory if 22 <= float(r["elapsed_s"]) < 44]
        memory_details = {
            "after_first_cycle_min_mib": min(float(r["consumer_rss_mb"]) for r in settled),
            "after_first_cycle_max_mib": max(float(r["consumer_rss_mb"]) for r in settled),
            "second_cycle_median_mib": float(np.median([float(r["consumer_rss_mb"]) for r in first])),
            "last_cycle_median_mib": float(np.median([float(r["consumer_rss_mb"]) for r in last])),
        }
        report["stress"].append({
            "experiment": experiment, "duration_s": paced["nominal_duration_s"], "offered_frames": 6600,
            "processed_frames": paced["processed_frames"],
            "saved": writer["committed_frames"], "replaced": paced["replaced_frames"],
            "expired": paced["expired_frames"], "process_ms": paced["metrics"]["process_ms"],
            "memory": memory_details, "peak_pending_records": writer["peak_pending_records"],
        })
    pixels = read_json(ROOT / "pixels/verification.json")
    assert pixels["status"] == "passed"
    report["pixels"] = pixels["videos"]
    kernel = read_csv(ROOT / "pixels/kernel_timing.csv")
    for method in ("allocate", "reuse"):
        rows = [r for r in kernel if r["method"] == method]
        report["kernel"][method] = {
            "samples": len(rows), "wall_ms": stats([float(r["wall_ms"]) for r in rows]),
            "minor_faults": sum(int(r["minor_faults"]) for r in rows),
            "major_faults": sum(int(r["major_faults"]) for r in rows),
        }
    report["pixel_and_full_input_equivalence_passed"] = True
    report["all_paced_runs_without_gaps"] = all(
        r["replaced_frames"] == r["expired_frames"] == 0 for r in report["replays"]
    ) and all(r["replaced"] == r["expired"] == 0 for r in report["stress"])
    report["status"] = (
        "passed" if report["all_paced_runs_without_gaps"] else "verified_with_replay_gaps"
    )
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    hashes = {n: digest(CODE / n) for n in before_hashes}
    for name in ("online_store.py", "async_store.py", "paced_replay.py"):
        assert hashes[name] == before_hashes[name], name
    (ROOT / "code_after.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(report["comparisons"], indent=2))
    print(json.dumps(report["kernel"], indent=2))
    print(json.dumps(report["stress"], indent=2))


if __name__ == "__main__":
    main()
