"""核对模拟 RTSP 经正式 Windows→WSL 链路的记录；只读数据，不访问设备。"""

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import av
import numpy as np

from customer_workspace.streamcopy_recording import write_json
from customer_workspace.tools.windows_camera_sim import consecutive_cycle_start


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_rows(path, rows, columns):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def video_hashes(path):
    """逐帧核对解码像素，不把 MP4 封装差异当成画面损坏。"""
    hashes, times = [], []
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 1
        for frame in container.decode(video=0):
            hashes.append(
                hashlib.sha256(frame.to_ndarray(format="yuv420p")).hexdigest()
            )
            times.append(float(frame.pts * frame.time_base))
    return hashes, times


def build_report(root, output):
    """逐个源帧解释去向，并独立检查录像像素、数据库及离线结果核对。"""
    output.mkdir(parents=True, exist_ok=False)
    inputs = read_json(root / "inputs.json")
    reports, all_gaps = [], []
    for name, provenance in inputs["runs"].items():
        folder = root / name
        session = read_json(folder / "gst_session.json")
        bridge = read_json(folder / "tracking_bridge.json")
        sender = read_json(folder / "tracking_sender.json")
        analysis = read_json(folder / "analysis_check.json")
        recording = read_json(folder / "recording_check.json")
        writer = read_json(folder / "tracking/writer_report.json")
        equivalent = read_json(folder / "equivalence.json")
        rows = read_rows(folder / "analysis_frames.csv")
        outcomes = read_rows(folder / "frame_outcomes.csv")
        processed = {int(row["source_frame"]): row for row in rows}
        decoded = {row["seq"]: row for row in bridge["decoded"]}
        received = {row["seq"] for row in bridge["received"]}
        recorded = {row["seq"]: row for row in session["record_frames"]}
        first, last = min(recorded), max(recorded)

        source_hashes, source_times = video_hashes(Path(provenance["original_video"]))
        saved_hashes, _ = video_hashes(folder / "recording/video.mp4")
        start = consecutive_cycle_start(saved_hashes, source_hashes)
        source_map, loop_frames = [], []
        for row in session["record_frames"]:
            index = row["video_frame"]
            original = (
                (start + index) % len(source_hashes) if start is not None else None
            )
            if original == 0 and index:
                loop_frames.append(index)
            source_map.append(
                {
                    "source_frame": row["seq"],
                    "video_frame": index,
                    "video_time_s": row["video_time_s"],
                    "original_frame": original,
                    "original_time_s": source_times[original]
                    if original is not None
                    else None,
                }
            )
        save_rows(
            output / (name + "_source_mapping.csv"), source_map, list(source_map[0])
        )

        all_frames, gaps = [], []
        previous = None
        for frame in session["input_frames"]:
            seq = frame["seq"]
            if seq < first:
                outcome = "before_recording_window"
            elif seq > last:
                outcome = "after_recording_window"
            elif seq in processed:
                outcome = "processed_and_saved"
                previous = processed[seq]
            elif seq in decoded:
                outcome = "decoded_not_analyzed"
            else:
                outcome = "not_decoded"
            record = recorded.get(seq, {})
            all_frames.append(
                {
                    "source_frame": seq,
                    "video_frame": record.get("video_frame"),
                    "video_time_s": record.get("video_time_s"),
                    "outcome": outcome,
                }
            )
            if first <= seq <= last and seq not in processed:
                current, following = decoded.get(seq), decoded.get(seq + 1)
                gaps.append(
                    {
                        "run": name,
                        "source_frame": seq,
                        "video_frame": record.get("video_frame"),
                        "video_time_s": record.get("video_time_s"),
                        "outcome": outcome,
                        "previous_analyzed_video_frame": previous["video_frame"]
                        if previous
                        else None,
                        "previous_processing_ms": float(previous["processing_call_ms"])
                        if previous
                        else None,
                        "decoded_to_next_publication_ms": (
                            following["published_wsl_s"] - current["published_wsl_s"]
                        )
                        * 1000
                        if current and following
                        else None,
                        "previous_analysis_end_minus_next_publication_ms": (
                            float(previous["completed_wsl_s"])
                            - following["published_wsl_s"]
                        )
                        * 1000
                        if previous and following
                        else None,
                    }
                )
        all_gaps.extend(gaps)
        save_rows(
            output / (name + "_all_frame_outcomes.csv"), all_frames, list(all_frames[0])
        )
        with sqlite3.connect(
            f"{(folder / 'tracking/tracking.sqlite').resolve().as_uri()}?mode=ro",
            uri=True,
        ) as db:
            integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
            database_frames = [
                row[0]
                for row in db.execute(
                    "SELECT source_frame FROM frames ORDER BY source_frame"
                )
            ]
        checks = {
            "recording_passed": recording["passed"],
            "no_duplicate_frame_records": (
                len(recorded) == len(session["record_frames"])
                and len(decoded) == len(bridge["decoded"])
                and len(received) == len(bridge["received"])
                and len(processed) == len(rows)
                and len(database_frames) == len(set(database_frames))
            ),
            "complete_window_decoded": set(recorded) == set(decoded) == received,
            "database_frames_equal_analysis": database_frames == sorted(processed),
            "database_integrity": integrity == "ok",
            "saved_counts_agree": len(rows)
            == analysis["processed"]
            == analysis["saved"]
            == writer["committed_frames"],
            "all_submitted_saved": writer["all_submitted_saved"]
            and writer["worker_finished"]
            and writer["pending_records"] == 0,
            "gaps_fully_accounted": len(gaps)
            == analysis["analysis_skipped"]
            == bridge["overwritten"]
            == sum(row["outcome"] != "processed" for row in outcomes),
            "recorded_pixels_match_source_cycle": start is not None
            and len(saved_hashes) == len(recorded),
            "same_input_tracking_equal_offline": equivalent["scientific_results_equal"],
            "no_pipeline_error": not analysis["pipeline_error"],
            "worker_closed": analysis["threads_closed"],
        }
        runtime_warnings = (
            (folder / "windows_stderr.log").read_text(errors="replace").strip()
        )
        report = {
            "run": name,
            "checks": checks,
            "accounting_passed": all(checks.values()),
            "every_frame_analyzed": len(rows) == len(recorded),
            "windows_received_total": len(session["input_frames"]),
            "before_window": first - 1,
            "after_window": len(session["input_frames"]) - last,
            "windows_received_window": recording["received_window_frames"],
            "recorded": recording["saved_frames"],
            "windows_sent_total_including_warmup": sender["sent_frames"],
            "sent_outside_window": sender["sent_frames"] - len(received),
            "wsl_received_window": len(received),
            "wsl_decoded_window": len(decoded),
            "analyzed": len(rows),
            "saved_analysis": len(database_frames),
            "skipped": len(gaps),
            "analysis_hz": analysis["analysis_hz"],
            "processing_ms": analysis["processing_ms"],
            "software_age_upper_bound_ms": analysis["software_age_upper_bound_ms"],
            "queue_wait_ms": {
                "median": float(np.median([float(r["queue_wait_ms"]) for r in rows])),
                "max": max(float(r["queue_wait_ms"]) for r in rows),
            },
            "rtp_sequence_missing": session["rtp_sequence_missing"],
            "queue_overruns": session["queue_overruns"],
            "save_queue_peak_frames": writer["peak_pending_records"],
            "source_cycle_start_frame": start,
            "source_loop_at_video_frames": loop_frames,
            "windows_stderr": runtime_warnings,
        }
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    columns = [
        "run",
        "source_frame",
        "video_frame",
        "video_time_s",
        "outcome",
        "previous_analyzed_video_frame",
        "previous_processing_ms",
        "decoded_to_next_publication_ms",
        "previous_analysis_end_minus_next_publication_ms",
    ]
    save_rows(output / "gap_details.csv", all_gaps, columns)
    result = {
        "recording_and_frame_accounting_passed": all(
            row["accounting_passed"] for row in reports
        ),
        "every_frame_analyzed": all(row["every_frame_analyzed"] for row in reports),
        "physical_hardware_tested": False,
        "latency_scope": "Conservative software age bound; Windows and WSL clocks are not subtracted",
        "runs": reports,
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="包含 inputs.json、A_01 和 B_01 的本次测试目录"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="必须不存在的新报告目录"
    )
    args = parser.parse_args()
    result = build_report(args.root, args.output)
    return int(not result["recording_and_frame_accounting_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
