"""离线核对平滑内存复用：完整 A/B 像素与追踪一致，交错测量同一平滑函数。"""

import argparse
import csv
import hashlib
import json
import resource
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from customer_workspace.microbubble_tracking.bubble_tracker import (
    OnlineBackground,
    SegParams,
    Tracker,
    TrackParams,
    combine_backgrounds,
    iter_frames,
    segment,
)
from customer_workspace.microbubble_tracking.online_replay import verify_inputs
from customer_workspace.microbubble_tracking.paced_replay import environment


def measure_blur(gray, output, writer, experiment, index):
    """交错顺序；输入相同，计时只包含 GaussianBlur，不包含像素比较。"""
    expected = cv2.GaussianBlur(gray, (0, 0), 1.0)
    for _ in range(3):
        cv2.GaussianBlur(gray, (0, 0), 1.0, dst=output)
    for repeat in range(30):
        order = ("allocate", "reuse") if repeat % 2 else ("reuse", "allocate")
        for method in order:
            before = resource.getrusage(resource.RUSAGE_THREAD)
            cpu = time.thread_time_ns()
            started = time.perf_counter_ns()
            actual = cv2.GaussianBlur(
                gray, (0, 0), 1.0, dst=output if method == "reuse" else None
            )
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            cpu_ms = (time.thread_time_ns() - cpu) / 1e6
            after = resource.getrusage(resource.RUSAGE_THREAD)
            assert np.array_equal(actual.view(np.uint8), expected.view(np.uint8))
            if method == "reuse":
                assert actual is output
            writer.writerow([
                experiment, index, repeat, method, wall_ms, cpu_ms,
                after.ru_minflt - before.ru_minflt,
                after.ru_majflt - before.ru_majflt,
            ])
            del actual  # 不把前一次独立分配的结果保留到下一次计时。
    peaks = {}
    for method in ("allocate", "reuse"):
        tracemalloc.start()
        try:
            actual = cv2.GaussianBlur(
                gray, (0, 0), 1.0, dst=output if method == "reuse" else None
            )
            peaks[method] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        del actual
    return peaks


def verify_video(code, experiment, output, timing_writer):
    """调用同一正式分割函数，仅改变 blurred_out；不复制旧识别算法。"""
    video = code / f"data/raw/exp{experiment}.mp4"
    reference = code / "results/B_offline/background.npy"
    pzt = code / f"data/pzt/exp{experiment}"
    provenance = verify_inputs(video, reference, pzt)
    selected = set(np.linspace(0, provenance["expected_frames"] - 1, 12, dtype=int))
    params = SegParams()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.close, params.close))
    backgrounds = []
    events = Counter()
    peaks = []
    with (
        (pzt / "video_frames.csv").open(encoding="utf-8-sig", newline="") as source,
        (output / f"pixels_{experiment}.csv").open("x", newline="") as log,
    ):
        writer = csv.writer(log)
        writer.writerow([
            "video_frame", "source_frame", "source_time_s", "pixels_checked",
            "different_pixels", "blur_sha256", "labels_sha256", "tracking_sha256",
            "allocate_blur_ms", "reuse_blur_ms",
        ])
        for (index, gray), row in zip(
            iter_frames(str(video)), csv.DictReader(source), strict=True
        ):
            assert index == int(row["frame_index"])
            source_frame, timestamp = int(row["source_seq"]), float(row["video_s"])
            if not backgrounds:
                initial = combine_backgrounds([np.load(reference)], gray, verbose=False)
                backgrounds = [OnlineBackground(initial), OnlineBackground(initial)]
                trackers = [Tracker(gray.shape, TrackParams(), retain_history=False,
                                    time_based=True, max_gap_s=8 / 60) for _ in range(2)]
                labels_out = [np.zeros(gray.shape, np.uint16) for _ in range(2)]
                blur_out = np.empty_like(gray)
                blur_out.fill(0)
                pointer = blur_out.ctypes.data
            results, timings = [None, None], [{}, {}]
            for method in ((0, 1) if index % 2 else (1, 0)):
                results[method] = segment(
                    gray, backgrounds[method].B, params, kernel,
                    workspace=backgrounds[method].workspace,
                    labels_out=labels_out[method],
                    blurred_out=blur_out if method else None,
                    timings=timings[method],
                )
            expected, actual = results
            assert actual[2] is blur_out and blur_out.ctypes.data == pointer
            assert np.array_equal(expected[2].view(np.uint8), actual[2].view(np.uint8))
            assert np.array_equal(expected[0], actual[0])
            assert len(expected[1]) == len(actual[1])
            assert all(np.array_equal(a, b) for a, b in zip(expected[1], actual[1]))
            tracked = [tracker.step(source_frame, result[0], result[1], timestamp_s=timestamp)
                       for tracker, result in zip(trackers, results)]
            for field in ("rows", "events"):
                assert tracked[0][field] == tracked[1][field], (experiment, index, field)
            assert [t.id for t in tracked[0]["ended"]] == [t.id for t in tracked[1]["ended"]]
            events.update(e["type"] for e in tracked[1]["events"])
            for background, result in zip(backgrounds, results):
                background.update(result[2], result[0])
            assert np.array_equal(backgrounds[0].B, backgrounds[1].B)
            for tracker in trackers:
                for track in tracker.tracks.values():
                    assert not np.shares_memory(track.xs, blur_out)
                    assert not np.shares_memory(track.ys, blur_out)
            writer.writerow([
                index, source_frame, timestamp, gray.size, 0,
                hashlib.sha256(blur_out).hexdigest(),
                hashlib.sha256(actual[0]).hexdigest(),
                hashlib.sha256(repr((tracked[1]["rows"], tracked[1]["events"])).encode()).hexdigest(),
                timings[0]["seg_blur_ms"], timings[1]["seg_blur_ms"],
            ])
            if index in selected:
                peaks.append(measure_blur(gray, blur_out, timing_writer, experiment, index))
            if index % 300 == 0:
                print(f"Exp. {experiment}: {index + 1} frames verified", flush=True)
        count = index + 1
    assert count == provenance["expected_frames"]
    return {
        "frames": count, "pixels_checked": count * gray.size, "different_pixels": 0,
        "shape": list(gray.shape), "buffer_bytes": blur_out.nbytes,
        "tracking_equal": True, "backgrounds_equal": True, "events": dict(events),
        "traced_allocation_peaks_bytes": peaks, "inputs": provenance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opencv-threads", type=int, default=24)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(args.opencv_threads)
    report = {"environment": environment(), "videos": {}}
    try:
        with (args.output_dir / "kernel_timing.csv").open("x", newline="") as log:
            writer = csv.writer(log)
            writer.writerow(["experiment", "video_frame", "repeat", "method", "wall_ms",
                             "thread_cpu_ms", "minor_faults", "major_faults"])
            for experiment in ("A", "B"):
                report["videos"][experiment] = verify_video(
                    Path(__file__).resolve().parent, experiment, args.output_dir, writer
                )
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
