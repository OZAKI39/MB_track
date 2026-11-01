"""离线核对 A/B 全视频的编号图、轮廓、轨迹和事件，并单独测重编号开销。"""

import argparse
import csv
import hashlib
import json
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .bubble_tracker import (
        OnlineBackground,
        SegParams,
        Tracker,
        TrackParams,
        combine_backgrounds,
        iter_frames,
        segment,
    )
    from .online_replay import verify_inputs
    from .paced_replay import environment
else:
    from bubble_tracker import (
        OnlineBackground,
        SegParams,
        Tracker,
        TrackParams,
        combine_backgrounds,
        iter_frames,
        segment,
    )
    from online_replay import verify_inputs
    from paced_replay import environment


def measure_remap(labels, workspace, writer, experiment, frame):
    """使用真实区域图和明确的试验映射表；不把映射后的图交给正式追踪。"""
    lookup = np.arange(int(labels.max()) + 1, dtype=np.uint16)
    lookup[1::2] = 0  # 覆盖删除目标和背景，原编号不必连续。
    expected = lookup[labels]
    target = labels.copy()
    workspace.remap_labels(target, lookup)
    assert np.array_equal(target, expected)
    # 预热之后交错顺序。数据恢复不计入重编号计时；两种方法输入完全相同。
    for repetition in range(20):
        methods = ("index", "reuse") if repetition % 2 else ("reuse", "index")
        for method in methods:
            np.copyto(target, labels)
            started = time.perf_counter()
            if method == "index":
                result = lookup[target]
            else:
                workspace.remap_labels(target, lookup)
                result = target
            elapsed = (time.perf_counter() - started) * 1000
            assert np.array_equal(result, expected)
            writer.writerow([experiment, frame, repetition, method, elapsed])
    peaks = {}
    for method in ("index", "reuse"):
        np.copyto(target, labels)
        tracemalloc.start()
        try:
            if method == "index":
                result = lookup[target]
            else:
                workspace.remap_labels(target, lookup)
            peaks[method] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    return peaks


def verify_video(code, experiment, reference_name, output, timing_writer):
    """共用现有分割函数的独立分配入口作为对照，不复制识别规则。"""
    video = code / f"data/raw/exp{experiment}.mp4"
    reference = code / f"results/{reference_name}_offline/background.npy"
    pzt = code / f"data/pzt/exp{experiment}"
    inputs = verify_inputs(video, reference, pzt)
    selected = set(np.linspace(0, inputs["expected_frames"] - 1, 12, dtype=int))
    params = SegParams()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.close, params.close))
    backgrounds = []
    trackers = []
    events = Counter()
    peaks = []
    with (
        (pzt / "video_frames.csv").open(encoding="utf-8-sig", newline="") as source,
        (output / f"exp{experiment}_frames.csv").open("x", newline="") as log,
    ):
        writer = csv.writer(log)
        writer.writerow(
            [
                "video_frame",
                "source_frame",
                "source_time_s",
                "regions",
                "labels_sha256",
                "contours_sha256",
                "tracking_sha256",
            ]
        )
        frames = iter_frames(str(video))
        for (index, gray), row in zip(frames, csv.DictReader(source), strict=True):
            assert index == int(row["frame_index"])
            assert row["experiment_id"] == inputs["experiment_id"]
            source_frame, timestamp = int(row["source_seq"]), float(row["video_s"])
            if not backgrounds:
                initial = combine_backgrounds([np.load(reference)], gray, verbose=False)
                backgrounds = [OnlineBackground(initial), OnlineBackground(initial)]
                trackers = [
                    Tracker(
                        gray.shape,
                        TrackParams(),
                        retain_history=False,
                        time_based=True,
                        max_gap_s=8 / 60,
                    )
                    for _ in range(2)
                ]
                label_output = np.zeros(gray.shape, np.uint16)
            expected, expected_contours, expected_blurred = segment(
                gray, backgrounds[0].B, params, kernel
            )
            labels, contours, blurred = segment(
                gray,
                backgrounds[1].B,
                params,
                kernel,
                workspace=backgrounds[1].workspace,
                labels_out=label_output,
            )
            assert labels is label_output
            assert np.array_equal(labels, expected), (experiment, index, "labels")
            assert np.array_equal(blurred, expected_blurred)
            assert len(contours) == len(expected_contours)
            assert all(
                np.array_equal(a, b) for a, b in zip(contours, expected_contours)
            )
            results = [
                tracker.step(source_frame, image, edges, timestamp_s=timestamp)
                for tracker, image, edges in zip(
                    trackers, (expected, labels), (expected_contours, contours)
                )
            ]
            for field in ("rows", "events"):
                assert results[0][field] == results[1][field], (
                    experiment,
                    index,
                    field,
                )
            assert [t.id for t in results[0]["ended"]] == [
                t.id for t in results[1]["ended"]
            ]
            events.update(e["type"] for e in results[1]["events"])
            backgrounds[0].update(expected_blurred, expected)
            backgrounds[1].update(blurred, labels)
            assert np.array_equal(backgrounds[0].B, backgrounds[1].B)
            contour_hash = hashlib.sha256()
            for contour in contours:
                contour_hash.update(contour)
            tracking_hash = hashlib.sha256(
                repr((results[1]["rows"], results[1]["events"])).encode()
            ).hexdigest()
            writer.writerow(
                [
                    index,
                    source_frame,
                    timestamp,
                    len(contours),
                    hashlib.sha256(labels).hexdigest(),
                    contour_hash.hexdigest(),
                    tracking_hash,
                ]
            )
            if index in selected:
                peaks.append(
                    measure_remap(
                        labels,
                        backgrounds[1].workspace,
                        timing_writer,
                        experiment,
                        index,
                    )
                )
            if index % 300 == 0:
                print(f"Verified Exp. {experiment}: {index + 1} frames", flush=True)
        count = index + 1
    assert count == inputs["expected_frames"]
    return {
        "frames": count,
        "inputs": inputs,
        "all_equal": True,
        "events": dict(events),
        "remap_temporary_peak_bytes": peaks,
        "fixed_selection_bytes": backgrounds[1].workspace.selection_mask.nbytes,
        "fixed_index_bytes": backgrounds[1].workspace._label_indices.nbytes,
        "fixed_output_bytes": label_output.nbytes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(24)
    report = {"environment": environment(), "videos": {}}
    try:
        with (args.output_dir / "remap_timing.csv").open("x", newline="") as log:
            writer = csv.writer(log)
            writer.writerow(["experiment", "video_frame", "repetition", "method", "ms"])
            for experiment, reference in (("A", "B"), ("B", "A")):
                report["videos"][experiment] = verify_video(
                    Path(__file__).resolve().parent,
                    experiment,
                    reference,
                    args.output_dir,
                    writer,
                )
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        (args.output_dir / "verification.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
