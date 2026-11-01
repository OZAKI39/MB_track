"""离线首帧对照；cold 表示无额外诊断准备，仍执行当前正式启动预热。"""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import av
import cv2
import numpy as np

from customer_workspace.microbubble_tracking.bubble_tracker import (
    Tracker,
    combine_backgrounds,
    segment,
)
from customer_workspace.microbubble_tracking.gc_replay_check import GCTrial
from customer_workspace.microbubble_tracking.online_tracking import FrameStamp, OnlineProcessor
from customer_workspace.microbubble_tracking.paced_replay import environment
from customer_workspace.microbubble_tracking.stall_probe import StallProbe


CONDITIONS = ("cold", "touch", "labels", "darkness", "components", "analysis")


def prepare(processor, gray, condition):
    """仅用于对照；不推进正式背景、帧号、目标编号或保存记录。

    touch 访问全部已有工作区，labels/darkness 分别只访问编号图/暗度图；
    components 只试运行区域标号函数；analysis 试运行现有分割和独立的
    临时追踪器。正式追踪器从未接收测试帧。
    """
    if condition not in CONDITIONS:
        raise ValueError("Unknown diagnostic condition")
    if processor.last_stamp is not None or processor.background.k != 0:
        raise ValueError("Diagnostic preparation requires a fresh processor")
    workspace = processor.background.workspace
    if condition == "touch":
        for array in (
            workspace._darkness,
            workspace.component_labels,
            workspace.selection_mask,
            workspace._label_indices,
            processor._labels,
        ):
            array.fill(0)
    elif condition == "labels":
        workspace.component_labels.fill(0)
    elif condition == "darkness":
        workspace._darkness.fill(0)
    elif condition == "components":
        mask = np.zeros(gray.shape, np.uint8)
        height, width = gray.shape
        mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = 1
        cv2.connectedComponentsWithStats(
            mask, labels=workspace.component_labels, connectivity=8
        )
    elif condition == "analysis":
        labels, contours, _ = segment(
            gray.astype(np.float32), processor.background.B,
            processor.seg, processor.kernel,
            workspace=workspace, labels_out=processor._labels,
        )
        temporary = Tracker(
            gray.shape, processor.tracker.tp, retain_history=False,
            time_based=True, max_gap_s=processor.tracker.max_gap_s,
        )
        temporary.step(0, labels, contours, timestamp_s=0.0)
        processor._labels.fill(0)


def run(video, reference, output, *, condition, frames=60, threads=24,
        diagnostics=False):
    """直接测同一处理接口，每种条件应在独立进程运行。

    使用真实前 N 帧、60 Hz 左右的固定播放时刻；此处不设最新帧替换队列，
    所有 N 帧都会处理，便于精确核对算法结果。这不是摄像头吞吐验收。
    """
    output.mkdir(parents=True, exist_ok=False)
    previous_threads = cv2.getNumThreads()
    rows = []
    processor = None
    probe = None
    status = "failed"
    try:
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            decoded = container.decode(stream)
            first = next(decoded)
            first_gray = cv2.cvtColor(first.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
            fps = float(stream.average_rate)
            background = combine_backgrounds(
                [np.load(reference, allow_pickle=False)], first_gray.astype(np.float32)
            )
            processor = OnlineProcessor(
                experiment_id="first-frame-diagnostic", stream_id="A-prefix",
                source_clock="synthetic diagnostic schedule: frame / nominal FPS",
                background=background,
                nominal_fps=fps, output_dir=output / "tracking", measure_stages=True,
            )
            cv2.setNumThreads(threads)
            initial_hash = hashlib.sha256(processor.background.B).hexdigest()
            before_threads = len(os.listdir("/proc/self/task"))
            started = time.perf_counter()
            prepare(processor, first_gray, condition)
            preparation_ms = (time.perf_counter() - started) * 1000
            assert hashlib.sha256(processor.background.B).hexdigest() == initial_hash
            assert processor.background.k == 0 and processor.last_stamp is None
            assert processor.tracker.next_id == 1 and not processor.tracker.tracks
            assert processor.store.status()["pending_records"] == 0
            after_threads = len(os.listdir("/proc/self/task"))
            if diagnostics:
                probe = StallProbe(output / "stages.csv", detailed=True)
            with GCTrial(output, "normal") as trial:
                try:
                    trial.begin_stream()
                    processor.start(first_gray)
                    origin = time.perf_counter() + 0.2
                    for index in range(frames):
                        if index == 0:
                            gray = first_gray
                        else:
                            frame = next(decoded)
                            gray = cv2.cvtColor(
                                frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY
                            )
                        deadline = origin + index / fps
                        time.sleep(max(0.0, deadline - time.perf_counter()))
                        stamp = FrameStamp(
                            processor.experiment_id, processor.stream_id,
                            index, index / fps,
                        )
                        if probe is not None:
                            probe.begin(index)
                        started = time.perf_counter()
                        result = processor.process(gray, stamp, diagnostics=probe)
                        finished = time.perf_counter()
                        if probe is not None:
                            probe.finish()
                        rows.append({
                            "frame": index, "start_s": started, "end_s": finished,
                            "wall_ms": (finished - started) * 1000,
                            **result["timings"],
                        })
                    status = "completed"
                finally:
                    processor.close(status)
        with (output / "timings.csv").open("x", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = {
            "condition": condition, "frames": frames, "environment": environment(),
            "diagnostics": diagnostics, "preparation_ms": preparation_ms,
            "formal_startup_preparation": processor.metadata["startup_preparation"],
            "analysis_preparation": json.loads(
                (processor.output / "analysis_preparation.json").read_text()
            ),
            "threads_before": before_threads, "threads_after_preparation": after_threads,
            "scientific_state_unchanged_after_preparation": True,
            "diagnostic_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Offline diagnostic; prefix processed without input replacement queue",
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"condition": condition, "first_ms": rows[0]["wall_ms"]}))
        return result
    finally:
        if processor is not None and not processor.closed:
            processor.close(status)
        if probe is not None:
            probe.close()
        cv2.setNumThreads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.frames <= 120 or args.threads < 1:
        parser.error("Use 1–120 frames and a positive thread count")
    code = Path(__file__).resolve().parent
    run(
        code / "data/raw/expA.mp4", code / "results/B_offline/background.npy",
        args.output_dir, condition=args.condition, frames=args.frames,
        threads=args.threads, diagnostics=args.diagnostics,
    )


if __name__ == "__main__":
    main()
