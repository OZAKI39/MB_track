"""停顿的小范围复现实验；只读原视频，复用正式分割，不推进实验追踪状态。"""

import argparse
import cProfile
import csv
import gc
import hashlib
import json
import os
import pstats
import time
from pathlib import Path

import av
import cv2
import numpy as np

if __package__:
    from .bubble_tracker import (
        SegmentationWorkspace,
        SegParams,
        combine_backgrounds,
        segment,
    )
    from .online_replay import file_hash
    from .paced_replay import paced_replay
    from .stall_probe import StallProbe
else:
    from bubble_tracker import (
        SegmentationWorkspace,
        SegParams,
        combine_backgrounds,
        segment,
    )
    from online_replay import file_hash
    from paced_replay import paced_replay
    from stall_probe import StallProbe


def conversion(video, output, threads, limit, split=False):
    """各方式单独解码同一视频，保存每帧 BGR/灰度指纹供逐像素一致核查。"""
    probe = StallProbe(output / "conversion_stalls.csv")
    cv2.setNumThreads(1)
    options = {} if threads == 0 else {"threads": threads}
    count = 0
    try:
        with (
            av.open(str(video)) as container,
            (output / "pixels.csv").open("x", newline="", encoding="utf-8") as file,
        ):
            stream = container.streams.video[0]
            stream.thread_count = 1
            frames = container.decode(stream)
            writer = csv.writer(file)
            writer.writerow(["video_frame", "pts", "bgr_sha256", "gray_sha256"])
            while not limit or count < limit:
                probe.begin(count)
                frame = next(frames, None)
                probe.mark("read_decode")
                if frame is None:
                    probe.finish()
                    break
                if split:
                    converted = frame.reformat(format="bgr24", **options)
                    probe.mark("reformat")
                    bgr = converted.to_ndarray()
                else:
                    bgr = frame.to_ndarray(format="bgr24", **options)
                probe.mark("convert_bgr")
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                probe.mark("convert_gray")
                writer.writerow(
                    [
                        count,
                        frame.pts,
                        hashlib.sha256(bgr).hexdigest(),
                        hashlib.sha256(gray).hexdigest(),
                    ]
                )
                probe.mark("hash_and_record")
                probe.finish()
                count += 1
    finally:
        probe.close()
    return {"frames": count, "conversion_threads": threads, "split_conversion": split}


def kernel(video, reference, output, threads, prefault, profile, memory_counters):
    """同一张真实首帧重复 12 次，分开首次调用与稳定开销，不更新背景。"""
    cv2.setNumThreads(threads)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        first = next(container.decode(stream))
        gray = cv2.cvtColor(first.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
    background = combine_backgrounds([np.load(reference)], gray.astype(np.float32))
    workspace = SegmentationWorkspace(background)
    if prefault:
        # 显式诊断首次写入这些内存页的成本；不修改背景和分割阈值。
        workspace._darkness.fill(0)
        workspace.component_labels.fill(0)
    params = SegParams()
    closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.close, params.close))
    probe = StallProbe(output / "kernel_stalls.csv", memory_counters=memory_counters)
    hashes = []
    profiler = cProfile.Profile()
    try:
        for index in range(12):
            timings = {}
            probe.begin(index)
            if profile and index == 0:
                profiler.enable()
            labels, contours, _ = segment(
                gray.astype(np.float32),
                background,
                params,
                closing,
                timings=timings,
                workspace=workspace,
                diagnostics=probe,
            )
            if profile and index == 0:
                profiler.disable()
            probe.finish()
            digest = hashlib.sha256(labels)
            for contour in contours:
                digest.update(contour)
            hashes.append(digest.hexdigest())
    finally:
        probe.close()
    if profile:
        profiler.dump_stats(output / "first_call.prof")
        with (output / "first_call_profile.txt").open("x") as file:
            pstats.Stats(profiler, stream=file).sort_stats("cumulative").print_stats(55)
    return {
        "opencv_threads": threads,
        "prefault": prefault,
        "profiled": profile,
        "output_hashes": hashes,
        "memory_counters": memory_counters,
        "identical_repeated_results": len(set(hashes)) == 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("kernel", "conversion", "gc_replay"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--prefault", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--memory-counters", action="store_true")
    parser.add_argument("--pzt", type=Path)
    parser.add_argument("--disable-gc", action="store_true")
    parser.add_argument("--split-conversion", action="store_true")
    args = parser.parse_args()
    if args.threads < 0 or args.max_frames < 0:
        parser.error("Thread and frame limits must be nonnegative")
    if args.mode == "kernel" and (args.reference is None or args.threads == 0):
        parser.error("Kernel mode requires a reference and positive threads")
    if args.mode == "gc_replay" and (args.reference is None or args.pzt is None):
        parser.error("GC replay requires a reference and PZT mapping")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if args.mode == "kernel":
        result = kernel(
            args.video,
            args.reference,
            args.output_dir,
            args.threads,
            args.prefault,
            args.profile,
            args.memory_counters,
        )
    elif args.mode == "conversion":
        result = conversion(
            args.video,
            args.output_dir,
            args.threads,
            args.max_frames,
            args.split_conversion,
        )
    else:
        # 只监听 GC：不逐步骤创建诊断行，检查细步骤探针是否干扰 GC 时机。
        probe = StallProbe(args.output_dir / "gc_only.csv")
        enabled = gc.isenabled()
        try:
            if args.disable_gc:
                gc.disable()
            result = paced_replay(
                args.video,
                args.reference,
                args.pzt,
                args.output_dir / "replay",
                measure_stages=True,
                opencv_threads=24,
                max_frames=args.max_frames,
            )
            result["gc_disabled_for_diagnostic"] = args.disable_gc
        finally:
            probe.close()
            if enabled:
                gc.enable()
    result.update(
        mode=args.mode,
        video=str(args.video.resolve()),
        video_sha256=file_hash(args.video),
        pid=os.getpid(),
        duration_s=time.perf_counter() - started,
        scope="Offline diagnostic; no hardware or experimental state advanced",
        code_sha256={
            name: file_hash(Path(__file__).with_name(name))
            for name in ("stall_diagnosis.py", "stall_probe.py", "bubble_tracker.py")
        },
    )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
