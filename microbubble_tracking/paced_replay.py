"""第 2 轮：按真实播放速度送帧，测量识别、写盘、跳帧和结果时效。"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import shutil
import time
import traceback
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np

if __package__:
    from .async_store import SaveSettings
    from .bubble_tracker import combine_backgrounds
    from .online_replay import file_hash, link_actions, verify_inputs
    from .online_store import export_tables
    from .online_tracking import TIMING_FIELDS, FrameStamp, OnlineProcessor
    from .paced_report import summarize
    from .paced_source import (
        DEFAULT_CONVERSION_WARMUP_ITERATIONS,
        LatestFrame,
        send_video,
    )
    from .stall_probe import HandoffTrace
else:
    from async_store import SaveSettings
    from bubble_tracker import combine_backgrounds
    from online_replay import file_hash, link_actions, verify_inputs
    from online_store import export_tables
    from online_tracking import TIMING_FIELDS, FrameStamp, OnlineProcessor
    from paced_report import summarize
    from paced_source import (
        DEFAULT_CONVERSION_WARMUP_ITERATIONS,
        LatestFrame,
        send_video,
    )
    from stall_probe import HandoffTrace


CONSUMER_FIELDS = (
    "video_frame",
    "source_frame",
    "video_s",
    "taken_s",
    "process_start_s",
    "finished_s",
    "send_late_ms",
    "queue_ms",
    "injected_ms",
    "process_ms",
    "core_ms",
    "age_ms",
    "active_tracks",
    "outcome",
)


def rss_mb(pid):
    """采样 WSL/Linux 实际驻留内存；进程已退出时留空，不虚报为零。"""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return None


def environment():
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "affinity_count": len(os.sched_getaffinity(0)),
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "numpy": np.__version__,
        "pyav": av.__version__,
        "perf_counter": vars(time.get_clock_info("perf_counter")),
        "monotonic": vars(time.get_clock_info("monotonic")),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def consume(
    processor,
    slot,
    source,
    done,
    cancel,
    output,
    origin,
    *,
    max_age_ms,
    extra_delay_ms,
    cancel_after_s,
    synthetic_replay=False,
    diagnostics=None,
    diagnose_details=False,
):
    """串行分析最新画面；异步时 process 耗时包含入队，磁盘耗时另记。"""
    next_sample = origin
    with (
        (
            HandoffTrace(output / "consumer_handoff.csv")
            if diagnose_details else nullcontext()
        ) as handoff,
        (output / "consumer_timing.csv").open(
            "x", newline="", encoding="utf-8", buffering=1
        ) as logfile,
        (output / "memory.csv").open(
            "x", newline="", encoding="utf-8", buffering=1
        ) as memory_file,
    ):
        fields = (
            (*CONSUMER_FIELDS, *TIMING_FIELDS)
            if processor.measure_stages
            else CONSUMER_FIELDS
        )
        writer = csv.DictWriter(logfile, fields)
        writer.writeheader()
        memory = csv.writer(memory_file)
        memory.writerow(
            [
                "elapsed_s",
                "consumer_rss_mb",
                "source_rss_mb",
                "active_tracks",
                "save_pending_records",
                "save_oldest_pending_ms",
            ]
        )
        while not cancel.is_set():
            if handoff is not None:
                handoff.mark("loop_control")
            now = time.perf_counter()
            if cancel_after_s and now - origin >= cancel_after_s:
                cancel.set()
                break
            if now >= next_sample:
                saving = (
                    processor.store.status() if processor.save_mode == "async" else {}
                )
                memory.writerow(
                    [
                        now - origin,
                        rss_mb(os.getpid()),
                        rss_mb(source.pid),
                        len(processor.tracker.tracks),
                        saving.get("pending_records", 0),
                        saving.get("oldest_pending_ms", 0),
                    ]
                )
                next_sample = now + 1.0
            if handoff is not None:
                handoff.mark("memory_status_sample")
            item = slot.take(handoff=handoff)
            if item is None:
                processor.poll_saved()
                if handoff is not None:
                    handoff.mark("poll_saved")
                if done.is_set():
                    # done 与 take 之间可能恰好送来末帧，再取一次防止漏尾帧。
                    item = slot.take(handoff=handoff)
                    if item is None:
                        break
                elif not source.is_alive():
                    raise RuntimeError("Sender exited without completing its report")
                else:
                    slot.available.wait(0.02)
                    if handoff is not None:
                        handoff.mark("wait_for_frame")
                    continue
            image, metadata = item
            taken = time.perf_counter()
            if handoff is not None:
                handoff.mark("take_return", ended_s=taken)
                handoff.finish()
            row = {
                "video_frame": metadata["video_frame"],
                "source_frame": metadata["source_frame"],
                "video_s": metadata["video_s"],
                "taken_s": taken,
                "send_late_ms": (metadata["published_s"] - metadata["deadline_s"])
                * 1000,
                "queue_ms": (taken - metadata["published_s"]) * 1000,
                "age_ms": (taken - metadata["deadline_s"]) * 1000,
            }
            if row["age_ms"] > max_age_ms:
                writer.writerow(dict(row, outcome="expired"))
                continue
            if extra_delay_ms and cancel.wait(extra_delay_ms / 1000):
                writer.writerow(dict(row, outcome="cancelled"))
                break
            started = time.perf_counter()
            row.update(process_start_s=started, injected_ms=(started - taken) * 1000)
            stamp = FrameStamp(
                processor.metadata["experiment_id"],
                processor.metadata["stream_id"],
                metadata["source_frame"],
                metadata["video_s"],
                video_frame=None if synthetic_replay else metadata["video_frame"],
                video_time_s=None if synthetic_replay else metadata["video_s"],
                received_windows_s=None
                if synthetic_replay
                else metadata["received_windows_s"],
                decoded_windows_s=None
                if synthetic_replay
                else metadata["decoded_windows_s"],
            )
            try:
                if diagnostics is not None:
                    diagnostics.begin(metadata["video_frame"])
                result = processor.process(image, stamp, diagnostics=diagnostics)
            except BaseException:
                writer.writerow(
                    dict(row, outcome="error", finished_s=time.perf_counter())
                )
                raise
            finished = time.perf_counter()
            if handoff is not None:
                handoff.begin(metadata["video_frame"], finished)
            if diagnostics is not None:
                diagnostics.finish(handoff=handoff)
            row.update(
                finished_s=finished,
                process_ms=(finished - started) * 1000,
                core_ms=result["frame"]["processing_ms"],
                age_ms=(finished - metadata["deadline_s"]) * 1000,
                active_tracks=len(processor.tracker.tracks),
                outcome="processed",
            )
            row.update(result.get("timings", {}))
            if handoff is not None:
                handoff.mark("consumer_row_build")
            writer.writerow(row)
            if handoff is not None:
                handoff.mark("consumer_csv")


def paced_replay(
    video,
    reference,
    pzt,
    output,
    *,
    max_frames=0,
    max_age_ms=100,
    extra_delay_ms=0,
    opencv_threads=None,
    cancel_after_s=0,
    measure_stages=False,
    save_mode="async",
    save_settings=None,
    repeats=1,
    diagnose_stalls=False,
    diagnose_details=False,
    conversion_threads=None,
    conversion_warmup_iterations=DEFAULT_CONVERSION_WARMUP_ITERATIONS,
    conversion_memory="reuse",
    gc_control=None,
):
    """1 倍速回放。所有新时间均来自同一 WSL 主机的 perf_counter。

    历史 Windows 时间仅透传给追踪记录，不参与本轮延迟相减。
    max_frames/extra_delay_ms/cancel_after_s 是显式诊断选项，默认完整正常回放。
    """
    if max_frames < 0 or max_age_ms <= 0 or extra_delay_ms < 0 or cancel_after_s < 0:
        raise ValueError("Invalid pacing limits")
    if diagnose_details and not diagnose_stalls:
        raise ValueError("Detailed diagnostics require diagnose_stalls")
    if opencv_threads is not None and opencv_threads < 1:
        raise ValueError("opencv_threads must be positive")
    if conversion_threads is not None and conversion_threads < 1:
        raise ValueError("conversion_threads must be positive")
    if conversion_memory not in ("allocate", "reuse"):
        raise ValueError("conversion_memory must be allocate or reuse")
    if (
        type(conversion_warmup_iterations) is not int
        or not 0 <= conversion_warmup_iterations <= 32
    ):
        raise ValueError("Conversion warm-up iterations must be an integer from 0 to 32")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    provenance = verify_inputs(video, reference, pzt)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        first = next(container.decode(stream))
        first_gray = cv2.cvtColor(first.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
        fps, resolution = float(stream.average_rate), float(stream.time_base)
    ref = np.load(reference, allow_pickle=False)
    if ref.shape != first_gray.shape:
        raise ValueError("Reference background and video sizes differ")
    background = combine_backgrounds([ref], first_gray.astype(np.float32))
    provenance["replay_mode"] = (
        "fixed original PTS schedule, 1x speed, latest pending frame"
    )
    prefix = "stress-replay:" if repeats > 1 else ""
    clock = (
        "synthetic repeated PTS; original mapping in source_timing.csv"
        if repeats > 1
        else "recording video PTS"
    )
    provenance["repeats"] = repeats
    # 构造成功即已完成分析端编号图的实际写入；必须先于下面的送帧进程创建/启动。
    processor = OnlineProcessor(
        experiment_id=prefix + provenance["experiment_id"],
        stream_id=prefix + "recording:" + provenance["video_sha256"],
        source_clock=clock,
        background=background,
        nominal_fps=fps,
        source_time_resolution_s=resolution,
        output_dir=output,
        provenance=provenance,
        measure_stages=measure_stages,
        diagnose_stalls=diagnose_stalls,
        save_mode=save_mode,
        save_settings=save_settings,
    )
    context = mp.get_context("spawn")
    slot = LatestFrame(context, first_gray.shape)
    ready, start, done, cancel = [context.Event() for _ in range(4)]
    origin = context.RawValue("d", 0.0)
    config = {
        "output": str(output.resolve()),
        "video": str(video.resolve()),
        "timestamps": str((pzt / "video_frames.csv").resolve()),
        "experiment_id": provenance["experiment_id"],
        "max_frames": max_frames,
        "expected_frames": provenance["expected_frames"],
        "repeats": repeats,
        "diagnose_stalls": diagnose_stalls,
        "conversion_threads": conversion_threads,
        "conversion_warmup_iterations": conversion_warmup_iterations,
        "conversion_memory": conversion_memory,
    }
    source = context.Process(
        target=send_video,
        args=(config, slot, ready, start, done, cancel, origin),
        name="paced-video-source",
    )
    previous_threads = cv2.getNumThreads()
    state, failure = "failed", None
    started_source = False
    probe = None
    try:
        if diagnose_stalls:
            if __package__:
                from .stall_probe import StallProbe
            else:
                from stall_probe import StallProbe
            probe = StallProbe(output / "consumer_stalls.csv", detailed=diagnose_details)
        if opencv_threads is not None:
            cv2.setNumThreads(opencv_threads)
        settings = {
            "environment": environment(),
            "nominal_fps": fps,
            "speed": 1.0,
            "synthetic_replay": repeats > 1,
            "repeats": repeats,
            "max_frames": max_frames,
            "max_age_ms": max_age_ms,
            "extra_delay_ms": extra_delay_ms,
            "cancel_after_s": cancel_after_s,
            "measure_stages": measure_stages,
            "diagnose_stalls": diagnose_stalls,
            "diagnose_details": diagnose_details,
            "conversion_threads": conversion_threads,
            "conversion_warmup_iterations": conversion_warmup_iterations,
            "conversion_memory": conversion_memory,
            "persistence": processor.metadata["persistence"],
            "pending_capacity": 1,
            "runtime_clock": "WSL host perf_counter shared by both local processes",
            "source_clock": clock,
            "code_hashes": {
                name: file_hash(Path(__file__).with_name(name))
                for name in (
                    "paced_replay.py",
                    "paced_source.py",
                    "frame_conversion.py",
                    "paced_report.py",
                    "online_replay.py",
                    "online_tracking.py",
                    "online_store.py",
                    "async_store.py",
                    "bubble_tracker.py",
                    "stall_probe.py",
                )
            },
        }
        (output / "pacing_settings.json").write_text(
            json.dumps(settings, indent=2) + "\n"
        )
        for name in ("actions.csv", "video_frames.csv"):
            shutil.copyfile(pzt / name, output / ("source_" + name))
        source.start()
        started_source = True
        ready_deadline = time.perf_counter() + 30
        while not ready.wait(0.05):
            if done.is_set() or not source.is_alive():
                raise RuntimeError(
                    "Sender failed before playback; see source_report.json"
                )
            if time.perf_counter() > ready_deadline:
                raise TimeoutError("Sender was not ready within 30 seconds")
        if gc_control is not None:
            gc_control.begin_stream()
        # 在真正执行 consume 的同一进程/线程预热；失败时不设置起点或发开始信号。
        processor.start(first_gray)
        origin.value = time.perf_counter() + 0.2
        start.set()
        consume(
            processor,
            slot,
            source,
            done,
            cancel,
            output,
            origin.value,
            max_age_ms=max_age_ms,
            extra_delay_ms=extra_delay_ms,
            cancel_after_s=cancel_after_s,
            synthetic_replay=repeats > 1,
            diagnostics=probe,
            diagnose_details=diagnose_details,
        )
        state = "cancelled" if cancel.is_set() else "completed"
    except KeyboardInterrupt:
        state, failure = "cancelled", "KeyboardInterrupt"
    except BaseException:  # noqa: BLE001 -- 保存失败原因，停止子进程，再保留已提交记录。
        failure = traceback.format_exc()
    finally:
        if probe is not None:
            probe.close()
        if state != "completed":
            cancel.set()
        if started_source:
            source.join(timeout=5)
            if source.is_alive():
                source.terminate()
                source.join(timeout=2)
                state, failure = (
                    "failed",
                    "Sender did not stop; its own process was terminated",
                )
        source_report_path = output / "source_report.json"
        source_report = (
            json.loads(source_report_path.read_text())
            if source_report_path.exists()
            else {}
        )
        if source_report.get("status") == "failed" or (
            state == "completed" and source_report.get("status") != "completed"
        ):
            state, failure = (
                "failed",
                source_report.get(
                    "error", "Sender did not complete the requested playback"
                ),
            )
        try:
            processor.close(state)
        except BaseException as exc:  # 收尾清理被取消也要保留已保存数据和运行状态。
            state = "cancelled" if isinstance(exc, KeyboardInterrupt) and state != "failed" else "failed"
            failure = (failure or "") + "\nSave shutdown: " + traceback.format_exc()
        finally:
            cv2.setNumThreads(previous_threads)
            # 仅离线 GC 对照提供此对象；保存线程结束后才恢复/清理。
            if gc_control is not None:
                try:
                    gc_control.finish()
                except BaseException as exc:  # 收尾清理被中断也要保留运行状态。
                    if isinstance(exc, KeyboardInterrupt) and state != "failed":
                        state = "cancelled"
                    else:
                        state = "failed"
                    failure = (failure or "") + "\nGC trial shutdown: " + traceback.format_exc()
        (output / "run_state.json").write_text(
            json.dumps(
                {"status": state, "error": failure, "origin_perf_s": origin.value},
                indent=2,
            )
            + "\n"
        )
    if state == "failed":
        raise RuntimeError(
            f"Paced replay failed; see run_state.json and writer_report.json in {output}: {failure}"
        )
    links = None
    if repeats == 1:
        links = link_actions(
            output / "tracking.sqlite",
            output / "source_actions.csv",
            output / "action_links.csv",
        )
    export_tables(output / "tracking.sqlite", output / "tables")
    report = summarize(output)
    report["action_links"] = links
    (output / "paced_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if state == "failed":
        raise RuntimeError(
            f"Paced replay failed; records preserved in {output}: {failure}"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--save-mode", choices=("sync", "async"), default="async")
    parser.add_argument("--save-queue-frames", type=int, default=32)
    parser.add_argument("--save-batch-frames", type=int, default=4)
    parser.add_argument("--save-max-pending-s", type=float, default=1.0)
    parser.add_argument("--save-checkpoint-policy", choices=("auto", "deferred"), default="auto")
    parser.add_argument("--save-wal-limit-mb", type=int, default=64)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Diagnostic repeated replay; synthetic timeline, no PZT linking",
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pzt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--opencv-threads", type=int, help="Benchmark-only OpenCV thread count"
    )
    parser.add_argument("--max-age-ms", type=float, default=100)
    parser.add_argument("--max-frames", type=int, default=0, help="0: complete video")
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Explicitly enable detailed per-frame diagnostics; off by default",
    )
    parser.add_argument(
        "--diagnose-details", action="store_true",
        help="Split contrast, selection and between-frame operations; requires --diagnose-stalls",
    )
    parser.add_argument(
        "--diagnose-stalls", action="store_true",
        help="Offline diagnostic CPU, page-fault and scheduling counters",
    )
    parser.add_argument(
        "--conversion-threads", type=int,
        help="Benchmark-only PyAV color conversion threads; default: library auto",
    )
    parser.add_argument(
        "--conversion-warmup-iterations", type=int,
        default=DEFAULT_CONVERSION_WARMUP_ITERATIONS,
        help="Startup conversions in the sender (0-32, default: 8); 0 disables for comparison",
    )
    parser.add_argument(
        "--conversion-memory", choices=("reuse", "allocate"), default="reuse",
        help="Reuse sender conversion buffers (default), or allocate for comparison",
    )
    parser.add_argument(
        "--extra-delay-ms", type=float, default=0, help="Diagnostic artificial delay"
    )
    parser.add_argument(
        "--cancel-after-s", type=float, default=0, help="Diagnostic cancellation"
    )
    parser.add_argument(
        "--measure-stages",
        action="store_true",
        help="Record per-frame processing stages",
    )
    args = parser.parse_args()
    output = args.output_dir or Path(
        __file__
    ).resolve().parent.parent / "validation" / (
        "paced_replay_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    report = paced_replay(
        args.video,
        args.reference,
        args.pzt,
        output,
        max_frames=args.max_frames,
        max_age_ms=args.max_age_ms,
        extra_delay_ms=args.extra_delay_ms,
        opencv_threads=args.opencv_threads,
        cancel_after_s=args.cancel_after_s,
        measure_stages=args.measure_stages or args.diagnostics,
        diagnose_stalls=args.diagnose_stalls or args.diagnostics,
        diagnose_details=args.diagnose_details or args.diagnostics,
        conversion_threads=args.conversion_threads,
        conversion_warmup_iterations=args.conversion_warmup_iterations,
        conversion_memory=args.conversion_memory,
        save_mode=args.save_mode,
        save_settings=SaveSettings(
            queue_frames=args.save_queue_frames,
            batch_frames=args.save_batch_frames,
            max_pending_s=args.save_max_pending_s,
            checkpoint_policy=args.save_checkpoint_policy,
            wal_limit_mb=args.save_wal_limit_mb,
        ),
        repeats=args.repeats,
    )
    print(json.dumps(report, indent=2))
    print(f"Records: {output.resolve()}")


if __name__ == "__main__":
    main()
