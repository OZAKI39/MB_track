"""Windows 独立录像与 WSL 新在线追踪共用同一视频输入；默认不控制 AFG。"""

import argparse
import csv
import json
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil

from customer_workspace.microbubble_tracking.online_replay import file_hash
from customer_workspace.microbubble_tracking.online_tracking import (
    FrameStamp,
    OnlineProcessor,
)
from customer_workspace.microbubble_tracking.stall_probe import StallProbe
from customer_workspace.parallel_camera_bridge import TrackingWindowsBridge
from customer_workspace.queue_diagnostics import append_span, scheduler_counters
from customer_workspace.streamcopy_recording import write_json
from customer_workspace.wsl_launcher import windows_path
from microbubble_experiment.camera import DEFAULT_CONFIG as CAMERA_CONFIG
from microbubble_experiment.configuration import ROOT


def add_tracking_arguments(parser, *, offline=True):
    """相机与 AFG 联合入口共用配置；背景必须明确指定，不能偷用其他视野。"""
    parser.add_argument(
        "--background",
        type=Path,
        required=True,
        help="同一视野、同一尺寸的已准备背景 .npy，灰度范围 0–255",
    )
    parser.add_argument("--camera-config", type=Path, default=CAMERA_CONFIG)
    parser.add_argument("--opencv-threads", type=int, default=24)
    parser.add_argument(
        "--waiting-frames",
        type=int,
        choices=(1, 2),
        default=1,
        help="解码后最多等待几帧；默认 1，2 仅供对照测试，满时替换最旧待取帧",
    )
    parser.add_argument(
        "--diagnose-analysis",
        action="store_true",
        help="仅短时排查：复用分析内部计时，记录 CPU、调度、缺页和自动清理",
    )
    parser.add_argument(
        "--diagnose-queue",
        action="store_true",
        help="仅短时排查：记录入队、出队、复制、锁等待和分析线程计数，并开启延迟确认",
    )
    parser.add_argument(
        "--measure-latency",
        action="store_true",
        help="仅诊断：记录各段时间，并通过原控制管道确认分析完成",
    )
    if offline:
        parser.add_argument(
            "--offline-video",
            type=Path,
            help="以原速度模拟输入；使用 Windows 同一录像和传图路线，不连接相机",
        )


def distribution(values):
    if not len(values):
        return dict.fromkeys(("median", "p95", "p99", "max"))
    return dict(
        zip(
            ("median", "p95", "p99", "max"),
            map(float, np.percentile(values, [50, 95, 99, 100])),
        )
    )


def write_frame_outcomes(output, rows):
    """逐帧说明录像窗口内哪些画面已分析；启动、结束处的缺口也必须留下。"""
    session_path = output / "gst_session.json"
    if not session_path.exists():
        return
    session = json.loads(session_path.read_text())
    bridge_path = output / "tracking_bridge.json"
    bridge = json.loads(bridge_path.read_text()) if bridge_path.exists() else {}
    decoded = {row["seq"] for row in bridge.get("decoded", [])}
    processed = {row["source_frame"] for row in rows}
    with (output / "frame_outcomes.csv").open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_frame", "video_frame", "video_time_s", "outcome"])
        for row in session["record_frames"]:
            outcome = (
                "processed"
                if row["seq"] in processed
                else (
                    "decoded_not_analyzed" if row["seq"] in decoded else "not_decoded"
                )
            )
            writer.writerow(
                [row["seq"], row["video_frame"], row["video_time_s"], outcome]
            )


def inspect_recording_result(output, metadata):
    """录像结束后逐帧核对文件，生成视频与接收/解码时刻的对应表。"""
    subprocess.run(
        [
            str(ROOT / "customer_workspace/.venv_gstreamer/Scripts/python.exe"),
            "-X",
            "utf8",
            "-B",
            "-m",
            "customer_workspace.parallel_camera",
            "--inspect",
            "--output",
            windows_path(output),
        ],
        cwd=ROOT,
        check=True,
        timeout=180,
    )
    from customer_workspace.parallel_video_report import check_branches
    from customer_workspace.streamcopy_report import make_report

    recording = make_report(output / "recording")
    session = json.loads((output / "gst_session.json").read_text())
    probe = json.loads((output / "recording/probe.json").read_text())
    checked = check_branches(session, probe, recording)
    write_json(output / "recording_check.json", checked)
    from customer_workspace.afg_video_report import linked_frames

    mapped = linked_frames(session, probe, metadata["experiment_id"])
    bridge_log = json.loads((output / "tracking_bridge.json").read_text())
    decoded = {row["seq"]: row["decoded_wsl_s"] for row in bridge_log["decoded"]}
    for row in mapped:
        row["stream_id"] = metadata["stream_id"]
        row["decoded_wsl_s"] = decoded.get(row["source_seq"])
    with (output / "video_frames.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(mapped[0]))
        writer.writeheader()
        writer.writerows(mapped)
    return checked


def run_check(args, control=None):
    """先预热实际分析线程，再允许录制；收尾先关断 AFG，最后排空已提交记录。"""
    offline_video = getattr(args, "offline_video", None)
    diagnose_queue = getattr(args, "diagnose_queue", False)
    diagnose_analysis = getattr(args, "diagnose_analysis", False)
    measure_latency = (
        getattr(args, "measure_latency", False) or diagnose_queue or diagnose_analysis
    )
    waiting_frames = getattr(args, "waiting_frames", 1)
    if waiting_frames not in (1, 2):
        raise ValueError("Waiting capacity must be 1 or 2 frames")
    latency_rows = []
    loop_events = []
    analysis_probe = None
    stage_rows = []
    if offline_video and control is not None:
        raise ValueError("Archived video simulation must never run an AFG controller")
    if not 1 <= args.seconds <= 90:
        raise ValueError("Capture duration must be 1–90 seconds")
    background_path = Path(args.background).resolve()
    background = np.load(background_path, allow_pickle=False).astype(np.float32)
    if (
        background.ndim != 2
        or not np.isfinite(background).all()
        or background.min() < 0
        or background.max() > 255
    ):
        raise ValueError("Background must be a finite 2D grayscale image in 0–255")
    if offline_video and not Path(offline_video).is_file():
        raise FileNotFoundError(offline_video)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_threads = cv2.getNumThreads()
    threads = getattr(args, "opencv_threads", 24)
    if threads < 1:
        raise ValueError("OpenCV thread count must be positive")
    cv2.setNumThreads(threads)
    processor = bridge = None
    rows, resources, errors = [], [], []
    window = None
    runtime = {"pipeline_start_wsl_s": time.perf_counter()}
    save_summary = {}
    status = "completed"
    cancelled = threading.Event()
    previous_sigint = None
    if threading.current_thread() is threading.main_thread():
        previous_sigint = signal.getsignal(signal.SIGINT)
        # 先完成当前帧的计算与提交，再处理取消，避免打断已提交但尚未登记的帧。
        signal.signal(signal.SIGINT, lambda *_: cancelled.set())
    try:
        if control is not None:
            control.start(output)
        link_file = output / "link_session.json"
        session = (
            json.loads(link_file.read_text())
            if link_file.exists()
            else {
                "experiment_id": str(uuid.uuid4()),
            }
        )
        if not link_file.exists():
            session["stream_id"] = str(uuid.uuid4())
            session["input_kind"] = "offline_video" if offline_video else "camera"
            write_json(link_file, session)
        config = {
            "engine": "microbubble_tracking.OnlineProcessor",
            "background": str(background_path),
            "background_sha256": file_hash(background_path),
            "shape": list(background.shape),
            "nominal_fps": 60,
            "opencv_threads": threads,
            "diagnose_stalls": False,
            "measure_latency": measure_latency,
            "diagnose_queue": diagnose_queue,
            "diagnose_analysis": diagnose_analysis,
            "waiting_frames": waiting_frames,
            "source_clock": "GStreamer source PTS",
            "mode": args.mode,
            "offline_video": str(Path(offline_video).resolve())
            if offline_video
            else None,
            "camera_config": str(getattr(args, "camera_config", CAMERA_CONFIG)),
            "transport_format": "H264",
            "analysis_format": "PyAV BGR24 -> OpenCV GRAY8",
            "recording": args.mode == "parallel",
            "source_frame_meaning": "received access-unit sequence; not a camera exposure counter",
            "interface_sha256": {
                name: file_hash(Path(__file__).with_name(name))
                for name in (
                    "parallel_camera.py",
                    "parallel_camera_bridge.py",
                    "parallel_video_check.py",
                )
            },
        }
        write_json(output / "analysis_config.json", config)
        processor = OnlineProcessor(
            experiment_id=session["experiment_id"],
            stream_id=session["stream_id"],
            source_clock="GStreamer source PTS",
            background=background,
            nominal_fps=60,
            source_time_resolution_s=1 / 90000,
            output_dir=output / "tracking",
            provenance=config,
            measure_stages=diagnose_analysis,
        )
        processor.start()
        if diagnose_analysis:
            analysis_probe = StallProbe(
                output / "consumer_stalls.csv", buffer_frames=2048, scheduler=True
            )
        runtime["analysis_ready_wsl_s"] = time.perf_counter()
        bridge = TrackingWindowsBridge(
            output,
            args.seconds,
            background.shape,
            camera_config=getattr(args, "camera_config", CAMERA_CONFIG),
            offline_video=offline_video,
            recording=args.mode == "parallel",
            measure_latency=measure_latency,
            waiting_frames=waiting_frames,
            diagnose_queue=diagnose_queue,
        ).start()
        armed = False
        next_poll = next_memory = 0
        memory_process = psutil.Process()
        receiver_memory = (
            psutil.Process(bridge.process.pid)
            if getattr(bridge.process, "pid", None)
            else None
        )
        deadline = time.perf_counter() + args.seconds + 60
        while time.perf_counter() < deadline:
            if cancelled.is_set():
                raise KeyboardInterrupt("Cancelled after completing the current frame")
            now = time.perf_counter()
            if now >= next_poll:
                if diagnose_queue:
                    span_start, span_cpu = time.perf_counter(), time.thread_time()
                next_poll = now + 0.05
                bridge.poll()
                if control is not None:
                    control.poll()
                if not armed and bridge.ready and (control is None or control.ready()):
                    write_json(
                        output / "start_recording.request", {"seconds": args.seconds}
                    )
                    armed = True
                if (
                    "recording_start_observed_wsl_s" not in runtime
                    and (output / "recording_started.json").exists()
                ):
                    runtime["recording_start_observed_wsl_s"] = now
                processor.poll_saved()
                if diagnose_queue:
                    append_span(
                        loop_events, "status_and_save_poll", span_start, span_cpu
                    )
            if now >= next_memory:
                if diagnose_queue:
                    span_start, span_cpu = time.perf_counter(), time.thread_time()
                memory = {
                    "elapsed_s": now - runtime["pipeline_start_wsl_s"],
                    "rss_mib": memory_process.memory_info().rss / 2**20,
                }
                if receiver_memory is not None:
                    try:
                        memory["decoder_rss_mib"] = (
                            receiver_memory.memory_info().rss / 2**20
                        )
                    except psutil.NoSuchProcess:
                        memory["decoder_rss_mib"] = None
                resources.append(memory)
                next_memory = now + 1
                if diagnose_queue:
                    append_span(loop_events, "memory_sample", span_start, span_cpu)
            if bridge.error:
                raise RuntimeError(bridge.error)
            packet = bridge.frames.take()
            if packet is not None:
                buffer_index, image, header = packet
                started = time.perf_counter()
                try:
                    stamp = FrameStamp(
                        experiment_id=header["experiment_id"],
                        stream_id=header["stream_id"],
                        source_frame=header["seq"],
                        source_time_s=header["source_time_s"],
                        video_frame=header["video_frame"],
                        video_time_s=header["video_time_s"],
                        received_windows_s=header["received_windows_mono"],
                        decoded_windows_s=None,
                    )
                    if diagnose_queue:
                        before_counters = scheduler_counters()
                        processor_cpu = time.thread_time()
                        processor_started = time.perf_counter()
                    if analysis_probe is not None:
                        analysis_probe.begin(header["seq"])
                    result = processor.process(image, stamp, diagnostics=analysis_probe)
                    completed = time.perf_counter()
                    if analysis_probe is not None:
                        analysis_probe.finish()
                        stage_rows.append(
                            {
                                "source_frame": header["seq"],
                                "video_frame": header["video_frame"],
                                **result["timings"],
                            }
                        )
                    if diagnose_queue:
                        processor_cpu_ms = (time.thread_time() - processor_cpu) * 1000
                        post_cpu = time.thread_time()
                    if measure_latency:
                        bridge.acknowledge(header, completed)
                        details = {}
                        if diagnose_queue:
                            after_counters = scheduler_counters()
                            details = {
                                **header["queue_detail"],
                                "processor_started_wsl_s": processor_started,
                                "processor_thread_cpu_ms": processor_cpu_ms,
                                "processor_counters": {
                                    key: after_counters[key] - before_counters[key]
                                    for key in before_counters
                                },
                            }
                        latency_rows.append(
                            {
                                "seq": header["seq"],
                                "video_frame": header["video_frame"],
                                **header["latency"],
                                "analysis_started_wsl_s": started,
                                "analysis_completed_wsl_s": completed,
                                **details,
                            }
                        )
                    rows.append(
                        {
                            **result["frame"],
                            "completed_wsl_s": completed,
                            "processing_call_ms": (completed - started) * 1000,
                            "queue_wait_ms": (started - header["published_wsl_s"])
                            * 1000,
                            "decode_and_convert_ms": (
                                header["decoded_wsl_s"] - header["arrived_wsl_s"]
                            )
                            * 1000,
                            "decoded_wsl_s": header["decoded_wsl_s"],
                            "request_ms": header["request_ms"],
                            "software_age_upper_bound_ms": header["sender_frame_age_ms"]
                            + header["request_ms"]
                            + (completed - header["arrived_wsl_s"]) * 1000,
                        }
                    )
                finally:
                    bridge.frames.release(buffer_index)
                if diagnose_queue:
                    append_span(
                        loop_events,
                        "post_analysis_and_release",
                        completed,
                        post_cpu,
                        seq=header["seq"],
                    )
            elif bridge.done.is_set():
                bridge.poll()
                if bridge.frames.pending.value >= 0:
                    continue
                if bridge.window is None:
                    raise RuntimeError(
                        bridge.error or "Receiver ended without a complete window"
                    )
                if control is None or control.finished():
                    window = bridge.window
                    break
                time.sleep(0.002)
            else:
                if diagnose_queue:
                    span_start, span_cpu = time.perf_counter(), time.thread_time()
                with bridge.frames.condition:
                    if bridge.frames.pending.value < 0:
                        bridge.frames.condition.wait(0.002)
                if diagnose_queue:
                    append_span(loop_events, "wait_for_frame", span_start, span_cpu)
        else:
            raise TimeoutError(
                "Camera/tracking window did not complete within the time limit"
            )
    except BaseException as exc:  # noqa: BLE001 -- 实验边界必须执行保存和设备收尾。
        status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        # 独立 AFG 执行器必须先完成关断；不能由图像线程强杀设备进程。
        for action in ([control.stop] if control is not None else []) + (
            [bridge.stop] if bridge is not None else []
        ):
            try:
                action()
            except BaseException as exc:  # noqa: BLE001 -- 保留错误，仍继续其余收尾。
                errors.append(f"Shutdown {type(exc).__name__}: {exc}")
        if processor is not None:
            try:
                save_summary = processor.close(
                    "cancelled"
                    if status == "cancelled"
                    else "failed"
                    if errors
                    else status
                )
            except BaseException as exc:  # noqa: BLE001 -- 保存失败不能冒充正常完成。
                errors.append(f"Saving {type(exc).__name__}: {exc}")
        if analysis_probe is not None:
            try:
                analysis_probe.close()
            except Exception as exc:  # noqa: BLE001 -- 诊断收尾失败也应保留实验记录。
                errors.append(f"Analysis diagnostics {type(exc).__name__}: {exc}")
            write_json(output / "analysis_stages.json", {"frames": stage_rows})
            write_json(
                output / "analysis_finished.json",
                {"finished_wsl_s": time.perf_counter()},
            )
        cv2.setNumThreads(previous_threads)
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        write_json(output / "runtime_origin.json", runtime)
    if (
        bridge is not None
        and bridge.error
        and not any(bridge.error in error for error in errors)
    ):
        errors.append(bridge.error)
    saved = save_summary.get("frames", 0)
    if saved != len(rows):
        errors.append(f"Analyzed/saved frame count mismatch: {len(rows)}/{saved}")
    if rows:
        with (output / "analysis_frames.csv").open("x", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_frame_outcomes(output, rows)
    if measure_latency:
        write_json(output / "latency_wsl_analysis.json", {"frames": latency_rows})
    if diagnose_queue:
        write_json(output / "queue_loop.json", {"events": loop_events})
    window = window or (bridge.window if bridge else None)
    write_json(output / "measurement_window.json", window)
    skipped = max(0, (window or {}).get("frames", 0) - len(rows))
    summary = {
        "error": errors or None,
        "resources": resources,
        "threads_closed": bridge is None or not bridge.process.is_alive(),
        "recorder_dropped": None,
        "analysis_skipped": skipped,
        "bridge_overwritten": bridge.frames.overwritten if bridge else 0,
        "tracking": save_summary,
    }
    write_json(output / "summary.json", summary)
    report = {
        "engine": "microbubble_tracking.OnlineProcessor",
        "window": window,
        "processed": len(rows),
        "saved": saved,
        "analysis_hz": len(rows) / args.seconds if window is not None else None,
        "window_completed": window is not None,
        "all_window_frames_analyzed": window is not None
        and len(rows) == window["frames"],
        "actual_tracking_results": len(rows),
        "processing_ms": distribution([r["processing_call_ms"] for r in rows]),
        "software_age_upper_bound_ms": distribution(
            [r["software_age_upper_bound_ms"] for r in rows]
        ),
        "completion_interval_ms": distribution(
            np.diff([r["completed_wsl_s"] for r in rows]) * 1000
        ),
        **{
            k: summary[k]
            for k in (
                "recorder_dropped",
                "analysis_skipped",
                "bridge_overwritten",
                "threads_closed",
            )
        },
        "pipeline_error": errors or None,
        "latency_scope": "Windows receipt to WSL result conservative bound; not exposure time",
        "tracking_summary": save_summary,
    }
    if errors:
        write_json(output / "analysis_check.json", report)
        raise RuntimeError("; ".join(errors))
    if args.mode == "parallel":
        try:
            checked = inspect_recording_result(output, processor.metadata)
            report["recording_passed"] = checked["passed"]
            report["recorder_dropped"] = checked["recording_extra_lost"]
        except BaseException as exc:
            report["recording_passed"] = False
            report["pipeline_error"] = [
                f"Recording validation {type(exc).__name__}: {exc}"
            ]
            write_json(output / "analysis_check.json", report)
            raise
    write_json(output / "analysis_check.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "parallel"), default="parallel")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--output-dir", type=Path)
    add_tracking_arguments(parser)
    args = parser.parse_args()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = (
            ROOT / "customer_workspace/validation" / f"camera_tracking_{stamp}"
        )
    report = run_check(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(
        bool(report["pipeline_error"] or not report.get("recording_passed", True))
    )


if __name__ == "__main__":
    raise SystemExit(main())
