"""录制并检查一段真实 Chronos 预览视频；复用接收器和录像器，不访问 AFG。

在 WSL 中运行。20 秒从收到第一帧算起，窗口之外的帧不写入录像。
检查逐帧时间、接收跳号、编码丢帧以及两份 MP4 的完整解码和帧对应。
"""

import argparse
import csv
import json
import math
import threading
import time
import traceback
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import av
import numpy as np

from microbubble_experiment.camera_bridge import LiveCameraFrameSource
from microbubble_experiment.configuration import ROOT, load_config
from microbubble_experiment.recording import DEFAULT_LAYERS, ExperimentVideoRecorder

GAP_LIMIT_S = 0.100
FIRST_FRAME_TIMEOUT_S = 15.0


def write_json(path: Path, value) -> None:
    """只创建新证据文件，禁止覆盖已有结果。"""
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def frame_keys(rows: list[dict]) -> list[tuple]:
    """用连接、来源帧号和收到时间识别同一帧，不依赖画面是否发生变化。"""
    return [
        (row["source_session_id"], row["source_frame_id"], row["arrival_monotonic"])
        for row in rows
    ]


def interval_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median_ms": None, "p99_ms": None, "maximum_ms": None}
    values_ms = np.asarray(values) * 1000
    return {
        "count": len(values),
        "median_ms": float(np.median(values_ms)),
        "p99_ms": float(np.percentile(values_ms, 99)),
        "maximum_ms": float(np.max(values_ms)),
    }


def check_frame_times(rows: list[dict], duration_s: float) -> tuple[dict, list[dict]]:
    """核对原始时间，不按名义帧率补时间；Windows 与 WSL 时钟分别比较。"""
    problems, table, arrival_intervals, decode_intervals = [], [], [], []
    missing_source_frames = 0
    discontinuities = []
    if not rows:
        return {"passed": False, "problems": ["没有保存任何帧"]}, []
    origin = rows[0]["arrival_monotonic"]
    previous = None
    for index, row in enumerate(rows):
        arrival = row["arrival_monotonic"]
        decode = row["host_decode_timestamp"].get("value")
        if not isinstance(decode, (float, int)) or not math.isfinite(decode):
            problems.append(f"第 {index + 1} 帧缺少有效的 Windows 解码时间")
        if not isinstance(arrival, (float, int)) or not math.isfinite(arrival):
            raise ValueError(f"第 {index + 1} 帧缺少有效的 WSL 接收时间")
        if row["record_index"] != index:
            problems.append(f"第 {index + 1} 条帧记录编号不连续")
        if row["source_mode"] != "LIVE_CAMERA":
            problems.append(f"第 {index + 1} 帧不是真实相机来源")
        if row["host_decode_timestamp"].get("clock") != "windows_monotonic":
            problems.append(f"第 {index + 1} 帧的解码时钟来源不符")
        if row["host_receive_timestamp"].get("clock") != "wsl_monotonic":
            problems.append(f"第 {index + 1} 帧的接收时钟来源不符")
        if row["host_receive_timestamp"].get("value") != arrival:
            problems.append(f"第 {index + 1} 帧的接收时间字段不一致")
        if row["pts_adjusted"]:
            problems.append(f"第 {index + 1} 帧发生播放时间修正")
        elapsed = arrival - origin
        if abs(elapsed - row["elapsed_since_first_frame_s"]) > 1e-8:
            problems.append(f"第 {index + 1} 帧相对时间不一致")
        delta, skipped = None, 0
        if previous:
            delta = arrival - previous["arrival_monotonic"]
            arrival_intervals.append(delta)
            if delta <= 0 or delta > GAP_LIMIT_S:
                discontinuities.append(
                    {"frame": index + 1, "clock": "wsl_receive", "interval_s": delta}
                )
            if row["source_session_id"] != previous["source_session_id"]:
                problems.append(f"第 {index + 1} 帧发生相机连接切换")
            else:
                sequence_delta = row["source_frame_id"] - previous["source_frame_id"]
                skipped = max(0, sequence_delta - 1)
                missing_source_frames += skipped
                if sequence_delta <= 0:
                    problems.append(f"第 {index + 1} 帧来源编号重复或倒退")
                prior_decode = previous["host_decode_timestamp"].get("value")
                if all(
                    isinstance(t, (float, int)) and math.isfinite(t)
                    for t in (decode, prior_decode)
                ):
                    decode_delta = decode - prior_decode
                    decode_intervals.append(decode_delta)
                    if decode_delta <= 0 or decode_delta > GAP_LIMIT_S:
                        discontinuities.append(
                            {
                                "frame": index + 1,
                                "clock": "windows_decode",
                                "interval_s": decode_delta,
                            }
                        )
        table.append(
            {
                "录像帧号": index + 1,
                "录像时间_秒": elapsed,
                "距上一帧_毫秒": None if delta is None else delta * 1000,
                "来源帧号": row["source_frame_id"],
                "缺少的来源帧数": skipped,
                "相机连接编号": row["source_session_id"],
                "Windows解码时间_秒": decode,
                "WSL收到时间_秒": arrival,
                "视频分段编号": row["part"],
                "分段内帧号": row["part_frame_index"],
                "分段内播放时间_秒": row["video_pts_us"] / 1e6,
            }
        )
        previous = row
    span = rows[-1]["arrival_monotonic"] - origin
    if span < duration_s - GAP_LIMIT_S or span >= duration_s:
        problems.append(
            f"画面首尾跨度 {span:.6f} 秒，未覆盖要求的 {duration_s:g} 秒窗口"
        )
    if missing_source_frames:
        problems.append(f"保存的相邻帧之间，来源编号共跳过 {missing_source_frames} 帧")
    if discontinuities:
        problems.append(
            f"发现 {len(discontinuities)} 处时间倒退、重复或超过 100 ms 的间隔"
        )
    return {
        "passed": not problems,
        "problems": problems,
        "frames": len(rows),
        "first_to_last_frame_s": span,
        "observed_receive_fps": (len(rows) - 1) / span if span > 0 else None,
        "source_sequence_missing_frames": missing_source_frames,
        "arrival_intervals": interval_stats(arrival_intervals),
        "decode_intervals": interval_stats(decode_intervals),
        "gap_limit_ms": GAP_LIMIT_S * 1000,
        "discontinuities": discontinuities,
    }, table


def check_video(path: Path, rows: list[dict], expected_wh: list[int]) -> dict:
    """从头到尾解码 MP4，逐帧核对播放时间和尺寸，不一次载入整段视频。"""
    problems = []
    count = 0
    container_duration_s = None
    try:
        with av.open(str(path)) as container:
            if container.duration is not None:
                container_duration_s = container.duration / av.time_base
            for frame in container.decode(video=0):
                if frame.pts is None or frame.time_base is None:
                    problems.append(f"第 {count + 1} 帧缺少播放时间")
                elif count < len(rows):
                    actual_s = float(frame.pts * frame.time_base)
                    expected_s = rows[count]["video_pts_us"] / 1e6
                    if abs(actual_s - expected_s) > 2e-6:
                        problems.append(f"第 {count + 1} 帧播放时间与记录不符")
                if [frame.width, frame.height] != expected_wh:
                    problems.append(f"第 {count + 1} 帧尺寸不符")
                count += 1
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        problems.append(f"无法完整解码：{type(exc).__name__}: {exc}")
    if count != len(rows) or count == 0:
        problems.append(f"视频解码 {count} 帧，逐帧记录 {len(rows)} 条")
    return {
        "file": path.name,
        "passed": not problems,
        "problems": problems,
        "decoded_frames": count,
        "metadata_rows": len(rows),
        "container_duration_s": container_duration_s,
    }


class RecordingWindow:
    """限制录像窗口并保留每次入队结果；不在相机回调里编码或写文件。"""

    def __init__(self, recorder, duration_s):
        self.recorder = recorder
        self.duration_s = duration_s
        self.first_arrival = None
        self.accepting = True
        self.lock = threading.Lock()
        self.packets = []

    def receive(self, packet):
        with self.lock:
            if not self.accepting:
                return
            if self.first_arrival is None:
                self.first_arrival = packet.arrival_monotonic
            if packet.arrival_monotonic - self.first_arrival >= self.duration_s:
                return
            accepted = self.recorder.submit(packet, 0, None, DEFAULT_LAYERS)
            self.packets.append(
                {
                    "source_session_id": packet.session_id,
                    "source_frame_id": packet.frame_id,
                    "arrival_monotonic": packet.arrival_monotonic,
                    "accepted": accepted,
                }
            )

    def freeze(self):
        with self.lock:
            self.accepting = False


def run_capture(output: Path, duration_s: float, config: dict) -> dict:
    """仅真实视频接收和录像；无 AFG 连接、参数写入或输出开关。"""
    output.mkdir(parents=True, exist_ok=False)
    config = {**config, "source_mode": "LIVE_CAMERA", "video_annotation_wait_ms": 0}
    write_json(output / "capture_config.json", config)
    transitions = []
    started = time.monotonic()
    problems = []
    recorder = None
    source = None
    window = None
    completed_window = False
    capture_finished = None
    recorder_report = {}
    try:
        recorder = ExperimentVideoRecorder(output, config, started)
        window = RecordingWindow(recorder, duration_s)

        def status(value):
            if not transitions or value != transitions[-1]["status"]:
                transitions.append(
                    {"elapsed_s": time.monotonic() - started, "status": value}
                )

        source = LiveCameraFrameSource(config, window.receive, status, output)
        source.start()
        announced = False
        next_progress_s = 5.0
        while True:
            now = time.monotonic()
            if recorder.error:
                raise RuntimeError(recorder.error)
            if transitions and transitions[-1]["status"].startswith("ERROR:"):
                raise RuntimeError(transitions[-1]["status"])
            if window.first_arrival is None:
                if now - started > FIRST_FRAME_TIMEOUT_S:
                    raise TimeoutError("15 秒内未收到相机画面")
            else:
                if not announced:
                    print(f"已收到第一帧，开始记录 {duration_s:g} 秒。", flush=True)
                    announced = True
                elapsed = now - window.first_arrival
                if elapsed >= duration_s:
                    completed_window = True
                    break
                if elapsed >= next_progress_s:
                    print(
                        f"已记录 {elapsed:.1f} 秒，已写入 {recorder.written} 帧。",
                        flush=True,
                    )
                    next_progress_s += 5
            time.sleep(0.01)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - 保留接收/编码错误，仍收尾保存。
        problems.append(f"{type(exc).__name__}: {exc}")
        write_json(
            output / "capture_error.json",
            {"error": problems[-1], "traceback": traceback.format_exc()},
        )
    finally:
        capture_finished = time.monotonic()
        if window:
            window.freeze()
        if source:
            try:
                source.stop()
            except Exception as exc:  # noqa: BLE001 - 相机收尾失败也必须继续关闭录像文件。
                problems.append(f"相机接收收尾失败：{type(exc).__name__}: {exc}")
        if recorder:
            try:
                recorder_report = recorder.close()
            except Exception as exc:  # noqa: BLE001 - 不将未正常封装的视频报告为成功。
                problems.append(f"录像收尾失败：{type(exc).__name__}: {exc}")
    write_json(
        output / "capture_session.json",
        {
            "requested_duration_s": duration_s,
            "first_frame_arrival_monotonic": window.first_arrival if window else None,
            "capture_finished_monotonic": capture_finished,
            "completed_window": completed_window,
            "status_transitions": transitions,
            "source_received": source.received if source else 0,
            "source_overwritten": source.overwritten if source else 0,
            "packets": window.packets if window else [],
            "problems": problems,
            "afg_accessed": False,
            "analysis_enabled": False,
        },
    )
    rows = read_jsonl(output / "video_frames.jsonl")
    timing, table = check_frame_times(rows, duration_s)
    problems.extend(timing["problems"])
    if not completed_window:
        problems.append("未完成指定时长的录像窗口")
    if not recorder_report.get("video_complete"):
        problems.append("原录像器未报告完整保存")
    if recorder_report.get("error"):
        problems.append(recorder_report["error"])
    received = window.packets if window else []
    accepted = [p for p in received if p["accepted"]]
    if frame_keys(accepted) != frame_keys(rows):
        problems.append("提交成功的画面与写入的逐帧记录不一致")
    upstream_skipped = sum(
        max(0, current["source_frame_id"] - previous["source_frame_id"] - 1)
        for previous, current in pairwise(received)
        if previous["source_session_id"] == current["source_session_id"]
    )
    videos = []
    for part in recorder_report.get("parts", []):
        part_rows = [row for row in rows if row["part"] == part["part"]]
        if [row["part_frame_index"] for row in part_rows] != list(
            range(len(part_rows))
        ):
            problems.append(f"第 {part['part']} 个视频分段的帧编号不连续")
        for key, size_key in (("raw", "encoded_wh"), ("overlay", "overlay_encoded_wh")):
            checked = check_video(output / part[key], part_rows, part[size_key])
            videos.append(checked)
            problems.extend(f"{part[key]}：{p}" for p in checked["problems"])
    if not videos:
        problems.append("没有生成可核对的视频")
    # 即使当前录像帧连续，也不能掩盖接收器曾经重连或解码失败。
    receiver_reports = []
    for path in sorted(output.glob("bridge_*/windows_receiver.json")):
        receiver = json.loads(path.read_text())["receiver"]
        receiver_reports.append(receiver)
        if receiver["failures"] or receiver["connections"] != 1:
            problems.append("相机接收器报告连接失败或重连")
    if len(receiver_reports) != 1:
        problems.append("缺少唯一的完整 Windows 接收报告")
    for path in sorted(output.glob("bridge_*/bridge_status.json")):
        bridge = json.loads(path.read_text())
        if bridge["error"] or bridge["exit_code"] != 0:
            problems.append(
                f"画面传输未正常结束：{bridge['error']}，退出码 {bridge['exit_code']}"
            )
    if table:
        with (output / "frame_times.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    report = {
        "passed": not problems,
        "problems": problems,
        "requested_duration_s": duration_s,
        "timing": timing,
        "videos": videos,
        "recorder_submitted": recorder_report.get("submitted_frames", 0),
        "recorder_written": recorder_report.get("written_frames", 0),
        "recorder_dropped": recorder_report.get("dropped_frames", 0),
        "upstream_sequence_skips_within_window": upstream_skipped,
        "receiver_reports": receiver_reports,
        "afg_accessed": False,
        "scope": "真实预览接收与录像；不含图像分析、GUI 负载或相机内部高速录像",
        "timestamp_meaning": "电脑解码及收到画面的时间，不是相机曝光时间",
        "upstream_camera_frame_loss": "相机源帧编号与曝光时间未知，不能统计解码前所有丢帧",
    }
    write_json(output / "recording_check.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 60:
        parser.error("录像时长必须在 1 到 60 秒之间")
    output = args.output_dir or ROOT / "customer_workspace/validation" / (
        "video_20s_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    try:
        result = run_capture(output, args.seconds, load_config())
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print(f"结果目录：{output}", flush=True)
        return 0 if result["passed"] else 1
    except Exception:  # noqa: BLE001 - 命令行明确返回失败并保留第三方错误栈。
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
