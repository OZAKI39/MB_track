"""Windows 本机 RTSP 模拟相机：只使用 F 盘视频副本，不导入实验或分析程序。

serve 按源视频速度循环发布 A、B；verify 在 Windows 接收并核对完整周期；
preview 使用 Windows FFplay 播放。正式源码留在 customer_workspace，
F 盘仅部署这个独立工具及它需要的视频、配置和可执行文件。
"""

import argparse
import csv
import ctypes
import hashlib
import itertools
import json
import os
import platform
import socket
import statistics
import subprocess
import threading
import time
from ctypes import wintypes
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path


def write_json(path, value):
    """测试记录只创建新文件，不覆盖以前的验证证据。"""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command, output, stem, timeout=60):
    """运行 Windows 工具，保留精确命令、输出和失败原因。"""
    write_json(output / (stem + ".command.json"), command)
    try:
        result = subprocess.run(
            command, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        (output / (stem + ".stdout.txt")).write_bytes(exc.stdout or b"")
        (output / (stem + ".stderr.txt")).write_bytes(exc.stderr or b"")
        write_json(output / (stem + ".timeout.json"), {"timeout_s": timeout})
        raise
    (output / (stem + ".stdout.txt")).write_bytes(result.stdout)
    (output / (stem + ".stderr.txt")).write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{stem} exited with {result.returncode}; see its stderr")
    return result.stdout.decode("utf-8", errors="replace")


def probe(root, video, output, stem):
    raw = run_logged(
        [
            str(root / "bin/ffprobe.exe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time,key_frame:stream:format",
            "-of",
            "json",
            str(video),
        ],
        output,
        stem,
    )
    result = json.loads(raw)
    write_json(output / (stem + ".json"), result)
    return result["streams"][0]


def parse_frame_hashes(text):
    return [
        line.split(",")[-1].strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def consecutive_cycle_start(observed, reference):
    """返回首帧在源循环的位置；任意中间缺帧、重复帧或损坏都使核对失败。"""
    if not observed:
        return None
    count = len(reference)
    for start, digest in enumerate(reference):
        if digest == observed[0] and all(
            value == reference[(start + offset) % count]
            for offset, value in enumerate(observed)
        ):
            return start
    return None


def frame_hashes(root, video, output, stem):
    """在 Windows 解码后逐帧计算 MD5；不经过微泡识别算法。"""
    text = run_logged(
        [
            str(root / "bin/ffmpeg.exe"),
            "-hide_banner",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-an",
            "-fps_mode",
            "passthrough",
            "-threads:v",
            "1",
            "-f",
            "framemd5",
            "-",
        ],
        output,
        stem,
    )
    return parse_frame_hashes(text)


def check_ports_free():
    for kind, port in [
        (socket.SOCK_STREAM, 8554),
        (socket.SOCK_DGRAM, 8000),
        (socket.SOCK_DGRAM, 8001),
    ]:
        with socket.socket(socket.AF_INET, kind) as check:
            # 不关闭其他服务，不修改防火墙；端口占用时直接报错。
            check.bind(("127.0.0.1", port))


class LocalCamera:
    """持有本次创建的 MediaMTX 与 FFmpeg 进程，结束时只关闭自己的进程。"""

    def __init__(self, root, output):
        self.root, self.output = root, output
        self.processes = []
        self.logs = []

    def launch(self, name, command, publisher=False):
        write_json(self.output / (name + ".command.json"), command)
        log = (self.output / (name + ".log")).open("xb")
        self.logs.append(log)
        process = subprocess.Popen(
            command,
            cwd=self.root,
            stdin=subprocess.PIPE if publisher else subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.processes.append((name, process, publisher))
        return process

    def start(self):
        check_ports_free()
        server = self.launch(
            "mediamtx",
            [str(self.root / "bin/mediamtx.exe"), str(self.root / "mediamtx.yml")],
        )
        deadline = time.perf_counter() + 10
        while True:
            if server.poll() is not None:
                raise RuntimeError("MediaMTX exited; see mediamtx.log")
            try:
                with socket.create_connection(("127.0.0.1", 8554), timeout=0.2):
                    break
            except OSError:
                if time.perf_counter() > deadline:
                    raise TimeoutError("Local RTSP listener did not start")
                time.sleep(0.05)
        for name in ("expA", "expB"):
            # 文件上传到本机服务器使用 TCP；被测接收端仍默认使用与相机相同的 UDP。
            self.launch(
                "publish_" + name,
                [
                    str(self.root / "bin/ffmpeg.exe"),
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-re",
                    "-stream_loop",
                    "-1",
                    "-i",
                    str(self.root / "videos" / (name + ".mp4")),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "copy",
                    "-rtsp_transport",
                    "tcp",
                    "-f",
                    "rtsp",
                    "rtsp://127.0.0.1:8554/" + name,
                ],
                publisher=True,
            )
        # 等待发布完成 RTSP 握手；验证会另外寻找完整循环，不隐瞒接入前的帧。
        time.sleep(1)
        self.check()

    def check(self):
        for name, process, _ in self.processes:
            if process.poll() is not None:
                raise RuntimeError(f"{name} stopped unexpectedly: {process.returncode}")

    def close(self):
        rows = []
        for name, process, publisher in reversed(self.processes):
            forced = False
            if process.poll() is None:
                if publisher:
                    try:
                        process.stdin.write(b"q\n")
                        process.stdin.flush()
                        process.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        forced = True
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            if process.stdin:
                process.stdin.close()
            rows.append(
                {
                    "name": name,
                    "pid": process.pid,
                    "exit_code": process.returncode,
                    "publisher_forced_termination": forced,
                }
            )
        for log in self.logs:
            log.close()
        write_json(self.output / "shutdown.json", rows)


def receive_cycle(root, source_name, reference, fps, output, transport):
    """接收至少两个循环，核对一个完整周期，保留全部接入数据和逐帧时间。

    接入先准备 2 秒，避免把建立解码状态时缺少时间戳的包写进正式录像。
    之后接收连续窗口，再选择完整匹配周期测量播放速度。
    时间是 Windows 收到 FFmpeg 解码校验行的时间，不冒充网络包到达时间。
    """
    output.mkdir(exist_ok=False)
    count = len(reference)
    target = 2 * count + round(2 * fps)
    captured = output / "received.mp4"
    command = [
        str(root / "bin/ffmpeg.exe"),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-copyts",
        "-start_at_zero",
        "-threads",
        "1",
        "-rtsp_transport",
        transport,
        "-timeout",
        "10000000",
        "-i",
        "rtsp://127.0.0.1:8554/" + source_name,
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-ss",
        "2",
        "-frames:v",
        str(target),
        "-video_track_timescale",
        "90000",
        "-n",
        str(captured),
        "-map",
        "0:v:0",
        "-an",
        "-fps_mode",
        "passthrough",
        "-threads:v",
        "1",
        "-ss",
        "2",
        "-frames:v",
        str(target),
        "-f",
        "framemd5",
        "-flush_packets",
        "1",
        "pipe:1",
    ]
    write_json(output / "receive.command.json", command)
    times, hashes = [], []
    started = time.perf_counter()
    with (output / "receive.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        timer = threading.Timer(target / fps + 25, process.terminate)
        timer.start()
        try:
            with (
                (output / "received.framemd5").open("xb") as raw,
                (output / "received_frames.csv").open(
                    "x", newline="", encoding="utf-8"
                ) as table,
            ):
                writer = csv.writer(table)
                writer.writerow(["decoded_index", "received_elapsed_s", "pts", "md5"])
                for line in process.stdout:
                    received = time.perf_counter() - started
                    raw.write(line)
                    if line.startswith(b"#") or not line.strip():
                        continue
                    parts = line.decode("ascii").strip().split(",")
                    digest = parts[-1].strip()
                    writer.writerow(
                        [len(hashes), f"{received:.9f}", parts[2].strip(), digest]
                    )
                    times.append(received)
                    hashes.append(digest)
            process.wait(timeout=5)
        finally:
            timer.cancel()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            process.stdout.close()
    matches = [
        i
        for i, digest in enumerate(hashes)
        if digest == reference[0] and hashes[i : i + count] == reference
    ]
    report = {
        "source": source_name,
        "transport": transport,
        "expected_cycle_frames": count,
        "capture_target": target,
        "receiver_startup_settle_s": 2.0,
        "capture_scope": "After a 2 s input warm-up; stream-copy also waits for a usable keyframe",
        "received_decoded_frames": len(hashes),
        "receive_exit_code": process.returncode,
        "full_cycle_starts": matches,
        "warnings": (output / "receive.log").read_text(errors="replace").strip(),
        "passed": False,
    }
    if not matches or process.returncode != 0:
        write_json(output / "result.json", report)
        raise RuntimeError(f"{source_name}: no complete matching cycle; see {output}")
    first = matches[-1]
    selected = times[first : first + count]
    gaps = [b - a for a, b in itertools.pairwise(selected)]
    actual_fps = (count - 1) / (selected[-1] - selected[0])
    report.update(
        selected_first_decoded_frame=first,
        selected_last_decoded_frame=first + count - 1,
        matched_source_frames=count,
        pixels_identical=True,
        receive_fps=actual_fps,
        source_fps=fps,
        receive_interval_median_ms=statistics.median(gaps) * 1000,
        receive_interval_p99_ms=sorted(gaps)[int(0.99 * (len(gaps) - 1))] * 1000,
        receive_interval_max_ms=max(gaps) * 1000,
        intervals_over_100ms=sum(gap > 0.1 for gap in gaps),
        timing_meaning="Windows receipt of decoded frame-hash lines, including decoding and pipe buffering",
    )
    captured_hashes = frame_hashes(root, captured, output, "saved_hashes")
    # FFmpeg 的复制输出与解码输出可从不同的起始包开始。逐帧对照源循环，
    # 明确记录边界差异；不能把两个数组下标直接相减，也不能忽略中间缺口。
    live_start = consecutive_cycle_start(hashes, reference)
    saved_start = consecutive_cycle_start(captured_hashes, reference)
    report["live_sequence_consecutive"] = live_start is not None
    report["saved_sequence_consecutive"] = saved_start is not None
    report["live_first_source_frame"] = live_start
    report["saved_first_source_frame"] = saved_start
    report["saved_capture_same_start_as_live"] = captured_hashes == hashes
    saved_stream = probe(root, captured, output, "saved_probe")
    report["saved_frames"] = int(saved_stream["nb_read_frames"])
    saved_matches = [
        i
        for i, digest in enumerate(captured_hashes)
        if digest == reference[0] and captured_hashes[i : i + count] == reference
    ]
    if not saved_matches:
        write_json(output / "result.json", report)
        raise RuntimeError("Saved recording has no complete source cycle")
    saved_first = saved_matches[-1]
    saved_probe = json.loads((output / "saved_probe.json").read_text(encoding="utf-8"))
    saved_times = [
        float(frame["best_effort_timestamp_time"]) for frame in saved_probe["frames"]
    ]
    saved_intervals = [b - a for a, b in itertools.pairwise(saved_times)]
    report["saved_interval_min_ms"] = min(saved_intervals) * 1000
    report["saved_interval_max_ms"] = max(saved_intervals) * 1000
    report["saved_timeline_regular"] = all(
        abs(interval - 1 / fps) <= 2 / 90000 for interval in saved_intervals
    )
    seek_frame = saved_probe["frames"][saved_first]
    if not seek_frame["key_frame"]:
        raise ValueError("The verified cycle does not begin on a keyframe")
    report["saved_cycle_start_frame"] = saved_first
    report["saved_cycle_start_time_s"] = seek_frame["best_effort_timestamp_time"]
    # 使用录像自身的帧时间和关键帧位置，不能用另一输出的下标或平均帧率猜测。
    complete = output / (source_name + "_complete.mp4")
    run_logged(
        [
            str(root / "bin/ffmpeg.exe"),
            "-hide_banner",
            "-v",
            "error",
            "-seek_timestamp",
            "1",
            "-ss",
            seek_frame["best_effort_timestamp_time"],
            "-i",
            str(captured),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-frames:v",
            str(count),
            "-video_track_timescale",
            "90000",
            "-n",
            str(complete),
        ],
        output,
        "extract_complete",
    )
    complete_hashes = frame_hashes(root, complete, output, "complete_hashes")
    report["complete_video_matches_source"] = complete_hashes == reference
    report["complete_video"] = str(complete)
    report["passed"] = bool(
        len(hashes) == target
        and report["live_sequence_consecutive"]
        and report["saved_sequence_consecutive"]
        and report["saved_frames"] == target
        and report["saved_timeline_regular"]
        and report["complete_video_matches_source"]
        and abs(actual_fps / fps - 1) <= 0.05
        and report["intervals_over_100ms"] == 0
        and not report["warnings"]
    )
    write_json(output / "result.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def player_window(user32, pid):
    """只找本次 FFplay 进程的可见窗口，避免关闭用户的其他播放器。"""
    matches = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def collect(window, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(window):
            matches.append(window)
        return True

    user32.EnumWindows(collect, 0)
    return matches[0] if matches else None


def preview(root, source, output, seconds, transport):
    """显示本机 RTSP；到时向自己的窗口发送正常关闭消息。

    循环直播没有文件结尾，FFplay 的 -t 配合 -autoexit 实测未能按时退出。
    因此按窗口出现后的墙钟时间结束预览，不将超时强杀当作播放通过。
    """
    command = [
        str(root / "bin/ffplay.exe"),
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "warning",
        "-rtsp_transport",
        transport,
        "-window_title",
        "MB Camera Sim - " + source,
        "-x",
        "960",
        "-y",
        "768",
        "-an",
        "rtsp://127.0.0.1:8554/" + source,
    ]
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.PostMessageW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    started = time.perf_counter()
    stem = "play_" + source
    write_json(output / (stem + ".command.json"), command)
    with (
        (output / (stem + ".stdout.txt")).open("xb") as stdout,
        (output / (stem + ".stderr.txt")).open("xb") as stderr,
    ):
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr
        )
        try:
            window = None
            while window is None:
                if process.poll() is not None or time.perf_counter() - started > 20:
                    raise RuntimeError(
                        "FFplay did not open its video window; see player log"
                    )
                window = player_window(user32, process.pid)
                time.sleep(0.05)
            opened = time.perf_counter()
            while time.perf_counter() - opened < seconds:
                if process.poll() is not None:
                    raise RuntimeError(
                        "FFplay closed before the requested preview duration"
                    )
                time.sleep(0.05)
            if not user32.PostMessageW(window, 0x0010, 0, 0):  # WM_CLOSE
                raise ctypes.WinError(ctypes.get_last_error())
            process.wait(timeout=10)
            if process.returncode:
                raise RuntimeError(
                    f"FFplay exited with {process.returncode}; see player log"
                )
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
    warnings = (
        (output / ("play_" + source + ".stderr.txt"))
        .read_text(errors="replace")
        .strip()
    )
    report = {
        "source": source,
        "player": "Windows FFplay",
        "exit_code": 0,
        "elapsed_s": time.perf_counter() - started,
        "requested_seconds": seconds,
        "window_found": True,
        "stop_method": "WM_CLOSE to this player PID after requested display time",
        "warnings": warnings,
        "human_review": False,
    }
    write_json(output / ("play_" + source + ".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "verify", "preview"))
    parser.add_argument("--root", type=Path, default=Path("F:/MB_camera_sim"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--transport", choices=("udp", "tcp"), default="udp")
    parser.add_argument("--source", choices=("expA", "expB"), default="expA")
    parser.add_argument("--seconds", type=float, default=22)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Run with Windows Python; no WSL analysis is used")
    if not 1 <= args.repeats <= 10 or not 1 <= args.seconds <= 3600:
        parser.error("repeats must be 1-10 and seconds must be 1-3600")
    root = args.root.resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    output = root / "runs" / (stamp + "_" + args.mode)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "environment.json",
        {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "script_sha256": file_hash(__file__),
            "mode": args.mode,
            "transport": args.transport,
            "root": str(root),
            "hardware": False,
            "wsl_analysis": False,
        },
    )
    if args.mode == "preview":
        preview(root, args.source, output, args.seconds, args.transport)
        return
    camera = LocalCamera(root, output)
    reports = []
    try:
        if args.mode == "verify":
            reference = {}
            for name in ("expA", "expB"):
                video = root / "videos" / (name + ".mp4")
                stream = probe(root, video, output, name + "_probe")
                if stream["codec_name"] != "h264":
                    raise ValueError(
                        "Input must already be H264; no re-encoding is permitted"
                    )
                fps = float(Fraction(stream["avg_frame_rate"]))
                hashes = frame_hashes(root, video, output, name + "_source_hashes")
                assert len(hashes) == int(stream["nb_read_frames"])
                reference[name] = (hashes, fps)
        camera.start()
        print("Local RTSP: rtsp://127.0.0.1:8554/expA and /expB", flush=True)
        if args.mode == "serve":
            print("Press Ctrl+C to stop this simulator and its publishers.", flush=True)
            while True:
                camera.check()
                time.sleep(0.2)
        else:
            for repeat in range(1, args.repeats + 1):
                for name, (hashes, fps) in reference.items():
                    camera.check()
                    print(f"Checking {name}, repetition {repeat}", flush=True)
                    reports.append(
                        receive_cycle(
                            root,
                            name,
                            hashes,
                            fps,
                            output / f"{name}_{repeat:02d}",
                            args.transport,
                        )
                    )
            playback = [
                preview(root, name, output, 22, args.transport) for name in reference
            ]
            server_warnings = [
                line
                for line in (output / "mediamtx.log")
                .read_text(errors="replace")
                .splitlines()
                if " WAR " in line or " ERR " in line
            ]
            publisher_warnings = {
                name: (output / ("publish_" + name + ".log"))
                .read_text(errors="replace")
                .strip()
                for name in reference
            }
            passed = (
                all(r["passed"] for r in reports)
                and not any(r["warnings"] for r in playback)
                and not server_warnings
                and not any(publisher_warnings.values())
            )
            write_json(
                output / "summary.json",
                {
                    "passed": passed,
                    "runs": reports,
                    "playback": playback,
                    "server_warnings": server_warnings,
                    "publisher_warnings": publisher_warnings,
                    "publisher_transport": "tcp",
                    "receiver_transport": args.transport,
                },
            )
            print(f"RESULT {output} passed={passed}", flush=True)
            if not passed:
                raise RuntimeError(
                    "Some acceptance checks failed; inspect summary.json"
                )
    except KeyboardInterrupt:
        print("Stopping local simulator.", flush=True)
    except BaseException as exc:
        write_json(
            output / "failure.json",
            {"error": f"{type(exc).__name__}: {exc}", "completed_runs": reports},
        )
        raise
    finally:
        camera.close()


if __name__ == "__main__":
    main()
