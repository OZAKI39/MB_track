"""由 Windows FFmpeg 独立保存 Chronos 压缩视频，不进入最新帧分析通道。

WSL 负责启动和离线检查；Windows 负责 RTSP 控制、UDP 接收及文件写入。
录像窗口使用视频原有时间，不代表相机曝光时间，也不访问 AFG。
"""

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from customer_workspace.wsl_launcher import (
    wait_for_windows,
    watch_launcher_input,
    windows_path,
)
from microbubble_experiment.camera import load_config
from microbubble_experiment.camera_rtsp import (
    ALLOWED,
    ChronosRtspControl,
    split_message,
)
from microbubble_experiment.desktop import is_wsl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLS = ROOT / "customer_workspace/tools/ffmpeg/bin"
STARTUP_SKIP_S = 0.1


class RecordingRtspControl(ChronosRtspControl):
    """复用原控制中继，记录交互；相机确认结束后关闭本次控制连接。"""

    def __init__(self, camera_ip, camera_port=554, *, rtsp_url=None):
        # Chronos 默认仍逐条重建控制连接；显式完整地址用于普通 RTSP 服务，
        # 需保持同一个 TCP 会话。RTP 视频始终直接进入原 GStreamer 接收端。
        self.endpoint = urlsplit(rtsp_url) if rtsp_url else None
        if self.endpoint is not None:
            if (
                self.endpoint.scheme != "rtsp"
                or self.endpoint.hostname != camera_ip
                or self.endpoint.username is not None
                or self.endpoint.password is not None
                or self.endpoint.fragment
            ):
                raise ValueError(
                    "RTSP URL must match camera_ip and contain no credentials or fragment"
                )
            camera_port = self.endpoint.port or 554
        super().__init__(camera_ip, camera_port)
        self.exchanges = []
        self.upstream = None
        self.pending_response = b""

    def start(self):
        super().start()
        if self.endpoint is not None:
            local = urlsplit(self.url)
            self.url = urlunsplit(
                (
                    local.scheme,
                    local.netloc,
                    self.endpoint.path or "/",
                    self.endpoint.query,
                    "",
                )
            )
        return self

    def _persistent_request(self, request):
        """为模拟 RTSP 源保留控制会话，并读取真实响应，不伪造确认成功。"""
        first, rest = request.split(b"\r\n", 1)
        method, uri, protocol = first.decode("ascii").split(" ", 2)
        if method not in ALLOWED or protocol != "RTSP/1.0":
            raise ValueError("Unsupported RTSP control method")
        parsed = urlsplit(uri)
        target = urlunsplit(
            (
                "rtsp",
                f"{self.camera_ip}:{self.camera_port}",
                parsed.path or "/",
                parsed.query,
                "",
            )
        )
        if self.upstream is None:
            self.upstream = socket.create_connection(
                (self.camera_ip, self.camera_port), timeout=2
            )
        self.upstream.sendall(f"{method} {target} {protocol}\r\n".encode() + rest)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            parts = split_message(self.pending_response)
            if parts:
                response, self.pending_response = parts
                self.requests += 1
                return response
            chunk = self.upstream.recv(8192)
            if not chunk:
                raise ConnectionError(
                    "RTSP source closed before completing its response"
                )
            self.pending_response += chunk
        raise TimeoutError("RTSP source response deadline exceeded")

    def close(self):
        try:
            super().close()
        finally:
            if self.upstream is not None:
                self.upstream.close()
                self.upstream = None

    def _camera_request(self, request):
        method = request.split(b" ", 1)[0].decode("ascii")
        event = {"method": method, "started_perf_counter_s": time.perf_counter()}
        self.exchanges.append(event)
        try:
            response = (
                self._persistent_request(request)
                if self.endpoint is not None
                else super()._camera_request(request)
            )
            event["response"] = response.split(b"\r\n", 1)[0].decode("ascii")
            if method == "TEARDOWN" and event["response"].startswith("RTSP/1.0 200 "):
                # 本中继仅服务本次录像。相机确认结束后不再等 FFmpeg
                # 关闭连接时的下一次 recv，避免 Windows 将主动断开报作故障。
                self.stop_event.set()
            return response
        except (OSError, ValueError) as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event["finished_perf_counter_s"] = time.perf_counter()


def write_json(path: Path, value) -> None:
    """创建新的证据文件，已有结果不能覆盖。"""
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def check_udp_ports(port: int) -> None:
    """在发出相机请求前检查端口，避免与现有接收进程争用。"""
    if port < 1024 or port > 65534 or port % 2:
        raise ValueError("UDP 起始端口必须为 1024–65534 内的偶数")
    sockets = []
    try:
        for number in (port, port + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sockets.append(sock)
            if os.name == "nt":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind(("0.0.0.0", number))
    finally:
        for sock in sockets:
            sock.close()


def capture_command(
    ffmpeg: Path,
    url: str,
    config: dict,
    output: Path,
    seconds: float,
    expected_fps: float = 60.0,
) -> list[str]:
    """沿用原 UDP 参数；保留来源时间，只封装压缩视频，不补帧或重编码。"""
    port = int(config.get("udp_port", 5000))
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "verbose",
        "-debug_ts",
        "-copyts",
        "-start_at_zero",
        "-copytb",
        "1",
        "-rtsp_transport",
        "udp",
        "-min_port",
        str(port),
        "-max_port",
        str(port + 1),
        "-timeout",
        str(config["open_timeout_ms"] * 1000),
        "-allowed_media_types",
        "video",
    ]
    if config.get("latency_mode") == "low":
        command.extend(
            [
                "-max_delay",
                "20000",
                "-reorder_queue_size",
                "32",
                "-buffer_size",
                "262144",
            ]
        )
    command.extend(
        [
            "-i",
            url,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            "copy",
            # Chronos 首包可能没有时间，即使它是关键帧也不能作为零点。
            # 输出端 -ss 跳过启动 100 ms，再由流复制等待下一个关键帧。
            "-ss",
            str(STARTUP_SKIP_S),
            # 连接后可能需等待关键帧。保存足量画面后再核对原始时长；
            # 此参数只限制数量，不重编码、补帧或修改播放速度。
            "-frames:v",
            str(math.ceil(seconds * expected_fps)),
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            "-progress",
            str(output / "progress.txt"),
            "-n",
            str(output / "video.mp4"),
        ]
    )
    return command


def run_capture(
    command: list[str], output: Path, seconds: float, cancel: threading.Event
) -> tuple[int, str | None]:
    """保存原始日志；取消或超时先请求 FFmpeg 收尾，仅管理本次子进程。"""
    reason = None
    with (output / "ffmpeg.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
        )
        deadline = time.perf_counter() + seconds + 30
        try:
            while process.poll() is None:
                try:
                    if cancel.wait(0.1):
                        reason = "用户取消或 WSL 启动端断开"
                        break
                    if time.perf_counter() >= deadline:
                        reason = "超过录像时长加 30 秒的总等待上限"
                        break
                except KeyboardInterrupt:
                    cancel.set()
            if reason:
                try:
                    process.stdin.write(b"q\n")
                    process.stdin.flush()
                except OSError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                    reason += "；FFmpeg 未及时退出，已终止本次子进程"
            return process.wait(), reason
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            try:
                process.stdin.close()
            except OSError:
                pass


def inspect_saved_video(ffmpeg: Path, ffprobe: Path, output: Path) -> dict:
    """录完后才解码检查；保存文件内的逐包、逐帧时间及严格解码日志。"""
    video = str(output / "video.mp4")
    probe = [
        str(ffprobe),
        "-v",
        "warning",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-count_packets",
        "-show_packets",
        "-show_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        video,
    ]
    decode = [
        str(ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-v",
        "warning",
        "-xerror",
        "-err_detect",
        "explode",
        "-i",
        video,
        "-map",
        "0:v:0",
        "-fps_mode",
        "passthrough",
        "-enc_time_base",
        "demux",
        "-f",
        "null",
        "-",
    ]
    write_json(output / "inspection_commands.json", {"probe": probe, "decode": decode})
    with (
        (output / "probe.json").open("xb") as result,
        (output / "probe.log").open("xb") as log,
    ):
        probe_result = subprocess.run(
            probe,
            stdout=result,
            stderr=log,
            timeout=90,
            check=False,
        )
    with (output / "decode.log").open("xb") as log:
        decode_result = subprocess.run(
            decode,
            stdout=log,
            stderr=log,
            timeout=90,
            check=False,
        )
    return {
        "probe_exit": probe_result.returncode,
        "decode_exit": decode_result.returncode,
    }


def record_windows(args, cancel: threading.Event) -> int:
    """真实录像入口；保留失败及部分视频，不自动重试或覆盖。"""
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    session = {
        "seconds": args.seconds,
        "expected_fps": args.expected_fps,
        "target_frames": math.ceil(args.seconds * args.expected_fps),
        "startup_skip_s": STARTUP_SKIP_S,
        "started_at": datetime.now().astimezone().isoformat(),
        "os": platform.platform(),
        "python": sys.version,
        "timestamp_meaning": "原始视频播放时间；不是曝光时刻或电脑收包时刻",
        "output_directory": str(output),
        "capture_exit": None,
        "stop_reason": None,
        "relay_errors": [],
    }
    relay = None
    try:
        config = load_config(args.config)
        session["config"] = config
        if config["rtsp_transport"] != "udp":
            raise ValueError("本轮独立路线仅验证 UDP；不能把 TCP 视频交给控制中继")
        ffmpeg = args.tools_dir.resolve() / "ffmpeg.exe"
        ffprobe = args.tools_dir.resolve() / "ffprobe.exe"
        for executable in (ffmpeg, ffprobe):
            version = subprocess.check_output(
                [str(executable), "-version"],
                text=True,
                encoding="utf-8",
                timeout=10,
            )
            session[executable.stem + "_version"] = version
        # debug_ts 是版本相关日志。升级工具后必须先确认解析格式，不能静默少计。
        if not session["ffmpeg_version"].startswith("ffmpeg version 9.0.2"):
            raise ValueError(
                "当前日志计数器已验证 FFmpeg 9.0.2，请先核验新版本日志格式"
            )
        check_udp_ports(int(config.get("udp_port", 5000)))
        if cancel.is_set():
            raise RuntimeError("录像启动前已取消")
        if config["rtsp_control_relay"]:
            relay = RecordingRtspControl(config["camera_ip"]).start()
            url = relay.url
        else:
            url = f"rtsp://{config['camera_ip']}/"
        command = capture_command(
            ffmpeg,
            url,
            config,
            output,
            args.seconds,
            args.expected_fps,
        )
        write_json(output / "command.json", command)
        started = time.perf_counter()
        print(f"Windows 正在独立保存 {args.seconds:g} 秒压缩视频。", flush=True)
        session["capture_exit"], session["stop_reason"] = run_capture(
            command,
            output,
            args.seconds,
            cancel,
        )
        session["capture_wall_seconds"] = time.perf_counter() - started
        # 先关闭相机控制连接，再离线检查；检查期间不继续接收画面。
        if relay:
            relay.close()
            session["relay_requests"] = relay.requests
            session["relay_errors"] = list(relay.errors)
            session["rtsp_exchanges"] = list(relay.exchanges)
            relay = None
        print("录像接收已结束，正在逐帧检查已保存文件。", flush=True)
        if (output / "video.mp4").exists():
            session.update(inspect_saved_video(ffmpeg, ffprobe, output))
    except Exception as exc:  # noqa: BLE001 -- 真实采集边界必须保存意外故障和完整堆栈。
        session["error"] = f"{type(exc).__name__}: {exc}"
        session["traceback"] = traceback.format_exc()
    finally:
        if relay:
            try:
                relay.close()
            except (OSError, RuntimeError) as exc:
                session["relay_close_error"] = str(exc)
            session["relay_requests"] = relay.requests
            session["relay_errors"] = list(relay.errors)
            session["rtsp_exchanges"] = list(relay.exchanges)
        session["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(output / "capture_session.json", session)
    return int(
        bool(
            session.get("error")
            or session["capture_exit"] != 0
            or session["stop_reason"]
            or session.get("probe_exit") != 0
            or session.get("decode_exit") != 0
            or session["relay_errors"]
            or session.get("relay_close_error")
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument(
        "--expected-fps",
        type=float,
        default=60,
        help="已核实的预览帧率，仅决定保存数量，不改变相机或补帧",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/chronos_config.json"
    )
    parser.add_argument("--tools-dir", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--output-dir", type=Path, help="必须不存在的结果目录")
    parser.add_argument("--launcher-stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 120:
        parser.error("本轮检查时长必须大于 0 且不超过 120 秒")
    if not math.isfinite(args.expected_fps) or not 1 <= args.expected_fps <= 240:
        parser.error("预期帧率必须在 1–240 之间；录像后仍核对真实播放时间")
    if args.output_dir is None:
        name = "streamcopy_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = ROOT / "customer_workspace/validation" / name
    if args.output_dir.exists():
        parser.error(f"结果目录已存在，不能覆盖：{args.output_dir}")
    if os.name == "nt":
        cancel = threading.Event()
        if args.launcher_stdin:
            threading.Thread(
                target=watch_launcher_input,
                args=(cancel.set,),
                daemon=True,
            ).start()
        return record_windows(args, cancel)
    if not is_wsl():
        parser.error("请从 WSL 运行，由现有 Windows 相机环境接收视频")
    machine = json.loads(
        (ROOT / "configs/machine.local.json").read_text(encoding="utf-8-sig")
    )
    command = [
        machine["windows_python"],
        "-X",
        "utf8",
        "-u",
        "-B",
        "-m",
        "customer_workspace.streamcopy_recording",
        "--seconds",
        str(args.seconds),
        "--expected-fps",
        str(args.expected_fps),
        "--config",
        windows_path(args.config),
        "--tools-dir",
        windows_path(args.tools_dir),
        "--output-dir",
        windows_path(args.output_dir),
        "--launcher-stdin",
    ]
    result = wait_for_windows(command, ROOT)
    if (args.output_dir / "capture_session.json").is_file():
        from customer_workspace.streamcopy_report import make_report

        report = make_report(args.output_dir)
        summary = {
            key: report.get(key)
            for key in ("passed", "problems", "counts", "video", "timing")
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return int(result != 0 or not report["passed"])
    print("Windows 未产生会话记录，本次录像未完成。", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
