"""Windows 单次接收、两路消费：压缩录像与最新图像分析。

GStreamer tee 后有独立队列。录像保留压缩帧，经 MPEG-TS 管道交给第一轮
FFmpeg 流复制；分析只在解码后替换旧图像，不能随意丢压缩参考帧。
新追踪入口通过 GET/QUIT 传递压缩帧和时间，不连接 AFG。
"""

import argparse
import json
import math
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

from customer_workspace.streamcopy_recording import (
    DEFAULT_TOOLS,
    RecordingRtspControl,
    inspect_saved_video,
    write_json,
)
from customer_workspace.tracking_latency import clock_description, parse_ack
from customer_workspace.video_clock import ClockServer, read_ready


def h264_nal_types(data: bytes) -> set[int]:
    """读取 Annex B 数据单元类型，只用于确认录像开头有 SPS/PPS/IDR。

    接入正在运行的视频流时，单凭 GStreamer 的关键帧标志还不够。
    此处不解码或改写图像；3 字节与 4 字节起始码都由同一模式识别。
    """
    return {
        data[match.end()] & 31
        for match in re.finditer(b"\x00\x00\x01", data)
        if match.end() < len(data)
    }


class ParallelCameraStream:
    """兼容原相机服务器的 snapshot/stats/stop 接口，持有本次分流资源。"""

    def __init__(
        self,
        config: dict,
        output: Path,
        seconds: float,
        offline_video=None,
        *,
        tracking=False,
        recording=True,
    ):
        import gstreamer_python

        # 官方 wheel 在 Windows 使用 lib 小写路径；WSL 共享目录区分大小写，
        # 这里显式补充实际的 Lib 路径，仅影响这个独立进程。
        typelibs = Path(gstreamer_python.__file__).parent / "Lib/girepository-1.0"
        os.environ["GI_TYPELIB_PATH"] = (
            str(typelibs) + os.pathsep + os.environ.get("GI_TYPELIB_PATH", "")
        )
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstRtsp", "1.0")
        from gi.repository import GLib, Gst, GstRtsp

        Gst.init(None)
        self.Gst, self.GLib, self.GstRtsp = Gst, GLib, GstRtsp
        self.config, self.output, self.seconds = config, Path(output), seconds
        self.offline_video = offline_video
        self.tracking = tracking
        self.recording = recording
        self.source_eos = False
        self.analysis_started = False
        self.record_dir = self.output / "recording"
        self.record_dir.mkdir(parents=True, exist_ok=False)
        self.target_frames = math.ceil(seconds * 60)
        self.started_at = time.monotonic()
        self.armed = False
        self.record_closed = False
        self.finish_requested = False
        self.record_eos = threading.Event()
        self.stopping = False
        self.errors, self.warnings, self.levels = [], [], []
        self.memory_samples = []
        self.memory_process = psutil.Process()
        self.ffmpeg_memory_process = None
        self.next_memory_sample = 0.0
        self.input_frames, self.record_frames, self.decoded_frames = [], [], []
        self.by_pts, self.decoded_by_pts = {}, {}
        self.latest = None
        self.queue_overruns = {"record_queue": 0, "decode_queue": 0}
        self.raw_replaced = 0
        self.rtp_packets = 0
        self.rtp_missing = 0
        self.rtp_previous = None
        self.relay = None
        self.clock_server = None
        self.link_session = read_ready(self.output / "link_session.json")
        self.ffmpeg = None
        self.ffmpeg_log = None
        self.thread = None
        self.loop = GLib.MainLoop()
        self.state = "connecting"

    def _error(self, message):
        self.errors.append(str(message))
        self.state = "error"
        path = self.output / "receiver_error.json"
        if not path.exists():
            write_json(path, {"error": str(message)})

    def start(self):
        Gst = self.Gst
        try:
            if self.link_session is not None:
                self.clock_server = ClockServer(
                    self.output, self.link_session["experiment_id"], self._health
                ).start()
            if self.offline_video:
                head = (
                    "filesrc name=file_input ! qtdemux ! h264parse config-interval=-1"
                )
                if self.tracking:
                    # 同一条接收/解码/录像链路按视频 PTS 播放，不能全速读文件冒充实机。
                    head += " ! identity sync=true"
            else:
                if self.config["rtsp_transport"] != "udp":
                    raise ValueError("本轮只验证 UDP 相机连接")
                self.relay = RecordingRtspControl(
                    self.config["camera_ip"], rtsp_url=self.config.get("rtsp_url")
                ).start()
                head = (
                    "rtspsrc name=camera protocols=udp latency=20 buffer-mode=none "
                    "drop-on-latency=false udp-buffer-size=262144 "
                    "do-rtsp-keep-alive=true ! rtph264depay wait-for-keyframe=true "
                    "! h264parse config-interval=-1"
                )
            description = head + (
                " ! video/x-h264,stream-format=byte-stream,alignment=au "
                "! tee name=split "
                "split. ! identity name=record_gate "
                "! queue name=record_queue leaky=no max-size-buffers=300 "
                "max-size-bytes=33554432 max-size-time=5000000000 "
                "! mpegtsmux name=transport alignment=7 "
                "! appsink name=compressed_sink emit-signals=true sync=false async=false "
                "max-buffers=8 drop=false wait-on-eos=false "
                "split. ! queue name=decode_queue leaky=no max-size-buffers=120 "
                "max-size-bytes=16777216 max-size-time=2000000000 "
                "! avdec_h264 max-threads=1 ! videoconvert "
                "! video/x-raw,format=BGR "
                "! queue name=latest_queue leaky=downstream max-size-buffers=1 "
                "max-size-bytes=0 max-size-time=0 "
                "! appsink name=latest_sink sync=false async=false max-buffers=1 "
                "leaky-type=downstream wait-on-eos=false"
            )
            if self.tracking:
                # 压缩帧有前后依赖，传输阶段不丢帧；WSL 解码后才保留最新图。
                description = description.split("! avdec_h264", 1)[0] + (
                    "! queue name=latest_queue leaky=no max-size-buffers=2 "
                    "max-size-bytes=0 max-size-time=0 "
                    "! appsink name=latest_sink sync=false async=false max-buffers=2 "
                    "drop=false wait-on-eos=false"
                )
            if not self.recording:
                description = description.replace(
                    "! mpegtsmux name=transport alignment=7 "
                    "! appsink name=compressed_sink emit-signals=true sync=false async=false "
                    "max-buffers=8 drop=false wait-on-eos=false",
                    "! fakesink name=compressed_sink sync=false async=false",
                )
            self.pipeline = Gst.parse_launch(description)
            if self.offline_video:
                self.pipeline.get_by_name("file_input").set_property(
                    "location", str(self.offline_video)
                )
                self.armed = True
            else:
                source = self.pipeline.get_by_name("camera")
                source.set_property("location", self.relay.url)
                port = int(self.config.get("udp_port", 5000))
                source.set_property("port-range", f"{port}-{port + 1}")
                source.connect("before-send", self._before_send)
                source.connect("pad-added", self._rtp_pad)
            self.pipeline.get_by_name("split").get_static_pad("sink").add_probe(
                Gst.PadProbeType.BUFFER,
                self._input_buffer,
            )
            self.pipeline.get_by_name("record_gate").get_static_pad("src").add_probe(
                Gst.PadProbeType.BUFFER,
                self._record_buffer,
            )
            if not self.tracking:
                self.pipeline.get_by_name("latest_queue").get_static_pad(
                    "sink"
                ).add_probe(
                    Gst.PadProbeType.BUFFER,
                    self._decoded_buffer,
                )
            else:
                self.pipeline.get_by_name("latest_queue").get_static_pad(
                    "sink"
                ).add_probe(
                    Gst.PadProbeType.BUFFER,
                    self._tracking_gate,
                )
            self.latest_sink = self.pipeline.get_by_name("latest_sink")
            sink = self.pipeline.get_by_name("compressed_sink")
            if self.recording:
                sink.connect("new-sample", self._write_compressed)
                sink.connect("eos", self._compressed_eos)
            for name in self.queue_overruns:
                self.pipeline.get_by_name(name).connect(
                    "overrun", self._queue_full, name
                )
            if not self.tracking:
                self.pipeline.get_by_name("latest_queue").connect(
                    "overrun", self._raw_full
                )
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._bus_message)
            if self.recording:
                self._start_ffmpeg()
            write_json(
                self.output / "gst_pipeline.json",
                {
                    "description": description,
                    "version": Gst.version_string(),
                    "latency_ms": 20,
                    "camera_connections": 0 if self.offline_video else 1,
                    "rtsp_source_url": self.config.get("rtsp_url"),
                    "rtsp_control_mode": "persistent"
                    if self.config.get("rtsp_url")
                    else "chronos_compatibility",
                },
            )
            self.GLib.timeout_add(100, self._monitor)
            self.thread = threading.Thread(
                target=self.loop.run, name="gst-main", daemon=True
            )
            self.thread.start()
            result = self.pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("GStreamer 不能进入播放状态")
            return self
        except Exception:  # 资源创建边界，清理后保留原始异常。
            self.stop()
            raise

    def _start_ffmpeg(self):
        command = [
            str(DEFAULT_TOOLS / "ffmpeg.exe"),
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-loglevel",
            "verbose",
            "-debug_ts",
            "-copyts",
            "-start_at_zero",
            "-copytb",
            "1",
            "-f",
            "mpegts",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            "-n",
            str(self.record_dir / "video.mp4"),
        ]
        write_json(self.record_dir / "command.json", command)
        self.ffmpeg_log = (self.record_dir / "ffmpeg.log").open("xb")
        self.ffmpeg = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=self.ffmpeg_log,
            stderr=self.ffmpeg_log,
            bufsize=65536,
        )
        self.ffmpeg_memory_process = psutil.Process(self.ffmpeg.pid)

    def _before_send(self, source, message):
        _, method, _, _ = message.parse_request()
        # 切到 NULL 会经过 PAUSED；本轮结束只需 TEARDOWN，不额外暂停相机。
        return method != self.GstRtsp.RTSPMethod.PAUSE

    def _rtp_pad(self, source, pad):
        pad.add_probe(self.Gst.PadProbeType.BUFFER, self._rtp_buffer)

    def _rtp_buffer(self, pad, info):
        buffer = info.get_buffer()
        header = buffer.extract_dup(0, min(12, buffer.get_size()))
        if len(header) >= 12:
            seq = int.from_bytes(header[2:4], "big")
            if self.rtp_previous is not None:
                difference = (seq - self.rtp_previous) % 65536
                if 1 < difference < 32768:
                    self.rtp_missing += difference - 1
            self.rtp_previous = seq
            self.rtp_packets += 1
        return self.Gst.PadProbeReturn.OK

    def _input_buffer(self, pad, info):
        buffer = info.get_buffer()
        row = {
            "seq": len(self.input_frames) + 1,
            "pts_ns": int(buffer.pts),
            "dts_ns": int(buffer.dts),
            "received_windows_mono": time.monotonic(),
            "keyframe": not buffer.has_flags(self.Gst.BufferFlags.DELTA_UNIT),
            "bytes": buffer.get_size(),
        }
        if row["keyframe"]:
            row["nal_types"] = sorted(
                h264_nal_types(buffer.extract_dup(0, buffer.get_size()))
            )
        self.input_frames.append(row)
        self.by_pts[buffer.pts] = row
        return self.Gst.PadProbeReturn.OK

    def _record_buffer(self, pad, info):
        Gst = self.Gst
        buffer = info.get_buffer()
        if self.record_closed:
            if not self.finish_requested:
                # 下一次回调才发 EOS，保证上一帧已经交给队列，避免结束
                # 标记从另一线程抢先到达而丢掉最后一帧。
                self.finish_requested = True
                self.GLib.idle_add(self._send_record_eos)
            return Gst.PadProbeReturn.DROP
        if not self.armed:
            return Gst.PadProbeReturn.DROP
        row = self.by_pts[buffer.pts]
        if not self.record_frames and (
            buffer.pts == Gst.CLOCK_TIME_NONE
            or not {5, 7, 8}.issubset(row.get("nal_types", []))
        ):
            return Gst.PadProbeReturn.DROP
        row["video_frame"] = len(self.record_frames)
        first_pts = (
            self.record_frames[0]["pts_ns"] if self.record_frames else row["pts_ns"]
        )
        row["video_time_s"] = (row["pts_ns"] - first_pts) / 1e9
        self.record_frames.append(dict(row))
        if len(self.record_frames) == 1:
            self.GLib.idle_add(self._mark_started)
        if len(self.record_frames) == self.target_frames:
            self.record_closed = True
        return Gst.PadProbeReturn.OK

    def _mark_started(self):
        write_json(self.output / "recording_started.json", self.record_frames[0])
        return False

    def _send_record_eos(self):
        # 在录像队列入口送结束标记。已接收的数据先全部写完，分析支路继续。
        self.pipeline.get_by_name("record_queue").get_static_pad("sink").send_event(
            self.Gst.Event.new_eos(),
        )
        return False

    def _write_compressed(self, sink):
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        try:
            self.ffmpeg.stdin.write(buffer.extract_dup(0, buffer.get_size()))
        except OSError as exc:
            self._error(f"录像管道写入失败：{exc}")
            return self.Gst.FlowReturn.ERROR
        return self.Gst.FlowReturn.OK

    def _compressed_eos(self, sink):
        try:
            self.ffmpeg.stdin.close()
        except OSError as exc:
            self._error(f"录像管道收尾失败：{exc}")
        self.record_eos.set()

    def _decoded_buffer(self, pad, info):
        buffer = info.get_buffer()
        received = self.by_pts.get(buffer.pts)
        if getattr(self, "tracking", False) and (
            received is None or "video_frame" not in received
        ):
            # 不让录像窗口外的新图覆盖最后一张待分析的录像帧。
            return self.Gst.PadProbeReturn.DROP
        if received is not None:
            decoded = time.monotonic()
            row = {
                "seq": received["seq"],
                "pts_ns": int(buffer.pts),
                "decoded_windows_mono": decoded,
                "input_to_decoded_ms": (decoded - received["received_windows_mono"])
                * 1000,
            }
            self.decoded_frames.append(row)
            self.decoded_by_pts[buffer.pts] = row
            self.state = "streaming"
        return self.Gst.PadProbeReturn.OK

    def _tracking_gate(self, pad, info):
        """窗口前允许解码预热，窗口后停止分析支路，不让 AFG 收尾造成积压。"""
        buffer = info.get_buffer()
        row = self.by_pts[buffer.pts]
        if self.record_closed and "video_frame" not in row:
            return self.Gst.PadProbeReturn.DROP
        if not self.analysis_started:
            if not {5, 7, 8}.issubset(row.get("nal_types", [])):
                return self.Gst.PadProbeReturn.DROP
            self.analysis_started = True
        return self.Gst.PadProbeReturn.OK

    def _queue_full(self, queue, name):
        self.queue_overruns[name] += 1
        if self.tracking:
            self._error(
                f"{name} reached capacity; compressed packets cannot be silently dropped"
            )

    def _raw_full(self, queue):
        self.raw_replaced += 1

    def _bus_message(self, bus, message):
        if message.type == self.Gst.MessageType.ERROR:
            error, detail = message.parse_error()
            self._error(f"{message.src.get_name()}: {error}; {detail}")
        elif message.type == self.Gst.MessageType.WARNING:
            warning, detail = message.parse_warning()
            self.warnings.append(f"{message.src.get_name()}: {warning}; {detail}")
        elif message.type == self.Gst.MessageType.EOS:
            self.source_eos = True

    def _health(self):
        """通过本机消息读取当前接收状态，不反复替换 Windows 共享文件。"""
        return {
            "experiment_id": self.link_session["experiment_id"],
            "windows_s": time.monotonic(),
            "last_frame_windows_s": self.input_frames[-1]["received_windows_mono"]
            if self.input_frames
            else None,
            "errors": list(self.errors),
        }

    def _monitor(self):
        if self.stopping:
            return False
        if (self.output / "start_recording.request").is_file():
            self.armed = True
        row = {"windows_mono": time.monotonic()}
        for name in ("record_queue", "decode_queue", "latest_queue"):
            queue = self.pipeline.get_by_name(name)
            row[name + "_buffers"] = queue.get_property("current-level-buffers")
            row[name + "_ns"] = queue.get_property("current-level-time")
        self.levels.append(row)
        if (
            not self.record_closed
            and self.state != "error"
            and self.input_frames
            and row["windows_mono"] - self.input_frames[-1]["received_windows_mono"] > 3
        ):
            self._error("接收端超过 3 秒未收到新视频帧，保留已有数据并停止本次测试")
        if row["windows_mono"] >= self.next_memory_sample:
            # 一秒一次，避免在每帧的数据传输回调中查询进程内存。
            memory = {"windows_mono": row["windows_mono"]}
            for name, process in (
                ("receiver", self.memory_process),
                ("ffmpeg", self.ffmpeg_memory_process),
            ):
                try:
                    if process is None:
                        continue
                    info = process.memory_info()
                    memory[name + "_rss_mib"] = info.rss / 2**20
                    memory[name + "_private_mib"] = getattr(info, "private", 0) / 2**20
                except psutil.Error as exc:
                    memory[name + "_error"] = str(exc)
            self.memory_samples.append(memory)
            self.next_memory_sample = row["windows_mono"] + 1
        record_finished = (
            self.record_eos.is_set() and self.ffmpeg.poll() is not None
            if getattr(self, "recording", True)
            else self.record_closed
        )
        if record_finished:
            path = self.output / "recording_done.json"
            if not path.exists() and self.record_frames:
                write_json(
                    path,
                    {
                        "seconds": self.seconds,
                        "first_seq": self.record_frames[0]["seq"],
                        "last_seq": self.record_frames[-1]["seq"],
                        "frames": len(self.record_frames),
                        "ffmpeg_exit": self.ffmpeg.returncode if self.ffmpeg else None,
                    },
                )
        return True

    def snapshot(self):
        sample = self.latest_sink.emit("try-pull-sample", 0)
        if sample is None:
            return self.latest
        buffer = sample.get_buffer()
        decoded = self.decoded_by_pts.get(buffer.pts)
        if decoded is None:
            return self.latest
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        data = buffer.extract_dup(0, buffer.get_size())
        # BGR 每行在 GStreamer 中按 4 字节对齐，不能假设任意宽度都没有填充。
        stride = ((width * 3 + 3) // 4) * 4
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, stride)
        frame = array[:, : width * 3].reshape(height, width, 3).copy()
        self.latest = (
            decoded["seq"],
            decoded["decoded_windows_mono"],
            datetime.now().astimezone().isoformat(),
            frame,
        )
        return self.latest

    def tracking_snapshot(self):
        """取一个 H.264 access unit；解码和灰度转换由 WSL 完成。"""
        sample = self.latest_sink.emit("try-pull-sample", 0)
        if sample is None:
            return None
        buffer = sample.get_buffer()
        row = self.by_pts.get(buffer.pts)
        if row is None:
            raise ValueError("Compressed frame has no source time correspondence")
        if buffer.pts == self.Gst.CLOCK_TIME_NONE:
            raise ValueError("Missing source PTS")
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        raw = buffer.extract_dup(0, buffer.get_size())
        frame = np.frombuffer(raw, np.uint8)
        header = dict(row)
        header.update(
            shape=[height, width],
            pixel_format="H264",
            payload_bytes=len(raw),
            source_time_s=int(buffer.pts) / 1e9,
            in_window="video_frame" in row,
        )
        return header, frame

    def stats(self):
        recent = self.decoded_frames[-300:]
        span = (
            recent[-1]["decoded_windows_mono"] - recent[0]["decoded_windows_mono"]
            if len(recent) > 1
            else 0
        )
        return {
            "state": self.state,
            "received_frames": len(self.decoded_frames),
            "connections": 1,
            "failures": len(self.errors),
            "receive_fps_last_5s": (len(recent) - 1) / span if span > 0 else 0,
            "last_error": self.errors[-1] if self.errors else "",
        }

    def stop(self):
        if self.stopping:
            return
        self.stopping = True
        self.record_closed = True
        if hasattr(self, "pipeline"):
            if self.recording and not self.record_eos.is_set():
                self._send_record_eos()
                self.record_eos.wait(3)
            self.pipeline.set_state(self.Gst.State.NULL)
            self.pipeline.get_state(3 * self.Gst.SECOND)
        self.loop.quit()
        if self.thread:
            self.thread.join(3)
        if self.ffmpeg:
            if not self.ffmpeg.stdin.closed:
                self.ffmpeg.stdin.close()
            try:
                self.ffmpeg.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg.kill()
                self.ffmpeg.wait(timeout=3)
                self.errors.append("FFmpeg 未按时收尾")
        if self.ffmpeg_log:
            self.ffmpeg_log.close()
        if self.relay:
            self.relay.close()
        if self.clock_server:
            self.clock_server.close()
        self.state = "stopped"
        write_json(
            self.output / "gst_session.json",
            {
                "seconds": self.seconds,
                "input_kind": "offline_video" if self.offline_video else "camera",
                "recording_enabled": self.recording,
                "tracking_protocol": self.tracking,
                "target_frames": self.target_frames,
                "errors": self.errors,
                "warnings": self.warnings,
                "input_frames": self.input_frames,
                "record_frames": self.record_frames,
                "decoded_frames": self.decoded_frames,
                "queue_levels": self.levels,
                "memory_samples": self.memory_samples,
                "queue_overruns": self.queue_overruns,
                "raw_queue_replaced": self.raw_replaced,
                "appsink_replaced": self.latest_sink.get_property("dropped")
                if hasattr(self, "latest_sink")
                else None,
                "rtp_packets": self.rtp_packets,
                "rtp_sequence_missing": self.rtp_missing,
                "relay_errors": self.relay.errors if self.relay else [],
                "rtsp_exchanges": self.relay.exchanges if self.relay else [],
                "capture_exit": self.ffmpeg.returncode if self.ffmpeg else None,
                "wall_seconds": time.monotonic() - self.started_at,
            },
        )


def inspect_recording(output: Path):
    session = json.loads((output / "gst_session.json").read_text())
    record_dir = output / "recording"
    result = inspect_saved_video(
        DEFAULT_TOOLS / "ffmpeg.exe", DEFAULT_TOOLS / "ffprobe.exe", record_dir
    )
    result.update(
        {
            "seconds": session["seconds"],
            "input_kind": session.get("input_kind", "camera"),
            "expected_fps": 60,
            "target_frames": session["target_frames"],
            "capture_exit": session["capture_exit"],
            "relay_errors": session["relay_errors"],
            "rtsp_exchanges": session["rtsp_exchanges"],
            "stop_reason": None,
            "capture_wall_seconds": session["wall_seconds"],
        }
    )
    if session["errors"]:
        result["error"] = session["errors"]
    write_json(record_dir / "capture_session.json", result)


def serve_tracking(args):
    """新版 GET 协议：压缩数据、源 PTS 和录像对应关系一起传给 WSL。"""
    if os.name == "nt":
        import msvcrt

        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    from microbubble_experiment.camera import load_config

    messages = queue.Queue(maxsize=1)
    measure_latency = getattr(args, "measure_latency", False)
    acknowledgements, timing_rows = [], []

    def commands():
        for line in sys.stdin.buffer:
            if measure_latency and line.startswith(b"ACK "):
                received = time.monotonic()
                try:
                    acknowledgements.append(parse_ack(line, received))
                except (ValueError, KeyError, TypeError):
                    messages.put(b"INVALID_ACK")
                continue
            messages.put(line.strip())
            if line.strip() == b"QUIT":
                return
        messages.put(b"QUIT")

    threading.Thread(target=commands, daemon=True).start()
    cfg = {} if args.offline_video else load_config(args.config)
    stream = ParallelCameraStream(
        cfg,
        args.output,
        args.seconds,
        args.offline_video,
        tracking=True,
        recording=not args.analysis_only,
    )
    previous = None
    sent = 0
    first_request = True
    try:
        stream.start()
        deadline = time.monotonic() + args.seconds + 60
        while time.monotonic() < deadline:
            try:
                command = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if command == b"QUIT":
                break
            if command != b"GET":
                raise ValueError("Unexpected tracking IPC command")
            get_started = time.monotonic() if measure_latency else None
            if first_request:
                # 首次请求可能包含 Windows 启动的数秒等待，单独答复，不能把
                # 这段启动时间加到第一张尚未产生的图像延迟上。
                ready = json.dumps(
                    {"protocol": 2, "status": "WAIT", "payload_bytes": 0}
                ).encode()
                sys.stdout.buffer.write(
                    b"CV2FRAME" + struct.pack("<I", len(ready)) + ready
                )
                sys.stdout.buffer.flush()
                first_request = False
                continue
            # 录像结束后不再取窗口之外的帧，避免末帧编号被覆盖。
            at_end = bool(
                stream.record_frames
                and stream.record_closed
                and previous == stream.record_frames[-1]["seq"]
            )
            packet = None if at_end else stream.tracking_snapshot()
            waiting_until = time.monotonic() + 0.05
            while (
                packet is None
                and not stream.errors
                and time.monotonic() < waiting_until
            ):
                if (
                    stream.record_closed
                    and stream.record_frames
                    and previous == stream.record_frames[-1]["seq"]
                ):
                    break
                time.sleep(0.001)
                packet = stream.tracking_snapshot()
            selected_at = time.monotonic() if measure_latency else None
            header = {"protocol": 2, "status": "WAIT", "payload_bytes": 0}
            payload = b""
            if stream.errors:
                header.update(status="ERROR", error=stream.errors[-1])
            elif packet is not None:
                values, image = packet
                if values["in_window"]:
                    previous = values["seq"]
                snapshot = time.monotonic()
                header.update(
                    values,
                    status="FRAME",
                    experiment_id=stream.link_session["experiment_id"],
                    stream_id=stream.link_session["stream_id"],
                    sender_frame_age_ms=(snapshot - values["received_windows_mono"])
                    * 1000,
                )
                payload = (
                    memoryview(image).cast("B")
                    if image.flags.c_contiguous
                    else image.tobytes()
                )
                sent += 1
            elif (
                stream.record_closed
                and stream.record_frames
                and previous == stream.record_frames[-1]["seq"]
            ):
                window = read_ready(args.output / "recording_done.json")
                if window:
                    header.update(status="END", window=window)
            elif stream.source_eos:
                header.update(
                    status="ERROR", error="Input ended before the requested window"
                )
            raw = json.dumps(header, separators=(",", ":")).encode()
            # 分开写头和已有图像空间，避免再拼接一个整帧大小的 bytes。
            write_started = time.monotonic() if measure_latency else None
            sys.stdout.buffer.write(b"CV2FRAME" + struct.pack("<I", len(raw)) + raw)
            if payload:
                sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
            if measure_latency and header["status"] == "FRAME":
                timing_rows.append(
                    {
                        "seq": header["seq"],
                        "in_window": header["in_window"],
                        "received_windows_s": header["received_windows_mono"],
                        "get_started_windows_s": get_started,
                        "selected_windows_s": selected_at,
                        "write_start_windows_s": write_started,
                        "write_end_windows_s": time.monotonic(),
                        "payload_bytes": len(payload),
                        "header_bytes": len(raw) + 12,
                    }
                )
            if header["status"] == "ERROR":
                raise RuntimeError(header["error"])
        else:
            raise TimeoutError("Tracking receiver exceeded its finite session duration")
    finally:
        stream.stop()
        if measure_latency:
            write_json(
                args.output / "latency_windows.json",
                {
                    "clock": clock_description(),
                    "sent": timing_rows,
                    "acknowledgements": acknowledgements,
                },
            )
        write_json(
            args.output / "tracking_sender.json",
            {
                "sent_frames": sent,
                "errors": stream.errors,
                "input_kind": "offline_video" if args.offline_video else "camera",
                "clock": "Windows monotonic for receipt; source PTS for motion",
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--offline-video", type=Path)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--measure-latency", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        inspect_recording(args.output)
        return
    if not 1 <= args.seconds <= 90:
        parser.error("每组录像限 1–90 秒")
    if args.tracking:
        serve_tracking(args)
        return
    if args.offline_video:
        stream = ParallelCameraStream(
            {}, args.output, args.seconds, args.offline_video
        ).start()
        try:
            stream.record_eos.wait(args.seconds + 30)
        finally:
            stream.stop()
        return
    from microbubble_experiment.camera_server import main as camera_server_main

    camera_server_main(
        [
            "--config",
            str(args.config),
            "--seconds",
            str(args.seconds + 60),
            "--summary",
            str(args.summary),
        ],
        stream_factory=lambda cfg: ParallelCameraStream(cfg, args.output, args.seconds),
    )


if __name__ == "__main__":
    main()
