"""Windows 到 WSL 的独立传图进程；有界等待区不覆盖正在分析的图像。"""

import json
import multiprocessing as mp
import os
import queue
import select
import signal
import struct
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from customer_workspace.microbubble_tracking.frame_conversion import ReusableConversion
from customer_workspace.streamcopy_recording import write_json
from customer_workspace.tracking_latency import clock_description, send_ack
from customer_workspace.wsl_launcher import windows_path
from microbubble_experiment.configuration import ROOT


class LatestFrames:
    """默认等待一帧；测试可等待两帧，满时替换最旧待取帧，取出保持先后顺序。

    另留一块正在接收、一块正在分析的空间；两帧等待不占用这两块空间。
    pending 始终指向下一张应取的图，保留原入口的等待与结束判断。
    """

    def __init__(self, context, shape, waiting_frames=1, diagnose_queue=False):
        if waiting_frames not in (1, 2):
            raise ValueError("Waiting capacity must be 1 or 2 frames")
        self.waiting_frames = waiting_frames
        self.diagnose_queue = diagnose_queue
        # 两端分别持有本地小日志，结束时各自保存；不经共享队列复制日志。
        self.producer_events, self.consumer_events = [], []
        self.shape = tuple(shape)
        slots = waiting_frames + 2
        self.pixels = [context.RawArray("B", int(np.prod(shape))) for _ in range(slots)]
        self.headers = [context.RawArray("B", 65536) for _ in range(slots)]
        self.sizes = context.RawArray("i", slots)
        self.source_frames = context.RawArray("q", slots)
        self.entered_times = context.RawArray("d", slots) if diagnose_queue else None
        self.states = context.RawArray(
            "i", slots
        )  # 0 free, 1 receiving, 2 pending, 3 analyzing
        self.pending = context.RawValue("i", -1)
        self.following = context.RawValue("i", -1)
        self.peak_waiting = context.RawValue("i", 0)
        self.replaced = context.RawValue("i", 0)
        self.condition = context.Condition()
        for memory in self.pixels:
            np.frombuffer(memory, np.uint8).fill(0)

    @property
    def overwritten(self):
        return self.replaced.value

    def acquire(self, seq=None):
        requested = time.perf_counter() if self.diagnose_queue else None
        cpu = time.thread_time() if self.diagnose_queue else None
        with self.condition:
            locked = time.perf_counter() if self.diagnose_queue else None
            for index, state in enumerate(self.states):
                if state == 0:
                    self.states[index] = 1
                    image = np.frombuffer(self.pixels[index], np.uint8).reshape(
                        self.shape
                    )
                    break
            else:
                raise RuntimeError("Frame buffer ownership violated")
        if self.diagnose_queue:
            self.producer_events.append(
                {
                    "operation": "acquire",
                    "seq": seq,
                    "slot": index,
                    "requested_wsl_s": requested,
                    "locked_wsl_s": locked,
                    "finished_wsl_s": time.perf_counter(),
                    "thread_cpu_ms": (time.thread_time() - cpu) * 1000,
                }
            )
        return index, image

    def publish(self, index, metadata):
        """发布完整帧，返回被替换的源帧号及等待数量，供逐帧核对。"""
        requested = time.perf_counter() if self.diagnose_queue else None
        cpu = time.thread_time() if self.diagnose_queue else None
        raw = json.dumps(metadata, separators=(",", ":")).encode()
        if len(raw) > 65536:
            raise ValueError("Frame metadata too large")
        lock_requested = time.perf_counter() if self.diagnose_queue else None
        with self.condition:
            locked = time.perf_counter() if self.diagnose_queue else None
            if self.states[index] != 1:
                raise RuntimeError("Only a receiving buffer can be published")
            discarded = None
            if self.pending.value >= 0 and (
                self.waiting_frames == 1 or self.following.value >= 0
            ):
                oldest = self.pending.value
                discarded = self.source_frames[oldest]
                self.states[oldest] = 0
                self.replaced.value += 1
                self.pending.value = self.following.value
            self.headers[index][: len(raw)] = raw
            self.sizes[index] = len(raw)
            self.source_frames[index] = metadata["seq"]
            if self.pending.value < 0:
                self.pending.value = index
            else:
                self.following.value = index
            self.states[index] = 2
            depth = 1 + int(self.following.value >= 0)
            self.peak_waiting.value = max(self.peak_waiting.value, depth)
            if self.diagnose_queue:
                entered = time.perf_counter()
                self.entered_times[index] = entered
            self.condition.notify()
        if self.diagnose_queue:
            self.producer_events.append(
                {
                    "operation": "publish",
                    "seq": metadata["seq"],
                    "slot": index,
                    "requested_wsl_s": requested,
                    "lock_requested_wsl_s": lock_requested,
                    "locked_wsl_s": locked,
                    "entered_wsl_s": entered,
                    "finished_wsl_s": time.perf_counter(),
                    "depth": depth,
                    "metadata_bytes": len(raw),
                    "thread_cpu_ms": (time.thread_time() - cpu) * 1000,
                }
            )
        return discarded, depth

    def take(self):
        requested = time.perf_counter() if self.diagnose_queue else None
        cpu = time.thread_time() if self.diagnose_queue else None
        with self.condition:
            locked = time.perf_counter() if self.diagnose_queue else None
            index = self.pending.value
            if index < 0:
                return None
            self.pending.value = self.following.value
            self.following.value = -1
            self.states[index] = 3
            if self.diagnose_queue:
                entered = self.entered_times[index]
                taken = time.perf_counter()
            header = json.loads(bytes(self.headers[index][: self.sizes[index]]))
        image = np.frombuffer(self.pixels[index], np.uint8).reshape(self.shape)
        if self.diagnose_queue:
            finished = time.perf_counter()
            detail = {
                "entered_wsl_s": entered,
                "taken_wsl_s": taken,
                "take_returned_wsl_s": finished,
            }
            header["queue_detail"] = detail
            self.consumer_events.append(
                {
                    "operation": "take",
                    "seq": header["seq"],
                    "slot": index,
                    "requested_wsl_s": requested,
                    "locked_wsl_s": locked,
                    "finished_wsl_s": finished,
                    **detail,
                    "thread_cpu_ms": (time.thread_time() - cpu) * 1000,
                }
            )
        return index, image, header

    def release(self, index):
        requested = time.perf_counter() if self.diagnose_queue else None
        cpu = time.thread_time() if self.diagnose_queue else None
        with self.condition:
            locked = time.perf_counter() if self.diagnose_queue else None
            if self.states[index] not in (1, 3):
                raise RuntimeError("Cannot release a buffer that is not owned")
            self.states[index] = 0
            seq = self.source_frames[index] if self.diagnose_queue else None
        if self.diagnose_queue:
            self.consumer_events.append(
                {
                    "operation": "release",
                    "seq": seq,
                    "slot": index,
                    "requested_wsl_s": requested,
                    "locked_wsl_s": locked,
                    "finished_wsl_s": time.perf_counter(),
                    "thread_cpu_ms": (time.thread_time() - cpu) * 1000,
                }
            )


def read_exact(stream, target, cancel=None):
    """直接写入共享空间，避免先拼接整幅图像的临时 bytes。"""
    view = memoryview(target).cast("B")
    offset = 0
    deadline = time.perf_counter() + 30
    while offset < len(view):
        if cancel is not None:
            if cancel.is_set():
                raise InterruptedError("Frame reception cancelled")
            if time.perf_counter() > deadline:
                raise TimeoutError("No complete camera message within 30 seconds")
            if not select.select([stream], [], [], 0.1)[0]:
                continue
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError("Windows frame pipe closed before a complete message")
        offset += count


def decoder_packet(compressed, header):
    """保留源播放时间；真实 RTSP 常没有 DTS，不把缺失标志当成巨大时间。"""
    packet = av.Packet(compressed)
    packet.pts = header["pts_ns"]
    packet.dts = None if header["dts_ns"] == 2**64 - 1 else header["dts_ns"]
    packet.time_base = Fraction(1, 10**9)
    return packet


def receive_frames(config, frames, cancel, done, result_pipe, completions=None):
    """管道读取独占一个小进程，避免多次 readinto 与分析争用 Python 执行锁。"""
    output = Path(config["output"])
    process = None
    received = []
    error = window = None
    converter = None
    converter_ready = False
    decoded_rows = []
    replacements = []
    copies = []
    latency_rows, ack_errors = [], []
    measure_latency = completions is not None
    writer_lock = threading.Lock()
    ack_stop = threading.Event()
    ack_thread = None
    try:
        # Ctrl+C 由父进程统一收尾；避免子进程在半个 H.264 包中间被打断。
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cv2.setNumThreads(1)
        decoder = av.CodecContext.create("h264", "r")
        decoder.thread_count = 1
        decoder.open()
        if config.get("offline_video"):
            # 模拟源先预热真实首帧；不会推进 Windows 的正式视频或追踪编号。
            with av.open(config["offline_video"]) as container:
                sample = next(container.decode(video=0))
            converter = ReusableConversion(sample, threads=1)
            for _ in range(8):
                converter.convert_bgr(sample)
                converter.to_gray()
                converter.release()
            converter_ready = True
            result_pipe.send({"decoder_ready": True})
        env = os.environ.copy()
        env.update(GST_DEBUG="2", GST_DEBUG_NO_COLOR="1", PYTHONDONTWRITEBYTECODE="1")
        env["WSLENV"] = ":".join(
            filter(
                None,
                [
                    env.get("WSLENV", ""),
                    "GST_DEBUG",
                    "GST_DEBUG_NO_COLOR",
                    "PYTHONDONTWRITEBYTECODE",
                ],
            )
        )
        with (output / "windows_stderr.log").open("xb") as log:
            process = subprocess.Popen(
                config["command"],
                cwd=ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                bufsize=0,
            )
            if measure_latency:

                def acknowledge_completed():
                    while not ack_stop.is_set():
                        try:
                            item = completions.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if item is None:
                            return
                        seq, timestamps = item
                        try:
                            send_ack(
                                process.stdin, writer_lock, "complete", seq, timestamps
                            )
                        except (OSError, ValueError) as exc:
                            ack_errors.append(str(exc))
                            return

                ack_thread = threading.Thread(
                    target=acknowledge_completed, name="latency-ack"
                )
                ack_thread.start()
            try:
                while not cancel.is_set():
                    requested = time.perf_counter()
                    if measure_latency:
                        with writer_lock:
                            process.stdin.write(b"GET\n")
                    else:
                        process.stdin.write(b"GET\n")
                    prefix = bytearray(12)
                    read_exact(process.stdout, prefix, cancel)
                    prefix_received = time.perf_counter() if measure_latency else None
                    size = struct.unpack("<I", prefix[8:])[0]
                    if prefix[:8] != b"CV2FRAME" or not 0 < size <= 65536:
                        raise ValueError("Invalid tracking frame header")
                    raw = bytearray(size)
                    read_exact(process.stdout, raw, cancel)
                    header = json.loads(raw)
                    if header.get("protocol") != 2:
                        raise ValueError("Tracking protocol version mismatch")
                    status = header["status"]
                    if status != "FRAME":
                        if header.get("payload_bytes") != 0:
                            raise ValueError("Unexpected payload in control response")
                        if status == "ERROR":
                            raise RuntimeError(header["error"])
                        if status == "END":
                            window = header["window"]
                            result_pipe.send({"window": window})
                            done.set()
                            # 保持相机/时钟服务，直到父进程完成 AFG 的独立收尾。
                            cancel.wait(config["seconds"] + 60)
                            break
                        if status != "WAIT":
                            raise ValueError(f"Unknown receiver status: {status}")
                        continue
                    if (
                        tuple(header["shape"]) != frames.shape
                        or header["pixel_format"] != "H264"
                        or not 0 < header["payload_bytes"] <= 16 * 1024 * 1024
                    ):
                        raise ValueError(
                            "Camera dimensions/format differ from prepared background"
                        )
                    compressed = bytearray(header["payload_bytes"])
                    read_exact(process.stdout, compressed, cancel)
                    arrived = time.perf_counter()
                    if measure_latency:
                        send_ack(
                            process.stdin,
                            writer_lock,
                            "received",
                            header["seq"],
                            {
                                "arrived_wsl_s": arrived,
                                "prefix_received_wsl_s": prefix_received,
                            },
                        )
                    header.update(
                        arrived_wsl_s=arrived, request_ms=(arrived - requested) * 1000
                    )
                    decode_started = time.perf_counter() if measure_latency else None
                    packet = decoder_packet(compressed, header)
                    decoded = decoder.decode(packet)
                    decode_finished = time.perf_counter() if measure_latency else None
                    if len(decoded) != 1 or decoded[0].pts != packet.pts:
                        raise ValueError(
                            "Expected one non-reordered frame per H264 packet"
                        )
                    if converter is None:
                        converter = ReusableConversion(decoded[0], threads=1)
                        for _ in range(8):
                            converter.convert_bgr(decoded[0])
                            converter.to_gray()
                            converter.release()
                        converter_ready = True
                        result_pipe.send({"decoder_ready": True})
                    converter.convert_bgr(decoded[0])
                    gray = converter.to_gray()
                    header["decoded_wsl_s"] = time.perf_counter()
                    if not header["in_window"]:
                        converter.release()
                        continue
                    index, image = frames.acquire(header["seq"])
                    copy_start = time.perf_counter() if frames.diagnose_queue else None
                    copy_cpu = time.thread_time() if frames.diagnose_queue else None
                    np.copyto(image, gray)
                    if frames.diagnose_queue:
                        copies.append(
                            {
                                "seq": header["seq"],
                                "slot": index,
                                "bytes": image.nbytes,
                                "started_wsl_s": copy_start,
                                "finished_wsl_s": time.perf_counter(),
                                "thread_cpu_ms": (time.thread_time() - copy_cpu) * 1000,
                            }
                        )
                    converter.release()
                    header["published_wsl_s"] = time.perf_counter()
                    if measure_latency:
                        header["latency"] = {
                            "decode_started_wsl_s": decode_started,
                            "decode_finished_wsl_s": decode_finished,
                            "arrived_wsl_s": arrived,
                            "converted_wsl_s": header["decoded_wsl_s"],
                            "shared_copy_ready_wsl_s": header["published_wsl_s"],
                        }
                    received.append(
                        {
                            k: header[k]
                            for k in (
                                "seq",
                                "video_frame",
                                "source_time_s",
                                "arrived_wsl_s",
                            )
                        }
                    )
                    decoded_rows.append(
                        {
                            k: header[k]
                            for k in (
                                "seq",
                                "video_frame",
                                "arrived_wsl_s",
                                "decoded_wsl_s",
                                "published_wsl_s",
                            )
                        }
                    )
                    discarded, depth = frames.publish(index, header)
                    if discarded is not None:
                        replacements.append(
                            {
                                "source_frame": discarded,
                                "replaced_by_source_frame": header["seq"],
                                "observed_wsl_s": time.perf_counter(),
                                "reason": "waiting_area_full_oldest_replaced",
                            }
                        )
                    if measure_latency:
                        latency_rows.append(
                            {
                                "seq": header["seq"],
                                **header["latency"],
                                "publish_finished_wsl_s": time.perf_counter(),
                                "waiting_depth_after_publish": depth,
                            }
                        )
            finally:
                if ack_thread is not None:
                    # 正常停止时父进程已排入结束标记；先发完完成确认，再发 QUIT。
                    ack_thread.join(3)
                    if ack_thread.is_alive():
                        ack_stop.set()
                        ack_thread.join(1)
                        ack_errors.append(
                            "Completion ACK worker did not drain normally"
                        )
                try:
                    process.stdin.write(b"QUIT\n")
                except (OSError, ValueError):
                    pass
                # 失败可能发生在帧头后；关闭读端让写端明确报错，不能无限等待。
                process.stdout.close()
                process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
                    error = "Camera receiver did not finish within 10 seconds"
    except BaseException as exc:  # noqa: BLE001 -- 子进程边界记录失败并通知父进程。
        if not (isinstance(exc, InterruptedError) and cancel.is_set()):
            error = f"{type(exc).__name__}: {exc}"
    finally:
        if converter is not None:
            if converter.in_use:
                converter.release()
            converter.close()
        if process is not None and process.returncode not in (0, None) and not error:
            error = f"Windows receiver exit code: {process.returncode}"
        if ack_errors and not error:
            error = "Latency acknowledgement failed: " + "; ".join(ack_errors)
        report = {
            "error": error,
            "received": received,
            "overwritten": frames.overwritten,
            "waiting_capacity": frames.waiting_frames,
            "peak_waiting_frames": frames.peak_waiting.value,
            "pending_at_shutdown": int(frames.pending.value >= 0)
            + int(frames.following.value >= 0),
            "replacement_events": replacements,
            "decoded": decoded_rows,
            "decoder_ready": converter_ready,
            "window": window,
            "exit_code": process.returncode if process else None,
        }
        write_json(output / "tracking_bridge.json", report)
        if frames.diagnose_queue:
            write_json(
                output / "queue_producer.json",
                {
                    "events": frames.producer_events,
                    "copies": copies,
                },
            )
        if measure_latency:
            write_json(
                output / "latency_wsl_decoder.json",
                {
                    "clock": clock_description(),
                    "frames": latency_rows,
                    "ack_errors": ack_errors,
                },
            )
        result_pipe.send({"error": error, "window": window})
        result_pipe.close()
        done.set()


class TrackingWindowsBridge:
    """管理一个传图进程；图像共享、状态用小消息传递，不经队列复制大图。"""

    def __init__(
        self,
        output,
        seconds,
        shape,
        *,
        camera_config,
        offline_video=None,
        recording=True,
        measure_latency=False,
        waiting_frames=1,
        diagnose_queue=False,
    ):
        self.output = Path(output)
        self.context = mp.get_context("spawn")
        self.frames = LatestFrames(self.context, shape, waiting_frames, diagnose_queue)
        self.stop_event, self.done = self.context.Event(), self.context.Event()
        self.reader, writer = self.context.Pipe(duplex=False)
        self.error = self.window = None
        self.ready = False
        self.completions = self.context.Queue(maxsize=128) if measure_latency else None
        command = [
            str(ROOT / "customer_workspace/.venv_gstreamer/Scripts/python.exe"),
            "-X",
            "utf8",
            "-u",
            "-B",
            "-m",
            "customer_workspace.parallel_camera",
            "--tracking",
            "--output",
            windows_path(self.output),
            "--seconds",
            str(seconds),
        ]
        if offline_video:
            command += ["--offline-video", windows_path(Path(offline_video))]
        else:
            command += ["--config", windows_path(Path(camera_config))]
        if not recording:
            command.append("--analysis-only")
        if measure_latency:
            command.append("--measure-latency")
        self.command = command
        self.process = self.context.Process(
            target=receive_frames,
            name="tracking-receiver",
            args=(
                {
                    "output": str(output),
                    "seconds": seconds,
                    "command": command,
                    "offline_video": str(offline_video) if offline_video else None,
                },
                self.frames,
                self.stop_event,
                self.done,
                writer,
                self.completions,
            ),
        )
        self.writer = writer

    def start(self):
        write_json(self.output / "launch.json", {"args": self.command, "protocol": 2})
        self.process.start()
        self.writer.close()
        return self

    def poll(self):
        while self.reader.poll():
            try:
                message = self.reader.recv()
            except EOFError:
                break
            self.error = message.get("error") or self.error
            self.window = message.get("window") or self.window
            self.ready = message.get("decoder_ready", self.ready)
        if self.process.exitcode is not None and not self.done.is_set():
            self.error = (
                self.error
                or f"Receiver process exited unexpectedly: {self.process.exitcode}"
            )

    def stop(self):
        if self.completions is not None:
            try:
                self.completions.put(None, timeout=2)
            except queue.Full:
                self.error = (
                    self.error or "Completion acknowledgement queue did not drain"
                )
        self.stop_event.set()
        if self.process.pid is not None:
            self.process.join(15)
            if self.process.is_alive():
                # 只报告无法收尾；不能杀掉拥有 Windows 子进程的进程并遗留接收器。
                self.error = (
                    self.error or "Receiver process has not stopped after 15 seconds"
                )
            self.poll()
        self.reader.close()
        if self.frames.diagnose_queue:
            write_json(
                self.output / "queue_consumer.json",
                {"events": self.frames.consumer_events},
            )
        if self.completions is not None:
            if self.error:
                # 子进程异常退出时不能无限等未被读取的诊断消息；失败会写入报告。
                self.completions.cancel_join_thread()
            self.completions.close()
            if not self.error:
                self.completions.join_thread()

    def acknowledge(self, header, completed_wsl_s):
        """诊断开启时提交完成确认；容量不足必须报错，不能静默漏掉确认。"""
        if self.completions is not None:
            self.completions.put_nowait(
                (
                    header["seq"],
                    {
                        "arrived_wsl_s": header["arrived_wsl_s"],
                        "completed_wsl_s": completed_wsl_s,
                    },
                )
            )
