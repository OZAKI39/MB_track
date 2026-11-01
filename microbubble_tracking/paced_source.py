"""独立进程按录像时间送帧；只保留一张待处理画面，不等待算法追上。"""

import csv
import json
import os
import time
import traceback
from pathlib import Path

import av
import cv2
import numpy as np

if __package__:
    from .frame_conversion import ReusableConversion
    from .online_replay import validated_frames
else:
    from frame_conversion import ReusableConversion
    from online_replay import validated_frames


DEFAULT_CONVERSION_WARMUP_ITERATIONS = 8

SLOT_FIELDS = (
    "video_frame",
    "source_frame",
    "video_s",
    "received_windows_s",
    "decoded_windows_s",
    "deadline_s",
    "published_s",
    "decode_ms",
)


class LatestFrame:
    """跨进程单帧槽：写入覆盖未取走的旧帧，取出时复制后立即释放锁。

    锁只保护约 1.3 MB 的灰度图及其时间记录；识别、写盘均在锁外执行。
    RawArray 不保存 numpy 视图，以免 spawn 时意外复制共享内存。
    """

    def __init__(self, context, shape):
        self.shape = tuple(shape)
        self.pixels = context.RawArray("B", int(np.prod(shape)))
        self.values = context.RawArray("d", len(SLOT_FIELDS))
        self.pending = context.RawValue("b", False)
        self.lock = context.Lock()
        self.available = context.Event()

    def put(self, gray, metadata, *, diagnostics=None):
        if gray.shape != self.shape or gray.dtype != np.uint8:
            raise ValueError("Frame size/type does not match the shared slot")
        with self.lock:
            if diagnostics is not None:
                diagnostics.mark("slot_lock")
            replaced = int(self.values[0]) if self.pending.value else None
            np.copyto(np.frombuffer(self.pixels, np.uint8).reshape(self.shape), gray)
            metadata = dict(metadata, published_s=time.perf_counter())
            self.values[:] = [metadata[key] for key in SLOT_FIELDS]
            self.pending.value = True
            self.available.set()
        return metadata, replaced

    def take(self, handoff=None):
        with self.lock:
            if handoff is not None:
                handoff.mark("take_lock")
            if not self.pending.value:
                return None
            image = np.frombuffer(self.pixels, np.uint8).reshape(self.shape).copy()
            if handoff is not None:
                handoff.mark("take_image_copy")
            metadata = dict(zip(SLOT_FIELDS, self.values[:]))
            if handoff is not None:
                handoff.mark("take_metadata")
            self.pending.value = False
            self.available.clear()
            if handoff is not None:
                handoff.mark("take_event_clear")
        metadata["video_frame"] = int(metadata["video_frame"])
        metadata["source_frame"] = int(metadata["source_frame"])
        return image, metadata


def wait_for_deadline(deadline_s, cancel, ahead_s):
    """保持绝对计划时刻；先可取消地等待，最后一小段用高精度计时循环。"""
    while not cancel.is_set():
        remaining = deadline_s - time.perf_counter()
        if remaining <= 0:
            return True
        if remaining > ahead_s:
            cancel.wait(remaining - ahead_s)
    return False


def calibrate_wait(cancel, path):
    """测本次 WSL 等待晚醒量，P99 加 0.5 ms 用作最后循环等待的提前量。"""
    rows = []
    for index in range(90):
        duration = (0.002, 0.005, 0.010)[index % 3]
        start = time.perf_counter()
        if cancel.wait(duration):
            raise InterruptedError("Cancelled during wait calibration")
        rows.append((duration * 1000, (time.perf_counter() - start - duration) * 1000))
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["requested_ms", "wake_late_ms"])
        writer.writerows(rows)
    p99_ms = float(np.percentile([r[1] for r in rows], 99))
    return {"wake_late_p99_ms": p99_ms, "ahead_ms": max(0, p99_ms) + 0.5}


def recording_frames(config):
    """流式读取；重复仅用于长时间负载检查，并保留原画面的独立对应表。

    正常回放不改变任何编号或时间。重复诊断使用虚拟回放编号与时间，
    不能解释为新的真实实验，也不能把跨轮接缝用于科学分析。
    """
    repeats = config.get("repeats", 1)
    offset = 0.0
    for cycle in range(repeats):
        with (
            av.open(config["video"]) as container,
            Path(config["timestamps"]).open(
                encoding="utf-8-sig", newline=""
            ) as timestamps,
        ):
            stream = container.streams.video[0]
            stream.thread_count = 1
            first_pts, count = None, 0
            for index, frame, row, pts_error in validated_frames(
                container.decode(stream),
                csv.DictReader(timestamps),
                config["experiment_id"],
            ):
                original_pts = float(row["video_s"])
                if first_pts is None:
                    first_pts = original_pts
                row = dict(
                    row,
                    replay_cycle=cycle,
                    original_video_frame=index,
                    original_source_frame=int(row["source_seq"]),
                    original_video_s=original_pts,
                )
                if repeats > 1:
                    index += cycle * config["expected_frames"]
                    row.update(
                        source_seq=index, video_s=offset + original_pts - first_pts
                    )
                yield index, frame, row, pts_error
                count += 1
            if count != config["expected_frames"]:
                raise ValueError("Repeated source cycle has an unexpected frame count")
            offset += original_pts - first_pts + 1 / float(stream.average_rate)


def warm_up_conversion(frame, cancel, iterations, conversion, output_path, converter=None):
    """在送帧线程中用真实首帧预热，既不推进录像，也不发布测试图像。

    第一遍真实转换负责申请并写入需要的空间，后续重复相同调用以初始化
    转换器和内存分配器。提供 converter 时预热正式复用的空间，结束后归还借用；
    否则沿用临时图像对照。两种模式都不推进正式画面或共享等待槽。
    """
    if type(iterations) is not int or not 0 <= iterations <= 32:
        raise ValueError("Conversion warm-up iterations must be an integer from 0 to 32")
    report = {
        "status": "failed",
        "pid": os.getpid(),
        "width": frame.width,
        "height": frame.height,
        "input_format": frame.format.name,
        "input_pts": frame.pts,
        "output_format": "bgr24 -> gray",
        "conversion_threads": conversion.get("threads"),
        "conversion_memory": "reuse" if converter is not None else "allocate",
        "iterations_requested": iterations,
        "iterations_completed": 0,
        "iterations": [],
        "started_s": time.perf_counter(),
    }
    bgr = gray = None
    try:
        for index in range(iterations):
            if cancel.is_set():
                raise InterruptedError("Cancelled during conversion warm-up")
            started_s = time.perf_counter()
            bgr = (
                converter.convert_bgr(frame) if converter is not None
                else frame.to_ndarray(format="bgr24", **conversion)
            )
            converted_s = time.perf_counter()
            # 保持正式流程的灰度范围与临时图像生命周期，不换算法或接口。
            gray = (
                converter.to_gray() if converter is not None
                else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            )
            finished_s = time.perf_counter()
            report["bgr_bytes"] = bgr.nbytes
            report["gray_bytes"] = gray.nbytes
            report["iterations"].append({
                "iteration": index,
                "convert_bgr_ms": (converted_s - started_s) * 1000,
                "compute_gray_ms": (finished_s - converted_s) * 1000,
            })
            del bgr
            bgr = None
            if converter is not None:
                converter.release()
            report["iterations_completed"] += 1
        if cancel.is_set():
            raise InterruptedError("Cancelled during conversion warm-up")
        report["status"] = "completed" if iterations else "disabled"
        return report
    except (InterruptedError, KeyboardInterrupt) as error:
        report.update(status="cancelled", error=str(error))
        raise
    except BaseException:  # 预热失败须留原因并交给既有退出处理。
        report["error"] = traceback.format_exc()
        raise
    finally:
        # 取消/异常也归还借用；固定空间由送帧进程继续持有。
        del bgr, gray
        if converter is not None and converter.in_use:
            converter.release()
        report["finished_s"] = time.perf_counter()
        report["duration_ms"] = (report["finished_s"] - report["started_s"]) * 1000
        with Path(output_path).open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")


def send_video(config, slot, ready, start, done, cancel, origin):
    """发送与处理完全独立；每条计划均由原始视频时间加同一个起点计算。"""
    output = Path(config["output"])
    report = {"status": "failed", "offered_frames": 0, "max_pts_error_s": 0.0}
    mapping_fields = (
        "replay_cycle",
        "original_video_frame",
        "original_source_frame",
        "original_video_s",
    )
    fields = [*SLOT_FIELDS, "replaced_video_frame", *mapping_fields]
    cv2.setNumThreads(1)
    probe = None
    converter = None
    try:
        memory_mode = config.get("conversion_memory", "reuse")
        if memory_mode not in ("allocate", "reuse"):
            raise ValueError("conversion_memory must be allocate or reuse")
        if config.get("diagnose_stalls"):
            if __package__:
                from .stall_probe import StallProbe
            else:
                from stall_probe import StallProbe
            probe = StallProbe(output / "source_stalls.csv")
        report["wait_calibration"] = calibrate_wait(
            cancel, output / "wait_calibration.csv"
        )
        with (
            (output / "source_timing.csv").open(
                "x", encoding="utf-8", newline="", buffering=1
            ) as log,
        ):
            report["decoder_threads"] = 1
            writer = csv.DictWriter(log, fields)
            writer.writeheader()
            frames = recording_frames(config)
            first_pts = None
            while not cancel.is_set():
                if probe is not None:
                    probe.begin(report["offered_frames"])
                decode_start = time.perf_counter()
                item = next(frames, None)
                if probe is not None:
                    probe.mark("read_decode")
                if item is None:
                    break
                index, frame, row, pts_error = item
                # 与第一轮保持相同的 BGR -> gray 转换，避免更换灰度范围。
                conversion = {}
                if config.get("conversion_threads") is not None:
                    conversion["threads"] = config["conversion_threads"]
                if first_pts is None:
                    warmup_started_s = time.perf_counter()
                    if memory_mode == "reuse":
                        converter = ReusableConversion(
                            frame, threads=config.get("conversion_threads")
                        )
                    report["conversion_memory"] = (
                        converter.metadata if converter is not None else {"mode": "allocate"}
                    )
                    report["conversion_warmup"] = warm_up_conversion(
                        frame,
                        cancel,
                        config.get(
                            "conversion_warmup_iterations",
                            DEFAULT_CONVERSION_WARMUP_ITERATIONS,
                        ),
                        conversion,
                        output / "source_warmup.json",
                        converter,
                    )
                    report["conversion_preparation_ms"] = (
                        time.perf_counter() - warmup_started_s
                    ) * 1000
                    if probe is not None:
                        probe.mark("startup_warmup")
                    # 独立记录预热；不把它算进正式帧的解码/转换耗时。
                    decode_start += time.perf_counter() - warmup_started_s
                bgr = (
                    converter.convert_bgr(frame) if converter is not None
                    else frame.to_ndarray(format="bgr24", **conversion)
                )
                if probe is not None:
                    probe.mark("convert_bgr")
                next_gray = (
                    converter.to_gray() if converter is not None
                    else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                )
                if probe is not None:
                    probe.mark("compute_gray")
                gray = next_gray
                del next_gray
                if probe is not None:
                    probe.mark("release_previous_gray")
                del bgr
                if probe is not None:
                    probe.mark("release_bgr")
                decode_ms = (time.perf_counter() - decode_start) * 1000
                pts = float(row["video_s"])
                if first_pts is None:
                    first_pts = pts
                    ready.set()
                    while not start.wait(0.05):
                        if cancel.is_set():
                            raise InterruptedError("Cancelled before playback")
                if probe is not None:
                    probe.mark("start_handshake")
                deadline = origin.value + (pts - first_pts)
                if not wait_for_deadline(
                    deadline, cancel, report["wait_calibration"]["ahead_ms"] / 1000
                ):
                    break
                if probe is not None:
                    probe.mark("deadline_wait")
                metadata, replaced = slot.put(
                    gray,
                    {
                        "video_frame": index,
                        "source_frame": int(row["source_seq"]),
                        "video_s": pts,
                        "received_windows_s": float(row["received_windows_s"]),
                        "decoded_windows_s": float(row["decoded_windows_s"]),
                        "deadline_s": deadline,
                        "decode_ms": decode_ms,
                    },
                    diagnostics=probe,
                )
                # put 已在锁内复制图像；等待区与追踪者不持有转换空间的视图。
                if converter is not None:
                    converter.release()
                if probe is not None:
                    probe.mark("slot_copy_notify")
                writer.writerow(
                    dict(
                        metadata,
                        replaced_video_frame=replaced,
                        **{key: row[key] for key in mapping_fields},
                    )
                )
                if probe is not None:
                    probe.mark("source_log_write")
                    probe.finish()
                report["offered_frames"] += 1
                report["max_pts_error_s"] = max(report["max_pts_error_s"], pts_error)
                if (
                    config["max_frames"]
                    and report["offered_frames"] >= config["max_frames"]
                ):
                    break
            total = config["expected_frames"] * config.get("repeats", 1)
            expected = min(config["max_frames"] or total, total)
            if not cancel.is_set() and report["offered_frames"] != expected:
                raise ValueError("Decoded video and expected frame counts differ")
            report["status"] = "cancelled" if cancel.is_set() else "completed"
            frames.close()
    except (InterruptedError, KeyboardInterrupt) as error:
        report.update(status="cancelled", error=str(error))
    except BaseException:  # noqa: BLE001 -- 子进程任何退出都必须通知消费者并保存原因。
        report["error"] = traceback.format_exc()
    finally:
        if converter is not None:
            if converter.in_use:
                converter.release()
            converter.close()
        if probe is not None:
            probe.close()
        (output / "source_report.json").write_text(json.dumps(report, indent=2) + "\n")
        done.set()
        slot.available.set()
