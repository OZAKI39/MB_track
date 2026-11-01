"""来一帧处理一帧的接口；不采集画面、不访问仪器、不读取未来帧。"""

import gc
import hashlib
import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .async_store import AsyncStore, SaveSettings
    from .bubble_tracker import (
        OnlineBackground,
        SegParams,
        Tracker,
        TrackParams,
        segment,
    )
    from .online_store import OnlineStore, snapshot_frame
else:
    from async_store import AsyncStore, SaveSettings
    from bubble_tracker import (
        OnlineBackground,
        SegParams,
        Tracker,
        TrackParams,
        segment,
    )
    from online_store import OnlineStore, snapshot_frame


TIMING_FIELDS = (
    "input_ms",
    "frame_setup_ms",
    "segment_ms",
    "track_ms",
    "background_ms",
    "pack_records_ms",
    "database_write_ms",
    "commit_ms",
    "read_updates_ms",
    "return_ms",
    "other_ms",
    "process_total_ms",
    "seg_blur_ms",
    "seg_contrast_ms",
    "seg_candidates_ms",
    "seg_components_ms",
    "seg_selection_ms",
    "seg_contours_ms",
    "track_pixels_ms",
    "track_features_ms",
    "track_matching_ms",
    "commit_cpu_ms",
)


@dataclass(frozen=True)
class FrameStamp:
    """源端身份与画面时间，不能用 WSL 收到调用的时间代替。

    source_frame 是源端连续编号；即使中间没有送来，也不可重新编号。
    source_time_s 属于本次流固定的 source_clock。video_frame 从 0 开始，
    与相机源编号不同；尚未建立录像对应关系时允许留空，不能猜测。
    Windows 接收/解码时刻仅原样记录，不与本机计时器相减。
    """

    experiment_id: str
    stream_id: str
    source_frame: int
    source_time_s: float
    video_frame: int | None = None
    video_time_s: float | None = None
    received_windows_s: float | None = None
    decoded_windows_s: float | None = None


class OnlineProcessor:
    """只持有背景和仍可能继续匹配的目标，历史逐帧写入 SQLite。

    background 必须是事先准备、对齐到当前视野的灰度背景（0–255）。
    默认异步保存：process() 返回表示帧已被有界保存队列接收，不代表落盘。
    poll_saved() 领取已落盘通知，close() 等待保存完成并核对帧数。
    save_mode="sync" 保留同步提交，供严格对照。接口由单个分析线程调用；
    保存线程只拿独立数据快照，不操作分析状态，也不擅自丢弃输入帧。
    """

    def __init__(
        self,
        *,
        experiment_id: str,
        stream_id: str,
        source_clock: str,
        background: np.ndarray,
        nominal_fps: float,
        output_dir: Path,
        seg: SegParams | None = None,
        tracking: TrackParams | None = None,
        max_gap_s: float | None = None,
        source_time_resolution_s: float = 0.0,
        provenance: dict | None = None,
        measure_stages: bool = False,
        diagnose_stalls: bool = False,
        save_mode: str = "async",
        save_settings: SaveSettings | None = None,
    ):
        if save_mode not in ("sync", "async"):
            raise ValueError("save_mode must be sync or async")
        self.save_mode = save_mode
        save_settings = save_settings or SaveSettings()
        for name, value in (
            ("experiment_id", experiment_id),
            ("stream_id", stream_id),
            ("source_clock", source_clock),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not math.isfinite(nominal_fps) or nominal_fps <= 0:
            raise ValueError("nominal_fps must be positive")
        bg = np.asarray(background, dtype=np.float32)
        if (
            bg.ndim != 2
            or min(bg.shape) < 10
            or not np.isfinite(bg).all()
            or bg.min() < 0
            or bg.max() > 255
        ):
            raise ValueError("background must be a finite grayscale image in [0,255]")
        self.seg = replace(seg) if seg is not None else SegParams()
        tp = replace(tracking) if tracking is not None else TrackParams()
        if (
            tp.min_len < 1
            or tp.max_gap < 1
            or not 0 <= tp.vel_alpha <= 1
            or self.seg.close < 1
        ):
            raise ValueError("Invalid tracking or segmentation parameters")
        max_gap_s = tp.max_gap / nominal_fps if max_gap_s is None else max_gap_s
        if not math.isfinite(source_time_resolution_s) or source_time_resolution_s < 0:
            raise ValueError("source_time_resolution_s must be finite and nonnegative")
        self.tracker = Tracker(
            bg.shape,
            tp,
            retain_history=False,
            time_based=True,
            max_gap_s=max_gap_s,
            time_tolerance_s=2 * source_time_resolution_s,
        )
        self.background = OnlineBackground(bg)
        # 申请空间不等于已完成首次写入；在构造返回、允许送帧之前实际访问整张图。
        # 这里只准备工作内存，不调用分割、追踪或背景更新，也不消耗正式帧号。
        label_preparation_started = time.perf_counter()
        self.background.workspace.component_labels.fill(0)
        label_preparation_finished = time.perf_counter()
        # 只供当前分析调用使用：step/update 均同步消费，保存队列不接收整张编号图。
        self._labels = np.zeros(bg.shape, np.uint16)
        # 平滑图也只在本帧同步使用；start() 由实际分析线程写入并预热整张图。
        self._blurred = np.empty_like(self.background.B)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.seg.close, self.seg.close)
        )
        self.experiment_id, self.stream_id = experiment_id, stream_id
        self.last_stamp = None
        self.last_video_frame = None
        self.last_video_time = None
        self.closed = self.failed = False
        self.stream_started = False
        self._analysis_owner = None
        self.gc_lifecycle = {
            "policy": "full collection before stream and after save; automatic settings unchanged",
        }
        self.failure_status = None
        self.peak_active = self.skipped_frames = 0
        self.output = Path(output_dir)
        self.measure_stages = measure_stages
        self.metadata = dict(
            schema_version=1,
            experiment_id=experiment_id,
            stream_id=stream_id,
            run_id=uuid.uuid4().hex,
            source_clock=source_clock,
            windows_clock="Windows monotonic receipt/decode fields; never subtract WSL time",
            nominal_fps=nominal_fps,
            max_gap_s=max_gap_s,
            status="running",
            source_time_resolution_s=source_time_resolution_s,
            time_tolerance_s=2 * source_time_resolution_s,
            segmentation=asdict(self.seg),
            tracking=asdict(tp),
            background_sha256=hashlib.sha256(bg.tobytes()).hexdigest(),
            implementation_sha256={
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in (
                    "bubble_tracker.py",
                    "online_tracking.py",
                    "online_store.py",
                    "async_store.py",
                )
            },
            provenance=provenance or {},
            velocity_units="px/s; raw displacement/dt and smoothed predictor both saved",
            confirmation="algorithmic, not human review or physical verification",
            processing_ms="segmentation, matching and background update; excludes disk commit",
            measure_stages=measure_stages,
            diagnose_stalls=diagnose_stalls,
            startup_preparation={
                "component_labels_ready": True,
                "component_labels_bytes": self.background.workspace.component_labels.nbytes,
                "component_labels_ms": (
                    label_preparation_finished - label_preparation_started
                ) * 1000,
                "component_labels_completed_perf_s": label_preparation_finished,
                "analysis_warmup_required": True,
                "analysis_warmup_report": "analysis_preparation.json",
                "clock": "analysis host perf_counter; not camera exposure time",
            },
            persistence=dict(mode=save_mode, **asdict(save_settings)),
        )
        encoded = json.dumps(
            self.metadata, ensure_ascii=False, indent=2, allow_nan=False
        )
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "session.json").write_text(encoded + "\n", encoding="utf-8")
        np.save(self.output / "background_initial.npy", bg)
        store_args = (self.output / "tracking.sqlite", self.metadata, tp.min_len)
        self.store = (
            AsyncStore(*store_args, save_settings)
            if save_mode == "async"
            else OnlineStore(
                *store_args, checkpoint_policy=save_settings.checkpoint_policy,
                wal_limit_mb=save_settings.wal_limit_mb,
            )
        )

    def _write_gc_lifecycle(self):
        (self.output / "gc_lifecycle.json").write_text(
            json.dumps(self.gc_lifecycle, indent=2) + "\n", encoding="utf-8"
        )

    def _collect_garbage(self, phase):
        """只在流的边界完整清理，不安装逐帧探针或改变进程级 GC 设置。"""
        record = {
            "status": "running", "generation": 2,
            "pid": os.getpid(), "thread_id": threading.get_native_id(),
            "enabled_before": gc.isenabled(),
            "thresholds_before": list(gc.get_threshold()),
        }
        self.gc_lifecycle[phase] = record
        failure = None
        started, cpu_started = time.perf_counter(), time.thread_time()
        record["started_perf_s"] = started
        try:
            if phase == "before_stream" and not gc.isenabled():
                raise RuntimeError("Automatic GC must be enabled before starting analysis")
            record["collected_objects"] = gc.collect(2)
            record["status"] = "completed"
        except BaseException as exc:
            failure = exc
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finished, cpu_finished = time.perf_counter(), time.thread_time()
        record.update(
            finished_perf_s=finished,
            wall_ms=(finished - started) * 1000,
            thread_cpu_ms=(cpu_finished - cpu_started) * 1000,
            enabled_after=gc.isenabled(),
            thresholds_after=list(gc.get_threshold()),
        )
        try:
            self._write_gc_lifecycle()
        except BaseException as report_error:
            if failure is None:
                raise
            failure.add_note(f"Could not save GC lifecycle: {report_error}")
        if failure is not None:
            raise failure

    def _warm_up_analysis(self, sample_frame):
        """在分析线程试跑现有算法；只复用工作区，临时轨迹不进入正式状态。"""
        workspace = self.background.workspace
        buffers = {
            "blurred": self._blurred,
            "darkness": workspace._darkness,
            "component_labels": workspace.component_labels,
            "selection_mask": workspace.selection_mask,
            "label_indices": workspace._label_indices,
            "output_labels": self._labels,
        }
        record = {
            "status": "running", "pid": os.getpid(),
            "thread_id": threading.get_native_id(),
            "shape": list(self.background.B.shape),
            "sample": "supplied frame" if sample_frame is not None else "synthetic only",
            "iterations": 3,
            "buffer_bytes": {name: array.nbytes for name, array in buffers.items()},
            "clock": "analysis host perf_counter; not exposure time",
        }
        started = time.perf_counter()
        record["started_perf_s"] = started
        background_hash = hashlib.sha256(self.background.B.tobytes()).hexdigest()
        failure = None
        try:
            # empty/zeros 可能仅申请了虚拟地址；先实际写完每个工作区。
            for array in buffers.values():
                array.fill(0)
            workspace.refresh_rows(0, self.background.B.shape[0])
            synthetic = self.background.B.astype(np.uint8)
            height, width = synthetic.shape
            synthetic[height // 4:height // 2, width // 4:width // 2] = 0
            synthetic[5 * height // 8:7 * height // 8,
                      5 * width // 8:7 * width // 8] = 0
            # 即使真实首帧没有目标，也先访问一次轮廓和形状计算路径。
            sample = synthetic if sample_frame is None else sample_frame
            temporary = Tracker(
                self.background.B.shape, replace(self.tracker.tp),
                retain_history=False, time_based=True,
                max_gap_s=self.tracker.max_gap_s,
                time_tolerance_s=self.tracker.time_tolerance_s,
            )
            target_counts = []
            for index in range(record["iterations"]):
                stamp = FrameStamp(
                    self.experiment_id, self.stream_id,
                    index, index / self.metadata["nominal_fps"],
                )
                gray = self._validate(synthetic if index == 0 else sample, stamp)
                labels, contours, _ = segment(
                    gray, self.background.B, self.seg, self.kernel,
                    workspace=workspace, labels_out=self._labels,
                    blurred_out=self._blurred,
                )
                temporary.step(index, labels, contours, timestamp_s=stamp.source_time_s)
                target_counts.append(len(contours))
            record["temporary_region_counts"] = target_counts
        except BaseException as exc:
            failure = exc
        # 清空计算缓存；从未调用正式 tracker.step、background.update 或 store.submit。
        try:
            for array in buffers.values():
                array.fill(0)
            unchanged = (
                hashlib.sha256(self.background.B.tobytes()).hexdigest() == background_hash
                and self.background.k == 0 and self.last_stamp is None
                and self.tracker.next_id == 1 and not self.tracker.tracks
                and self.tracker.previous_frame is None
            )
            record["scientific_state_unchanged"] = unchanged
            if not unchanged:
                raise RuntimeError("Analysis warm-up changed scientific state")
        except BaseException as cleanup_error:
            if failure is None:
                failure = cleanup_error
            else:
                failure.add_note(f"Warm-up cleanup also failed: {cleanup_error}")
        record.update(
            status="completed" if failure is None else "failed",
            finished_perf_s=time.perf_counter(),
        )
        record["duration_ms"] = (record["finished_perf_s"] - started) * 1000
        if failure is not None:
            record["error"] = f"{type(failure).__name__}: {failure}"
        try:
            (self.output / "analysis_preparation.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
        except BaseException as report_error:
            if failure is None:
                raise
            failure.add_note(f"Could not save warm-up report: {report_error}")
        if failure is not None:
            raise failure

    def start(self, sample_frame: np.ndarray | None = None):
        """在分析线程预热并清理，返回后才允许正式送帧；重复调用不重做。

        应由实际执行 process 的分析线程调用。正式回放入口在发出开始信号前
        传入已取得的第一张画面，只用于准备，不消耗它的编号或写入记录。
        未提供样例时使用背景生成的临时画面；原始输入和正式背景都不修改。
        """
        if self.closed or self.failed:
            raise RuntimeError("Processor is closed or failed; start a new session")
        owner = (os.getpid(), threading.get_native_id())
        if self.stream_started:
            if owner != self._analysis_owner:
                raise RuntimeError("Analysis must run in the process and thread that called start")
            return
        try:
            self._warm_up_analysis(sample_frame)
            self._collect_garbage("before_stream")
            ready = {
                "status": "ready", "pid": owner[0], "thread_id": owner[1],
                "ready_perf_s": time.perf_counter(),
                "clock": "analysis host perf_counter; not exposure time",
            }
            (self.output / "analysis_ready.json").write_text(
                json.dumps(ready, indent=2) + "\n", encoding="utf-8"
            )
        except BaseException as exc:
            self.failed = True
            self.failure_status = (
                "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            )
            raise
        self._analysis_owner = owner
        self.stream_started = True

    def poll_saved(self):
        """非阻塞领取已保存通知；空闲时也可调用，用来及时发现后台保存错误。"""
        if self.save_mode == "sync":
            return []
        try:
            return self.store.poll()
        except Exception:
            self.failed, self.failure_status = True, "failed"
            raise

    def _validate(self, image: np.ndarray, stamp: FrameStamp) -> np.ndarray:
        if not isinstance(stamp, FrameStamp):
            raise ValueError("stamp must be a FrameStamp")
        if (stamp.experiment_id, stamp.stream_id) != (
            self.experiment_id,
            self.stream_id,
        ):
            raise ValueError("Experiment or stream identity does not match")
        for name in ("source_frame", "video_frame"):
            value = getattr(stamp, name)
            if value is None and name == "video_frame":
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "source_time_s",
            "video_time_s",
            "received_windows_s",
            "decoded_windows_s",
        ):
            value = getattr(stamp, name)
            if value is None and name != "source_time_s":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if (stamp.video_frame is None) != (stamp.video_time_s is None):
            raise ValueError("video_frame and video_time_s must be supplied together")
        if self.last_stamp is not None:
            if stamp.source_frame <= self.last_stamp.source_frame:
                raise ValueError("Source frame numbers must increase")
            if stamp.source_time_s <= self.last_stamp.source_time_s:
                raise ValueError("Source timestamps must increase")
        if stamp.video_frame is not None and self.last_video_frame is not None:
            if (
                stamp.video_frame <= self.last_video_frame
                or stamp.video_time_s <= self.last_video_time
            ):
                raise ValueError("Video frame numbers and timestamps must increase")
        image = np.asarray(image)
        if image.shape == (*self.background.B.shape, 3) and image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape != self.background.B.shape or image.dtype not in (
            np.uint8,
            np.float32,
        ):
            raise ValueError(
                "Expected matching-size uint8 BGR, or uint8/float32 grayscale"
            )
        # uint8 在类型上已保证有限且处于 0–255；只有 float32 需要逐像素核查。
        if image.dtype == np.float32 and (
            not np.isfinite(image).all() or image.min() < 0 or image.max() > 255
        ):
            raise ValueError("Image values must be finite and in [0,255]")
        return image.astype(np.float32, copy=False)

    def process(self, image: np.ndarray, stamp: FrameStamp, *, diagnostics=None) -> dict:
        """返回当帧轮廓和观测；保存通知可能来自前面已提交的帧。

        输入不合法时记录拒绝原因并抛 ValueError，不推进状态。计算/写盘
        失败则停止本实例。events / track_updates 仅包含已落盘的状态修订，
        异步时请按每条记录的 reported_frame / anchor_frame 对应原画面。
        saved_updates 同时提供这些修订所在帧的保存确认；不包含未来信息。
        diagnostics 仅供离线诊断：调用方负责 begin/finish，默认不启用。
        """
        if self.closed or self.failed:
            raise RuntimeError("Processor is closed or failed; start a new session")
        # 兼容直接调用；正式入口应显式 start，避免预热和清理占用首帧时间。
        if not self.stream_started:
            self.start()
        elif (os.getpid(), threading.get_native_id()) != self._analysis_owner:
            raise RuntimeError("Analysis must run in the process and thread that called start")
        call_started = time.perf_counter() if self.measure_stages else 0.0
        timings = {} if self.measure_stages else None
        notices = []
        try:
            gray = self._validate(image, stamp)
        except (ValueError, TypeError) as exc:
            try:
                self.store.reject(str(exc), stamp)
            except Exception:
                self.failed = True
                self.failure_status = "failed"
                raise
            raise
        after_validation = time.perf_counter() if self.measure_stages else 0.0
        if diagnostics is not None:
            diagnostics.mark("input")
        previous = {
            i: dict(frame=t.last, time_s=t.time_s, cx=t.cx, cy=t.cy)
            for i, t in self.tracker.tracks.items()
        }
        gap = (
            0
            if self.last_stamp is None
            else stamp.source_frame - self.last_stamp.source_frame - 1
        )
        dt = (
            None
            if self.last_stamp is None
            else stamp.source_time_s - self.last_stamp.source_time_s
        )
        try:
            started = time.perf_counter()
            if diagnostics is not None:
                diagnostics.mark("frame_setup")
            labels, contours, blurred = segment(
                gray,
                self.background.B,
                self.seg,
                self.kernel,
                timings=timings,
                workspace=self.background.workspace,
                diagnostics=diagnostics,
                labels_out=self._labels,
                blurred_out=self._blurred,
            )
            after_segment = time.perf_counter() if self.measure_stages else 0.0
            result = self.tracker.step(
                stamp.source_frame,
                labels,
                contours,
                timestamp_s=stamp.source_time_s,
                timings=timings,
            )
            after_tracking = time.perf_counter() if self.measure_stages else 0.0
            if diagnostics is not None:
                diagnostics.mark("tracking")
            self.background.update(blurred, labels)
            after_background = time.perf_counter() if self.measure_stages else 0.0
            if diagnostics is not None:
                diagnostics.mark("background")
            frame = asdict(stamp)
            frame.update(
                run_id=self.metadata["run_id"],
                gap_frames=gap,
                delta_time_s=dt,
                processing_ms=(time.perf_counter() - started) * 1000,
                active_tracks=len(self.tracker.tracks),
            )
            before_store = time.perf_counter() if self.measure_stages else 0.0
            if self.save_mode == "async":
                packet_started = time.perf_counter()
                packet = snapshot_frame(
                    frame,
                    result,
                    self.tracker.tracks,
                    previous,
                    self.tracker.tp.min_len,
                )
                notices = self.store.submit(packet)
                store_timings = dict(
                    pack_records_ms=(time.perf_counter() - packet_started) * 1000,
                    database_write_ms=0.0,
                    commit_ms=0.0,
                    commit_cpu_ms=0.0,
                )
            else:
                store_timings = self.store.append(
                    frame, result, self.tracker.tracks, previous
                )
            after_store = time.perf_counter() if self.measure_stages else 0.0
            if diagnostics is not None:
                diagnostics.mark("submit")
            if self.save_mode == "async":
                updates = {
                    name: [row for notice in notices for row in notice[name]]
                    for name in ("events", "track_updates")
                }
            else:
                updates = self.store.updates(stamp.source_frame)
            after_updates = time.perf_counter() if self.measure_stages else 0.0
        except BaseException as exc:
            self.failed = True
            self.failure_status = (
                "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            )
            raise
        self.last_stamp = stamp
        if stamp.video_frame is not None:
            self.last_video_frame, self.last_video_time = (
                stamp.video_frame,
                stamp.video_time_s,
            )
        self.skipped_frames += gap
        self.peak_active = max(self.peak_active, len(self.tracker.tracks))
        for row in result["rows"]:
            row["source_frame"] = row.pop("frame")
            row["track_id"] = row.pop("id")
            row["vx_px_s"], row["vy_px_s"] = row.pop("vx"), row.pop("vy")
        response = dict(
            frame=frame,
            observations=result["rows"],
            contours=result["contours"],
            saved_updates=notices,
            persistence=(
                self.store.status()
                if self.save_mode == "async"
                else {"last_committed_source_frame": stamp.source_frame}
            ),
            **updates,
        )
        if self.measure_stages:
            finished = time.perf_counter()
            timings.update(
                input_ms=(after_validation - call_started) * 1000,
                frame_setup_ms=(
                    started - after_validation + before_store - after_background
                )
                * 1000,
                segment_ms=(after_segment - started) * 1000,
                track_ms=(after_tracking - after_segment) * 1000,
                background_ms=(after_background - after_tracking) * 1000,
                read_updates_ms=(after_updates - after_store) * 1000,
                return_ms=(finished - after_updates) * 1000,
                **store_timings,
            )
            total_ms = (finished - call_started) * 1000
            # 主阶段可相加；seg_* / track_* 是内部拆分，不能再重复加一次。
            timings["other_ms"] = total_ms - sum(
                timings[key] for key in TIMING_FIELDS[:10]
            )
            timings["process_total_ms"] = total_ms
            response["timings"] = timings
        return response

    def close(self, status: str = "completed") -> dict:
        """停止接收、等待保存退出，再清理；任何收尾错误都不能掩盖原始错误。"""
        if status not in ("completed", "cancelled", "failed"):
            raise ValueError("Unknown completion status")
        if self.closed:
            if getattr(self, "close_error", None):
                raise RuntimeError(self.close_error)
            return self.summary
        status = self.failure_status or status
        self.closed = True
        errors = []
        try:
            try:
                self.summary = self.store.finish(status)
            except KeyboardInterrupt as interrupt:
                # 用户在等待保存时取消，也先尽力排空，而不是遗留后台保存线程。
                try:
                    self.summary = self.store.finish("cancelled")
                except BaseException as finish_error:
                    interrupt.add_note(f"Draining records after cancellation failed: {finish_error}")
                raise
            # 失败后内存可能已推进，不能把未提交的背景冒充可靠结果。
            if status != "failed" and not self.failed:
                np.save(self.output / "background_final.npy", self.background.B)
        except BaseException as exc:
            errors.append(exc)
        self.tracker.tracks.clear()
        try:
            self.store.close()
        except BaseException as exc:
            errors.append(exc)
        try:
            if self.save_mode == "async" and self.store.worker.is_alive():
                # 超时不是“保存完成”：不与仍在写盘的线程同时执行收尾清理。
                self.gc_lifecycle["after_save"] = {
                    "status": "skipped", "reason": "save_worker_still_running",
                }
                self._write_gc_lifecycle()
            else:
                self.gc_lifecycle["save_finished_perf_s"] = time.perf_counter()
                self._collect_garbage("after_save")
        except BaseException as exc:
            errors.append(exc)
        if errors:
            self.failed = True
            self.close_error = f"{type(errors[0]).__name__}: {errors[0]}"
            self.summary = dict(
                getattr(self, "summary", {}),
                status="cancelled" if isinstance(errors[0], KeyboardInterrupt) else "failed",
                error=self.close_error,
            )
        if self.save_mode == "async":
            self.summary["persistence"] = self.store.status()
        try:
            (self.output / "summary.json").write_text(
                json.dumps(self.summary, indent=2) + "\n", encoding="utf-8"
            )
        except BaseException as exc:
            errors.append(exc)
            self.failed = True
            self.close_error = f"{type(errors[0]).__name__}: {errors[0]}"
            self.summary.update(
                status="cancelled" if isinstance(errors[0], KeyboardInterrupt) else "failed",
                error=self.close_error,
            )
        if errors:
            for later in errors[1:]:
                errors[0].add_note(f"Additional shutdown error: {type(later).__name__}: {later}")
            raise errors[0]
        return self.summary

    def __enter__(self):
        try:
            self.start()
        except BaseException as exc:
            try:
                self.close("cancelled" if isinstance(exc, KeyboardInterrupt) else "failed")
            except BaseException as close_error:
                exc.add_note(f"Also failed to close online records: {close_error}")
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        status = (
            "completed"
            if exc is None
            else "cancelled"
            if isinstance(exc, KeyboardInterrupt)
            else "failed"
        )
        try:
            self.close(status)
        except BaseException as close_error:
            if exc is None:
                raise
            exc.add_note(f"Also failed to close online records: {close_error}")
