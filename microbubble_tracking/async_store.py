"""有界的保存线程：复用 OnlineStore，分析端不等待磁盘提交。

队列容量包括正在写入的帧；异常立即停止接收，已接收的数据尽力排空。
所有数据库操作都在同一个保存线程中进行，SQLite 仍使用 FULL 同步。
"""

import csv
import json
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass

if __package__:
    from .online_store import OnlineStore
else:
    from online_store import OnlineStore


@dataclass(frozen=True)
class SaveSettings:
    queue_frames: int = 32
    batch_frames: int = 4
    max_pending_s: float = 1.0
    close_timeout_s: float = 30.0
    checkpoint_policy: str = "auto"
    wal_limit_mb: int = 64

    def __post_init__(self):
        for value in (self.queue_frames, self.batch_frames, self.wal_limit_mb):
            if type(value) is not int or value < 1:
                raise ValueError("Save queue and batch sizes must be positive integers")
        for value in (self.max_pending_s, self.close_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Save deadlines must be finite and positive")
        if self.checkpoint_policy not in ("auto", "deferred"):
            raise ValueError("checkpoint_policy must be auto or deferred")


class SaveError(RuntimeError):
    """数据未能按保存约定完成；细节保存在 writer_report.json。"""


class AsyncStore:
    """单生产者接口；仅提交、poll、finish 由分析线程调用。"""

    def __init__(self, path, metadata, min_observations, settings):
        self.path, self.settings = path, settings
        self.jobs = queue.Queue(maxsize=settings.queue_frames)
        self.notifications = queue.Queue(maxsize=settings.queue_frames * 2 + 1)
        self.lock = threading.Lock()
        self.ready, self.stopping, self.done = [threading.Event() for _ in range(3)]
        self.pending = {}
        self.submitted = self.committed = self.sequence = self.peak_pending = 0
        self.last_committed_frame = None
        self.error = None
        self.error_details = None
        self.requested_status = "completed"
        self.summary = None
        self.finish_ms = self.close_ms = None
        self.worker = threading.Thread(
            target=self._run,
            args=(metadata, min_observations),
            name="tracking-record-writer",
            daemon=True,
        )
        self.worker.start()
        if not self.ready.wait(settings.close_timeout_s):
            self._fail("Save worker did not initialize before its deadline")
            self.stopping.set()
        self.check_health()

    def _fail(self, message, **details):
        with self.lock:
            if self.error is None:
                self.error, self.error_details = str(message), details
            self.stopping.set()

    def _check_age(self):
        with self.lock:
            oldest = next(iter(self.pending.values()), None)
        if (
            oldest
            and time.perf_counter() - oldest["submitted_s"]
            > self.settings.max_pending_s
        ):
            self._fail("Save backlog exceeded max_pending_s", oldest=oldest)

    def check_health(self):
        self._check_age()
        if self.error:
            raise SaveError(self.error)

    def status(self):
        with self.lock:
            oldest = next(iter(self.pending.values()), None)
            return {
                "submitted_frames": self.submitted,
                "committed_frames": self.committed,
                "pending_records": len(self.pending),
                "peak_pending_records": self.peak_pending,
                "oldest_pending_ms": (time.perf_counter() - oldest["submitted_s"])
                * 1000
                if oldest
                else 0.0,
                "last_committed_source_frame": self.last_committed_frame,
                "error": self.error,
            }

    def poll(self, *, check=True):
        """领取已提交事务的通知；历史始终在数据库中，不依赖调用方保存通知。"""
        updates = []
        while True:
            try:
                updates.append(self.notifications.get_nowait())
            except queue.Empty:
                break
        if check:
            self.check_health()
        return updates

    def _submit(self, kind, payload, source_frame=None):
        self.check_health()
        if self.stopping.is_set():
            raise SaveError("Save worker is closing; no new submissions accepted")
        with self.lock:
            # 健康检查与取得锁之间，保存线程也可能刚好失败。
            # 必须在同一把锁内再次检查，避免线程退出后仍接收一帧。
            if self.error or self.stopping.is_set():
                raise SaveError(self.error or "Save worker is closing")
            full = len(self.pending) >= self.settings.queue_frames
            if not full:
                self.sequence += 1
                item = {
                    "sequence": self.sequence,
                    "kind": kind,
                    "source_frame": source_frame,
                    "submitted_s": time.perf_counter(),
                }
                self.pending[self.sequence] = item
                self.submitted += kind == "frame"
                self.peak_pending = max(self.peak_pending, len(self.pending))
                # 单生产者，且 pending 也计算在途任务；这里不会等待队列空间。
                self.jobs.put_nowait((item, payload))
        if full:
            self._fail(
                "Save queue capacity reached; frame was not accepted",
                rejected_source_frame=source_frame,
            )
            raise SaveError(self.error)

    def submit(self, packet):
        notices = self.poll()
        self._submit("frame", packet, packet[0]["source_frame"])
        return notices

    def reject(self, reason, stamp):
        self._submit("rejection", (reason, repr(stamp)[:2000]))

    def _run(self, metadata, min_observations):
        store = None
        probe = None
        try:
            store = OnlineStore(
                self.path,
                metadata,
                min_observations,
                checkpoint_policy=self.settings.checkpoint_policy,
                wal_limit_mb=self.settings.wal_limit_mb,
            )
            if metadata.get("diagnose_stalls", False):
                if __package__:
                    from .stall_probe import StallProbe
                else:
                    from stall_probe import StallProbe
                probe = StallProbe(self.path.with_name("writer_stalls.csv"))
            with self.path.with_name("writer_timing.csv").open(
                "x", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file,
                    (
                        "source_frame",
                        "submitted_s",
                        "started_s",
                        "committed_s",
                        "queue_ms",
                        "durable_ms",
                        "batch_frames",
                        "database_ms",
                        "pack_records_ms",
                        "statements_ms",
                        "commit_ms",
                        "commit_cpu_ms",
                        "pending_after_commit",
                        "transaction_id",
                        "observations",
                        "events",
                        "contour_bytes",
                        "wal_bytes_after_commit",
                    ),
                )
                writer.writeheader()
                self.ready.set()
                while not (self.stopping.is_set() and self.jobs.empty()):
                    try:
                        first = self.jobs.get(timeout=0.02)
                    except queue.Empty:
                        if not self.stopping.is_set():
                            store.maintain(idle=True)
                        continue
                    batch = [first]
                    # 只合并已到达的数据，不为凑满一批而刻意等待。
                    while len(batch) < self.settings.batch_frames:
                        try:
                            batch.append(self.jobs.get_nowait())
                        except queue.Empty:
                            break
                    self._write_batch(store, batch, writer, diagnostics=probe)
                    # 先确认 FULL 提交并发通知，再单独处理可延期的日志整理。
                    if not self.stopping.is_set():
                        store.maintain(idle=self.jobs.empty())
                    self._check_age()
                file.flush()
        except BaseException as exc:  # noqa: BLE001 -- 线程退出必须向调用方报告，不能静默失败。
            self._fail(f"{type(exc).__name__}: {exc}")
        finally:
            if probe is not None:
                try:
                    probe.close()
                except Exception as exc:  # noqa: BLE001 -- 诊断失败不能跳过数据库关闭与通知。
                    self._fail(f"Closing writer diagnostics failed: {exc}")
            self.ready.set()
            if store is not None:
                finish_started = time.perf_counter()
                try:
                    self.summary = store.finish(
                        "failed" if self.error else self.requested_status
                    )
                    if (
                        self.summary["frames"] != self.committed
                        or self.submitted != self.committed
                    ):
                        self._fail("Submitted and committed frame counts do not match")
                        self.summary = store.finish("failed")
                except BaseException as exc:  # noqa: BLE001 -- 关闭异常也必须保留首次故障。
                    self._fail(f"Final database check failed: {exc}")
                finally:
                    self.finish_ms = (time.perf_counter() - finish_started) * 1000
                    close_started = time.perf_counter()
                    try:
                        store.close()
                    except BaseException as exc:  # noqa: BLE001 -- 通知主线程关闭失败。
                        self._fail(f"Closing database failed: {exc}")
                    finally:
                        self.close_ms = (time.perf_counter() - close_started) * 1000
            self._write_report()
            self.done.set()

    def _write_batch(self, store, batch, writer, *, diagnostics=None):
        # 输入拒绝记录少见；先保持它与前后帧的顺序，再继续合批。
        if any(item["kind"] != "frame" for item, _ in batch):
            for item, payload in batch:
                if item["kind"] == "frame":
                    self._write_batch(
                        store,
                        [(item, payload)],
                        writer,
                        diagnostics=diagnostics,
                    )
                else:
                    store.reject(*payload, serialized=True)
                    with self.lock:
                        del self.pending[item["sequence"]]
            return
        started = time.perf_counter()
        if diagnostics is not None:
            diagnostics.begin(batch[0][0]["source_frame"])
        packets = [payload for _, payload in batch]
        if diagnostics is None:
            results, timing = store.append_batch(packets)
        else:
            results, timing = store.append_batch(packets, diagnostics=diagnostics)
        committed = time.perf_counter()
        if diagnostics is not None:
            diagnostics.finish()
        self._check_age()
        with self.lock:
            for item, _ in batch:
                del self.pending[item["sequence"]]
                self.committed += 1
                self.last_committed_frame = item["source_frame"]
            pending = len(self.pending)
        detail = {
            "transaction_id": batch[0][0]["sequence"],
            "observations": sum(len(packet[1]["rows"]) for packet in packets),
            "events": sum(len(packet[1]["events"]) for packet in packets),
            "contour_bytes": sum(
                len(contour)
                for packet in packets
                for _, contour in packet[1]["contours"]
            ),
            "wal_bytes_after_commit": store.wal_bytes(),
        }
        for (item, _), (_, updates) in zip(batch, results):
            writer.writerow(
                dict(
                    source_frame=item["source_frame"],
                    submitted_s=item["submitted_s"],
                    started_s=started,
                    committed_s=committed,
                    queue_ms=(started - item["submitted_s"]) * 1000,
                    durable_ms=(committed - item["submitted_s"]) * 1000,
                    batch_frames=len(batch),
                    pending_after_commit=pending,
                    **timing,
                    **detail,
                )
            )
            notice = dict(
                source_frame=item["source_frame"], committed_s=committed, **updates
            )
            # 通知也有界；提交和正常关闭都会领取，防止慢调用方导致历史常驻内存。
            while True:
                try:
                    self.notifications.put(notice, timeout=0.02)
                    break
                except queue.Full:
                    self._check_age()

    def _write_report(self, *, timed_out=False):
        with self.lock:
            missing = [dict(item) for item in self.pending.values()]
        report = dict(
            self.status(),
            settings=asdict(self.settings),
            error_details=self.error_details,
            uncommitted=missing,
            all_submitted_saved=self.submitted == self.committed,
            worker_finished=self.worker is threading.current_thread(),
            summary=self.summary,
            finalize_ms=self.finish_ms,
            database_close_ms=self.close_ms,
        )
        try:
            # 超时报告由分析线程写，最终报告只由保存线程写；避免同时截断同一文件。
            name = "writer_timeout.json" if timed_out else "writer_report.json"
            self.path.with_name(name).write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            self._fail(f"Could not save writer report: {exc}")

    def finish(self, status):
        """停止接收、排空、核对；失败或超时不能作为正常完成返回。"""
        self.requested_status = status
        self.stopping.set()
        deadline = time.perf_counter() + self.settings.close_timeout_s
        while not self.done.wait(0.01):
            self.poll(check=False)
            if time.perf_counter() >= deadline:
                self._fail("Timed out draining save queue; records remain incomplete")
                self._write_report(timed_out=True)
                raise SaveError(self.error)
        self.poll(check=False)
        self.worker.join()
        if self.error:
            raise SaveError(self.error)
        return dict(self.summary, persistence=self.status())

    def close(self):
        # 正常生命周期统一由 finish 排空及关闭；不从分析线程触碰 SQLite。
        pass
