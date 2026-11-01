"""逐帧事务日志：历史轮廓、临时编号和事件修订留在磁盘，不留在活跃内存。"""

import csv
import json
import os
import sqlite3
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = (
    "cx",
    "cy",
    "area",
    "eq_diam",
    "major",
    "minor",
    "angle",
    "circularity",
    "solidity",
)


@dataclass(frozen=True)
class TrackWriteState:
    """保存所需的少量目标状态，不把图像或活跃像素数组交给保存线程。"""

    id: int
    first: int
    last: int
    observations: int


def annotate_observation(row, track, old, source_time, min_observations):
    """同步和异步共用同一观测整理规则。"""
    dt = source_time - old["time_s"] if old is not None else None
    row.update(
        status_at_frame="confirmed"
        if track.observations >= min_observations
        else "tentative",
        velocity_valid=old is not None,
        dt_s=dt,
        raw_vx_px_s=(row["cx"] - old["cx"]) / dt if old else None,
        raw_vy_px_s=(row["cy"] - old["cy"]) / dt if old else None,
    )


def snapshot_frame(frame, result, live, previous, min_observations):
    """生成与分析状态脱离的快照；以后处理新帧、修改返回值都不会改动待存数据。"""
    tracks = {}
    for row in result["rows"]:
        track = live[row["id"]]
        annotate_observation(
            row, track, previous.get(track.id), frame["source_time_s"], min_observations
        )
        tracks[track.id] = TrackWriteState(
            track.id, track.first, track.last, track.observations
        )
    saved = {
        "rows": [dict(row) for row in result["rows"]],
        "events": [dict(event, ids=list(event["ids"])) for event in result["events"]],
        "contours": [
            (track_id, np.asarray(contour, dtype="<i4").reshape(-1, 2).tobytes())
            for track_id, contour in result["contours"]
        ],
        "ended": [
            TrackWriteState(t.id, t.first, t.last, t.observations)
            for t in result["ended"]
        ],
    }
    return (
        dict(frame),
        saved,
        tracks,
        {key: dict(value) for key, value in previous.items()},
    )


class OnlineStore:
    """每次返回前提交一帧；SQLite 失败会向调用者报错，不静默丢记录。"""

    def __init__(
        self,
        path: Path,
        metadata: dict,
        min_observations: int,
        *,
        checkpoint_policy="auto",
        wal_limit_mb=64,
    ):
        if checkpoint_policy not in ("auto", "deferred"):
            raise ValueError("checkpoint_policy must be auto or deferred")
        if type(wal_limit_mb) is not int or wal_limit_mb < 1:
            raise ValueError("wal_limit_mb must be a positive integer")
        self.path = Path(path).resolve()
        self.checkpoint_policy = checkpoint_policy
        self.wal_limit_bytes = wal_limit_mb * 1024**2
        self.last_commit_s = None
        self.needs_checkpoint = False
        self.frames_since_checkpoint = 0
        self.min_observations = min_observations
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA cache_size=-2048")
        self.db.execute(
            "PRAGMA wal_autocheckpoint="
            + ("256" if checkpoint_policy == "auto" else "0")
        )
        self.db.executescript("""
            CREATE TABLE session (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE frames (
                source_frame INTEGER PRIMARY KEY, video_frame INTEGER UNIQUE,
                source_time_s REAL NOT NULL, video_time_s REAL,
                received_windows_s REAL, decoded_windows_s REAL,
                gap_frames INTEGER NOT NULL, delta_time_s REAL,
                processing_ms REAL NOT NULL, active_tracks INTEGER NOT NULL
            );
            CREATE TABLE tracks (
                track_id INTEGER PRIMARY KEY, first_frame INTEGER NOT NULL,
                last_frame INTEGER NOT NULL, observations INTEGER NOT NULL,
                confirmed_frame INTEGER, ended_frame INTEGER, end_reason TEXT,
                FOREIGN KEY(first_frame) REFERENCES frames(source_frame),
                FOREIGN KEY(last_frame) REFERENCES frames(source_frame)
            );
            CREATE TABLE observations (
                source_frame INTEGER NOT NULL, track_id INTEGER NOT NULL,
                status_at_frame TEXT NOT NULL, velocity_valid INTEGER NOT NULL,
                dt_s REAL, raw_vx_px_s REAL, raw_vy_px_s REAL,
                vx_px_s REAL NOT NULL, vy_px_s REAL NOT NULL,
                cx REAL, cy REAL, area INTEGER, eq_diam REAL, major REAL,
                minor REAL, angle REAL, circularity REAL, solidity REAL,
                contour_i32 BLOB NOT NULL,
                PRIMARY KEY(source_frame, track_id),
                FOREIGN KEY(source_frame) REFERENCES frames(source_frame),
                FOREIGN KEY(track_id) REFERENCES tracks(track_id)
            );
            CREATE TABLE track_updates (
                sequence INTEGER PRIMARY KEY, track_id INTEGER NOT NULL,
                reported_frame INTEGER NOT NULL, state TEXT NOT NULL,
                FOREIGN KEY(track_id) REFERENCES tracks(track_id)
            );
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY, kind TEXT NOT NULL,
                anchor_frame INTEGER NOT NULL, reported_frame INTEGER NOT NULL,
                interval_from_s REAL NOT NULL, interval_to_s REAL NOT NULL,
                participants TEXT NOT NULL, result_track INTEGER NOT NULL,
                x REAL, y REAL, evidence_gap INTEGER NOT NULL,
                status TEXT NOT NULL, decided_frame INTEGER,
                FOREIGN KEY(anchor_frame) REFERENCES frames(source_frame),
                FOREIGN KEY(reported_frame) REFERENCES frames(source_frame)
            );
            CREATE INDEX pending_events ON events(status);
            CREATE TABLE event_updates (
                sequence INTEGER PRIMARY KEY, event_id INTEGER NOT NULL,
                reported_frame INTEGER NOT NULL, status TEXT NOT NULL,
                FOREIGN KEY(event_id) REFERENCES events(event_id)
            );
            CREATE TABLE input_rejections (
                sequence INTEGER PRIMARY KEY, reason TEXT NOT NULL,
                supplied_stamp TEXT NOT NULL
            );
            CREATE INDEX track_update_frame ON track_updates(reported_frame);
            CREATE INDEX event_update_frame ON event_updates(reported_frame);
        """)
        with self.db:
            self.db.executemany(
                "INSERT INTO session VALUES (?, ?)",
                [
                    (k, json.dumps(v, ensure_ascii=False, allow_nan=False))
                    for k, v in metadata.items()
                ],
            )
        # 必须在实际写入连接上读设置；另开只读连接会得到它自己的同步默认值。
        settings = {
            key: self.db.execute(f"PRAGMA {key}").fetchone()[0]
            for key in (
                "journal_mode",
                "synchronous",
                "wal_autocheckpoint",
                "page_size",
                "cache_size",
                "auto_vacuum",
                "busy_timeout",
                "foreign_keys",
            )
        }
        settings.update(
            database=str(self.path),
            sqlite_version=sqlite3.sqlite_version,
            device_id=os.stat(self.path).st_dev,
            pid=os.getpid(),
            writer_thread_id=threading.get_native_id(),
            checkpoint_policy=checkpoint_policy,
            wal_limit_bytes=self.wal_limit_bytes,
            checkpoint_interval_frames=128,
        )
        try:
            self.path.with_name("database_settings.json").write_text(
                json.dumps(settings, indent=2) + "\n", encoding="utf-8"
            )
            self.maintenance_file = self.path.with_name(
                "database_maintenance.csv"
            ).open(
                "x",
                encoding="utf-8",
                newline="",
                buffering=1,
            )
        except BaseException:
            self.db.close()
            raise
        self.maintenance_writer = csv.DictWriter(
            self.maintenance_file,
            (
                "operation",
                "reason",
                "started_s",
                "finished_s",
                "wall_ms",
                "cpu_ms",
                "wal_before_bytes",
                "wal_after_bytes",
                "busy",
                "log_pages",
                "copied_pages",
                "error",
            ),
        )
        self.maintenance_writer.writeheader()

    def wal_bytes(self):
        """当前日志文件占用；仅作空间门限，不把它冒充新增写入字节数。"""
        try:
            return self.path.with_name(self.path.name + "-wal").stat().st_size
        except FileNotFoundError:
            return 0

    def checkpoint(self, reason, *, mode="TRUNCATE"):
        """单独测量日志写回；不等待外部读者，不删除未完成的日志。"""
        if mode not in ("PASSIVE", "TRUNCATE"):
            raise ValueError("Unsupported checkpoint mode")
        before = self.wal_bytes()
        started, cpu = time.perf_counter(), time.thread_time()
        result, error = (-1, -1, -1), ""
        timeout = self.db.execute("PRAGMA busy_timeout").fetchone()[0]
        try:
            self.db.execute("PRAGMA busy_timeout=0")
            result = tuple(
                self.db.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
            )
            if result[0]:
                raise RuntimeError(
                    "Checkpoint blocked by a reader; committed data remain in WAL"
                )
            self.needs_checkpoint = result[1] != result[2]
            self.frames_since_checkpoint = 0
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            finished, cpu_finished = time.perf_counter(), time.thread_time()
            self.db.execute(f"PRAGMA busy_timeout={timeout}")
            self.maintenance_writer.writerow(
                {
                    "operation": "checkpoint_" + mode.lower(),
                    "reason": reason,
                    "started_s": started,
                    "finished_s": finished,
                    "wall_ms": (finished - started) * 1000,
                    "cpu_ms": (cpu_finished - cpu) * 1000,
                    "wal_before_bytes": before,
                    "wal_after_bytes": self.wal_bytes(),
                    "busy": result[0],
                    "log_pages": result[1],
                    "copied_pages": result[2],
                    "error": error,
                }
            )

    def maintain(self, *, idle=False):
        """连续输入时延后整理；超过日志门限必须整理，不能无限增长。"""
        if self.checkpoint_policy != "deferred" or not self.needs_checkpoint:
            return
        if self.wal_bytes() >= self.wal_limit_bytes:
            self.checkpoint("wal_size_limit")
        elif idle and self.frames_since_checkpoint:
            if time.perf_counter() - self.last_commit_s >= 0.1:
                self.checkpoint("idle_100ms", mode="PASSIVE")
            elif self.frames_since_checkpoint >= 128:
                # 保留已分配的日志空间，下一轮可循环使用，避免反复扩展文件。
                self.checkpoint("interval_128_frames", mode="PASSIVE")

    def reject(self, reason: str, stamp, *, serialized=False) -> None:
        """无效输入不会推进追踪状态，但保留拒绝原因。"""
        with self.db:
            self.db.execute(
                "INSERT INTO input_rejections(reason,supplied_stamp) VALUES (?,?)",
                (reason, stamp if serialized else repr(stamp)[:2000]),
            )

    def _track(self, track, frame: int) -> None:
        previous = self.db.execute(
            "SELECT confirmed_frame FROM tracks WHERE track_id=?", (track.id,)
        ).fetchone()
        confirmed = frame if track.observations >= self.min_observations else None
        self.db.execute(
            """
            INSERT INTO tracks(track_id,first_frame,last_frame,observations,confirmed_frame)
            VALUES (?,?,?,?,?) ON CONFLICT(track_id) DO UPDATE SET
            last_frame=excluded.last_frame, observations=excluded.observations,
            confirmed_frame=COALESCE(tracks.confirmed_frame,excluded.confirmed_frame)
        """,
            (track.id, track.first, track.last, track.observations, confirmed),
        )
        if previous is None:
            self.db.execute(
                "INSERT INTO track_updates(track_id,reported_frame,state) VALUES (?,?,?)",
                (track.id, frame, "tentative"),
            )
        if confirmed is not None and (previous is None or previous[0] is None):
            self.db.execute(
                "INSERT INTO track_updates(track_id,reported_frame,state) VALUES (?,?,?)",
                (track.id, frame, "confirmed"),
            )

    def _end_track(self, track_id: int, frame: int, reason: str) -> None:
        self.db.execute(
            "UPDATE tracks SET ended_frame=?,end_reason=? WHERE track_id=?",
            (frame, reason, track_id),
        )
        self.db.execute(
            "INSERT INTO track_updates(track_id,reported_frame,state) VALUES (?,?,?)",
            (track_id, frame, reason),
        )

    def _resolve_events(self, frame: int, finishing=False) -> None:
        # 只读仍待确认的事件；已完成的历史事件不重新载入内存。
        pending = self.db.execute(
            "SELECT event_id,participants FROM events WHERE status='pending'"
        ).fetchall()
        for event in pending:
            participants = [
                self.db.execute(
                    "SELECT confirmed_frame,end_reason FROM tracks WHERE track_id=?",
                    (i,),
                ).fetchone()
                for i in json.loads(event["participants"])
            ]
            if all(p["confirmed_frame"] is not None for p in participants):
                status = "confirmed_tracks"
            elif any(
                p["confirmed_frame"] is None
                and p["end_reason"] in ("merge", "lost", "exit")
                for p in participants
            ):
                status = "rejected_short_track"
            elif finishing:
                status = "unresolved_at_end"
            else:
                continue
            self.db.execute(
                "UPDATE events SET status=?,decided_frame=? WHERE event_id=?",
                (status, frame, event["event_id"]),
            )
            self.db.execute(
                "INSERT INTO event_updates(event_id,reported_frame,status) VALUES (?,?,?)",
                (event["event_id"], frame, status),
            )

    def append(
        self, frame: dict, result: dict, live: dict, previous: dict, *, commit=True
    ) -> dict:
        """同一事务保存整帧，并分开测量观测打包、数据库操作与提交等待。"""
        started = time.perf_counter()
        source_frame = frame["source_frame"]
        source_time = frame["source_time_s"]
        contours = dict(result["contours"])
        packing_s = time.perf_counter() - started
        with self.db if commit else nullcontext():
            self.db.execute(
                "INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?)",
                tuple(
                    frame[k]
                    for k in (
                        "source_frame",
                        "video_frame",
                        "source_time_s",
                        "video_time_s",
                        "received_windows_s",
                        "decoded_windows_s",
                        "gap_frames",
                        "delta_time_s",
                        "processing_ms",
                        "active_tracks",
                    )
                ),
            )
            for row in result["rows"]:
                track = live[row["id"]]
                self._track(track, source_frame)
                packing_started = time.perf_counter()
                old = previous.get(track.id)
                annotate_observation(
                    row, track, old, source_time, self.min_observations
                )
                values = [
                    source_frame,
                    track.id,
                    row["status_at_frame"],
                    int(old is not None),
                    row["dt_s"],
                    row["raw_vx_px_s"],
                    row["raw_vy_px_s"],
                    row["vx"],
                    row["vy"],
                ]
                values.extend(
                    float(row[k]) if k != "area" else int(row[k]) for k in FEATURES
                )
                contour = contours[track.id]
                values.append(
                    contour
                    if isinstance(contour, bytes)
                    else np.asarray(contour, dtype="<i4").reshape(-1, 2).tobytes()
                )
                packing_s += time.perf_counter() - packing_started
                self.db.execute(
                    "INSERT INTO observations VALUES ("
                    + ",".join("?" for _ in values)
                    + ")",
                    values,
                )

            reasons = {}
            for event in result["events"]:
                if event["type"] in ("merge", "lost", "exit"):
                    reasons[event["ids"][0]] = event["type"]
            for track in result["ended"]:
                self._end_track(track.id, source_frame, reasons[track.id])

            for event in result["events"]:
                earlier = [previous[i]["time_s"] for i in event["ids"] if i in previous]
                lower = (
                    min(earlier)
                    if earlier
                    else source_time - (frame["delta_time_s"] or 0)
                )
                evidence_gap = frame["gap_frames"] > 0 or any(
                    source_frame - previous[i]["frame"] > 1
                    for i in event["ids"]
                    if i in previous
                )
                cursor = self.db.execute(
                    """
                    INSERT INTO events(kind,anchor_frame,reported_frame,interval_from_s,
                        interval_to_s,participants,result_track,x,y,evidence_gap,status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                    (
                        event["type"],
                        event["frame"],
                        source_frame,
                        lower,
                        source_time,
                        json.dumps(event["ids"]),
                        event["result"],
                        float(event["x"]),
                        float(event["y"]),
                        int(evidence_gap),
                        "pending",
                    ),
                )
                self.db.execute(
                    "INSERT INTO event_updates(event_id,reported_frame,status) VALUES (?,?,?)",
                    (cursor.lastrowid, source_frame, "pending"),
                )
            self._resolve_events(source_frame)
            before_commit = time.perf_counter()
            cpu_before_commit = time.thread_time()
        # 仍由原来的 with self.db 完成提交；包括 FULL 同步和可能的 WAL 检查点。
        # 数据库操作计时也包含编号/事件整理，不能解释为纯硬盘写入时间。
        timing = {
            "pack_records_ms": packing_s * 1000,
            "database_write_ms": (before_commit - started - packing_s) * 1000,
            "commit_ms": (time.perf_counter() - before_commit) * 1000,
            "commit_cpu_ms": (time.thread_time() - cpu_before_commit) * 1000,
        }
        if commit:
            self.last_commit_s = time.perf_counter()
            self.needs_checkpoint = True
            self.frames_since_checkpoint += 1
            self.maintain(idle=True)
        return timing

    def append_batch(self, packets, *, diagnostics=None):
        """复用逐帧写入，至多一批一次 FULL 提交；提交前不对外确认成功。

        每帧状态更新在写入当帧后立即复制，避免同批后面的帧改变早期事件状态。
        任一帧失败时，整个批次由同一事务回滚。
        """
        results = []
        started = time.perf_counter()
        with self.db:
            for packet in packets:
                timing = self.append(*packet, commit=False)
                updates = self.updates(packet[0]["source_frame"])
                results.append((timing, updates))
            before_commit = time.perf_counter()
            cpu_before_commit = time.thread_time()
            if diagnostics is not None:
                diagnostics.mark("database_statements")
        committed, cpu_committed = time.perf_counter(), time.thread_time()
        self.last_commit_s = committed
        self.needs_checkpoint = True
        self.frames_since_checkpoint += len(packets)
        if diagnostics is not None:
            diagnostics.mark("database_commit")
        packing_ms = sum(timing["pack_records_ms"] for timing, _ in results)
        return results, {
            "database_ms": (before_commit - started) * 1000,
            "pack_records_ms": packing_ms,
            "statements_ms": (before_commit - started) * 1000 - packing_ms,
            "commit_ms": (committed - before_commit) * 1000,
            "commit_cpu_ms": (cpu_committed - cpu_before_commit) * 1000,
        }

    def finish(self, status: str) -> dict:
        """录像结束不伪造 lost；未充分观察的目标和事件保留未决状态。"""
        # 以最后提交成功的帧为准；提交后的其他操作也可能抛出异常。
        frame = self.db.execute("SELECT MAX(source_frame) FROM frames").fetchone()[0]
        with self.db:
            if frame is not None:
                active = self.db.execute(
                    "SELECT track_id FROM tracks WHERE ended_frame IS NULL"
                ).fetchall()
                for row in active:
                    self._end_track(row[0], frame, "stream_" + status)
                self._resolve_events(frame, finishing=True)
            self.db.execute(
                "INSERT OR REPLACE INTO session VALUES ('status',?)",
                (json.dumps(status),),
            )
        return {
            "status": status,
            "frames": self.db.execute("SELECT count(*) FROM frames").fetchone()[0],
            "observations": self.db.execute(
                "SELECT count(*) FROM observations"
            ).fetchone()[0],
            "tracks": self.db.execute("SELECT count(*) FROM tracks").fetchone()[0],
            "confirmed_tracks": self.db.execute(
                "SELECT count(*) FROM tracks WHERE confirmed_frame IS NOT NULL"
            ).fetchone()[0],
            "events_by_status": dict(
                self.db.execute("SELECT status,count(*) FROM events GROUP BY status")
            ),
            "input_rejections": self.db.execute(
                "SELECT count(*) FROM input_rejections"
            ).fetchone()[0],
            "skipped_source_frames": self.db.execute(
                "SELECT COALESCE(SUM(gap_frames),0) FROM frames"
            ).fetchone()[0],
            "peak_active_tracks": self.db.execute(
                "SELECT COALESCE(MAX(active_tracks),0) FROM frames"
            ).fetchone()[0],
        }

    def updates(self, frame: int) -> dict:
        """同时返回首次发现和延迟确认；明确区分事件画面与报告画面。"""
        events = self.db.execute(
            """
            SELECT u.sequence,u.reported_frame AS update_frame,u.status AS update_status,e.*
            FROM event_updates u
            JOIN events e USING(event_id) WHERE u.reported_frame=? ORDER BY u.sequence
        """,
            (frame,),
        )
        tracks = self.db.execute(
            "SELECT * FROM track_updates WHERE reported_frame=? ORDER BY sequence",
            (frame,),
        )
        return dict(
            events=[dict(row) for row in events],
            track_updates=[dict(row) for row in tracks],
        )

    def close(self) -> None:
        try:
            self.checkpoint("close")
        finally:
            try:
                self.db.close()
            finally:
                self.maintenance_file.close()


def export_tables(database: Path, output: Path) -> None:
    """流式导出可读 CSV；轮廓继续原样保存在数据库中，不一次载入历史。"""
    output.mkdir(exist_ok=False)
    db = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        metadata = {
            k: json.loads(v) for k, v in db.execute("SELECT key,value FROM session")
        }
        identity = [metadata[k] for k in ("experiment_id", "stream_id", "run_id")]
        for table in (
            "frames",
            "tracks",
            "events",
            "event_updates",
            "track_updates",
            "input_rejections",
            "observations",
        ):
            columns = [
                r[1]
                for r in db.execute(f"PRAGMA table_info({table})")
                if r[1] != "contour_i32"
            ]
            with (output / (table + ".csv")).open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(["experiment_id", "stream_id", "run_id", *columns])
                for row in db.execute(f"SELECT {','.join(columns)} FROM {table}"):
                    writer.writerow([*identity, *row])
    finally:
        db.close()
