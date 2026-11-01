"""保存线程的独立快照、合批事务、积压、磁盘错误及停止排空验证。"""

import csv
import gc
import json
import sqlite3
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from customer_workspace.microbubble_tracking.async_store import SaveError, SaveSettings
from customer_workspace.microbubble_tracking.online_store import OnlineStore
from customer_workspace.microbubble_tracking.tests.test_online_tracking import (
    image,
    processor,
    rows,
    stamp,
)


def async_processor(path, **settings):
    return processor(path, save_mode="async", save_settings=SaveSettings(**settings))


def block_first_write(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = OnlineStore.append_batch

    def blocked(store, packets):
        if not entered.is_set():
            entered.set()
            assert release.wait(5), "test must release writer"
        return original(store, packets)

    monkeypatch.setattr(OnlineStore, "append_batch", blocked)
    return entered, release


def test_returns_before_commit_and_copies_mutable_results(tmp_path, monkeypatch):
    entered, release = block_first_write(monkeypatch)
    p = async_processor(tmp_path)
    result = p.process(image((20, 20, 12, 12)), stamp(274, 0.0))
    assert entered.wait(1)
    assert result["persistence"]["submitted_frames"] == 1
    assert not rows(p.output, "SELECT * FROM frames")
    expected_contour = result["contours"][0][1].copy()
    expected_x = result["observations"][0]["cx"]
    result["contours"][0][1][:] = 0
    result["observations"][0]["cx"] = -123
    p.tracker.tracks[1].cx = -999
    release.set()
    summary = p.close()
    observed = rows(p.output, "SELECT * FROM observations")[0]
    assert observed["cx"] == expected_x
    assert observed["contour_i32"] == expected_contour.astype("<i4").tobytes()
    assert summary["frames"] == 1
    assert summary["persistence"]["pending_records"] == 0
    assert not p.store.worker.is_alive()


def test_async_matches_sync_with_gaps_events_and_delayed_confirmation(
    tmp_path, monkeypatch
):
    entered, release = block_first_write(monkeypatch)
    async_p = async_processor(tmp_path / "async", queue_frames=64, batch_frames=32)
    sync_p = processor(tmp_path / "sync")
    whole = image((20, 25, 40, 14))
    split = image((20, 25, 14, 14), (46, 25, 14, 14))
    expected_events = []
    for index, frame in enumerate([whole] * 5 + [split] * 5 + [whole, image(), whole]):
        number = index if index < 11 else index + 10
        expected = sync_p.process(frame, stamp(number))
        actual = async_p.process(frame, stamp(number))
        assert actual["observations"] == expected["observations"]
        expected_events.extend(expected["events"])
    assert entered.wait(1)
    with pytest.raises(ValueError):
        async_p.process(image(), stamp(0))
    with pytest.raises(ValueError):
        sync_p.process(image(), stamp(0))
    release.set()
    notices = []
    deadline = time.perf_counter() + 3
    while async_p.store.status()["committed_frames"] < 13:
        notices.extend(async_p.poll_saved())
        assert time.perf_counter() < deadline
        time.sleep(0.001)
    # 通知可能正在排入：关闭会领取末尾通知，所以先等待 worker 已发完本批。
    while sum(len(n["events"]) for n in notices) < len(expected_events):
        notices.extend(async_p.poll_saved())
        assert time.perf_counter() < deadline
        time.sleep(0.001)
    assert [e for n in notices for e in n["events"]] == expected_events
    async_p.close()
    sync_p.close()
    for table in (
        "frames",
        "observations",
        "tracks",
        "events",
        "event_updates",
        "track_updates",
        "input_rejections",
    ):
        left, right = (
            rows(sync_p.output, f"SELECT * FROM {table}"),
            rows(async_p.output, f"SELECT * FROM {table}"),
        )
        for values in (left, right):
            for row in values:
                row.pop("processing_ms", None)
        assert left == right, table
    assert np.array_equal(sync_p.background.B, async_p.background.B)


def test_full_queue_reports_rejected_frame_and_drains_accepted(tmp_path, monkeypatch):
    entered, release = block_first_write(monkeypatch)
    p = async_processor(tmp_path, queue_frames=2)
    p.process(image(), stamp(0))
    assert entered.wait(1)
    p.process(image(), stamp(1))
    with pytest.raises(SaveError, match="capacity"):
        p.process(image(), stamp(2))
    release.set()
    with pytest.raises(SaveError):
        p.close()
    report = json.loads((p.output / "writer_report.json").read_text())
    assert report["submitted_frames"] == report["committed_frames"] == 2
    assert report["error_details"]["rejected_source_frame"] == 2
    assert report["summary"]["status"] == "failed"
    assert not (p.output / "background_final.npy").exists()


def test_old_pending_record_reports_error_even_before_queue_fills(
    tmp_path, monkeypatch
):
    entered, release = block_first_write(monkeypatch)
    p = async_processor(tmp_path, max_pending_s=0.03)
    p.process(image(), stamp(0))
    assert entered.wait(1)
    time.sleep(0.04)
    with pytest.raises(SaveError, match="backlog"):
        p.poll_saved()
    release.set()
    with pytest.raises(SaveError):
        p.close()
    assert len(rows(p.output, "SELECT * FROM frames")) == 1


def test_failed_batch_rolls_back_and_names_uncommitted_frames(tmp_path, monkeypatch):
    entered, release = block_first_write(monkeypatch)
    original = OnlineStore.append

    def fail_on_two(store, frame, *args, **kwargs):
        result = original(store, frame, *args, **kwargs)
        if frame["source_frame"] == 2:
            raise sqlite3.OperationalError("injected disk failure")
        return result

    monkeypatch.setattr(OnlineStore, "append", fail_on_two)
    p = async_processor(tmp_path, batch_frames=4)
    p.process(image(), stamp(0))
    assert entered.wait(1)
    for number in (1, 2, 3, 4):
        p.process(image(), stamp(number))
    release.set()
    with pytest.raises(SaveError, match="injected disk failure"):
        p.close()
    report = json.loads((p.output / "writer_report.json").read_text())
    assert [r["source_frame"] for r in report["uncommitted"]] == [1, 2, 3, 4]
    assert [r["source_frame"] for r in rows(p.output, "SELECT * FROM frames")] == [0]
    assert report["summary"]["status"] == "failed"
    assert not report["all_submitted_saved"]


def test_cancel_drains_and_does_not_create_disappearance(tmp_path, monkeypatch):
    entered, release = block_first_write(monkeypatch)
    p = async_processor(tmp_path)
    for number in range(5):
        p.process(image((20, 20, 12, 12)), stamp(number))
    assert entered.wait(1)
    release.set()
    summary = p.close("cancelled")
    assert summary["frames"] == 5 and summary["status"] == "cancelled"
    assert (
        rows(p.output, "SELECT end_reason FROM tracks")[0]["end_reason"]
        == "stream_cancelled"
    )
    assert [r["kind"] for r in rows(p.output, "SELECT kind FROM events")] == ["appear"]


def test_close_timeout_is_explicit_and_does_not_claim_completed(tmp_path, monkeypatch):
    entered, release = block_first_write(monkeypatch)
    p = async_processor(tmp_path)
    p.process(image(), stamp(0))
    assert entered.wait(1)
    # 只缩短本次关闭的等待期限，避免建库耗时先触发另一个启动超时。
    p.store.settings = replace(p.store.settings, close_timeout_s=0.1)
    try:
        with pytest.raises(SaveError, match="Timed out"):
            p.close()
        report = json.loads((p.output / "writer_timeout.json").read_text())
        assert not report["worker_finished"] and not report["all_submitted_saved"]
    finally:
        release.set()
        assert p.store.done.wait(2)
    assert not (p.output / "background_final.npy").exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"queue_frames": 0},
        {"batch_frames": 1.5},
        {"max_pending_s": float("nan")},
        {"close_timeout_s": 0},
    ],
)
def test_invalid_save_settings(settings):
    with pytest.raises(ValueError):
        SaveSettings(**settings)


def test_writer_diagnostics_use_writer_thread_and_preserve_results(tmp_path):
    callbacks = list(gc.callbacks)
    measured = processor(tmp_path / "measured", save_mode="async", diagnose_stalls=True)
    plain = processor(tmp_path / "plain", save_mode="async")
    with measured, plain:
        for number in range(8):
            frame = image((20 + number, 20, 12, 12))
            measured.process(frame, stamp(number))
            plain.process(frame, stamp(number))
    assert gc.callbacks == callbacks
    meta = json.loads((measured.output / "writer_stalls.meta.json").read_text())
    assert meta["native_thread_id"] == measured.store.worker.native_id
    assert meta["native_thread_id"] != threading.get_native_id()
    with (measured.output / "writer_stalls.csv").open() as file:
        log = list(csv.DictReader(file))
    assert {r["stage"] for r in log} == {
        "database_statements", "database_commit", "tail"
    }
    assert all(float(r["wall_ms"]) >= 0 for r in log)
    for table in ("frames", "observations", "tracks", "events", "track_updates",
                  "event_updates", "input_rejections"):
        expected = rows(plain.output, f"SELECT * FROM {table}")
        actual = rows(measured.output, f"SELECT * FROM {table}")
        for values in (expected, actual):
            for row in values:
                row.pop("processing_ms", None)
        assert actual == expected, table
    report = json.loads((measured.output / "writer_report.json").read_text())
    assert report["all_submitted_saved"]
    assert report["submitted_frames"] == report["committed_frames"] == 8


def test_diagnostic_close_failure_still_closes_database_and_worker(tmp_path, monkeypatch):
    from customer_workspace.microbubble_tracking.stall_probe import StallProbe

    original = StallProbe.close

    def fail_close(probe):
        original(probe)
        raise OSError("injected diagnostic write failure")

    monkeypatch.setattr(StallProbe, "close", fail_close)
    p = processor(tmp_path, save_mode="async", diagnose_stalls=True)
    p.process(image(), stamp(0))
    with pytest.raises(SaveError, match="diagnostic write failure"):
        p.close()
    assert p.store.done.is_set() and not p.store.worker.is_alive()
    assert len(rows(p.output, "SELECT * FROM frames")) == 1


def test_failure_between_health_check_and_enqueue_cannot_accept_frame(tmp_path):
    """人为固定这次交错：健康检查刚结束，保存线程在入队锁之前宣布失败。"""
    p = async_processor(tmp_path)
    original_lock = p.store.lock

    class FailAtEnqueueLock:
        def __enter__(self):
            original_lock.acquire()
            p.store.error = "injected concurrent worker failure"
            p.store.stopping.set()

        def __exit__(self, *args):
            original_lock.release()

    p.store.check_health = lambda: None
    p.store.lock = FailAtEnqueueLock()
    try:
        with pytest.raises(SaveError, match="concurrent worker failure"):
            p.store._submit("frame", None, 7)
    finally:
        p.store.lock = original_lock
    assert p.store.submitted == 0 and p.store.jobs.empty()
    with pytest.raises(SaveError):
        p.close()
