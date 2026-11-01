"""安全同步不降级；日志整理时机、完整性和异常退出后的已确认记录。"""

import csv
import json
import multiprocessing as mp
import os
import sqlite3
import threading
import time

import pytest

from customer_workspace.microbubble_tracking.async_store import SaveSettings
from customer_workspace.microbubble_tracking.online_store import OnlineStore
from customer_workspace.microbubble_tracking.tests.test_online_tracking import (
    image,
    processor,
    stamp,
)


def make_store(path, policy="deferred", limit=1):
    path.mkdir()
    return OnlineStore(
        path / "tracking.sqlite", {}, 5, checkpoint_policy=policy, wal_limit_mb=limit
    )


@pytest.mark.parametrize("policy,expected", [("auto", 256), ("deferred", 0)])
def test_actual_connection_keeps_wal_full_and_reports_settings(
    tmp_path, policy, expected
):
    store = make_store(tmp_path / "run", policy)
    try:
        assert store.db.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert store.db.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == expected
        saved = json.loads((tmp_path / "run/database_settings.json").read_text())
        assert saved["synchronous"] == 2
        assert saved["writer_thread_id"] == threading.get_native_id()
        assert saved["database"] == str((tmp_path / "run/tracking.sqlite").resolve())
    finally:
        store.close()


def test_size_limit_compacts_only_committed_wal_and_preserves_data(tmp_path):
    store = make_store(tmp_path / "run")
    with store.db:
        store.db.execute("INSERT INTO session VALUES ('large',zeroblob(1200000))")
    store.needs_checkpoint = True
    store.last_commit_s = time.perf_counter()
    assert store.wal_bytes() > 1024**2
    store.maintain()
    assert store.wal_bytes() == 0
    assert (
        store.db.execute(
            "SELECT length(value) FROM session WHERE key='large'"
        ).fetchone()[0]
        == 1200000
    )
    store.close()
    with (tmp_path / "run/database_maintenance.csv").open() as file:
        log = list(csv.DictReader(file))
    assert log[0]["reason"] == "wal_size_limit"
    assert log[0]["busy"] == "0"


def test_idle_does_not_repeatedly_checkpoint_clean_database(tmp_path):
    store = make_store(tmp_path / "run")
    store.last_commit_s = time.perf_counter() - 1
    store.needs_checkpoint = True
    store.frames_since_checkpoint = 1
    store.maintain(idle=True)
    store.maintain(idle=True)
    store.close()
    with (tmp_path / "run/database_maintenance.csv").open() as file:
        log = list(csv.DictReader(file))
    assert [r["reason"] for r in log] == ["idle_100ms", "close"]


def test_reader_blocking_checkpoint_keeps_wal_and_restores_timeout(tmp_path):
    store = make_store(tmp_path / "run")
    reader = sqlite3.connect(store.path)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM session").fetchone()
        with store.db:
            store.db.execute("INSERT INTO session VALUES ('durable','yes')")
        with pytest.raises(RuntimeError, match="committed data remain in WAL"):
            store.checkpoint("test_reader")
        assert store.wal_bytes() > 0
        assert store.db.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        reader.rollback()
        assert (
            reader.execute("SELECT value FROM session WHERE key='durable'").fetchone()[
                0
            ]
            == "yes"
        )
    finally:
        reader.close()
        store.close()


def test_acknowledgement_precedes_optional_checkpoint(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = OnlineStore.maintain

    def blocked(store, *, idle=False):
        if store.last_commit_s is not None and not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(store, idle=idle)

    monkeypatch.setattr(OnlineStore, "maintain", blocked)
    p = processor(
        tmp_path / "run",
        save_mode="async",
        save_settings=SaveSettings(checkpoint_policy="deferred"),
    )
    try:
        p.process(image(), stamp(0))
        assert entered.wait(2)
        assert p.store.status()["committed_frames"] == 1
        assert len(p.poll_saved()) == 1
        with sqlite3.connect(p.output / "tracking.sqlite") as connection:
            assert connection.execute("SELECT count(*) FROM frames").fetchone()[0] == 1
    finally:
        release.set()
        p.close()


def test_periodic_passive_checkpoint_reuses_file_space(tmp_path):
    store = make_store(tmp_path / "run")
    with store.db:
        store.db.execute("INSERT INTO session VALUES ('data','saved')")
    before = store.wal_bytes()
    store.last_commit_s = time.perf_counter()
    store.needs_checkpoint = True
    store.frames_since_checkpoint = 128
    store.maintain(idle=True)
    assert store.wal_bytes() == before
    assert store.frames_since_checkpoint == 0 and not store.needs_checkpoint
    store.close()
    with (tmp_path / "run/database_maintenance.csv").open() as file:
        log = list(csv.DictReader(file))
    assert log[0]["operation"] == "checkpoint_passive"
    assert log[0]["reason"] == "interval_128_frames"


def crash_after_acknowledgement(path, pipe):
    """只退出本测试新建的子进程；刻意不执行 finish/checkpoint。"""
    p = processor(
        path,
        save_mode="async",
        save_settings=SaveSettings(checkpoint_policy="deferred"),
    )
    for index in range(5):
        p.process(image(), stamp(index))
    deadline = time.perf_counter() + 5
    while p.store.status()["committed_frames"] < 5:
        p.poll_saved()
        if time.perf_counter() > deadline:
            os._exit(2)
        time.sleep(0.001)
    pipe.send(p.store.status()["committed_frames"])
    os._exit(0)


def test_acknowledged_frames_survive_process_exit_without_checkpoint(tmp_path):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(
        target=crash_after_acknowledgement, args=(tmp_path, sender)
    )
    child.start()
    try:
        assert receiver.poll(10)
        assert receiver.recv() == 5
        child.join(5)
        assert child.exitcode == 0
        assert (tmp_path / "run/tracking.sqlite-wal").exists()
        with sqlite3.connect(tmp_path / "run/tracking.sqlite") as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert db.execute("SELECT count(*) FROM frames").fetchone()[0] == 5
    finally:
        if child.is_alive():
            child.terminate()
            child.join()
        receiver.close()
        sender.close()


@pytest.mark.parametrize(
    "settings",
    [{"checkpoint_policy": "off"}, {"wal_limit_mb": 0}, {"wal_limit_mb": 1.5}],
)
def test_invalid_maintenance_settings(settings):
    with pytest.raises(ValueError):
        SaveSettings(**settings)
