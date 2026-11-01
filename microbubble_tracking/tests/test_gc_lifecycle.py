"""正式清理只发生在流边界；数据先保存，GC 不改配置，不掩盖原始故障。"""

import gc
import json
import threading
from dataclasses import replace

import pytest

from customer_workspace.microbubble_tracking import online_tracking
from customer_workspace.microbubble_tracking.async_store import SaveError
from customer_workspace.microbubble_tracking.tests.test_async_store import block_first_write
from customer_workspace.microbubble_tracking.tests.test_online_tracking import (
    image, processor, rows, stamp,
)


@pytest.mark.parametrize("save_mode", ["sync", "async"])
@pytest.mark.parametrize("interruption", [None, RuntimeError, KeyboardInterrupt])
def test_two_collections_preserve_settings_and_drain_saved_frames(
    tmp_path, monkeypatch, save_mode, interruption
):
    calls = []
    settings = gc.get_threshold(), gc.isenabled(), gc.get_debug()
    owner = threading.get_native_id()
    p = processor(tmp_path, save_mode=save_mode)

    def collect(generation):
        assert generation == 2 and threading.get_native_id() == owner
        assert (gc.get_threshold(), gc.isenabled(), gc.get_debug()) == settings
        if not calls:
            assert p.last_stamp is None and p.background.k == 0
            assert p.tracker.next_id == 1 and not p.tracker.tracks
        else:
            assert p.closed and len(rows(p.output, "SELECT * FROM frames")) == 6
            if save_mode == "async":
                assert not p.store.worker.is_alive()
                assert p.store.status()["pending_records"] == 0
        calls.append(generation)
        return 0

    def forbidden(*args):
        raise AssertionError("Production lifecycle must not change automatic GC settings")

    monkeypatch.setattr(gc, "collect", collect)
    for name in ("enable", "disable", "set_threshold", "set_debug"):
        monkeypatch.setattr(gc, name, forbidden)
    caught = None
    try:
        with p:
            p.start()  # 重复调用不重复清理。
            for number in range(6):
                p.process(image((20 + number, 20, 12, 12)), stamp(274 + number))
                assert calls == [2]
            if interruption is not None:
                raise interruption("injected analysis stop")
    except (RuntimeError, KeyboardInterrupt) as exc:
        caught = exc
    assert (type(caught) if caught else None) is interruption
    assert calls == [2, 2]
    assert [row["source_frame"] for row in rows(
        p.output, "SELECT source_frame FROM frames ORDER BY source_frame"
    )] == list(range(274, 280))
    report = json.loads((p.output / "gc_lifecycle.json").read_text())
    for phase in ("before_stream", "after_save"):
        assert report[phase]["status"] == "completed"
        assert report[phase]["thresholds_before"] == list(settings[0])
        assert report[phase]["enabled_after"]
    assert report["save_finished_perf_s"] <= report["after_save"]["started_perf_s"]
    p.close()
    assert calls == [2, 2]


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_start_failure_closes_empty_writer_and_preserves_original_error(
    tmp_path, monkeypatch, exception
):
    p = processor(tmp_path, save_mode="async")
    calls = []

    def collect(generation):
        calls.append(generation)
        if len(calls) == 1:
            raise exception("injected startup collection failure")
        assert not p.store.worker.is_alive()
        return 0

    monkeypatch.setattr(gc, "collect", collect)
    with pytest.raises(exception, match="startup collection failure"):
        with p:
            pytest.fail("Failed startup must not accept a frame")
    assert p.closed and not p.store.worker.is_alive()
    assert not rows(p.output, "SELECT * FROM frames")
    assert calls == [2, 2]


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_shutdown_collection_error_cannot_lose_saved_frames(tmp_path, monkeypatch, exception):
    p = processor(tmp_path, save_mode="async")
    p.start()
    for number in range(6):
        p.process(image(), stamp(number))

    def collect(generation):
        assert not p.store.worker.is_alive()
        assert len(rows(p.output, "SELECT * FROM frames")) == 6
        raise exception("injected final collection failure")

    monkeypatch.setattr(gc, "collect", collect)
    with pytest.raises(exception, match="final collection failure"):
        p.close()
    report = json.loads((p.output / "gc_lifecycle.json").read_text())
    assert report["after_save"]["status"] == "failed"
    saved = json.loads((p.output / "writer_report.json").read_text())
    assert saved["all_submitted_saved"] and saved["committed_frames"] == 6
    with pytest.raises(RuntimeError, match="final collection failure"):
        p.close()


def test_original_analysis_error_is_not_masked_by_final_gc_error(tmp_path, monkeypatch):
    p = processor(tmp_path, save_mode="async")
    p.start()

    def fail(generation):
        raise RuntimeError("second failure in GC")

    with pytest.raises(ValueError, match="original analysis failure") as caught:
        with p:
            p.process(image(), stamp(0))
            monkeypatch.setattr(gc, "collect", fail)
            raise ValueError("original analysis failure")
    assert any("second failure in GC" in note for note in caught.value.__notes__)
    assert len(rows(p.output, "SELECT * FROM frames")) == 1


def test_cancellation_during_save_wait_still_drains_before_gc(tmp_path, monkeypatch):
    p = processor(tmp_path, save_mode="async")
    p.start()
    for number in range(6):
        p.process(image(), stamp(number))
    original = p.store.finish
    calls = []

    def interrupted(status):
        calls.append(status)
        if len(calls) == 1:
            raise KeyboardInterrupt("cancel while waiting for save")
        return original(status)

    monkeypatch.setattr(p.store, "finish", interrupted)
    with pytest.raises(KeyboardInterrupt, match="waiting for save"):
        p.close()
    assert calls == ["completed", "cancelled"]
    assert not p.store.worker.is_alive()
    assert len(rows(p.output, "SELECT * FROM frames")) == 6
    report = json.loads((p.output / "gc_lifecycle.json").read_text())
    assert report["after_save"]["status"] == "completed"


def test_save_timeout_skips_final_gc_while_worker_is_alive(tmp_path, monkeypatch):
    entered, release = block_first_write(monkeypatch)
    p = processor(tmp_path, save_mode="async")
    p.start()
    p.process(image(), stamp(0))
    assert entered.wait(1)
    p.store.settings = replace(p.store.settings, close_timeout_s=0.05)

    def forbidden(generation):
        raise AssertionError("Cannot collect as if saving had finished")

    monkeypatch.setattr(gc, "collect", forbidden)
    try:
        with pytest.raises(SaveError, match="Timed out"):
            p.close()
        report = json.loads((p.output / "gc_lifecycle.json").read_text())
        assert report["after_save"] == {
            "status": "skipped", "reason": "save_worker_still_running",
        }
    finally:
        release.set()
        assert p.store.done.wait(2)


def test_disabled_automatic_gc_is_rejected_without_changing_it(tmp_path):
    p = processor(tmp_path, save_mode="async")
    originally_enabled = gc.isenabled()
    try:
        gc.disable()
        with pytest.raises(RuntimeError, match="Automatic GC must be enabled"):
            with p:
                pytest.fail("Disabled automatic GC must not be silently accepted")
        assert not gc.isenabled() and not p.store.worker.is_alive()
    finally:
        if originally_enabled:
            gc.enable()
