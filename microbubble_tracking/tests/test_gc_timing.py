"""GC 对照必须恢复进程设置，异常/取消时仍让保存线程完成收尾。"""

import gc
import json

import pytest

from customer_workspace.microbubble_tracking.gc_replay_check import GCTrial
from customer_workspace.microbubble_tracking.tests.test_online_tracking import (
    image,
    processor,
    rows,
    stamp,
)


@pytest.mark.parametrize("mode", ["normal", "precollect", "defer_full"])
@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_restore_and_drain_writer_on_normal_failure_and_cancel(tmp_path, mode, failure):
    output = tmp_path / "gc"
    output.mkdir()
    thresholds, enabled, debug = gc.get_threshold(), gc.isenabled(), gc.get_debug()
    callbacks = list(gc.callbacks)
    caught = None
    p = None
    try:
        with (
            GCTrial(output, mode) as trial,
            processor(tmp_path / "tracking", save_mode="async") as p,
        ):
            trial.begin_stream()
            assert gc.isenabled()
            assert gc.get_threshold()[:2] == thresholds[:2]
            assert gc.get_threshold()[2] == (
                1_000_000 if mode == "defer_full" else thresholds[2]
            )
            for index in range(6):
                p.process(image((20 + index, 20, 12, 12)), stamp(index))
            if failure is not None:
                raise failure("injected experiment interruption")
    except (RuntimeError, KeyboardInterrupt) as exc:
        caught = exc
    assert (type(caught) if caught else None) is failure
    assert gc.get_threshold() == thresholds and gc.isenabled() == enabled
    assert gc.get_debug() == debug and gc.callbacks == callbacks
    assert not p.store.worker.is_alive()
    assert len(rows(p.output, "SELECT * FROM frames")) == 6
    saved = json.loads((p.output / "writer_report.json").read_text())
    assert saved["all_submitted_saved"] and saved["pending_records"] == 0
    assert saved["summary"]["status"] == (
        "completed"
        if failure is None
        else "cancelled"
        if failure is KeyboardInterrupt
        else "failed"
    )
    assert (output / "gc_events.csv").exists()
    trial.finish()  # 重复关闭不能再次清理、改设置或覆盖已有文件。


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_cleanup_exception_still_restores_settings(tmp_path, monkeypatch, failure):
    thresholds, callbacks = gc.get_threshold(), list(gc.callbacks)
    with GCTrial(tmp_path, "defer_full") as trial:
        trial.begin_stream()

        def fail(*args):
            raise failure("injected collect failure")

        monkeypatch.setattr(gc, "collect", fail)
        with pytest.raises(failure, match="collect failure"):
            trial.finish()
        assert gc.get_threshold() == thresholds
        assert gc.isenabled() and gc.callbacks == callbacks


def test_audit_frees_retained_objects_and_keeps_existing_garbage(tmp_path):
    existing = list(gc.garbage)
    debug = gc.get_debug()
    with GCTrial(tmp_path, "precollect", audit_objects=True) as trial:
        trial.begin_stream()
        loop = []
        loop.append(loop)
        del loop
        gc.collect(2)
        assert len(gc.garbage) > len(existing)
    assert gc.garbage == existing
    assert gc.get_debug() == debug
    assert trial.report["unreachable_objects"]["count"] > 0


def test_default_replay_does_not_install_gc_trial():
    import inspect

    from customer_workspace.microbubble_tracking.paced_replay import paced_replay

    assert inspect.signature(paced_replay).parameters["gc_control"].default is None
