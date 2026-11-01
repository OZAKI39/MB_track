"""分析端准备必须先于送帧，不推进科学状态，失败时不启动后台资源。"""

import json
import os
import threading
import time
import weakref
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from customer_workspace.microbubble_tracking import online_tracking, paced_replay
from customer_workspace.microbubble_tracking.tests.test_online_tracking import (
    image,
    processor,
    rows,
    stamp,
)


def observe_label_write(monkeypatch, *, failure=False):
    """用非零内容开始，确认完整实际写入，而不依赖新内存碰巧为零。"""
    observed = {}
    original = online_tracking.OnlineBackground

    class ObservedLabels(np.ndarray):
        def fill(self, value):
            observed["write_called"] = True
            if failure:
                raise MemoryError("injected label preparation failure")
            super().fill(value)
            observed.setdefault("write_finished_s", time.perf_counter())

    def background(initial):
        result = original(initial)
        labels = np.full(initial.shape, -123, np.int32).view(ObservedLabels)
        result.workspace.component_labels = labels
        observed["labels"] = labels
        return result

    monkeypatch.setattr(online_tracking, "OnlineBackground", background)
    return observed


@pytest.mark.parametrize("save_mode", ["sync", "async"])
def test_construction_prepares_labels_without_processing_a_frame(
    tmp_path, monkeypatch, save_mode
):
    observed = observe_label_write(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation must not process a frame")

    with monkeypatch.context() as startup:
        startup.setattr(online_tracking, "segment", forbidden)
        startup.setattr(online_tracking.Tracker, "step", forbidden)
        p = processor(tmp_path / "tracking", save_mode=save_mode)
    with p:
        assert observed["write_called"]
        assert p.background.workspace.component_labels is observed["labels"]
        assert not np.any(observed["labels"])
        np.testing.assert_array_equal(p.background.B, np.full((80, 80), 200, np.float32))
        assert p.background.k == 0 and p.last_stamp is None
        assert p.tracker.next_id == 1 and not p.tracker.tracks
        assert not rows(p.output, "SELECT * FROM frames")
        metadata = json.loads((p.output / "session.json").read_text())
        prepared = metadata["startup_preparation"]
        assert prepared["component_labels_ready"]
        assert prepared["component_labels_bytes"] == observed["labels"].nbytes
        assert prepared["component_labels_ms"] >= 0
        assert observed["write_finished_s"] <= prepared["component_labels_completed_perf_s"]
        assert prepared["component_labels_completed_perf_s"] <= time.perf_counter()
        result = p.process(image((20, 20, 12, 12)), stamp(274, 1.25))
        assert result["frame"]["source_frame"] == 274
        assert result["frame"]["source_time_s"] == 1.25
        assert result["observations"][0]["track_id"] == 1
        assert p.background.workspace.component_labels is observed["labels"]
    assert len(rows(p.output, "SELECT * FROM frames")) == 1


@pytest.mark.parametrize("failure", [False, True])
def test_analysis_preparation_precedes_even_creation_of_sender_context(
    tmp_path, monkeypatch, failure
):
    observed = observe_label_write(monkeypatch, failure=failure)
    created = []
    original = online_tracking.OnlineProcessor

    def capture_processor(*args, **kwargs):
        result = original(*args, **kwargs)
        created.append(result)
        return result

    class Video:
        streams = SimpleNamespace(video=[SimpleNamespace(
            average_rate=60, time_base=Fraction(1, 90000)
        )])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, stream):
            return iter([SimpleNamespace(to_ndarray=lambda **kwargs: np.full(
                (80, 100, 3), 200, np.uint8
            ))])

    class SenderCreationReached(Exception):
        pass

    def create_sender_context(*args):
        assert not failure, "Sender must not be created after a preparation failure"
        assert observed["write_called"] and not np.any(observed["labels"])
        assert created[0].metadata["startup_preparation"]["component_labels_ready"]
        raise SenderCreationReached

    monkeypatch.setattr(paced_replay, "OnlineProcessor", capture_processor)
    monkeypatch.setattr(paced_replay, "verify_inputs", lambda *args: {
        "experiment_id": "test", "video_sha256": "test-hash", "expected_frames": 1,
    })
    monkeypatch.setattr(paced_replay.av, "open", lambda *args: Video())
    monkeypatch.setattr(paced_replay.np, "load", lambda *args, **kwargs: np.full(
        (80, 100), 200, np.float32
    ))
    monkeypatch.setattr(paced_replay, "combine_backgrounds", lambda refs, frame: refs[0])
    monkeypatch.setattr(paced_replay.mp, "get_context", create_sender_context)
    try:
        with pytest.raises(MemoryError if failure else SenderCreationReached):
            paced_replay.paced_replay(
                tmp_path / "video", tmp_path / "reference", tmp_path / "pzt",
                tmp_path / "result",
            )
        assert observed["write_called"]
        if failure:
            assert not created and not (tmp_path / "result").exists()
    finally:
        for p in created:
            p.close()


@pytest.mark.parametrize("save_mode", ["sync", "async"])
@pytest.mark.parametrize("format_name", ["gray", "bgr", "float", "fallback"])
def test_start_prepares_real_analysis_thread_without_scientific_changes(
    tmp_path, monkeypatch, save_mode, format_name
):
    p = processor(tmp_path / "warm", save_mode=save_mode)
    plain = processor(tmp_path / "plain", save_mode=save_mode)
    # Test-only control: skip the new operation while retaining the same GC policy.
    monkeypatch.setattr(plain, "_warm_up_analysis", lambda sample: None)
    sample = image((20, 20, 12, 12))
    if format_name == "bgr":
        sample = np.repeat(sample[:, :, None], 3, axis=2)
    elif format_name == "float":
        sample = sample.astype(np.float32)
    original_sample = sample.copy()
    sample.flags.writeable = False
    background = p.background.B.copy()
    workspace = p.background.workspace
    buffers = (p._blurred, workspace._darkness, workspace.component_labels,
               workspace.selection_mask, workspace._label_indices, p._labels)
    for array in buffers:
        array.fill(19)
    temporary_refs = []
    original_tracker = online_tracking.Tracker

    def temporary_tracker(*args, **kwargs):
        tracker = original_tracker(*args, **kwargs)
        temporary_refs.append(weakref.ref(tracker))
        return tracker

    monkeypatch.setattr(online_tracking, "Tracker", temporary_tracker)
    with plain:
        try:
            p.start(None if format_name == "fallback" else sample)
            preparation = (p.output / "analysis_preparation.json").read_bytes()
            p.start(sample)  # Must not warm again or consume a second synthetic frame.
            assert (p.output / "analysis_preparation.json").read_bytes() == preparation
            assert len(temporary_refs) == 1 and temporary_refs[0]() is None
            assert all(not np.any(array) for array in buffers)
            np.testing.assert_array_equal(p.background.B, background)
            np.testing.assert_array_equal(sample, original_sample)
            np.testing.assert_array_equal(workspace._denominator, background + 1)
            assert p.background.k == 0 and p.last_stamp is None
            assert p.tracker.next_id == 1 and not p.tracker.tracks
            assert not rows(p.output, "SELECT * FROM frames")
            ready = json.loads((p.output / "analysis_ready.json").read_text())
            prepared = json.loads(preparation)
            assert prepared["pid"] == ready["pid"] == os.getpid()
            assert prepared["thread_id"] == ready["thread_id"] == threading.get_native_id()
            assert prepared["scientific_state_unchanged"]
            assert prepared["finished_perf_s"] <= ready["ready_perf_s"]
            # True source numbers include a gap; warm-up must not invent frame zero.
            for number in (274, 275, 278, 279, 280, 281):
                frame = image((20 + number - 274, 20, 12, 12))
                actual = p.process(frame, stamp(number))
                expected = plain.process(frame, stamp(number))
                assert actual["observations"] == expected["observations"]
                for (aid, ac), (eid, ec) in zip(actual["contours"], expected["contours"], strict=True):
                    assert aid == eid
                    np.testing.assert_array_equal(ac, ec)
                np.testing.assert_array_equal(p.background.B, plain.background.B)
        finally:
            p.close()
    for table in ("observations", "tracks", "events", "event_updates", "track_updates"):
        assert rows(p.output, f"SELECT * FROM {table}") == rows(plain.output, f"SELECT * FROM {table}")


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_warmup_failure_never_reports_ready_or_leaves_writer_running(tmp_path, monkeypatch, failure):
    p = processor(tmp_path, save_mode="async")

    def fail(*args, **kwargs):
        raise failure("injected warm-up failure")

    monkeypatch.setattr(online_tracking, "segment", fail)
    with pytest.raises(failure, match="injected warm-up failure"):
        with p:
            pytest.fail("Must not start after failed warm-up")
    assert p.closed and p.failed and not p.stream_started
    assert not p.store.worker.is_alive()
    assert not (p.output / "analysis_ready.json").exists()
    assert not rows(p.output, "SELECT * FROM frames")
    report = json.loads((p.output / "analysis_preparation.json").read_text())
    assert report["status"] == "failed" and report["scientific_state_unchanged"]


def test_prepared_processor_rejects_analysis_on_another_thread(tmp_path):
    errors = []
    with processor(tmp_path) as p:
        def wrong_thread():
            for action in (p.start, lambda: p.process(image(), stamp(0))):
                try:
                    action()
                except RuntimeError as exc:
                    errors.append(str(exc))

        thread = threading.Thread(target=wrong_thread)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive() and len(errors) == 2
        assert all("process and thread" in error for error in errors)
        assert not rows(p.output, "SELECT * FROM frames")
        assert p.process(image((20, 20, 12, 12)), stamp(0))["observations"][0]["track_id"] == 1


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_sender_start_signal_requires_completed_analysis_preparation(tmp_path, monkeypatch, failure):
    events = []
    shared_origin = SimpleNamespace(value=0.0)
    output = tmp_path / "result"
    pzt = tmp_path / "pzt"
    pzt.mkdir()
    for name in ("actions.csv", "video_frames.csv"):
        (pzt / name).write_text("test\n")

    class Video:
        streams = SimpleNamespace(video=[SimpleNamespace(average_rate=60, time_base=Fraction(1, 90000))])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, stream):
            return iter([SimpleNamespace(to_ndarray=lambda **kwargs: np.full((80, 80, 3), 200, np.uint8))])

    class Source:
        def __init__(self, **kwargs):
            pass

        def start(self):
            events[0].set()  # Sender is prepared, but cannot publish before events[1].
            (output / "source_report.json").write_text('{"status":"completed"}')

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    def event():
        result = threading.Event()
        events.append(result)
        return result

    context = SimpleNamespace(Event=event, RawValue=lambda *args: shared_origin, Process=Source)
    original = online_tracking.OnlineProcessor._warm_up_analysis

    def warm_up(self, sample):
        assert not events[1].is_set() and shared_origin.value == 0
        if failure:
            raise failure("injected preparation failure")
        original(self, sample)

    consumed = []

    def consume(processor, *args, **kwargs):
        ready = json.loads((output / "analysis_ready.json").read_text())
        assert events[1].is_set() and ready["ready_perf_s"] < shared_origin.value
        assert processor.last_stamp is None and processor.tracker.next_id == 1
        consumed.append(True)

    monkeypatch.setattr(paced_replay, "verify_inputs", lambda *args: {
        "experiment_id": "test", "video_sha256": "test-hash", "expected_frames": 1,
    })
    monkeypatch.setattr(paced_replay.av, "open", lambda *args: Video())
    monkeypatch.setattr(paced_replay.np, "load", lambda *args, **kwargs: np.full((80, 80), 200, np.float32))
    monkeypatch.setattr(paced_replay, "combine_backgrounds", lambda refs, frame: refs[0])
    monkeypatch.setattr(paced_replay.mp, "get_context", lambda *args: context)
    monkeypatch.setattr(paced_replay, "LatestFrame", lambda *args: None)
    monkeypatch.setattr(online_tracking.OnlineProcessor, "_warm_up_analysis", warm_up)
    monkeypatch.setattr(paced_replay, "consume", consume)
    monkeypatch.setattr(paced_replay, "link_actions", lambda *args: {})
    monkeypatch.setattr(paced_replay, "summarize", lambda *args: {})
    if failure is RuntimeError:
        with pytest.raises(RuntimeError, match="Paced replay failed"):
            paced_replay.paced_replay(tmp_path / "video", tmp_path / "bg", pzt, output)
    else:
        paced_replay.paced_replay(tmp_path / "video", tmp_path / "bg", pzt, output)
    assert bool(consumed) == (failure is None)
    assert events[1].is_set() == (failure is None)
    if failure:
        assert shared_origin.value == 0 and not (output / "analysis_ready.json").exists()
    writer = json.loads((output / "writer_report.json").read_text())
    assert writer["worker_finished"] and writer["submitted_frames"] == 0
