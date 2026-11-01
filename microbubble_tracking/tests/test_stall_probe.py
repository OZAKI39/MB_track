"""诊断记录不得改变轮廓；正常结束和中途退出均关闭文件及 GC 回调。"""

import csv
import gc
import time
from itertools import pairwise

import cv2
import numpy as np
import pytest

from customer_workspace.microbubble_tracking.bubble_tracker import (
    SegmentationWorkspace,
    SegParams,
    segment,
)
from customer_workspace.microbubble_tracking.stall_probe import HandoffTrace, StallProbe


def test_buffered_probe_flushes_tail_and_keeps_scheduler_and_gc(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "customer_workspace.microbubble_tracking.stall_probe.scheduler_statistics_status",
        lambda: {"enabled": True, "source": "test"},
    )
    callbacks, thresholds = list(gc.callbacks), gc.get_threshold()
    path = tmp_path / "buffered.csv"
    probe = StallProbe(path, buffer_frames=8, scheduler=True)
    for frame in (5, 6):
        probe.begin(frame)
        gc.collect(0)
        probe.mark("work")
        probe.finish()
    assert probe.buffered_frames == 2
    assert len(probe.pending_rows) == 4
    probe.close()
    probe.close()
    rows = list(csv.DictReader(path.open()))
    assert [row["frame"] for row in rows] == ["5", "5", "6", "6"]
    assert all(float(row["runnable_wait_ms"]) >= 0 for row in rows)
    assert len(list(csv.DictReader(path.with_suffix(".gc.csv").open()))) >= 2
    assert probe.scheduler_file.closed
    assert gc.callbacks == callbacks
    assert gc.get_threshold() == thresholds


def test_disabled_kernel_statistics_are_missing_not_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "customer_workspace.microbubble_tracking.stall_probe.scheduler_statistics_status",
        lambda: {"enabled": False, "source": "test"},
    )
    path = tmp_path / "unavailable.csv"
    probe = StallProbe(path, scheduler=True)
    try:
        probe.begin(0)
        probe.finish()
    finally:
        probe.close()
    assert probe.scheduler_file is None
    with path.open() as file:
        assert next(csv.DictReader(file))["runnable_wait_ms"] == ""


def test_basic_probe_does_not_scan_threads(tmp_path, monkeypatch):
    def reject_scan(path):
        raise AssertionError(f"Unexpected thread scan: {path}")

    monkeypatch.setattr("os.listdir", reject_scan)
    probe = StallProbe(tmp_path / "basic.csv")
    try:
        probe.begin(0)
        probe.mark("work")
        probe.finish()
    finally:
        probe.close()
    rows = list(csv.DictReader((tmp_path / "basic.csv").open()))
    assert all(row["threads"] == "" for row in rows)


@pytest.mark.parametrize("detailed", [False, True])
def test_diagnostics_preserve_segmentation_and_save_each_stage(tmp_path, detailed):
    image = np.full((80, 90), 180, np.float32)
    image[15:28, 30:53] = 75
    background = np.full_like(image, 180)
    workspace = SegmentationWorkspace(background)
    params = SegParams()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expected = segment(image, background, params, kernel, workspace=workspace)
    path = tmp_path / "stages.csv"
    probe = StallProbe(path, detailed=detailed)
    try:
        probe.begin(274)
        actual = segment(
            image,
            background,
            params,
            kernel,
            workspace=workspace,
            diagnostics=probe,
        )
        probe.finish()
    finally:
        probe.close()
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[2], expected[2])
    assert len(actual[1]) == len(expected[1])
    for left, right in zip(actual[1], expected[1]):
        np.testing.assert_array_equal(left, right)
    rows = list(csv.DictReader(path.open()))
    assert {r["frame"] for r in rows} == {"274"}
    stages = [r["stage"] for r in rows]
    detail_stages = {
        "contrast_divide", "contrast_subtract", "contrast_median", "contrast_center",
        "selection_strong_core", "selection_allocate",
    }
    if detailed:
        assert detail_stages <= set(stages)
    assert [stage for stage in stages if stage not in detail_stages] == [
        "blur",
        "contrast",
        "candidates",
        "components",
        "selection",
        "find_contours",
        "draw_contours",
        "filter_areas",
        "tail",
    ]
    assert all(float(r["wall_ms"]) >= 0 for r in rows)
    assert all(float(r["thread_cpu_ms"]) >= 0 for r in rows)
    assert all(a["end_s"] == b["start_s"] for a, b in pairwise(rows))


def test_close_records_partial_frame_and_removes_gc_callback(tmp_path):
    callbacks = list(gc.callbacks)
    path = tmp_path / "partial.csv"
    probe = StallProbe(path)
    probe.begin(7)
    gc.collect()
    probe.mark("before_error")
    probe.close()
    assert gc.callbacks == callbacks
    rows = list(csv.DictReader(path.open()))
    assert [r["stage"] for r in rows] == ["before_error", "tail"]
    gc_rows = list(csv.DictReader(path.with_suffix(".gc.csv").open()))
    assert any(r["frame"] == "7" and r["generation"] == "2" for r in gc_rows)


def test_diagnostic_frame_lifecycle_is_explicit(tmp_path):
    probe = StallProbe(tmp_path / "lifecycle.csv")
    try:
        with pytest.raises(RuntimeError, match="not started"):
            probe.mark("invalid")
        probe.begin(1)
        with pytest.raises(RuntimeError, match="still open"):
            probe.begin(2)
    finally:
        probe.close()


def test_detailed_contrast_is_identical_and_handoff_is_contiguous(tmp_path):
    rng = np.random.default_rng(61)
    background = rng.uniform(120, 210, (80, 90)).astype(np.float32)
    image = rng.uniform(40, 180, background.shape).astype(np.float32)
    workspace = SegmentationWorkspace(background)
    expected = workspace.compute(image).copy()
    probe = StallProbe(tmp_path / "details.csv", detailed=True)
    path = tmp_path / "handoff.csv"
    started = time.perf_counter()
    try:
        with HandoffTrace(path) as handoff:
            probe.begin(4)
            actual = workspace.compute(image, diagnostics=probe).copy()
            handoff.begin(4, started)
            probe.finish(handoff=handoff)
            ended = time.perf_counter()
            handoff.mark("take_return", ended_s=ended)
            handoff.finish()
    finally:
        probe.close()
    np.testing.assert_array_equal(actual, expected)
    rows = list(csv.DictReader(path.open()))
    assert float(rows[0]["start_s"]) == started
    assert float(rows[-1]["end_s"]) == ended
    assert all(a["end_s"] == b["start_s"] for a, b in pairwise(rows))
    assert sum(float(r["wall_ms"]) for r in rows) == pytest.approx((ended - started) * 1000)
    assert {r["after_frame"] for r in rows} == {"4"}
    assert "diagnostic_csv" in {r["stage"] for r in rows}


def test_handoff_waiting_history_stays_bounded(tmp_path):
    path = tmp_path / "waiting.csv"
    with HandoffTrace(path) as handoff:
        handoff.begin(7, time.perf_counter())
        for _ in range(1000):
            handoff.mark("wait_for_frame")
            assert len(handoff.rows) < 256
    rows = list(csv.DictReader(path.open()))
    assert sum(r["stage"] == "wait_for_frame" for r in rows) == 1000
    assert sum(r["stage"] == "trace_flush" for r in rows) >= 3
