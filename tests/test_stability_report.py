"""持续运行的趋势判定不能将缺记录或持续增长误认为稳定。"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from customer_workspace.parallel_camera import ParallelCameraStream
from customer_workspace.stability_check import run_series
from customer_workspace.stability_report import memory_problems, series_stats


@pytest.fixture
def criteria():
    path = Path(__file__).parents[1] / "configs/video_stability.json"
    return json.loads(path.read_text())


def test_stable_memory_with_small_logging_growth_passes(criteria):
    times = np.arange(62)
    stats = series_stats(times, 100 + 0.05 * times, 62)
    assert not memory_problems("receiver", stats, 62, criteria)
    assert stats["samples"] == 62
    assert stats["tail_change"] < 3


def test_continuous_memory_growth_is_reported(criteria):
    times = np.arange(62)
    stats = series_stats(times, 100 + times, 62)
    assert memory_problems("receiver", stats, 62, criteria)
    assert stats["late_slope_per_second"] == pytest.approx(1)


@pytest.mark.parametrize("times", [[], list(range(5)), list(range(0, 62, 3))])
def test_missing_memory_records_do_not_pass(times, criteria):
    stats = series_stats(times, [100] * len(times), 62)
    assert memory_problems("receiver", stats, 62, criteria)


def test_startup_before_recording_does_not_change_window_statistics(criteria):
    times = np.arange(-10, 70)
    values = np.where(times < 0, 1000, 100)
    stats = series_stats(times, values, 62)
    assert stats["samples"] == 62
    assert stats["max"] == 100
    assert not memory_problems("receiver", stats, 62, criteria)


def test_one_memory_jump_is_still_visible_in_total_growth(criteria):
    times = np.arange(62)
    stats = series_stats(times, np.where(times < 15, 100, 180), 62)
    assert memory_problems("receiver", stats, 62, criteria)
    assert stats["tail_change"] == 80


def test_failed_run_is_preserved_and_other_runs_continue(
    tmp_path, monkeypatch, criteria
):
    calls = []

    def capture(args):
        calls.append(args.output_dir)
        if len(calls) == 2:
            raise RuntimeError("independent test failure")
        return {"recording_passed": True, "analysis_hz": 7}

    monkeypatch.setattr("customer_workspace.stability_check.run_check", capture)
    outcomes = run_series(tmp_path / "series", criteria, tmp_path / "geometry.json")
    assert len(calls) == 4
    assert not outcomes[1]["capture_completed"]
    assert "independent test failure" in outcomes[1]["error"]
    assert outcomes[2]["capture_completed"]
    assert len({row["wsl_pid"] for row in outcomes}) == 1


def test_original_output_is_not_overwritten(tmp_path, criteria):
    with pytest.raises(FileExistsError):
        run_series(tmp_path, criteria, tmp_path / "geometry.json")


@pytest.mark.parametrize("record_closed", [False, True])
def test_stale_input_is_reported_only_during_capture(
    tmp_path, monkeypatch, record_closed
):
    stream = ParallelCameraStream.__new__(ParallelCameraStream)
    stream.stopping = False
    stream.output = tmp_path
    stream.state = "streaming"
    stream.record_closed = record_closed
    stream.input_frames = [{"received_windows_mono": 6}]
    stream.levels = []
    stream.errors = []
    stream.next_memory_sample = 100
    stream.record_eos = threading.Event()
    queue = SimpleNamespace(get_property=lambda key: 0)
    stream.pipeline = SimpleNamespace(get_by_name=lambda name: queue)
    monkeypatch.setattr("customer_workspace.parallel_camera.time.monotonic", lambda: 10)
    stream._monitor()
    stream._monitor()
    if record_closed:
        assert not stream.errors
    else:
        assert len(stream.errors) == 1
        assert "3 秒" in stream.errors[0]
        assert (tmp_path / "receiver_error.json").exists()
