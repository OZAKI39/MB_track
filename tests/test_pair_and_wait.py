"""检查纯记录判定、真实软件时钟及取消；不模拟或连接 AFG。"""

import math
import threading
from time import monotonic, perf_counter

import pytest

from customer_workspace.afg_timing import TimedRig
from customer_workspace.amplitude_pair_diagnostic import (
    classify_readback,
    spin_until,
    summarize_rounds,
)
from customer_workspace.experiment_runner import (
    ExperimentCancelled,
    ExperimentRunner,
    wait_until,
)
from customer_workspace.timing_benchmark import wait_until as benchmark_wait_until
from customer_workspace.wait_calibration import choose_advance_ms


@pytest.mark.parametrize(
    "x,y,value,failed,expected",
    [
        (0.001, 0.1, 0.1, False, "passed"),
        (0.001, 0.1, 0.001, False, "second_lost"),
        (0.1, 0.001, 0.001, False, "passed"),
        (0.1, 0.001, 0.1, False, "second_lost"),
        (0.001, 0.1, 0.05, False, "abnormal"),
        (0.001, 0.1, math.nan, False, "abnormal"),
        (0.001, 0.1, None, True, "query_failed"),
        (0.001, 0.1, 0.001, True, "query_failed"),
    ],
)
def test_readback_classification(x, y, value, failed, expected):
    assert classify_readback(value, x, y, failed) == expected


def test_zero_bytes_do_not_count_as_loss_and_errors_are_separate():
    rows = [
        {
            "index": 1,
            "result": "passed",
            "actual_interval_ms": 10.1,
            "error_queue_code": -102,
        },
        {"index": 2, "result": "second_lost", "actual_interval_ms": 10.2},
        {
            "index": 3,
            "result": "query_failed",
            "actual_interval_ms": 10.3,
            "amplitude_query": {"returned_byte_count": 0},
        },
        {"index": 4, "result": "abnormal", "actual_interval_ms": None},
    ]
    stats = summarize_rounds(rows)
    assert stats["total_rounds"] == 4
    assert all(
        stats[key] == 1 for key in ("passed", "second_lost", "query_failed", "abnormal")
    )
    assert stats["zero_byte_query_failures"] == 1
    assert stats["error_queue_nonzero"] == 1
    assert stats["actual_interval_samples"] == 3
    assert stats["actual_interval_median_ms"] == pytest.approx(10.2)
    assert stats["actual_interval_min_ms"] == pytest.approx(10.1)
    assert stats["actual_interval_max_ms"] == pytest.approx(10.3)


def test_advance_uses_worst_duration_p99_and_adds_margin():
    assert choose_advance_ms({10: [20.709] * 100, 60: [17.218] * 100}) == 22


@pytest.mark.parametrize("records", [{}, {10: []}, {10: [math.nan]}, {10: [-1]}])
def test_invalid_calibration_is_rejected(records):
    with pytest.raises(ValueError):
        choose_advance_ms(records)


@pytest.mark.parametrize("advance", [0, 0.001, 0.02])
def test_hybrid_wait_does_not_return_before_fixed_deadline(advance):
    deadline = monotonic() + 0.005
    wait_until(deadline, threading.Event(), advance)
    assert monotonic() >= deadline


@pytest.mark.parametrize("advance", [0.001, 1.0])
def test_hybrid_wait_is_cancellable_during_sleep_and_spin(advance):
    cancel = threading.Event()
    timer = threading.Timer(0.01, cancel.set)
    timer.start()
    try:
        with pytest.raises(ExperimentCancelled):
            wait_until(monotonic() + 0.2, cancel, advance)
    finally:
        timer.join()


def test_pair_spin_uses_its_fixed_deadline_and_checks_cancel():
    cancel = threading.Event()
    deadline = perf_counter() + 0.002
    spin_until(deadline, cancel)
    assert perf_counter() >= deadline
    cancel.set()
    with pytest.raises(ExperimentCancelled):
        spin_until(perf_counter() - 1, cancel)


@pytest.mark.parametrize("advance", [-1, math.inf, math.nan])
def test_invalid_wake_advance_is_rejected(advance):
    with pytest.raises(ValueError):
        wait_until(monotonic(), threading.Event(), advance)


def test_formal_runner_uses_shared_wait_and_fixed_origin():
    # 只测试真实软件时钟，不连接仪器，也不模拟 AFG。
    rig = TimedRig(None)
    runner = ExperimentRunner(rig)
    assert not rig.sessions
    assert benchmark_wait_until is wait_until
    assert runner._use_hybrid_wait
    runner._start_s = monotonic()
    for planned_s in (0.003, 0.030):
        runner._wait_until(planned_s)
        assert monotonic() >= runner._start_s + planned_s
    assert not rig.sessions


@pytest.mark.parametrize("planned_s", [0.020, 1.0])
def test_formal_runner_wait_can_be_cancelled(planned_s):
    runner = ExperimentRunner(TimedRig(None))
    runner._start_s = monotonic()
    timer = threading.Timer(0.001, runner._cancel.set)
    timer.start()
    try:
        with pytest.raises(ExperimentCancelled):
            runner._wait_until(planned_s)
    finally:
        timer.join()


def test_custom_clock_wait_interface_keeps_its_original_timebase():
    # 只注入软件时钟；保留调用方已有的 clock/wait 接口。
    current_s = [100.0]
    waits = []

    def clock():
        return current_s[0]

    def wait(seconds):
        waits.append(seconds)
        current_s[0] += seconds

    runner = ExperimentRunner(TimedRig(None), clock=clock, wait=wait)
    runner._start_s = clock()
    runner._wait_until(5.0)
    runner._wait_until(6.0)
    assert waits == [5.0, 1.0]
