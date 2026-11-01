"""用明确的时间记录检查统计和失败判定，不模拟或连接 AFG。"""

import math

import pytest

from customer_workspace.benchmark_report import distribution, summarize_samples
from customer_workspace.timing_benchmark import validate_protocol


def recorded_samples(delays, method="schedule"):
    """构造数据分析的输入行；不实现仪器接口，也不执行任何命令。"""
    rows = []
    for index, delay in enumerate(delays):
        planned = index * 0.020
        sent = planned + delay / 1000 if method == "schedule" else index * 0.100
        returned = sent + 0.0001
        reply = sent + delay / 1000 if method == "readback" else None
        target = 0.1 if index % 2 == 0 else 0.001
        rows.append(
            {
                "index": index + 1,
                "state": "ACTIVE" if index % 2 == 0 else "IDLE",
                "status": "sent" if method == "schedule" else "readback_ok",
                "planned_s": planned if method == "schedule" else None,
                "ready_s": planned,
                "call_started_s": sent,
                "command_sent_s": sent,
                "command_write_returned_s": returned,
                "call_finished_s": (reply if reply is not None else returned) + 0.0001,
                "confirmed_s": reply,
                "amplitude_reply_received_s": reply,
                "query_started_s": returned,
                "query": "AMPLitude?" if reply is not None else None,
                "amplitude_requested_vpp": target,
                "amplitude_readback_vpp": target if reply is not None else None,
            }
        )
    return rows


def test_oscillating_delays_have_no_accumulating_trend():
    stats = summarize_samples(recorded_samples([1, 2, 3, 4, 5] * 100), "schedule", 20)
    assert stats["samples"] == 500
    assert stats["delay_ms"]["mean"] == pytest.approx(3)
    assert abs(stats["trend"]["slope_ms_per_100_commands"]) < 0.01
    assert stats["trend"]["first_window_mean_ms"] == pytest.approx(3)
    assert stats["trend"]["last_window_mean_ms"] == pytest.approx(3)
    assert stats["sent_at_least_one_interval_late"] == 0


def test_accumulating_delay_exposes_trend_and_missed_intervals():
    stats = summarize_samples(
        recorded_samples([index * 0.1 for index in range(500)]), "schedule", 20
    )
    assert stats["trend"]["slope_ms_per_100_commands"] == pytest.approx(10)
    assert stats["trend"]["fitted_total_change_ms"] == pytest.approx(49.9)
    assert stats["trend"]["last_window_mean_ms"] - stats["trend"][
        "first_window_mean_ms"
    ] == pytest.approx(45)
    assert stats["previous_call_finished_after_next_deadline"] > 250


def test_roundtrip_reports_observed_slowest_reply_and_separate_states():
    stats = summarize_samples(
        recorded_samples([2, 3, 8, 4], "readback"), "readback", 20
    )
    assert stats["slowest"]["index"] == 3
    assert stats["delay_ms"]["max"] == pytest.approx(8)
    assert stats["by_state"]["ACTIVE"]["mean"] == pytest.approx(5)
    assert "trend" not in stats


@pytest.mark.parametrize(
    "change",
    [
        {"confirmed_s": None},
        {"amplitude_readback_vpp": 0.001},
        {"status": "failed"},
        {"query_started_s": math.nan},
        {"command_write_returned_s": 1.0},
    ],
)
def test_failed_or_invalid_readbacks_do_not_enter_statistics(change):
    samples = recorded_samples([2, 3], "readback")
    samples[0].update(change)
    with pytest.raises(ValueError):
        summarize_samples(samples, "readback", 20)


def test_schedule_cannot_hide_drift_by_shifting_planned_times():
    samples = recorded_samples([2, 3])
    samples[1]["planned_s"] += 0.003
    with pytest.raises(ValueError, match="固定起点"):
        summarize_samples(samples, "schedule", 20)


def test_percentiles_use_documented_nearest_rank():
    result = distribution(list(range(1, 501)))
    assert result["p95"] == 475 and result["p99"] == 495


@pytest.mark.parametrize(
    "count,interval",
    [
        (1, 20),
        (3, 20),
        (True, 20),
        (500, 0),
        (500, -1),
        (500, math.nan),
        (500, math.inf),
    ],
)
def test_protocol_rejected_before_hardware(count, interval):
    with pytest.raises(ValueError):
        validate_protocol("schedule", count, interval)


def scheduled_readback_samples(lateness_ms, reply_ms, interval_ms=50):
    """构造有共同计划起点的已记录数据，两个耗时明确属于同一条指令。"""
    rows = recorded_samples(reply_ms, "readback")
    for index, (row, lateness) in enumerate(zip(rows, lateness_ms)):
        planned = index * interval_ms / 1000
        offset = planned + lateness / 1000 - row["command_sent_s"]
        for key in (
            "call_started_s",
            "command_sent_s",
            "command_write_returned_s",
            "query_started_s",
            "amplitude_reply_received_s",
            "confirmed_s",
            "call_finished_s",
        ):
            row[key] += offset
        row["planned_s"] = planned
        row["ready_s"] = planned
    return rows


def test_budget_uses_same_command_instead_of_sum_of_independent_maxima():
    rows = scheduled_readback_samples([8, 2], [20, 42])
    stats = summarize_samples(rows, "scheduled_readback", 50)
    assert stats["planned_to_reply_ms"]["max"] == pytest.approx(44)
    assert stats["sending_delay_ms"]["max"] + stats["delay_ms"]["max"] == pytest.approx(
        50
    )
    worst = stats["slowest_planned_to_reply"]
    assert worst["index"] == 2
    assert worst["lateness_ms"] == pytest.approx(2)
    assert worst["command_to_reply_ms"] == pytest.approx(42)
    assert stats["minimum_margin_ms"] == pytest.approx(6)
    assert stats["all_replies_with_5ms_reserve"] is True


def test_completion_can_miss_reserve_without_crossing_next_deadline():
    rows = scheduled_readback_samples([4, 6], [40, 40])
    stats = summarize_samples(rows, "scheduled_readback", 50)
    assert stats["replies_exceeding_reserve_target"] == 1
    assert stats["replies_at_or_after_next_deadline"] == 0
    assert stats["all_replies_with_5ms_reserve"] is False


def test_overdue_reply_is_counted_and_not_hidden_by_later_recovery():
    rows = scheduled_readback_samples([1, 2], [50, 30])
    stats = summarize_samples(rows, "scheduled_readback", 50)
    assert stats["planned_to_reply_ms"]["max"] == pytest.approx(51)
    assert stats["minimum_margin_ms"] == pytest.approx(-1)
    assert stats["replies_at_or_after_next_deadline"] == 1
    assert stats["previous_call_finished_after_next_deadline"] == 1


def test_55ms_trial_keeps_same_five_millisecond_reserve():
    rows = scheduled_readback_samples([5, 8], [40, 41], interval_ms=55)
    stats = summarize_samples(rows, "scheduled_readback", 55)
    assert stats["reserve_target_ms"] == 50
    assert stats["planned_to_reply_ms"]["max"] == pytest.approx(49)
    assert stats["all_replies_with_5ms_reserve"] is True
    validate_protocol("scheduled_readback", 500, 55)


def test_scheduled_readback_rejects_a_moving_plan():
    rows = scheduled_readback_samples([5, 6], [25, 26])
    rows[1]["planned_s"] = 0.056
    with pytest.raises(ValueError, match="固定起点"):
        summarize_samples(rows, "scheduled_readback", 50)


def test_pre_query_wait_counts_toward_reply_time_and_reserve():
    rows = scheduled_readback_samples([4, 6], [30, 30])
    for row in rows:
        row["before_query_min_ms"] = 10.0
        row["write_to_query_ms"] = 10.0
        for key in (
            "query_started_s",
            "amplitude_reply_received_s",
            "confirmed_s",
            "call_finished_s",
        ):
            row[key] += 0.010
    stats = summarize_samples(rows, "scheduled_readback", 50)
    assert stats["write_to_query_ms"]["min"] == pytest.approx(10)
    assert stats["delay_ms"]["max"] == pytest.approx(40)
    assert stats["planned_to_reply_ms"]["max"] == pytest.approx(46)
    assert stats["replies_exceeding_reserve_target"] == 1
    assert stats["all_replies_with_5ms_reserve"] is False


@pytest.mark.parametrize("actual_gap_ms,recorded_gap_ms", [(9, 9), (10, 0)])
def test_missing_wait_or_incorrect_wait_record_is_rejected(
    actual_gap_ms, recorded_gap_ms
):
    rows = scheduled_readback_samples([4, 6], [30, 30])
    for row in rows:
        row["before_query_min_ms"] = 10.0
        row["query_started_s"] += actual_gap_ms / 1000
        row["write_to_query_ms"] = recorded_gap_ms
    with pytest.raises(ValueError, match="查询前等待"):
        summarize_samples(rows, "scheduled_readback", 50)
