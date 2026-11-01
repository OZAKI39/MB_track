"""只核对已批准的时间表和读回记录判断，不模拟或连接 AFG。"""

from copy import deepcopy
from dataclasses import replace

import pytest

from customer_workspace.experiment_runner import CHANNEL_NAMES, PROJECT_ROOT, load_plan
from customer_workspace.four_pzt_validation import (
    CASES,
    check_times,
    compare_readback,
    expected_at,
)
from microbubble_experiment.afg import load_config


def plan_for(case):
    path = (
        PROJECT_ROOT / "customer_workspace/configs/four_pzt" / f"{CASES[case][0]}.json"
    )
    return load_plan(path, load_config())


@pytest.mark.parametrize("case,count", [(1, 6), (2, 6), (3, 5), (4, 6), (5, 5), (6, 3)])
def test_approved_plans_stop_at_twenty_and_leave_room_for_readbacks(case, count):
    plan = plan_for(case)
    assert plan.duration_s == 20
    assert plan.steps[-1].state == "STOP"
    times = check_times(plan, 10 if case == 6 else None)
    assert len(times) == count
    assert all(0 < t < 20 for t in times)
    assert all(not state["output"] for state in expected_at(plan, 20).values())


@pytest.mark.parametrize(
    "case,active_by_check",
    [
        (1, [(), ("PZT1",), ("PZT2",), ("PZT3",), ("PZT4",), ()]),
        (
            2,
            [
                (),
                ("PZT1", "PZT2"),
                ("PZT3", "PZT4"),
                ("PZT1", "PZT3"),
                ("PZT2", "PZT4"),
                (),
            ],
        ),
        (3, [(), CHANNEL_NAMES, (), CHANNEL_NAMES, ()]),
    ],
)
def test_amplitude_sequences_match_approved_channel_choices(case, active_by_check):
    plan = plan_for(case)
    for at_s, active in zip(check_times(plan, None), active_by_check):
        state = expected_at(plan, at_s)
        assert (
            tuple(
                name for name in CHANNEL_NAMES if state[name]["amplitude_vpp"] > 0.001
            )
            == active
        )
        assert all(
            value["output"] and value["frequency_hz"] == 2500000
            for value in state.values()
        )


def test_only_requested_channel_changes_frequency():
    plan = plan_for(4)
    for at_s, chosen in zip([3, 7, 11, 15], CHANNEL_NAMES):
        state = expected_at(plan, at_s)
        for name in CHANNEL_NAMES:
            assert state[name]["frequency_hz"] == (
                2600000 if name == chosen else 2500000
            )
            assert state[name]["amplitude_vpp"] == 0.001
    assert all(
        state["frequency_hz"] == 2500000 for state in expected_at(plan, 19).values()
    )


def test_combined_case_has_four_distinct_targets_and_restores_all():
    plan = plan_for(5)
    for at_s in (3, 13):
        state = expected_at(plan, at_s)
        assert [state[name]["frequency_hz"] for name in CHANNEL_NAMES] == [
            2520000,
            2540000,
            2560000,
            2580000,
        ]
        assert [state[name]["amplitude_vpp"] for name in CHANNEL_NAMES] == [
            0.02,
            0.04,
            0.06,
            0.08,
        ]
    for at_s in (1, 9, 19):
        assert all(
            state["frequency_hz"] == 2500000 and state["amplitude_vpp"] == 0.001
            for state in expected_at(plan, at_s).values()
        )


def test_cancellation_is_checked_while_all_four_are_active():
    plan = plan_for(6)
    assert check_times(plan, 10) == [1, 3, 9]
    assert all(
        state["output"] and state["amplitude_vpp"] == 0.1
        for state in expected_at(plan, 9).values()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("output", False),
        ("frequency_hz", 2600000),
        ("amplitude_vpp", 0.1),
        ("amplitude_vpp", float("nan")),
    ],
)
def test_unselected_channel_mismatch_is_detected(field, value):
    expected = expected_at(plan_for(1), 3)
    actual = deepcopy(expected)
    actual["PZT4"][field] = value
    problems = compare_readback(expected, actual)
    assert problems and all("PZT4" in problem for problem in problems)
    assert expected["PZT4"] == {
        "output": True,
        "frequency_hz": 2500000,
        "amplitude_vpp": 0.001,
    }


def test_missing_channel_is_not_treated_as_a_pass():
    expected = expected_at(plan_for(2), 7)
    actual = deepcopy(expected)
    del actual["PZT2"]
    assert compare_readback(expected, actual) == ["PZT2 缺少读回"]
    assert compare_readback(expected, expected) == []


def test_readback_rejects_a_window_too_close_to_next_step():
    plan = plan_for(1)
    steps = (plan.steps[0], replace(plan.steps[1], at_s=1.2), *plan.steps[2:])
    with pytest.raises(ValueError, match="稳定读回"):
        check_times(replace(plan, steps=steps), None)
