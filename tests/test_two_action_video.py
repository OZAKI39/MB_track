"""双动作计划和报告的离线检查，不连接或模拟仪器。"""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from customer_workspace.afg_video_check import load_video_plan
from customer_workspace.afg_video_report import check_plan_execution
from customer_workspace.four_pzt_validation import check_times

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PLAN = CONFIGS / "two_action_video_20s.json"


def expected_actions():
    return [
        {
            "kind": "amplitude",
            "planned_s": at_s,
            "channel": channel,
            "state": state,
            "amplitude_vpp": amplitude,
        }
        for at_s, channel, state, amplitude in (
            (4, "PZT1", "ACTIVE", 0.8),
            (8, "PZT1", "IDLE", 0.001),
            (8, "PZT3", "ACTIVE", 0.8),
            (12, "PZT3", "IDLE", 0.001),
            (12, "PZT1", "ACTIVE", 0.8),
            (16, "PZT1", "IDLE", 0.001),
        )
    ]


def stable_checks():
    return [{"step_at_s": at_s, "passed": True} for at_s in (0, 4, 8, 12, 16)]


def test_two_actions_keep_all_previous_settings():
    plan = load_video_plan(PLAN)
    previous = load_video_plan(CONFIGS / "four_pzt_video_20s.json")
    assert plan.settings == previous.settings
    assert [s.active_channels for s in plan.steps] == [
        (),
        ("PZT1",),
        ("PZT3",),
        ("PZT1",),
        (),
        (),
    ]
    assert [s.at_s for s in plan.steps] == [0, 4, 8, 12, 16, 20]
    assert check_times(plan, None) == [1, 5, 9, 13, 17]
    assert all(not s.frequencies_hz for s in plan.steps)


def test_six_actions_and_five_checks_pass_for_the_new_plan():
    report = check_plan_execution(
        asdict(load_video_plan(PLAN)), expected_actions(), stable_checks()
    )
    assert report["passed"]
    assert len(report["expected_actions"]) == 6


@pytest.mark.parametrize(
    "change",
    [
        "wrong_channel",
        "wrong_amplitude",
        "wrong_time",
        "missing_action",
        "reordered_actions",
        "duplicate_check",
        "failed_check",
    ],
)
def test_correct_counts_cannot_hide_wrong_actions_or_checks(change):
    actions, checks = expected_actions(), stable_checks()
    if change == "wrong_channel":
        actions[2]["channel"] = "PZT2"
    elif change == "wrong_amplitude":
        actions[2]["amplitude_vpp"] = 0.1
    elif change == "wrong_time":
        actions[2]["planned_s"] = 9
    elif change == "missing_action":
        actions.pop()
    elif change == "reordered_actions":
        actions[1], actions[2] = actions[2], actions[1]
    elif change == "duplicate_check":
        checks[2] = copy.deepcopy(checks[1])
    else:
        checks[2]["passed"] = False
    assert not check_plan_execution(asdict(load_video_plan(PLAN)), actions, checks)[
        "passed"
    ]


def test_old_four_channel_report_is_still_valid():
    # 已有实测文件只作回归检查；不把历史数值当作新实验结论。
    base = CONFIGS.parent / "validation/avsync_20261025-161356_rerun"
    if not base.exists():
        pytest.skip("本机历史验收文件不存在")
    actions = json.loads((base / "report_normal/link_report.json").read_text())[
        "actions"
    ]
    checks = json.loads((base / "normal/afg_link_summary.json").read_text())["checks"]
    result = check_plan_execution(
        asdict(load_video_plan(CONFIGS / "four_pzt_video_20s.json")), actions, checks
    )
    assert result["passed"]
    assert len(result["expected_actions"]) == 12
    assert len(result["expected_check_steps_s"]) == 7


def test_frozen_plan_keeps_the_validated_settings(tmp_path):
    plan = load_video_plan(PLAN)
    frozen = tmp_path / "selected_plan.json"
    frozen.write_text(json.dumps(asdict(plan)))
    assert load_video_plan(frozen) == plan


@pytest.mark.parametrize(
    "change", ["duration", "frequency_change", "too_short_for_readback", "over_voltage"]
)
def test_unsupported_plan_is_rejected_before_device_start(tmp_path, change):
    raw = json.loads(PLAN.read_text())
    if change == "duration":
        raw["duration_s"] = raw["steps"][-1]["at_s"] = 19
    elif change == "frequency_change":
        raw["steps"][1]["frequencies_hz"] = {"PZT1": 2600000}
    elif change == "too_short_for_readback":
        raw["steps"][1]["at_s"] = 1
    else:
        raw["settings"]["PZT1"]["amplitude_vpp"] = 0.9
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_video_plan(path)
