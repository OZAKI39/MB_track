"""检查时间表和命令行转交；不创建 AFG 模拟会话，也不触发真实输出。"""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from customer_workspace import experiment_runner
from customer_workspace.experiment_runner import ExperimentStep, load_plan
from microbubble_experiment.afg import DEFAULT_CONFIG, ChannelSettings, load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "configs/pzt1_pulse.json"


@pytest.fixture(autouse=True)
def block_hardware_and_process_launch(monkeypatch):
    """普通测试拒绝连接设备或启动子进程，避免默认实机入口被误调用。"""

    def unexpected_call(*args, **kwargs):
        raise AssertionError("普通测试不能连接设备或启动 Windows 实验")

    monkeypatch.setattr(experiment_runner, "Rig", unexpected_call)
    monkeypatch.setattr(experiment_runner, "TimedRig", unexpected_call)
    monkeypatch.setattr(experiment_runner, "launch_from_wsl", unexpected_call)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def plan(config):
    return load_plan(EXAMPLE, config)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True, "14"])
def test_invalid_duration_is_rejected(config, plan, duration):
    with pytest.raises((ValueError, TypeError)):
        replace(plan, duration_s=duration).validate(config)


@pytest.mark.parametrize(
    "steps",
    [
        (),
        (ExperimentStep(1, "IDLE"),),
        (ExperimentStep(0, "ACTIVE", ("PZT1",)),),
        (ExperimentStep(0, "IDLE"), ExperimentStep(0, "IDLE")),
        (ExperimentStep(0, "IDLE"), ExperimentStep(-1, "IDLE")),
        (ExperimentStep(0, "IDLE"), ExperimentStep(14, "IDLE")),
        (
            ExperimentStep(0, "IDLE"),
            ExperimentStep(5, "IDLE"),
            ExperimentStep(4, "IDLE"),
        ),
        (ExperimentStep(0, "IDLE"), ExperimentStep(float("nan"), "IDLE")),
        (ExperimentStep(0, "IDLE"), ExperimentStep(True, "IDLE")),
        (ExperimentStep(0, "IDLE"), ExperimentStep(5, "ACTIVE", ("PZT5",))),
        (ExperimentStep(0, "IDLE"), ExperimentStep(5, "ACTIVE", ("PZT1", "PZT1"))),
        (ExperimentStep(0, "IDLE"), ExperimentStep(5, "ACTIVE")),
        (ExperimentStep(0, "IDLE"), ExperimentStep(5, "IDLE", ("PZT1",))),
        (ExperimentStep(0, "IDLE"), ExperimentStep(5, "STOP")),
        (ExperimentStep(0, "IDLE"), ExperimentStep(5, "UNKNOWN")),
    ],
)
def test_invalid_steps_are_rejected(config, plan, steps):
    with pytest.raises((ValueError, TypeError)):
        replace(plan, steps=steps + (ExperimentStep(14, "STOP"),)).validate(config)


def test_stop_is_required_at_end(config, plan):
    with pytest.raises(ValueError, match="最后一步"):
        replace(plan, steps=plan.steps[:-1]).validate(config)
    with pytest.raises(ValueError, match="最后一步"):
        replace(plan, duration_s=15).validate(config)


def test_sample_plan_and_only_changed_channels(config, plan):
    plan.validate(config)
    assert [step.state for step in plan.steps] == [
        "IDLE",
        "ACTIVE",
        "IDLE",
        "IDLE",
        "STOP",
    ]
    current = dict.fromkeys(experiment_runner.CHANNEL_NAMES, "IDLE")
    assert plan.steps[0].changes_from(current) == []
    assert plan.steps[1].changes_from(current) == [("PZT1", "ACTIVE")]
    current["PZT1"] = "ACTIVE"
    assert plan.steps[1].changes_from(current) == []
    assert plan.steps[2].changes_from(current) == [("PZT1", "IDLE")]


@pytest.mark.parametrize("combined", [False, True])
def test_frequency_plans_keep_two_commands_at_one_planned_time(config, combined):
    filename = "pzt1_frequency_amplitude.json" if combined else "pzt1_frequency.json"
    plan = load_plan(EXAMPLE.with_name(filename), config)
    assert [step.at_s for step in plan.steps] == [0, 5, 6, 9, 14]
    assert plan.steps[1].frequencies_hz == {"PZT1": 2600000}
    assert plan.steps[2].frequencies_hz == {"PZT1": 2500000}
    assert (plan.steps[1].state == "ACTIVE") is combined
    assert plan.steps[2].state == "IDLE"


@pytest.mark.parametrize(
    "frequencies",
    [
        {"PZT5": 2600000},
        {"PZT1": 0},
        {"PZT1": -1},
        {"PZT1": 10000001},
        {"PZT1": float("nan")},
        {"PZT1": float("inf")},
        {"PZT1": True},
        {"PZT1": "2600000"},
        [],
        None,
    ],
)
def test_invalid_step_frequencies_are_rejected(config, plan, frequencies):
    steps = list(plan.steps)
    steps[1] = replace(steps[1], frequencies_hz=frequencies)
    with pytest.raises((ValueError, TypeError)):
        replace(plan, steps=tuple(steps)).validate(config)


def test_stop_cannot_contain_a_frequency_change(config, plan):
    steps = (
        *plan.steps[:-1],
        replace(plan.steps[-1], frequencies_hz={"PZT1": 2600000}),
    )
    with pytest.raises(ValueError, match="STOP"):
        replace(plan, steps=steps).validate(config)


def test_lower_previous_active_channels_before_raising_new_ones():
    current = {"PZT1": "IDLE", "PZT2": "ACTIVE", "PZT3": "IDLE", "PZT4": "ACTIVE"}
    step = ExperimentStep(5, "ACTIVE", ("PZT1", "PZT4"))
    assert step.changes_from(current) == [("PZT2", "IDLE"), ("PZT1", "ACTIVE")]


@pytest.mark.parametrize(
    "settings",
    [
        ChannelSettings(amplitude_vpp=0.801),
        ChannelSettings(frequency_hz=10_000_001),
        ChannelSettings(mode="burst"),
        ChannelSettings(amplitude_vpp=0.001),
        ChannelSettings(amplitude_vpp=0.0015),
    ],
)
def test_driver_limits_and_continuous_mode_are_checked(config, plan, settings):
    modified = dict(plan.settings, PZT4=settings)
    with pytest.raises(ValueError):
        replace(plan, settings=modified).validate(config)


def test_all_four_settings_are_required(config, plan):
    with pytest.raises(ValueError):
        replace(plan, settings={"PZT1": plan.settings["PZT1"]}).validate(config)


@pytest.mark.parametrize(
    "change",
    [
        "unknown_top_key",
        "unknown_step_key",
        "missing_voltage",
        "string_outputs",
        "unknown_setting",
    ],
)
def test_json_mistakes_are_rejected(config, tmp_path, change):
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    if change == "unknown_top_key":
        raw["duraton_s"] = 14
    elif change == "unknown_step_key":
        raw["steps"][0]["time"] = 0
    elif change == "missing_voltage":
        del raw["settings"]["PZT1"]["amplitude_vpp"]
    elif change == "string_outputs":
        raw["steps"][1]["active_channels"] = "PZT1"
    else:
        raw["settings"]["PZT1"]["voltage"] = 0.1
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_plan(path, config)


def test_old_switch_plan_is_not_silently_reinterpreted(config, tmp_path):
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["steps"] = [{"at_s": 0, "outputs_on": []}]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="旧 outputs_on"):
        load_plan(path, config)


def test_invalid_plan_stops_before_launch(tmp_path, capsys):
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")
    assert experiment_runner.main([str(path)]) == 1
    captured = capsys.readouterr()
    assert "计划必须包含" in captured.out
    assert "Traceback (most recent call last)" in captured.err
    assert "ValueError: 计划必须包含" in captured.err


def test_cli_help_does_not_connect(capsys):
    with pytest.raises(SystemExit) as exc:
        experiment_runner.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "直接控制真实仪器" in help_text
    assert "--execute" not in help_text


def test_removed_execute_argument_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        experiment_runner.main([str(EXAMPLE), "--execute"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --execute" in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="此项检查 WSL/Linux 的启动转交")
def test_cli_directly_delegates_execution(monkeypatch, tmp_path):
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return 7

    monkeypatch.setattr(experiment_runner, "launch_from_wsl", launch)
    result = experiment_runner.main(
        [
            str(EXAMPLE),
            "--output-dir",
            str(tmp_path / "result"),
            "--windows-python",
            "/mnt/f/custom/python.exe",
        ]
    )
    assert result == 7
    assert len(calls) == 1
    assert calls[0][0] == (
        experiment_runner.PROJECT_ROOT,
        EXAMPLE,
        DEFAULT_CONFIG,
        tmp_path / "result",
        Path("/mnt/f/custom/python.exe"),
    )
    assert calls[0][1] == {"repeat": 1}


@pytest.mark.parametrize("repeat", ["0", "-1"])
def test_invalid_repeat_is_rejected_before_launch(repeat):
    with pytest.raises(SystemExit) as exc:
        experiment_runner.main([str(EXAMPLE), "--repeat", repeat])
    assert exc.value.code == 2
