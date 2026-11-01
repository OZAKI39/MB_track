"""WSL 启动、路径转换和取消转交测试；所有子进程均被替换。"""

import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from customer_workspace import wsl_launcher


@pytest.fixture
def launch_environment(tmp_path, monkeypatch):
    project = tmp_path / "含空格的 project"
    (project / "configs").mkdir(parents=True)
    python = tmp_path / "Windows Python" / "python.exe"
    python.parent.mkdir()
    python.touch()
    (project / "configs/machine.local.json").write_text(
        json.dumps({"windows_afg_python": str(python)}), encoding="utf-8"
    )
    monkeypatch.setattr(wsl_launcher, "is_wsl", lambda: True)
    monkeypatch.setattr(
        wsl_launcher, "windows_path", lambda path: "WIN:" + str(path.resolve())
    )
    wait = Mock(return_value=13)
    monkeypatch.setattr(wsl_launcher, "wait_for_windows", wait)
    return project, python, wait


def test_launch_preserves_literal_paths_and_returns_exit_code(launch_environment):
    project, python, wait = launch_environment
    plan = project / "计划 & sample.json"
    config = project / "customer_workspace/configs/afg_video.json"
    output = project / "结果 空格"
    assert wsl_launcher.launch_from_wsl(project, plan, config, output) == 13
    command, cwd = wait.call_args.args
    assert command == [
        str(python),
        "-X",
        "utf8",
        "-u",
        "-B",
        "-m",
        "customer_workspace.experiment_runner",
        "WIN:" + str(plan),
        "--config",
        "WIN:" + str(config),
        "--output-dir",
        "WIN:" + str(output),
        "--launcher-stdin",
    ]
    assert cwd == project


def test_default_output_is_left_to_windows_local_time(launch_environment):
    project, _, wait = launch_environment
    wsl_launcher.launch_from_wsl(project, Path("plan.json"), Path("afg.json"))
    assert "--output-dir" not in wait.call_args.args[0]


def test_direct_execution_always_watches_cancellation(launch_environment):
    project, _, wait = launch_environment
    wsl_launcher.launch_from_wsl(project, Path("plan.json"), Path("afg.json"))
    command = wait.call_args.args[0]
    assert "--execute" not in command
    assert "--launcher-stdin" in command


def test_repeat_count_is_forwarded_to_windows(launch_environment):
    project, _, wait = launch_environment
    wsl_launcher.launch_from_wsl(
        project, Path("plan.json"), Path("afg.json"), repeat=10
    )
    command = wait.call_args.args[0]
    assert command[command.index("--repeat") + 1] == "10"
    assert "--launcher-stdin" in command


def test_benchmark_uses_same_launcher_and_cancellation(launch_environment):
    project, _, wait = launch_environment
    wsl_launcher.launch_from_wsl(
        project,
        Path("plan.json"),
        Path("afg.json"),
        module="customer_workspace.timing_benchmark",
        extra_args=["--method", "schedule", "--interval-ms", "20", "--count", "500"],
    )
    command = wait.call_args.args[0]
    assert command[command.index("-m") + 1] == "customer_workspace.timing_benchmark"
    assert command[-7:] == [
        "--method",
        "schedule",
        "--interval-ms",
        "20",
        "--count",
        "500",
        "--launcher-stdin",
    ]


def test_explicit_python_overrides_machine_config(launch_environment, tmp_path):
    project, _, wait = launch_environment
    override = tmp_path / "other.exe"
    override.touch()
    (project / "configs/machine.local.json").write_text("invalid json")
    wsl_launcher.launch_from_wsl(
        project, Path("plan"), Path("config"), windows_python=override
    )
    assert wait.call_args.args[0][0] == str(override)


@pytest.mark.parametrize(
    "contents",
    [
        None,
        {},
        {"windows_afg_python": ""},
        {"windows_afg_python": "/missing/python.exe"},
    ],
)
def test_missing_configuration_never_starts_process(launch_environment, contents):
    project, _, wait = launch_environment
    config = project / "configs/machine.local.json"
    if contents is None:
        config.unlink()
    else:
        config.write_text(json.dumps(contents))
    with pytest.raises((ValueError, FileNotFoundError)):
        wsl_launcher.launch_from_wsl(project, Path("plan"), Path("config"))
    wait.assert_not_called()


def test_existing_output_never_starts_process(launch_environment):
    project, _, wait = launch_environment
    with pytest.raises(FileExistsError):
        wsl_launcher.launch_from_wsl(project, Path("plan"), Path("config"), project)
    wait.assert_not_called()


def test_non_wsl_linux_is_rejected(launch_environment, monkeypatch):
    project, _, wait = launch_environment
    monkeypatch.setattr(wsl_launcher, "is_wsl", lambda: False)
    with pytest.raises(OSError, match="不是 WSL"):
        wsl_launcher.launch_from_wsl(project, Path("plan"), Path("config"))
    wait.assert_not_called()


def test_path_conversion_uses_absolute_argument_without_shell(tmp_path, monkeypatch):
    path = tmp_path / "名称 with spaces &.json"
    run = Mock(return_value=Mock(stdout="C:\\test path\\plan.json\n"))
    monkeypatch.setattr(wsl_launcher.subprocess, "run", run)
    assert wsl_launcher.windows_path(path) == "C:\\test path\\plan.json"
    assert run.call_args.args[0] == ["wslpath", "-w", str(path)]
    assert "shell" not in run.call_args.kwargs
    assert run.call_args.kwargs["timeout"] == 5


def test_empty_converted_path_is_rejected(monkeypatch):
    monkeypatch.setattr(
        wsl_launcher.subprocess, "run", Mock(return_value=Mock(stdout="\n"))
    )
    with pytest.raises(OSError, match="无法"):
        wsl_launcher.windows_path(Path("plan.json"))


@pytest.mark.parametrize("interrupts", [0, 1, 2])
def test_parent_wait_forwards_cancel_once_and_never_kills(
    monkeypatch, tmp_path, interrupts
):
    process = Mock()
    process.wait.side_effect = [KeyboardInterrupt() for _ in range(interrupts)] + [130]
    popen = Mock(return_value=process)
    monkeypatch.setattr(wsl_launcher.subprocess, "Popen", popen)
    assert (
        wsl_launcher.wait_for_windows(["python.exe", "-m", "example"], tmp_path) == 130
    )
    if interrupts:
        process.stdin.write.assert_called_once_with("cancel\n")
        process.stdin.flush.assert_called_once()
    else:
        process.stdin.write.assert_not_called()
    process.stdin.close.assert_called_once()
    process.kill.assert_not_called()
    process.terminate.assert_not_called()
    assert popen.call_args.kwargs["start_new_session"] is True
    assert "shell" not in popen.call_args.kwargs


def test_broken_cancel_pipe_does_not_kill_or_mask_exit_status(monkeypatch, tmp_path):
    process = Mock()
    process.wait.side_effect = [KeyboardInterrupt(), 1]
    process.stdin.write.side_effect = BrokenPipeError("closed")
    process.stdin.close.side_effect = BrokenPipeError("closed")
    monkeypatch.setattr(wsl_launcher.subprocess, "Popen", Mock(return_value=process))
    assert wsl_launcher.wait_for_windows(["python.exe"], tmp_path) == 1
    process.kill.assert_not_called()
    process.terminate.assert_not_called()


@pytest.mark.parametrize("message", ["cancel\n", "", "unexpected\n"])
def test_windows_receives_cancellation_or_parent_disconnect(monkeypatch, message):
    monkeypatch.setattr(wsl_launcher.sys, "stdin", io.StringIO(message))
    cancel = Mock()
    wsl_launcher.watch_launcher_input(cancel)
    cancel.assert_called_once()


def test_input_failure_requests_cancellation(monkeypatch):
    stream = Mock()
    stream.readline.side_effect = OSError("pipe lost")
    monkeypatch.setattr(wsl_launcher.sys, "stdin", stream)
    cancel = Mock()
    wsl_launcher.watch_launcher_input(cancel)
    cancel.assert_called_once()
