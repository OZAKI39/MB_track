"""用同一录像测内存压力与转换停顿；只观测，不关闭应用或修改系统配置。

Windows 总缺页计数含文件读取，不能冒充 WSL 页文件读取次数；后者需要 WPR。
Linux 与 Windows 各自保留采样时间，禁止直接相减两侧的程序计时器。
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
CODE = Path(__file__).resolve().parent


def windows_path(path):
    """转换显式路径，不拼接 shell 命令。"""
    return subprocess.check_output(
        ["wslpath", "-w", str(path.absolute())], text=True
    ).strip()


def windows_snapshot():
    """记录物理内存、应用占用和 WSL 配置；不改变配置。"""
    command = r"""
    $ErrorActionPreference = 'Stop'
    $os = Get-CimInstance Win32_OperatingSystem
    $path = Join-Path $env:USERPROFILE '.wslconfig'
    $config = if (Test-Path -LiteralPath $path) {
        [System.IO.File]::ReadAllText($path)
    } else { $null }
    $processes = @(Get-Process | Sort-Object WorkingSet64 -Descending |
        Select-Object -First 25 Id, ProcessName, WorkingSet64, PrivateMemorySize64)
    [ordered]@{
        unix_ms = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        total_physical_bytes = [long]$os.TotalVisibleMemorySize * 1024
        available_bytes = [long]$os.FreePhysicalMemory * 1024
        processes = $processes
        wsl_config_path = $path
        wsl_config = $config
    } | ConvertTo-Json -Depth 4 -Compress
    """
    result = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def linux_sample():
    """全机内存和换页累计数，并单独辨认 Pylance；不保存进程命令行。"""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    processes = []
    for process in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
        try:
            info = process.info
            rss = info["memory_info"].rss
            pylance = any(
                "/extensions/ms-python.vscode-pylance-" in argument
                and argument.endswith("/dist/server.bundle.js")
                for argument in (info["cmdline"] or [])
            )
            if rss >= 100 * 1024**2 or pylance:
                processes.append(
                    {
                        "pid": info["pid"],
                        "name": info["name"],
                        "rss_bytes": rss,
                        "pylance": pylance,
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue
    counters = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, value = line.split()
        if key in ("pgmajfault", "pswpin", "pswpout", "oom_kill"):
            counters[key] = int(value)
    return {
        "perf_counter_s": time.perf_counter(),
        "unix_s": time.time(),
        "total_bytes": memory.total,
        "available_bytes": memory.available,
        "used_bytes": memory.used,
        "cached_bytes": memory.cached,
        "swap_used_bytes": swap.used,
        "vmstat": counters,
        "memory_pressure": Path("/proc/pressure/memory").read_text(),
        "processes": sorted(processes, key=lambda row: -row["rss_bytes"]),
    }


def watch_linux(path, stop, errors):
    """每秒记录一次；监测失败必须由主线程报告。"""
    try:
        with path.open("x", encoding="utf-8", buffering=1) as file:
            while not stop.is_set():
                file.write(json.dumps(linux_sample()) + "\n")
                stop.wait(1.0)
    except Exception as error:  # noqa: BLE001 -- 线程错误交回主线程，不能静默缺测。
        errors.append(repr(error))


def run_replay(command, log, timeout_s):
    """超时或取消先让回放正常收尾；必要时仅终止本次回放的进程组。"""
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_s)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
        raise


def build_replay_command(args, output):
    """内存观察不隐式开启逐帧诊断；调用者必须明确传入 diagnostics=True。"""
    command = [
        sys.executable,
        str(CODE / "paced_replay.py"),
        "--video",
        str(args.video.absolute()),
        "--reference",
        str(args.reference.absolute()),
        "--pzt",
        str(args.pzt.absolute()),
        "--output-dir",
        str(output / "replay"),
        "--repeats",
        str(args.repeats),
        "--opencv-threads",
        "24",
        "--conversion-threads",
        "1",
    ]
    if getattr(args, "diagnostics", False):
        command.append("--diagnostics")
    return command


def run_check(args):
    """运行现有回放，前后保留主机快照；不另建分析或保存链路。"""
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=False)
    settings = vars(args).copy()
    settings["diagnostics"] = getattr(args, "diagnostics", False)
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in settings.items()
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    before = windows_snapshot()
    (output / "windows_before.json").write_text(json.dumps(before, indent=2) + "\n")
    stop = threading.Event()
    errors = []
    watcher = threading.Thread(
        target=watch_linux,
        args=(output / "linux_memory.jsonl", stop, errors),
        name="memory-observer",
        daemon=True,
    )
    host_command = [
        str(POWERSHELL),
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        windows_path(CODE / "host_stall_watch.ps1"),
        "-Output",
        windows_path(output / "windows_memory.jsonl"),
        "-DurationSeconds",
        str(args.duration_limit_s),
        "-StopFile",
        windows_path(output / "stop_host.request"),
    ]
    replay_command = build_replay_command(args, output)
    (output / "commands.json").write_text(
        json.dumps(
            {
                "host": host_command,
                "replay": replay_command,
            },
            indent=2,
        )
        + "\n"
    )
    report = {"status": "failed", "label": args.label}
    with (output / "host.log").open("x") as host_log:
        host = subprocess.Popen(host_command, stdout=host_log, stderr=subprocess.STDOUT)
        watcher.start()
        try:
            # 给 Windows 观测程序一个有界启动机会，避免基线没有内存记录。
            deadline = time.monotonic() + 30
            while not (output / "windows_memory.jsonl").exists():
                if host.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Windows observer did not start; see host.log")
                time.sleep(0.1)
            report["replay_start_s"] = time.perf_counter()
            with (output / "replay.log").open("x") as log:
                returncode = run_replay(replay_command, log, args.duration_limit_s)
            report["replay_end_s"] = time.perf_counter()
            report["replay_exit_code"] = returncode
            if returncode != 0:
                raise RuntimeError("Replay failed; see replay.log")
            report["status"] = "completed"
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            stop.set()
            watcher.join(timeout=5)
            (output / "stop_host.request").touch(exist_ok=False)
            try:
                host.wait(timeout=20)
            except subprocess.TimeoutExpired:
                host.terminate()  # 只结束本工具启动的只读计数器进程。
                host.wait(timeout=5)
                errors.append("Windows observer exceeded shutdown deadline")
            report["host_exit_code"] = host.returncode
            report["observer_errors"] = errors
            if errors or host.returncode != 0:
                report["status"] = "failed"
            (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    after = windows_snapshot()
    (output / "windows_after.json").write_text(json.dumps(after, indent=2) + "\n")
    if report["status"] != "completed":
        raise RuntimeError("Memory observation failed; see run.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pzt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration-limit-s", type=int, default=150)
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Opt in to detailed per-frame timing; otherwise keep essential replay records",
    )
    args = parser.parse_args()
    if args.repeats < 1 or not 30 <= args.duration_limit_s <= 3600:
        parser.error("Use positive repeats and a 30–3600 second limit")
    print(json.dumps(run_check(args), indent=2))


if __name__ == "__main__":
    main()
