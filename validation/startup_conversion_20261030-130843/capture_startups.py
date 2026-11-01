"""Capture unchanged replay startups plus a measured Windows/WSL clock bridge."""

import base64
import csv
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATHS = json.loads((HERE / "paths.json").read_text())
ROOT = Path(PATHS["root"])
HOST = Path(PATHS["host"])
CODE = ROOT / "customer_workspace/microbubble_tracking"
PYTHON = ROOT / ".venv/bin/python"
PS = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
WSL = "/mnt/c/Windows/System32/wsl.exe"
PERFROOT = ROOT / "customer_workspace/dependencies/native_perf/installed/usr"
PERF = PERFROOT / "lib/linux-tools-6.8.0-147/perf"
LIB = PERFROOT / "lib/x86_64-linux-gnu"


def windows_path(path):
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def clock_bridge(process, stop, errors):
    """Bound each Windows QPC reply by Linux request and response times."""
    try:
        with (HERE / "clock_bridge.csv").open("x", newline="", buffering=1) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "wsl_before_s",
                    "wsl_after_s",
                    "linux_unix_s",
                    "windows_unix_s",
                    "windows_qpc",
                    "windows_frequency",
                    "roundtrip_ms",
                ]
            )
            while not stop.is_set():
                start = time.perf_counter()
                process.stdin.write("\n")
                process.stdin.flush()
                row = json.loads(process.stdout.readline())
                end = time.perf_counter()
                writer.writerow(
                    [
                        start,
                        end,
                        time.time(),
                        row["unix_s"],
                        row["qpc"],
                        row["frequency"],
                        (end - start) * 1000,
                    ]
                )
                stop.wait(0.1)
    except Exception as error:
        errors.append(repr(error))


def main():
    for name in ("windows_stall_capture.ps1", "windows_stall_capture.wprp"):
        shutil.copy2(CODE / name, HOST / name)
    capture = HOST / "capture"
    arguments = (
        f'-NoProfile -ExecutionPolicy Bypass -File "{windows_path(HOST / "windows_stall_capture.ps1")}" '
        f'-Output "{windows_path(capture)}" -DurationSeconds 90'
    )
    command = (
        "$ErrorActionPreference='Stop'; "
        "Start-Process powershell.exe -Verb RunAs -PassThru -ArgumentList '"
        + arguments.replace("'", "''")
        + "' | Select-Object -ExpandProperty Id"
    )
    encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
    with (HERE / "elevation.log").open("x") as log:
        subprocess.run(
            [PS, "-NoProfile", "-EncodedCommand", encoded],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    deadline = time.monotonic() + 60
    while not (capture / "recording.json").exists():
        if (capture / "error.json").exists() or time.monotonic() > deadline:
            raise RuntimeError("Windows trace did not start")
        time.sleep(0.1)
    print("Windows trace started", flush=True)
    clock_command = r"""
while ($null -ne [Console]::ReadLine()) {
    [Console]::WriteLine((@{
        unix_s=([DateTime]::UtcNow.Ticks - 621355968000000000)/10000000.0
        qpc=[System.Diagnostics.Stopwatch]::GetTimestamp()
        frequency=[System.Diagnostics.Stopwatch]::Frequency
    } | ConvertTo-Json -Compress))
}
"""
    stop = threading.Event()
    errors = []
    bridge = subprocess.Popen(
        [PS, "-NoProfile", "-NonInteractive", "-Command", clock_command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    watcher = threading.Thread(
        target=clock_bridge, args=(bridge, stop, errors), daemon=True
    )
    watcher.start()
    commands = []
    try:
        # Allow clock samples on both sides of every diagnostic startup.
        time.sleep(1.5)
        for index in range(1, 9):
            run = HERE / f"startup_{index:02d}"
            command = [
                WSL,
                "-d",
                "Ubuntu",
                "-u",
                "root",
                "--",
                "env",
                f"LD_LIBRARY_PATH={LIB}",
                str(PERF),
                "record",
                "--clockid",
                "mono",
                "--switch-events",
                "-F",
                "999",
                "-e",
                "cpu-clock",
                "--call-graph",
                "dwarf,4096",
                "-m",
                "512",
                "-o",
                str(HERE / f"startup_{index:02d}.data"),
                "--",
                "runuser",
                "-u",
                "lzy",
                "--",
                "env",
                "-u",
                "LD_LIBRARY_PATH",
                str(PYTHON),
                str(CODE / "paced_replay.py"),
                "--video",
                str(CODE / "data/raw/expB.mp4"),
                "--reference",
                str(CODE / "results/A_offline/background.npy"),
                "--pzt",
                str(CODE / "data/pzt/expB"),
                "--output-dir",
                str(run),
                "--max-frames",
                "60",
                "--opencv-threads",
                "24",
                "--conversion-threads",
                "1",
                "--measure-stages",
                "--diagnose-stalls",
            ]
            commands.append(command)
            (HERE / "replay_commands.json").write_text(json.dumps(commands, indent=2))
            with (HERE / f"startup_{index:02d}.log").open("x") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=25,
                )
            print("Completed startup", index, flush=True)
            if errors:
                raise RuntimeError(str(errors))
        time.sleep(0.3)
    finally:
        stop.set()
        watcher.join(timeout=5)
        if watcher.is_alive():
            bridge.terminate()
            watcher.join(timeout=5)
        else:
            bridge.stdin.close()
        bridge.wait(timeout=5)
        (HERE / "clock_errors.json").write_text(json.dumps(errors))
        (capture / "stop.request").touch()
    deadline = time.monotonic() + 90
    while not (capture / "finished.json").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Windows stop not confirmed")
        time.sleep(0.5)
    assert json.loads((capture / "finished.json").read_text())["stop_exit_code"] == 0
    print("Windows trace stopped", flush=True)


if __name__ == "__main__":
    main()
