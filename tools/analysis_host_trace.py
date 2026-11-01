"""限时 Windows 系统记录与 WSL 时间核对；不更改系统设置，不暂停其他程序。"""

import argparse
import base64
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from customer_workspace.streamcopy_recording import write_json
from customer_workspace.wsl_launcher import windows_path
from microbubble_experiment.configuration import ROOT

POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def clock_bridge(output, stop):
    """Windows 回答时间位于 WSL 请求/回复之间；保留整个区间，不假设双向等时。"""
    script = (
        "while ($null -ne [Console]::ReadLine()) { "
        "[Console]::WriteLine((@{ unix_s=([DateTime]::UtcNow.Ticks - 621355968000000000)/10000000.0; "
        "qpc=[System.Diagnostics.Stopwatch]::GetTimestamp(); "
        "frequency=[System.Diagnostics.Stopwatch]::Frequency } | ConvertTo-Json -Compress)) }"
    )
    with (
        (output / "clock_bridge.log").open("x") as log,
        (output / "clock_bridge.jsonl").open("x", buffering=1) as rows,
    ):
        process = subprocess.Popen(
            [POWERSHELL, "-NoProfile", "-Command", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )
        try:
            while not stop.is_set():
                started = time.perf_counter()
                process.stdin.write("clock\n")
                process.stdin.flush()
                row = json.loads(process.stdout.readline())
                row.update(send_s=started, receive_s=time.perf_counter())
                rows.write(json.dumps(row) + "\n")
                stop.wait(0.25)
        finally:
            process.stdin.close()
            process.wait(timeout=5)


def run(root, simulator):
    host = Path("/mnt/c/Users/lzy/AppData/Local/Temp") / root.name
    host.mkdir(exist_ok=False)
    for name in ("windows_stall_capture.ps1", "windows_stall_capture.wprp"):
        shutil.copyfile(
            ROOT / "customer_workspace/microbubble_tracking" / name, host / name
        )
    capture = host / "capture"
    arguments = f'-NoProfile -ExecutionPolicy Bypass -File "{windows_path(host / "windows_stall_capture.ps1")}" -Output "{windows_path(capture)}" -DurationSeconds 180'
    script = (
        "$ErrorActionPreference='Stop'; Start-Process powershell.exe -Verb RunAs -PassThru -ArgumentList '"
        + arguments.replace("'", "''")
        + "' | Select-Object -ExpandProperty Id"
    )
    command = [
        POWERSHELL,
        "-NoProfile",
        "-EncodedCommand",
        base64.b64encode(script.encode("utf-16le")).decode("ascii"),
    ]
    write_json(
        root / "host_capture_location.json",
        {
            "host": str(host),
            "capture": str(capture),
            "duration_limit_s": 180,
            "elevation_command": command,
        },
    )
    with (root / "elevation.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=log, check=True, timeout=90)
    deadline = time.perf_counter() + 45
    while not (capture / "recording.json").exists():
        if (capture / "error.json").exists() or time.perf_counter() > deadline:
            raise RuntimeError("Windows system recording did not start")
        time.sleep(0.2)
    print("HOST RECORDING (automatic stop <= 180 s)", flush=True)
    stop = threading.Event()
    clock = threading.Thread(
        target=clock_bridge, args=(root, stop), name="diagnostic-clock"
    )
    clock.start()
    try:
        # 给独立时钟进程留出启动时间，尽量在首帧之前得到核对记录。
        time.sleep(2)
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "customer_workspace.tools.analysis_stall_check",
                "native",
                str(root),
                "--simulator-run",
                str(simulator),
            ],
            cwd=ROOT,
            check=True,
        )
    finally:
        (capture / "stop.request").touch()
        stop.set()
        clock.join(10)
        deadline = time.perf_counter() + 90
        while (
            not (capture / "finished.json").exists() and time.perf_counter() < deadline
        ):
            time.sleep(0.25)
        if not (capture / "finished.json").exists():
            raise RuntimeError(
                "Windows system recording did not finish; inspect the host output"
            )
        state = json.loads((capture / "finished.json").read_text(encoding="utf-8-sig"))
        write_json(root / "host_finished.json", state)
        if state["stop_exit_code"]:
            raise RuntimeError("Windows recording could not be saved")
        print("HOST RECORDING CLOSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--simulator-run", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.simulator_run.resolve())


if __name__ == "__main__":
    main()
