"""Bounded Windows/WSL observation of current A/B playback; no device commands."""

import base64
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from customer_workspace.microbubble_tracking.memory_pressure_check import (
    POWERSHELL, windows_path, windows_snapshot,
)


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CODE = ROOT / "customer_workspace/microbubble_tracking"
HOST = Path("/mnt/c/Users/lzy/AppData/Local/Temp") / OUT.name
CAPTURE = HOST / "capture"
PERF = ROOT / "customer_workspace/dependencies/native_perf/installed/usr/lib/linux-tools-6.8.0-147/perf"
LIB = ROOT / "customer_workspace/dependencies/native_perf/installed/usr/lib/x86_64-linux-gnu"


def clock_bridge(stop, errors):
    command = (
        "while ($null -ne [Console]::ReadLine()) { [Console]::WriteLine((@{ "
        "unix_s=([DateTime]::UtcNow.Ticks - 621355968000000000)/10000000.0; "
        "qpc=[System.Diagnostics.Stopwatch]::GetTimestamp(); "
        "frequency=[System.Diagnostics.Stopwatch]::Frequency } | ConvertTo-Json -Compress)) }"
    )
    with (OUT / "clock_bridge.log").open("x") as log:
        process = subprocess.Popen(
            [str(POWERSHELL), "-NoProfile", "-Command", command],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            text=True, bufsize=1,
        )
        try:
            with (OUT / "clock_bridge.csv").open("x", newline="", buffering=1) as file:
                writer = csv.writer(file)
                writer.writerow(["send_s", "receive_s", "unix_s", "qpc", "frequency"])
                while not stop.is_set():
                    started = time.perf_counter()
                    process.stdin.write("clock\n")
                    process.stdin.flush()
                    reply = json.loads(process.stdout.readline())
                    finished = time.perf_counter()
                    writer.writerow([started, finished, reply["unix_s"], reply["qpc"], reply["frequency"]])
                    stop.wait(0.5)
        except Exception as error:
            errors.append(repr(error))
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)


def main():
    HOST.mkdir(exist_ok=False)
    for name in ("windows_stall_capture.ps1", "windows_stall_capture.wprp"):
        shutil.copyfile(CODE / name, HOST / name)
    before = windows_snapshot()
    (OUT / "windows_before.json").write_text(json.dumps(before, indent=2) + "\n")
    argument = (
        '-NoProfile -ExecutionPolicy Bypass -File "'
        + windows_path(HOST / "windows_stall_capture.ps1")
        + '" -Output "' + windows_path(CAPTURE) + '" -DurationSeconds 180'
    )
    command = (
        "$ErrorActionPreference='Stop'; Start-Process powershell.exe -Verb RunAs "
        "-PassThru -ArgumentList '" + argument.replace("'", "''") + "' | Select-Object -ExpandProperty Id"
    )
    encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
    with (OUT / "host_elevation.log").open("x") as log:
        subprocess.run([str(POWERSHELL), "-NoProfile", "-EncodedCommand", encoded],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
    deadline = time.monotonic() + 60
    while not (CAPTURE / "recording.json").exists():
        if (CAPTURE / "error.json").exists() or time.monotonic() > deadline:
            raise RuntimeError("Host recording did not start; see host capture state")
        time.sleep(0.2)
    print("Windows recording started", flush=True)
    stop, errors = threading.Event(), []
    clock = threading.Thread(target=clock_bridge, args=(stop, errors), daemon=True)
    clock.start()
    records = {"host": str(HOST), "duration_limit_s": 180, "runs": []}
    try:
        for label, repeats in (("A", 1), ("B", 6)):
            run = OUT / ("host_" + label)
            observe = ["/mnt/c/Windows/System32/wsl.exe", "-d", "Ubuntu", "-u", "root", "--",
                       sys.executable, str(CODE / "native_stall_watch.py"), str(run),
                       str(OUT / ("host_watch_" + label + ".jsonl")), "--duration", str(repeats * 22 + 15),
                       "--perf", str(PERF), "--perf-lib", str(LIB),
                       "--perf-targets", "consumer", "writer", "--frequency", "999"]
            replay = [sys.executable, "-m", "customer_workspace.microbubble_tracking.paced_replay",
                      "--video", str(CODE / f"data/raw/exp{label}.mp4"),
                      "--reference", str(CODE / "results/B_offline/background.npy"),
                      "--pzt", str(CODE / f"data/pzt/exp{label}"), "--output-dir", str(run),
                      "--repeats", str(repeats), "--opencv-threads", "24", "--conversion-threads", "1", "--diagnostics"]
            record = {"label": label, "observer": observe, "replay": replay}
            records["runs"].append(record)
            (OUT / "host_commands.json").write_text(json.dumps(records, indent=2) + "\n")
            print("START host", label, flush=True)
            with (OUT / f"host_observer_{label}.log").open("x") as native_log, (OUT / f"host_replay_{label}.log").open("x") as replay_log:
                observer = subprocess.Popen(observe, stdout=native_log, stderr=subprocess.STDOUT)
                try:
                    result = subprocess.run(replay, cwd=ROOT, stdout=replay_log, stderr=subprocess.STDOUT,
                                            timeout=repeats * 22 + 20)
                    record["replay_exit"] = result.returncode
                finally:
                    record["observer_exit"] = observer.wait(timeout=25)
            (OUT / "host_commands.json").write_text(json.dumps(records, indent=2) + "\n")
            if record["replay_exit"] or record["observer_exit"]:
                raise RuntimeError("Correlated replay or native observation failed")
            print("DONE host", label, flush=True)
    finally:
        stop.set()
        clock.join(timeout=5)
        (CAPTURE / "stop.request").touch()
        (OUT / "clock_errors.json").write_text(json.dumps(errors) + "\n")
    deadline = time.monotonic() + 60
    while not (CAPTURE / "finished.json").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Host capture has not confirmed shutdown")
        time.sleep(0.2)
    target = OUT / "host_capture"
    target.mkdir()
    for name in ("recording.json", "finished.json", "start.log", "stop.log", "host.etl"):
        shutil.copyfile(CAPTURE / name, target / name)
    print("Windows recording stopped and archived", flush=True)


if __name__ == "__main__":
    main()
