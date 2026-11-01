"""选取 GPU 慢帧对应的 Windows 事件窗口；保留跨系统时间核对的不确定范围。"""

import argparse
import json
import shutil
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from customer_workspace.tools.analysis_host_trace import POWERSHELL
from customer_workspace.tools.gpu_decode_check import save_json
from customer_workspace.tools.gpu_decode_report import read_csv, read_json, stats
from customer_workspace.tools.gpu_decode_root_report import trace_frames
from customer_workspace.wsl_launcher import windows_path
from microbubble_experiment.configuration import ROOT


def clock_mapping(root, etl):
    """校验本机 ETL 头与实测 QPC；中点仅用于选窗口，不作为单向延迟。"""
    with etl.open("rb") as file:
        header = file.read(512)
    origin = struct.unpack_from("<Q", header, 88)[0]
    frequency = struct.unpack_from("<Q", header, 360)[0]
    filetime = struct.unpack_from("<Q", header, 368)[0]
    clocks = [json.loads(line) for line in (root / "clock_bridge.jsonl").open()]
    clocks = [row for row in clocks if 0 < row["receive_s"] - row["send_s"] < 0.004]
    if len(clocks) < 10 or any(row["frequency"] != frequency for row in clocks):
        raise ValueError("Insufficient clock correspondence or unexpected ETL header")
    wsl = np.array([(row["send_s"] + row["receive_s"]) / 2 for row in clocks])
    host = np.array([(row["qpc"] - origin) / frequency * 1000 for row in clocks])
    utc_error = np.array([row["unix_s"] for row in clocks]) - (
        filetime / 1e7 - 11644473600 + host / 1000
    )
    if abs(np.median(utc_error)) > 0.05:
        raise ValueError("ETL origin does not agree with Windows UTC checks")
    return (
        wsl,
        host,
        clocks,
        {
            "origin_qpc": origin,
            "frequency": frequency,
            "utc_check_median_ms": float(np.median(utc_error) * 1000),
            "rtt_ms": stats(
                [(row["receive_s"] - row["send_s"]) * 1000 for row in clocks]
            ),
        },
    )


def prepare(root, output):
    """每次回放选择最长锁定、最长解码、最长同步帧及前一帧。"""
    output.mkdir(exist_ok=False)
    location = read_json(root / "capture_location.json")
    etl = Path(location["capture"]) / "host.etl"
    archived = root / "etl_archive.json"
    if archived.exists():
        etl = Path(read_json(archived)["current"])
    wsl, host, clocks, clock_info = clock_mapping(root, etl)
    report = {"etl": str(etl), "clock": clock_info, "runs": {}, "windows": []}
    for database in sorted(root.glob("profile_*.sqlite")):
        name = database.stem.removeprefix("profile_")
        timings = read_csv(root / f"traced_{name}/frames.csv")
        frames = trace_frames(database, timings)
        save_json(output / f"frames_{name}.json", frames)
        report["runs"][name] = {
            "frames": len(frames),
            "total_ms": stats([row["total_ms"] for row in frames]),
            "max_lock_ms": max(row["lock_ms"] for row in frames),
        }
        picked = set()
        for field in ("lock_ms", "decode_ms", "total_ms"):
            picked.update(
                row["frame"] for row in sorted(frames, key=lambda r: -r[field])[:2]
            )
        picked.update(max(0, frame - 1) for frame in list(picked))
        for index in sorted(picked):
            timing, frame = timings[index], frames[index]
            start = float(timing["decode_start_s"])
            end = start + float(timing["total_ms"]) / 1000
            if not wsl[0] <= start < end <= wsl[-1]:
                raise ValueError("Frame outside measured clock range")
            a, b = np.interp([start, end], wsl, host)
            pos = int(np.searchsorted(wsl, start))
            nearby = clocks[max(0, pos - 1) : pos + 2]
            report["windows"].append(
                {
                    "run": name,
                    "frame": index,
                    "start_ms": float(a),
                    "end_ms": float(b),
                    "clock_half_rtt_max_ms": max(
                        (row["receive_s"] - row["send_s"]) * 500 for row in nearby
                    ),
                    "timing": timing,
                    "trace": frame,
                }
            )
    save_json(output / "windows.json", report)
    print(output / "windows.json", flush=True)


def prepare_native(root, output):
    """原生 Windows 帧自带 QPC；直接对应 ETW 同一时钟，不借用 WSL 时钟。"""
    output.mkdir(exist_ok=False)
    location = read_json(root / "capture_location.json")
    etl = Path(location["capture"]) / "host.etl"
    if (root / "etl_archive.json").exists():
        etl = Path(read_json(root / "etl_archive.json")["current"])
    with etl.open("rb") as file:
        header = file.read(512)
    origin, frequency = (
        struct.unpack_from("<Q", header, 88)[0],
        struct.unpack_from("<Q", header, 360)[0],
    )
    report = {
        "etl": str(etl),
        "clock": {
            "origin_qpc": origin,
            "frequency": frequency,
            "scope": "Same Windows QPC clock; no cross-system clock subtraction",
        },
        "runs": {},
        "windows": [],
    }
    for run in sorted(root.glob("traced_*")):
        if not run.is_dir():
            continue
        name = run.name.removeprefix("traced_")
        environment = read_json(run / "environment.json")
        if environment["qpc_frequency"] != frequency:
            raise ValueError("Native frame clock differs from ETW clock")
        frames = read_csv(run / "frames.csv")
        report["runs"][name] = {
            "frames": len(frames),
            "pid": environment["pid"],
            "tid": environment["tid"],
            "decode_ready_ms": stats([float(f["decode_ready_ms"]) for f in frames]),
        }
        picked = set()
        for key in ("decode_ready_ms", "stream_sync_ms", "late_ms"):
            picked.update(
                int(f["frame"])
                for f in sorted(frames, key=lambda f: -float(f[key]))[:4]
            )
        for index in sorted(picked):
            frame = frames[index]
            report["windows"].append(
                {
                    "run": name,
                    "frame": index,
                    "start_ms": (int(frame["qpc_start"]) - origin) / frequency * 1000,
                    "end_ms": (int(frame["qpc_end"]) - origin) / frequency * 1000,
                    "timing": frame,
                    "pid": environment["pid"],
                    "tid": environment["tid"],
                }
            )
    if not report["windows"]:
        raise ValueError("No native Windows frame records")
    save_json(output / "windows.json", report)
    print(output / "windows.json", flush=True)


def export(root, output, dependency_dir, origins=False):
    """用现有官方 TraceEvent DLL 读取 ETL；DLL 放在 Windows 本地避免 UNC 加载限制。"""
    report = read_json(output / "windows.json")
    location = read_json(root / "capture_location.json")
    host = Path(location["host"])
    libraries = host / f"traceevent_{output.name}"
    libraries.mkdir(exist_ok=False)
    for source in dependency_dir.glob("*.dll"):
        shutil.copyfile(source, libraries / source.name)
    source = ROOT / "customer_workspace/tools/gpu_host_events.cs"
    shutil.copyfile(source, libraries / source.name)
    dll = windows_path(libraries / "Microsoft.Diagnostics.Tracing.TraceEvent.dll")
    reader = windows_path(libraries / source.name)
    destination = host / f"{output.name}_selected_events.tsv"
    starts = ",".join(str(row["start_ms"] - 50) for row in report["windows"])
    ends = ",".join(str(row["end_ms"] + 50) for row in report["windows"])
    script = (
        "$ErrorActionPreference='Stop'\n"
        f"[Reflection.Assembly]::LoadFrom('{dll}') | Out-Null\n"
        f"Add-Type -Path '{reader}' -ReferencedAssemblies '{dll}'\n"
        f"[GpuHostEvents]::Read('{windows_path(Path(report['etl']))}',"
        f"'{windows_path(destination)}',[double[]]@({starts}),[double[]]@({ends}))\n"
    )
    if origins:
        script = (
            "$ErrorActionPreference='Stop'\n"
            f"[Reflection.Assembly]::LoadFrom('{dll}') | Out-Null\n"
            f"Add-Type -Path '{reader}' -ReferencedAssemblies '{dll}'\n"
            f"[GpuHostEvents]::ReadOrigins('{windows_path(Path(report['etl']))}',"
            f"'{windows_path(destination)}')\n"
        )
    script_path = output / "export.ps1"
    script_path.write_text(script, encoding="utf-8")
    command = [
        POWERSHELL,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        windows_path(script_path),
    ]
    with (output / "export.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=log, timeout=180, check=True)
    shutil.move(destination, output / "selected_events.tsv")
    print(output / "selected_events.tsv", flush=True)


def pairs(output):
    """仅匹配同一进程、线程、函数的嵌套起止；跨缺失窗口不拼接。"""
    stacks = defaultdict(list)
    matched = []
    previous = None
    with (output / "selected_events.tsv").open() as file:
        for line in file:
            fields = line.rstrip("\n").split("\t")
            stamp = float(fields[0])
            if previous is not None and stamp - previous > 100:
                stacks.clear()
            previous = stamp
            payload = dict(field.split("=", 1) for field in fields[5:] if "=" in field)
            if fields[1] != "Microsoft-Windows-DxgKrnl" or "Function" not in payload:
                continue
            key = (int(fields[3]), int(fields[4]), payload["Function"])
            if fields[2] == "Profiler/Start":
                stacks[key].append(stamp)
            elif fields[2] == "Profiler/Stop" and stacks[key]:
                start = stacks[key].pop()
                matched.append(
                    {
                        "pid": key[0],
                        "tid": key[1],
                        "name": key[2],
                        "start_ms": start,
                        "end_ms": stamp,
                        "duration_ms": stamp - start,
                    }
                )
    save_json(output / "host_functions.json", matched)
    print("Matched functions:", len(matched), flush=True)
    if not matched:
        raise RuntimeError(
            "No GPU function events were captured; kernel-only evidence retained"
        )


def adapter_locks(output):
    """将同一显卡锁的请求、获得和释放配对，区分等待与实际持有。"""
    stacks = defaultdict(list)
    locks, acpi = [], []
    previous = None
    with (output / "selected_events.tsv").open() as file:
        for line in file:
            fields = line.rstrip("\n").split("\t")
            stamp = float(fields[0])
            if previous is not None and stamp - previous > 100:
                stacks.clear()
            previous = stamp
            if fields[1] != "Microsoft-Windows-DxgKrnl":
                continue
            data = dict(field.split("=", 1) for field in fields[5:] if "=" in field)
            event, pid, tid = fields[2], int(fields[3]), int(fields[4])
            if event == "ACPIEvent" or event.startswith("DdiNotifyAcpiEvent/"):
                acpi.append(
                    {"time_ms": stamp, "event": event, "pid": pid, "tid": tid, **data}
                )
            adapter = data.get("DxgAdapter")
            key = (pid, tid, adapter)
            if event == "TraceAdapterLock/Start":
                stacks[key].append(
                    {
                        "start_ms": stamp,
                        "pid": pid,
                        "tid": tid,
                        "adapter": adapter,
                        "exclusive": data["Exclusive"] == "True",
                    }
                )
            elif (
                event == "TraceAdapterLock"
                and data.get("Acquired") == "True"
                and stacks[key]
            ):
                stacks[key][-1]["acquired_ms"] = stamp
            elif event == "TraceAdapterLock/Stop" and stacks[key]:
                row = stacks[key].pop()
                if "acquired_ms" in row:
                    row.update(
                        end_ms=stamp,
                        wait_ms=row["acquired_ms"] - row["start_ms"],
                        held_ms=stamp - row["acquired_ms"],
                    )
                    locks.append(row)
    save_json(output / "adapter_locks.json", locks)
    save_json(output / "acpi_events.json", acpi)
    print("Adapter locks:", len(locks), "ACPI events:", len(acpi), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traceevent-dir", type=Path, required=True)
    parser.add_argument(
        "--native", action="store_true", help="读取 Windows 原生 QPC 帧记录"
    )
    parser.add_argument(
        "--origins", action="store_true", help="读取全程 ACPI、WMI 与显卡锁事件"
    )
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    if args.native:
        prepare_native(root, output)
    else:
        prepare(root, output)
    export(root, output, args.traceevent_dir.resolve(), args.origins)
    pairs(output)
    adapter_locks(output)


if __name__ == "__main__":
    main()
