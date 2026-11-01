"""将限时 Windows 记录与慢帧对应；保留时钟误差，不声称逐线程跨系统精确同步。"""

import argparse
import csv
import json
import re
import struct
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from customer_workspace.streamcopy_recording import write_json
from customer_workspace.tools.analysis_stall_report import collect_run
from customer_workspace.tools.rtsp_tracking_report import read_json
from customer_workspace.wsl_launcher import windows_path

XPERF = (
    "/mnt/c/Program Files (x86)/Windows Kits/10/Windows Performance Toolkit/xperf.exe"
)


def run_xperf(etl, target, options, log):
    command = [
        XPERF,
        "-i",
        windows_path(etl),
        "-o",
        windows_path(target),
        "-a",
        *options,
    ]
    with log.open("x") as stream:
        subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=90
        )
    return command


def summarize_window(rows, start_us, end_us):
    """只累计时间窗口内的已观测 CPU 片段；多核心时间相加可大于窗口长度。"""
    running, ready = {}, {}
    usage = Counter()
    faults, waits = [], []
    for row in rows:
        stamp = int(row[1])
        if row[0] == "CSwitch":
            cpu = int(row[16])
            if cpu in running:
                begun, process, tid = running[cpu]
                if int(row[9]) == tid:
                    usage[process] += (
                        max(0, min(end_us, stamp) - max(start_us, begun)) / 1000
                    )
            running[cpu] = (stamp, row[2], int(row[3]))
            if "vmmem" in row[8].lower() and row[12] == "Ready":
                ready[int(row[9])] = (stamp, row[13], row[2])
            if "vmmem" in row[2].lower() and int(row[3]) in ready:
                begun, reason, other = ready.pop(int(row[3]))
                overlap = max(0, min(end_us, stamp) - max(start_us, begun)) / 1000
                if overlap:
                    waits.append(
                        {
                            "tid": int(row[3]),
                            "overlap_ms": overlap,
                            "reason": reason,
                            "next_process": other,
                        }
                    )
        if (
            row[0] in ("HardFault", "InMemoryStoreFault", "DiskRead")
            and start_us <= stamp <= end_us
        ):
            faults.append(row)
    return {
        "process_cpu_ms": dict(usage.most_common()),
        "fault_and_disk_events": faults,
        "vmmem_ready_waits": sorted(waits, key=lambda x: -x["overlap_ms"]),
    }


def build(root, output):
    output.mkdir(exist_ok=False)
    location = read_json(root / "host_capture_location.json")
    host = Path(location["host"])
    etl = Path(location["capture"]) / "host.etl"
    with etl.open("rb") as file:
        header = file.read(512)
    origin = struct.unpack_from("<Q", header, 88)[0]
    frequency = struct.unpack_from("<Q", header, 360)[0]
    filetime = struct.unpack_from("<Q", header, 368)[0]
    clocks = [json.loads(line) for line in (root / "clock_bridge.jsonl").open()]
    clocks = [r for r in clocks if 0 < r["receive_s"] - r["send_s"] < 0.004]
    if len(clocks) < 10 or not all(r["frequency"] == frequency for r in clocks):
        raise ValueError("Insufficient clock correspondence or unexpected ETL clock")
    x = np.array([(r["send_s"] + r["receive_s"]) / 2 for r in clocks])
    y = np.array([(r["qpc"] - origin) / frequency for r in clocks])
    utc_error = (
        np.array([r["unix_s"] - (filetime / 1e7 - 11644473600) for r in clocks]) - y
    )
    if abs(np.median(utc_error)) > 0.05:
        raise ValueError("ETL clock header does not agree with measured Windows time")
    residuals = [
        y[i] - np.interp(x[i], x[[i - 1, i + 1]], y[[i - 1, i + 1]])
        for i in range(1, len(x) - 1)
    ]
    report = {
        "etl": str(etl),
        "origin_qpc": origin,
        "frequency": frequency,
        "clock_unix_check_median_s": float(np.median(utc_error)),
        "rtt_ms": list(
            map(
                float,
                np.percentile(
                    [(r["receive_s"] - r["send_s"]) * 1000 for r in clocks],
                    [50, 99, 100],
                ),
            )
        ),
        "interpolation_residual_abs_p99_ms": float(
            np.percentile(np.abs(residuals), 99) * 1000
        ),
        "windows": [],
    }
    for label in ("A_r3", "B_r3"):
        summary, frames = collect_run(root / label)
        slow = sorted(summary["slow"], key=lambda r: -r["external_call_ms"])[:8]
        if not slow:
            slow = [summary["first_frame"]]
        chosen = {r["source_frame"]: r for r in slow}
        # 相邻正常帧作为同一段系统负载的参照；不把它当作完全相同内容。
        for row in slow:
            candidates = [
                f
                for f in frames
                if 0 < row["source_frame"] - f["source_frame"] <= 6
                and f["external_call_ms"] < 12
            ]
            if candidates:
                normal = candidates[-1]
                chosen[normal["source_frame"]] = normal
        for seq, frame in sorted(chosen.items()):
            start, end = frame["start_s"], frame["end_s"]
            if not x[0] <= start <= end <= x[-1]:
                raise ValueError("Frame not bracketed by clock samples")
            a, b = np.interp([start, end], x, y)
            low, high = max(0, int((a - 0.03) * 1e6)), int((b + 0.03) * 1e6)
            native = host / f"etw_{label}_{seq}.txt"
            command = run_xperf(
                etl,
                native,
                ["dumper", "-range", str(low), str(high)],
                output / f"{label}_{seq}.export.log",
            )
            events = []
            with native.open(errors="replace") as file:
                for record in csv.reader(file):
                    record = [value.strip() for value in record]
                    if (
                        len(record) > 1
                        and record[1].isdigit()
                        and low <= int(record[1]) <= high
                    ):
                        events.append(record)
            write_json(output / f"{label}_{seq}.events.json", events)
            j = int(np.searchsorted(x, start))
            local = clocks[max(0, j - 2) : j + 3]
            result = {
                "run": label,
                "source_frame": seq,
                "video_frame": frame["video_frame"],
                "start_wsl_s": start,
                "end_wsl_s": end,
                "analysis_ms": frame["wall_ms"],
                "main_thread_cpu_ms": frame["thread_cpu_ms"],
                "runnable_wait_ms": frame["runnable_wait_ms"],
                "start_etw_s": float(a),
                "end_etw_s": float(b),
                "clock_samples": local,
                "local_half_rtt_max_ms": max(
                    (c["receive_s"] - c["send_s"]) * 500 for c in local
                ),
                "command": command,
                **summarize_window(events, a * 1e6, b * 1e6),
            }
            report["windows"].append(result)
            print(
                label,
                seq,
                "ms",
                round(frame["wall_ms"], 3),
                "host CPU",
                list(result["process_cpu_ms"].items())[:3],
                flush=True,
            )
    trace_stats = host / "trace_stats.txt"
    run_xperf(etl, trace_stats, ["tracestats"], output / "trace_stats.log")
    text = trace_stats.read_text(errors="replace")
    (output / "trace_stats.txt").write_text(text)
    losses = re.findall(r"Total # Lost (?:Buffers|Events)\s*:\s*(\d+)", text)
    report["no_lost_events_or_buffers"] = len(losses) == 2 and all(
        int(value) == 0 for value in losses
    )
    write_json(output / "summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
