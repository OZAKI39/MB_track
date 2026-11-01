"""关联只读 ACPI/WMI 记录与原生解码时间；所有时间使用 Windows QPC。"""

import argparse
import csv
import json
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from customer_workspace.tools.gpu_decode_check import save_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def summary(values):
    if not values:
        return None
    return dict(
        zip(
            ("median", "p95", "p99", "max"),
            map(float, np.percentile(values, [50, 95, 99, 100])),
        )
    )


def run(root, output=None):
    analysis = root / "analysis"
    output = output or root / "trigger_report"
    output.mkdir(exist_ok=False)
    events, methods, calls = [], [], []
    with (analysis / "selected_events.tsv").open() as file:
        for fields in csv.reader(file, delimiter="\t"):
            payload = dict(field.split("=", 1) for field in fields[5:] if "=" in field)
            row = {
                "time_ms": float(fields[0]),
                "provider": fields[1],
                "event": fields[2],
                "pid": int(fields[3]),
                "tid": int(fields[4]),
                **payload,
            }
            if fields[2] == "ACPIEvent" and payload.get("Id") == "197":
                events.append(row)
            if (
                fields[2] == "AmlMethodTrace"
                and payload.get("AmlMethodState") == "started"
            ):
                methods.append(row)
            if fields[2] == "EventID(11)" and "::GetSetULong" in payload.get(
                "Operation", ""
            ):
                # 客户端 PID 是实际请求方；事件 PID 只是 Windows 的 WMI 服务进程。
                calls.append(
                    {
                        "time_ms": row["time_ms"],
                        "client_pid": int(payload["ClientProcessId"].replace(",", "")),
                        "group": payload["GroupOperationId"],
                        "operation": payload["Operation"],
                    }
                )
    links = []
    for event in events:
        earlier_calls = [r for r in calls if 0 <= event["time_ms"] - r["time_ms"] < 100]
        earlier_methods = [
            r
            for r in methods
            if r["AmlMethodName"] == r"\_SB.AMW0.WMBC"
            and 0 <= event["time_ms"] - r["time_ms"] < 20
        ]
        links.append(
            {
                "notification": event,
                "preceding_wmi": earlier_calls[-1] if earlier_calls else None,
                "preceding_wmbc": earlier_methods[-1] if earlier_methods else None,
            }
        )
    save_json(output / "notification_sources.json", links)
    save_json(output / "wmi_calls.json", calls)
    save_json(output / "aml_methods.json", methods)

    clocks = read_json(analysis / "windows.json")["clock"]
    frequency, origin = clocks["frequency"], clocks["origin_qpc"]

    def milliseconds(qpc):
        return (int(qpc) - origin) / frequency * 1000

    locks = read_json(analysis / "adapter_locks.json")
    driver_notifications = [
        row
        for row in read_json(analysis / "acpi_events.json")
        if row["event"] == "DdiNotifyAcpiEvent/Start" and row.get("Event") == "197"
    ]
    c5_locks = []
    for notification in driver_notifications:
        matching = [
            row
            for row in locks
            if row["exclusive"]
            and row["pid"] == notification["pid"]
            and row["tid"] == notification["tid"]
            and row["acquired_ms"] <= notification["time_ms"] <= row["end_ms"]
        ]
        if len(matching) != 1:
            raise ValueError(f"Cannot identify unique C5 lock: {notification}")
        c5_locks.append(matching[0])
    save_json(output / "c5_exclusive_locks.json", c5_locks)
    frames = []
    for path in sorted(root.glob("traced_*")):
        environment = read_json(path / "environment.json")
        with (path / "frames.csv").open() as file:
            for row in csv.DictReader(file):
                start, end = (
                    milliseconds(row["qpc_start"]),
                    milliseconds(row["qpc_end"]),
                )
                frames.append(
                    {
                        "run": path.name,
                        "frame": int(row["frame"]),
                        "start_ms": start,
                        "end_ms": end,
                        "decode_ms": float(row["decode_ready_ms"]),
                        "pid": environment["pid"],
                        "tid": environment["tid"],
                    }
                )
    # 仅合计本次解码线程、当前调用区间内的锁等待；不把其他进程等待算入。
    decode_threads = {(f["pid"], f["tid"]) for f in frames}
    decode_locks = defaultdict(list)
    for row in locks:
        key = (row["pid"], row["tid"])
        if key in decode_threads:
            decode_locks[key].append(row)
    lock_starts = {}
    for key, rows in decode_locks.items():
        rows.sort(key=lambda row: row["start_ms"])
        lock_starts[key] = [row["start_ms"] for row in rows]
    for frame in frames:
        key = (frame["pid"], frame["tid"])
        starts = lock_starts.get(key, [])
        first = bisect_left(starts, frame["start_ms"])
        last = bisect_right(starts, frame["end_ms"])
        waits = [
            r
            for r in decode_locks[key][first:last]
            if r["acquired_ms"] <= frame["end_ms"]
        ]
        frame["lock_wait_ms"] = sum(r["wait_ms"] for r in waits)
    if (root / "phases.json").exists():
        phases = read_json(root / "phases.json")["phases"]
    else:
        phases = [
            {
                "name": "pilot",
                "start_qpc": origin,
                "end_qpc": origin
                + int((max(f["end_ms"] for f in frames) + 1000) * frequency / 1000),
            }
        ]
    result = []
    for phase in phases:
        start, end = milliseconds(phase["start_qpc"]), milliseconds(phase["end_qpc"])
        selected = [f for f in frames if start <= f["start_ms"] < end]
        notifications = [e for e in events if start <= e["time_ms"] < end]
        phase_calls = [c for c in calls if start <= c["time_ms"] < end]
        # 独占持有是系统操作，不等于解码线程真的等待；分别报告。
        held = [r for r in c5_locks if start <= r["start_ms"] < end]
        result.append(
            {
                "phase": phase["name"],
                "duration_s": (end - start) / 1000,
                "start_ms": start,
                "end_ms": end,
                "c5_count": len(notifications),
                "wmi_calls": len(phase_calls),
                "frames": len(selected),
                "decode_ms": summary([f["decode_ms"] for f in selected]),
                "frame_lock_wait_ms": summary([f["lock_wait_ms"] for f in selected]),
                "c5_exclusive_held_ms": summary([r["held_ms"] for r in held]),
                "slowest": sorted(selected, key=lambda f: -f["decode_ms"])[:5],
            }
        )
    save_json(output / "phases_summary.json", result)
    save_json(output / "frame_lock_waits.json", frames)
    save_json(
        output / "source_counts.json",
        {
            "c5": len(events),
            "wmi_calls": len(calls),
            "wmi_clients": dict(Counter(r["client_pid"] for r in calls)),
            "method_counts": dict(Counter(r["AmlMethodName"] for r in methods)),
            "matched_notifications": sum(
                bool(r["preceding_wmi"] and r["preceding_wmbc"]) for r in links
            ),
            "limitations": "Temporal links use 100 ms WMI / 20 ms AML windows; these are not ACPI argument tracing.",
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output_dir)
