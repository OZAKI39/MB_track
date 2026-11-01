"""汇总同录像的内存对照；Windows 系统总缺页与 WSL 页文件读取分开统计。"""

import argparse
import csv
import json
from itertools import pairwise
from pathlib import Path

import numpy as np


def distribution(values):
    """空样本保留为缺失，不将缺失误报为零。"""
    values = list(values)
    if not values:
        return {"count": 0, "median": None, "p99": None, "max": None, "min": None}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def pagefile_reads(path):
    """从 xperf hardfault -bytes 结果中只提取 WSL 的页文件读取。

    此数值是整段 ETW 记录的累计量，含启动和收尾；不等于算法独占的读盘量。
    同一页面重复读入会重复计数，因此累计字节可以大于物理内存容量。
    """
    result = {"faults": 0, "bytes": 0}
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file, skipinitialspace=True)
        header = [value.strip() for value in next(reader, [])]
        if (
            len(header) != 4
            or header[:2] != ["NumFaults", "TotalBytes"]
            or header[-1] != "FileName"
        ):
            raise ValueError("Expected an xperf hardfault -bytes report")
        for row in reader:
            if len(row) != 4 or not row[0].strip().isdigit():
                continue
            process, filename = row[2].strip(), row[3].strip().lower()
            if process.startswith("vmmemWSL (") and filename.endswith("\\pagefile.sys"):
                result["faults"] += int(row[0])
                result["bytes"] += int(row[1])
    return result


def counter_delta(rows, field):
    """累计计数若倒退就报错，不将重置后的值算成负速率。"""
    values = [int(row[field]) for row in rows]
    if len(values) < 2:
        return None
    if any(right < left for left, right in pairwise(values)):
        raise ValueError(f"Counter reset detected: {field}")
    return values[-1] - values[0]


def summarize_run(directory):
    """保留启动期与正式逐帧数据；首帧单列，但总体统计不删除慢样本。"""
    directory = Path(directory)
    diagnostic_path = directory / "replay/source_stalls.csv"
    settings = json.loads((directory / "replay/pacing_settings.json").read_text())
    diagnostics_enabled = settings.get("diagnose_stalls", False)
    conversion = []
    # 明确关闭时可以缺少细分数据；要求开启却缺文件时仍报错，不掩盖采集失败。
    if diagnostic_path.exists() or diagnostics_enabled:
        with diagnostic_path.open(newline="") as file:
            conversion = [
                row for row in csv.DictReader(file) if row["stage"] == "convert_bgr"
            ]
    values = [float(row["wall_ms"]) for row in conversion]
    linux = read_jsonl(directory / "linux_memory.jsonl")
    windows = read_jsonl(directory / "windows_memory.jsonl")
    mib = 1024**2
    result = {
        "run": str(directory),
        "status": json.loads((directory / "run.json").read_text()),
        "conversion_timing_recorded": bool(conversion),
        "conversion_ms": distribution(values),
        "conversion_over_10_ms": sum(value > 10 for value in values) if values else None,
        "conversion_over_20_ms": sum(value > 20 for value in values) if values else None,
        "conversion_over_50_ms": sum(value > 50 for value in values) if values else None,
        "first_conversion_ms": values[0] if values else None,
        "slowest_conversions": sorted(
            conversion, key=lambda row: -float(row["wall_ms"])
        )[:10],
        "linux_available_mib": distribution(
            row["available_bytes"] / mib for row in linux
        ),
        "linux_used_mib": distribution(row["used_bytes"] / mib for row in linux),
        "linux_swap_used_mib": distribution(
            row["swap_used_bytes"] / mib for row in linux
        ),
        "pylance_rss_mib": distribution(
            sum(p["rss_bytes"] for p in row["processes"] if p["pylance"]) / mib
            for row in linux
        ),
        "windows_available_mib": distribution(
            row["memory"]["AvailableMBytes"] for row in windows
        ),
        "windows_committed_mib": distribution(
            row["memory"]["CommittedBytes"] / mib for row in windows
        ),
        "windows_commit_headroom_mib": distribution(
            (row["memory"]["CommitLimit"] - row["memory"]["CommittedBytes"]) / mib
            for row in windows
        ),
        "wsl_host_resident_mib": distribution(
            sum(p["WorkingSet"] for p in row["wsl_process"]) / mib for row in windows
        ),
        "linux_counter_changes": {
            name: counter_delta([row["vmstat"] for row in linux], name)
            for name in ("pgmajfault", "pswpin", "pswpout", "oom_kill")
        },
        "windows_all_process_page_inputs": counter_delta(
            [row["memory"] for row in windows], "PagesInputPerSec"
        ),
        "limitations": (
            "Windows PagesInput includes file-backed reads across all processes. "
            "It is NOT WSL pagefile traffic. No direct cross-OS clock subtraction. "
            "CPU/host monitoring can perturb the run; compare like-for-like recordings."
            + ("" if conversion else " Per-frame conversion timing was not recorded.")
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [summarize_run(path) for path in args.runs]
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")
    for row in results:
        print(
            row["run"],
            row["conversion_ms"],
            "over 10 ms:",
            row["conversion_over_10_ms"],
        )


if __name__ == "__main__":
    main()
