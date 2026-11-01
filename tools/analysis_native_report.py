"""从已保存的 perf 记录中提取慢帧内部函数样本；采样不是每项时间的完整计数。"""

import argparse
import bisect
import os
import subprocess
from collections import Counter
from pathlib import Path

from customer_workspace.streamcopy_recording import write_json
from customer_workspace.tools.analysis_stall_report import collect_run
from customer_workspace.tools.rtsp_tracking_report import read_json
from microbubble_experiment.configuration import ROOT


def build(root, name, output):
    summary, _ = collect_run(root / name)
    slow = {row["source_frame"]: row for row in summary["slow"]}
    # 即使首帧未超时，也保留以便核对首次内存访问。
    first = summary["first_frame"]
    slow.setdefault(first["source_frame"], first)
    windows = sorted(
        [
            (
                float(stage["start_s"]),
                float(stage["end_s"]),
                row["source_frame"],
                stage["stage"],
            )
            for row in slow.values()
            for stage in row["stages"]
        ]
    )
    starts = [row[0] for row in windows]
    tid = read_json(root / name / "consumer_stalls.meta.json")["native_thread_id"]
    perf_root = ROOT / "customer_workspace/dependencies/native_perf/installed/usr/lib"
    command = [
        str(perf_root / "linux-tools-6.8.0-147/perf"),
        "script",
        "-i",
        str(root / f"{name}_native.data"),
        "--ns",
        "-F",
        "tid,time,event,ip,sym,dso",
    ]
    env = dict(os.environ, LD_LIBRARY_PATH=str(perf_root / "x86_64-linux-gnu"))
    selected, totals = [], Counter()

    def record(block):
        if not block:
            return
        fields = block[0].split()
        if len(fields) < 3 or not fields[0].isdigit() or int(fields[0]) != tid:
            return
        try:
            stamp = float(fields[1].rstrip(":"))
        except ValueError:
            return
        pos = bisect.bisect_right(starts, stamp) - 1
        if pos < 0 or stamp > windows[pos][1]:
            return
        _, _, frame, stage = windows[pos]
        totals[stage] += 1
        selected.append(
            {"source_frame": frame, "stage": stage, "time_s": stamp, "stack": block}
        )

    with (output / f"{name}.perf_script.log").open("x") as errors:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors, text=True, env=env
        )
        block = []
        for line in process.stdout:
            if line.strip():
                block.append(line.rstrip())
            else:
                record(block)
                block = []
        record(block)
        if process.wait():
            raise RuntimeError("perf script failed; see its error log")
    if not selected:
        raise ValueError(
            "No samples matched the analyzed windows; inspect clock and header format"
        )
    result = {
        "command": command,
        "tid": tid,
        "frames": list(slow),
        "samples_per_stage": dict(totals),
        "samples": selected,
    }
    write_json(output / f"{name}.json", result)
    print(name, "samples", len(selected), "stages", dict(totals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    for name in ("A_r3", "B_r3"):
        build(args.root.resolve(), name, args.output_dir.resolve())


if __name__ == "__main__":
    main()
