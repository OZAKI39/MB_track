"""Windows 独立对照：不增加解码任务、单路、双路，然后倒序重复。

只启动已有 F 盘解码测试，不改变电源、服务或设备配置。阶段边界使用
Windows QPC；不在测量期间轮询显卡管理接口，以免测量本身唤醒显卡。
"""

import argparse
import ctypes
import json
import os
import subprocess
import time
from pathlib import Path


def run(root: Path, output: Path):
    if os.name != "nt":
        raise RuntimeError("Run this diagnostic with Windows Python")
    output.mkdir(parents=True, exist_ok=False)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    frequency = ctypes.c_int64()
    if not kernel.QueryPerformanceFrequency(ctypes.byref(frequency)):
        raise ctypes.WinError(ctypes.get_last_error())

    def qpc():
        value = ctypes.c_int64()
        if not kernel.QueryPerformanceCounter(ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.value

    phases = [
        ("quiet_1", ()),
        ("single_A", ("A",)),
        ("dual_AB", ("A", "B")),
        ("dual_BA", ("B", "A")),
        ("single_B", ("B",)),
        ("quiet_2", ()),
    ]
    records = {"qpc_frequency": frequency.value, "phases": []}
    try:
        for name, videos in phases:
            processes, logs = [], []
            record = {"name": name, "start_qpc": qpc(), "runs": []}
            records["phases"].append(record)
            print("START", name, flush=True)
            try:
                for video in videos:
                    target = output / f"traced_{name}_{video}"
                    command = [
                        str(root / "python312/python.exe"),
                        "-u",
                        "-m",
                        "customer_workspace.tools.gpu_native_decode",
                        "--video",
                        str(root / f"videos/exp{video}.mp4"),
                        "--output-dir",
                        str(target),
                    ]
                    log = (output / f"{name}_{video}.log").open("x")
                    logs.append(log)
                    process = subprocess.Popen(command, stdout=log, stderr=log)
                    processes.append(process)
                    record["runs"].append({"path": str(target), "pid": process.pid})
                if not videos:
                    time.sleep(22)
                deadline = time.monotonic() + 45
                for process in processes:
                    process.wait(timeout=max(0.1, deadline - time.monotonic()))
                    if process.returncode:
                        raise RuntimeError(f"Decode failed: PID {process.pid}")
            finally:
                # 仅收尾本脚本创建的测试子进程，不影响其他用户程序。
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)
                for log in logs:
                    log.close()
                record["end_qpc"] = qpc()
            print("DONE", name, flush=True)
    finally:
        (output / "phases.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.native_root, args.output_dir)
