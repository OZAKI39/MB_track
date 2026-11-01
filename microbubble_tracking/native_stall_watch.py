"""只读采样诊断进程的内核等待位置；需要读取 /proc/.../stack 的权限。

仅观察指定运行目录登记的三个线程，不改变调度、磁盘或系统配置。
与 perf 样本、StallProbe 共用 Linux monotonic 时间。采样没有覆盖的短等待
不能解释为不存在；系统总计数也不能直接归因于某一个进程。
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def read_optional(path):
    try:
        return path.read_text().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
        return f"unavailable: {type(error).__name__}"


def watch(
    run, output, duration, interval, perf=None, perf_lib=None,
    perf_targets=("source", "writer"), frequency=199,
    stop_file=None,
):
    """限时观察登记线程；perf 目标和采样率显式指定，默认保持旧设置。"""
    if perf:
        for suffix in (".data", ".perf.log"):
            if output.with_suffix(suffix).exists():
                raise FileExistsError(output.with_suffix(suffix))
    identities = {}
    deadline = time.monotonic() + duration
    next_system = 0.0
    recorder = None
    perf_log = None
    with output.open("x", encoding="utf-8", buffering=1) as file:
        try:
            while time.monotonic() < deadline:
                if stop_file is not None and Path(stop_file).exists():
                    break
                started = time.monotonic()
                rows = []
                for name in ("consumer", "source", "writer"):
                    if name not in identities:
                        path = run / f"{name}_stalls.meta.json"
                        if path.exists():
                            try:
                                identities[name] = json.loads(path.read_text())
                            except json.JSONDecodeError:
                                # 登记文件可能尚未写完；下一次采样再读取。
                                continue
                    if name not in identities:
                        continue
                    identity = identities[name]
                    task = Path(
                        f"/proc/{identity['pid']}/task/{identity['native_thread_id']}"
                    )
                    if not task.exists():
                        continue
                    rows.append(
                        {
                            "name": name,
                            "tid": identity["native_thread_id"],
                            "syscall": read_optional(task / "syscall"),
                            "wchan": read_optional(task / "wchan"),
                            "stack": read_optional(task / "stack"),
                            "stat": read_optional(task / "stat"),
                            "schedstat": read_optional(task / "schedstat"),
                        }
                    )
                record = {"time_s": started, "threads": rows}
                if (
                    perf
                    and recorder is None
                    and set(perf_targets) <= identities.keys()
                ):
                    tids = ",".join(
                        str(identities[name]["native_thread_id"])
                        for name in perf_targets
                    )
                    command = [
                        str(perf),
                        "record",
                        "--clockid",
                        "mono",
                        "--switch-events",
                        "-t",
                        tids,
                        "-F",
                        str(frequency),
                        "-e",
                        "cpu-clock",
                        "-e",
                        "syscalls:sys_enter_fdatasync",
                        "-e",
                        "syscalls:sys_exit_fdatasync",
                        "-e",
                        "syscalls:sys_enter_fsync",
                        "-e",
                        "syscalls:sys_exit_fsync",
                        "--call-graph",
                        "dwarf,4096",
                        "-m",
                        "512",
                        "-o",
                        str(output.with_suffix(".data")),
                    ]
                    env = dict(os.environ)
                    if perf_lib:
                        env["LD_LIBRARY_PATH"] = str(perf_lib)
                    perf_log = output.with_suffix(".perf.log").open("x")
                    recorder = subprocess.Popen(
                        command,
                        env=env,
                        stdout=perf_log,
                        stderr=subprocess.STDOUT,
                    )
                    record["perf_command"] = command
                if started >= next_system:
                    record["system"] = {
                        name: read_optional(Path("/proc") / name)
                        for name in ("diskstats", "vmstat", "stat", "pressure/io")
                    }
                    next_system = started + 0.25
                record["sample_ms"] = (time.monotonic() - started) * 1000
                file.write(json.dumps(record) + "\n")
                if identities and not rows and (run / "run_state.json").exists():
                    break
                time.sleep(max(0.0, interval - (time.monotonic() - started)))
        finally:
            if recorder is not None:
                if recorder.poll() is None:
                    recorder.send_signal(signal.SIGINT)
                try:
                    returncode = recorder.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    # 只结束本观察器启动的 perf，绝不结束被观察的实验进程。
                    recorder.kill()
                    recorder.wait()
                    raise RuntimeError("Native recording did not finish within 15 s") from None
                finally:
                    perf_log.close()
                if returncode not in (0, -signal.SIGINT):
                    raise RuntimeError(f"Native recording failed: exit {returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--duration", type=float, default=90)
    parser.add_argument("--interval", type=float, default=0.01)
    parser.add_argument("--perf", type=Path)
    parser.add_argument("--perf-lib", type=Path)
    parser.add_argument(
        "--perf-targets", nargs="+", choices=("consumer", "source", "writer"),
        default=("source", "writer"),
    )
    parser.add_argument("--frequency", type=int, default=199)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    if args.duration <= 0 or args.interval < 0.001 or not 1 <= args.frequency <= 1000:
        parser.error("Use a positive duration, interval >= 1 ms and frequency 1-1000 Hz")
    watch(
        args.run, args.output, args.duration, args.interval, args.perf, args.perf_lib,
        args.perf_targets, args.frequency, args.stop_file,
    )


if __name__ == "__main__":
    main()
