"""按需记录当前分析线程的时间和调度计数；不扫描线程列表，不逐帧写盘。"""

import resource
import threading
import time
from pathlib import Path


def scheduler_counters():
    """读取当前 Linux 线程等待 CPU 的累计时间；不能将主线程 CPU 当作 OpenCV 总 CPU。"""
    path = Path(f"/proc/self/task/{threading.get_native_id()}/schedstat")
    fields = path.read_text().split()
    usage = resource.getrusage(resource.RUSAGE_THREAD)
    return {
        "runnable_wait_s": int(fields[1]) / 1e9,
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "voluntary_switches": usage.ru_nvcsw,
        "involuntary_switches": usage.ru_nivcsw,
    }


def append_span(events, stage, start, cpu_start, **values):
    events.append(
        {
            "stage": stage,
            "started_wsl_s": start,
            "finished_wsl_s": time.perf_counter(),
            "thread_cpu_ms": (time.thread_time() - cpu_start) * 1000,
            **values,
        }
    )
