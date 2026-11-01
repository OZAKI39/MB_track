"""可选的离线停顿诊断；记录墙钟、线程 CPU、缺页和切换，不改变计算。

主线程 CPU 不含 OpenCV 工作线程；process_cpu_ms 包括同进程所有线程。
墙钟减去线程 CPU 只能说明没有在该线程执行，不能单独区分锁等待、
操作系统调度和 WSL 宿主暂停。minor_faults 是内存页首次映射等软缺页，
不是磁盘读入；major_faults 才表示需要 I/O 的缺页。
这些计数仅反映当前 Linux 内核。Windows 为 WSL 虚拟机回读页文件时，
Linux 的 major_faults 可能仍为零；线程 CPU 接近墙钟也不能单独排除宿主机等待。
"""

import csv
import gc
import gzip
import json
import os
import resource
import threading
import time
from pathlib import Path

FIELDS = (
    "frame",
    "stage",
    "start_s",
    "end_s",
    "wall_ms",
    "thread_cpu_ms",
    "process_cpu_ms",
    "minor_faults",
    "major_faults",
    "voluntary_switches",
    "involuntary_switches",
    "process_minor_faults",
    "process_major_faults",
    "input_blocks",
    "output_blocks",
    "threads",
    "thread_user_ms",
    "thread_system_ms",
    "compact_stall",
    "compact_fail",
    "compact_success",
    "allocstall_normal",
    "allocstall_movable",
    "runnable_wait_ms",
)

MEMORY_FIELDS = FIELDS[-6:-1]


def scheduler_statistics_status():
    """未启用内核统计时，schedstat 的零值没有诊断意义；只在启动时检查。"""
    switch = Path("/proc/sys/kernel/sched_schedstats")
    if switch.exists():
        return {"enabled": switch.read_text().strip() == "1", "source": str(switch)}
    config = Path("/proc/config.gz")
    if config.exists():
        with gzip.open(config, "rt") as file:
            for line in file:
                if line.strip() == "# CONFIG_SCHEDSTATS is not set":
                    return {"enabled": False, "source": "CONFIG_SCHEDSTATS=n"}
    return {"enabled": None, "source": "Kernel scheduler statistics not verified"}


class HandoffTrace:
    """拆分分析结束到下次取帧的时间；仅显式诊断使用。

    连续端点覆盖整个交接区间；每个小段包含上次计时后的记账开销。
    正常在取到下一帧后批量写入，因此写本文件的耗时落在下一帧的
    injected_ms 内，不冒充图像处理时间。长时间等帧时每 256 段落盘，
    避免诊断数据无限积累；这种落盘单独记为 trace_flush。
    """

    def __init__(self, path):
        # 跨帧持有文件，统一由本类 __exit__ 关闭。
        self.file = Path(path).open("x", newline="", encoding="utf-8")  # noqa: SIM115
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["after_frame", "stage", "start_s", "end_s", "wall_ms", "thread_cpu_ms"]
        )
        self.previous = None
        self.rows = []

    def begin(self, frame, started_s):
        self.frame = frame
        self.previous = (started_s, time.thread_time())

    def mark(self, stage, *, ended_s=None):
        if self.previous is None:
            return
        now = time.perf_counter() if ended_s is None else ended_s
        cpu = time.thread_time()
        previous, old_cpu = self.previous
        self.rows.append(
            (
                self.frame, stage, previous, now, (now - previous) * 1000,
                (cpu - old_cpu) * 1000,
            )
        )
        self.previous = (now, cpu)
        if len(self.rows) >= 256:
            self.writer.writerows(self.rows)
            self.rows.clear()
            self.mark("trace_flush")

    def finish(self):
        self.writer.writerows(self.rows)
        self.rows.clear()
        self.previous = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        try:
            self.mark("stopped")
            self.finish()
        finally:
            self.file.close()


class StallProbe:
    """一个调用线程使用一个探针；按指定帧数批量写日志，内存有界。

    默认每帧保存以兼容原诊断；短测可缓存更多帧，close() 排空余下记录。
    scheduler 只读取当前线程已打开的调度计数文件，不扫描线程列表。
    """

    def __init__(self, path, *, memory_counters=False, detailed=False,
                 buffer_frames=1, scheduler=False):
        if buffer_frames < 1:
            raise ValueError("buffer_frames must be positive")
        self.memory_counters = memory_counters
        self.detailed = detailed
        self.buffer_frames = buffer_frames
        self.buffered_frames = 0
        self.pending_rows = []
        self.pending_gc_rows = []
        self.closed = False
        self.scheduler_file = None
        scheduler_status = scheduler_statistics_status() if scheduler else None
        if scheduler and scheduler_status["enabled"] is True:
            # 缓存当前线程的句柄，避免逐步骤打开文件；不扫描其他线程。
            scheduler_path = Path(f"/proc/self/task/{threading.get_native_id()}/schedstat")
            self.scheduler_file = scheduler_path.open()
        # 文件跨多帧使用，由 close() 与整个诊断生命周期一起关闭。
        self.file = Path(path).open("x", newline="", encoding="utf-8")  # noqa: SIM115
        self.writer = csv.DictWriter(self.file, FIELDS)
        self.writer.writeheader()
        self.gc_file = (
            Path(path).with_suffix(".gc.csv").open("x", newline="", encoding="utf-8")
        )
        self.gc_writer = csv.writer(self.gc_file)
        self.gc_writer.writerow(
            ["frame", "thread_id", "generation", "start_s", "end_s", "wall_ms"]
        )
        self.gc_starts = {}
        self.gc_rows = []
        self.rows = []
        self.frame = -1
        self.previous = None
        Path(path).with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "native_thread_id": threading.get_native_id(),
                    "perf_counter_s": time.perf_counter(),
                    "unix_s": time.time(),
                    "memory_counters": memory_counters,
                    "scheduler": scheduler,
                    "scheduler_statistics": scheduler_status,
                    "buffer_frames": buffer_frames,
                }, indent=2,
            ) + "\n", encoding="utf-8",
        )
        gc.callbacks.append(self._gc_callback)

    def _gc_callback(self, phase, info):
        key = (threading.get_native_id(), info["generation"])
        now = time.perf_counter()
        if phase == "start":
            self.gc_starts[key] = now
        else:
            start = self.gc_starts.pop(key, None)
            if start is not None:
                self.gc_rows.append(
                    (self.frame, *key, start, now, (now - start) * 1000)
                )

    def _snapshot(self):
        memory = {}
        if self.memory_counters:
            for line in Path("/proc/vmstat").read_text().splitlines():
                name, value = line.split()
                if name in MEMORY_FIELDS:
                    memory[name] = int(value)
        thread = resource.getrusage(resource.RUSAGE_THREAD)
        process = resource.getrusage(resource.RUSAGE_SELF)
        runnable_wait_ns = None
        if self.scheduler_file is not None:
            self.scheduler_file.seek(0)
            runnable_wait_ns = int(self.scheduler_file.read().split()[1])
        return (
            time.perf_counter(),
            time.thread_time(),
            time.process_time(),
            thread,
            process,
            memory,
            runnable_wait_ns,
        )

    def begin(self, frame):
        if self.previous is not None:
            raise RuntimeError("Previous diagnostic frame is still open")
        self.frame = frame
        self.previous = self._snapshot()

    def mark(self, stage):
        current = self._snapshot()
        if self.previous is None:
            raise RuntimeError("Diagnostic frame was not started")
        old, self.previous = self.previous, current
        a, b = old[3], current[3]
        p, q = old[4], current[4]
        self.rows.append(
            dict(
                frame=self.frame,
                stage=stage,
                start_s=old[0],
                end_s=current[0],
                wall_ms=(current[0] - old[0]) * 1000,
                thread_cpu_ms=(current[1] - old[1]) * 1000,
                process_cpu_ms=(current[2] - old[2]) * 1000,
                minor_faults=b.ru_minflt - a.ru_minflt,
                major_faults=b.ru_majflt - a.ru_majflt,
                voluntary_switches=b.ru_nvcsw - a.ru_nvcsw,
                involuntary_switches=b.ru_nivcsw - a.ru_nivcsw,
                process_minor_faults=q.ru_minflt - p.ru_minflt,
                process_major_faults=q.ru_majflt - p.ru_majflt,
                input_blocks=q.ru_inblock - p.ru_inblock,
                output_blocks=q.ru_oublock - p.ru_oublock,
                threads="",
                thread_user_ms=(b.ru_utime - a.ru_utime) * 1000,
                thread_system_ms=(b.ru_stime - a.ru_stime) * 1000,
                runnable_wait_ms=(current[6] - old[6]) / 1e6
                if current[6] is not None else "",
                **{
                    name: current[5][name] - old[5][name]
                    if self.memory_counters
                    else ""
                    for name in MEMORY_FIELDS
                },
            )
        )

    def finish(self, handoff=None):
        self.mark("tail")
        if handoff is not None:
            handoff.mark("diagnostic_tail")
        # 普通停顿探针不扫描线程目录；完整扫描仅属于显式详细诊断。
        # 否则诊断本身的文件系统访问可能影响下一帧到达时的交接。
        if self.detailed:
            self.rows[-1]["threads"] = len(os.listdir("/proc/self/task"))
        if handoff is not None:
            handoff.mark("diagnostic_thread_count")
        self.pending_rows.extend(self.rows)
        gc_rows, self.gc_rows = self.gc_rows, []
        self.pending_gc_rows.extend(gc_rows)
        self.buffered_frames += 1
        if self.buffered_frames >= self.buffer_frames:
            self.flush()
        if handoff is not None:
            handoff.mark("diagnostic_csv")
        self.rows.clear()
        self.previous = None
        if handoff is not None:
            handoff.mark("diagnostic_gc_and_clear")

    def close(self):
        if self.closed:
            return
        if self.previous is not None:
            self.finish()
        gc.callbacks.remove(self._gc_callback)
        try:
            self.pending_gc_rows.extend(self.gc_rows)
            self.flush()
        finally:
            self.file.close()
            self.gc_file.close()
            if self.scheduler_file is not None:
                self.scheduler_file.close()
            self.closed = True

    def flush(self):
        """只保存诊断记录；实验数据仍由原有保存线程负责。"""
        self.writer.writerows(self.pending_rows)
        self.gc_writer.writerows(self.pending_gc_rows)
        self.pending_rows.clear()
        self.pending_gc_rows.clear()
        self.buffered_frames = 0
