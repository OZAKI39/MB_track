"""独立诊断 NVDEC 的逐帧延迟；不改变正式程序或设备设置。

同一份压缩包分别在正常等待、CPU 忙等、间隔内 GPU 轻负载和连续送帧下运行。
GPU 轻负载只是短期诊断干预；不会锁频、改变电源设置或用作正式优化。
"""

import argparse
import csv
import ctypes
import resource
import threading
import time
from pathlib import Path

import numpy as np

from customer_workspace.tools.gpu_decode_check import (
    GpuDecoder,
    load_packets,
    save_json,
    wait_until,
)


class GpuStatus:
    """只读 NVML 频率和功耗状态；不支持的读数保留错误码。"""

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.handle = ctypes.c_void_p()
        self.check(self.lib.nvmlInit_v2())
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)))

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML returned {code}")

    def read(self):
        values = {"observed_s": time.perf_counter()}
        for label, kind in (("graphics_mhz", 0), ("memory_mhz", 2), ("video_mhz", 3)):
            value = ctypes.c_uint()
            code = self.lib.nvmlDeviceGetClockInfo(
                self.handle, kind, ctypes.byref(value)
            )
            values[label] = value.value if not code else f"error:{code}"
        for label, function in (
            ("pstate", "nvmlDeviceGetPerformanceState"),
            ("power_mw", "nvmlDeviceGetPowerUsage"),
        ):
            value = ctypes.c_uint()
            code = getattr(self.lib, function)(self.handle, ctypes.byref(value))
            values[label] = value.value if not code else f"error:{code}"
        return values

    def close(self):
        self.lib.nvmlShutdown()


def scheduler_state(path):
    """Linux 当前线程的运行时间、可运行但没获得 CPU 的时间及调度次数。"""
    return tuple(map(int, path.read_text().split()))


def write_rows(path, rows):
    with path.open("x", newline="") as file:
        if rows:
            writer = csv.DictWriter(file, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    packets, metadata = load_packets(args.video)
    if args.frames > len(packets):
        raise ValueError("Requested more frames than the source contains")
    save_json(args.output_dir / "input.json", metadata)
    gpu = GpuDecoder((metadata["height"], metadata["width"]))
    monitor = GpuStatus()
    scheduler = Path(f"/proc/self/task/{threading.get_native_id()}/schedstat")
    # 每次只有一个线程运行约 0.5 ms；不跨越计划送帧时刻，不持续占用全部显卡。
    warm_kernel = gpu.cp.RawKernel(
        r"""extern "C" __global__ void stay_active(unsigned long long duration) {
            unsigned long long start, now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
            do {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            } while (now - start < duration);
        }""",
        "stay_active",
    )
    warm_kernel.compile()
    rows, observations = [], []
    profiling = False
    try:
        for packet in packets[:8]:
            frame = gpu.decode(packet)
            view = gpu.view(frame)
            gpu.process(view)
            del view, frame
        if args.profile:
            gpu.cp.cuda.profiler.start()
            profiling = True
        origin = time.perf_counter() + 0.1
        for index, packet in enumerate(packets[: args.frames]):
            planned = origin + packet["time_s"] - packets[0]["time_s"]
            if args.wait == "hybrid":
                wait_until(planned)
            elif args.wait == "spin":
                while time.perf_counter() < planned:
                    pass
            elif args.wait == "gpu":
                while planned - time.perf_counter() > 0.002:
                    with gpu.stream:
                        warm_kernel((1,), (1,), (np.uint64(500_000),))
                    gpu.stream.synchronize()
                while time.perf_counter() < planned:
                    pass
            sched_before = scheduler_state(scheduler)
            usage_before = resource.getrusage(resource.RUSAGE_THREAD)
            begin = time.perf_counter_ns()
            data = gpu.nvc.PacketData()
            data.bsl = packet["data"].size
            data.bsl_data = packet["data"].ctypes.data
            data.pts = data.dts = packet["pts"]
            data.decode_flag = int(gpu.nvc.VideoPacketFlag.ENDOFPICTURE) | int(
                gpu.nvc.VideoPacketFlag.TIMESTAMP
            )
            c0, t0 = time.thread_time_ns(), time.perf_counter_ns()
            frames = gpu.decoder.Decode(data)
            t1, c1 = time.perf_counter_ns(), time.thread_time_ns()
            gpu.decoder.SyncOnCUStream()
            t2, c2 = time.perf_counter_ns(), time.thread_time_ns()
            gpu.stream.synchronize()
            t3, c3 = time.perf_counter_ns(), time.thread_time_ns()
            usage_after = resource.getrusage(resource.RUSAGE_THREAD)
            sched_after = scheduler_state(scheduler)
            if len(frames) != 1:
                raise RuntimeError("Expected one immediately available frame")
            row = {
                "frame": index,
                "source_pts": packet["pts"],
                "source_time_s": packet["time_s"],
                "decode_start_s": t0 / 1e9,
                "decode_end_s": t1 / 1e9,
                "late_ms": max(0, begin / 1e9 - planned) * 1000
                if args.wait != "none"
                else "",
                "packet_prepare_ms": (t0 - begin) / 1e6,
                "decode_ms": (t1 - t0) / 1e6,
                "decode_cpu_ms": (c1 - c0) / 1e6,
                "decoder_sync_ms": (t2 - t1) / 1e6,
                "stream_sync_ms": (t3 - t2) / 1e6,
                "stream_sync_cpu_ms": (c3 - c2) / 1e6,
                "total_ms": (t3 - begin) / 1e6,
                "linux_runqueue_ms": (sched_after[1] - sched_before[1]) / 1e6,
                "voluntary_switches": usage_after.ru_nvcsw - usage_before.ru_nvcsw,
                "involuntary_switches": usage_after.ru_nivcsw - usage_before.ru_nivcsw,
                "minor_faults": usage_after.ru_minflt - usage_before.ru_minflt,
                "major_faults": usage_after.ru_majflt - usage_before.ru_majflt,
            }
            rows.append(row)
            view = gpu.view(frames[0])
            gpu.process(view)
            del view, frames
            # 所有方案用同样采样频率；读数在解码计时范围外，成本不混入解码。
            if index % 15 == 0:
                observations.append({"frame": index, **monitor.read()})
        summary = {
            "wait": args.wait,
            "frames": len(rows),
            "profiled": args.profile,
            "stages": {
                key: dict(
                    zip(
                        ("median", "p95", "p99", "max"),
                        map(
                            float,
                            np.percentile(
                                [row[key] for row in rows], [50, 95, 99, 100]
                            ),
                        ),
                    )
                )
                for key in rows[0]
                if key.endswith("_ms") and (key != "late_ms" or args.wait != "none")
            },
            "scheduler_scope": "Linux calling thread only; does not expose Windows host scheduling",
        }
        save_json(args.output_dir / "summary.json", summary)
        print(args.output_dir, summary["stages"]["total_ms"], flush=True)
    finally:
        if profiling:
            gpu.cp.cuda.profiler.stop()
        write_rows(args.output_dir / "frames.csv", rows)
        write_rows(args.output_dir / "gpu_status.csv", observations)
        monitor.close()
        gpu.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument(
        "--wait", choices=("hybrid", "spin", "gpu", "none"), default="hybrid"
    )
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    run(args)


if __name__ == "__main__":
    main()
