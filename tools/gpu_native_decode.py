"""Windows/WSL 共用的独立 NVDEC 对照：固定 60 FPS，无相机、追踪或设备控制。

复用既有压缩包读取、零延迟解码参数和等待方法。通过 CUDA Driver API 管理
上下文与流，两端使用同一实现，不引入图像处理内核或逐帧磁盘写入。
正常计时只输出显存图像；--verify 单独下载完整图像做逐像素核对。
"""

import argparse
import csv
import ctypes as ct
import importlib.metadata
import json
import os
import platform
import threading
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import PyNvVideoCodec as nvc

from customer_workspace.tools.gpu_decode_check import (
    cpu_decoder,
    decode_cpu,
    load_packets,
    save_json,
    wait_until,
)


class CudaDriver:
    """仅创建本测试的 CUDA 资源；不修改设备设置或其他进程的上下文。"""

    def __init__(self):
        self.windows = os.name == "nt"
        self.library = (
            ct.WinDLL("nvcuda.dll") if self.windows else ct.CDLL("libcuda.so.1")
        )
        self.context, self.stream = ct.c_void_p(), ct.c_void_p()
        self.init = self.bind("cuInit", [ct.c_uint])
        self.retain = self.bind(
            "cuDevicePrimaryCtxRetain", [ct.POINTER(ct.c_void_p), ct.c_int]
        )
        self.release = self.bind("cuDevicePrimaryCtxRelease_v2", [ct.c_int])
        self.set_current = self.bind("cuCtxSetCurrent", [ct.c_void_p])
        self.create_stream = self.bind(
            "cuStreamCreate", [ct.POINTER(ct.c_void_p), ct.c_uint]
        )
        self.destroy_stream = self.bind("cuStreamDestroy_v2", [ct.c_void_p])
        self.synchronize = self.bind("cuStreamSynchronize", [ct.c_void_p])
        self.download = self.bind(
            "cuMemcpyDtoH_v2", [ct.c_void_p, ct.c_uint64, ct.c_size_t]
        )
        self.pointer_attribute = self.bind(
            "cuPointerGetAttribute", [ct.c_void_p, ct.c_int, ct.c_uint64]
        )
        self.init(0)
        self.retain(ct.byref(self.context), 0)
        self.set_current(self.context)
        self.create_stream(ct.byref(self.stream), 1)  # CU_STREAM_NON_BLOCKING。

    def bind(self, name, arguments):
        function = getattr(self.library, name)
        function.argtypes, function.restype = arguments, ct.c_int

        def checked(*args):
            status = function(*args)
            if status:
                raise RuntimeError(f"{name} failed: CUDA status {status}")
            return status

        return checked

    def close(self):
        if self.stream.value:
            self.synchronize(self.stream)
            self.destroy_stream(self.stream)
            self.stream = ct.c_void_p()
        if self.context.value:
            self.release(0)
            self.context = ct.c_void_p()


class WindowsClock:
    """Windows 帧记录直接保存 QPC，随后与同一时钟的 ETW 事件比较。"""

    def __init__(self):
        self.frequency = None
        if os.name == "nt":
            kernel = ct.WinDLL("kernel32", use_last_error=True)
            self.counter = kernel.QueryPerformanceCounter
            self.counter.argtypes, self.counter.restype = (
                [ct.POINTER(ct.c_int64)],
                ct.c_int,
            )
            frequency = ct.c_int64()
            if not kernel.QueryPerformanceFrequency(ct.byref(frequency)):
                raise ct.WinError(ct.get_last_error())
            self.frequency = frequency.value

    def read(self):
        if self.frequency is None:
            return ""
        value = ct.c_int64()
        if not self.counter(ct.byref(value)):
            raise ct.WinError(ct.get_last_error())
        return value.value


def make_packet(packet):
    data = nvc.PacketData()
    data.bsl, data.bsl_data = packet["data"].size, packet["data"].ctypes.data
    data.pts = data.dts = packet["pts"]
    data.decode_flag = int(nvc.VideoPacketFlag.ENDOFPICTURE) | int(
        nvc.VideoPacketFlag.TIMESTAMP
    )
    return data


def write_rows(path, rows):
    if rows:
        with path.open("x", newline="") as file:
            writer = csv.DictWriter(file, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def percentiles(values):
    return dict(
        zip(
            ("median", "p95", "p99", "max"),
            map(float, np.percentile(values, [50, 95, 99, 100])),
        )
    )


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    packets, metadata = load_packets(args.video)
    # B 的容器帧率为约 60.00006；保留原 PTS，统一按用户指定的 60 FPS 送帧。
    if abs(float(Fraction(metadata["fps"])) - 60) > 0.01:
        raise ValueError("This comparison requires a nominal 60 FPS input")
    packets = packets[: args.frames] if args.frames else packets
    clock = WindowsClock()
    save_json(output / "input.json", metadata)
    save_json(
        output / "environment.json",
        {
            "os": platform.platform(),
            "python": platform.python_version(),
            "executable": os.sys.executable,
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
            "qpc_frequency": clock.frequency,
            "perf_counter": vars(time.get_clock_info("perf_counter")),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("pynvvideocodec", "numpy", "av")
            },
            "scope": "H.264 packet to synchronized GPU frame; no image processing; no image download during timing",
        },
    )
    driver, decoder, rows, verification = None, None, [], []
    completed = False
    try:
        started = time.perf_counter()
        driver = CudaDriver()
        decoder = nvc.CreateDecoder(
            gpuid=0,
            codec=nvc.cudaVideoCodec.H264,
            cudacontext=driver.context.value,
            cudastream=driver.stream.value,
            usedevicememory=True,
            latency=nvc.DisplayDecodeLatencyType.ZERO,
        )
        for packet in packets[:8]:
            frames = decoder.Decode(make_packet(packet))
            decoder.SyncOnCUStream()
            driver.synchronize(driver.stream)
            if len(frames) != 1:
                raise ValueError("Warm-up did not return one frame")
            device_output = {
                "shape": list(frames[0].shape),
                "strides": list(frames[0].strides),
                "dlpack_device": list(frames[0].__dlpack_device__()),
            }
            memory_type = ct.c_uint()
            driver.pointer_attribute(
                ct.byref(memory_type), 2, frames[0].GetPtrToPlane(0)
            )
            if memory_type.value != 2:  # CU_MEMORYTYPE_DEVICE。
                raise ValueError("Decoder output is not device memory")
            device_output["memory_type"] = memory_type.value
            del frames
        # 启动信息在开始送帧前写完，避免 F 盘文件写入拖延第二帧。
        save_json(output / "device_output.json", device_output)
        preparation_ms = (time.perf_counter() - started) * 1000
        reference = cpu_decoder() if args.verify else None
        origin = time.perf_counter() + 0.1
        for index, packet in enumerate(packets):
            planned = origin + index / 60
            if not args.verify:
                wait_until(planned)
            qpc_start = clock.read()
            begin = time.perf_counter_ns()
            data = make_packet(packet)
            prepared = time.perf_counter_ns()
            frames = decoder.Decode(data)
            decoded = time.perf_counter_ns()
            decoder.SyncOnCUStream()
            sync_called = time.perf_counter_ns()
            driver.synchronize(driver.stream)
            ready = time.perf_counter_ns()
            qpc_end = clock.read()
            if len(frames) != 1:
                raise ValueError(f"Incorrect immediate frame count at {index}")
            frame = frames[0]
            rows.append(
                {
                    "frame": index,
                    "source_pts": packet["pts"],
                    # 当前 ZERO 模式返回 PTS=0；用输入顺序及单独的全帧像素核对关联。
                    "decoder_pts": frame.getPTS(),
                    "source_time_s": packet["time_s"],
                    "qpc_start": qpc_start,
                    "qpc_end": qpc_end,
                    "planned_s": planned,
                    "start_s": begin / 1e9,
                    "end_s": ready / 1e9,
                    "late_ms": max(0, begin / 1e9 - planned) * 1000
                    if not args.verify
                    else "",
                    "prepare_ms": (prepared - begin) / 1e6,
                    "decode_call_ms": (decoded - prepared) / 1e6,
                    "decoder_sync_ms": (sync_called - decoded) / 1e6,
                    "stream_sync_ms": (ready - sync_called) / 1e6,
                    "decode_ready_ms": (ready - begin) / 1e6,
                }
            )
            if args.verify:
                raw = decode_cpu(reference, packet, Fraction(metadata["time_base"]))
                expected = raw.to_ndarray(format="nv12")
                pitch = frame.strides[0]
                native = np.empty((metadata["height"] * 3 // 2, pitch), np.uint8)
                driver.download(
                    native.ctypes.data, frame.GetPtrToPlane(0), native.nbytes
                )
                actual = native[:, : metadata["width"]]
                difference = int(np.count_nonzero(actual != expected))
                verification.append({"frame": index, "different_pixels": difference})
                if difference:
                    raise ValueError(
                        f"GPU/CPU pixel mismatch at frame {index}: {difference}"
                    )
            del frame, frames
        end = nvc.PacketData()
        end.bsl = end.bsl_data = 0
        end.decode_flag = int(nvc.VideoPacketFlag.ENDOFSTREAM)
        if decoder.Decode(end):
            raise ValueError("Unexpected delayed frames at EOF")
        decoder.SyncOnCUStream()
        driver.synchronize(driver.stream)
        completed = True
        summary = {
            "completed": True,
            "frames": len(rows),
            "warmup_frames": min(8, len(packets)),
            "preparation_ms": preparation_ms,
            "verification": args.verify,
            "full_image_downloads": len(verification),
            "pixel_mismatches": sum(r["different_pixels"] for r in verification),
            "stages": {
                key: percentiles([row[key] for row in rows])
                for key in rows[0]
                if key.endswith("_ms") and (key != "late_ms" or not args.verify)
            },
            "over_16_67_ms": sum(row["decode_ready_ms"] > 1000 / 60 for row in rows),
        }
        save_json(output / "summary.json", summary)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "frames": len(rows),
                    "decode_ready_ms": summary["stages"]["decode_ready_ms"],
                    "verified": args.verify,
                }
            ),
            flush=True,
        )
    except BaseException as error:
        save_json(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "frames_recorded": len(rows),
            },
        )
        raise
    finally:
        write_rows(output / "frames.csv", rows)
        write_rows(output / "pixel_check.csv", verification)
        decoder = None
        if driver is not None:
            driver.close()
        save_json(
            output / "closed.json", {"completed": completed, "rows_saved": len(rows)}
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frames", type=int, default=0, help="0 means the complete video"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Separate full-pixel check; no speed claims",
    )
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
