"""独立验证 NVDEC→显存→CUDA 图像处理；不接入正式追踪或任何设备。

CPU/GPU 使用同一组 Annex B 压缩包，要求无 B 帧、一包一帧、顺序立即输出。
GPU 图像示例是亮度平面的 3×3 均值、归一化暗度和阈值，仅验证显存内计算，
不是微泡识别算法。核对模式主动下载结果；计时模式不下载任何完整图像。
"""

import argparse
import csv
import ctypes
import hashlib
import json
import platform
import sys
import time
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import av
import cv2
import numpy as np


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_packets(path):
    """先读取压缩包，计时不包含读盘；不提前解码完整图像。"""
    packets = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context
        if codec.name != "h264" or codec.has_b_frames:
            raise ValueError("Only H.264 without B-frames is allowed in this test")
        metadata = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": stream.width,
            "height": stream.height,
            "codec": codec.name,
            "pixel_format": codec.format.name,
            "fps": str(stream.average_rate),
            "time_base": str(stream.time_base),
            "has_b_frames": codec.has_b_frames,
        }
        bitstream = av.BitStreamFilterContext("h264_mp4toannexb", stream)
        for packet in container.demux(stream):
            if not packet.size:
                continue
            for converted in bitstream.filter(packet):
                if converted.pts is None:
                    raise ValueError("Missing packet timestamp")
                packets.append(
                    {
                        "data": np.frombuffer(bytes(converted), dtype=np.uint8),
                        "pts": converted.pts,
                        "time_s": float(converted.pts * stream.time_base),
                        "key": converted.is_keyframe,
                    }
                )
        if bitstream.filter(None):
            raise ValueError("Unexpected delayed bitstream-filter output")
    if not packets or not packets[0]["key"]:
        raise ValueError("A key frame is required at the beginning")
    if any(b["time_s"] <= a["time_s"] for a, b in pairwise(packets)):
        raise ValueError("Packet times must increase without reordering")
    metadata["packets"] = len(packets)
    metadata["compressed_bytes"] = sum(p["data"].size for p in packets)
    return packets, metadata


def cpu_decoder():
    """与正式 WSL 接收端相同：PyAV H.264、单解码线程。"""
    decoder = av.CodecContext.create("h264", "r")
    decoder.thread_count = 1
    return decoder


def decode_cpu(decoder, packet, time_base):
    data = av.Packet(packet["data"])
    data.pts = data.dts = packet["pts"]
    data.time_base = time_base
    frames = decoder.decode(data)
    if len(frames) != 1:
        raise ValueError(f"Expected one immediate CPU frame, received {len(frames)}")
    frame = frames[0]
    if frame.pict_type == av.video.frame.PictureType.B:
        raise ValueError("B-frames are not allowed")
    if frame.pts != packet["pts"]:
        raise ValueError("CPU frame timestamp does not match its packet")
    return frame


def cpu_luma(frame):
    """只建立原解码亮度平面的视图，包含真实行跨度。"""
    return np.ndarray(
        (frame.height, frame.width),
        dtype=np.uint8,
        buffer=frame.planes[0],
        strides=(frame.planes[0].line_size, 1),
    )


def cpu_image(luma, sums, darkness, mask):
    """与 CUDA 示例相同的边缘延拓和整数阈值；复用输出数组。"""
    cv2.boxFilter(
        luma,
        cv2.CV_32S,
        (3, 3),
        dst=sums,
        normalize=False,
        borderType=cv2.BORDER_REPLICATE,
    )
    np.divide(sums, np.float32(2295), out=darkness, casting="unsafe")
    np.subtract(np.float32(1), darkness, out=darkness)
    np.less(sums, 1377, out=mask)


class GpuDecoder:
    """单线程持有解码器及 CUDA 工作区；每帧用完后才允许下一次解码。"""

    def __init__(self, shape):
        import cupy as cp
        import PyNvVideoCodec as nvc

        self.cp, self.nvc = cp, nvc
        cp.cuda.Device(0).use()
        cp.zeros(1)  # 创建同一设备的 primary context。
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.decoder = nvc.CreateDecoder(
            gpuid=0,
            codec=nvc.cudaVideoCodec.H264,
            cudacontext=cp.cuda.driver.ctxGetCurrent(),
            cudastream=self.stream.ptr,
            usedevicememory=True,
            latency=nvc.DisplayDecodeLatencyType.ZERO,
        )
        self.shape = shape
        with self.stream:
            self.darkness = cp.empty(shape, cp.float32)
            self.mask = cp.empty(shape, cp.uint8)
            self.darkness.fill(0)
            self.mask.fill(0)
        self.kernel = cp.RawKernel(
            r"""
        extern "C" __global__ void image_demo(
            const unsigned char* src, long long stride, int width, int height,
            float* darkness, unsigned char* mask) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= width * height) return;
            int x = i % width, y = i / width, sum = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = min(max(y + dy, 0), height - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = min(max(x + dx, 0), width - 1);
                    sum += src[yy * stride + xx];
                }
            }
            darkness[i] = 1.0f - (float)sum / 2295.0f;
            mask[i] = sum < 1377;
        }
        """,
            "image_demo",
            options=("--fmad=false",),
        )
        self.kernel.compile()
        self.stream.synchronize()

    def decode(self, packet):
        """显式等待解码完成；返回的图像仍在显存，没有调用 get/asnumpy。"""
        data = self.nvc.PacketData()
        data.bsl = packet["data"].size
        data.bsl_data = packet["data"].ctypes.data
        data.pts = data.dts = packet["pts"]
        data.decode_flag = int(self.nvc.VideoPacketFlag.ENDOFPICTURE) | int(
            self.nvc.VideoPacketFlag.TIMESTAMP
        )
        frames = self.decoder.Decode(data)
        self.decoder.SyncOnCUStream()
        self.stream.synchronize()
        if len(frames) != 1:
            raise ValueError(
                f"Expected one immediate GPU frame, received {len(frames)}"
            )
        return frames[0]

    def view(self, frame):
        """DLPack 共享原显存，逐帧检查地址相同，绝不静默复制至 CPU。"""
        with self.stream:
            array = self.cp.from_dlpack(frame)
        if array.data.ptr != frame.GetPtrToPlane(0):
            raise ValueError("DLPack did not share the decoder's device address")
        height, width = self.shape
        if array.shape != (height * 3 // 2, width) or array.dtype != self.cp.uint8:
            raise ValueError(f"Unexpected NV12 layout: {array.shape}, {array.dtype}")
        return array

    def process(self, array):
        height, width = self.shape
        with self.stream:
            self.kernel(
                ((height * width + 255) // 256,),
                (256,),
                (
                    array,
                    np.int64(array.strides[0]),
                    np.int32(width),
                    np.int32(height),
                    self.darkness,
                    self.mask,
                ),
            )
        self.stream.synchronize()

    def close(self):
        if self.decoder is not None:
            self.stream.synchronize()
            data = self.nvc.PacketData()
            data.bsl, data.bsl_data = 0, 0
            data.decode_flag = int(self.nvc.VideoPacketFlag.ENDOFSTREAM)
            if self.decoder.Decode(data):
                raise ValueError("Unexpected delayed GPU frames at end of input")
            self.decoder.SyncOnCUStream()
            self.decoder = None


def wait_until(deadline):
    """固定计划时刻；记录实际晚到，不把等下一帧混入解码时间。"""
    remaining = deadline - time.perf_counter()
    if remaining > 0.001:
        time.sleep(remaining - 0.001)
    while time.perf_counter() < deadline:
        pass


def benchmark(args, output, packets, metadata):
    """逐帧同步完成、不提前解码未来画面；预热另记，正式从原第 0 帧开始。"""
    shape = (metadata["height"], metadata["width"])
    time_base = Fraction(metadata["time_base"])
    gpu = None
    prepared = time.perf_counter()
    cv2.setNumThreads(24)
    if args.engine == "gpu":
        gpu = GpuDecoder(shape)
    else:
        decoder = cpu_decoder()
        sums = np.zeros(shape, np.int32)
        darkness = np.zeros(shape, np.float32)
        mask = np.zeros(shape, bool)
    rows = []
    evidence = None
    profile_started = False
    audit = None
    try:
        # IDR 开头允许同一解码器重新开始；计时与核对均验证一包立即对应一帧。
        for packet in packets[:8]:
            if gpu:
                frame = gpu.decode(packet)
                array = gpu.view(frame)
                gpu.process(array)
                evidence = {
                    "device": list(frame.__dlpack_device__()),
                    "decoder_pointer": frame.GetPtrToPlane(0),
                    "cupy_pointer": array.data.ptr,
                    "shape": list(array.shape),
                    "strides": list(array.strides),
                    "allocation_type": gpu.cp.cuda.runtime.pointerGetAttributes(
                        array.data.ptr
                    ).type,
                }
                del array, frame
            else:
                frame = decode_cpu(decoder, packet, time_base)
                cpu_image(cpu_luma(frame), sums, darkness, mask)
        preparation_ms = (time.perf_counter() - prepared) * 1000
        if args.profile:
            if gpu is None:
                raise ValueError("CUDA profiling is only available with --engine gpu")
            gpu.cp.cuda.profiler.start()
            profile_started = True
        if args.copy_audit:
            audit = ctypes.CDLL(str(args.copy_audit.resolve()))
            audit.audit_start.argtypes = [ctypes.c_char_p]
            audit.audit_start.restype = ctypes.c_int
            if audit.audit_start(str(output / "cuda_copies.csv").encode()) != 0:
                raise RuntimeError("CUDA copy observer could not start")
        origin = time.perf_counter() + (0.1 if args.paced else 0)
        for index, packet in enumerate(packets):
            if args.frames and index >= args.frames:
                break
            planned = origin + packet["time_s"] - packets[0]["time_s"]
            if args.paced:
                wait_until(planned)
            start = time.perf_counter()
            if gpu:
                frame = gpu.decode(packet)
                decoded = time.perf_counter()
                array = gpu.view(frame)
                imported = time.perf_counter()
                gpu.process(array)
                done = time.perf_counter()
                decoder_pts = frame.getPTS()
                del array, frame
            else:
                frame = decode_cpu(decoder, packet, time_base)
                decoded = time.perf_counter()
                luma = cpu_luma(frame)
                imported = time.perf_counter()
                cpu_image(luma, sums, darkness, mask)
                done = time.perf_counter()
                decoder_pts = frame.pts
            rows.append(
                {
                    "frame": index,
                    "source_pts": packet["pts"],
                    "source_time_s": packet["time_s"],
                    "decoder_pts": decoder_pts,
                    "compressed_bytes": packet["data"].size,
                    "planned_s": planned if args.paced else "",
                    "submitted_s": start,
                    "decoded_s": decoded,
                    "finished_s": done,
                    "late_ms": max(0, start - planned) * 1000 if args.paced else "",
                    "decode_ready_ms": (decoded - start) * 1000,
                    "view_ms": (imported - decoded) * 1000,
                    "image_ms": (done - imported) * 1000,
                    "packet_to_image_ms": (done - start) * 1000,
                    "scheduled_to_image_ms": (done - planned) * 1000
                    if args.paced
                    else "",
                }
            )
        if profile_started:
            gpu.cp.cuda.profiler.stop()
            profile_started = False
        save_json(
            output / "run.json",
            {
                "engine": args.engine,
                "paced": args.paced,
                "frames": len(rows),
                "preparation_ms": preparation_ms,
                "warmup_frames": 8,
                "full_image_downloads_during_timing": 0,
                "gpu_memory": evidence,
                "scope": "Packet submission to native decode completion, then identical image demo; no tracking, RTSP or cross-system pipeline",
            },
        )
    finally:
        if audit is not None:
            audit.audit_stop()
        if profile_started:
            gpu.cp.cuda.profiler.stop()
        if gpu:
            gpu.close()
        with (output / "frames.csv").open("x", newline="") as file:
            if rows:
                writer = csv.DictWriter(file, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def verify(output, packets, metadata):
    """全帧核对原生 YUV 及 GPU 运算；下载在此显式执行，不计入速度对比。"""
    from customer_workspace.microbubble_tracking.frame_conversion import (
        ReusableConversion,
    )

    shape = (metadata["height"], metadata["width"])
    decoder, gpu = cpu_decoder(), GpuDecoder(shape)
    sums, darkness, mask = (
        np.zeros(shape, np.int32),
        np.zeros(shape, np.float32),
        np.zeros(shape, bool),
    )
    rows, preview = [], None
    legacy = None
    try:
        for packet in packets[:8]:
            decode_cpu(decoder, packet, Fraction(metadata["time_base"]))
            frame = gpu.decode(packet)
            array = gpu.view(frame)
            gpu.process(array)
            del array, frame
        for index, packet in enumerate(packets):
            raw = decode_cpu(decoder, packet, Fraction(metadata["time_base"]))
            frame = gpu.decode(packet)
            array = gpu.view(frame)
            gpu.process(array)
            native = array.get(stream=gpu.stream)
            actual_darkness = gpu.darkness.get(stream=gpu.stream)
            actual_mask = gpu.mask.get(stream=gpu.stream)
            expected = raw.to_ndarray(format="nv12")
            cpu_image(cpu_luma(raw), sums, darkness, mask)
            if legacy is None:
                legacy = ReusableConversion(raw, threads=1)
            legacy.convert_bgr(raw)
            gray = legacy.to_gray()
            # 原追踪输入为 YUV→BGR→灰度，不能把未经转换的 Y 平面冒充它。
            difference = native[: shape[0]].astype(np.int16) - gray
            row = {
                "frame": index,
                "source_pts": packet["pts"],
                "gpu_pts": frame.getPTS(),
                "native_different_pixels": int(np.count_nonzero(native != expected)),
                "native_max_error": int(
                    np.max(np.abs(native.astype(np.int16) - expected))
                ),
                "image_max_error": float(np.max(np.abs(actual_darkness - darkness))),
                "mask_different_pixels": int(np.count_nonzero(actual_mask != mask)),
                "native_y_vs_legacy_gray_mean_abs": float(np.mean(np.abs(difference))),
                "native_y_vs_legacy_gray_max_abs": int(np.max(np.abs(difference))),
                "shared_device_address": array.data.ptr == frame.GetPtrToPlane(0),
            }
            rows.append(row)
            if index == 300:
                preview = (native[: shape[0]].copy(), gray.copy(), actual_mask.copy())
            legacy.release()
            del array, frame
    finally:
        gpu.close()
        if legacy is not None:
            legacy.close()
        with (output / "pixels.csv").open("x", newline="") as file:
            if rows:
                writer = csv.DictWriter(file, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = {
        "warmup_frames": 8,
        "frames": len(rows),
        "native_different_pixels": sum(r["native_different_pixels"] for r in rows),
        "native_max_error": max(r["native_max_error"] for r in rows),
        "image_max_error": max(r["image_max_error"] for r in rows),
        "mask_different_pixels": sum(r["mask_different_pixels"] for r in rows),
        "all_addresses_shared": all(r["shared_device_address"] for r in rows),
        "decoder_timestamp_matches": sum(r["gpu_pts"] == r["source_pts"] for r in rows),
        "native_y_vs_legacy_gray_mean_abs": float(
            np.mean([r["native_y_vs_legacy_gray_mean_abs"] for r in rows])
        ),
        "native_y_vs_legacy_gray_max_abs": max(
            r["native_y_vs_legacy_gray_max_abs"] for r in rows
        ),
    }
    save_json(output / "verification.json", summary)
    if preview:
        for name, image in zip(("native_y", "legacy_gray", "gpu_demo_mask"), preview):
            cv2.imwrite(
                str(output / f"{name}.png"),
                image * 255 if name.endswith("mask") else image,
            )
    if (
        summary["native_different_pixels"]
        or summary["mask_different_pixels"]
        or summary["image_max_error"] > 1e-6
    ):
        raise ValueError("GPU pixel equivalence check failed; see verification.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bench", "verify"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="0 means all frames")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="CUDA profiler capture around measured frames only",
    )
    parser.add_argument(
        "--copy-audit", type=Path, help="Optional CUPTI copy observer library"
    )
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    if args.copy_audit and (
        args.profile or args.engine != "gpu" or args.mode != "bench"
    ):
        parser.error("Copy audit needs a GPU bench without a second CUDA profiler")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    packets, metadata = load_packets(args.video)
    save_json(args.output_dir / "input.json", metadata)
    save_json(
        args.output_dir / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "os": platform.platform(),
            "av": av.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    )
    try:
        if args.mode == "verify":
            verify(args.output_dir, packets, metadata)
        else:
            benchmark(args, args.output_dir, packets, metadata)
    except BaseException as exc:
        save_json(
            args.output_dir / "failure.json",
            {"type": type(exc).__name__, "error": str(exc)},
        )
        raise
    print(args.output_dir, "completed", flush=True)


if __name__ == "__main__":
    main()
