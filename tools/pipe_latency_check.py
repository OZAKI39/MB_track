"""用已备好的真实 H.264 包测 Windows→WSL 管道；不等待下一帧、不解码。"""

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from customer_workspace.tracking_latency import (
    clock_description,
    forward_bounds_ms,
    parse_ack,
    send_ack,
)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def prepare_packets(video, fixture):
    """只更换 H.264 包的边界形式，不重编码；准备过程不计入传输时间。"""
    import av

    hashes, sizes = [], []
    with av.open(str(video)) as container, fixture.open("xb") as target:
        stream = container.streams.video[0]
        if stream.codec_context.name != "h264":
            raise ValueError("This benchmark requires an H.264 source")
        bitstream = av.bitstream.BitStreamFilterContext("h264_mp4toannexb", stream)
        for packet in container.demux(stream):
            for filtered in bitstream.filter(packet):
                payload = bytes(filtered)
                target.write(struct.pack("<I", len(payload)))
                target.write(payload)
                sizes.append(len(payload))
                hashes.append(hashlib.sha256(payload).hexdigest())
    return {"source": str(video.resolve()), "sizes": sizes, "sha256": hashes}


def windows_server(fixture, output, warmup):
    """所有包先装入内存，再宣布就绪；每次 GET 都有现成的数据。"""
    packets = []
    with fixture.open("rb") as source:
        while raw := source.read(4):
            size = struct.unpack("<I", raw)[0]
            payload = source.read(size)
            if len(payload) != size:
                raise ValueError("Truncated fixture")
            packets.append(payload)
    sys.stdout.buffer.write(b"READY\n")
    sys.stdout.buffer.flush()
    rows, acknowledgements = [], []
    try:
        for seq in range(warmup + len(packets)):
            if sys.stdin.buffer.readline() != b"GET\n":
                raise ValueError("Expected GET")
            get_started = time.monotonic()
            source_index = (seq - warmup) % len(packets)
            payload = packets[source_index]
            header = {
                "protocol": 2,
                "status": "FRAME",
                "pixel_format": "H264",
                "shape": [1024, 1280],
                "seq": seq,
                "payload_bytes": len(payload),
                "video_frame": source_index,
                "video_time_s": source_index / 60,
                "source_time_s": source_index / 60,
                "in_window": seq >= warmup,
                "experiment_id": "ready-payload-latency",
                "stream_id": "h264",
                "received_windows_mono": get_started,
                "sender_frame_age_ms": 0,
            }
            raw = json.dumps(header, separators=(",", ":")).encode()
            started = time.monotonic()
            sys.stdout.buffer.write(b"CV2FRAME" + struct.pack("<I", len(raw)) + raw)
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
            finished = time.monotonic()
            command = sys.stdin.buffer.readline()
            acknowledged = time.monotonic()
            if not command.startswith(b"ACK "):
                raise ValueError("Expected received acknowledgement")
            ack = parse_ack(command, acknowledged)
            if ack["seq"] != seq or ack["kind"] != "received":
                raise ValueError("Acknowledgement does not match frame")
            rows.append(
                {
                    "seq": seq,
                    "source_frame": source_index,
                    "measured": seq >= warmup,
                    "payload_bytes": len(payload),
                    "header_bytes": len(raw) + 12,
                    "get_started_windows_s": get_started,
                    "write_start_windows_s": started,
                    "write_end_windows_s": finished,
                }
            )
            acknowledgements.append(ack)
        if sys.stdin.buffer.readline() != b"QUIT\n":
            raise ValueError("Expected QUIT")
    finally:
        save_json(
            output,
            {
                "clock": clock_description(),
                "sent": rows,
                "acknowledgements": acknowledgements,
            },
        )


def run(video, output, warmup):
    from customer_workspace.parallel_camera_bridge import read_exact
    from customer_workspace.wsl_launcher import windows_path
    from microbubble_experiment.configuration import ROOT

    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "h264_packets.bin"
    metadata = prepare_packets(video, fixture)
    save_json(output / "input.json", metadata)
    command = [
        str(ROOT / "customer_workspace/.venv_gstreamer/Scripts/python.exe"),
        "-X",
        "utf8",
        "-u",
        "-B",
        "-m",
        "customer_workspace.tools.pipe_latency_check",
        "--windows-server",
        "--fixture",
        windows_path(fixture),
        "--output",
        windows_path(output / "windows.json"),
        "--warmup",
        str(warmup),
    ]
    lock = threading.Lock()
    rows = []
    with (output / "windows.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            bufsize=0,
        )
        try:
            ready = bytearray(6)
            read_exact(process.stdout, ready)
            if ready != b"READY\n":
                raise ValueError("Windows payload server did not become ready")
            for seq in range(warmup + len(metadata["sizes"])):
                requested = time.perf_counter()
                process.stdin.write(b"GET\n")
                prefix = bytearray(12)
                read_exact(process.stdout, prefix)
                prefix_received = time.perf_counter()
                if prefix[:8] != b"CV2FRAME":
                    raise ValueError("Invalid prefix")
                raw = bytearray(struct.unpack("<I", prefix[8:])[0])
                read_exact(process.stdout, raw)
                header = json.loads(raw)
                payload = bytearray(header["payload_bytes"])
                read_exact(process.stdout, payload)
                arrived = time.perf_counter()
                send_ack(
                    process.stdin,
                    lock,
                    "received",
                    seq,
                    {
                        "prefix_received_wsl_s": prefix_received,
                        "arrived_wsl_s": arrived,
                    },
                )
                # 完成计时后才校验字节；不把哈希计算算进管道。
                index = header["video_frame"]
                if hashlib.sha256(payload).hexdigest() != metadata["sha256"][index]:
                    raise ValueError("H.264 payload changed during transfer")
                rows.append(
                    {
                        "seq": seq,
                        "requested_wsl_s": requested,
                        "arrived_wsl_s": arrived,
                        "ready_transaction_ms": (arrived - requested) * 1000,
                    }
                )
            process.stdin.write(b"QUIT\n")
            process.wait(timeout=15)
            if process.returncode:
                raise RuntimeError(f"Windows server exit: {process.returncode}")
        finally:
            process.stdin.close()
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
    native = json.loads((output / "windows.json").read_text())
    for row, sent, ack in zip(
        rows, native["sent"], native["acknowledgements"], strict=True
    ):
        lower, upper = forward_bounds_ms(sent, ack)
        row.update(
            forward_lower_ms=lower,
            forward_upper_ms=upper,
            measured=sent["measured"],
            payload_bytes=sent["payload_bytes"],
        )
        if lower < 0 or upper + 0.001 < lower:
            raise ValueError("Causal timing bounds violated")
    save_json(
        output / "wsl.json",
        {"clock": clock_description(), "frames": rows, "all_payloads_identical": True},
    )
    print(f"Measured {len(metadata['sizes'])} ready H.264 packets: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--windows-server", action="store_true")
    args = parser.parse_args()
    if args.windows_server:
        windows_server(args.fixture, args.output, args.warmup)
    else:
        run(args.video, args.output, args.warmup)


if __name__ == "__main__":
    main()
