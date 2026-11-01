"""离线核对 Windows 相机路线输出与原 PyAV 灰度图，不接触设备。"""

import argparse
import json
import time
import uuid
from pathlib import Path

import av
import cv2
import numpy as np

from customer_workspace.microbubble_tracking.frame_conversion import ReusableConversion
from customer_workspace.parallel_camera_bridge import (
    TrackingWindowsBridge,
    decoder_packet,
)
from customer_workspace.streamcopy_recording import write_json


def run(video, output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "link_session.json",
        {
            "experiment_id": str(uuid.uuid4()),
            "stream_id": str(uuid.uuid4()),
            "input_kind": "offline_video",
        },
    )
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        decoded = next(container.decode(stream))
        expected = cv2.cvtColor(decoded.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
    bridge = TrackingWindowsBridge(
        output,
        1,
        expected.shape,
        camera_config=None,
        offline_video=video,
        recording=False,
    ).start()
    actual = index = None
    try:
        deadline = time.perf_counter() + 30
        while time.perf_counter() < deadline:
            bridge.poll()
            if bridge.error:
                raise RuntimeError(bridge.error)
            packet = bridge.frames.take()
            if packet is not None:
                slot, image, header = packet
                if actual is None:
                    actual = image.copy()
                    index = header["video_frame"]
                bridge.frames.release(slot)
            elif bridge.done.is_set():
                break
            else:
                time.sleep(0.001)
        else:
            raise TimeoutError("Input check timed out")
    finally:
        bridge.stop()
    if actual is None or index != 0:
        raise RuntimeError("Did not receive original frame 0")
    np.save(output / "received_gray.npy", actual)
    difference = actual.astype(int) - expected.astype(int)
    result = {
        "min_error": int(difference.min()),
        "max_error": int(difference.max()),
        "different_pixels": int(np.count_nonzero(difference)),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "gray_range_received": [int(actual.min()), int(actual.max())],
        "gray_range_expected": [int(expected.min()), int(expected.max())],
    }
    write_json(output / "pixel_check.json", result)
    print(json.dumps(result))


def check_camera_timestamps(video, output):
    """模拟历史真实 RTSP 的缺失 DTS 和非零 PTS，不改写原视频。"""
    output.mkdir(parents=True, exist_ok=False)
    decoder = av.CodecContext.create("h264", "r")
    decoder.thread_count = 1
    decoder.open()
    count = 0
    converter = None
    try:
        with av.open(str(video)) as compressed, av.open(str(video)) as original:
            stream = compressed.streams.video[0]
            annex_b = av.BitStreamFilterContext("h264_mp4toannexb", stream)
            expected_frames = original.decode(video=0)
            for packet in compressed.demux(stream):
                if packet.pts is None:
                    continue
                for unit in annex_b.filter(packet):
                    pts = 5183333334 + round(float(unit.pts * unit.time_base) * 1e9)
                    decoded = decoder.decode(
                        decoder_packet(
                            bytes(unit),
                            {
                                "pts_ns": pts,
                                "dts_ns": 2**64 - 1,
                            },
                        )
                    )
                    if len(decoded) != 1 or decoded[0].pts != pts:
                        raise ValueError("Camera-style source PTS was not preserved")
                    if converter is None:
                        converter = ReusableConversion(decoded[0], threads=1)
                    converter.convert_bgr(decoded[0])
                    gray = converter.to_gray()
                    expected = cv2.cvtColor(
                        next(expected_frames).to_ndarray(format="bgr24"),
                        cv2.COLOR_BGR2GRAY,
                    )
                    if not np.array_equal(gray, expected):
                        raise ValueError(
                            "Decoded camera-style pixels differ from original"
                        )
                    converter.release()
                    count += 1
                if count >= 30:
                    break
    finally:
        if converter is not None:
            if converter.in_use:
                converter.release()
            converter.close()
    if count != 30:
        raise ValueError("Fewer than 30 frames verified")
    result = {
        "frames_checked": count,
        "missing_dts_supported": True,
        "nonzero_source_pts_preserved": True,
        "pixel_difference": 0,
        "input": str(video),
        "scope": "simulated RTSP timestamp metadata, not live hardware",
    }
    write_json(output / "pixel_check.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-timestamps-only", action="store_true")
    args = parser.parse_args()
    if args.camera_timestamps_only:
        check_camera_timestamps(args.video, args.output)
    else:
        run(args.video, args.output)
