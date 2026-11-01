"""固定转换空间的像素、借用顺序、慢消费者及异常保护。"""

import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor

import av
import cv2
import numpy as np
import pytest

from customer_workspace.microbubble_tracking.frame_conversion import ReusableConversion
from customer_workspace.microbubble_tracking.paced_source import LatestFrame


def frame_for(seed, width=34, height=26):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (height, width, 3), np.uint8)
    return av.VideoFrame.from_ndarray(pixels, format="bgr24").reformat(format="yuv420p")


def stamp(index):
    return {
        "video_frame": index,
        "source_frame": 100 + index,
        "video_s": index / 60,
        "received_windows_s": 1000 + index / 60,
        "decoded_windows_s": 1001 + index / 60,
        "deadline_s": index / 60,
        "decode_ms": 0,
    }


@pytest.mark.parametrize("size", [(32, 24), (34, 26), (1280, 1024)])
@pytest.mark.parametrize("threads", [None, 1])
def test_pixels_match_pyav_and_addresses_are_stable(size, threads):
    first = frame_for(0, *size)
    converter = ReusableConversion(first, threads=threads)
    addresses = (converter.bgr.ctypes.data, converter.gray.ctypes.data)
    kwargs = {} if threads is None else {"threads": threads}
    try:
        for seed in range(6):
            frame = frame_for(seed, *size)
            expected = frame.to_ndarray(format="bgr24", **kwargs)
            bgr = converter.convert_bgr(frame)
            gray = converter.to_gray()
            np.testing.assert_array_equal(bgr, expected)
            np.testing.assert_array_equal(
                gray, cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)
            )
            assert (bgr.ctypes.data, gray.ctypes.data) == addresses
            assert frame.format.name == "yuv420p"
            assert frame.colorspace == 2 and frame.color_range == 0
            converter.release()
    finally:
        if converter.in_use:
            converter.release()
        converter.close()


def test_live_lease_blocks_overwrite_and_close():
    first, second = frame_for(1), frame_for(2)
    converter = ReusableConversion(first, threads=1)
    bgr = converter.convert_bgr(first)
    gray = converter.to_gray()
    expected_bgr, expected_gray = bgr.copy(), gray.copy()
    with pytest.raises(RuntimeError, match="still in use"):
        converter.convert_bgr(second)
    with pytest.raises(RuntimeError, match="in use"):
        converter.close()
    np.testing.assert_array_equal(bgr, expected_bgr)
    np.testing.assert_array_equal(gray, expected_gray)
    converter.release()
    with pytest.raises(RuntimeError, match="No frame"):
        converter.to_gray()
    with pytest.raises(RuntimeError, match="No frame"):
        converter.release()
    converter.close()
    converter.close()  # 收尾可以安全重入。
    with pytest.raises(RuntimeError, match="closed"):
        converter.convert_bgr(second)


@pytest.mark.parametrize("change", ["size", "format", "range", "colorspace"])
def test_changed_input_is_rejected_without_overwriting_buffer(change):
    first = frame_for(1)
    converter = ReusableConversion(first, threads=1)
    before = converter.bgr.copy()
    frame = frame_for(2, width=36 if change == "size" else 34)
    if change == "format":
        frame = frame.reformat(format="rgb24")
    if change == "range":
        frame.color_range = 2
    if change == "colorspace":
        frame.colorspace = 1
    with pytest.raises(ValueError, match="changed"):
        converter.convert_bgr(frame)
    np.testing.assert_array_equal(converter.bgr, before)
    assert not converter.in_use
    converter.close()


def test_different_thread_cannot_use_buffer():
    frame = frame_for(1)
    converter = ReusableConversion(frame, threads=1)
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(converter.convert_bgr, frame)
        with pytest.raises(RuntimeError, match="different thread"):
            future.result()
    assert not converter.in_use
    converter.close()


def test_slow_consumer_keeps_its_frame_while_sender_reuses_storage():
    """消费者停在上一帧读取期间，发送端继续转换；旧图和编号仍必须完整。"""
    first = frame_for(1)
    converter = ReusableConversion(first, threads=1)
    slot = LatestFrame(mp.get_context("spawn"), (first.height, first.width))
    converter.convert_bgr(first)
    original = converter.to_gray().copy()
    slot.put(converter.gray, stamp(0))
    converter.release()  # put 返回前已完成复制。
    reading, finish_reading = threading.Event(), threading.Event()

    def slow_reader():
        image, metadata = slot.take()
        reading.set()
        assert finish_reading.wait(5)
        np.testing.assert_array_equal(image, original)
        assert metadata["source_frame"] == 100

    with ThreadPoolExecutor(max_workers=1) as reader:
        future = reader.submit(slow_reader)
        assert reading.wait(5)
        try:
            for index in range(1, 6):
                converter.convert_bgr(frame_for(index + 1))
                gray = converter.to_gray()
                slot.put(gray, stamp(index))
                converter.release()
            latest, metadata = slot.take()
            np.testing.assert_array_equal(latest, converter.gray)
            assert metadata["source_frame"] == 105
        finally:
            finish_reading.set()
        future.result()
    converter.close()


def test_incomplete_native_conversion_is_not_published(monkeypatch):
    frame = frame_for(1)
    converter = ReusableConversion(frame, threads=1)
    monkeypatch.setattr(converter, "_scale", lambda *args: -1)
    with pytest.raises(RuntimeError, match="Incomplete conversion"):
        converter.convert_bgr(frame)
    assert not converter.in_use
    converter.close()


def test_startup_pixel_mismatch_stops_before_playback(monkeypatch):
    original = ReusableConversion.convert_bgr

    def wrong_pixels(self, frame):
        result = original(self, frame)
        result[0, 0, 0] ^= 1
        return result

    monkeypatch.setattr(ReusableConversion, "convert_bgr", wrong_pixels)
    with pytest.raises(ValueError, match="differs from PyAV"):
        ReusableConversion(frame_for(1), threads=1)


def test_unvalidated_color_range_requires_explicit_original_path():
    frame = frame_for(1)
    frame.color_range = 2
    with pytest.raises(ValueError, match="conversion-memory allocate"):
        ReusableConversion(frame, threads=1)
