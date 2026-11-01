"""预热必须保持原始像素、帧序号和时间，且在取消或失败时不发布测试帧。"""

import json
import threading
import time
import weakref
from types import SimpleNamespace

import av
import cv2
import numpy as np
import pytest

from customer_workspace.microbubble_tracking import paced_source


def video_frame(seed=0):
    pixels = np.random.default_rng(seed).integers(0, 256, (24, 32, 3), np.uint8)
    frame = av.VideoFrame.from_ndarray(pixels, format="bgr24").reformat(
        format="yuv420p"
    )
    frame.pts = seed * 1500
    return frame


def test_warmup_preserves_pixels_metadata_and_releases_temporary_images(
    tmp_path, monkeypatch
):
    frame = video_frame()
    expected = frame.to_ndarray(format="bgr24", threads=1).copy()
    original_metadata = (
        frame.pts,
        frame.format.name,
        frame.color_range,
        frame.colorspace,
    )
    references = []
    convert = cv2.cvtColor

    def observed_gray(*args, **kwargs):
        gray = convert(*args, **kwargs)
        references.append(weakref.ref(gray))
        return gray

    monkeypatch.setattr(paced_source.cv2, "cvtColor", observed_gray)
    report = paced_source.warm_up_conversion(
        frame, threading.Event(), 8, {"threads": 1}, tmp_path / "warmup.json"
    )
    assert report["status"] == "completed"
    assert report["iterations_completed"] == 8
    assert (
        frame.pts,
        frame.format.name,
        frame.color_range,
        frame.colorspace,
    ) == original_metadata
    np.testing.assert_array_equal(frame.to_ndarray(format="bgr24", threads=1), expected)
    assert all(reference() is None for reference in references)
    assert json.loads((tmp_path / "warmup.json").read_text())["duration_ms"] >= 0


def test_zero_iterations_does_not_convert(tmp_path):
    def forbidden(**kwargs):
        raise AssertionError("Disabled warm-up must not convert")

    frame = SimpleNamespace(
        width=32,
        height=24,
        format=SimpleNamespace(name="yuv420p"),
        pts=0,
        to_ndarray=forbidden,
    )
    result = paced_source.warm_up_conversion(
        frame, threading.Event(), 0, {}, tmp_path / "warmup.json"
    )
    assert result["status"] == "disabled"
    assert result["iterations_completed"] == 0


@pytest.mark.parametrize("count", [-1, 33, True, 1.5])
def test_warmup_rejects_invalid_count(tmp_path, count):
    with pytest.raises(ValueError, match="0 to 32"):
        paced_source.warm_up_conversion(
            video_frame(), threading.Event(), count, {}, tmp_path / "warmup.json"
        )


class RecordingSlot:
    def __init__(self):
        self.available = threading.Event()
        self.images = []
        self.metadata = []

    def put(self, image, metadata, *, diagnostics=None):
        self.images.append(image.copy())
        metadata = dict(metadata, published_s=time.perf_counter())
        self.metadata.append(metadata)
        return metadata, None


def source_inputs(tmp_path, monkeypatch, warmup_iterations=8):
    frames = [video_frame(index) for index in range(3)]
    yielded = []

    def recording(config):
        for index, frame in enumerate(frames):
            yielded.append(index)
            yield (
                index,
                frame,
                {
                    "source_seq": 274 + index,
                    "video_s": index / 60,
                    "received_windows_s": 1000 + index / 60,
                    "decoded_windows_s": 1001 + index / 60,
                    "replay_cycle": 0,
                    "original_video_frame": index,
                    "original_source_frame": 274 + index,
                    "original_video_s": index / 60,
                },
                0.0,
            )

    monkeypatch.setattr(paced_source, "recording_frames", recording)
    monkeypatch.setattr(paced_source, "calibrate_wait", lambda *args: {"ahead_ms": 1})
    monkeypatch.setattr(paced_source, "wait_for_deadline", lambda *args: True)
    config = {
        "output": str(tmp_path),
        "expected_frames": 3,
        "max_frames": 0,
        "conversion_warmup_iterations": warmup_iterations,
        "conversion_threads": 1,
    }
    ready, start, done, cancel = [threading.Event() for _ in range(4)]
    start.set()
    slot = RecordingSlot()
    return frames, yielded, config, slot, ready, start, done, cancel


def test_sender_warms_before_ready_without_consuming_or_publishing_extra_frames(
    tmp_path, monkeypatch
):
    frames, yielded, config, slot, ready, start, done, cancel = source_inputs(
        tmp_path, monkeypatch
    )
    original = paced_source.warm_up_conversion

    def observed_warmup(*args):
        assert not ready.is_set()
        assert slot.images == []
        assert yielded == [0]
        return original(*args)

    monkeypatch.setattr(paced_source, "warm_up_conversion", observed_warmup)
    paced_source.send_video(
        config, slot, ready, start, done, cancel, SimpleNamespace(value=10.0)
    )
    assert ready.is_set() and done.is_set()
    assert yielded == [0, 1, 2]
    assert [row["source_frame"] for row in slot.metadata] == [274, 275, 276]
    assert [row["video_frame"] for row in slot.metadata] == [0, 1, 2]
    assert [row["deadline_s"] for row in slot.metadata] == pytest.approx(
        [10 + i / 60 for i in range(3)]
    )
    for frame, actual in zip(frames, slot.images, strict=True):
        expected = cv2.cvtColor(
            frame.to_ndarray(format="bgr24", threads=1), cv2.COLOR_BGR2GRAY
        )
        np.testing.assert_array_equal(actual, expected)
    report = json.loads((tmp_path / "source_report.json").read_text())
    assert report["offered_frames"] == 3
    assert report["status"] == "completed"


@pytest.mark.parametrize("failure", ["cancel", "conversion_error"])
def test_failed_or_cancelled_warmup_never_starts_sender(tmp_path, monkeypatch, failure):
    _frames, yielded, config, slot, ready, start, done, cancel = source_inputs(
        tmp_path, monkeypatch
    )
    convert = cv2.cvtColor
    calls = 0

    def interrupted_conversion(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            if failure == "cancel":
                cancel.set()
            else:
                raise RuntimeError("test conversion failure")
        return convert(*args, **kwargs)

    monkeypatch.setattr(paced_source.cv2, "cvtColor", interrupted_conversion)
    paced_source.send_video(
        config, slot, ready, start, done, cancel, SimpleNamespace(value=10.0)
    )
    assert done.is_set() and not ready.is_set()
    assert not slot.images
    assert yielded == [0]
    report = json.loads((tmp_path / "source_report.json").read_text())
    warmup = json.loads((tmp_path / "source_warmup.json").read_text())
    assert report["offered_frames"] == 0
    expected = "cancelled" if failure == "cancel" else "failed"
    assert warmup["status"] == report["status"] == expected
