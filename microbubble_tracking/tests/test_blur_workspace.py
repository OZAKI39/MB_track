"""平滑输出复用：逐位相同、独立结果兼容、错误缓冲拒绝及异步保存隔离。"""

import cv2
import numpy as np
import pytest

from customer_workspace.microbubble_tracking.bubble_tracker import (
    SegmentationWorkspace,
    SegParams,
    segment,
)
from customer_workspace.microbubble_tracking.tests.test_async_store import (
    async_processor,
    block_first_write,
)
from customer_workspace.microbubble_tracking.tests.test_online_tracking import image, rows, stamp


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("sigma", [0.1, 1.0, 2.5])
def test_blurred_destination_exact_and_independent_default(dtype, strided, sigma):
    background = np.full((81, 101), 200, dtype)
    workspace = SegmentationWorkspace(background)
    rng = np.random.default_rng(112)
    source = rng.uniform(0, 255, (81, 202 if strided else 101)).astype(dtype)
    frame = source[:, ::2] if strided else source
    original = frame.copy()
    frame.flags.writeable = False
    output = np.empty(background.shape, dtype)
    params = SegParams(blur=sigma)
    kernel = np.ones((5, 5), np.uint8)
    expected = segment(frame, background, params, kernel, workspace=workspace)
    actual = segment(frame, background, params, kernel, workspace=workspace,
                     blurred_out=output)
    assert actual[2] is output
    assert np.array_equal(actual[2].view(np.uint8), expected[2].view(np.uint8))
    assert np.array_equal(actual[0], expected[0])
    assert len(actual[1]) == len(expected[1])
    assert all(np.array_equal(a, b) for a, b in zip(actual[1], expected[1]))
    saved = expected[2].copy()
    segment(background.copy(), background, params, kernel, workspace=workspace,
            blurred_out=output)
    np.testing.assert_array_equal(expected[2], saved)
    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize("kind", [
    "dtype", "shape", "strided", "readonly", "input", "background", "darkness",
    "component_labels", "output_labels",
])
def test_invalid_blurred_storage_rejected_before_writing(kind):
    frame = np.full((80, 80), 200, np.float32)
    background = frame.copy()
    workspace = SegmentationWorkspace(background)
    labels = np.zeros(frame.shape, np.uint16)
    output = np.zeros_like(frame)
    if kind == "dtype":
        output = output.astype(np.float64)
    elif kind == "shape":
        output = output[:-1].copy()
    elif kind == "strided":
        output = np.zeros((80, 160), np.float32)[:, ::2]
    elif kind == "readonly":
        output.flags.writeable = False
    elif kind == "input":
        output = frame
    elif kind == "background":
        output = background
    elif kind == "darkness":
        output = workspace._darkness
    elif kind == "component_labels":
        output = workspace.component_labels.view(np.float32)
    elif kind == "output_labels":
        labels = output.view(np.uint16).reshape(-1)[:frame.size].reshape(frame.shape)
    original_frame, original_background = frame.copy(), background.copy()
    with pytest.raises(ValueError, match="blurred_out"):
        segment(frame, background, SegParams(), np.ones((5, 5), np.uint8),
                workspace=workspace, labels_out=labels, blurred_out=output)
    np.testing.assert_array_equal(frame, original_frame)
    np.testing.assert_array_equal(background, original_background)


def test_pending_writer_never_reads_reused_blur(tmp_path, monkeypatch):
    entered, release = block_first_write(monkeypatch)
    p = async_processor(tmp_path, queue_frames=32, max_pending_s=5)
    pointer = p._blurred.ctypes.data
    retained = []
    original_blur = cv2.GaussianBlur
    destinations = []

    def observed_blur(*args, **kwargs):
        destinations.append(kwargs.get("dst"))
        return original_blur(*args, **kwargs)

    monkeypatch.setattr(cv2, "GaussianBlur", observed_blur)
    try:
        p.start(image())
        assert len(destinations) == 3
        assert all(a is p._blurred for a in destinations)
        assert not p._blurred.any()
        for index in range(10):
            result = p.process(image((12 + index * 3, 25, 12, 12)), stamp(index))
            assert p._blurred.ctypes.data == pointer
            retained.append((result, result["contours"][0][1].copy()))
        assert len(destinations) == 13
        assert all(a is p._blurred for a in destinations)
        assert entered.wait(1)
        assert not rows(p.output, "SELECT * FROM frames")
        # 故意破坏工作图，再放行保存线程；若它持有图像引用，下面核对必然失败。
        p._blurred.fill(np.nan)
        for result, expected in retained:
            np.testing.assert_array_equal(result["contours"][0][1], expected)
    finally:
        release.set()
        p.close()
    saved = rows(p.output, "SELECT * FROM observations ORDER BY source_frame")
    assert len(saved) == len(retained) == 10
    for row, (result, expected) in zip(saved, retained):
        assert row["cx"] == result["observations"][0]["cx"]
        assert row["contour_i32"] == expected.astype("<i4").tobytes()
    assert p.summary["persistence"]["pending_records"] == 0
