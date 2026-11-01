"""暗度缓存必须逐像素等于原式；覆盖更新行、阈值附近和跨帧数组复用。"""

import cv2
import numpy as np
import pytest
from customer_workspace.microbubble_tracking.bubble_tracker import (
    OnlineBackground,
    SegmentationWorkspace,
    SegParams,
    segment,
)


def reference(blurred, background):
    darkness = 1.0 - blurred / (background + 1.0)
    darkness -= np.median(darkness[::8, ::8])
    return darkness


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(1, 1), (13, 19), (105, 83)])
def test_matches_original_bits_and_preserves_inputs(dtype, shape):
    random = np.random.default_rng(103)
    background = random.uniform(0, 255, shape).astype(dtype)
    cache = SegmentationWorkspace(background)
    pointer = None
    original_background = background.copy()
    for _ in range(8):
        blurred = random.uniform(0, 255, shape).astype(dtype)
        before = blurred.copy()
        actual = cache.compute(blurred)
        assert actual.tobytes() == reference(blurred, background).tobytes()
        assert np.array_equal(blurred, before)
        assert np.array_equal(background, original_background)
        assert not np.shares_memory(actual, blurred)
        assert not np.shares_memory(actual, background)
        if pointer is not None:
            assert actual.ctypes.data == pointer
        pointer = actual.ctypes.data


def test_noncontiguous_inputs_and_explicit_partial_refresh():
    random = np.random.default_rng(23)
    background = random.uniform(0, 255, (101, 154)).astype(np.float32)[:, ::2]
    blurred = random.uniform(0, 255, (101, 154)).astype(np.float32)[:, ::2]
    cache = SegmentationWorkspace(background)
    background[10:29] += np.float32(0.25)
    cache.refresh_rows(10, 29)
    assert cache.compute(blurred).tobytes() == reference(blurred, background).tobytes()


def test_threshold_neighbours_remain_identical():
    background = np.full((81, 83), 199.0, np.float32)
    blurred = background + 1.0
    candidates = []
    for threshold in (0.06, 0.12):
        value = np.float32(200 * (1 - threshold))
        candidates.extend(
            (
                np.nextafter(value, np.float32(-np.inf)),
                value,
                np.nextafter(value, np.float32(np.inf)),
            )
        )
    blurred[20:23, 20:22] = np.array(candidates).reshape(3, 2)
    expected = reference(blurred, background)
    actual = SegmentationWorkspace(background).compute(blurred)
    assert actual.tobytes() == expected.tobytes()
    for threshold in (0.06, 0.12):
        assert np.array_equal(actual > threshold, expected > threshold)


def test_online_update_keeps_entire_cache_current_across_uneven_bands():
    random = np.random.default_rng(56)
    bg = OnlineBackground(random.uniform(50, 200, (103, 79)).astype(np.float32))
    labels = np.zeros(bg.B.shape, np.uint16)
    labels[20:40, 20:50] = 1
    for number in range(43):
        blurred = random.uniform(30, 230, bg.B.shape).astype(np.float32)
        before = bg.B.copy()
        bg.update(blurred, labels)
        start = (number % 10) * bg.B.shape[0] // 10
        stop = (number % 10 + 1) * bg.B.shape[0] // 10
        assert np.array_equal(bg.B[:start], before[:start])
        assert np.array_equal(bg.B[stop:], before[stop:])
        assert (
            bg.workspace.compute(blurred).tobytes()
            == reference(blurred, bg.B).tobytes()
        )


def test_segment_outputs_are_independent_of_reused_darkness():
    background = np.full((90, 110), 190, np.float32)
    cache = SegmentationWorkspace(background)
    params = SegParams()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.close, params.close))
    first = background.copy()
    first[20:35, 25:60] = 50
    expected_labels, expected_contours, expected_blurred = segment(
        first, background, params, kernel
    )
    labels, contours, blurred = segment(
        first, background, params, kernel, workspace=cache
    )
    second = background.copy()
    second[40:60, 65:90] = 70
    segment(second, background, params, kernel, workspace=cache)
    assert np.array_equal(labels, expected_labels)
    assert np.array_equal(blurred, expected_blurred)
    assert len(contours) == len(expected_contours)
    assert all(np.array_equal(a, b) for a, b in zip(contours, expected_contours))
    with pytest.raises(ValueError, match="different background"):
        segment(first, background.copy(), params, kernel, workspace=cache)


def test_incompatible_cache_inputs_fail_instead_of_silently_changing_precision():
    cache = SegmentationWorkspace(np.zeros((10, 10), np.float32))
    for frame in (np.zeros((10, 10), np.float64), np.zeros((10, 11), np.float32)):
        with pytest.raises(ValueError, match="shape and dtype"):
            cache.compute(frame)
    for start, stop in ((-1, 5), (3, 2), (0, 11)):
        with pytest.raises(ValueError, match="update rows"):
            cache.refresh_rows(start, stop)
    with pytest.raises(ValueError, match="floating-point"):
        SegmentationWorkspace(np.zeros((10, 10), np.uint8))
