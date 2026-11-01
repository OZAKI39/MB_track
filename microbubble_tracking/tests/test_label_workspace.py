"""区域编号复用：像素/轮廓不变，跨帧结果和未保存快照不被覆盖。"""

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
from customer_workspace.microbubble_tracking.tests.test_online_tracking import (
    image,
    rows,
    stamp,
)


def test_owned_and_borrowed_labels_match_reference_including_area_rejection():
    background = np.full((131, 110), 200, np.float32)
    workspace = SegmentationWorkspace(background)
    output = np.empty(background.shape, np.uint16)
    params = SegParams(close=1, min_area=30, blur=0.1)
    kernel = np.ones((1, 1), np.uint8)
    # 5×5 区域能通过轮廓粗筛（16 >= 15），但会被像素面积筛除（25 < 30）。
    mixed = background.copy()
    mixed[10:15, 10:15] = 40
    mixed[40:55, 40:60] = 40
    mixed[125:130, 100:105] = 40
    tiny = background.copy()
    tiny[10:15, 10:15] = 40
    border = background.copy()
    border[:15, :20] = 40
    border[4:9, 4:9] = 200  # 保留原来的填孔规则。
    saved = []
    pointers = [
        a.ctypes.data
        for a in (output, workspace.selection_mask, workspace._label_indices)
    ]
    for frame in (mixed, tiny, border, background, mixed):
        expected, expected_contours, expected_blurred = segment(
            frame, background, params, kernel
        )
        owned, owned_contours, owned_blurred = segment(
            frame, background, params, kernel, workspace=workspace
        )
        actual, contours, blurred = segment(
            frame, background, params, kernel, workspace=workspace, labels_out=output
        )
        assert actual is output
        assert np.array_equal(actual, expected)
        assert np.array_equal(owned, expected)
        assert np.array_equal(blurred, expected_blurred)
        assert np.array_equal(owned_blurred, expected_blurred)
        assert len(contours) == len(owned_contours) == len(expected_contours)
        for a, b, c in zip(contours, owned_contours, expected_contours):
            assert np.array_equal(a, c) and np.array_equal(b, c)
            assert not np.shares_memory(a, output)
            assert not np.shares_memory(a, workspace.selection_mask)
        saved.append((owned, owned.copy()))
        assert pointers == [
            a.ctypes.data
            for a in (output, workspace.selection_mask, workspace._label_indices)
        ]
        assert all(np.array_equal(old, snapshot) for old, snapshot in saved)
        if frame is mixed:
            assert len(contours) == 1 and actual.max() == 1
        if frame is tiny or frame is background:
            assert not contours and not actual.any()


def test_remap_covers_full_uint16_range_and_partial_last_band():
    workspace = SegmentationWorkspace(np.zeros((133, 79), np.float32))
    lookup = np.arange(65536, dtype=np.uint16)[::-1].copy()
    rng = np.random.default_rng(31)
    labels = rng.integers(0, 65536, (133, 79), dtype=np.uint16)
    labels[0, :3] = (0, 256, 65535)
    expected = lookup[labels]
    pointer = labels.ctypes.data
    workspace.remap_labels(labels, lookup)
    assert np.array_equal(labels, expected)
    assert labels.ctypes.data == pointer
    assert workspace._label_indices.dtype == np.intp
    assert workspace._label_indices.shape == (64, 79)


@pytest.mark.parametrize("kind", ["dtype", "shape", "strided", "readonly", "alias"])
def test_invalid_output_is_rejected_without_modifying_inputs(kind):
    background = np.full((80, 80), 200, np.float32)
    frame = background.copy()
    workspace = SegmentationWorkspace(background)
    output = np.zeros(frame.shape, np.uint16)
    if kind == "dtype":
        output = np.zeros(frame.shape, np.int32)
    elif kind == "shape":
        output = np.zeros((79, 80), np.uint16)
    elif kind == "strided":
        output = np.zeros((80, 160), np.uint16)[:, ::2]
    elif kind == "readonly":
        output.flags.writeable = False
    else:
        output = frame.view(np.uint16).reshape(-1)[: frame.size].reshape(frame.shape)
    before = frame.copy()
    with pytest.raises(ValueError, match="labels_out"):
        segment(
            frame,
            background,
            SegParams(),
            np.ones((5, 5), np.uint8),
            workspace=workspace,
            labels_out=output,
        )
    assert np.array_equal(frame, before)


def test_pending_save_and_returned_contours_survive_label_buffer_reuse(
    tmp_path, monkeypatch
):
    entered, release = block_first_write(monkeypatch)
    processor = async_processor(tmp_path, queue_frames=32, max_pending_s=5)
    retained = []
    pointer = processor._labels.ctypes.data
    try:
        for index in range(10):
            result = processor.process(
                image((12 + index * 3, 25, 12, 12)), stamp(index)
            )
            assert processor._labels.ctypes.data == pointer
            for track in processor.tracker.tracks.values():
                assert not np.shares_memory(track.xs, processor._labels)
                assert not np.shares_memory(track.ys, processor._labels)
            contour = result["contours"][0][1]
            retained.append((result, contour.copy()))
        assert entered.wait(1)
        assert not rows(processor.output, "SELECT * FROM frames")
        # 所有绘制/追踪已用完整图，但保存线程仍未处理第 1 帧。
        processor._labels.fill(65535)
        assert all(
            np.array_equal(result["contours"][0][1], expected)
            for result, expected in retained
        )
    finally:
        release.set()
        processor.close()
    saved = rows(processor.output, "SELECT * FROM observations ORDER BY source_frame")
    assert len(saved) == len(retained) == 10
    for row, (result, expected) in zip(saved, retained):
        assert row["cx"] == result["observations"][0]["cx"]
        assert row["contour_i32"] == expected.astype("<i4").tobytes()
    assert processor.summary["persistence"]["pending_records"] == 0
