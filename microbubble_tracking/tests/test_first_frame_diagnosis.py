"""诊断准备不得消耗真实帧、改变背景或污染正式目标编号。"""

import numpy as np
import pytest

from customer_workspace.microbubble_tracking.first_frame_diagnosis import CONDITIONS, prepare
from customer_workspace.microbubble_tracking.online_tracking import FrameStamp, OnlineProcessor


def make_processor(output):
    return OnlineProcessor(
        experiment_id="diagnostic-test", stream_id="camera",
        source_clock="test", background=np.full((96, 128), 180, np.float32),
        nominal_fps=60, output_dir=output, save_mode="sync",
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_preparation_preserves_background_ids_and_subsequent_results(tmp_path, condition):
    image = np.full((96, 128), 180, np.uint8)
    image[24:42, 40:65] = 50
    with make_processor(tmp_path / "plain") as plain, make_processor(tmp_path / "ready") as ready:
        prepare(ready, image, condition)
        np.testing.assert_array_equal(ready.background.B, plain.background.B)
        assert ready.background.k == 0
        assert ready.last_stamp is None
        assert ready.tracker.next_id == 1
        assert not ready.tracker.tracks
        assert ready.store.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 0
        for frame in range(8):
            shifted = np.roll(image, frame, axis=1)
            stamp = FrameStamp("diagnostic-test", "camera", frame, frame / 60)
            expected = plain.process(shifted, stamp)
            actual = ready.process(shifted, stamp)
            assert actual["observations"] == expected["observations"]
            assert actual["events"] == expected["events"]
            assert actual["track_updates"] == expected["track_updates"]
            for (left_id, left), (right_id, right) in zip(
                actual["contours"], expected["contours"], strict=True
            ):
                assert left_id == right_id
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(ready.background.B, plain.background.B)


def test_preparation_rejects_active_processor_and_unknown_condition(tmp_path):
    image = np.full((96, 128), 180, np.uint8)
    with make_processor(tmp_path / "active") as processor:
        with pytest.raises(ValueError, match="Unknown"):
            prepare(processor, image, "invalid")
        processor.process(image, FrameStamp("diagnostic-test", "camera", 0, 0.0))
        with pytest.raises(ValueError, match="fresh processor"):
            prepare(processor, image, "analysis")
