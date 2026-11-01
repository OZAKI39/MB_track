"""有限等待区：默认行为、先后顺序、满区替换、进程间所有权和结束排空。"""

import argparse
import multiprocessing as mp

import numpy as np
import pytest

from customer_workspace.parallel_camera_bridge import LatestFrames
from customer_workspace.parallel_video_check import add_tracking_arguments


def publish_number(frames, number):
    index, image = frames.acquire()
    image.fill(number)
    return frames.publish(index, {"seq": number, "source_time_s": number / 60})


def fill_in_child(frames, finished):
    for number in range(2, 32):
        publish_number(frames, number)
    finished.set()


def test_cli_keeps_one_waiting_frame_unless_explicitly_selected():
    parser = argparse.ArgumentParser()
    add_tracking_arguments(parser)
    assert parser.parse_args(["--background", "test.npy"]).waiting_frames == 1
    assert not parser.parse_args(["--background", "test.npy"]).diagnose_queue
    assert not parser.parse_args(["--background", "test.npy"]).diagnose_analysis
    assert (
        parser.parse_args(
            ["--background", "test.npy", "--waiting-frames", "2"]
        ).waiting_frames
        == 2
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--background", "test.npy", "--waiting-frames", "3"])


@pytest.mark.parametrize("capacity", [1, 2])
def test_optional_diagnostics_keep_timestamps_and_shared_view(capacity):
    frames = LatestFrames(mp.get_context("spawn"), (8, 12), capacity, True)
    publish_number(frames, 7)
    index, image, header = frames.take()
    detail = header["queue_detail"]
    assert detail["entered_wsl_s"] <= detail["taken_wsl_s"]
    assert detail["taken_wsl_s"] <= detail["take_returned_wsl_s"]
    assert np.shares_memory(image, np.frombuffer(frames.pixels[index], np.uint8))
    assert np.all(image == 7)
    frames.release(index)
    assert [event["operation"] for event in frames.producer_events] == [
        "acquire",
        "publish",
    ]
    assert [event["operation"] for event in frames.consumer_events] == [
        "take",
        "release",
    ]
    assert all(event["seq"] == 7 for event in frames.consumer_events)


def test_default_does_not_record_diagnostic_data():
    frames = LatestFrames(mp.get_context("spawn"), (8, 12))
    publish_number(frames, 1)
    index, _, header = frames.take()
    frames.release(index)
    assert "queue_detail" not in header
    assert frames.entered_times is None
    assert frames.producer_events == frames.consumer_events == []


@pytest.mark.parametrize("capacity", [0, 3, -1])
def test_unsupported_capacity_rejected(capacity):
    with pytest.raises(ValueError, match="capacity"):
        LatestFrames(mp.get_context("spawn"), (8, 12), capacity)


def test_two_waiting_frames_keep_order_and_drop_only_oldest_when_full():
    frames = LatestFrames(mp.get_context("spawn"), (8, 12), 2)
    assert publish_number(frames, 1) == (None, 1)
    borrowed, image, _ = frames.take()
    assert publish_number(frames, 2) == (None, 1)
    assert publish_number(frames, 3) == (None, 2)
    assert publish_number(frames, 4) == (2, 2)
    assert np.all(image == 1)
    assert frames.overwritten == 1
    frames.release(borrowed)
    for expected in (3, 4):
        index, current, header = frames.take()
        assert header["seq"] == expected
        assert header["source_time_s"] == expected / 60
        assert np.all(current == expected)
        frames.release(index)
    assert frames.take() is None
    assert frames.pending.value == frames.following.value == -1
    assert list(frames.states) == [0, 0, 0, 0]


@pytest.mark.parametrize("capacity", [1, 2])
def test_spawned_receiver_cannot_overwrite_analysis_and_tail_drains(capacity):
    context = mp.get_context("spawn")
    frames = LatestFrames(context, (8, 12), capacity)
    publish_number(frames, 1)
    borrowed, image, _ = frames.take()
    finished = context.Event()
    receiver = context.Process(target=fill_in_child, args=(frames, finished))
    receiver.start()
    try:
        assert finished.wait(10)
        receiver.join(5)
        assert receiver.exitcode == 0
        assert np.all(image == 1)
        assert frames.peak_waiting.value == capacity
        assert frames.overwritten == 30 - capacity
        frames.release(borrowed)
        for expected in range(32 - capacity, 32):
            index, current, header = frames.take()
            assert header["seq"] == expected
            assert np.all(current == expected)
            frames.release(index)
        assert frames.take() is None
        assert all(state == 0 for state in frames.states)
    finally:
        if receiver.is_alive():
            receiver.terminate()
            receiver.join(5)
