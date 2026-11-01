"""核对录像完整性判定；只使用测试图像和时间记录，不连接任何设备。"""

from copy import deepcopy

import numpy as np
import pytest

from customer_workspace.video_recording_check import check_frame_times, check_video
from microbubble_experiment.camera_bridge import LiveCameraFrameSource
from microbubble_experiment.recording import VideoPair


def frame_rows():
    return [
        {
            "record_index": index,
            "source_mode": "LIVE_CAMERA",
            "source_session_id": "test-connection",
            "source_frame_id": index + 100,
            "arrival_monotonic": 1000 + elapsed,
            "host_decode_timestamp": {
                "clock": "windows_monotonic",
                "value": 500 + elapsed,
            },
            "host_receive_timestamp": {
                "clock": "wsl_monotonic",
                "value": 1000 + elapsed,
            },
            "elapsed_since_first_frame_s": elapsed,
            "pts_adjusted": False,
            "part": 1,
            "part_frame_index": index,
            "video_pts_us": round(elapsed * 1e6),
        }
        for index, elapsed in enumerate((0, 0.03, 0.07, 0.12))
    ]


def test_actual_variable_frame_intervals_are_preserved():
    result, table = check_frame_times(frame_rows(), 0.15)
    assert result["passed"]
    assert result["source_sequence_missing_frames"] == 0
    assert [row["录像时间_秒"] for row in table] == pytest.approx([0, 0.03, 0.07, 0.12])
    assert result["arrival_intervals"]["maximum_ms"] == pytest.approx(50)


def test_skipped_decoded_frames_are_not_called_complete():
    rows = frame_rows()
    for row in rows[2:]:
        row["source_frame_id"] += 3
    result, table = check_frame_times(rows, 0.15)
    assert not result["passed"]
    assert result["source_sequence_missing_frames"] == 3
    assert table[2]["缺少的来源帧数"] == 3


@pytest.mark.parametrize(
    "problem",
    [
        "reconnect",
        "duplicate",
        "decode_backwards",
        "receive_backwards",
        "missing_clock",
        "pts_adjusted",
        "wrong_source",
    ],
)
def test_invalid_time_or_identity_is_rejected(problem):
    rows = frame_rows()
    row = rows[2]
    if problem == "reconnect":
        row["source_session_id"] = "new-connection"
    elif problem == "duplicate":
        row["source_frame_id"] = rows[1]["source_frame_id"]
    elif problem == "decode_backwards":
        row["host_decode_timestamp"]["value"] = 499
    elif problem == "receive_backwards":
        row["arrival_monotonic"] = 999
    elif problem == "missing_clock":
        row["host_decode_timestamp"] = {}
    elif problem == "pts_adjusted":
        row["pts_adjusted"] = True
    else:
        row["source_mode"] = "REALTIME_REPLAY"
    result, _ = check_frame_times(rows, 0.15)
    assert not result["passed"] and result["problems"]


def test_short_recording_cannot_pass_twenty_seconds():
    result, _ = check_frame_times(frame_rows(), 20)
    assert not result["passed"]
    assert any("未覆盖" in problem for problem in result["problems"])


def test_gap_is_reported_instead_of_filled():
    rows = frame_rows()
    for row in rows[2:]:
        row["arrival_monotonic"] += 0.2
        row["host_receive_timestamp"]["value"] += 0.2
        row["elapsed_since_first_frame_s"] += 0.2
    result, table = check_frame_times(rows, 0.35)
    assert not result["passed"]
    assert len(table) == 4
    assert result["discontinuities"] == [
        {"frame": 3, "clock": "wsl_receive", "interval_s": pytest.approx(0.24)}
    ]


def test_no_frames_is_not_success():
    result, table = check_frame_times([], 20)
    assert not result["passed"] and not table


def test_real_encoded_video_is_fully_checked(tmp_path):
    rows = frame_rows()
    pair = VideoPair(tmp_path, 1, (48, 64, 3))
    for index, row in enumerate(rows):
        image = np.full((48, 64, 3), index * 50, dtype=np.uint8)
        pair.write(image, image, row["video_pts_us"])
    pair.close()
    path = tmp_path / "experiment_raw.mp4"
    assert check_video(path, rows, [64, 48])["passed"]
    assert not check_video(path, rows[:-1], [64, 48])["passed"]
    wrong = deepcopy(rows)
    wrong[2]["video_pts_us"] += 5000
    assert not check_video(path, wrong, [64, 48])["passed"]
    assert not check_video(path, rows, [32, 48])["passed"]


def test_broken_video_is_not_success(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    result = check_video(path, frame_rows(), [64, 48])
    assert not result["passed"]
    assert any("无法完整解码" in problem for problem in result["problems"])


def test_live_source_can_create_its_session_before_hardware_start(tmp_path):
    """仅验证 UUID 会话路径能交给接收器；通过工厂参数终止，不启动相机。"""
    calls = []
    statuses = []

    def stop_before_device(output, project, python, seconds):
        calls.append(output)
        raise RuntimeError("TEST_STOP_BEFORE_HARDWARE")

    source = LiveCameraFrameSource(
        {"windows_python": "unused", "bridge_session_s": 120},
        lambda packet: None,
        statuses.append,
        tmp_path,
        bridge_factory=stop_before_device,
    )
    source.run()
    assert len(calls) == 1 and calls[0].name.startswith("bridge_")
    assert statuses[-1] == "ERROR:RuntimeError('TEST_STOP_BEFORE_HARDWARE')"
