"""独立录像计数和时间验收的离线测试，不访问相机或 AFG。"""

import json
from fractions import Fraction

import pytest

from customer_workspace.streamcopy_recording import (
    RecordingRtspControl,
    capture_command,
)
from customer_workspace.streamcopy_report import (
    analyze,
    check_timestamps,
    compare_packets,
    make_report,
    read_log_packets,
)
from microbubble_experiment.camera_rtsp import ChronosRtspControl


def packet(pts):
    return {"pts": pts, "dts": pts, "pts_s": pts / 60}


def log_for(points):
    lines = []
    for pts in points:
        fields = f"pkt_pts:{pts} pkt_pts_time:{pts / 60} pkt_dts:{pts}"
        lines.append(f"[vist] demuxer -> {fields}")
        lines.append(f"[vist] demuxer+ffmpeg -> {fields}")
        lines.append(f"[vost] muxer <- pts:{pts} pts_time:{pts / 60} dts:{pts}")
    lines.extend(
        [
            f"Input stream #0:0 (video): {len(points)} packets read",
            f"Output stream #0:0 (video): {len(points)} packets muxed",
        ]
    )
    return "\n".join(lines)


def test_parse_all_stages_without_double_counting():
    parsed = read_log_packets(log_for([0, 1, 2]))
    assert all(len(rows) == 3 for rows in parsed.values())


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("3 packets read", "4 packets read"),
        lambda text: text.replace("demuxer+ffmpeg ->", "unknown ->", 1),
        lambda text: text.replace("pkt_pts:0", "pkt_pts:NOPTS", 1),
        lambda text: text.split("Input stream")[0],
    ],
)
def test_incomplete_or_unknown_logs_fail_closed(change):
    with pytest.raises(ValueError):
        read_log_packets(change(log_for([0, 1, 2])))


def test_end_boundary_is_not_counted_as_recording_loss():
    rows = [packet(i) for i in range(62)]
    counts = compare_packets(rows, rows[:60], rows[:60], Fraction(1, 60), 1)
    assert counts["after_window_packets"] == 2
    assert counts["received_window_frames"] == 60
    assert counts["missing_in_saved"] == 0
    assert counts["ordered_timestamps_match"]


def test_missing_frame_detected_despite_equal_total_count():
    rows = [packet(i) for i in range(60)]
    damaged = rows[:10] + rows[11:] + [rows[-1]]
    counts = compare_packets(rows, rows, damaged, Fraction(1, 60), 1)
    assert counts["missing_in_saved"] == 1
    assert counts["unexpected_in_saved"] == 1
    assert not counts["ordered_timestamps_match"]


def test_clock_unit_mismatch_is_not_silently_accepted():
    with pytest.raises(ValueError, match="时间单位"):
        compare_packets([packet(1)], [packet(1)], [packet(1)], Fraction(1, 90), 1)


def frames(points):
    return [{"pts": i, "duration": 1, "key_frame": int(i == 0)} for i in points]


def test_regular_video_time_is_complete():
    timing, rows = check_timestamps(frames(range(60)), Fraction(1, 60), 1, Fraction(60))
    assert timing["covers_requested_window"]
    assert timing["gaps_over_1_5_frames"] == 0
    assert len(rows) == 60


@pytest.mark.parametrize(
    "points,field",
    [
        ([0, 1, 3, 4], "gaps_over_1_5_frames"),
        ([0, 1, 1, 2], "non_increasing"),
        ([0, 2, 1, 3], "non_increasing"),
    ],
)
def test_missing_repeated_and_backward_times(points, field):
    timing, _ = check_timestamps(frames(points), Fraction(1, 60), 1, Fraction(60))
    assert timing[field] > 0


def test_short_video_cannot_pass():
    timing, _ = check_timestamps(frames(range(30)), Fraction(1, 60), 1, Fraction(60))
    assert not timing["covers_requested_window"]


def test_missing_evidence_cannot_pass(tmp_path):
    report, _ = analyze(tmp_path)
    assert not report["passed"]
    assert report["problems"]


def test_old_report_never_overwritten(tmp_path):
    path = tmp_path / "recording_check.json"
    path.write_text('{"original": true}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        make_report(tmp_path)
    assert json.loads(path.read_text()) == {"original": True}


def test_command_copies_without_frame_replacement(tmp_path):
    config = {"udp_port": 5000, "open_timeout_ms": 5000, "latency_mode": "low"}
    command = capture_command(
        tmp_path / "ffmpeg.exe", "rtsp://127.0.0.1/", config, tmp_path, 20
    )
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-frames:v") + 1] == "1200"
    assert command.index("-ss") > command.index("-i")
    assert command[command.index("-ss") + 1] == "0.1"
    assert "-r" not in command
    assert "-n" in command
    assert "-y" not in command


def test_keyframe_warmup_and_constant_mp4_shift():
    untimed = {"pts": None, "dts": None, "pts_s": None}
    received = [untimed] + [packet(i) for i in range(80)]
    muxed = received[11:71]
    saved = [packet(i) for i in range(60)]
    counts = compare_packets(received, muxed, saved, Fraction(1, 60), 1)
    assert counts["before_window_packets"] == 10
    assert counts["startup_untimed_packets"] == 1
    assert counts["after_window_packets"] == 10
    assert counts["ordered_timestamps_match"]


def test_missing_timestamp_after_start_fails():
    rows = [packet(0), {"pts": None, "dts": None, "pts_s": None}, packet(2)]
    with pytest.raises(ValueError, match="录像开始后"):
        compare_packets(
            rows, [packet(0), packet(2)], [packet(0), packet(2)], Fraction(1, 60), 1
        )


def test_output_seek_offset_preserves_each_frame_and_interval():
    received = [packet(i) for i in range(80)]
    muxed = [packet(i) for i in range(4, 64)]
    saved = [packet(i) for i in range(60)]
    counts = compare_packets(received, muxed, saved, Fraction(1, 60), 1, 0.1)
    assert counts["before_window_packets"] == 10
    assert counts["output_seek_ticks"] == 6
    assert counts["ordered_timestamps_match"]


def test_end_boundary_one_clock_tick_early_is_not_a_lost_frame():
    received = [{"pts": i, "dts": i, "pts_s": i / 90000} for i in (0, 1499, 2999)]
    saved = [dict(row, duration=1500) for row in received[:2]]
    counts = compare_packets(received, received[:2], saved, Fraction(1, 90000), 1 / 30)
    assert counts["received_window_frames"] == 2
    assert counts["after_window_packets"] == 1
    assert counts["ordered_timestamps_match"]


def test_nonconstant_source_timestamp_rewrite_fails():
    log = log_for([0, 1, 2]).replace(
        "demuxer -> pkt_pts:2",
        "demuxer -> pkt_pts:3",
    )
    with pytest.raises(ValueError, match="非常量改写"):
        read_log_packets(log)


@pytest.mark.parametrize(
    "response,closed",
    [
        (b"RTSP/1.0 200 OK\r\n\r\n", True),
        (b"RTSP/1.0 454 Session Not Found\r\n\r\n", False),
    ],
)
def test_only_successful_teardown_closes_owned_control(monkeypatch, response, closed):
    # 仅验证结束响应处理，不创建网络连接。
    monkeypatch.setattr(
        ChronosRtspControl, "_camera_request", lambda self, request: response
    )
    relay = RecordingRtspControl("127.0.0.1")
    assert (
        relay._camera_request(b"TEARDOWN rtsp://127.0.0.1/ RTSP/1.0\r\n\r\n")
        == response
    )
    assert relay.stop_event.is_set() == closed
    assert relay.exchanges[0]["response"] == response.split(b"\r\n")[0].decode()
