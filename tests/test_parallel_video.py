"""分流边界与验收逻辑的离线检查，不连接相机、AFG 或 PZT。"""

import copy
from types import SimpleNamespace

import pytest

from customer_workspace.parallel_camera import ParallelCameraStream, h264_nal_types
from customer_workspace.parallel_video_report import check_branches, compare_analysis


def test_annex_b_three_and_four_byte_start_codes():
    data = b"\x00\x00\x00\x01\x67abc\x00\x00\x01\x68def\x00\x00\x01\x65ghi"
    assert h264_nal_types(data) == {5, 7, 8}
    assert h264_nal_types(b"\x00\x00\x01") == set()


@pytest.mark.parametrize("nal_types", [[], [5], [7, 8], [1, 7, 8]])
def test_recording_does_not_start_without_complete_headers(nal_types):
    stream, invoke, _ = gate_for_test()
    assert invoke(1, nal_types) == "DROP"
    assert stream.record_frames == []


def gate_for_test():
    # 只模拟数据包与调度接口，检查结束标记不能越过最后一帧。
    scheduled = []
    stream = ParallelCameraStream.__new__(ParallelCameraStream)
    stream.Gst = SimpleNamespace(
        CLOCK_TIME_NONE=2**64 - 1,
        PadProbeReturn=SimpleNamespace(OK="OK", DROP="DROP"),
    )
    stream.GLib = SimpleNamespace(idle_add=lambda fn: scheduled.append(fn))
    stream.armed = True
    stream.record_closed = stream.finish_requested = False
    stream.record_frames = []
    stream.by_pts = {}
    stream.target_frames = 2

    def invoke(seq, types):
        stream.by_pts[seq] = {"seq": seq, "pts_ns": seq, "nal_types": types}
        info = SimpleNamespace(get_buffer=lambda: SimpleNamespace(pts=seq))
        return stream._record_buffer(None, info)

    return stream, invoke, scheduled


def test_end_marker_waits_until_last_frame_was_forwarded():
    stream, invoke, scheduled = gate_for_test()
    assert invoke(1, [5, 7, 8]) == "OK"
    assert invoke(2, [1]) == "OK"
    assert stream.record_closed
    assert stream._send_record_eos not in scheduled
    assert invoke(3, [1]) == "DROP"
    assert stream._send_record_eos in scheduled
    assert [row["seq"] for row in stream.record_frames] == [1, 2]


def branch_evidence():
    frames = [
        {"seq": 3, "pts_ns": 0, "nal_types": [5, 7, 8]},
        {"seq": 4, "pts_ns": 16666667},
    ]
    session = {
        "record_frames": frames,
        "input_frames": frames,
        "target_frames": 2,
        "errors": [],
        "warnings": [],
        "relay_errors": [],
        "rtp_sequence_missing": 0,
        "queue_overruns": {"record_queue": 0, "decode_queue": 0},
        "queue_levels": [],
        "rtsp_exchanges": [{"method": "PLAY"}],
        "decoded_frames": frames,
    }
    probe = {
        "packets_and_frames": [
            {"type": "packet", "pts": 0},
            {"type": "packet", "pts": 1500},
        ],
        "streams": [{"time_base": "1/90000"}],
    }
    return session, probe, {"passed": True, "problems": []}


def test_all_three_recording_stages_must_agree():
    session, probe, recording = branch_evidence()
    assert check_branches(session, probe, recording)["passed"]
    probe["packets_and_frames"].pop()
    assert not check_branches(session, probe, recording)["passed"]


def test_timestamps_are_checked_even_when_counts_match():
    session, probe, recording = branch_evidence()
    probe["packets_and_frames"][1]["pts"] = 3000
    assert not check_branches(session, probe, recording)["passed"]


@pytest.mark.parametrize(
    "kind", ["queue", "network", "reconnect", "incomplete_headers"]
)
def test_recording_transport_problems_cannot_pass(kind):
    session, probe, recording = branch_evidence()
    if kind == "queue":
        session["queue_overruns"]["record_queue"] = 1
    elif kind == "network":
        session["rtp_sequence_missing"] = 1
    elif kind == "reconnect":
        session["rtsp_exchanges"].append({"method": "PLAY"})
    else:
        session["record_frames"][0]["nal_types"] = [5]
    assert not check_branches(session, probe, recording)["passed"]


def analysis_evidence():
    row = {
        "analysis_hz": 8,
        "software_age_upper_bound_ms": {"p95": 200},
        "pipeline_error": None,
        "recorder_dropped": 0,
        "threads_closed": True,
        "actual_detector_and_piv_results": 160,
        "processed": 160,
        "completion_interval_ms": {"max": 300},
    }
    criteria = {
        "analysis_at_least_hz": 5.6,
        "analysis_rate_ratio_to_baseline_min": 0.8,
        "maximum_analysis_completion_gap_ms": 1000,
    }
    return row, copy.deepcopy(row), criteria


def test_parallel_analysis_without_slowdown_passes():
    assert compare_analysis(*analysis_evidence())["passed"]


@pytest.mark.parametrize("kind", ["slow", "stale", "no_analysis", "stalled", "error"])
def test_fast_empty_or_delayed_analysis_cannot_pass(kind):
    baseline, parallel, criteria = analysis_evidence()
    if kind == "slow":
        parallel["analysis_hz"] = 5
    elif kind == "stale":
        parallel["software_age_upper_bound_ms"]["p95"] = 300
    elif kind == "no_analysis":
        parallel["actual_detector_and_piv_results"] = 0
    elif kind == "stalled":
        parallel["completion_interval_ms"]["max"] = 1200
    else:
        parallel["pipeline_error"] = "failure"
    assert not compare_analysis(baseline, parallel, criteria)["passed"]
