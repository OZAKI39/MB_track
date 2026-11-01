"""审查视频的时间与记录边界测试；不连接相机或 AFG。"""

import json
from fractions import Fraction
from types import SimpleNamespace

import av
import pytest

from customer_workspace.pzt_video_review import (
    check_frame_time,
    load_evidence,
    status_at,
    verify_video,
)


def action(state="ACTIVE", start=3.01, end=3.03, confirmed=4.10):
    return {
        "state": state,
        "command_video": {"video_from_s": start, "video_to_s": end},
        "confirmation_video": {"mapped": True, "video_to_s": confirmed},
    }


def test_future_action_and_confirmation_are_not_shown_early():
    event = action()
    assert status_at([event], 3.0) == (None, "Preset idle")
    assert status_at([event], 3.02) == (event, "Switching")
    assert status_at([event], 3.03) == (event, "Awaiting readback")
    assert status_at([event], 4.10) == (event, "Readback received")


def test_previous_confirmation_does_not_confirm_new_command():
    active = action()
    idle = action("IDLE", 6.01, 6.03, 7.10)
    assert status_at([active, idle], 6.05) == (idle, "Awaiting readback")


def test_missing_reply_stays_unconfirmed():
    event = action()
    event["confirmation_video"] = {"mapped": False}
    assert status_at([event], 20) == (event, "Unconfirmed")


def test_actual_fractional_frame_time_is_used():
    frame = SimpleNamespace(pts=1501, time_base=Fraction(1, 90000))
    expected = {"frame_index": 1, "video_s": 1501 / 90000}
    assert check_frame_time(frame, expected) == Fraction(1501, 90000)
    with pytest.raises(ValueError, match="播放时间"):
        check_frame_time(frame, {"frame_index": 1, "video_s": 1 / 60})


def test_missing_frame_time_is_rejected():
    with pytest.raises(ValueError, match="没有播放时间"):
        check_frame_time(SimpleNamespace(pts=None, time_base=None), {})


def test_other_experiment_report_is_rejected(tmp_path):
    experiment, report = tmp_path / "experiment", tmp_path / "report"
    for name in (
        "recording/video.mp4",
        "afg_experiment/plan.json",
        "afg_experiment/events.jsonl",
        "gst_session.json",
        "recording/probe.json",
    ):
        path = experiment / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for name in ("link_session.json", "afg_link_summary.json", "afg_ready.json"):
        (experiment / name).write_text(json.dumps({"experiment_id": "correct"}))
    report.mkdir()
    (report / "link_report.json").write_text(json.dumps({"experiment_id": "different"}))
    with pytest.raises(ValueError, match="不属于同一次实验"):
        load_evidence(experiment, report)


def make_timed_video(path):
    """生成很小的无设备图像夹具；故意使用不等间隔来检查有无重造时间。"""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=60)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 90000)
        stream.codec_context.max_b_frames = 0
        for pts in (0, 1501, 3000, 4500):
            frame = av.VideoFrame(64, 48, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            frame.pts, frame.time_base = pts, Fraction(1, 90000)
            frame.duration = 1500
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_encoded_video_keeps_nonuniform_times_and_is_fully_decoded(tmp_path):
    path = tmp_path / "tiny.mp4"
    make_timed_video(path)
    times = [Fraction(pts, 90000) for pts in (0, 1501, 3000, 4500)]
    result = verify_video(
        path, times, Fraction(1, 15), (64, 48), tmp_path / "preview.png"
    )
    assert result["decoded_frames"] == 4
    assert result["all_frame_times_identical"]
    assert (tmp_path / "preview.png").exists()


@pytest.mark.parametrize("change", ["missing_frame", "wrong_time", "wrong_duration"])
def test_video_verification_detects_missing_or_retimed_frames(tmp_path, change):
    path = tmp_path / "tiny.mp4"
    make_timed_video(path)
    times = [Fraction(pts, 90000) for pts in (0, 1501, 3000, 4500)]
    end = Fraction(1, 15)
    if change == "missing_frame":
        times.append(Fraction(6000, 90000))
    elif change == "wrong_time":
        times[1] = Fraction(1, 60)
    else:
        end += Fraction(1, 60)
    with pytest.raises(ValueError):
        verify_video(path, times, end, (64, 48), tmp_path / "preview.png")
