"""只测试日志关联和时钟边界，不模拟或连接 AFG。"""

import copy
from pathlib import Path

import pytest

from customer_workspace.afg_video_report import amplitude_confirmation, frame_range
from customer_workspace.experiment_runner import load_plan
from customer_workspace.video_clock import (
    ClockServer,
    clock_interval,
    probe_clock,
    read_health,
    read_ready,
)
from microbubble_experiment.afg import DEFAULT_CONFIG, load_config

ROOT = Path(__file__).resolve().parents[2]


def frames():
    return [
        {"frame_index": i, "video_s": i * 0.02, "received_windows_s": 100 + i * 0.02}
        for i in range(5)
    ]


def test_event_is_bracketed_by_received_frames():
    result = frame_range(frames(), 100.03)
    assert result["first_frame"] == 1
    assert result["last_frame"] == 2
    assert result["video_from_s"] == 0.02
    assert result["video_to_s"] == 0.04


def test_clock_margin_expands_the_corresponding_range():
    result = frame_range(frames(), 100.04, 0.001)
    assert (result["first_frame"], result["last_frame"]) == (1, 3)


@pytest.mark.parametrize("at", [99, 101])
def test_outside_video_is_not_extrapolated(at):
    assert not frame_range(frames(), at)["mapped"]


def test_reversed_receive_times_are_rejected():
    with pytest.raises(ValueError):
        frame_range(list(reversed(frames())), 100.03)


def test_two_way_clock_interval_does_not_assume_symmetric_delay():
    assert clock_interval(
        {"a1": 10, "b2": 10.03, "b3": 10.04, "a4": 10.10}
    ) == pytest.approx((-0.06, 0.03))


def test_other_host_clock_is_not_used():
    camera = {
        "experiment_id": "wrong",
        "host": "another-computer",
        "os": "nt",
        "clock": "windows_monotonic",
    }
    with pytest.raises(ValueError):
        probe_clock(camera, "this-experiment")


def test_incomplete_marker_is_not_read_as_ready(tmp_path):
    path = tmp_path / "marker.json"
    assert read_ready(path) is None
    path.write_text('{"status":')
    assert read_ready(path) is None


def test_local_health_messages_do_not_replace_shared_files(tmp_path):
    health = {"experiment_id": "test", "errors": [], "last_frame_windows_s": 42}
    server = ClockServer(tmp_path, "test", lambda: dict(health)).start()
    try:
        camera = read_ready(tmp_path / "camera_clock.json")
        assert read_health(camera, "test")["last_frame_windows_s"] == 42
        health["last_frame_windows_s"] = 43
        assert read_health(camera, "test")["last_frame_windows_s"] == 43
        assert not (tmp_path / "camera_health.json").exists()
    finally:
        server.close()


def test_800mv_plan_uses_the_shared_limit():
    assert DEFAULT_CONFIG == ROOT / "customer_workspace/configs/afg_video.json"
    config = load_config()
    assert all(ch["max_input_vpp"] == 0.8 for ch in config["channels"].values())
    path = ROOT / "customer_workspace/configs/four_pzt_video_20s.json"
    plan = load_plan(path, config)
    assert all(s.amplitude_vpp == 0.8 for s in plan.settings.values())
    assert [s.at_s for s in plan.steps] == [0, 3, 6, 9, 12, 15, 18, 20]
    assert plan.steps[5].active_channels == ("PZT1", "PZT3")

    # 显式传入较低的配置上限时，时间表检查仍应拒绝超限参数。
    restricted = copy.deepcopy(config)
    restricted["channels"]["PZT1"]["max_input_vpp"] = 0.5
    with pytest.raises(ValueError):
        load_plan(path, restricted)


def confirmation_evidence():
    command = "SOURce1:VOLTage:LEVel:IMMediate:AMPLitude?"
    event = {"channel": "PZT1", "planned_s": 3, "amplitude_requested_vpp": 0.8}
    afg = {
        "checks": [
            {
                "step_at_s": 3,
                "passed": True,
                "queries": [
                    {
                        "target": "PZT1",
                        "command": command,
                        "started_s": 4.0,
                        "finished_s": 4.1,
                    }
                ],
            }
        ],
        "communication": [
            {
                "event": "reply",
                "device": "A",
                "command": command,
                "reply": "0.8",
                "received_windows_s": 104.05,
            }
        ],
    }
    return event, afg, {"channels": {"PZT1": {"device": "A"}}}, 100


def test_confirmation_requires_matching_step_channel_value_and_time():
    evidence = confirmation_evidence()
    assert amplitude_confirmation(*evidence)["received_windows_s"] == 104.05
    for key, value in [
        ("device", "B"),
        ("reply", "0.001"),
        ("received_windows_s", 105.0),
    ]:
        altered = copy.deepcopy(evidence)
        altered[1]["communication"][0][key] = value
        assert amplitude_confirmation(*altered) is None


def test_a_later_final_check_cannot_confirm_an_intermediate_action():
    event, afg, config, origin = confirmation_evidence()
    afg["checks"][0]["step_at_s"] = 18
    assert amplitude_confirmation(event, afg, config, origin) is None
