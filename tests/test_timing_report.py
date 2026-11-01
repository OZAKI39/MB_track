"""只用动作记录样例检查时间换算与汇总，不创建 AFG 会话。"""

import csv
import json

import pytest

from customer_workspace.afg_timing import OutputTiming
from customer_workspace.timing_report import (
    parameter_rows,
    pulse_rows,
    write_timing_report,
)


def action(on, sent, confirmed, *, status="ok"):
    """构造带已知时间的记录，用于检查统计公式。"""
    return {
        "event": "action_finished",
        "action": "set_output",
        "channel": "PZT1",
        "output_requested": on,
        "status": status,
        "planned_s": 5 if on else 6,
        "actual_start_s": sent - 0.1,
        "actual_finish_s": confirmed + 0.01,
        "output_timing": {
            "command_sent_s": sent,
            "command_write_returned_s": sent + 0.001,
            "confirmed_s": confirmed,
        },
    }


def test_times_are_relative_and_failed_switch_is_not_confirmed():
    timing = OutputTiming("OUTPut1:STATe ON", "OUTPut1:STATe?")
    timing.command_sent = 105.2
    timing.write_returned = 105.201
    timing.state_reply_received = 105.22
    timing.state_reply = "1"
    assert timing.relative_to(100)["confirmed_s"] is None
    assert timing.relative_to(100)["state_reply_received_s"] == pytest.approx(5.22)
    timing.succeeded = True
    result = timing.relative_to(100)
    assert result["command_sent_s"] == pytest.approx(5.2)
    assert result["confirmed_s"] == pytest.approx(5.22)
    assert result["command_to_confirmation_s"] == pytest.approx(0.02)


def test_missing_response_stays_missing_even_if_marked_successful():
    timing = OutputTiming("OUTPut1:STATe ON", "OUTPut1:STATe?", succeeded=True)
    assert timing.relative_to(100)["confirmed_s"] is None
    assert timing.relative_to(100)["command_to_confirmation_s"] is None


def test_write_only_timing_does_not_claim_confirmation():
    timing = OutputTiming("SOURce1:VOLTage:LEVel:IMMediate:AMPLitude 0.1Vpp", None)
    timing.command_sent = 105.001
    timing.write_returned = 105.002
    timing.succeeded = True
    record = timing.relative_to(100, reply_name="amplitude")
    assert record["query"] is None
    assert record["confirmed_s"] is None
    assert record["amplitude_reply"] is None
    assert record["command_write_returned_s"] == pytest.approx(5.002)


def send_action(state, sent):
    event = amplitude_action(state, sent, sent + 0.01, status="sent")
    event["action"] = "send_amplitude"
    event["amplitude_timing"]["confirmed_s"] = None
    event["actual_start_s"] = sent - 0.0001
    return event


def frequency_action(sent):
    event = send_action("ACTIVE", sent)
    event["action"] = "send_frequency"
    event["frequency_requested_hz"] = 2600000
    event["frequency_timing"] = event.pop("amplitude_timing")
    event["actual_finish_s"] = sent + 0.0011
    return event


def test_same_step_parameter_report_keeps_individual_times_and_gap():
    amplitude = send_action("ACTIVE", 5.003)
    amplitude["amplitude_requested_vpp"] = 0.1
    rows = parameter_rows([frequency_action(5.001), amplitude], 1)
    assert [row["parameter"] for row in rows] == ["frequency", "amplitude"]
    assert [row["requested_value"] for row in rows] == [2600000, 0.1]
    assert [row["planned_s"] for row in rows] == [5, 5]
    assert rows[0]["send_lateness_ms"] == pytest.approx(1)
    assert rows[1]["send_lateness_ms"] == pytest.approx(3)
    assert rows[0]["previous_same_step_command_gap_ms"] is None
    assert rows[1]["previous_same_step_command_gap_ms"] == pytest.approx(2)


def test_failed_frequency_write_keeps_missing_time_and_error():
    event = frequency_action(5.001)
    event["status"] = "failed"
    event["error"] = "write failed"
    event["frequency_timing"]["command_write_returned_s"] = None
    rows = parameter_rows([event], 1)
    assert rows[0]["error"] == "write failed"
    assert rows[0]["write_ms"] is None


@pytest.mark.parametrize("combined", [False, True])
def test_frequency_report_accepts_new_modes_and_exports_commands(tmp_path, combined):
    mode = "frequency_amplitude_write_only" if combined else "frequency_write_only"
    summary = {
        "status": "completed",
        "data_reliable": True,
        "control_mode": mode,
        "timeline_started_utc": "2026-10-25T12:00:00+00:00",
    }
    events = [frequency_action(5.001)]
    if combined:
        events += [send_action("ACTIVE", 5.003), send_action("IDLE", 6.003)]
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    report = write_timing_report(tmp_path, [tmp_path], 1)
    assert report["control_mode"] == mode
    assert report["recorded_parameter_commands"] == len(events)
    assert report["per_command_confirmation"] is False
    assert report["paired_intervals"] == int(combined)
    with (tmp_path / "parameter_commands.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(events) and rows[0]["parameter"] == "frequency"


def test_write_only_pairs_send_times_and_leaves_confirmation_empty():
    rows = pulse_rows([send_action("ACTIVE", 5.001), send_action("IDLE", 6.002)], 1)
    row = rows[0]
    assert row["control_mode"] == "amplitude_write_only"
    assert row["active_schedule_delay_ms"] == pytest.approx(1)
    assert row["idle_schedule_delay_ms"] == pytest.approx(2)
    assert row["active_write_ms"] == pytest.approx(1)
    assert row["active_confirmed_s"] is None
    assert row["idle_confirmed_s"] is None
    assert row["active_confirmation_delay_ms"] is None
    assert row["active_local_prepare_ms"] == pytest.approx(0.1)


def test_write_only_records_reject_fabricated_confirmation():
    event = send_action("ACTIVE", 5.001)
    event["amplitude_timing"]["confirmed_s"] = 5.01
    with pytest.raises(ValueError, match="不能包含逐次设备确认"):
        pulse_rows([event], 1)


def test_unreliable_round_is_saved_and_excluded(tmp_path):
    directories = []
    for index, (status, reliable) in enumerate(
        [("completed", True), ("unreliable", False), ("completed", False)], 1
    ):
        directory = tmp_path / f"run_{index:02d}"
        directory.mkdir()
        directories.append(directory)
        summary = {
            "status": status,
            "data_reliable": reliable,
            "control_mode": "amplitude_write_only",
            "timeline_started_utc": "2026-10-23T12:00:00+00:00",
            "error": "instrument error" if status == "unreliable" else None,
        }
        (directory / "summary.json").write_text(json.dumps(summary))
        events = [send_action("ACTIVE", 5.001), send_action("IDLE", 6.002)]
        (directory / "events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events)
        )
    report = write_timing_report(tmp_path, directories, 3)
    assert report["reliable_runs"] == 1
    assert report["unreliable_runs"] == 1
    assert report["paired_intervals"] == 1
    assert report["per_command_confirmation"] is False
    assert "active_confirmation_delay_ms" not in report["statistics_ms"]["PZT1"]
    with (tmp_path / "timing.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1 and rows[0]["active_confirmed_s"] == ""
    with (tmp_path / "all_attempts.csv").open(encoding="utf-8-sig") as file:
        attempts = list(csv.DictReader(file))
    assert len(attempts) == 3 and attempts[1]["error"] == "instrument error"


def test_amplitude_confirmation_has_explicit_field_names():
    timing = OutputTiming(
        "SOURce1:VOLTage:LEVel:IMMediate:AMPLitude 0.1Vpp",
        "SOURce1:VOLTage:LEVel:IMMediate:AMPLitude?",
    )
    timing.command_sent = 105.1
    timing.write_returned = 105.101
    timing.state_reply_received = 105.12
    timing.state_reply = "0.100"
    result = timing.relative_to(100, reply_name="amplitude")
    assert result["confirmed_s"] is None
    timing.succeeded = True
    result = timing.relative_to(100, reply_name="amplitude")
    assert result["amplitude_reply"] == "0.100"
    assert result["confirmed_s"] == pytest.approx(5.12)
    assert "state_reply" not in result


def amplitude_action(state, sent, confirmed, *, status="ok"):
    """调幅日志样例，不模拟设备或通信。"""
    event = action(state == "ACTIVE", sent, confirmed, status=status)
    event["action"] = "apply_live"
    event["state_requested"] = state
    event["amplitude_timing"] = event.pop("output_timing")
    del event["output_requested"]
    return event


def test_amplitude_pair_does_not_use_old_on_off_names():
    rows = pulse_rows(
        [amplitude_action("ACTIVE", 5.1, 5.12), amplitude_action("IDLE", 6.1, 6.13)], 1
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["control_mode"] == "amplitude_states"
    assert row["active_command_sent_s"] == 5.1
    assert row["active_confirmed_s"] == 5.12
    assert row["idle_command_sent_s"] == 6.1
    assert row["idle_confirmed_s"] == 6.13
    assert row["active_confirmation_delay_ms"] == pytest.approx(20)
    assert row["idle_confirmation_delay_ms"] == pytest.approx(30)
    assert "off_confirmed_s" not in row


def test_failed_idle_and_stop_do_not_fabricate_amplitude_confirmation():
    active = amplitude_action("ACTIVE", 5.1, 5.12)
    failed = amplitude_action("IDLE", 6.1, 6.13, status="failed")
    assert pulse_rows([active, failed, amplitude_action("IDLE", 9.1, 9.12)], 1) == []
    assert (
        pulse_rows([active, {"event": "shutdown_finished", "action": "all_off"}], 1)
        == []
    )


@pytest.mark.parametrize("missing_summary", [False, True])
def test_amplitude_report_uses_matching_definition(tmp_path, missing_summary):
    directory = tmp_path / "run_01"
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "control_mode": "amplitude_states",
                "timeline_started_utc": "2026-10-23T12:00:00+00:00",
            }
        )
    )
    events = [
        amplitude_action("ACTIVE", 5.1, 5.12),
        amplitude_action("IDLE", 6.1, 6.13),
    ]
    (directory / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )
    run_dirs = [directory]
    if missing_summary:
        incomplete = tmp_path / "run_02"
        incomplete.mkdir()
        run_dirs.append(incomplete)
    report = write_timing_report(tmp_path, run_dirs, len(run_dirs))
    assert report["paired_intervals"] == 1
    assert report["control_mode"] == "amplitude_states"
    assert report["confirmed_definition"].startswith("amplitude_reply")
    assert report["statistics_ms"]["PZT1"]["idle_confirmation_delay_ms"][
        "mean"
    ] == pytest.approx(30)
    if missing_summary:
        assert report["runs"][1]["status"] == "incomplete"


def test_mixed_old_and_new_measurements_are_rejected(tmp_path):
    paths = []
    for mode in ("output_switch", "amplitude_states"):
        path = tmp_path / mode
        path.mkdir()
        paths.append(path)
        (path / "summary.json").write_text(
            json.dumps({"status": "failed", "control_mode": mode})
        )
    with pytest.raises(ValueError, match="不能把旧输出开关"):
        write_timing_report(tmp_path, paths, 2)


def test_four_times_and_delays_pair_only_the_pulse():
    events = [
        action(False, 0.2, 0.21),
        action(True, 5.2, 5.22),
        action(False, 6.1, 6.13),
        action(False, 9.2, 9.21),
    ]
    rows = pulse_rows(events, 2)
    assert len(rows) == 1
    row = rows[0]
    assert row["run_index"] == 2
    assert [
        row[key]
        for key in (
            "on_command_sent_s",
            "on_confirmed_s",
            "off_command_sent_s",
            "off_confirmed_s",
        )
    ] == [5.2, 5.22, 6.1, 6.13]
    assert row["on_confirmation_delay_ms"] == pytest.approx(20)
    assert row["off_confirmation_delay_ms"] == pytest.approx(30)
    assert row["on_schedule_delay_ms"] == pytest.approx(200)
    assert row["on_before_call_ms"] + row["on_precheck_ms"] == pytest.approx(200)


def test_failed_or_unpaired_switch_is_not_counted():
    assert pulse_rows([action(True, 5.2, 5.22)], 1) == []
    assert (
        pulse_rows(
            [
                action(True, 5.2, 5.22),
                action(False, 6.1, 6.13, status="failed"),
                action(False, 9.2, 9.21),
            ],
            1,
        )
        == []
    )


@pytest.mark.parametrize("sent", [None, float("nan"), 5.25])
def test_missing_or_reversed_times_are_rejected(sent):
    event = action(True, 5.2, 5.22)
    event["output_timing"]["command_sent_s"] = sent
    with pytest.raises(ValueError):
        pulse_rows([event], 1)


def test_repeated_on_pairs_with_first_on():
    rows = pulse_rows(
        [action(True, 5.2, 5.22), action(True, 5.5, 5.52), action(False, 6.1, 6.13)], 1
    )
    assert rows[0]["on_command_sent_s"] == 5.2


def test_report_keeps_failed_run_and_excludes_it_from_statistics(tmp_path):
    run_dirs = []
    for index, status in enumerate(("completed", "failed", "completed"), 1):
        directory = tmp_path / f"run_{index:02d}"
        directory.mkdir()
        run_dirs.append(directory)
        (directory / "summary.json").write_text(
            json.dumps(
                {"status": status, "timeline_started_utc": "2026-10-23T12:00:00+00:00"}
            )
        )
        events = [action(True, 5.2, 5.22 + index * 0.01), action(False, 6.1, 6.13)]
        (directory / "events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events)
        )
    report = write_timing_report(tmp_path, run_dirs, requested=10)
    assert report["requested_runs"] == 10
    assert report["completed_runs"] == 2
    assert report["paired_pulses"] == 2
    assert report["runs"][1]["status"] == "failed"
    stats = report["statistics_ms"]["PZT1"]["on_confirmation_delay_ms"]
    assert stats["count"] == 2
    assert stats["mean"] == pytest.approx(40)
    assert stats["range"] == pytest.approx(20)
    with (tmp_path / "timing.csv").open(encoding="utf-8-sig", newline="") as file:
        assert [row["run_index"] for row in csv.DictReader(file)] == ["1", "3"]
    with pytest.raises(FileExistsError):
        write_timing_report(tmp_path, run_dirs, requested=10)
