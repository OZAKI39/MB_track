"""只读核对本批真实记录，独立重算 CSV 指标；不连接仪器，不覆盖报告。"""

import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path("customer_workspace/validation/write_only_20261023-190015")
BASELINE = Path("customer_workspace/validation/amplitude_states_20261023-174926")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    batch = ROOT / "experiments"
    report = read_json(batch / "timing_summary.json")
    with (batch / "timing.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    with (batch / "all_attempts.csv").open(encoding="utf-8-sig", newline="") as file:
        attempts = list(csv.DictReader(file))
    assert len(rows) == len(attempts) == 30
    assert report["completed_runs"] == report["reliable_runs"] == 30
    assert report["unreliable_runs"] == report["failed_runs"] == 0
    assert report["paired_intervals"] == 30
    assert report["per_command_confirmation"] is False
    baseline_plan = read_json(BASELINE / "experiments/run_01/plan.json")
    samples = {}
    for index, row in enumerate(rows, 1):
        run = batch / f"run_{index:02d}"
        summary = read_json(run / "summary.json")
        events = [
            json.loads(line)
            for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert read_json(run / "plan.json") == baseline_plan, run
        assert summary["status"] == "completed" and summary["data_reliable"], run
        assert summary["all_outputs_off_confirmed"], run
        assert summary["final_state"] == "STOP" and not summary["shutdown_errors"], run
        assert int(row["run_index"]) == index and row["channel"] == "PZT1", run
        assert row["timeline_started_utc"] == summary["timeline_started_utc"], run
        for phase in ("precheck", "postcheck"):
            audit = summary[phase]
            assert audit["passed"] and not audit["problems"], (run, phase)
            assert set(audit["channels"]) == {"PZT1", "PZT2", "PZT3", "PZT4"}
            for actual in audit["channels"].values():
                assert actual["output"] is False
                assert actual["frequency_hz"] == 2_500_000
                assert actual["amplitude_vpp"] == 0.001
                assert actual["waveform"] == "SIN" and actual["load_ohm"] == 50
                assert actual["offset_v"] == actual["phase_deg"] == 0
                assert actual["frequency_mode"] == "CW"
                assert not any(actual["modulation"].values())
                assert not any(
                    actual[key]
                    for key in ("burst_enabled", "concurrent", "peer_concurrent")
                )
            for alias in ("A", "B"):
                assert audit["io_counts"][alias] == {
                    "write_calls": 2,
                    "read_calls": 40,
                    "query_calls": int(phase == "postcheck"),
                    "query_commands": 40 + int(phase == "postcheck"),
                }, (run, phase, alias)
                assert all(
                    int(reply.split(",")[0]) == 0
                    for reply in audit["error_queue_replies"][alias]
                )
        starts = [event for event in events if event["event"] == "step_started"]
        assert [(event["planned_s"], event["state"]) for event in starts] == [
            (0, "IDLE"),
            (5, "ACTIVE"),
            (6, "IDLE"),
            (9, "IDLE"),
            (14, "STOP"),
        ], run
        actions = [event for event in events if event["event"] == "action_finished"]
        outputs = [event for event in actions if event["action"] == "set_output"]
        assert len(outputs) == 4
        assert {event["channel"] for event in outputs} == set(
            summary["precheck"]["channels"]
        )
        assert all(
            event["result"] is True and event["actual_start_s"] is None
            for event in outputs
        ), run
        changes = [event for event in actions if event["action"] == "send_amplitude"]
        assert len(changes) == 2, run
        for event, state, target, planned in zip(
            changes, ("active", "idle"), (0.1, 0.001), (5, 6)
        ):
            timing = event["amplitude_timing"]
            assert event["status"] == "sent" and event["channel"] == "PZT1"
            assert event["state_requested"] == state.upper()
            assert event["amplitude_requested_vpp"] == target
            assert event["planned_s"] == planned
            assert timing["command"] == (
                f"SOURce1:VOLTage:LEVel:IMMediate:AMPLitude {target:.12g}Vpp"
            )
            assert all(
                timing[key] is None
                for key in (
                    "query",
                    "amplitude_reply_received_s",
                    "amplitude_reply",
                    "confirmed_s",
                    "command_to_confirmation_s",
                )
            )
            assert event["result"]["device_confirmed"] is False
            assert event["result"]["io_delta"] == {
                "write_calls": 1,
                "read_calls": 0,
                "query_calls": 0,
                "query_commands": 0,
            }
            sent = timing["command_sent_s"]
            returned = timing["command_write_returned_s"]
            assert (
                planned
                <= event["actual_start_s"]
                <= sent
                <= returned
                <= event["actual_finish_s"]
            )
            assert float(row[f"{state}_command_sent_s"]) == sent
            assert float(row[f"{state}_command_write_returned_s"]) == returned
            assert (
                row[f"{state}_confirmed_s"]
                == row[f"{state}_confirmation_delay_ms"]
                == ""
            )
            metrics = {
                "schedule_delay_ms": (sent - planned) * 1000,
                "before_call_ms": (event["actual_start_s"] - planned) * 1000,
                "local_prepare_ms": (sent - event["actual_start_s"]) * 1000,
                "write_ms": (returned - sent) * 1000,
            }
            for key, value in metrics.items():
                metric = f"{state}_{key}"
                assert math.isclose(float(row[metric]), value, abs_tol=1e-9)
                samples.setdefault(metric, []).append(value)
        shutdown = next(
            event for event in events if event["event"] == "shutdown_finished"
        )
        postcheck = next(
            event for event in actions if event["action"] == "check_state_after"
        )
        assert (
            14
            <= shutdown["actual_start_s"]
            <= shutdown["actual_finish_s"]
            <= postcheck["actual_start_s"]
        )
    statistics = {}
    for metric, values in samples.items():
        statistics[metric] = {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "sample_stddev": stdev(values),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }
        for key, value in statistics[metric].items():
            assert math.isclose(
                report["statistics_ms"]["PZT1"][metric][key], value, abs_tol=1e-9
            )
    before_hashes = read_json(ROOT / "source_hashes.json")
    after_hashes = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in before_hashes
    }
    assert before_hashes == after_hashes, "测量结束时源码或配置发生变化"
    fault = read_json(ROOT / "error_injection_result.json")
    assert fault["passed"] and fault["excluded_from_timing_statistics"]
    assert fault["summary_status"] == "unreliable"
    assert fault["audit"]["error_queue_replies"]["A"][0] == '-102,"Syntax error"'
    postflight = read_json(ROOT / "postflight.json")
    assert postflight["returncode"] == 0
    channels = json.loads(postflight["stdout"])["channels"]
    assert len(channels) == 4 and all(
        value["output"] is False for value in channels.values()
    )
    baseline_report = read_json(BASELINE / "experiments/timing_summary.json")
    comparison = {}
    for state in ("active", "idle"):
        metric = f"{state}_schedule_delay_ms"
        previous = baseline_report["statistics_ms"]["PZT1"][metric]["mean"]
        current = statistics[metric]["mean"]
        comparison[state] = {
            "previous_mean_ms": previous,
            "current_mean_ms": current,
            "mean_reduction_percent": (1 - current / previous) * 100,
        }
    result = {
        "checked_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "normal_runs": 30,
        "verified_write_only_actions": 60,
        "full_queries_per_channel_per_boundary": 20,
        "all_normal_runs_reliable": True,
        "normal_errors": 0,
        "intentional_error_test_passed": True,
        "source_and_config_unchanged_during_measurement": True,
        "source_hashes_at_verification": after_hashes,
        "plan_and_config_match_previous_three_run_baseline": True,
        "csv_matches_raw_events": True,
        "postchecks_after_stop": True,
        "final_outputs_off": True,
        "statistics_ms": statistics,
        "comparison": comparison,
        "limits": [
            "Only PC-side send times; device acceptance and voltage timing were not measured.",
            "Boundary checks cannot rule out transient changes during a round.",
            "Baseline has 3 runs; new benchmark has 30 runs at another time.",
            "Hardware latency data covers PZT1 changes; all four channels were prepared and checked.",
        ],
    }
    with (ROOT / "verification.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        json.dumps(
            {"passed": True, "normal_runs": 30, "comparison": comparison}, indent=2
        )
    )


if __name__ == "__main__":
    main()
