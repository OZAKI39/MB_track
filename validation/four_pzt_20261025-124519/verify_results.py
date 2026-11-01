"""只读核对本批原始记录并导出汇总；从项目根目录运行，不连接仪器。"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
CHANNELS = [f"PZT{i}" for i in range(1, 5)]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_json(name, value):
    with (ROOT / name).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def save_csv(name, rows):
    with (ROOT / name).open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def expected_states(plan, at_s):
    """独立地从存档时间表重算目标，不导入验收模块的比较函数。"""
    states = {
        name: {
            "output": True,
            "frequency_hz": plan["settings"][name]["frequency_hz"],
            "amplitude_vpp": 0.001,
        }
        for name in CHANNELS
    }
    for step in plan["steps"]:
        if step["at_s"] > at_s:
            break
        if step["state"] == "STOP":
            for state in states.values():
                state["output"] = False
            break
        for name in CHANNELS:
            states[name]["frequency_hz"] = step.get("frequencies_hz", {}).get(
                name, states[name]["frequency_hz"]
            )
            states[name]["amplitude_vpp"] = (
                plan["settings"][name]["amplitude_vpp"]
                if name in step.get("active_channels", [])
                else 0.001
            )
    return states


def main():
    frozen = read_json(ROOT / "execution_hashes.json")
    assert all(
        hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest
        for name, digest in frozen.items()
    ), "执行代码或配置发生变化"
    case_dirs = sorted(p for p in ROOT.glob("[0-9][0-9]_*") if p.is_dir())
    assert len(case_dirs) == 6
    summary_rows, command_rows, readback_rows, step_rows = [], [], [], []
    total_queries, total_errors_checked = 0, 0
    minimum_query_gap_ms = float("inf")
    maximum_snapshot_ms = 0.0
    for case, folder in enumerate(case_dirs, 1):
        result = read_json(folder / "validation_summary.json")
        experiment = read_json(folder / "experiment/summary.json")
        plan = read_json(folder / "experiment/plan.json")["plan"]
        events = read_lines(folder / "experiment/events.jsonl")
        snapshots = read_lines(folder / "stable_checks.jsonl")
        commands = read_csv(folder / "experiment/parameter_commands.csv")
        flat = read_csv(folder / "stable_channels.csv")
        postflight = read_json(folder / "postflight.json")
        assert result["case"] == case and result["passed"] and not result["problems"]
        assert read_json(ROOT / f"case_{case}_launch.json")["returncode"] == 0
        assert result["completed_s"] <= 25
        assert result["experiment_status"] == (
            "cancelled" if case == 6 else "completed"
        )
        assert experiment["data_reliable"] is (case != 6)
        assert (
            experiment["all_outputs_off_confirmed"]
            and not experiment["shutdown_errors"]
        )
        assert postflight["all_outputs_off"] and all(
            c["output"] is False for c in postflight["channels"].values()
        )
        for phase in ("precheck", "postcheck"):
            check = experiment[phase]
            assert check["passed"] and not check["problems"]
            for replies in check["error_queue_replies"].values():
                assert all(int(r.split(",", 1)[0]) == 0 for r in replies)
                total_errors_checked += len(replies)
        planned_checks = (
            [1, 3, 7, 11, 15, 19]
            if case in (1, 2, 4)
            else [1, 3, 8, 13, 19]
            if case == 3
            else [1, 3, 9, 13, 19]
            if case == 5
            else [1, 3, 9]
        )
        assert [s["planned_check_s"] for s in snapshots] == planned_checks
        actions = [
            e
            for e in events
            if e["event"] == "action_finished"
            and e.get("action") in ("send_amplitude", "send_frequency")
        ]
        assert len(actions) == len(commands) == [8, 14, 16, 8, 32, 4][case - 1]
        for action, row in zip(actions, commands):
            parameter = action["action"].removeprefix("send_")
            timing = action[f"{parameter}_timing"]
            sent = timing["command_sent_s"]
            returned = timing["command_write_returned_s"]
            assert action["planned_s"] <= action["actual_start_s"] <= sent <= returned
            assert (
                action["status"] == "sent"
                and action["result"]["device_confirmed"] is False
            )
            assert action["result"]["io_delta"] == {
                "write_calls": 1,
                "read_calls": 0,
                "query_calls": 0,
                "query_commands": 0,
            }
            assert timing["confirmed_s"] is None and timing["query"] is None
            assert float(row["command_sent_s"]) == sent
            assert float(row["write_returned_s"]) == returned
            assert float(row["call_started_s"]) == action["actual_start_s"]
            assert row["parameter"] == parameter and row["channel"] == action["channel"]
            target_key = "frequency_hz" if parameter == "frequency" else "amplitude_vpp"
            target = expected_states(plan, action["planned_s"])[action["channel"]][
                target_key
            ]
            assert float(row["requested_value"]) == target
            assert (
                abs(
                    float(row["send_lateness_ms"]) - (sent - action["planned_s"]) * 1000
                )
                < 1e-9
            )
            command_rows.append({"case": case, **row})
        assert len(flat) == 4 * len(snapshots)
        flat_index = 0
        for snapshot in snapshots:
            expected = expected_states(plan, snapshot["step_at_s"])
            assert snapshot["expected"] == expected == snapshot["channels"]
            assert snapshot["passed"] and not snapshot["problems"]
            assert (
                snapshot["planned_check_s"]
                <= snapshot["started_s"]
                < snapshot["finished_s"]
                < snapshot["deadline_s"]
            )
            maximum_snapshot_ms = max(
                maximum_snapshot_ms,
                (snapshot["finished_s"] - snapshot["started_s"]) * 1000,
            )
            previous_returns = [
                float(r["write_returned_s"])
                for r in commands
                if float(r["planned_s"]) <= snapshot["step_at_s"]
            ]
            if previous_returns:
                gap_ms = (snapshot["started_s"] - max(previous_returns)) * 1000
                assert gap_ms >= 100
                minimum_query_gap_ms = min(minimum_query_gap_ms, gap_ms)
            queries = snapshot["queries"]
            assert len(queries) == 14
            prior_finish = snapshot["started_s"]
            for q in queries:
                assert "error" not in q and q["reply"]
                assert (
                    prior_finish
                    <= q["started_s"]
                    <= q["finished_s"]
                    <= snapshot["finished_s"]
                )
                prior_finish = q["finished_s"]
            for index, name in enumerate(CHANNELS):
                for q, key in zip(
                    queries[index * 3 : index * 3 + 3],
                    ("output", "frequency_hz", "amplitude_vpp"),
                ):
                    assert q["target"] == name
                    value = (
                        bool(int(q["reply"])) if key == "output" else float(q["reply"])
                    )
                    assert value == expected[name][key]
                row = flat[flat_index]
                flat_index += 1
                assert row["channel"] == name and row["check_passed"] == "True"
                assert float(row["planned_check_s"]) == snapshot["planned_check_s"]
                for key in ("frequency_hz", "amplitude_vpp"):
                    assert (
                        float(row[f"actual_{key}"])
                        == float(row[f"expected_{key}"])
                        == expected[name][key]
                    )
                assert row["actual_output"] == row["expected_output"] == "True"
                readback_rows.append({"case": case, **row})
            for q in queries[-2:]:
                assert q["command"] == "SYSTem:ERRor:NEXT?"
                assert int(q["reply"].split(",", 1)[0]) == 0
                assert snapshot["error_queue_replies"][q["target"]] == q["reply"]
            total_queries += len(queries)
            total_errors_checked += 2
        for planned_s in sorted({float(r["planned_s"]) for r in commands}):
            group = [r for r in commands if float(r["planned_s"]) == planned_s]
            first_sent = min(float(r["command_sent_s"]) for r in group)
            last_sent = max(float(r["command_sent_s"]) for r in group)
            last_returned = max(float(r["write_returned_s"]) for r in group)
            assert not any(
                first_sent <= q["started_s"] <= last_returned
                for s in snapshots
                for q in s["queries"]
            )
            step_rows.append(
                {
                    "case": case,
                    "planned_s": planned_s,
                    "commands": len(group),
                    "first_sent_s": first_sent,
                    "last_sent_s": last_sent,
                    "last_returned_s": last_returned,
                    "first_to_last_send_ms": (last_sent - first_sent) * 1000,
                }
            )
        if case == 6:
            cancel = result["cancel_request"]["actual_s"]
            assert cancel >= 10 and all(
                float(r["command_sent_s"]) < cancel for r in commands
            )
            assert [e["planned_s"] for e in events if e["event"] == "step_started"] == [
                0,
                2,
            ]
        else:
            assert result["execution_error"] is None
        summary_rows.append(
            {
                "case": case,
                "name": result["name"],
                "passed": True,
                "completed_s": result["completed_s"],
                "stable_checks": len(snapshots),
                "channel_readbacks": len(flat),
                "parameter_commands": len(commands),
                "maximum_command_lateness_ms": max(
                    float(r["send_lateness_ms"]) for r in commands
                ),
                "maximum_same_step_send_span_ms": max(
                    r["first_to_last_send_ms"] for r in step_rows if r["case"] == case
                ),
                "shutdown_started_s": result["shutdown"]["actual_start_s"],
                "shutdown_confirmed_s": result["shutdown"]["actual_finish_s"],
                "independent_all_outputs_off": True,
            }
        )
    final = read_json(ROOT / "postflight_final.json")
    assert final["returncode"] == 0 and len(final["status"]["channels"]) == 4
    assert all(c["output"] is False for c in final["status"]["channels"].values())
    verification = {
        "passed": True,
        "cases_passed": 6,
        "cases_expected_cancelled": 1,
        "stable_checks": 31,
        "channel_readbacks": len(readback_rows),
        "parameter_commands": len(command_rows),
        "stable_queries": total_queries,
        "error_replies_checked": total_errors_checked,
        "nonzero_error_replies": 0,
        "minimum_last_write_to_snapshot_ms": minimum_query_gap_ms,
        "maximum_snapshot_duration_ms": maximum_snapshot_ms,
        "execution_sources_unchanged": True,
        "independent_final_outputs_off": True,
        "physical_signal_timing_measured": False,
    }
    assert (
        len(readback_rows) == 124 and len(command_rows) == 82 and total_queries == 434
    )
    save_csv("summary.csv", summary_rows)
    save_csv("all_commands.csv", command_rows)
    save_csv("all_readbacks.csv", readback_rows)
    save_csv("step_timings.csv", step_rows)
    save_json("summary.json", summary_rows)
    save_json("verification.json", verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
