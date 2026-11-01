"""从真实记录导出参数写入和 ACTIVE/IDLE 时间；不连接设备，不混淆设备确认。"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


def parameter_rows(events: list[dict], run_index: int) -> list[dict]:
    """导出每条调频/调幅记录及同一步两条命令的间隔；失败和缺失时间保留。"""
    rows = []
    previous_sent = {}
    for event in events:
        if event.get("event") != "action_finished" or event.get("action") not in (
            "send_frequency",
            "send_amplitude",
        ):
            continue
        parameter = "frequency" if event["action"] == "send_frequency" else "amplitude"
        timing = event.get(f"{parameter}_timing", {})
        if timing.get("confirmed_s") is not None:
            raise ValueError("只发送参数记录不能包含设备确认")
        sent = timing.get("command_sent_s")
        returned = timing.get("command_write_returned_s")
        times = [event["actual_start_s"], sent, returned, event["actual_finish_s"]]
        present = [value for value in times if value is not None]
        if any(not math.isfinite(value) for value in present) or present != sorted(
            present
        ):
            raise ValueError("参数命令时间顺序无效")
        if event["status"] == "sent" and len(present) != len(times):
            raise ValueError("成功发送的参数命令缺少时间")
        planned = event["planned_s"]
        key = (event["channel"], planned)
        previous = previous_sent.get(key)
        unit = "Hz" if parameter == "frequency" else "Vpp"
        value_key = (
            "frequency_requested_hz"
            if parameter == "frequency"
            else "amplitude_requested_vpp"
        )
        rows.append(
            {
                "run_index": run_index,
                "channel": event["channel"],
                "parameter": parameter,
                "requested_value": event.get(value_key),
                "unit": unit,
                "planned_s": planned,
                "call_started_s": event["actual_start_s"],
                "command_sent_s": sent,
                "write_returned_s": returned,
                "send_lateness_ms": None
                if sent is None or planned is None
                else (sent - planned) * 1000,
                "write_ms": None
                if sent is None or returned is None
                else (returned - sent) * 1000,
                "previous_same_step_command_gap_ms": None
                if previous is None or sent is None
                else (sent - previous) * 1000,
                "status": event["status"],
                "error": event.get("error"),
            }
        )
        if sent is not None:
            previous_sent[key] = sent
    return rows


def pulse_rows(events: list[dict], run_index: int) -> list[dict]:
    """将 ACTIVE 与后续 IDLE 配对；兼容旧 ON/OFF 记录但不混淆两种含义。

    逐次核查模式要求设备确认；只发送模式仅要求写入返回成功。失败或
    缺少通信时间的动作不能进入统计，未配对动作不补造结束时间。
    """
    pending = {}
    rows = []
    for event in events:
        if event.get("event") != "action_finished":
            continue
        action = event.get("action")
        if action in ("apply_live", "send_amplitude") and event.get(
            "state_requested"
        ) in (
            "ACTIVE",
            "IDLE",
        ):
            mode = (
                "amplitude_write_only"
                if action == "send_amplitude"
                else "amplitude_states"
            )
            enabled = event["state_requested"] == "ACTIVE"
            timing = event.get("amplitude_timing")
            timing_key = "amplitude_timing"
            start_prefix, end_prefix = "active", "idle"
        elif action == "set_output":
            mode = "output_switch"
            enabled = event["output_requested"]
            timing = event.get("output_timing")
            timing_key = "output_timing"
            start_prefix, end_prefix = "on", "off"
        else:
            continue
        channel = event["channel"]
        pending_key = (channel, mode)
        write_only = mode == "amplitude_write_only"
        expected_status = "sent" if write_only else "ok"
        if (
            event["status"] != expected_status
            or not timing
            or (not write_only and timing["confirmed_s"] is None)
        ):
            pending.pop(pending_key, None)
            continue
        if write_only and timing["confirmed_s"] is not None:
            raise ValueError("只发送记录不能包含逐次设备确认时间")
        sent = timing["command_sent_s"]
        written = timing["command_write_returned_s"]
        confirmed = timing["confirmed_s"]
        times = [
            event["actual_start_s"],
            sent,
            written,
        ]
        if not write_only:
            times.append(confirmed)
        times.append(event["actual_finish_s"])
        if not all(
            isinstance(value, (float, int)) and math.isfinite(value) for value in times
        ):
            raise ValueError(f"第 {run_index} 次 {channel} 缺少有效通信时间")
        if list(times) != sorted(times):
            raise ValueError(f"第 {run_index} 次 {channel} 通信时间顺序不正确")
        if enabled:
            pending.setdefault(pending_key, event)
        elif pending_key in pending:
            on = pending.pop(pending_key)
            on_timing = on[timing_key]
            earlier_finished = (
                on_timing["command_write_returned_s"]
                if write_only
                else on_timing["confirmed_s"]
            )
            if sent < earlier_finished:
                raise ValueError(f"第 {run_index} 次 {channel} 关闭早于开启确认")
            rows.append(
                {
                    "run_index": run_index,
                    "channel": channel,
                    "control_mode": mode,
                    f"{start_prefix}_command_sent_s": on_timing["command_sent_s"],
                    f"{start_prefix}_confirmed_s": on_timing["confirmed_s"],
                    f"{start_prefix}_command_write_returned_s": on_timing[
                        "command_write_returned_s"
                    ],
                    f"{end_prefix}_command_sent_s": sent,
                    f"{end_prefix}_confirmed_s": confirmed,
                    f"{end_prefix}_command_write_returned_s": written,
                    f"{start_prefix}_confirmation_delay_ms": None
                    if write_only
                    else (on_timing["confirmed_s"] - on_timing["command_sent_s"])
                    * 1000,
                    f"{end_prefix}_confirmation_delay_ms": None
                    if write_only
                    else (confirmed - sent) * 1000,
                    f"{start_prefix}_schedule_delay_ms": (
                        on_timing["command_sent_s"] - on["planned_s"]
                    )
                    * 1000,
                    f"{end_prefix}_schedule_delay_ms": (sent - event["planned_s"])
                    * 1000,
                    f"{start_prefix}_before_call_ms": (
                        on["actual_start_s"] - on["planned_s"]
                    )
                    * 1000,
                    f"{start_prefix}_{'local_prepare' if write_only else 'precheck'}_ms": (
                        on_timing["command_sent_s"] - on["actual_start_s"]
                    )
                    * 1000,
                    f"{end_prefix}_before_call_ms": (
                        event["actual_start_s"] - event["planned_s"]
                    )
                    * 1000,
                    f"{end_prefix}_{'local_prepare' if write_only else 'precheck'}_ms": (
                        sent - event["actual_start_s"]
                    )
                    * 1000,
                    f"{start_prefix}_write_ms": (
                        on_timing["command_write_returned_s"]
                        - on_timing["command_sent_s"]
                    )
                    * 1000,
                    f"{end_prefix}_write_ms": (written - sent) * 1000,
                }
            )
    return rows


def describe(values: list[float]) -> dict:
    """报告均值、中位数、样本标准差和范围；不擅自设定稳定性合格线。"""
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "sample_stddev": stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def write_timing_report(output_dir: Path, run_dirs: list[Path], requested: int) -> dict:
    """关断完成后汇总文件；失败运行保留状态，统计只使用完整成功的实验。"""
    rows = []
    runs = []
    commands = []
    for index, run_dir in enumerate(run_dirs, 1):
        summary_path = run_dir / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {"status": "incomplete"}
        )
        runs.append(
            {
                "run_index": index,
                "directory": run_dir.name,
                "status": summary["status"],
                "control_mode": summary.get("control_mode", "output_switch")
                if summary_path.exists()
                else None,
                "data_reliable": summary.get("data_reliable"),
                "error": summary.get("error"),
            }
        )
        event_path = run_dir / "events.jsonl"
        events = (
            [
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
            ]
            if event_path.exists()
            else []
        )
        for row in parameter_rows(events, index):
            row["run_status"] = summary["status"]
            row["data_reliable"] = summary.get("data_reliable")
            commands.append(row)
        if summary["status"] != "completed":
            continue
        if (
            summary.get("control_mode", "").endswith("_write_only")
            and summary.get("data_reliable") is not True
        ):
            continue
        for row in pulse_rows(events, index):
            if summary.get("control_mode") == "frequency_amplitude_write_only":
                row["control_mode"] = summary["control_mode"]
            row["timeline_started_utc"] = summary["timeline_started_utc"]
            rows.append(row)

    modes = {run["control_mode"] for run in runs if run["control_mode"] is not None} | {
        row["control_mode"] for row in rows
    }
    if len(modes) > 1:
        raise ValueError("不能把旧输出开关与新幅值状态记录混合成同一份统计")
    mode = next(iter(modes), "amplitude_states")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["channel"]].append(row)
    statistics = {}
    for channel, channel_rows in grouped.items():
        statistics[channel] = {
            metric: describe(
                [row[metric] for row in channel_rows if row[metric] is not None]
            )
            for metric in channel_rows[0]
            if metric.endswith("_ms")
            and any(row[metric] is not None for row in channel_rows)
        }
    report = {
        "requested_runs": requested,
        "completed_runs": sum(run["status"] == "completed" for run in runs),
        "reliable_runs": sum(
            run["status"] == "completed" and run["data_reliable"] is True
            for run in runs
        ),
        "unreliable_runs": sum(run["status"] == "unreliable" for run in runs),
        "failed_runs": sum(run["status"] == "failed" for run in runs),
        "paired_pulses": len(rows),
        "paired_intervals": len(rows),
        "recorded_parameter_commands": len(commands),
        "control_mode": mode,
        "runs": runs,
        "statistics_ms": statistics,
        "time_reference": "seconds_since_each_run_timeline_started",
        "command_sent_definition": "immediately_before_VISA_write_call",
        "confirmed_definition": (
            "not_measured_write_only_commands_with_round_boundary_checks"
            if mode.endswith("_write_only")
            else "amplitude_reply_received_at_PC_and_subsequently_validated_by_driver"
            if mode == "amplitude_states"
            else "output_state_reply_received_at_PC_and_subsequently_validated_by_driver"
        ),
        "per_command_confirmation": not mode.endswith("_write_only"),
        "physical_voltage_timing_verified": False,
    }
    if commands:
        with (output_dir / "parameter_commands.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(commands[0]))
            writer.writeheader()
            writer.writerows(commands)
    if rows:
        with (output_dir / "timing.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "timing_summary.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    if runs:
        with (output_dir / "all_attempts.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)
    return report
