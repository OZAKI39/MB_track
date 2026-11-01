"""真实 AFG 专项测试：定时只发送、连续读回或定时发送并查询幅值。

正式 Experiment Runner 不导入此模块。查询回复仅表示参数读回，不能证明
模拟电压已稳定。调幅后至少等待 10 ms 才发查询，等待包含在总耗时中。
时间表期间记录暂存内存，全部关断后再写文件。
"""

import argparse
import csv
import json
import math
import os
import threading
import traceback
from dataclasses import asdict, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from time import monotonic, sleep

from customer_workspace.afg_timing import RoundUnreliable, TimedRig
from customer_workspace.benchmark_report import distribution, summarize_samples
from customer_workspace.experiment_runner import (
    CHANNEL_NAMES,
    IDLE_AMPLITUDE_VPP,
    PROJECT_ROOT,
    WAKE_AHEAD_S,
    ExperimentCancelled,
    check_cancelled,
    load_plan,
    wait_until,
)
from customer_workspace.wsl_launcher import launch_from_wsl, watch_launcher_input
from microbubble_experiment.afg import DEFAULT_CONFIG, AFGError, load_config

BEFORE_QUERY_MIN_MS = 10.0


def validate_protocol(method: str, count: int, interval_ms: float) -> None:
    """连接设备前拒绝无效参数；偶数条指令保证正常序列以 IDLE 结束。"""
    if method not in ("schedule", "readback", "scheduled_readback"):
        raise ValueError("method 必须为 schedule、readback 或 scheduled_readback")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2 or count % 2:
        raise ValueError("count 必须为不小于 2 的偶数，表示调幅指令条数")
    if not math.isfinite(interval_ms) or interval_ms <= 0:
        raise ValueError("interval-ms 必须为正的有限毫秒数")


def sample_command(
    rig: TimedRig,
    row: dict,
    start_s: float,
    readback: bool,
    cancel_event: threading.Event,
) -> None:
    """复用只发送调幅；查询测试在写入返回后等待，失败保留部分时间。"""
    with rig.lock:
        session, channel = rig._target(row["channel"])
        query = f"SOURce{channel['channel']}:VOLTage:LEVel:IMMediate:AMPLitude?"
        before = dict(session.io_counts)
        rig.last_amplitude_timing = None
        row["call_started_s"] = monotonic() - start_s
        try:
            rig.send_amplitude(row["channel"], row["amplitude_requested_vpp"])
            timing = rig.last_amplitude_timing
            if readback:
                # 恢复同一个计时对象，让会话在原始 read/query 返回处采集时间。
                timing.query = query
                timing.succeeded = False
                session.timing = timing
                row["before_query_min_ms"] = BEFORE_QUERY_MIN_MS
                # 实测可规避紧邻调幅与查询触发的 VI_ERROR_IO；不能移到查询后。
                # 沿用上一轮诊断的 sleep，避免 Event.wait 在本环境中拉长等待。
                check_cancelled(cancel_event)
                sleep(BEFORE_QUERY_MIN_MS / 1000)
                check_cancelled(cancel_event)
                row["query_started_s"] = monotonic() - start_s
                reply = rig._query(session, query)
                actual = float(reply)
                if not math.isfinite(actual) or not math.isclose(
                    actual, row["amplitude_requested_vpp"], rel_tol=1e-6, abs_tol=1e-9
                ):
                    raise AFGError(
                        f"幅值回复不符：请求 {row['amplitude_requested_vpp']}，收到 {reply!r}"
                    )
                row["amplitude_readback_vpp"] = actual
                timing.succeeded = True
            row["status"] = "readback_ok" if readback else "sent"
        except BaseException as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["error_notes"] = getattr(exc, "__notes__", [])
            raise
        finally:
            session.timing = None
            row["call_finished_s"] = monotonic() - start_s
            timing = rig.last_amplitude_timing
            if timing is not None:
                row.update(timing.relative_to(start_s, reply_name="amplitude"))
                if row.get("query_started_s") is not None:
                    row["write_to_query_ms"] = (
                        row["query_started_s"] - row["command_write_returned_s"]
                    ) * 1000
                if timing.command_sent is not None:
                    if row["planned_s"] is not None:
                        row["lateness_ms"] = (
                            timing.command_sent - start_s - row["planned_s"]
                        ) * 1000
                    if timing.write_returned is not None:
                        row["write_ms"] = (
                            timing.write_returned - timing.command_sent
                        ) * 1000
                    if timing.state_reply_received is not None:
                        row["command_to_reply_ms"] = (
                            timing.state_reply_received - timing.command_sent
                        ) * 1000
                        if row["planned_s"] is not None:
                            row["planned_to_reply_ms"] = (
                                timing.state_reply_received - start_s - row["planned_s"]
                            ) * 1000
            row["io_delta"] = {
                key: value - before[key] for key, value in session.io_counts.items()
            }


def save_results(output_dir: Path, samples: list[dict], summary: dict) -> None:
    """关断和核查后写原始记录、平面 CSV 及摘要；从不覆盖已有文件。"""
    with (output_dir / "samples.jsonl").open("x", encoding="utf-8") as file:
        for row in samples:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    if samples:
        fields = list(dict.fromkeys(key for row in samples for key in row))
        with (output_dir / "samples.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in samples:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                        for key, value in row.items()
                    }
                )
    with (output_dir / "summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)


def run_benchmark(
    rig, plan, output_dir, method, channel, count, interval_ms, cancel_event
):
    """一次连接只跑一种测量；失败、取消或核查异常后先 STOP，再保存证据。"""
    validate_protocol(method, count, interval_ms)
    scheduled = method in ("schedule", "scheduled_readback")
    readback = method in ("readback", "scheduled_readback")
    plan.validate(rig.config)
    if channel not in CHANNEL_NAMES:
        raise ValueError("未知 PZT 通道")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "method": method,
        "channel": channel,
        "requested_commands": count,
        "interval_ms": interval_ms if scheduled else None,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "failed",
        "data_reliable": False,
        "physical_voltage_timing_verified": False,
        "max_allowed_schedule_lateness_s": 1.0,
        "before_query_min_ms": BEFORE_QUERY_MIN_MS if readback else 0.0,
        "before_query_wait_method": "time.sleep" if readback else None,
        "schedule_wait_method": "Event.wait then perf_counter spin"
        if scheduled
        else None,
        "schedule_wake_ahead_ms": WAKE_AHEAD_S * 1000 if scheduled else None,
    }
    with (output_dir / "configuration.json").open("x", encoding="utf-8") as file:
        json.dump(
            {
                "protocol": summary,
                "settings": {
                    name: asdict(value) for name, value in plan.settings.items()
                },
                "afg_config": rig.config,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    idle_settings = {
        name: replace(value, amplitude_vpp=IDLE_AMPLITUDE_VPP)
        for name, value in plan.settings.items()
    }
    expected = dict(idle_settings)
    samples = []
    problem = None
    start_s = None
    connected = False
    try:
        check_cancelled(cancel_event)
        summary["devices"] = rig.connect()
        connected = True
        failures = rig.all_off()
        if failures:
            raise AFGError("准备关断未确认：" + "; ".join(failures))
        check_cancelled(cancel_event)
        rig.apply(idle_settings)
        summary["precheck"] = rig.check_state(idle_settings, "before")
        if not summary["precheck"]["passed"]:
            raise RoundUnreliable(summary["precheck"])
        for name in CHANNEL_NAMES:
            check_cancelled(cancel_event)
            if rig.set_output(name, True) is not True:
                raise AFGError(f"{name} 开启未确认")
        check_cancelled(cancel_event)
        counts_before = {
            alias: dict(session.io_counts) for alias, session in rig.sessions.items()
        }
        summary["timeline_started_utc"] = datetime.now(UTC).isoformat()
        start_s = monotonic()
        for index in range(count):
            planned_s = index * interval_ms / 1000 if scheduled else None
            ready_s = monotonic() - start_s
            if planned_s is not None:
                wait_until(start_s + planned_s, cancel_event)
                # 此诊断允许观察短时落后和追赶；严重积压时结束，避免无限补发。
                if monotonic() - start_s - planned_s > 1.0:
                    raise TimeoutError(
                        f"第 {index + 1} 条已落后计划超过 1 秒，停止测试"
                    )
            check_cancelled(cancel_event)
            target = (
                plan.settings[channel] if index % 2 == 0 else idle_settings[channel]
            )
            row = {
                "index": index + 1,
                "channel": channel,
                "state": "ACTIVE" if index % 2 == 0 else "IDLE",
                "amplitude_requested_vpp": target.amplitude_vpp,
                "planned_s": planned_s,
                "ready_s": ready_s,
            }
            samples.append(row)
            try:
                sample_command(rig, row, start_s, readback, cancel_event)
            finally:
                if row.get("command_write_returned_s") is not None:
                    expected[channel] = target
        summary["measurement_io_counts"] = {
            alias: {
                key: value - counts_before[alias][key]
                for key, value in session.io_counts.items()
            }
            for alias, session in rig.sessions.items()
        }
        summary["status"] = "completed"
    except BaseException as exc:  # noqa: BLE001 - 保存后重抛，中断同样必须先关断。
        problem = exc
        summary["status"] = (
            "cancelled"
            if isinstance(exc, (ExperimentCancelled, KeyboardInterrupt))
            else "unreliable"
            if isinstance(exc, RoundUnreliable)
            else "failed"
        )
    finally:
        shutdown_start = monotonic()
        try:
            failures = rig.all_off() if connected else []
        except BaseException as exc:  # noqa: BLE001 - 关断异常不能覆盖最早的失败原因。
            failures = [f"{type(exc).__name__}: {exc}"]
        summary["shutdown_started_s"] = (
            None if start_s is None else shutdown_start - start_s
        )
        summary["shutdown_finished_s"] = (
            None if start_s is None else monotonic() - start_s
        )
        summary["all_outputs_off_confirmed"] = connected and not failures
        if not connected:
            summary["shutdown_skipped_reason"] = (
                "设备连接未完成，未开始测试，最终状态未读回"
            )
        summary["shutdown_errors"] = failures
        if failures:
            summary["status"] = "failed"
            if problem is None:
                problem = AFGError("关断未确认：" + "; ".join(failures))
        if start_s is not None and not failures:
            try:
                summary["postcheck_started_s"] = monotonic() - start_s
                summary["postcheck"] = rig.check_state(expected, "after")
                if not summary["postcheck"]["passed"]:
                    raise RoundUnreliable(summary["postcheck"])
            except BaseException as exc:  # noqa: BLE001 - 已尝试关断，保留轮末核查错误。
                if problem is None:
                    problem = exc
                    summary["status"] = (
                        "unreliable" if isinstance(exc, RoundUnreliable) else "failed"
                    )
                else:
                    problem.add_note(f"轮末核查异常：{exc}")
        try:
            rig.close()
        except BaseException as exc:  # noqa: BLE001 - 关闭连接前再次关断失败也必须记录。
            summary["close_error"] = f"{type(exc).__name__}: {exc}"
            summary["all_outputs_off_confirmed"] = False
            summary["status"] = "failed"
            if problem is None:
                problem = exc
        summary["recorded_commands"] = len(samples)
        summary["finished_utc"] = datetime.now(UTC).isoformat()
        summary["data_reliable"] = summary["status"] == "completed" and problem is None
        try:
            summary["statistics"] = (
                summarize_samples(samples, method, interval_ms)
                if summary["data_reliable"]
                else None
            )
            summary["actual_command_intervals_ms"] = (
                distribution(
                    [
                        (following["command_sent_s"] - previous["command_sent_s"])
                        * 1000
                        for previous, following in pairwise(samples)
                    ]
                )
                if summary["data_reliable"]
                else None
            )
        except Exception as exc:  # noqa: BLE001 - 分析失败仍保留原始测量和关断结果。
            summary["analysis_error"] = f"{type(exc).__name__}: {exc}"
            summary["status"] = "failed"
            summary["data_reliable"] = False
            if problem is None:
                problem = exc
        if isinstance(problem, RoundUnreliable):
            summary["validation_failure"] = problem.report
        if problem is not None:
            summary["error"] = f"{type(problem).__name__}: {problem}"
            summary["error_notes"] = getattr(problem, "__notes__", [])
            summary["error_traceback"] = "".join(traceback.format_exception(problem))
        save_results(output_dir, samples, summary)
    if problem is not None:
        raise problem
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="真实 AFG 延迟专项测试；自动由 WSL 转交 Windows"
    )
    parser.add_argument(
        "plan", type=Path, help="沿用此 JSON 的四路频率和幅值；不执行其中的 steps"
    )
    parser.add_argument(
        "--method",
        choices=("schedule", "readback", "scheduled_readback"),
        required=True,
    )
    parser.add_argument("--channel", choices=CHANNEL_NAMES, default="PZT1")
    parser.add_argument(
        "--count", type=int, default=500, help="调幅指令条数，必须为偶数"
    )
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=20,
        help="schedule／scheduled_readback 的计划间隔；readback 不按时刻表等待",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="AFG 设备配置，默认 %(default)s",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--windows-python", type=Path)
    parser.add_argument("--launcher-stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        validate_protocol(args.method, args.count, args.interval_ms)
        config = load_config(args.config)
        plan = load_plan(args.plan, config)
        if os.name != "nt":
            return launch_from_wsl(
                PROJECT_ROOT,
                args.plan,
                args.config,
                args.output_dir,
                args.windows_python,
                module="customer_workspace.timing_benchmark",
                extra_args=[
                    "--method",
                    args.method,
                    "--channel",
                    args.channel,
                    "--count",
                    str(args.count),
                    "--interval-ms",
                    str(args.interval_ms),
                ],
            )
        output_dir = (
            args.output_dir
            or PROJECT_ROOT
            / "customer_workspace/validation"
            / f"{args.method}_{datetime.now():%Y%m%d-%H%M%S}"
        )
        if output_dir.exists():
            raise FileExistsError(f"结果目录已存在：{output_dir}")
        rig = TimedRig(config)
        cancel_event = threading.Event()

        def cancel():
            cancel_event.set()
            rig.request_stop()

        if args.launcher_stdin:
            threading.Thread(
                target=watch_launcher_input, args=(cancel,), daemon=True
            ).start()
        print(
            f"开始 {args.method}：{args.count} 条调幅指令，通道 {args.channel}；记录 {output_dir}",
            flush=True,
        )
        summary = run_benchmark(
            rig,
            plan,
            output_dir,
            args.method,
            args.channel,
            args.count,
            args.interval_ms,
            cancel_event,
        )
        print(json.dumps(summary["statistics"], ensure_ascii=False, indent=2))
        return 0
    except (ExperimentCancelled, KeyboardInterrupt):
        print("测试已取消，请检查结果目录中的关断记录。")
        return 130
    except Exception as exc:  # noqa: BLE001 - 命令行报告包含 VISA 错误在内的失败。
        traceback.print_exception(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
