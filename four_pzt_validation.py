"""四路 PZT 的六项实机验收；控制动作复用正式执行器，只在稳定时段读回。

每项准备完成后计时，20 秒进入 STOP，25 秒内完成核查和记录；第六项
在 10 秒请求取消。普通实验入口不导入本模块，也不增加这些中途查询。
"""

import argparse
import csv
import json
import math
import os
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from customer_workspace.afg_timing import RoundUnreliable, TimedRig
from customer_workspace.experiment_runner import (
    CHANNEL_NAMES,
    IDLE_AMPLITUDE_VPP,
    PROJECT_ROOT,
    ExperimentCancelled,
    ExperimentPlan,
    ExperimentRunner,
    load_plan,
)
from customer_workspace.timing_report import parameter_rows, write_timing_report
from customer_workspace.wsl_launcher import launch_from_wsl, watch_launcher_input
from microbubble_experiment.afg import DEFAULT_CONFIG, AFGError, Rig, load_config

CASES = {
    1: ("01_sequential", "四路轮流工作"),
    2: ("02_groups", "不同组合工作"),
    3: ("03_all_together", "四路共同工作与待机"),
    4: ("04_frequencies", "四路独立调频"),
    5: ("05_combined", "四路分别设置频率和幅值"),
    6: ("06_cancel", "工作途中取消"),
}
CHECK_DELAY_S = 1.0
NEXT_STEP_MARGIN_S = 0.25
CASE_LIMIT_S = 25.0


def expected_at(plan: ExperimentPlan, at_s: float) -> dict:
    """按计划推算三项应读回的设置；不以已经发出的命令代替实际读回。"""
    expected = {
        name: {
            "output": True,
            "frequency_hz": setting.frequency_hz,
            "amplitude_vpp": IDLE_AMPLITUDE_VPP,
        }
        for name, setting in plan.settings.items()
    }
    for step in plan.steps:
        if step.at_s > at_s:
            break
        if step.state == "STOP":
            for state in expected.values():
                state["output"] = False
            break
        for name, frequency in step.frequencies_hz.items():
            expected[name]["frequency_hz"] = frequency
        for name in CHANNEL_NAMES:
            expected[name]["amplitude_vpp"] = (
                plan.settings[name].amplitude_vpp
                if name in step.active_channels
                else IDLE_AMPLITUDE_VPP
            )
    return expected


def compare_readback(expected: dict, actual: dict) -> list[str]:
    """核对四路；幅值和频率容差与原驱动一致，不改变原始读回值。"""
    problems = []
    for name in CHANNEL_NAMES:
        if name not in actual:
            problems.append(f"{name} 缺少读回")
            continue
        target, state = expected[name], actual[name]
        if state.get("output") is not target["output"]:
            problems.append(f"{name} 输出状态不符")
        for key, tolerance in (
            ("frequency_hz", max(1e-6, target["frequency_hz"] * 1e-9)),
            ("amplitude_vpp", 0.00051),
        ):
            value = state.get(key)
            if (
                value is None
                or not math.isfinite(value)
                or abs(value - target[key]) > tolerance
            ):
                problems.append(f"{name} {key}：应为 {target[key]}，读到 {value}")
    return problems


def check_times(plan: ExperimentPlan, cancel_at_s: float | None) -> list[float]:
    """每段开始后 1 秒检查；取消前再检查一次，确认取消时四路原处于工作设置。"""
    times = []
    for step, following in zip(plan.steps, plan.steps[1:]):
        limit = (
            min(following.at_s, cancel_at_s)
            if cancel_at_s is not None
            else following.at_s
        )
        at_s = step.at_s + CHECK_DELAY_S
        if at_s >= limit - NEXT_STEP_MARGIN_S:
            raise ValueError("稳定读回需要切换后 1 秒，并在下一动作前留出时间")
        times.append(at_s)
        if cancel_at_s is not None and following.at_s >= cancel_at_s:
            before_cancel = cancel_at_s - 1.0
            if before_cancel > at_s:
                times.append(before_cancel)
            break
    return times


class ValidationRunner(ExperimentRunner):
    """复用全部正式控制与收尾，在等待下一个步骤时插入验收读回。"""

    def __init__(self, rig, plan, cancel_event, cancel_at_s=None):
        super().__init__(rig, cancel_event=cancel_event)
        self.plan = plan
        self.cancel_at_s = cancel_at_s
        self.checks = []
        self.cancel_request = None
        self._previous_step = None

    def _query_before(self, name, session, command, deadline_s, report):
        """一次查询，不重试；临时缩短会话超时，使失败尽早进入关断。"""
        self._check_cancelled()
        remaining_s = deadline_s - monotonic()
        if remaining_s <= 0:
            raise TimeoutError("稳定读回已到截止时间，停止本项")
        record = {
            "target": name,
            "command": command,
            "started_s": self._elapsed(monotonic()),
        }
        report["queries"].append(record)
        visa_session = session._session
        original_timeout = visa_session.timeout
        try:
            # query 内含写入和读取；两者各至多使用剩余预算的一半。
            visa_session.timeout = max(
                1, min(original_timeout, 250, int(remaining_s * 500))
            )
            reply = self.rig._query(session, command)
            record["reply"] = reply
            if monotonic() >= deadline_s:
                raise TimeoutError("稳定读回超过截止时间，停止本项")
            self._check_cancelled()
            return reply
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["error_notes"] = getattr(exc, "__notes__", [])
            raise
        finally:
            record["finished_s"] = self._elapsed(monotonic())
            visa_session.timeout = original_timeout

    def _snapshot(self, step, planned_s, limit_s):
        super()._wait_until(planned_s)
        report = {
            "phase": "stable",
            "step_at_s": step.at_s,
            "planned_check_s": planned_s,
            "deadline_s": limit_s - NEXT_STEP_MARGIN_S,
            "started_s": self._elapsed(monotonic()),
            "expected": expected_at(self.plan, step.at_s),
            "channels": {},
            "queries": [],
            "error_queue_replies": {},
            "problems": [],
            "passed": False,
        }
        self.checks.append(report)
        deadline = self._start_s + report["deadline_s"]
        before = {alias: dict(s.io_counts) for alias, s in self.rig.sessions.items()}
        try:
            # 避免严重晚发后立即查询，重现此前紧邻调幅与查询的通信问题。
            for timing in (
                self.rig.last_frequency_timing,
                self.rig.last_amplitude_timing,
            ):
                if (
                    timing is not None
                    and timing.write_returned is not None
                    and monotonic() - timing.write_returned < 0.100
                ):
                    raise TimeoutError("距离上次参数写入不足 100 ms，停止本项")
            with self.rig.lock:
                for name in CHANNEL_NAMES:
                    session, mapping = self.rig._target(name)
                    channel = mapping["channel"]
                    actual = {}
                    report["channels"][name] = actual
                    for key, command in (
                        ("output", f"OUTPut{channel}:STATe?"),
                        ("frequency_hz", f"SOURce{channel}:FREQuency:FIXed?"),
                        (
                            "amplitude_vpp",
                            f"SOURce{channel}:VOLTage:LEVel:IMMediate:AMPLitude?",
                        ),
                    ):
                        reply = self._query_before(
                            name, session, command, deadline, report
                        )
                        value = (
                            self.rig._state(reply) if key == "output" else float(reply)
                        )
                        if key != "output" and not math.isfinite(value):
                            raise AFGError(f"{name} {key} 回复不是有限数值：{reply!r}")
                        actual[key] = value
                for alias, session in self.rig.sessions.items():
                    reply = self._query_before(
                        alias, session, "SYSTem:ERRor:NEXT?", deadline, report
                    )
                    report["error_queue_replies"][alias] = reply
                    if int(reply.split(",", 1)[0]) != 0:
                        report["problems"].append(f"{alias} 错误队列：{reply}")
            report["problems"].extend(
                compare_readback(report["expected"], report["channels"])
            )
            if report["problems"]:
                raise RoundUnreliable(report)
            report["passed"] = True
        except ExperimentCancelled:
            report["problems"].append("读回期间请求取消")
            raise
        except RoundUnreliable:
            raise
        except Exception as exc:
            report["problems"].append(f"{type(exc).__name__}: {exc}")
            raise RoundUnreliable(report) from exc
        finally:
            report["finished_s"] = self._elapsed(monotonic())
            report["io_counts"] = {
                alias: {
                    key: value - before[alias][key]
                    for key, value in s.io_counts.items()
                }
                for alias, s in self.rig.sessions.items()
            }
        print(f"  {planned_s:g} 秒稳定读回：四路设置均符合计划", flush=True)

    def _wait_until(self, planned_s):
        if self._previous_step is not None:
            limit = (
                min(planned_s, self.cancel_at_s)
                if self.cancel_at_s is not None
                else planned_s
            )
            self._snapshot(
                self._previous_step, self._previous_step.at_s + CHECK_DELAY_S, limit
            )
            if self.cancel_at_s is not None and planned_s >= self.cancel_at_s:
                extra_check_s = self.cancel_at_s - 1.0
                if extra_check_s > self._previous_step.at_s + CHECK_DELAY_S:
                    self._snapshot(self._previous_step, extra_check_s, self.cancel_at_s)
                super()._wait_until(self.cancel_at_s)
                self.cancel_request = {
                    "planned_s": self.cancel_at_s,
                    "actual_s": self._elapsed(monotonic()),
                }
                self.cancel()
                self._check_cancelled()
        super()._wait_until(planned_s)
        self._previous_step = next(
            step for step in self.plan.steps if step.at_s == planned_s
        )


def save_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def run_case(config, plan, output, case_number, cancel_event):
    """单项完成后先独立确认关闭，再保存验收记录；任何失败由批次调用方停止。"""
    output.mkdir(parents=True, exist_ok=False)
    cancel_at_s = 10.0 if case_number == 6 else None
    planned_checks = check_times(plan, cancel_at_s)
    rig = TimedRig(config)
    runner = ValidationRunner(rig, plan, cancel_event, cancel_at_s)
    problems = []
    execution_error = None
    try:
        with rig:
            runner.run(plan, output / "experiment")
    except BaseException as exc:  # noqa: BLE001 - 真实中断也要保存与复查关断。
        execution_error = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        if not (
            isinstance(exc, ExperimentCancelled) and runner.cancel_request is not None
        ):
            problems.append(execution_error["error"])

    postflight = {
        "checked_utc": datetime.now(UTC).isoformat(),
        "all_outputs_off": False,
    }
    try:
        with Rig(config) as observer:
            postflight["devices"] = dict(observer.identities)
            postflight["channels"] = observer.status()
        postflight["all_outputs_off"] = len(postflight["channels"]) == 4 and all(
            state["output"] is False for state in postflight["channels"].values()
        )
        if not postflight["all_outputs_off"]:
            problems.append("独立复查未确认四路全部关闭")
    except BaseException as exc:  # noqa: BLE001 - 失败证据仍须落盘。
        postflight["error"] = f"{type(exc).__name__}: {exc}"
        problems.append(postflight["error"])
    postflight["finished_s"] = runner._elapsed(monotonic())
    save_json(output / "postflight.json", postflight)

    summary_path = output / "experiment/summary.json"
    experiment = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    expected_status = "cancelled" if case_number == 6 else "completed"
    if experiment.get("status") != expected_status:
        problems.append(
            f"预期 {expected_status}，实际 {experiment.get('status', '未开始')}"
        )
    if case_number == 6 and runner.cancel_request is None:
        problems.append("未执行预定的第 10 秒取消")
    for key in ("precheck", "postcheck"):
        if not experiment.get(key, {}).get("passed"):
            problems.append(f"{key} 未通过")
    if not experiment.get("all_outputs_off_confirmed"):
        problems.append("执行器未确认四路全部关闭")
    if [r["planned_check_s"] for r in runner.checks] != planned_checks or not all(
        r["passed"] for r in runner.checks
    ):
        problems.append("稳定时段检查不完整或未通过")

    with (output / "stable_checks.jsonl").open("x", encoding="utf-8") as file:
        for report in runner.checks:
            file.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
    flat = []
    for report in runner.checks:
        for name in CHANNEL_NAMES:
            actual = report["channels"].get(name, {})
            target = report["expected"][name]
            flat.append(
                {
                    "planned_check_s": report["planned_check_s"],
                    "channel": name,
                    "expected_frequency_hz": target["frequency_hz"],
                    "actual_frequency_hz": actual.get("frequency_hz"),
                    "expected_amplitude_vpp": target["amplitude_vpp"],
                    "actual_amplitude_vpp": actual.get("amplitude_vpp"),
                    "expected_output": target["output"],
                    "actual_output": actual.get("output"),
                    "started_s": report["started_s"],
                    "finished_s": report["finished_s"],
                    "check_passed": report["passed"],
                }
            )
    if flat:
        with (output / "stable_channels.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    events_path = output / "experiment/events.jsonl"
    commands = []
    shutdown = None
    if events_path.exists():
        try:
            write_timing_report(output / "experiment", [output / "experiment"], 1)
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            commands = parameter_rows(events, 1)
            shutdown = next(
                (e for e in events if e["event"] == "shutdown_finished"), None
            )
            if runner.cancel_request and any(
                r["command_sent_s"] is not None
                and r["command_sent_s"] >= runner.cancel_request["actual_s"]
                for r in commands
            ):
                problems.append("取消后仍发送了普通参数命令")
        except Exception as exc:  # noqa: BLE001 - 汇总失败也要保留验收与关断结果。
            problems.append(f"汇总失败：{type(exc).__name__}: {exc}")
    finished_s = runner._elapsed(monotonic())
    if finished_s is None or finished_s > CASE_LIMIT_S:
        problems.append(f"本项未在 25 秒内完成，完成时刻为 {finished_s}")
    lateness = [
        r["send_lateness_ms"] for r in commands if r["send_lateness_ms"] is not None
    ]
    summary = {
        "case": case_number,
        "name": CASES[case_number][1],
        "passed": not problems,
        "problems": problems,
        "experiment_status": experiment.get("status"),
        "timeline_started_utc": experiment.get("timeline_started_utc"),
        "completed_s": finished_s,
        "limit_s": CASE_LIMIT_S,
        "stable_checks": len(runner.checks),
        "stable_channel_readbacks": len(flat),
        "parameter_commands": len(commands),
        "maximum_command_lateness_ms": max(lateness) if lateness else None,
        "cancel_request": runner.cancel_request,
        "shutdown": shutdown,
        "independent_all_outputs_off": postflight["all_outputs_off"],
        "execution_error": execution_error,
        "physical_signal_timing_measured": False,
    }
    save_json(output / "validation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, nargs="?")
    parser.add_argument("--case", type=int, choices=CASES, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="AFG 设备配置，默认 %(default)s",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows-python", type=Path)
    parser.add_argument("--launcher-stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        plan_path = (
            args.plan
            or PROJECT_ROOT
            / "customer_workspace/configs/four_pzt"
            / f"{CASES[args.case][0]}.json"
        )
        config = load_config(args.config)
        plan = load_plan(plan_path, config)
        if plan.duration_s != 20:
            raise ValueError("六项验收的时间表必须在第 20 秒 STOP，留下 5 秒收尾")
        check_times(plan, 10.0 if args.case == 6 else None)
        if args.case == 6 and not all(
            state["amplitude_vpp"] > IDLE_AMPLITUDE_VPP
            for state in expected_at(plan, 9).values()
        ):
            raise ValueError("取消测试必须在第 9 秒保持四路工作")
        if os.name != "nt":
            return launch_from_wsl(
                PROJECT_ROOT,
                plan_path,
                args.config,
                args.output_dir,
                args.windows_python,
                module="customer_workspace.four_pzt_validation",
                extra_args=["--case", str(args.case)],
            )
        cancel = threading.Event()
        if args.launcher_stdin:
            threading.Thread(
                target=watch_launcher_input, args=(cancel.set,), daemon=True
            ).start()
        if cancel.is_set():
            raise ExperimentCancelled("验收已取消")
        print(f"第 {args.case} 项：{CASES[args.case][1]}；准备后开始计时。", flush=True)
        summary = run_case(config, plan, args.output_dir, args.case, cancel)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if summary["passed"] else 1
    except (KeyboardInterrupt, ExperimentCancelled):
        print("验收已取消，请查看关断记录。", flush=True)
        return 130
    except Exception:  # noqa: BLE001 - 命令行保留完整错误并返回失败状态。
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
