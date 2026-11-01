"""按相对时间表控制四个 PZT，复用现有 AFG 驱动并保存逐项执行记录。

本模块不会在导入时连接仪器。运行命令会检查计划并直接执行真实实验；
WSL 会自动调用已有 Windows Python。记录的是软件调用和设备读回时间，
不是示波器测得的电压变化时刻，也不保证四路同时切换。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from time import monotonic, perf_counter
from typing import TextIO

from customer_workspace.afg_timing import RoundUnreliable, TimedRig
from customer_workspace.timing_report import write_timing_report
from customer_workspace.wsl_launcher import launch_from_wsl, watch_launcher_input
from microbubble_experiment.afg import (
    DEFAULT_CONFIG,
    AFGError,
    ChannelSettings,
    Rig,
    load_config,
)

CHANNEL_NAMES = ("PZT1", "PZT2", "PZT3", "PZT4")
IDLE_AMPLITUDE_VPP = 0.001
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Windows 实测：各等待长度 P99 最大 20.709 ms，加 1 ms 后向上取整。
# 原始记录：validation/pair_wait_20261023-213421/wait_calibration/
WAKE_AHEAD_S = 0.022


class ExperimentCancelled(RuntimeError):
    """用户请求停止；执行器仍会尝试关闭所有输出并保存记录。"""


def check_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise ExperimentCancelled("实验已取消")


def wait_until(
    deadline_s: float,
    cancel_event: threading.Event,
    wake_ahead_s: float = WAKE_AHEAD_S,
) -> None:
    """按 monotonic 绝对时刻等待；正式实验与专项测试共用已验证的方法。

    先可取消休眠，最后用 perf_counter 循环并检查取消。显式转换剩余
    时间，不依赖两个时钟起点相同；原计划不随上一次完成时间移动。
    """
    if not math.isfinite(wake_ahead_s) or wake_ahead_s < 0:
        raise ValueError("提前量必须为非负有限秒数")
    while True:
        check_cancelled(cancel_event)
        remaining_s = deadline_s - monotonic()
        if remaining_s <= 0:
            return
        if remaining_s > wake_ahead_s:
            cancel_event.wait(remaining_s - wake_ahead_s)
        else:
            spin_deadline = perf_counter() + remaining_s
            while perf_counter() < spin_deadline:
                check_cancelled(cancel_event)
            check_cancelled(cancel_event)
            return


@dataclass(frozen=True)
class ExperimentStep:
    """同一步先改指定频率，再按 ACTIVE/IDLE 调幅；STOP 关闭全部输出。"""

    at_s: float
    state: str
    active_channels: tuple[str, ...] = ()
    frequencies_hz: dict[str, float] = field(default_factory=dict)

    def changes_from(self, current: dict[str, str]) -> list[tuple[str, str]]:
        """仅返回需要调幅的通道，先降至 IDLE 再升至 ACTIVE；STOP 另行关断。"""
        changes = []
        if self.state == "STOP":
            return changes
        for state in ("IDLE", "ACTIVE"):
            for name in CHANNEL_NAMES:
                target = "ACTIVE" if name in self.active_channels else "IDLE"
                if target == state and current[name] != target:
                    changes.append((name, target))
        return changes


@dataclass(frozen=True)
class ExperimentPlan:
    """四路进入 IDLE 后开始计时；duration_s 时必须进入 STOP。"""

    duration_s: float
    settings: dict[str, ChannelSettings]
    steps: tuple[ExperimentStep, ...]

    def validate(self, config: dict) -> None:
        """在任何设备写入之前检查整张时间表及现有驱动的参数限制。"""
        _validate_seconds(self.duration_s, "duration_s")
        if self.duration_s <= 0:
            raise ValueError("duration_s 必须大于 0")
        if set(self.settings) != set(CHANNEL_NAMES):
            raise ValueError("settings 必须恰好包含 PZT1、PZT2、PZT3、PZT4")
        for name in CHANNEL_NAMES:
            settings = self.settings[name]
            settings.validate(config["channels"][name])
            # 只对连续正弦波调幅，不隐式发出扫频或 burst 触发。
            if settings.mode != "continuous":
                raise ValueError(f"{name}: 时间表目前只支持 continuous 模式")
            if settings.amplitude_vpp <= IDLE_AMPLITUDE_VPP:
                raise ValueError(f"{name}: ACTIVE 幅值必须大于 IDLE 的 0.001 Vpp")
            replace(settings, amplitude_vpp=IDLE_AMPLITUDE_VPP).validate(
                config["channels"][name]
            )
        if not self.steps or self.steps[0].at_s != 0 or self.steps[0].state != "IDLE":
            raise ValueError("第一步必须从 0 秒开始，状态为 IDLE")
        if self.steps[-1].state != "STOP" or self.steps[-1].at_s != self.duration_s:
            raise ValueError("最后一步必须在 duration_s 时进入 STOP")
        previous_s = -1.0
        for step in self.steps:
            _validate_seconds(step.at_s, "at_s")
            if not previous_s < step.at_s <= self.duration_s:
                raise ValueError("步骤时间必须严格递增，且不超过 duration_s")
            if step.state not in ("IDLE", "ACTIVE", "STOP"):
                raise ValueError("state 只允许 IDLE、ACTIVE、STOP")
            if step.state == "STOP" and step.at_s != self.duration_s:
                raise ValueError("STOP 只能是时间表的最后一步")
            if (step.state == "ACTIVE") != bool(step.active_channels):
                raise ValueError(
                    "ACTIVE 必须指定 active_channels；IDLE/STOP 不允许指定"
                )
            if len(set(step.active_channels)) != len(step.active_channels):
                raise ValueError("active_channels 不能包含重复通道")
            if not set(step.active_channels).issubset(CHANNEL_NAMES):
                raise ValueError("active_channels 只允许 PZT1、PZT2、PZT3、PZT4")
            if not isinstance(step.frequencies_hz, dict) or not set(
                step.frequencies_hz
            ).issubset(CHANNEL_NAMES):
                raise ValueError("frequencies_hz 必须为通道名称到频率的对象")
            if step.state == "STOP" and step.frequencies_hz:
                raise ValueError("STOP 只关闭输出，不允许同时调频")
            for name, frequency in step.frequencies_hz.items():
                if isinstance(frequency, bool) or not isinstance(
                    frequency, (int, float)
                ):
                    raise TypeError("frequencies_hz 中的频率必须为数值")
                replace(self.settings[name], frequency_hz=frequency).validate(
                    config["channels"][name]
                )
            previous_s = step.at_s


def _validate_seconds(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是秒数，不能是字符串或布尔值")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} 必须是有限的非负秒数")


def load_plan(path: str | Path, config: dict) -> ExperimentPlan:
    """读取明确列出四路频率和输入电压的 JSON；拒绝误拼字段。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or set(raw) != {"duration_s", "settings", "steps"}:
        raise ValueError("计划必须包含且仅包含 duration_s、settings、steps")
    if not isinstance(raw["settings"], dict) or not isinstance(raw["steps"], list):
        raise TypeError("settings 必须为对象，steps 必须为列表")
    settings = {}
    for name, values in raw["settings"].items():
        if not isinstance(values, dict) or not {"frequency_hz", "amplitude_vpp"} <= set(
            values
        ):
            raise ValueError(f"{name}: 必须明确填写 frequency_hz 和 amplitude_vpp")
        try:
            settings[name] = ChannelSettings(**values)
        except TypeError as exc:
            raise ValueError(f"{name}: 参数字段不正确：{exc}") from exc
    steps = []
    for step in raw["steps"]:
        if (
            not isinstance(step, dict)
            or not {"at_s", "state"} <= set(step)
            or not set(step) <= {"at_s", "state", "active_channels", "frequencies_hz"}
        ):
            raise ValueError(
                "每一步需要 at_s、state，可选 active_channels、frequencies_hz；"
                "旧 outputs_on 格式不再执行"
            )
        active_channels = step.get("active_channels", [])
        if not isinstance(active_channels, list) or not all(
            isinstance(name, str) for name in active_channels
        ):
            raise ValueError("active_channels 必须为通道名称列表")
        steps.append(
            ExperimentStep(
                step["at_s"],
                step["state"],
                tuple(active_channels),
                step.get("frequencies_hz", {}),
            )
        )
    plan = ExperimentPlan(raw["duration_s"], settings, tuple(steps))
    plan.validate(config)
    return plan


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_event(log: TextIO, event: dict) -> None:
    """逐条刷入文件；异常退出前已经写出的行仍可单独读取。"""
    log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    log.flush()


class ExperimentRunner:
    """
    独占使用一个已连接的 Rig，同步执行一次计划。
    rig 可以理解成程序里的仪器遥控器：程序通过它连接仪器、设置参数、打开或关闭输出

    run() 会阻塞调用线程，应从独立命令行或专用工作线程调用。
    cancel() 可从另一线程调用；已经开始的 USB 调用仍需返回或超时。
    正常运行先可取消休眠，最后循环等到固定计划时刻。
    显式传入 clock/wait 时，保留原有软件测试等待接口，不用于实机计时。
    """

    def __init__(
        self,
        rig: Rig,
        *,
        clock: Callable[[], float] = monotonic,
        wait: Callable[[float], object] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        """把需要的东西准备好"""
        self.rig = rig
        self._clock = clock
        self._cancel = cancel_event if cancel_event is not None else threading.Event()
        self._wait = wait if wait is not None else self._cancel.wait
        self._use_hybrid_wait = clock is monotonic and wait is None
        self._run_lock = threading.Lock()
        self._has_run = False
        self._start_s: float | None = None

    def cancel(self) -> None:
        """唤醒时间表等待，并要求现有驱动停止后续普通写入。"""
        self._cancel.set()
        self.rig.request_stop()

    def _check_cancelled(self) -> None:
        """检查是否有人要求停止"""
        check_cancelled(self._cancel)

    def _elapsed(self, now_s: float) -> float | None:
        """计算实验已经进行了多少秒"""
        return None if self._start_s is None else now_s - self._start_s

    def _timeline_origin(self) -> float:
        """默认准备完成即开始；联合录像入口可等待共同的视频起点。"""
        return self._clock()

    def _output_timing(self, action: str) -> dict:
        """
        取出某次设备操作的详细时间记录。
        例如发送调频命令、收到回复的时间，供后面记录使用
        保留原驱动接口；使用计时驱动时附加实际命令和回复时间。
        """
        if not isinstance(self.rig, TimedRig) or self._start_s is None:
            return {}
        
        if action == "set_output":
            timing = self.rig.last_output_timing
            key, reply_name = "output_timing", "state"
        elif action == "send_frequency":
            timing = self.rig.last_frequency_timing
            key, reply_name = "frequency_timing", "frequency"
        elif action in ("apply_live", "send_amplitude"):
            timing = self.rig.last_amplitude_timing
            key, reply_name = "amplitude_timing", "amplitude"
        else:
            return {}
        
        if timing is None:
            return {}
        return {key: timing.relative_to(self._start_s, reply_name=reply_name)}

    def _wait_until(self, planned_s: float) -> None:
        """
        等到时间表指定的时刻。 例如传入 5，就是等到实验开始后的第 5 秒
        """
        # 始终相对同一个起点等待，避免每一步的 USB 耗时累加到后续时间表。
        if self._use_hybrid_wait:
            wait_until(self._start_s + planned_s, self._cancel)
            return
        
        # 显式注入的软件测试时钟沿用自身等待函数，不混入真实循环时钟。
        while True:
            self._check_cancelled()
            remaining_s = self._start_s + planned_s - self._clock()
            if remaining_s <= 0:
                return
            self._wait(remaining_s)

    def _call(
        self,
        log: TextIO,
        action: str,
        operation: Callable,
        *args,
        planned_s: float | None = None,
        latest_start_s: float | None = None,
        state_requested: str | None = None,
    ):
        """
        执行一个设备操作，并记录过程
        """
        details = {"action": action, "planned_s": planned_s}
        if state_requested is not None:
            details["state_requested"] = state_requested

        if action == "set_output":
            details.update(channel=args[0], output_requested=args[1])
        elif action == "apply_live":
            details.update(
                channel=args[0], amplitude_requested_vpp=args[1].amplitude_vpp
            )
        elif action == "send_amplitude":
            details.update(channel=args[0], amplitude_requested_vpp=args[1])
        elif action == "send_frequency":
            details.update(channel=args[0], frequency_requested_hz=args[1])

        _write_event(log, {"event": "action_requested", "utc": _utc_now(), **details})
        started_s = self._clock()
        timing = {
            "started_utc": _utc_now(),
            "actual_start_s": self._elapsed(started_s),
            "lateness_s": None
            if planned_s is None
            else started_s - self._start_s - planned_s,
        }

        try:
            # 日志写入也可能耗时；写完后再次检查，不能补发过期的开启操作。
            if action == "set_output" and isinstance(self.rig, TimedRig):
                self.rig.last_output_timing = None
            elif action in ("apply_live", "send_amplitude") and isinstance(
                self.rig, TimedRig
            ):
                self.rig.last_amplitude_timing = None
            elif action == "send_frequency" and isinstance(self.rig, TimedRig):
                self.rig.last_frequency_timing = None

            if action != "check_state_after":
                self._check_cancelled()

            if (
                latest_start_s is not None
                and self._elapsed(started_s) >= latest_start_s
            ):
                raise TimeoutError(f"{args[0]}: 当前时段已过，不再补发参数命令")
            
            result = operation(*args)
            if action == "all_off" and result:
                raise AFGError("未能确认全部关闭：" + "; ".join(result))
            if action == "set_output" and result is not args[1]:
                raise AFGError(f"{args[0]}: 开关读回与请求不一致")
            if action == "apply_live" and result["output"] is not True:
                raise AFGError(f"{args[0]}: 调幅后未保持输出开启")
            if (
                action in ("check_state_before", "check_state_after")
                and not result["passed"]
            ):
                raise RoundUnreliable(result)
        except BaseException as exc:
            finished_s = self._clock()
            try:
                _write_event(
                    log,
                    {
                        "event": "action_finished",
                        **details,
                        **timing,
                        **self._output_timing(action),
                        "actual_finish_s": self._elapsed(finished_s),
                        "duration_s": finished_s - started_s,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "error_notes": getattr(exc, "__notes__", []),
                        **(
                            {"audit": exc.report}
                            if isinstance(exc, RoundUnreliable)
                            else {}
                        ),
                    },
                )
            except (OSError, ValueError, TypeError) as log_error:
                exc.add_note(f"同时无法写入动作失败记录：{log_error}")
            raise
        finished_s = self._clock()
        _write_event(
            log,
            {
                "event": "action_finished",
                **details,
                **timing,
                **self._output_timing(action),
                "actual_finish_s": self._elapsed(finished_s),
                "duration_s": finished_s - started_s,
                "status": "sent"
                if action in ("send_amplitude", "send_frequency")
                else "ok",
                "result": result,
            },
        )
        return result

    def run(self, plan: ExperimentPlan, output_dir: str | Path) -> dict:
        """
        运行、保存记录并返回摘要；出错或取消时，先收尾再重新抛出异常。

        output_dir 必须不存在，避免覆盖既有实验。参数无效、目录已存在或
        尚未连接四路时不进行设备写入。每个实例只执行一次。
        """
        with self._run_lock:
            if self._has_run:
                raise RuntimeError("此执行器已经使用过，请为新实验创建新实例")
            self._has_run = True
        # 固定本次参数，避免调用者随后修改原字典影响正在运行的实验。
        plan = ExperimentPlan(
            plan.duration_s,
            dict(plan.settings),
            tuple(
                replace(step, frequencies_hz=dict(step.frequencies_hz))
                for step in plan.steps
            ),
        )
        plan.validate(self.rig.config)
        if set(self.rig.sessions) != set(self.rig.config["devices"]):
            raise AFGError("实验需要先连接 A、B 两台 AFG")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        with (output_dir / "plan.json").open("x", encoding="utf-8") as file:
            json.dump(
                {"plan": asdict(plan), "afg_config": self.rig.config},
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        with (output_dir / "events.jsonl").open("x", encoding="utf-8") as log:
            return self._run_plan(plan, output_dir, log)

    def _run_plan(self, plan: ExperimentPlan, output_dir: Path, log: TextIO) -> dict:
        """
        实际完成一次实验
        """
        control_mode = "amplitude_write_only"
        if any(step.frequencies_hz for step in plan.steps):
            control_mode = (
                "frequency_amplitude_write_only"
                if any(step.state == "ACTIVE" for step in plan.steps)
                else "frequency_write_only"
            )
        summary = {
            "status": "completed",
            "created_utc": _utc_now(),
            "planned_duration_s": plan.duration_s,
            "timeline_started_utc": None,
            "output_dir": str(output_dir),
            "error": None,
            "timing_basis": "software_calls_and_driver_readback_not_physical_voltage",
            "control_mode": control_mode,
            "parameter_order": ["frequency", "amplitude"],
            "idle_amplitude_vpp": IDLE_AMPLITUDE_VPP,
            "data_reliable": False,
            "per_command_confirmation": False,
            "schedule_wait_method": "Event.wait then perf_counter spin"
            if self._use_hybrid_wait
            else "custom clock/wait",
            "schedule_wake_ahead_ms": WAKE_AHEAD_S * 1000
            if self._use_hybrid_wait
            else None,
        }
        idle_settings = {
            name: replace(setting, amplitude_vpp=IDLE_AMPLITUDE_VPP)
            for name, setting in plan.settings.items()
        }
        expected_settings = dict(idle_settings)
        # 时间表内先在内存中记日志；STOP 后落盘，避免文件刷新推迟调幅命令。
        timeline_log = StringIO()
        problem: BaseException | None = None
        try:
            self._check_cancelled()
            self._call(log, "all_off", self.rig.all_off, state_requested="STOP")
            self._check_cancelled()
            self._call(log, "apply", self.rig.apply, idle_settings)
            summary["precheck"] = self._call(
                log, "check_state_before", self.rig.check_state, idle_settings, "before"
            )
            for name in CHANNEL_NAMES:
                self._call(
                    log,
                    "set_output",
                    self.rig.set_output,
                    name,
                    True,
                    state_requested="IDLE",
                )
            current_states = dict.fromkeys(CHANNEL_NAMES, "IDLE")
            self._check_cancelled()
            self._start_s = self._timeline_origin()
            summary["timeline_started_utc"] = _utc_now()
            _write_event(
                log,
                {"event": "timeline_started", "utc": summary["timeline_started_utc"]},
            )
            for index, step in enumerate(plan.steps):
                self._wait_until(step.at_s)
                end_s = (
                    plan.steps[index + 1].at_s
                    if index + 1 < len(plan.steps)
                    else plan.duration_s
                )
                _write_event(
                    timeline_log,
                    {
                        "event": "step_started",
                        "planned_s": step.at_s,
                        "actual_s": self._elapsed(self._clock()),
                        "state": step.state,
                        "active_channels": step.active_channels,
                        "frequencies_hz": step.frequencies_hz,
                    },
                )
                if step.state == "STOP":
                    break  # finally 中统一 all_off，取消或异常也走同一路径。
                # 两种参数属于同一个计划时刻；连续发出，不在两条之间加等待或查询。
                for name in CHANNEL_NAMES:
                    if name not in step.frequencies_hz:
                        continue
                    frequency = step.frequencies_hz[name]
                    if frequency == expected_settings[name].frequency_hz:
                        continue
                    self._call(
                        timeline_log,
                        "send_frequency",
                        self.rig.send_frequency,
                        name,
                        frequency,
                        planned_s=step.at_s,
                        latest_start_s=end_s,
                        state_requested=step.state,
                    )
                    expected_settings[name] = replace(
                        expected_settings[name], frequency_hz=frequency
                    )
                for name, target_state in step.changes_from(current_states):
                    target = (
                        plan.settings[name]
                        if target_state == "ACTIVE"
                        else idle_settings[name]
                    )
                    self._call(
                        timeline_log,
                        "send_amplitude",
                        self.rig.send_amplitude,
                        name,
                        target.amplitude_vpp,
                        planned_s=step.at_s,
                        latest_start_s=end_s if target_state == "ACTIVE" else None,
                        state_requested=target_state,
                    )
                    current_states[name] = target_state
                    expected_settings[name] = replace(
                        expected_settings[name], amplitude_vpp=target.amplitude_vpp
                    )
                if (step.state == "ACTIVE" or step.frequencies_hz) and self._elapsed(
                    self._clock()
                ) >= end_s:
                    raise TimeoutError("本段操作超过下一边界，停止实验并关闭全部输出")
        except BaseException as exc:  # noqa: BLE001 - 保存后会重抛，中断也必须先关断。
            problem = exc
            summary["status"] = (
                "cancelled"
                if isinstance(exc, (ExperimentCancelled, KeyboardInterrupt))
                else "unreliable"
                if isinstance(exc, RoundUnreliable)
                else "failed"
            )
            summary["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, RoundUnreliable):
                summary["validation_failure"] = exc.report
        finally:
            # 收尾不能依赖日志正常写入；磁盘出错时也必须先尝试关闭仪器。
            shutdown_started_s = self._clock()
            try:
                failures = self.rig.all_off()
            except BaseException as exc:  # noqa: BLE001 - 关断被中断时不能报告已关闭。
                failures = [f"{type(exc).__name__}: {exc}"]
            shutdown_finished_s = self._clock()
            summary.update(
                finished_utc=_utc_now(),
                elapsed_s=self._elapsed(shutdown_finished_s),
                all_outputs_off_confirmed=not failures,
                shutdown_errors=failures,
                final_state="STOP" if not failures else "UNKNOWN",
            )
            if failures:
                summary["status"] = "failed"
                if problem is None:
                    problem = AFGError("收尾时未能确认全部关闭：" + "; ".join(failures))
                    summary["error"] = str(problem)
            try:
                log.write(timeline_log.getvalue())
                log.flush()
                _write_event(
                    log,
                    {
                        "event": "shutdown_finished",
                        "action": "all_off",
                        "state_requested": "STOP",
                        "planned_s": plan.duration_s
                        if summary["status"] == "completed"
                        else None,
                        "actual_start_s": self._elapsed(shutdown_started_s),
                        "actual_finish_s": self._elapsed(shutdown_finished_s),
                        "duration_s": shutdown_finished_s - shutdown_started_s,
                        "errors": failures,
                    },
                )
            except (OSError, ValueError, TypeError) as exc:
                summary["status"] = "failed"
                summary["logging_error"] = f"{type(exc).__name__}: {exc}"
                if problem is None:
                    problem = exc
                    summary["error"] = summary["logging_error"]
            if self._start_s is not None and not failures:
                try:
                    summary["postcheck"] = self._call(
                        log,
                        "check_state_after",
                        self.rig.check_state,
                        expected_settings,
                        "after",
                    )
                except BaseException as exc:  # noqa: BLE001 - 已关断；保留核查中断并在保存摘要后重抛。
                    if isinstance(exc, RoundUnreliable):
                        summary["validation_failure"] = exc.report
                    if problem is None:
                        problem = exc
                        summary["status"] = (
                            "unreliable"
                            if isinstance(exc, RoundUnreliable)
                            else "failed"
                        )
                        summary["error"] = f"{type(exc).__name__}: {exc}"
                    else:
                        problem.add_note(
                            f"轮末核查也未通过：{type(exc).__name__}: {exc}"
                        )
            else:
                summary["postcheck_skipped_reason"] = "未开始时间表或关断未确认"
            summary["data_reliable"] = (
                summary["status"] == "completed"
                and summary.get("precheck", {}).get("passed", False)
                and summary.get("postcheck", {}).get("passed", False)
                and not failures
            )
            summary["finished_utc"] = _utc_now()
            timeline_log.close()
            if problem is not None:
                summary["error_notes"] = getattr(problem, "__notes__", [])
                summary["error_traceback"] = "".join(
                    traceback.format_exception(problem)
                )
            with (output_dir / "summary.json").open("x", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        if problem is not None:
            raise problem
        return summary


def run_experiments(
    rig: TimedRig,
    plan: ExperimentPlan,
    output_dir: Path,
    repeat: int,
    cancel_event: threading.Event,
) -> dict:
    """
    同一连接连续运行相同计划；每次重新准备，任何失败或取消都停止整批。
    按照同一份时间表，把实验依次做指定次数，最后把这些实验的记录汇总起来
    """
    if repeat < 1:
        raise ValueError("重复次数必须至少为 1")
    if repeat > 1:
        output_dir.mkdir(parents=True, exist_ok=False)

    run_dirs = []
    problem: BaseException | None = None

    try:
        with rig:
            for index in range(1, repeat + 1):
                runner = ExperimentRunner(rig, cancel_event=cancel_event)       # 为这一次实验准备执行工具 runner
                runner._check_cancelled()

                run_dir = output_dir / f"run_{index:02d}" if repeat > 1 else output_dir
                run_dirs.append(run_dir)

                print(f"开始第 {index}/{repeat} 次实验：{run_dir}", flush=True)
                result = runner.run(plan, run_dir)
                print(
                    f"第 {index}/{repeat} 次完成；全部关闭确认："
                    f"{result['all_outputs_off_confirmed']}",
                    flush=True,
                )
    except BaseException as exc:
        problem = exc
        raise
    finally:
        # 汇总安排在关断与连接关闭之后，避免写 CSV 干扰开关时间。
        if output_dir.exists():
            try:
                report = write_timing_report(output_dir, run_dirs, repeat)
            except Exception as exc:
                # 保留最先发生的设备或取消异常。
                if problem is None:
                    raise
                problem.add_note(f"另外无法生成计时汇总：{exc}")
                print(f"计时汇总未完成：{exc}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    """
    检查时间表后直接运行真实实验；WSL 自动转交已有 Windows 环境。
    只控制 PZT，不录像
    """

    parser = argparse.ArgumentParser(
        description="四路 PZT 时间表执行器：检查计划后直接控制真实仪器"
    )

    # 要执行的实验时间表 json 的文件路径，例如 customer_workspace/configs/pzt1_pulse.json;
    # 它是必填参数，用来告诉程序本次采用哪份时间表。
    parser.add_argument("plan", type=Path, help="实验时间表 JSON 文件路径（必填）")

    # record device info, including:
    # 设备身份：用序列号识别 A、B 两台 AFG，避免因 USB 插接顺序变化而认错;
    # 各实际 PZT 与通道对应关系;
    # 参数上限：统一配置目前允许最高 0.8 Vpp、10 MHz，实际值由时间表指定;
    # 通信设置：驱动选择、通信超时和设备状态读取方式.
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="AFG 设备配置，默认 %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="必须不存在的结果目录，默认 records/年月日-时分秒（运行电脑的本地时间）",
    )
    parser.add_argument(
        "--windows-python",
        type=Path,
        help="WSL 使用的 Windows Python 路径；默认读取本机 windows_afg_python 配置",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="连续重复同一时间表的次数，默认 1"
    )
    parser.add_argument("--launcher-stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat 必须至少为 1")

    try:
        # 打开文件，检查实验安排有没有问题
        config = load_config(args.config)
        plan = load_plan(args.plan, config)

        # VISA 是控制信号发生器的通信标准，通常靠 Windows 上安装的 NI-VISA 等驱动，
        # 通过 USB、GPIB 或网口和设备通信。
        # 仪器驱动和硬件连接仍由 Windows 管理；WSL 负责启动并转交路径。
        # 判断：现在是不是在 Windows 中运行
        if os.name != "nt":
            # 如果从 WSL 启动，就会进入这一段
            # 找到 Windows Python，让它重新运行这个程序，然后等待它结束
            return launch_from_wsl(
                PROJECT_ROOT,
                args.plan,
                args.config,
                args.output_dir,
                args.windows_python,
                repeat=args.repeat,
            )

        # 因为已经是 Windows，就跳过这个 if，继续执行下面的代码
        output_dir = (
            args.output_dir
            or PROJECT_ROOT / "records" / datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        # 在连接前也检查目录，明确拒绝覆盖已有记录。
        if output_dir.exists():
            raise FileExistsError(f"结果目录已存在：{output_dir}")
        print(f"实验记录将保存到：{output_dir}", flush=True)

        rig = TimedRig(config)              # 后面用来操作仪器的工具
        cancel_event = threading.Event()

        def cancel_batch():
            # 提前写好“收到停止要求时怎么办”
            cancel_event.set()
            rig.request_stop()

        if args.launcher_stdin:
            threading.Thread(
                target=watch_launcher_input,
                args=(cancel_batch,),
                name="experiment-cancel",
                daemon=True,
            ).start()
        if cancel_event.is_set():
            raise ExperimentCancelled("实验已取消")

        # 真正开始做实验，并等待它完成
        # 使用这些设备，按照这份时间表，重复指定次数，把记录存到指定位置，同时留意有没有停止要求
        result = run_experiments(rig, plan, output_dir, args.repeat, cancel_event)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    except (KeyboardInterrupt, ExperimentCancelled):
        print("实验已取消；请检查输出目录中的关闭结果。")
        return 130

    except Exception as exc:  # noqa: BLE001 - 命令行统一报告包含第三方驱动在内的错误。
        print(f"实验未完成：{type(exc).__name__}: {exc}")
        traceback.print_exception(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
