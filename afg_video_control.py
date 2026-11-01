"""Windows AFG 联合录像入口：复用四路执行与稳定读回，独立负责关断。"""

import argparse
import json
import sys
import threading
import time
import traceback
from pathlib import Path

from customer_workspace.afg_timing import TimedRig, TimedSession
from customer_workspace.experiment_runner import ExperimentCancelled, load_plan
from customer_workspace.four_pzt_validation import ValidationRunner
from customer_workspace.streamcopy_recording import write_json
from customer_workspace.video_clock import probe_clock, read_health, read_ready
from customer_workspace.wsl_launcher import watch_launcher_input
from microbubble_experiment.afg import Rig, load_config


class RecordedSession(TimedSession):
    """只观察真实 VISA 调用，不增加命令；STOP 的每路写入和读回也单独留档。"""

    def __init__(self, session, alias, records):
        super().__init__(session)
        self.alias, self.records = alias, records

    def write(self, command):
        row = {
            "event": "write",
            "device": self.alias,
            "command": command,
            "started_windows_s": time.monotonic(),
        }
        self.records.append(row)
        try:
            result = super().write(command)
            row["status"] = "returned"
            return result
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["finished_windows_s"] = time.monotonic()

    def query(self, command):
        row = {
            "event": "query",
            "device": self.alias,
            "command": command,
            "started_windows_s": time.monotonic(),
        }
        self.records.append(row)
        try:
            result = super().query(command)
            row["reply"] = result.strip()
            return result
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["finished_windows_s"] = time.monotonic()

    def _record_reply(self, command, result, received):
        super()._record_reply(command, result, received)
        self.records.append(
            {
                "event": "reply",
                "device": self.alias,
                "command": command,
                "reply": result.strip(),
                "received_windows_s": received,
            }
        )


class RecordedRig(TimedRig):
    def __init__(self, config):
        super().__init__(config)
        self.communication = []

    def connect(self):
        identities = super().connect()
        for alias, session in self.sessions.items():
            self.sessions[alias] = RecordedSession(
                session._session, alias, self.communication
            )
        return identities


def wait_file(path, cancelled, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cancelled.is_set():
            raise ExperimentCancelled("等待录像时被取消")
        value = read_ready(path)
        if value is not None:
            return value
        cancelled.wait(0.02)
    raise TimeoutError(f"等待文件超时：{path.name}")


class LinkedRunner(ValidationRunner):
    """每段继续用原验收读回；以首个正式录像帧的 Windows 接收时间为零点。"""

    def __init__(self, rig, plan, cancel, output, experiment_id):
        super().__init__(rig, plan, cancel)
        self.output, self.experiment_id = output, experiment_id
        self.abort_reason = None
        self.origin = None

    def _timeline_origin(self):
        write_json(
            self.output / "afg_ready.json",
            {
                "experiment_id": self.experiment_id,
                "ready_windows_s": time.monotonic(),
                "idle_vpp": 0.001,
            },
        )
        try:
            frame = wait_file(self.output / "recording_started.json", self._cancel)
        except ExperimentCancelled:
            if self.abort_reason:
                raise RuntimeError(self.abort_reason) from None
            raise
        origin = frame["received_windows_mono"]
        if not 0 <= time.monotonic() - origin < 1:
            raise RuntimeError("录像起点过旧或不属于同一时钟，禁止启动动作")
        self.origin = {
            "experiment_id": self.experiment_id,
            "origin_windows_s": origin,
            "first_frame_seq": frame["seq"],
            "first_frame_pts_ns": frame["pts_ns"],
            "observed_windows_s": time.monotonic(),
        }
        write_json(self.output / "afg_origin.json", self.origin)
        return origin

    def _check_cancelled(self):
        if self.abort_reason:
            raise RuntimeError(self.abort_reason)
        super()._check_cancelled()

    def _wait_until(self, planned_s):
        try:
            super()._wait_until(planned_s)
        except ExperimentCancelled:
            if self.abort_reason:
                raise RuntimeError(self.abort_reason) from None
            raise


def guardian(runner, camera, mode, stop_event, requests):
    """相机停止到达或 WSL 管道取消时，唤醒原执行器；不从守护线程争用 VISA。"""
    while not stop_event.wait(0.05):
        try:
            health = read_health(camera, runner.experiment_id)
        except (OSError, ValueError, KeyError) as exc:
            runner.abort_reason = f"无法读取相机实时状态：{type(exc).__name__}: {exc}"
            requests.append(
                {
                    "kind": "camera_failure",
                    "reason": runner.abort_reason,
                    "windows_s": time.monotonic(),
                }
            )
            runner.cancel()
            return
        now = time.monotonic()
        if health is None or health.get("experiment_id") != runner.experiment_id:
            reason = "相机状态记录缺失或实验身份不符"
        elif now - health["windows_s"] > 3 or health["errors"]:
            reason = "相机心跳过期或报告错误"
        elif (
            health["last_frame_windows_s"] is None
            or now - health["last_frame_windows_s"] > 3
        ):
            reason = "相机连续超过3秒没有新视频帧"
        else:
            reason = None
        if reason:
            runner.abort_reason = reason
            requests.append(
                {"kind": "camera_failure", "reason": reason, "windows_s": now}
            )
            runner.cancel()
            return
        if (
            runner._start_s is not None
            and mode != "normal"
            and now - runner._start_s >= 4.5
        ):
            requests.append({"kind": mode, "planned_s": 4.5, "windows_s": now})
            if mode == "fault":
                runner.abort_reason = "验收主动触发程序异常；没有向AFG发送非法命令"
            runner.cancel()
            return


def run(args):
    output = args.output
    session = wait_file(output / "link_session.json", threading.Event())
    cancel = threading.Event()
    rig = RecordedRig(load_config(args.config))
    plan = load_plan(args.plan, rig.config)
    runner = LinkedRunner(rig, plan, cancel, output, session["experiment_id"])
    threading.Thread(
        target=watch_launcher_input, args=(runner.cancel,), daemon=True
    ).start()
    stop_guard = threading.Event()
    guard = None
    camera = None
    summary = {
        "experiment_id": session["experiment_id"],
        "validation_mode": args.mode,
        "error": None,
        "checks": [],
        "requests": [],
        "postflight": {"all_outputs_off": False},
    }
    try:
        camera = wait_file(output / "camera_clock.json", cancel)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if cancel.is_set():
                raise ExperimentCancelled("等待相机画面时被取消")
            health = read_health(camera, session["experiment_id"])
            if health.get("last_frame_windows_s") is not None:
                break
            cancel.wait(0.05)
        else:
            raise TimeoutError("相机尚未收到画面，不准备 AFG 输出")
        summary["clock_before"] = probe_clock(camera, session["experiment_id"])
        guard = threading.Thread(
            target=guardian,
            args=(runner, camera, args.mode, stop_guard, summary["requests"]),
            daemon=True,
        )
        guard.start()
        with rig:
            summary["devices"] = dict(rig.identities)
            runner.run(plan, output / "afg_experiment")
    except BaseException as exc:  # noqa: BLE001 -- 真实中断必须留档，原 runner 已先执行关断。
        summary["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        stop_guard.set()
        if guard:
            guard.join(2)
        # 独立新连接只读复核；如果读到开启，立即再次尝试关断，不能仅写报告。
        try:
            with Rig(rig.config) as observer:
                summary["postflight"]["devices"] = dict(observer.identities)
                channels = observer.status()
                summary["postflight"]["initial_channels"] = channels
                if any(row["output"] for row in channels.values()):
                    summary["postflight"]["recovery_errors"] = observer.all_off()
                    channels = observer.status()
                summary["postflight"]["channels"] = channels
                summary["postflight"]["all_outputs_off"] = len(channels) == 4 and all(
                    not row["output"] for row in channels.values()
                )
                summary["postflight"]["confirmed_windows_s"] = time.monotonic()
            summary["clock_after"] = probe_clock(camera, session["experiment_id"])
        except Exception as exc:  # noqa: BLE001 -- 即使关断复查失败，也必须保留动作证据。
            summary["postflight"]["error"] = f"{type(exc).__name__}: {exc}"
        summary["checks"] = runner.checks
        summary["origin"] = runner.origin
        summary["communication"] = rig.communication
        experiment = read_ready(output / "afg_experiment/summary.json") or {}
        expected = {"normal": "completed", "cancel": "cancelled", "fault": "failed"}[
            args.mode
        ]
        summary["expected_status"] = expected
        summary["actual_status"] = experiment.get("status")
        summary["passed"] = (
            experiment.get("status") == expected
            and experiment.get("all_outputs_off_confirmed", False)
            and experiment.get("precheck", {}).get("passed", False)
            and experiment.get("postcheck", {}).get("passed", False)
            and summary["postflight"]["all_outputs_off"]
            and "error" not in summary["postflight"]
            and "clock_before" in summary
            and "clock_after" in summary
            and bool(runner.checks)
            and all(row["passed"] for row in runner.checks)
        )
        summary["finished_windows_s"] = time.monotonic()
        write_json(output / "afg_link_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("normal", "cancel", "fault"), default="normal"
    )
    result = run(parser.parse_args())
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "status": result["actual_status"],
                "outputs_off": result["postflight"]["all_outputs_off"],
            },
            ensure_ascii=False,
        )
    )
    sys.exit(0 if result["passed"] else 1)
