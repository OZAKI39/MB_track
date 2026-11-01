"""真实 AFG 的关闭输出通信排查；不启用任何 PZT，不自动重试失败查询。

复用现有连接、参数设置和关断。原生读取仅额外保留 viRead 在出错时的
返回字节数及缓冲区；不会修改 VISA 错误处理器、驱动或系统配置。
"""

import argparse
import ctypes
import json
import math
import os
import threading
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

from customer_workspace.afg_timing import TimedRig
from customer_workspace.experiment_runner import IDLE_AMPLITUDE_VPP, load_plan
from customer_workspace.timing_benchmark import save_results, wait_until
from microbubble_experiment.afg import DEFAULT_CONFIG, AFGError, load_config


def read_with_evidence(session, row: dict, size: int) -> str:
    """沿用 PyVISA 的 viRead 签名，保留异常时仍有效的 retCount 和原始字节。

    普通 PyVISA 会先抛出错误，再离开局部缓冲区，调用方看不到部分数据。
    本函数不将错误返回的数据当作成功回复；任何异常照常重抛并终止本组。
    """
    from pyvisa import constants
    from pyvisa.ctwrapper.types import ViUInt32

    buffer = ctypes.create_string_buffer(size)
    returned = ViUInt32()
    try:
        status = session.visalib.lib.viRead(
            session.session, buffer, size, ctypes.byref(returned)
        )
        row["visa_read_status"] = int(status)
    except Exception as exc:
        row["visa_read_status"] = getattr(exc, "error_code", None)
        raise
    finally:
        row["returned_byte_count"] = returned.value
        row["returned_bytes_hex"] = buffer.raw[: returned.value].hex()
    if status == constants.StatusCode.success_max_count_read:
        raise AFGError("诊断标量回复未完整结束，不继续读取或重试")
    return buffer.raw[: returned.value].decode(session.encoding).strip()


def validate_options(args) -> None:
    """连接设备之前检查次数、间隔和缓冲区大小。"""
    if args.count < 2 or args.count % 2:
        raise ValueError("count 必须是至少为 2 的偶数")
    for key in ("interval_ms", "before_query_ms", "before_read_ms"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} 必须是非负有限数")
    if args.interval_ms <= 0 or not 32 <= args.read_size <= 20480:
        raise ValueError("interval-ms 必须为正，read-size 必须在 32 至 20480 之间")


def run_diagnostic(args) -> dict:
    """一次连接只测一种条件；失败停止，保持输出关闭后保存完整证据。"""
    validate_options(args)
    config = load_config(args.config)
    config["query_batch_mode"] = args.batch_mode
    plan = load_plan(args.plan, config)
    idle = {
        name: replace(setting, amplitude_vpp=IDLE_AMPLITUDE_VPP)
        for name, setting in plan.settings.items()
    }
    expected = dict(idle)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rig = TimedRig(config)
    rows = []
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "failed",
        "data_reliable": False,
        "protocol": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "outputs_enabled_by_diagnostic": False,
    }
    with (args.output_dir / "configuration.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {"protocol": summary["protocol"], "afg_config": config},
            stream,
            ensure_ascii=False,
            indent=2,
        )
    problem = None
    connected = False
    try:
        summary["devices"] = rig.connect()
        connected = True
        summary["initial_channels"] = {
            name: rig.read_channel(name) for name in config["channels"]
        }
        if any(state["output"] for state in summary["initial_channels"].values()):
            raise AFGError("诊断开始前必须四路关闭；发现输出开启，停止排查")
        if args.mode == "write_query":
            rig.apply(idle)
            summary["precheck"] = rig.check_state(idle, "before")
            if not summary["precheck"]["passed"]:
                raise AFGError("轮前核查未通过")
        wrapper, channel = rig._target(args.channel)
        session = wrapper._session
        session.chunk_size = args.read_size
        query = f"SOURce{channel['channel']}:VOLTage:LEVel:IMMediate:AMPLitude?"
        summary["session"] = {
            "resource": session.resource_name,
            "visa_library": str(session.visalib.library_path),
            "timeout_ms": session.timeout,
            "read_termination": session.read_termination,
            "write_termination": session.write_termination,
            "query_delay_s": session.query_delay,
        }
        cancel = threading.Event()
        started = monotonic()
        summary["timeline_started_utc"] = datetime.now(UTC).isoformat()
        for index in range(args.count):
            planned = index * args.interval_ms / 1000
            wait_until(started + planned, cancel)
            if monotonic() - started - planned > 1:
                raise TimeoutError("诊断落后计划超过 1 秒，停止补发")
            target = (
                plan.settings[args.channel] if index % 2 == 0 else idle[args.channel]
            )
            amplitude = (
                target.amplitude_vpp
                if args.mode == "write_query"
                else summary["initial_channels"][args.channel]["amplitude_vpp"]
            )
            row = {
                "index": index + 1,
                "planned_s": planned,
                "expected_amplitude_vpp": amplitude,
                "query": query,
            }
            rows.append(row)
            try:
                if args.mode == "write_query":
                    row["stage"] = "amplitude_write"
                    row["command"] = query[:-1] + f" {amplitude:g}Vpp"
                    row["command_sent_s"] = monotonic() - started
                    rig._write(wrapper, row["command"])
                    row["write_returned_s"] = monotonic() - started
                    expected[args.channel] = target
                if args.before_query_ms:
                    sleep(args.before_query_ms / 1000)
                row["stage"] = "query_write"
                row["query_sent_s"] = monotonic() - started
                session.write(query)
                row["query_write_returned_s"] = monotonic() - started
                if args.before_read_ms:
                    sleep(args.before_read_ms / 1000)
                row["stage"] = "read"
                row["read_started_s"] = monotonic() - started
                reply = (
                    read_with_evidence(session, row, args.read_size)
                    if args.reader == "native"
                    else session.read().strip()
                )
                row["reply_received_s"] = monotonic() - started
                row["reply"] = reply
                row["stage"] = "match"
                actual = float(reply)
                if not math.isfinite(actual) or not math.isclose(
                    actual, amplitude, rel_tol=1e-6, abs_tol=1e-9
                ):
                    raise AFGError(f"幅值回复不符：期望 {amplitude}，收到 {reply!r}")
                row["status"] = "readback_ok"
            except BaseException as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["error_code"] = getattr(exc, "error_code", None)
                raise
            finally:
                row["finished_s"] = monotonic() - started
        summary["status"] = "completed"
    except BaseException as exc:  # noqa: BLE001 - 中断也必须保存证据并关断。
        problem = exc
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["error_traceback"] = traceback.format_exc()
    finally:
        try:
            failures = rig.all_off() if connected else ["设备连接未完成"]
            summary["shutdown_errors"] = failures
            summary["all_outputs_off_confirmed"] = connected and not failures
            if connected and not failures:
                if args.mode == "write_query":
                    summary["postcheck"] = rig.check_state(expected, "after")
                    if not summary["postcheck"]["passed"]:
                        raise AFGError("轮末核查未通过")
                else:
                    summary["final_channels"] = {
                        name: rig.read_channel(name) for name in config["channels"]
                    }
                    for session in rig.sessions.values():
                        rig._errors(session)
                    summary["error_queue_replies"] = {
                        alias: list(session.error_replies)
                        for alias, session in rig.sessions.items()
                    }
        except BaseException as exc:  # noqa: BLE001 - 清理失败不能丢失最早的错误。
            summary["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            if problem is None:
                problem = exc
        try:
            rig.close()
        except BaseException as exc:  # noqa: BLE001 - 关闭失败也要保存部分记录。
            summary["close_error"] = f"{type(exc).__name__}: {exc}"
            summary["all_outputs_off_confirmed"] = False
            if problem is None:
                problem = exc
        summary["recorded_commands"] = len(rows)
        summary["successful_replies"] = sum(
            row.get("status") == "readback_ok" for row in rows
        )
        summary["finished_utc"] = datetime.now(UTC).isoformat()
        summary["data_reliable"] = (
            problem is None
            and summary["status"] == "completed"
            and summary.get("all_outputs_off_confirmed") is True
        )
        if not summary["data_reliable"]:
            summary["status"] = "failed"
        save_results(args.output_dir, rows, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="四路保持关闭的真实 AFG 通信诊断")
    parser.add_argument("--mode", choices=("query_only", "write_query"), required=True)
    parser.add_argument(
        "--channel", choices=("PZT1", "PZT2", "PZT3", "PZT4"), default="PZT1"
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--interval-ms", type=float, default=50)
    parser.add_argument("--before-query-ms", type=float, default=0)
    parser.add_argument("--before-read-ms", type=float, default=0)
    parser.add_argument("--read-size", type=int, default=20480)
    parser.add_argument("--reader", choices=("pyvisa", "native"), default="native")
    parser.add_argument(
        "--batch-mode", choices=("legacy_lines", "sequential"), default="legacy_lines"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="AFG 设备配置，默认 %(default)s",
    )
    parser.add_argument(
        "--plan", type=Path, default=Path("customer_workspace/configs/pzt1_pulse.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error(
            "在 WSL 中使用现有 Windows python.exe -m customer_workspace.visa_diagnostics"
        )
    result = run_diagnostic(args)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "recorded_commands",
                    "successful_replies",
                    "error",
                    "all_outputs_off_confirmed",
                    "cleanup_error",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["data_reliable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
