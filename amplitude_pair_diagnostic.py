"""输出始终关闭的两次调幅诊断；不改变正式实验流程。

T 从第一次写入开始计时，纯 perf_counter 循环等待，不调用 wait_until。
两次写入之间不查询；第二次写入返回至少 100 ms 后，各查询幅值和错误一次。
"""

import argparse
import csv
import json
import math
import os
import threading
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter, sleep
from zoneinfo import ZoneInfo

from customer_workspace.afg_timing import TimedRig
from customer_workspace.experiment_runner import (
    CHANNEL_NAMES,
    PROJECT_ROOT,
    ExperimentCancelled,
    load_plan,
)
from customer_workspace.timing_benchmark import save_results
from customer_workspace.visa_diagnostics import read_with_evidence
from customer_workspace.wsl_launcher import launch_from_wsl, watch_launcher_input
from microbubble_experiment.afg import DEFAULT_CONFIG, AFGError, Rig, load_config

DIRECTIONS = {"up": (0.001, 0.1), "down": (0.1, 0.001)}
RESULTS = ("passed", "second_lost", "abnormal", "query_failed")


def check_cancelled(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise ExperimentCancelled("两次调幅诊断已取消")


def spin_until(deadline_s: float, cancel: threading.Event) -> None:
    """只检查 perf_counter 和取消标志；此等待故意占用 CPU，不主动休眠。"""
    while True:
        check_cancelled(cancel)
        if perf_counter() >= deadline_s:
            return


def classify_readback(value, x: float, y: float, query_failed: bool) -> str:
    """查询失败优先；不将零字节、部分回复或缺失数据判为第二条丢失。"""
    if query_failed:
        return "query_failed"
    if value is None or not math.isfinite(value):
        return "abnormal"
    if math.isclose(value, y, rel_tol=1e-6, abs_tol=1e-9):
        return "passed"
    if math.isclose(value, x, rel_tol=1e-6, abs_tol=1e-9):
        return "second_lost"
    return "abnormal"


def summarize_rounds(rows: list[dict]) -> dict:
    """四类结果互斥；错误队列非零另外计数，不改写幅值判断。"""
    counts = {result: 0 for result in RESULTS}
    for index, row in enumerate(rows, 1):
        if row["index"] != index or row["result"] not in counts:
            raise ValueError("轮次或判定结果无效")
        counts[row["result"]] += 1
    intervals = [
        r["actual_interval_ms"] for r in rows if r.get("actual_interval_ms") is not None
    ]
    if any(not math.isfinite(value) or value < 0 for value in intervals):
        raise ValueError("实际间隔无效")
    return {
        "total_rounds": len(rows),
        **counts,
        "error_queue_nonzero": sum(
            r.get("error_queue_code") not in (None, 0) for r in rows
        ),
        "zero_byte_query_failures": sum(
            r["result"] == "query_failed"
            and any(
                r.get(key, {}).get("returned_byte_count") == 0
                for key in ("amplitude_query", "error_query")
            )
            for r in rows
        ),
        "actual_interval_samples": len(intervals),
        "actual_interval_median_ms": median(intervals) if intervals else None,
        "actual_interval_min_ms": min(intervals) if intervals else None,
        "actual_interval_max_ms": max(intervals) if intervals else None,
    }


def query_with_evidence(session, command: str, origin_s: float) -> dict:
    """各发一次查询并原生读取，保存零字节及部分回复；不重试这次查询。"""
    record = {"command": command, "started_s": perf_counter() - origin_s}
    try:
        session.write(command)
        record["write_returned_s"] = perf_counter() - origin_s
        record["reply"] = read_with_evidence(session, record, 20480)
        if record["returned_byte_count"] == 0:
            raise AFGError("查询返回 0 字节")
    except Exception as exc:  # noqa: BLE001 - 错误作为诊断数据保留。
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_code"] = getattr(exc, "error_code", None)
    finally:
        record["finished_s"] = perf_counter() - origin_s
    return record


def clear_after_failed_query(session, row: dict, query_name: str) -> None:
    """失败不重试；清理通信缓冲，避免残留回复被误认为下一次查询的回复。"""
    recovery = {"after": query_name, "operation": "VISA clear"}
    row.setdefault("recovery", []).append(recovery)
    try:
        session.clear()
        recovery["succeeded"] = True
    except Exception as exc:
        recovery["succeeded"] = False
        recovery["error"] = f"{type(exc).__name__}: {exc}"
        raise AFGError("查询失败后无法清理通信缓冲，停止测试") from exc


def measure_pair(
    session, row: dict, channel_number: int, cancel: threading.Event
) -> None:
    """两次 write 紧邻所需循环；计时窗口中不打印、保存文件或查询。"""
    x, y = row["x_vpp"], row["y_vpp"]
    prefix = f"SOURce{channel_number}:VOLTage:LEVel:IMMediate:AMPLitude"
    command_x, command_y = f"{prefix} {x:g}Vpp", f"{prefix} {y:g}Vpp"
    row.update(command_x=command_x, command_y=command_y, result="abnormal")
    first_started = first_finished = second_started = second_finished = None
    try:
        check_cancelled(cancel)
        first_started = perf_counter()
        session.write(command_x)
        first_finished = perf_counter()
        spin_until(first_started + row["planned_interval_ms"] / 1000, cancel)
        second_started = perf_counter()
        session.write(command_y)
        second_finished = perf_counter()
        # 这里允许多等，不参与两条写入的目标间隔；测量实际值以确认至少 100 ms。
        settle_deadline = second_finished + 0.100
        while True:
            check_cancelled(cancel)
            remaining = settle_deadline - perf_counter()
            if remaining <= 0:
                break
            sleep(remaining)
        check_cancelled(cancel)
        amplitude = query_with_evidence(session, prefix + "?", first_started)
        row["amplitude_query"] = amplitude
        if "error" in amplitude:
            row["result"] = "query_failed"
            clear_after_failed_query(session, row, "amplitude")
        errors = query_with_evidence(session, "SYSTem:ERRor?", first_started)
        row["error_query"] = errors
        row["error_queue_reply"] = errors.get("reply")
        if "error" in errors:
            row["result"] = "query_failed"
            clear_after_failed_query(session, row, "error_queue")
        query_failed = "error" in amplitude or "error" in errors
        try:
            value = float(amplitude["reply"]) if "error" not in amplitude else None
        except (ValueError, KeyError):
            value = None
        row["readback_vpp"] = (
            value if value is not None and math.isfinite(value) else None
        )
        row["result"] = classify_readback(value, x, y, query_failed)
        if "error" not in errors:
            try:
                row["error_queue_code"] = int(errors["reply"].split(",", 1)[0])
            except (ValueError, KeyError):
                row["error_queue_parse_error"] = "错误队列回复无法解析"
                row["result"] = "abnormal" if not query_failed else "query_failed"
    except BaseException as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        row["first_write_started_counter_s"] = first_started
        row["first_write_returned_counter_s"] = first_finished
        row["second_write_started_counter_s"] = second_started
        row["second_write_returned_counter_s"] = second_finished
        row["actual_interval_ms"] = (
            None if second_started is None else (second_started - first_started) * 1000
        )
        row["first_write_ms"] = (
            None if first_finished is None else (first_finished - first_started) * 1000
        )
        row["second_write_ms"] = (
            None
            if second_finished is None
            else (second_finished - second_started) * 1000
        )
        query_start = row.get("amplitude_query", {}).get("started_s")
        row["settle_before_query_ms"] = (
            None
            if query_start is None
            else (query_start - (second_finished - first_started)) * 1000
        )


def run_group(
    config, plan, directory, channel, interval_ms, direction, count, loss_limit, cancel
):
    """单组独立连接、准备并核查；达到丢失上限结束，通信无法恢复时停止。"""
    directory.mkdir(parents=True, exist_ok=False)
    x, y = DIRECTIONS[direction]
    idle = {
        name: replace(setting, amplitude_vpp=0.001)
        for name, setting in plan.settings.items()
    }
    rig = TimedRig(config)
    rows = []
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "channel": channel,
        "interval_ms": interval_ms,
        "direction": direction,
        "x_vpp": x,
        "y_vpp": y,
        "requested_rounds": count,
        "loss_limit": loss_limit,
        "outputs_enabled_by_diagnostic": False,
        "wait_method": "perf_counter spin only",
        "interval_origin": "first write start",
        "settle_minimum_ms": 100,
    }
    problem = None
    connected = False
    try:
        check_cancelled(cancel)
        summary["devices"] = rig.connect()
        connected = True
        failures = rig.all_off()
        if failures:
            raise AFGError("准备关断未确认：" + "; ".join(failures))
        check_cancelled(cancel)
        rig.apply(idle)
        summary["precheck"] = rig.check_state(idle, "before")
        if not summary["precheck"]["passed"]:
            raise AFGError("准备核查未通过")
        wrapper, mapping = rig._target(channel)
        session = wrapper._session
        summary["visa_library"] = str(session.visalib.library_path)
        losses = 0
        for index in range(1, count + 1):
            check_cancelled(cancel)
            row = {
                "index": index,
                "direction": direction,
                "planned_interval_ms": interval_ms,
                "x_vpp": x,
                "y_vpp": y,
                "started_utc": datetime.now(UTC).isoformat(),
            }
            rows.append(row)
            measure_pair(session, row, mapping["channel"], cancel)
            losses += row["result"] == "second_lost"
            if index % 50 == 0 or losses >= loss_limit:
                print(
                    f"T={interval_ms:g} ms {direction}: {index}/{count}，第二条丢失 {losses}",
                    flush=True,
                )
            if losses >= loss_limit:
                summary["status"] = "loss_limit_reached"
                break
        else:
            summary["status"] = "completed"
    except BaseException as exc:  # noqa: BLE001 - 中断同样必须先关断并保存。
        problem = exc
        summary["status"] = (
            "cancelled"
            if isinstance(exc, (ExperimentCancelled, KeyboardInterrupt))
            else "failed"
        )
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["error_traceback"] = traceback.format_exc()
    finally:
        try:
            failures = rig.all_off() if connected else ["设备未完成连接"]
            summary["shutdown_errors"] = failures
            summary["all_outputs_off_confirmed"] = connected and not failures
            if failures:
                raise AFGError("组末关断未确认：" + "; ".join(failures))
            if connected and not failures:
                summary["final_channels"] = {
                    name: rig.read_channel(name) for name in CHANNEL_NAMES
                }
                if any(state["output"] for state in summary["final_channels"].values()):
                    raise AFGError("组末检测到输出开启")
        except BaseException as exc:  # noqa: BLE001 - 保存关断失败，仍关闭连接并写记录。
            summary["all_outputs_off_confirmed"] = False
            summary["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            problem = problem or exc
        try:
            rig.close()
        except BaseException as exc:  # noqa: BLE001 - 清理异常不能覆盖原始测量。
            summary["all_outputs_off_confirmed"] = False
            summary["close_error"] = f"{type(exc).__name__}: {exc}"
            problem = problem or exc
        if problem is not None and summary["status"] not in ("failed", "cancelled"):
            summary["status"] = "failed"
        summary["statistics"] = summarize_rounds(rows)
        summary["finished_utc"] = datetime.now(UTC).isoformat()
        save_results(directory, rows, summary)
    if problem is not None:
        raise problem
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "customer_workspace/configs/pzt1_pulse.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="AFG 设备配置，默认 %(default)s",
    )
    parser.add_argument("--channel", choices=CHANNEL_NAMES, default="PZT1")
    parser.add_argument(
        "--intervals-ms", type=float, nargs="+", default=[60, 40, 20, 10, 0]
    )
    parser.add_argument(
        "--directions", nargs="+", choices=DIRECTIONS, default=list(DIRECTIONS)
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--loss-limit", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--windows-python", type=Path)
    parser.add_argument("--launcher-stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.count < 1
        or args.loss_limit < 1
        or any(not math.isfinite(t) or t < 0 for t in args.intervals_ms)
    ):
        parser.error("轮数及丢失上限必须为正，T 必须为非负有限毫秒数")
    if len(set(args.intervals_ms)) != len(args.intervals_ms) or len(
        set(args.directions)
    ) != len(args.directions):
        parser.error("间隔及方向不能重复")
    config = load_config(args.config)
    plan = load_plan(args.plan, config)
    for amplitude in (0.001, 0.1):
        replace(plan.settings[args.channel], amplitude_vpp=amplitude).validate(
            config["channels"][args.channel]
        )
    output = (
        args.output_dir
        or PROJECT_ROOT
        / "customer_workspace/validation"
        / f"amplitude_pairs_{datetime.now(ZoneInfo('Europe/Zurich')):%Y%m%d-%H%M%S}"
    )
    if os.name != "nt":
        return launch_from_wsl(
            PROJECT_ROOT,
            args.plan,
            args.config,
            output,
            args.windows_python,
            module="customer_workspace.amplitude_pair_diagnostic",
            extra_args=[
                "--channel",
                args.channel,
                "--count",
                str(args.count),
                "--loss-limit",
                str(args.loss_limit),
                "--intervals-ms",
                *map(str, args.intervals_ms),
                "--directions",
                *args.directions,
            ],
        )
    output.mkdir(parents=True, exist_ok=False)
    cancel = threading.Event()
    if args.launcher_stdin:
        threading.Thread(
            target=watch_launcher_input, args=(cancel.set,), daemon=True
        ).start()
    exit_code = 0
    try:
        with (output / "configuration.json").open("x", encoding="utf-8") as file:
            json.dump(
                {
                    "arguments": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    "afg_config": config,
                    "plan": json.loads(args.plan.read_text(encoding="utf-8")),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        for interval_ms in args.intervals_ms:
            for direction in args.directions:
                directory = output / f"T_{interval_ms:g}ms_{direction}"
                run_group(
                    config,
                    plan,
                    directory,
                    args.channel,
                    interval_ms,
                    direction,
                    args.count,
                    args.loss_limit,
                    cancel,
                )
    except BaseException as exc:  # noqa: BLE001 - 中断也必须独立核查并保存已完成组。
        traceback.print_exception(exc)
        exit_code = (
            130 if isinstance(exc, (ExperimentCancelled, KeyboardInterrupt)) else 1
        )
    finally:
        # 重新连接，只读确认；与每组使用的会话独立，不再设置幅值或输出。
        postflight = {"checked_utc": datetime.now(UTC).isoformat()}
        try:
            with Rig(config) as observer:
                postflight["channels"] = {
                    name: observer.read_channel(name) for name in CHANNEL_NAMES
                }
            postflight["all_outputs_off_confirmed"] = all(
                not state["output"] for state in postflight["channels"].values()
            )
            if not postflight["all_outputs_off_confirmed"]:
                raise AFGError("独立核查发现输出开启")
        except Exception as exc:  # noqa: BLE001 - 独立核查失败必须明确记录。
            postflight["all_outputs_off_confirmed"] = False
            postflight["error"] = f"{type(exc).__name__}: {exc}"
            exit_code = exit_code or 1
        with (output / "postflight.json").open("x", encoding="utf-8") as file:
            json.dump(postflight, file, ensure_ascii=False, indent=2)
        # 失败组同样进入矩阵汇总，不能只列成功返回的组。
        results = []
        for interval_ms in args.intervals_ms:
            for direction in args.directions:
                path = output / f"T_{interval_ms:g}ms_{direction}" / "summary.json"
                if path.exists():
                    summary = json.loads(path.read_text(encoding="utf-8"))
                    results.append(
                        {
                            "interval_ms": interval_ms,
                            "direction": direction,
                            "status": summary["status"],
                            **summary["statistics"],
                        }
                    )
        with (output / "groups.json").open("x", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
        if results:
            with (output / "groups.csv").open(
                "x", encoding="utf-8-sig", newline=""
            ) as file:
                writer = csv.DictWriter(file, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
