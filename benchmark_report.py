"""专项测试的纯数据核对、统计和绘图；导入或绘图均不访问仪器。"""

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path
from statistics import linear_regression, mean

from customer_workspace.timing_report import describe


def distribution(values: list[float]) -> dict:
    """沿用既有统计，增加按最近秩计算的第 95、99 百分位。"""
    ordered = sorted(values)
    return {
        **describe(values),
        "p95": ordered[math.ceil(len(values) * 0.95) - 1],
        "p99": ordered[math.ceil(len(values) * 0.99) - 1],
    }


def summarize_samples(samples: list[dict], method: str, interval_ms: float) -> dict:
    """核对时间顺序并计算指标；拒绝将失败或缺少回复的记录当作成功。"""
    if len(samples) < 2 or method not in ("schedule", "readback", "scheduled_readback"):
        raise ValueError("需要至少两条记录及明确的测试方法")
    scheduled = method in ("schedule", "scheduled_readback")
    readback = method in ("readback", "scheduled_readback")
    delays = []
    sending_delays = []
    writes = []
    query_gaps = []
    for index, row in enumerate(samples, 1):
        if row["index"] != index or row["status"] != (
            "sent" if method == "schedule" else "readback_ok"
        ):
            raise ValueError("记录序号不连续或存在未成功动作")
        times = [
            row["call_started_s"],
            row["command_sent_s"],
            row["command_write_returned_s"],
        ]
        if readback:
            times.extend([row["query_started_s"], row["amplitude_reply_received_s"]])
            if row["confirmed_s"] != row[
                "amplitude_reply_received_s"
            ] or not math.isclose(
                row["amplitude_readback_vpp"],
                row["amplitude_requested_vpp"],
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                raise ValueError("查询回复没有通过核验")
            delay = (row["amplitude_reply_received_s"] - row["command_sent_s"]) * 1000
            if "before_query_min_ms" in row:
                minimum = row["before_query_min_ms"]
                gap = (row["query_started_s"] - row["command_write_returned_s"]) * 1000
                if (
                    not math.isfinite(minimum)
                    or minimum < 0
                    or not math.isfinite(gap)
                    or gap + 1e-6 < minimum
                    or not math.isclose(gap, row["write_to_query_ms"], abs_tol=1e-6)
                ):
                    raise ValueError("查询前等待不足或等待记录不一致")
                query_gaps.append(gap)
        else:
            if row["confirmed_s"] is not None or row["query"] is not None:
                raise ValueError("只发送测试不能包含逐次查询确认")
        if scheduled:
            planned = (index - 1) * interval_ms / 1000
            if not math.isclose(row["planned_s"], planned, abs_tol=1e-9):
                raise ValueError("计划时间未按固定起点生成")
            lateness = (row["command_sent_s"] - planned) * 1000
            if not math.isfinite(lateness) or lateness < 0:
                raise ValueError("发送延迟无效或存在提前发送")
            sending_delays.append(lateness)
            if not readback:
                delay = lateness
        times.append(row["call_finished_s"])
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in times
        ) or times != sorted(times):
            raise ValueError("通信时间缺失、非有限或顺序错误")
        if delay < 0:
            raise ValueError("存在提前发送或负耗时")
        delays.append(delay)
        writes.append((row["command_write_returned_s"] - row["command_sent_s"]) * 1000)
    result = {
        "samples": len(samples),
        "metric": "schedule_lateness_ms"
        if method == "schedule"
        else "command_to_reply_ms",
        "delay_ms": distribution(delays),
        "write_ms": distribution(writes),
        "by_state": {},
        "slowest": {
            "index": samples[delays.index(max(delays))]["index"],
            "delay_ms": max(delays),
        },
    }
    if query_gaps:
        if len(query_gaps) != len(samples):
            raise ValueError("部分记录缺少查询前等待信息")
        result["write_to_query_ms"] = distribution(query_gaps)
    for state in ("ACTIVE", "IDLE"):
        values = [value for row, value in zip(samples, delays) if row["state"] == state]
        if values:
            result["by_state"][state] = distribution(values)
    if scheduled:
        indices = list(range(1, len(samples) + 1))
        fit = linear_regression(indices, sending_delays)
        window = min(50, len(samples) // 2)
        result["trend"] = {
            "slope_ms_per_100_commands": fit.slope * 100,
            "fitted_total_change_ms": fit.slope * (len(samples) - 1),
            "intercept_ms": fit.intercept,
            "first_window_mean_ms": mean(sending_delays[:window]),
            "last_window_mean_ms": mean(sending_delays[-window:]),
            "window_commands": window,
            "block_means_ms": [
                mean(sending_delays[start : start + window])
                for start in range(0, len(sending_delays), window)
            ],
        }
        result["sent_at_least_one_interval_late"] = sum(
            value >= interval_ms for value in sending_delays
        )
        result["previous_call_finished_after_next_deadline"] = sum(
            previous["call_finished_s"] > following["planned_s"]
            for previous, following in pairwise(samples)
        )
        result["late_ready_max_ms"] = max(
            max(0, (row["ready_s"] - row["planned_s"]) * 1000) for row in samples
        )
    if method == "scheduled_readback":
        # 必须配对同一条指令；两组独立最大值相加不是实际观测到的最慢一次。
        totals = [
            (row["amplitude_reply_received_s"] - row["planned_s"]) * 1000
            for row in samples
        ]
        slowest = totals.index(max(totals))
        reserve_target = interval_ms - 5
        result["sending_delay_ms"] = distribution(sending_delays)
        result["planned_to_reply_ms"] = distribution(totals)
        result["slowest_planned_to_reply"] = {
            "index": samples[slowest]["index"],
            "lateness_ms": sending_delays[slowest],
            "command_to_reply_ms": delays[slowest],
            "total_ms": totals[slowest],
        }
        result["reserve_target_ms"] = reserve_target
        result["replies_exceeding_reserve_target"] = sum(
            value > reserve_target for value in totals
        )
        result["replies_above_45_ms"] = sum(value > 45 for value in totals)
        result["replies_at_or_after_next_deadline"] = sum(
            value >= interval_ms for value in totals
        )
        result["minimum_margin_ms"] = interval_ms - max(totals)
        result["all_replies_with_5ms_reserve"] = max(totals) <= reserve_target
    return result


def plot_reports(directories: list[Path], output_dir: Path) -> list[Path]:
    """从成功且核查可靠的原始记录重算统计；输出新目录下的 PNG 和 SVG。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "completed" or summary["data_reliable"] is not True:
            raise ValueError(f"{directory} 未通过核查，请先查看失败原因")
        rows = [
            json.loads(line)
            for line in (directory / "samples.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        stats = summarize_samples(rows, summary["method"], summary["interval_ms"] or 20)
        if stats != summary["statistics"]:
            raise ValueError(f"{directory} 的摘要与原始数据不一致")
        records.append((directory, summary, rows, stats))
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for directory, summary, rows, stats in records:
        indices = [row["index"] for row in rows]
        fig, axes = plt.subplots(2, 1, figsize=(12, 7.5))
        if summary["method"] == "schedule":
            delays = [(row["command_sent_s"] - row["planned_s"]) * 1000 for row in rows]
            trend = stats["trend"]
            axes[0].plot(
                indices,
                delays,
                ".-",
                color="#1672b8",
                linewidth=0.6,
                markersize=2,
                label="Each command",
            )
            axes[0].plot(
                indices,
                [
                    trend["intercept_ms"]
                    + trend["slope_ms_per_100_commands"] * index / 100
                    for index in indices
                ],
                color="#bf3e35",
                label="Linear trend",
            )
            axes[0].axhline(
                summary["interval_ms"],
                color="#8b8b8b",
                linestyle="--",
                label="One full interval late",
            )
            axes[0].set(
                ylabel="Sending lateness (ms)",
                title=f"First {trend['window_commands']} mean {trend['first_window_mean_ms']:.3f} ms | Last {trend['window_commands']} mean {trend['last_window_mean_ms']:.3f} ms | Trend {trend['slope_ms_per_100_commands']:+.3f} ms / 100 commands",
            )
            axes[0].set_ylim(0, max(summary["interval_ms"], max(delays)) * 1.1)
            axes[1].plot(
                indices,
                [
                    (row["command_write_returned_s"] - row["command_sent_s"]) * 1000
                    for row in rows
                ],
                ".-",
                linewidth=0.6,
                markersize=2,
                label="VISA write duration",
            )
            axes[1].plot(
                indices,
                [max(0, row["ready_s"] - row["planned_s"]) * 1000 for row in rows],
                linewidth=0.7,
                alpha=0.7,
                label="Already late before waiting",
            )
            axes[1].set(
                ylabel="Milliseconds",
                title="Write cost and lateness carried into the next iteration",
            )
            title = f"Method 1 | {summary['channel']} | {summary['interval_ms']:g} ms interval | {len(rows)} commands"
            footer = "Software sending schedule only; no per-command query. Stable sending does not prove absence of an instrument-side queue."
        elif summary["method"] == "scheduled_readback":
            totals = [
                (row["amplitude_reply_received_s"] - row["planned_s"]) * 1000
                for row in rows
            ]
            sending = [
                (row["command_sent_s"] - row["planned_s"]) * 1000 for row in rows
            ]
            replies = [
                (row["amplitude_reply_received_s"] - row["command_sent_s"]) * 1000
                for row in rows
            ]
            axes[0].plot(
                indices,
                totals,
                ".-",
                color="#6b53a6",
                linewidth=0.7,
                markersize=3,
                label="Sending lateness + reply time (same command)",
            )
            axes[0].axhline(
                stats["reserve_target_ms"],
                color="#c08118",
                linestyle="--",
                label=f"5 ms reserve target: {stats['reserve_target_ms']:g} ms",
            )
            axes[0].axhline(
                summary["interval_ms"],
                color="#bf3e35",
                linestyle="--",
                label=f"Next planned command: {summary['interval_ms']:g} ms",
            )
            slowest = stats["slowest_planned_to_reply"]
            axes[0].set(
                ylabel="Planned time to reply (ms)",
                title=f"Maximum {slowest['total_ms']:.3f} = {slowest['lateness_ms']:.3f} sending + {slowest['command_to_reply_ms']:.3f} reply ms | Command {slowest['index']}",
            )
            axes[0].set_ylim(0, max(summary["interval_ms"], max(totals)) * 1.12)
            axes[1].plot(indices, sending, linewidth=0.7, label="Sending lateness")
            axes[1].plot(
                indices, replies, linewidth=0.7, label="Command to amplitude reply"
            )
            axes[1].set(
                xlabel="Command number",
                ylabel="Milliseconds",
                title="Two measured parts of each command's total",
            )
            title = f"Scheduled readback | {summary['channel']} | {summary['interval_ms']:g} ms interval | {len(rows)} commands"
            footer = "Totals pair each command's actual sending delay and amplitude reply. Parameter readback does not measure voltage settling."
        else:
            delays = [
                (row["amplitude_reply_received_s"] - row["command_sent_s"]) * 1000
                for row in rows
            ]
            for state, color in (("ACTIVE", "#1672b8"), ("IDLE", "#e47922")):
                selected = [
                    (row["index"], delay)
                    for row, delay in zip(rows, delays)
                    if row["state"] == state
                ]
                axes[0].scatter(
                    [item[0] for item in selected],
                    [item[1] for item in selected],
                    s=8,
                    color=color,
                    label=state,
                )
                axes[1].hist(
                    [item[1] for item in selected],
                    bins=80,
                    range=(min(delays), max(delays)),
                    alpha=0.6,
                    color=color,
                    label=state,
                )
            axes[0].set(
                ylabel="Command to reply (ms)",
                title=f"Mean {stats['delay_ms']['mean']:.3f} ms | P99 {stats['delay_ms']['p99']:.3f} ms | Maximum {stats['delay_ms']['max']:.3f} ms at command {stats['slowest']['index']}",
            )
            axes[1].set(
                xlabel="Command to reply (ms)",
                ylabel="Count",
                title="Measured response-time distribution",
            )
            title = f"Method 2 | {summary['channel']} | {len(rows)} write + amplitude query pairs"
            footer = "Includes write, query, USB/driver and host scheduling. Matching parameter readback is not measured voltage settling."
        if summary.get("before_query_min_ms", 0) > 0:
            footer += f"\nIncludes the actual pre-query wait (minimum {summary['before_query_min_ms']:g} ms after amplitude write returns)."
        axes[0].set_xlabel("Command number")
        if summary["method"] == "schedule":
            axes[1].set_xlabel("Command number")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=9)
            axis.set_ylim(bottom=0)
        fig.suptitle(title, fontsize=14)
        fig.text(0.5, 0.02, footer, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        try:
            for extension in ("png", "svg"):
                path = output_dir / f"{directory.name}.{extension}"
                with path.open("xb") as file:
                    fig.savefig(file, format=extension, dpi=160)
                outputs.append(path)
        finally:
            plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从专项测试原始记录核对统计并生成图表，不连接仪器"
    )
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="必须不存在的新图表目录"
    )
    args = parser.parse_args()
    for path in plot_reports(args.directories, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
