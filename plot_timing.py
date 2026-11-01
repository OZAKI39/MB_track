"""将只发送模式的实测 CSV 绘图，可与上一版逐次核查调幅记录比较。"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # 先选无窗口绘图后端。


def read_rows(directory: Path) -> tuple[list[dict], dict]:
    """读取已结束批次的报告；拒绝空数据和无效数值。"""
    report = json.loads((directory / "timing_summary.json").read_text(encoding="utf-8"))
    with (directory / "timing.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("没有可绘制的可靠实验记录")
    for row in rows:
        for key in (
            "active_schedule_delay_ms",
            "idle_schedule_delay_ms",
            "active_write_ms",
            "idle_write_ms",
            "active_command_sent_s",
            "idle_command_sent_s",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"{key} 包含无效数值")
    return rows, report


def plot_report(directory: Path, baseline: Path | None = None) -> list[Path]:
    """生成 PNG/SVG；只比较共同的发送指标，不把缺失确认画成零延迟。"""
    rows, report = read_rows(directory)
    if report["control_mode"] != "amplitude_write_only":
        raise ValueError("主数据必须是只发送模式")
    if len({row["channel"] for row in rows}) != 1:
        raise ValueError("此图要求单个 PZT 通道的数据")
    outputs = [directory / "delay_comparison.png", directory / "delay_comparison.svg"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("图文件已存在，拒绝覆盖历史结果")
    baseline_rows = None
    if baseline is not None:
        baseline_rows, baseline_report = read_rows(baseline)
        if baseline_report["control_mode"] != "amplitude_states":
            raise ValueError("基线需要上一版逐次核查的调幅记录")
        if {row["channel"] for row in baseline_rows} != {
            row["channel"] for row in rows
        }:
            raise ValueError("基线与本轮的 PZT 通道不同")
        current_plan = json.loads(
            (directory / report["runs"][0]["directory"] / "plan.json").read_text(
                encoding="utf-8"
            )
        )
        baseline_plan = json.loads(
            (
                baseline / baseline_report["runs"][0]["directory"] / "plan.json"
            ).read_text(encoding="utf-8")
        )
        if current_plan != baseline_plan:
            raise ValueError("基线的时间表或设备参数不同，不能直接比较")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    indices = [int(row["run_index"]) for row in rows]
    colors = {"active": "#1672b8", "idle": "#e47922"}
    for axis, suffix, title in (
        (axes[0, 0], "schedule_delay_ms", "Planned time to command sent"),
        (axes[0, 1], "write_ms", "VISA write call duration"),
    ):
        for state, color in colors.items():
            values = [float(row[f"{state}_{suffix}"]) for row in rows]
            axis.plot(
                indices,
                values,
                ".-",
                color=color,
                linewidth=1,
                markersize=5,
                label=f"{state.upper()} (mean {mean(values):.3f} ms)",
            )
            axis.axhline(
                mean(values), color=color, linestyle="--", alpha=0.45, linewidth=0.9
            )
        axis.set(title=title, xlabel="Run number", ylabel="Milliseconds")
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8)

    axis = axes[1, 0]
    if baseline_rows is not None:
        groups = [
            ("Previous ACTIVE", baseline_rows, "active"),
            ("New ACTIVE", rows, "active"),
            ("Previous IDLE", baseline_rows, "idle"),
            ("New IDLE", rows, "idle"),
        ]
        values = [
            [float(row[f"{state}_schedule_delay_ms"]) for row in records]
            for _, records, state in groups
        ]
        means = [mean(group) for group in values]
        deviations = [stdev(group) if len(group) > 1 else 0 for group in values]
        axis.bar(
            range(4),
            means,
            yerr=deviations,
            capsize=4,
            color=[colors[state] for _, _, state in groups],
            alpha=0.75,
        )
        axis.set_xticks(
            range(4),
            [f"{label}\n(n={len(records)})" for label, records, _ in groups],
            fontsize=9,
        )
        for index, value in enumerate(means):
            axis.text(
                index,
                value + deviations[index] + 2,
                f"{value:.2f}",
                ha="center",
                fontsize=9,
            )
        axis.set_ylim(
            0,
            max(value + deviation for value, deviation in zip(means, deviations)) * 1.2,
        )
        axis.set(
            title="Sending delay: mean ± sample standard deviation",
            ylabel="Milliseconds",
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No baseline supplied",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()

    axis = axes[1, 1]
    spacing = [
        (float(row["idle_command_sent_s"]) - float(row["active_command_sent_s"])) * 1000
        for row in rows
    ]
    axis.plot(indices, spacing, ".-", color="#6b53a6", linewidth=1)
    axis.set(
        title="ACTIVE to IDLE command spacing",
        xlabel="Run number",
        ylabel="Milliseconds (not physical pulse width)",
    )
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{rows[0]['channel']} write-only timing | {report['reliable_runs']} reliable / {len(report['runs'])} attempts",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        f"Unreliable: {report['unreliable_runs']} | Failed: {report['failed_runs']} | Dashed lines: means",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.025,
        "PC-side sending times only. No per-command confirmation or physical voltage timing was measured.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91), h_pad=2.5)
    try:
        for path in outputs:
            with path.open("xb") as file:
                fig.savefig(file, format=path.suffix[1:], dpi=160)
    finally:
        plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制只发送模式的延迟结果，不连接设备")
    parser.add_argument(
        "directory", type=Path, help="含 timing.csv 和 timing_summary.json 的批次目录"
    )
    parser.add_argument("--baseline", type=Path, help="上一版逐次核查调幅的批次目录")
    args = parser.parse_args()
    for path in plot_report(args.directory, args.baseline):
        print(path)


if __name__ == "__main__":
    main()
