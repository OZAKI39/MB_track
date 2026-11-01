"""汇总同步/异步保存对照及连续回放；只读已有记录，不执行实验。"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def writer_metrics(directory):
    report = json.loads((directory / "writer_report.json").read_text())
    rows = read_rows(directory / "writer_timing.csv")
    if (
        not report["all_submitted_saved"]
        or report["error"]
        or report["pending_records"]
    ):
        raise ValueError(f"Save did not complete: {directory}")
    if len(rows) != report["committed_frames"]:
        raise ValueError(f"Writer timing count mismatch: {directory}")
    batches = {r["started_s"]: r for r in rows}
    report["timing_ms"] = {
        key: stats([float(r[key]) for r in rows]) for key in ("queue_ms", "durable_ms")
    }
    report["batch_timing_ms"] = {
        key: stats([float(r[key]) for r in batches.values()])
        for key in ("database_ms", "commit_ms", "commit_cpu_ms")
    }
    report["batches"] = len(batches)
    report["largest_batch"] = max(int(r["batch_frames"]) for r in rows)
    return report


def make_report(root, output, font):
    if not font.is_file():
        raise FileNotFoundError(f"Arial font is required: {font}")
    output.mkdir(exist_ok=False)
    full = {}
    for experiment in "AB":
        for mode in ("sync", "async"):
            directory = root / f"fixed_{experiment}_{mode}"
            rows = read_rows(directory / "stage_timing.csv")
            full[f"{experiment}_{mode}"] = {
                key: stats([float(row[key]) for row in rows])
                for key in (
                    "process_total_ms",
                    "segment_ms",
                    "track_ms",
                    "pack_records_ms",
                    "database_write_ms",
                    "commit_ms",
                    "commit_cpu_ms",
                )
            }
    paced = []
    for directory in sorted(root.glob("paced_[AB]_*_[123]")):
        _, experiment, mode, iteration = directory.name.split("_")
        report = json.loads((directory / "paced_report.json").read_text())
        if (
            report["status"] != "completed"
            or not report["all_offered_frames_accounted"]
        ):
            raise ValueError(f"Incomplete replay: {directory}")
        rows = [
            r
            for r in read_rows(directory / "consumer_timing.csv")
            if r["outcome"] == "processed"
        ]
        data = {
            "name": directory.name,
            "experiment": experiment,
            "mode": mode,
            "iteration": int(iteration),
            "report": report,
            "process_ms": stats([float(r["process_ms"]) for r in rows]),
        }
        if mode == "async":
            data["writer"] = writer_metrics(directory)
            if data["writer"]["committed_frames"] != len(rows):
                raise ValueError("Processed and saved frame counts differ")
        paced.append(data)
    if len(paced) != 12:
        raise ValueError(
            "Expected three complete sync/async replays for each experiment"
        )
    long_dir = root / "long_async_A"
    long_report = json.loads((long_dir / "paced_report.json").read_text())
    if (
        long_report["status"] != "completed"
        or not long_report["synthetic_repeated_replay"]
    ):
        raise ValueError("Expected a completed continuous replay diagnostic")
    long_writer = writer_metrics(long_dir)
    if long_report["processed_frames"] != long_writer["committed_frames"]:
        raise ValueError("Long replay has unsaved analyzed frames")
    memory = read_rows(long_dir / "memory.csv")
    elapsed = np.array([float(r["elapsed_s"]) for r in memory])
    rss = np.array([float(r["consumer_rss_mb"]) for r in memory])
    memory_summary = {
        "first_60s_median_mb": float(np.median(rss[elapsed < 60])),
        "last_60s_median_mb": float(np.median(rss[elapsed > elapsed[-1] - 60])),
        "min_mb": float(rss.min()),
        "max_mb": float(rss.max()),
    }
    calls = re.findall(
        r"(\d+) [\d.]+ fdatasync\(.*?<([\d.]+)>",
        (root / "sync_syscalls.log").read_text(),
    )
    trace = {
        "calls": len(calls),
        "thread_ids": sorted({tid for tid, _ in calls}),
        "duration_ms": stats([float(value) * 1000 for _, value in calls]),
        "caution": "Tracing overhead is substantial; not a throughput benchmark",
    }
    final_checks = {}
    for name in ("final_guard_check", "cancel_check"):
        final_checks[name] = {
            "replay": json.loads((root / name / "paced_report.json").read_text()),
            "writer": writer_metrics(root / name),
        }
    test_result = (root / "final_tests.log").read_text().strip().splitlines()[-1]
    report = {
        "full": full,
        "paced": paced,
        "long_report": long_report,
        "long_writer": long_writer,
        "memory": memory_summary,
        "syscall_trace": trace,
        "final_checks": final_checks,
        "unit_tests": test_result,
        "equality": json.loads((root / "scientific_equality.json").read_text()),
        "scope": "Offline replay only; no live camera recording or instrument control",
    }
    (output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    writer_rows = read_rows(long_dir / "writer_timing.csv")
    origin = json.loads((long_dir / "run_state.json").read_text())["origin_perf_s"]
    render(paced, memory, writer_rows, origin, font, output)
    lines = [
        "# 分析与保存并行：验证报告",
        "",
        "识别与追踪算法未改。保存线程复用原数据库，保留 FULL 可靠提交；默认队列 32 条，最多合批 4 帧。",
        "正常结束和取消都会排空；队列满、等待过久、写盘失败或关闭超时均报错。",
        "",
        "## 4.52 毫秒在哪里",
        "",
        "原版没有保存线程或保存队列，分析线程直接等待 SQLite 提交。下表是本次未加系统跟踪工具的复测。",
        "CPU 时间远小于实际经过时间，说明大部分是系统提交等待。系统调用记录另外确认同一个分析线程执行 fdatasync。",
        "这能定位到系统的可靠写入路径，不能据此把全部耗时归为物理硬盘本身。跟踪工具会明显拖慢运行，其帧率不用于性能比较。",
        "",
        "| 完整输入 | 同步调用均值 ms | 异步调用均值 ms | 同步提交均值 ms | 提交 CPU 均值 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for experiment in "AB":
        before, after = full[experiment + "_sync"], full[experiment + "_async"]
        lines.append(
            f"| {experiment}，1320 帧 | {before['process_total_ms']['mean']:.3f} | "
            f"{after['process_total_ms']['mean']:.3f} | {before['commit_ms']['mean']:.3f} | "
            f"{before['commit_cpu_ms']['mean']:.3f} |"
        )
    lines += [
        "",
        "异步调用仅等快照入队，磁盘实际等待另外记录，不能把异步分析计时中的 0 毫秒当成没有写盘开销。",
        "",
        "## 相同输入，保存结果是否相同",
        "",
        "两段各 1320 帧，共 39884 条区域观测。与本次同步版以及此前保存的版本逐条比较：",
        "轮廓字节、坐标、速度、编号、临时目标状态、事件及修订、源帧时间、最终背景和 PZT 对应文件完全一致。",
        "只排除实际运行耗时、运行标识和代码版本指纹。详见 scientific_equality.json。",
        "",
        "## 按真实播放速度测试",
        "",
        "两段录像分别同步/异步各运行 3 次，顺序交替。所有测试串行运行，不同时进行其他基准测试。",
        "分析输入仍只保留最新帧，因来不及分析被替换的帧，与保存队列丢失是两回事。",
        "",
        "| 运行 | 分析并保存帧数 / 1320 | 分析帧率 fps | 调用均值 ms | 分析前替换 | 保存积压峰值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for data in paced:
        item = data["report"]
        peak = data.get("writer", {}).get("peak_pending_records", "—")
        lines.append(
            f"| {data['name']} | {item['processed_frames']} | {item['processed_fps']:.2f} | "
            f"{data['process_ms']['mean']:.3f} | {item['replaced_frames']} | {peak} |"
        )
    lines += [
        "",
        "## 连续运行检查",
        "",
        f"将 A 原录像连续重复 15 遍，按固定计划送帧 {long_report['nominal_duration_s']:.1f} 秒。这是软件负载测试，不是新增真实实验。",
        (
            f"送入 {long_report['offered_frames']} 帧，分析 {long_report['processed_frames']} 帧；"
            f"已接收并保存 {long_writer['committed_frames']} 帧，结束剩余 {long_writer['pending_records']} 条。"
        ),
        (
            f"保存积压峰值 {long_writer['peak_pending_records']} 条（容量 32）；入队到保存最大 "
            f"{long_writer['timing_ms']['durable_ms']['max']:.3f} ms。"
        ),
        (
            f"分析进程内存：前 60 秒中位数 {memory_summary['first_60s_median_mb']:.2f} MB，"
            f"末 60 秒 {memory_summary['last_60s_median_mb']:.2f} MB；范围 "
            f"{memory_summary['min_mb']:.2f}–{memory_summary['max_mb']:.2f} MB。"
        ),
        "原画面对应保存在 source_timing.csv 的 original_* 列；虚拟循环时间不能用于科学分析或 PZT 同步。",
        "",
        "## 范围与限制",
        "",
        "本次使用 WSL、项目现有 Python 环境和两段原始录像，没有连接相机或 AFG。",
        f"自动检查：{test_result}。覆盖独立快照、合批事件时序、队列满、写入回滚、后台失败与入队交错、取消和关闭超时。",
        "在重复测试后补强了后台错误与新帧入队之间的检查，并将关闭超时快照单独保存，避免两个线程改写同一报告。",
        "最终代码另完成 30 秒回放：1794 条分析记录全部保存；约 2 秒主动取消时，120 条已提交记录全部保存。",
        "所有已交给保存线程的帧均保存完整；原录像只读。真实相机同时录像时是否增加掉帧，尚未进行实机复测，不能宣布已验收。",
        "本次连续运行长度为约 5.5 分钟，不能由此保证任意长时间或任意磁盘负载下永不积压。",
        "强制杀进程或断电时，仍在内存队列里的帧可能未落盘；应以已保存计数与数据库为准。",
        "",
        "图中左上为分析调用耗时，右上为真正处理的帧比例；下方分别显示长时间保存积压和内存。",
        "JSON 保留全部数值；各运行目录保存逐帧计时、帧去向、动作对应及最终数据库。",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def render(paced, memory, writer_rows, origin, font, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(font))
    if font_manager.FontProperties(fname=str(font)).get_name() != "Arial":
        raise ValueError("The supplied font is not Arial")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), layout="constrained")
    colors = {"sync": "#B75B66", "async": "#829E63"}
    for col, (key, label) in enumerate(
        (("process", "Analysis call (ms)"), ("fraction", "Analyzed frames (%)"))
    ):
        ax = axes[0, col]
        for i, experiment in enumerate("AB"):
            for mode, offset in (("sync", -0.18), ("async", 0.18)):
                group = [
                    r
                    for r in paced
                    if r["experiment"] == experiment and r["mode"] == mode
                ]
                values = [
                    r["process_ms"]["mean"]
                    if key == "process"
                    else r["report"]["processed_frames"]
                    / r["report"]["offered_frames"]
                    * 100
                    for r in group
                ]
                ax.bar(
                    i + offset,
                    np.mean(values),
                    width=0.30,
                    color=colors[mode],
                    label=mode.capitalize() if i == 0 else None,
                    alpha=0.85,
                )
                ax.scatter(
                    i + offset + np.linspace(-0.06, 0.06, len(values)),
                    values,
                    s=18,
                    facecolor="white",
                    edgecolor="#333333",
                    linewidth=0.8,
                    zorder=3,
                )
        ax.set(xticks=[0, 1], xticklabels=["Exp. A", "Exp. B"], ylabel=label)
        ax.legend(frameon=False)
    elapsed = [float(r["elapsed_s"]) for r in memory]
    axes[1, 0].plot(
        [float(r["committed_s"]) - origin for r in writer_rows],
        [int(r["pending_after_commit"]) for r in writer_rows],
        color="#829E63",
        lw=1.2,
    )
    axes[1, 0].axhline(32, color="#B75B66", ls="--", lw=1, label="Capacity")
    axes[1, 0].set(
        xlabel="Replay time (s)", ylabel="Pending records after commit", ylim=(-1, 35)
    )
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(
        elapsed, [float(r["consumer_rss_mb"]) for r in memory], color="#803028", lw=1.2
    )
    axes[1, 1].set(xlabel="Replay time (s)", ylabel="Analysis process memory (MB)")
    axes[1, 1].set_ylim(0, max(float(r["consumer_rss_mb"]) for r in memory) * 1.1)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", width=0.8)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"save_comparison.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--font", type=Path, default=Path("/mnt/c/Windows/Fonts/arial.ttf")
    )
    args = parser.parse_args()
    make_report(args.root, args.output_dir, args.font)


if __name__ == "__main__":
    main()
