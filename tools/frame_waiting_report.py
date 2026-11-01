"""汇总一帧/两帧等待区的重复测试：跳帧、同钟反馈、交接等待和结果正确性。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from customer_workspace.streamcopy_recording import write_json
from customer_workspace.tools.tracking_latency_report import live_run, save_csv, stats


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def draw_comparison(runs, output):
    """图表只呈现实测值；使用 Windows 已安装的真实 Arial。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import MaxNLocator

    font = Path("/mnt/c/Windows/Fonts/arial.ttf")
    if not font.is_file():
        raise FileNotFoundError("Arial is required for the comparison figure")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    metrics = [
        ("skipped", "Skipped analysis frames"),
        ("feedback_p99_ms", "Feedback P99 (ms)"),
        ("wait_p99_ms", "Handoff wait P99 (ms)"),
    ]
    for row, experiment in enumerate(("A", "B")):
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for capacity, color, shift in ((1, "#397C8C", -0.18), (2, "#DFA23B", 0.18)):
                selected = sorted(
                    [
                        r
                        for r in runs
                        if r["experiment"] == experiment and r["capacity"] == capacity
                    ],
                    key=lambda r: r["repeat"],
                )
                ax.bar(
                    np.arange(len(selected)) + 1 + shift,
                    [r[metric] for r in selected],
                    width=0.36,
                    color=color,
                    label=f"{capacity} waiting frame" + ("s" if capacity == 2 else ""),
                )
            ax.set_xticks([1, 2, 3])
            ax.set_xlabel("Repeat")
            ax.set_ylabel(label)
            ax.set_title(f"Exp. {experiment}")
            ax.set_ylim(bottom=0)
            if metric == "skipped":
                ax.set_ylim(0, max(1, ax.get_ylim()[1]))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=2,
        frameon=False,
    )
    fig.savefig(output / "comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def build(root, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = read_json(root / "test_plan.json")
    inputs = read_json(root / "inputs.json")
    integrity = read_json(root / "integrity_report/summary.json")
    audited = {r["run"]: r for r in integrity["runs"]}
    runs, pooled, detailed = [], {}, {}
    for name, config in inputs["runs"].items():
        folder = root / name
        latency = live_run(folder, output)
        bridge = read_json(folder / "tracking_bridge.json")
        analysis = read_json(folder / "analysis_check.json")
        recorded = read_json(folder / "recording_check.json")
        frames = read_rows(output / f"{name}_per_frame.csv")
        outcomes = read_rows(folder / "frame_outcomes.csv")
        skipped = {
            int(r["source_frame"]) for r in outcomes if r["outcome"] != "processed"
        }
        replaced = [r["source_frame"] for r in bridge["replacement_events"]]
        capacity, experiment = config["waiting_frames"], config["experiment"]
        queue_checks = {
            "capacity_as_requested": bridge["waiting_capacity"] == capacity,
            "queue_never_exceeded_capacity": bridge["peak_waiting_frames"] <= capacity,
            "tail_drained": bridge["pending_at_shutdown"] == 0,
            "all_skips_have_replacement_event": skipped == set(replaced),
            "replacement_counts_match": len(replaced)
            == len(set(replaced))
            == analysis["analysis_skipped"]
            == bridge["overwritten"],
        }
        if (
            not all(queue_checks.values())
            or not latency["passed"]
            or not audited[name]["accounting_passed"]
        ):
            raise ValueError(f"Integrity, queue or ACK check failed: {name}")
        feedback = latency["stages"]["feedback_total_ms"]
        wait = latency["stages"]["publish_wait_take_ms"]
        source_rss = read_json(folder / "summary.json")["resources"]
        row = {
            "run": name,
            "experiment": experiment,
            "capacity": capacity,
            "repeat": config["repeat"],
            "recorded": recorded["saved_frames"],
            "decoded": len(bridge["decoded"]),
            "analyzed": analysis["processed"],
            "saved": analysis["saved"],
            "skipped": analysis["analysis_skipped"],
            "feedback_median_ms": feedback["median"],
            "feedback_p95_ms": feedback["p95"],
            "feedback_p99_ms": feedback["p99"],
            "feedback_max_ms": feedback["max"],
            "wait_median_ms": wait["median"],
            "wait_p95_ms": wait["p95"],
            "wait_p99_ms": wait["p99"],
            "wait_max_ms": wait["max"],
            "feedback_over_33ms": sum(
                float(r["feedback_total_ms"]) > 1000 / 30 for r in frames
            ),
            "feedback_over_50ms": sum(
                float(r["feedback_total_ms"]) > 50 for r in frames
            ),
            "wait_over_one_frame": sum(
                float(r["publish_wait_take_ms"]) > 1000 / 60 for r in frames
            ),
            "peak_waiting": bridge["peak_waiting_frames"],
            "analysis_peak_rss_mib": max(r["rss_mib"] for r in source_rss),
            "decoder_peak_rss_mib": max(
                r.get("decoder_rss_mib") or 0 for r in source_rss
            ),
            "source_cycle_start_frame": audited[name]["source_cycle_start_frame"],
        }
        runs.append(row)
        detailed[name] = {
            "queue_checks": queue_checks,
            "latency": latency,
            "integrity_checks": audited[name]["checks"],
        }
        pooled.setdefault((experiment, capacity), []).extend(frames)
    save_csv(output / "runs.csv", runs)
    groups, decisions = {}, {}
    for (experiment, capacity), frames in pooled.items():
        matching = [
            r
            for r in runs
            if r["experiment"] == experiment and r["capacity"] == capacity
        ]
        groups[f"{experiment}_q{capacity}"] = {
            "recorded": sum(r["recorded"] for r in matching),
            "analyzed": sum(r["analyzed"] for r in matching),
            "skipped": sum(r["skipped"] for r in matching),
            "feedback_ms": stats([float(r["feedback_total_ms"]) for r in frames]),
            "wait_ms": stats([float(r["publish_wait_take_ms"]) for r in frames]),
            "feedback_over_50ms": sum(r["feedback_over_50ms"] for r in matching),
            "wait_over_one_frame": sum(r["wait_over_one_frame"] for r in matching),
        }
    for experiment in ("A", "B"):
        original, candidate = groups[f"{experiment}_q1"], groups[f"{experiment}_q2"]
        reduction = (
            (1 - candidate["skipped"] / original["skipped"])
            if original["skipped"]
            else None
        )
        deltas = {
            q: candidate["feedback_ms"][q] - original["feedback_ms"][q]
            for q in ("p95", "p99", "max")
        }
        decisions[experiment] = {
            "skip_reduction_fraction": reduction,
            "feedback_change_ms": deltas,
            "fewer_skips": reduction is not None
            and reduction >= plan["skip_reduction_fraction"],
            "feedback_tail_within_margin": all(
                deltas[q] <= plan["maximum_p95_p99_increase_ms"] for q in ("p95", "p99")
            ),
            "wait_p99_below_one_frame": candidate["wait_ms"]["p99"]
            <= plan["maximum_handoff_p99_ms"],
        }
    promising = all(
        r["fewer_skips"]
        and r["feedback_tail_within_margin"]
        and r["wait_p99_below_one_frame"]
        for r in decisions.values()
    )
    result = {
        "plan": plan,
        "runs": runs,
        "groups": groups,
        "decisions": decisions,
        "meets_comparison_criteria": promising,
        "default_waiting_frames": 1,
        "details": detailed,
        "hardware_tested": False,
    }
    write_json(output / "summary.json", result)
    draw_comparison(runs, output)
    lines = [
        "# 一帧与两帧等待区对照",
        "",
        "A、B 各在两种设置下重复 3 次，每次 22 秒；12 组都开启上一轮相同的 Windows 同钟完成确认计时。",
        "正式默认仍为一帧；两帧只通过 `--waiting-frames 2` 显式启用。",
        "",
        "## 本轮判断",
        "",
        (
            "两帧方案达到本轮预先设定的比较标准：跳帧减少，反馈尾部增加在限度内。可作为下一步候选，尚不更改默认。"
            if promising
            else "两帧方案没有同时达到全部比较标准；保留单帧默认。应结合下面的逐视频结果，区分改善、代价和证据不足。"
        ),
        "",
        "预先写入 test_plan.json 的标准：每段视频合计跳帧至少减少一半；反馈 P95/P99 各增加不超过 2 ms；交接等待 P99 不超过一帧约 16.7 ms。这是本次工程比较标准，不是曝光或设备控制的安全期限。",
        "",
        "## 三次合计",
        "",
        "| 视频 | 等待帧数 | 录像/解码 | 分析/保存 | 跳帧 | 反馈中位数 | P95 | P99 | 最大 | 等待 P99 | 等待最大 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in groups.items():
        fb, wait = group["feedback_ms"], group["wait_ms"]
        lines.append(
            f"| {name[0]} | {name[-1]} | {group['recorded']} | {group['analyzed']} | {group['skipped']} | {fb['median']:.3f} | {fb['p95']:.3f} | {fb['p99']:.3f} | {fb['max']:.3f} | {wait['p99']:.3f} | {wait['max']:.3f} |"
        )
    lines += [
        "",
        "时间单位均为 ms；合计分位数由全部逐帧记录重新计算，不是对三组 P99 取平均。跳过的帧不伪造分析完成确认。",
        "",
    ]
    for name, decision in decisions.items():
        reduction = decision["skip_reduction_fraction"]
        change = decision["feedback_change_ms"]
        text = (
            "原方案无跳帧，无法判断减少比例"
            if reduction is None
            else f"跳帧减少 {reduction:.1%}"
        )
        lines.append(
            f"- {name}：{text}；反馈 P95 改变 {change['p95']:+.3f} ms，P99 {change['p99']:+.3f} ms，最大值 {change['max']:+.3f} ms。"
        )
        original, candidate = groups[f"{name}_q1"], groups[f"{name}_q2"]
        lines.append(
            f"  交接等待超过一帧时长的画面由 {original['wait_over_one_frame']} 帧变为 {candidate['wait_over_one_frame']} 帧；最长等待由 {original['wait_ms']['max']:.3f} ms 变为 {candidate['wait_ms']['max']:.3f} ms。"
        )
    lines += [
        "",
        "## 每次结果",
        "",
        "| 运行 | 跳帧 | 反馈 P95 | 反馈 P99 | 反馈最大 | 等待 P99 | 反馈超过 50 ms | 等待超过一帧 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in runs:
        lines.append(
            f"| {r['run']} | {r['skipped']} | {r['feedback_p95_ms']:.3f} | {r['feedback_p99_ms']:.3f} | {r['feedback_max_ms']:.3f} | {r['wait_p99_ms']:.3f} | {r['feedback_over_50ms']} | {r['wait_over_one_frame']} |"
        )
    lines += [
        "",
        "![Queue comparison](comparison.png)",
        "",
        "图中每行对应一段视频，每列依次比较跳帧数、反馈 P99、交接等待 P99。蓝色是一帧等待，黄色是两帧；横轴为第几次重复。完整数字在 runs.csv。",
        "",
        "## 如何保证没有以错误结果换取速度",
        "",
        "- 等待区按先来先处理；满时仅替换最旧的待取帧。正在分析和正在写入的空间各自独占；一帧方案共 3 块，两帧方案共 4 块。",
        "- 多一块 1280×1024 灰度图及帧头，预分配空间约增加 1.31 MiB；实际进程峰值内存在 runs.csv，不能把两个进程的共享驻留内存简单相加。",
        "- 每次替换保存被跳过帧号、替换它的新帧号及 WSL 时间；逐一核对所有分析跳帧。等待数量没有超过限制，正常结束等待区全部排空。",
        "- 12 组录像均逐帧核对原 A/B 视频循环中的像素；每组所有实际分析帧重新离线计算，核对轮廓、编号、运动、事件和源时间记录。数据库与确认消息均核对帧数；详细证据在 ../integrity_report/ 和各组 equivalence.json。",
        "- 两种设置可能实际分析不同的帧，也可能从视频不同位置接入，因此不能要求两次独立运行的编号完全相同。核对的是各自相同输入下的结果，不把正常的输入差异误判成算法改变。",
        "",
        "## 计时边界及限制",
        "",
        "完整反馈：Windows 在 H.264 拆包解析后、分流前记录收到帧，到 Windows 收到同一帧的分析完成确认。两点使用同一个 Windows 时钟。",
        "交接等待：WSL 完成共享内存复制、准备发布，到开始分析；包含发布、唤醒、排队及取帧。只相减 WSL 本机时间。",
        "这些值不包含相机曝光、前端 RTSP 接收缓冲，也不包含数据库最终全部写盘和 AFG 动作。分析完成是已计算并提交保存，实验结束另核对全部保存完成。",
        "短测没有建立无限期的低延迟保证；两帧空间只应对短时成批到达，不能解决处理速度长期低于输入速度。极端停顿仍可造成旧帧或替换。",
        "各组是独立回放，帧到达起伏和系统调度也会影响尾部；这些是本次观测差异，不能把每一次最慢反馈的变化全部归因于等待区容量。",
        "模拟视频循环只是负载测试，不是新实验。所有采集完成后才做离线重算，不用重算任务干扰计时；giolibproxy.dll 警告只记录，不修改 Windows 环境。",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {"meets_comparison_criteria": promising, "decisions": decisions}, indent=2
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)


if __name__ == "__main__":
    main()
