"""汇总相对暗度优化的实测与核查，生成简明报告和 Arial 图表。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STAGES = (
    "seg_contrast_ms",
    "seg_components_ms",
    "segment_ms",
    "background_ms",
    "process_total_ms",
)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def make_report(root, output, font):
    if not font.is_file():
        raise FileNotFoundError("Actual Arial font is required")
    output.mkdir(exist_ok=False)
    fixed = {}
    for name in [f"before_fixed_{letter}" for letter in "AB"] + [
        f"final_fixed_{letter}_{repeat}" for letter in "AB" for repeat in (1, 2, 3)
    ]:
        rows = read_rows(root / name / "stage_timing.csv")
        fixed[name] = {key: stats([float(row[key]) for row in rows]) for key in STAGES}
    verification = json.loads((root / "final_verification.json").read_text())
    pixels = json.loads(
        (root / "pixel_verification/pixel_verification.json").read_text()
    )
    for result in pixels.values():
        if result["different_bits"] or result["stale_cache_pixels"]:
            raise ValueError("Pixel verification failed")
    paced = {}
    for name in [f"before_paced_{letter}" for letter in "AB"] + [
        f"final_paced_{letter}_{repeat}" for letter in "AB" for repeat in (1, 2, 3)
    ]:
        report = json.loads((root / name / "paced_report.json").read_text())
        writer = json.loads((root / name / "writer_report.json").read_text())
        if (
            report["status"] != "completed"
            or not writer["all_submitted_saved"]
            or writer["error"]
        ):
            raise ValueError(f"Run did not finish correctly: {name}")
        if writer["committed_frames"] != report["processed_frames"]:
            raise ValueError("Analyzed and saved frame counts differ")
        paced[name] = {"replay": report, "writer": writer}
    kernel = json.loads((root / "kernel_candidates/kernel_report.json").read_text())
    stability = json.loads((root / "stability_cache/paced_report.json").read_text())
    stability_writer = json.loads(
        (root / "stability_cache/writer_report.json").read_text()
    )
    if (
        stability["status"] != "completed"
        or stability_writer["pending_records"]
        or stability_writer["error"]
    ):
        raise ValueError("Continuous replay did not finish correctly")
    if stability["processed_frames"] != stability_writer["committed_frames"]:
        raise ValueError("Continuous replay has unsaved analyzed frames")
    memory_rows = read_rows(root / "stability_cache/memory.csv")
    elapsed = np.array([float(row["elapsed_s"]) for row in memory_rows])
    memory = np.array([float(row["consumer_rss_mb"]) for row in memory_rows])
    memory_summary = {
        "after_warmup_median_mb": float(
            np.median(memory[(elapsed >= 5) & (elapsed < 20)])
        ),
        "last_15s_median_mb": float(np.median(memory[elapsed >= elapsed[-1] - 15])),
        "peak_mb": float(memory.max()),
    }
    tests = (root / "final_tests.log").read_text().strip().splitlines()[-1]
    report = {
        "fixed": fixed,
        "paced": paced,
        "kernel": kernel,
        "pixel_verification": pixels,
        "scientific_verification": verification,
        "tests": tests,
        "stability": stability,
        "stability_writer": stability_writer,
        "memory": memory_summary,
    }
    (output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "stages.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "stage", "mean_ms", "median_ms", "p99_ms", "max_ms"])
        for name, stages in fixed.items():
            for stage, values in stages.items():
                writer.writerow([name, stage, *values.values()])
    draw(fixed, output, font)
    lines = [
        "# 相对暗度计算：减少重复操作",
        "",
        "本轮保留原数学公式、全图抽样中位数、阈值、背景更新规则和追踪规则。",
        "最终采用 NumPy 除法，缓存 B + 1、只刷新背景实际更新的行，并复用暗度和区域标号工作区。",
        "现有整段录像入口、逐帧接口和真实速度回放都已接入，运行命令不变。",
        "",
        "## 相同完整输入的耗时",
        "",
        "修改前 A、B 各完整处理一次；最终方案各完整处理三次。以下为每帧平均值，最终列取三次均值。",
        "视频解码、初始背景准备及最后表格导出不在处理计时内。异步保存方式沿用上一轮，不在本轮改动。",
        "",
        "| 视频 | 暗度：原版 ms | 暗度：最终 ms | 整帧：原版 ms | 整帧：最终 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for letter in "AB":
        before = fixed[f"before_fixed_{letter}"]
        after = [fixed[f"final_fixed_{letter}_{i}"] for i in (1, 2, 3)]
        lines.append(
            f"| {letter} | {before['seg_contrast_ms']['mean']:.3f} | "
            f"{np.mean([r['seg_contrast_ms']['mean'] for r in after]):.3f} | "
            f"{before['process_total_ms']['mean']:.3f} | "
            f"{np.mean([r['process_total_ms']['mean'] for r in after]):.3f} |"
        )
    before_bg = np.mean(
        [fixed[f"before_fixed_{letter}"]["background_ms"]["mean"] for letter in "AB"]
    )
    after_bg = np.mean(
        [
            fixed[f"final_fixed_{letter}_{i}"]["background_ms"]["mean"]
            for letter in "AB"
            for i in (1, 2, 3)
        ]
    )
    lines += [
        "",
        f"刷新缓存算在背景更新中：该阶段由 {before_bg:.3f} 变为 {after_bg:.3f} ms/帧。整帧时间包含这笔开销。",
        "各次均值、P99 和最大值见 stages.csv；图中的圆点为三次实际运行，柱高为均值。",
        "",
        "## 为什么采用这一个实现",
        "",
        "候选分别比较了原式、当帧原地运算、固定数组、缓存分母、缓存倒数、OpenCV 运算以及分行计算。",
        "小步骤测试会对同一画面反复计算，数据已在内存中，因此只用于筛候选，不能用它的耗时代替完整视频的测量。",
        "",
        "| 候选 | 小步骤均值 ms | 与原式不同的像素累计 | 默认阈值判断不同累计 |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "original": "原式",
        "numpy_inplace": "本帧暗度数组内完成减法",
        "numpy_buffers": "固定的分母、暗度数组",
        "numpy_cached_divide": "缓存 B + 1，保留除法",
        "numpy_cached_reciprocal": "缓存倒数，改为乘法",
        "opencv_buffers": "OpenCV 固定数组",
        "opencv_cached_divide": "OpenCV 缓存分母",
        "numpy_tiles_32": "每批计算 32 行",
        "numpy_tiles_64": "每批计算 64 行",
        "numpy_tiles_128": "每批计算 128 行",
    }
    for name, values in kernel["timing_ms"].items():
        check = kernel["checks"][name]
        lines.append(
            f"| {labels[name]} | {values['mean']:.3f} | {check['different_pixels']} | {check['threshold_differences']} |"
        )
    lines += [
        "",
        "乘倒数出现很小的浮点差异；这 24 张抽查画面在默认阈值下没有判断差异，但不满足逐像素一致要求，因此未采用。",
        "仅复用暗度数组时，暗度步骤变快，但后面的区域标号变慢，整帧退步。这些试验保存在 cached_fixed_*、allocated_fixed_A 和 denominator_fixed_A。",
        "最终给原来的 connectedComponentsWithStats 提供固定输出数组，消除了本次实测的标号耗时回退。",
        "函数、连通方式和区域判断规则保持不变；尚未确定早期回退在底层分配器或线程调度中的具体原因，不作推断性结论。",
        "",
        "## 是否改变识别结果",
        "",
        (
            f"两段原视频共 {sum(r['frames'] for r in pixels.values())} 帧、"
            f"{sum(r['pixels'] for r in pixels.values()):,} 个像素位置的暗度值，按浮点位模式逐一比较，差异为 0。"
        ),
        "每次背景更新后再核对整张 B + 1 缓存，过期像素为 0。",
        "三次完整回放分别核对轮廓字节、坐标、形状、速度、编号、事件及修订、源帧号与时间、最终背景和 PZT 对应文件，全部一致。",
        "科学比较只排除运行耗时、运行标识和代码版本指纹。没有重新人工标注，也不据此声称算法识别准确率提高。",
        f"自动检查：{tests}。覆盖阈值附近、非连续数组、浮点精度、非整除的背景分区、跨帧复用及此前保存/取消/失败检查。",
        "",
        "## 按真实速度回放",
        "",
        "| 运行 | 已分析并保存 / 输入 | 分析帧率 fps | 调用中位数 ms | 调用 P99 ms | 保存队列峰值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in paced.items():
        replay, writer = values["replay"], values["writer"]
        lines.append(
            f"| {name} | {replay['processed_frames']} / {replay['offered_frames']} | "
            f"{replay['processed_fps']:.2f} | {replay['metrics']['process_ms']['median']:.3f} | "
            f"{replay['metrics']['process_ms']['p99']:.3f} | {writer['peak_pending_records']} |"
        )
    lines += [
        "",
        "这是原录像按 60 fps 送入的 WSL 软件测试。未处理的源画面仍按既有规则明确记为替换或过期，不能说这些画面也完成了分析。",
        "所有已提交给保存线程的结果均完整保存，结束剩余队列为 0。没有连接相机或 AFG，也没有实测同时录像或显示界面的负载。",
        "",
        "## 缓存如何保持正确",
        "",
        "B + 1 在开始时算一次；OnlineBackground.update 每帧更新约十分之一的行后，立即刷新相同行的缓存。",
        "不换乘倒数，不跳过亮区，不裁剪视野；全图中位数采样和阈值保持原样。",
        "一个 1280×1024 float32 视频流固定使用约 15 MiB 工作区，三个数组不会按历史帧数增加。",
        (
            f"完整进程的实测峰值：修改前约 {max(v['replay']['memory_sampled_mb']['consumer_rss_mb']['max'] for n, v in paced.items() if n.startswith('before')):.1f} MB，"
            f"修改后约 {max(v['replay']['memory_sampled_mb']['consumer_rss_mb']['max'] for n, v in paced.items() if n.startswith('final')):.1f} MB。"
        ),
        (
            f"额外连续重复回放 {stability['nominal_duration_s']:.0f} 秒："
            f"分析并保存 {stability_writer['committed_frames']} / {stability['offered_frames']} 帧，"
            f"保存队列峰值 {stability_writer['peak_pending_records']} 条，结束为 0。"
        ),
        (
            f"内存中位数：运行第 5–20 秒 {memory_summary['after_warmup_median_mb']:.2f} MB，"
            f"最后 15 秒 {memory_summary['last_15s_median_mb']:.2f} MB。"
            "这是重复原录像的负载诊断，使用单独的虚拟流编号，不生成新的真实实验或 PZT 对应。"
        ),
        "自行修改背景时必须刷新相应行，更换背景或图像规格则创建新工作区；一个工作区只能由所属分析线程使用。",
        "完整原始数据、旧结果和失败候选均保留。所有图中文字为英文，字体使用实际 Arial。",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def draw(fixed, output, font):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(font))
    if font_manager.FontProperties(fname=str(font)).get_name() != "Arial":
        raise ValueError("Font must be Arial")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, key, label in zip(
        axes,
        ("seg_contrast_ms", "process_total_ms"),
        ("Relative darkness (ms/frame)", "Analysis call (ms/frame)"),
    ):
        for index, letter in enumerate("AB"):
            before = fixed[f"before_fixed_{letter}"][key]["mean"]
            after = [fixed[f"final_fixed_{letter}_{i}"][key]["mean"] for i in (1, 2, 3)]
            ax.bar(
                index - 0.18,
                before,
                width=0.30,
                color="#B75B66",
                label="Before" if index == 0 else None,
            )
            ax.bar(
                index + 0.18,
                np.mean(after),
                width=0.30,
                color="#829E63",
                label="After" if index == 0 else None,
            )
            ax.scatter(
                index + 0.18 + np.array([-0.06, 0, 0.06]),
                after,
                facecolors="white",
                edgecolors="#333333",
                s=22,
                zorder=3,
            )
        ax.set(xticks=[0, 1], xticklabels=["Exp. A", "Exp. B"], ylabel=label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    axes[0].legend(frameon=False, ncol=2, loc="upper center")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            output / f"contrast_comparison.{extension}", dpi=220, facecolor="white"
        )
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
