"""汇总正式链路的分析分项、慢帧、CPU/调度与自动清理；只读取已有数据。"""

import argparse
import sqlite3
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from customer_workspace.streamcopy_recording import write_json
from customer_workspace.tools.rtsp_tracking_report import read_json, read_rows
from customer_workspace.tools.tracking_latency_report import save_csv, stats

GROUPS = {
    "input": "input",
    "frame_setup": "setup",
    "blur": "smoothing",
    "contrast": "darkness",
    "candidates": "regions",
    "components": "regions",
    "selection": "regions",
    "find_contours": "contours",
    "draw_contours": "contours",
    "filter_areas": "contours",
    "tracking": "tracking",
    "background": "background",
    "submit": "record_submit",
    "tail": "return",
}


def collect_run(folder):
    """所有阶段连续相加；GC 是阶段内部重叠时间，不能再重复加一次。"""
    stages = read_rows(folder / "consumer_stalls.csv")
    frames = {
        int(row["source_frame"]): row
        for row in read_rows(folder / "analysis_frames.csv")
    }
    gc_rows = read_rows(folder / "consumer_stalls.gc.csv")
    metadata = read_json(folder / "consumer_stalls.meta.json")
    tid = metadata["native_thread_id"]
    scheduler_available = (metadata.get("scheduler_statistics") or {}).get("enabled")
    # 早期探针未保存开关；本次额外保存的内核配置证明零值不可用。
    availability = folder.parent / "scheduler_availability.json"
    if availability.exists() and "# CONFIG_SCHEDSTATS is not set" in read_json(
        availability
    ).get("kernel_config", []):
        scheduler_available = False
    grouped = defaultdict(list)
    for row in stages:
        if int(row["frame"]) in frames:
            grouped[int(row["frame"])].append(row)
    output, measured = [], []
    for frame, spans in grouped.items():
        source = frames[frame]
        start, end = float(spans[0]["start_s"]), float(spans[-1]["end_s"])
        groups = defaultdict(float)
        for span in spans:
            groups[GROUPS[span["stage"]]] += float(span["wall_ms"])
        gc_overlap = 0.0
        gc_events = []
        for event in gc_rows:
            if int(event["thread_id"]) != tid:
                continue
            part = (
                max(
                    0.0,
                    min(end, float(event["end_s"]))
                    - max(start, float(event["start_s"])),
                )
                * 1000
            )
            if part:
                gc_overlap += part
                gc_events.append({**event, "overlap_ms": part})
        row = {
            "source_frame": frame,
            "video_frame": int(source["video_frame"]),
            "start_s": start,
            "end_s": end,
            "wall_ms": (end - start) * 1000,
            "external_call_ms": float(source["processing_call_ms"]),
            "gc_overlap_ms": gc_overlap,
            **{key + "_ms": groups[key] for key in dict.fromkeys(GROUPS.values())},
        }
        for key in (
            "thread_cpu_ms",
            "process_cpu_ms",
            "thread_user_ms",
            "thread_system_ms",
            "runnable_wait_ms",
            "minor_faults",
            "major_faults",
            "voluntary_switches",
            "involuntary_switches",
        ):
            values = [float(span[key]) for span in spans if span[key] != ""]
            row[key] = sum(values) if len(values) == len(spans) else None
        if scheduler_available is not True:
            row["runnable_wait_ms"] = None
        if abs(row["wall_ms"] - sum(float(span["wall_ms"]) for span in spans)) > 0.001:
            raise ValueError("Stage times do not add up")
        if any(a["end_s"] != b["start_s"] for a, b in pairwise(spans)):
            raise ValueError("Stage timing is not contiguous")
        output.append(row)
        measured.append({**row, "stages": spans, "gc_events": gc_events})
    if set(grouped) != frames.keys():
        raise ValueError("Stage records do not cover all analyzed frames")
    medians = {
        stage: float(
            np.median(
                [
                    float(row["wall_ms"])
                    for row in stages
                    if row["stage"] == stage and int(row["frame"]) in frames
                ]
            )
        )
        for stage in GROUPS
    }
    for row in measured:
        row["excess_by_stage_ms"] = {
            stage["stage"]: float(stage["wall_ms"]) - medians[stage["stage"]]
            for stage in row["stages"]
        }
    slow = [row for row in measured if row["external_call_ms"] > 1000 / 60]
    return {
        "frames": len(output),
        "scheduler_statistics_available": scheduler_available,
        "stage_medians_ms": medians,
        "groups": {
            key: stats([row[key] for row in output if row[key] is not None])
            if any(row[key] is not None for row in output)
            else None
            for key in output[0]
            if key.endswith("_ms")
        },
        "slow": slow,
        "first_frame": measured[0],
        "gc_events_in_analysis": sum(bool(row["gc_events"]) for row in measured),
    }, output


def replay_comparison(root, summaries):
    """比较相同输入历史的核心计算时间；不是相同负载的实时性能对照。"""
    output = []
    for name, summary in summaries.items():
        folder = root / name
        if not read_json(folder / "equivalence.json")["scientific_results_equal"]:
            raise ValueError("Scientific outputs differ")
        actual = sqlite3.connect(
            f"file:{folder / 'tracking/tracking.sqlite'}?mode=ro", uri=True
        )
        replay = sqlite3.connect(
            f"file:{folder / 'verification_reference/tracking.sqlite'}?mode=ro",
            uri=True,
        )
        try:
            for row in summary["slow"]:
                query = "SELECT processing_ms, active_tracks FROM frames WHERE source_frame=?"
                a = actual.execute(query, (row["source_frame"],)).fetchone()
                b = replay.execute(query, (row["source_frame"],)).fetchone()
                output.append(
                    {
                        "run": name,
                        "source_frame": row["source_frame"],
                        "original_frame": row.get("original_frame"),
                        "realtime_core_ms": a[0],
                        "replay_core_ms": b[0],
                        "active_tracks": a[1],
                        "replay_active_tracks": b[1],
                    }
                )
        finally:
            actual.close()
            replay.close()
    return output


def write_charts(output, summaries):
    """画出已测慢帧的时间构成；使用实际 Arial 字体。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/mnt/c/Windows/Fonts/arial.ttf")
    if not font.exists():
        raise FileNotFoundError("Arial is required")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "font.size": 10})
    selected = sorted(
        [(name, row) for name, run in summaries.items() for row in run["slow"]],
        key=lambda x: x[1]["external_call_ms"],
        reverse=True,
    )[:12]
    if not selected:
        return
    groups = [
        ("input_ms", "Input", "#B6C9DB"),
        ("smoothing_ms", "Smoothing", "#C8E0D8"),
        ("darkness_ms", "Darkness", "#E1C48C"),
        ("regions_ms", "Regions", "#82A585"),
        ("contours_ms", "Contours", "#E6A28F"),
        ("tracking_ms", "Tracking", "#B75B66"),
        ("record_submit_ms", "Submit", "#AA99B7"),
    ]
    fig, ax = plt.subplots(figsize=(10, max(4, len(selected) * 0.36 + 1)))
    left = np.zeros(len(selected))
    for key, label, color in groups:
        values = np.array([row[key] for _, row in selected])
        ax.barh(range(len(selected)), values, left=left, color=color, label=label)
        left += values
    other = np.array([row["wall_ms"] for _, row in selected]) - left
    ax.barh(range(len(selected)), other, left=left, color="#D5D5D5", label="Other")
    ax.axvline(1000 / 60, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(
        range(len(selected)),
        [f"{name} · frame {row['video_frame']}" for name, row in selected],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Analysis time (ms)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output / "slow_frames.png", dpi=220)
    fig.savefig(output / "slow_frames.pdf")
    plt.close(fig)


def build(root, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = read_json(root / "inputs.json")["runs"]
    summaries = {}
    for name, row in inputs.items():
        if not row["diagnose_analysis"]:
            continue
        summaries[name], rows = collect_run(root / name)
        mapping_file = root / "integrity_report" / f"{name}_source_mapping.csv"
        if mapping_file.exists():
            mapping = {
                int(r["source_frame"]): int(r["original_frame"])
                for r in read_rows(mapping_file)
            }
            for frame in rows:
                frame["original_frame"] = mapping[frame["source_frame"]]
            for frame in summaries[name]["slow"]:
                frame["original_frame"] = mapping[frame["source_frame"]]
        save_csv(output / f"{name}_frames.csv", rows)
    write_json(output / "summary.json", summaries)
    comparisons = replay_comparison(root, summaries)
    save_csv(output / "same_input_replay.csv", comparisons)
    write_charts(output, summaries)
    lines = [
        "# 微泡分析内部慢帧诊断",
        "",
        "本轮解释见 [主要结论](../FINDINGS.md)，下表保留可核对的逐帧数据。",
        "",
        "单帧等待、原算法、原线程数；F 盘 RTSP 模拟输入。普通对照 A/B 各一次，分项诊断各三次，其中最后一次另带 Linux 与 Windows 系统观察。",
        "",
        "## 分析耗时",
        "",
        "| 组 | 分析帧数 | 中位 ms | P99 ms | 最慢 ms | >16.67 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in inputs:
        rows = read_rows(root / name / "analysis_frames.csv")
        times = [float(row["processing_call_ms"]) for row in rows]
        v = stats(times)
        lines.append(
            f"| {name} | {len(rows)} | {v['median']:.3f} | {v['p99']:.3f} | {v['max']:.3f} | {sum(t > 1000 / 60 for t in times)} |"
        )
    lines += [
        "",
        "## 每个慢帧",
        "",
        "以下列出相对于本组各步骤中位值，增加最多的步骤。GC 已包含在步骤时间里，不能再重复相加。主线程 CPU 时间不包括 OpenCV 其他工作线程，也不能单凭接近墙钟就排除 Windows 宿主影响。",
        "",
        "| 组 / 接收帧 | 原视频帧 | 总调用 ms | 增量最大步骤 | 该步骤 ms | 主线程 CPU ms | 等待 CPU ms | GC 重叠 ms |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in summaries.items():
        for row in summary["slow"]:
            worst = max(row["excess_by_stage_ms"], key=row["excess_by_stage_ms"].get)
            span = next(s for s in row["stages"] if s["stage"] == worst)
            lines.append(
                f"| {name} / {row['source_frame']} | {row.get('original_frame', '')} | {row['external_call_ms']:.3f} | {worst} | {float(span['wall_ms']):.3f} | {row['thread_cpu_ms']:.3f} | {row['runnable_wait_ms'] if row['runnable_wait_ms'] is not None else '不可用'} | {row['gc_overlap_ms']:.3f} |"
            )
    lines += [
        "",
        "## 数据解释",
        "",
        "- smoothing：平滑；darkness：相对暗度；regions：候选阈值、闭运算、连通区域和保留区域；contours：找轮廓、填充和按面积筛选；tracking：目标匹配与轨迹；record_submit：整理本帧记录并提交独立保存线程。",
        "- input 包含输入检查和 uint8→float32；setup、background、return 分别为帧信息准备、背景更新、结果整理。不能只统计用户列出的六项而丢掉这些时间。",
        "- 分项记录保持连续；其中包括前一步计时记账的少量成本。probe 的详细扫描关闭，日志按批保存。开启诊断会改变运行负担和 GC 时机，不能把诊断运行当作正式性能验收。",
        "- Linux minor_faults 是内存页首次映射等软缺页，并不等同于硬盘读取。major_faults 为零也不能排除 Windows 为 WSL 回读页文件。",
        "- 本机内核 CONFIG_SCHEDSTATS 未开启，原始 schedstat 零值无效；报告将等待 CPU 时间标为不可用，不能据此断言没有调度等待。",
        "- Windows 与 WSL 通过请求/回复区间对应时间，没有直接相减原始时钟；Windows 窗口的关联误差必须另见 host_report。",
        "- 录像像素、帧去向、数据库及同输入离线重算结果见 ../integrity_report/summary.json。",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "REPORT.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
