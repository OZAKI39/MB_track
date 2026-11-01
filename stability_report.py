"""第三轮结果核对：完整录像、前后段延迟、队列与两端内存趋势。"""

import argparse
import csv
import gzip
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from customer_workspace.parallel_video_report import check_branches
from customer_workspace.streamcopy_recording import write_json


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def series_stats(times, values, seconds: float) -> dict:
    """报告首尾十秒及后半程趋势，避免把初始化的内存分配当成持续积压。"""
    pairs = [
        (t, v) for t, v in zip(times, values) if 0 <= t < seconds and v is not None
    ]
    if not pairs:
        return {"samples": 0}
    times, values = map(np.asarray, zip(*pairs))
    first = values[times < 10]
    last = values[times >= seconds - 10]
    late = times >= seconds / 2
    late_start = values[(times >= seconds / 2) & (times < seconds / 2 + 10)]
    if not len(first) or not len(last) or not len(late_start):
        return {"samples": len(values), "incomplete": True}
    return {
        "samples": len(values),
        "first_10s_median": float(np.median(first)),
        "last_10s_median": float(np.median(last)),
        "first_10s_mean": float(np.mean(first)),
        "last_10s_mean": float(np.mean(last)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "tail_change": float(np.median(last) - np.median(first)),
        "late_change": float(np.median(last) - np.median(late_start)),
        "slope_per_second": float(np.polyfit(times, values, 1)[0]),
        "late_slope_per_second": float(np.polyfit(times[late], values[late], 1)[0]),
    }


def memory_problems(
    name: str, stats: dict, seconds: float, criteria: dict
) -> list[str]:
    """小幅记录数据增长与持续增长分开；缺少监测数据不能自动通过。"""
    if stats["samples"] < seconds * criteria[
        "monitor_min_samples_per_second"
    ] or stats.get("incomplete"):
        return [f"{name} 内存记录不足"]
    sustained = (
        stats["late_change"] > criteria["memory_late_growth_limit_mib"]
        and stats["late_slope_per_second"] > criteria["memory_late_slope_limit_mib_s"]
    )
    if sustained or stats["tail_change"] > criteria["memory_total_growth_limit_mib"]:
        return [f"{name} 内存有需要排查的持续增长"]
    return []


def check_run(folder: Path, criteria: dict) -> dict:
    """复用第二轮逐帧核对，再检查本轮保存的时间序列。"""
    seconds = criteria["seconds"]
    session = read_json(folder / "gst_session.json")
    summary = read_json(folder / "summary.json")
    analysis = read_json(folder / "analysis_check.json")
    origin = read_json(folder / "runtime_origin.json")
    recording = check_branches(
        session,
        read_json(folder / "recording/probe.json"),
        read_json(folder / "recording/recording_check.json"),
    )
    problems = list(recording["problems"])
    if criteria.get("analysis_engine") and criteria["analysis_engine"] != analysis.get("engine"):
        problems.append("分析程序与本次验收配置不符")
    if session["seconds"] != seconds or session["target_frames"] != round(seconds * 60):
        problems.append("实际录像目标与本轮实验长度不符")
    frames = session["record_frames"]
    first_pts = frames[0]["pts_ns"]
    windows_start = frames[0]["received_windows_mono"]
    frame_times = {r["seq"]: (r["pts_ns"] - first_pts) / 1e9 for r in frames}
    if analysis.get("engine") == "microbubble_tracking.OnlineProcessor":
        with (folder / "analysis_frames.csv").open(newline="") as file:
            states = list(csv.DictReader(file))
        times = np.array([float(r["video_time_s"]) for r in states])
        ages = np.array([float(r["software_age_upper_bound_ms"]) for r in states])
    else:
        # 保留旧报告的读取能力，不再调用旧分析程序。
        with gzip.open(folder / "states.jsonl.gz", "rt", encoding="utf-8") as file:
            states = [json.loads(line) for line in file]
        states = [r for r in states if r["frame_id"] in frame_times]
        times = np.array([frame_times[r["frame_id"]] for r in states])
        ages = np.array([r["timings"]["end_to_end_software_ms"] for r in states])
    first_ages, last_ages = ages[times < 20], ages[times >= seconds - 20]
    if not len(first_ages) or not len(last_ages):
        raise ValueError("没有足够的前后段分析记录")
    first_rate, last_rate = len(first_ages) / 20, len(last_ages) / 20
    latency = {}
    for label, percentile in (("median", 50), ("p95", 95)):
        first, last = map(
            float,
            (
                np.percentile(first_ages, percentile),
                np.percentile(last_ages, percentile),
            ),
        )
        limit = max(
            first * criteria["latency_tail_ratio_max"],
            first + criteria["latency_tail_increase_ms"],
        )
        latency[label] = {"first_20s_ms": first, "last_20s_ms": last, "limit_ms": limit}
        if last > limit:
            problems.append(f"后20秒的分析延迟 {label} 明显高于前20秒")
    if analysis["analysis_hz"] < criteria["analysis_min_hz"]:
        problems.append("整段分析速度低于约定要求")
    if last_rate < first_rate * criteria["analysis_tail_rate_ratio_min"]:
        problems.append("后20秒分析速度下降超过约定比例")
    if (
        analysis["completion_interval_ms"]["max"] is None
        or analysis["completion_interval_ms"]["max"] > criteria["completion_gap_max_ms"]
    ):
        problems.append("分析出现超过约定上限的停顿")
    if analysis.get("actual_tracking_results", analysis.get("actual_detector_and_piv_results", 0)) < 0.9 * analysis["processed"]:
        problems.append("没有充分执行实际分析，不能用空转速度验收")
    if (
        analysis["pipeline_error"]
        or analysis["recorder_dropped"]
        or not analysis["threads_closed"]
    ):
        problems.append("分析程序、记录或线程收尾存在异常")
    queues = {}
    for name in ("record_queue", "decode_queue"):
        stats = series_stats(
            [r["windows_mono"] - windows_start for r in session["queue_levels"]],
            [r[name + "_buffers"] for r in session["queue_levels"]],
            seconds,
        )
        queues[name] = stats
        if stats["samples"] < seconds * 5 or stats.get("incomplete"):
            problems.append(f"{name} 队列记录不足")
        elif (
            stats["last_10s_mean"] - stats["first_10s_mean"]
            > criteria["queue_tail_increase_frames"]
            and stats["slope_per_second"] > criteria["queue_slope_max_frames_s"]
        ):
            problems.append(f"{name} 有持续增长的等待数据")
    memory = {}
    plot_data = {"analysis_t": times.tolist(), "analysis_age_ms": ages.tolist()}
    for key in (
        "receiver_rss_mib",
        "receiver_private_mib",
        "ffmpeg_rss_mib",
        "ffmpeg_private_mib",
    ):
        rows = session["memory_samples"]
        mt = [r["windows_mono"] - windows_start for r in rows]
        mv = [r.get(key) for r in rows]
        memory[key] = series_stats(mt, mv, seconds)
        problems.extend(memory_problems(key, memory[key], seconds, criteria))
        plot_data[key] = {"t": mt, "values": mv}
    # WSL 的开始标记有约0.1秒量级传递误差，只作内存趋势的粗对齐。
    offset = origin["recording_start_observed_wsl_s"] - origin["pipeline_start_wsl_s"]
    mt = [r["elapsed_s"] - offset for r in summary["resources"]]
    mv = [r["rss_mib"] for r in summary["resources"]]
    memory["wsl_rss_mib"] = series_stats(mt, mv, seconds)
    problems.extend(memory_problems("WSL", memory["wsl_rss_mib"], seconds, criteria))
    plot_data["wsl_rss_mib"] = {"t": mt, "values": mv}
    if any("decoder_rss_mib" in row for row in summary["resources"]):
        values = [row.get("decoder_rss_mib") for row in summary["resources"]]
        memory["decoder_rss_mib"] = series_stats(mt, values, seconds)
        problems.extend(memory_problems("WSL decoder", memory["decoder_rss_mib"], seconds, criteria))
        plot_data["decoder_rss_mib"] = {"t": mt, "values": values}
    plot_data["queues"] = [
        {
            "t": r["windows_mono"] - windows_start,
            "record": r["record_queue_buffers"],
            "decode": r["decode_queue_buffers"],
        }
        for r in session["queue_levels"]
    ]
    bins = []
    # 最后12秒合在一段，不用只有2秒的小样本夸大波动。
    edges = [*range(0, int(seconds) - 9, 10), seconds]
    for start, end in pairwise(edges):
        selected = (times >= start) & (times < end)
        count = int(np.sum(selected))
        row = {
            "start_s": start,
            "end_s": end,
            "processed": count,
            "analysis_hz": count / (end - start),
            "age_median_ms": float(np.median(ages[selected])) if count else None,
            "age_p95_ms": float(np.percentile(ages[selected], 95)) if count else None,
        }
        bins.append(row)
        if row["analysis_hz"] < criteria["analysis_min_hz"]:
            problems.append(f"{start}–{end}秒分析速度低于要求")
    return {
        "passed": not problems,
        "problems": problems,
        "recording": recording,
        "analysis": analysis,
        "first_20s_hz": first_rate,
        "last_20s_hz": last_rate,
        "latency_comparison": latency,
        "queues": queues,
        "memory": memory,
        "bins": bins,
        "plot_data": plot_data,
    }


def plot_runs(output: Path, runs: list[dict], seconds: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    from customer_workspace.pzt_video_review import find_font

    font_manager.fontManager.addfont(str(find_font(None)))
    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for number, run in enumerate(runs, 1):
        if "plot_data" not in run:
            continue
        data = run["plot_data"]
        centers = [(r["start_s"] + r["end_s"]) / 2 for r in run["bins"]]
        label = f"Run {number}"
        axes[0].plot(
            centers, [r["analysis_hz"] for r in run["bins"]], "o-", label=label
        )
        axes[1].plot(centers, [r["age_p95_ms"] for r in run["bins"]], "o-", label=label)
        axes[2].plot(
            [r["t"] for r in data["queues"]],
            [r["record"] for r in data["queues"]],
            label=label,
            alpha=0.8,
        )
        for key, style, name in (
            ("wsl_rss_mib", "-", "WSL analysis"),
            ("receiver_rss_mib", "--", "Windows receiver"),
            ("ffmpeg_rss_mib", ":", "FFmpeg"),
        ):
            row = data[key]
            axes[3].plot(
                row["t"], row["values"], linestyle=style, label=f"{label} {name}"
            )
    for ax, title in zip(
        axes,
        (
            "Processed frames / s",
            "Software age P95 (ms)",
            "Recording queue (frames)",
            "Resident memory (MiB)",
        ),
    ):
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(ncol=4, fontsize=8)
        ax.set_xlim(0, seconds)
    axes[0].axhline(60 if any(r.get("analysis", {}).get("engine") for r in runs) else 7, color="grey", linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Time from recording start (s)")
    fig.suptitle(f"Recording and tracking: {len(runs)} runs of {seconds:g} s")
    fig.tight_layout()
    fig.savefig(output / "stability.png", dpi=150)
    plt.close(fig)


def create_report(series_dir: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    criteria = read_json(series_dir / "acceptance_criteria.json")
    outcomes = read_json(series_dir / "series.json")
    runs = []
    for outcome in outcomes:
        if not outcome["capture_completed"]:
            runs.append({"passed": False, "problems": [outcome["error"]]})
            continue
        try:
            runs.append(check_run(Path(outcome["directory"]), criteria))
        except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
            runs.append(
                {
                    "passed": False,
                    "problems": [f"核对失败：{type(exc).__name__}: {exc}"],
                }
            )
    problems = []
    if len(outcomes) != criteria["runs"] or len(outcomes) < 4:
        problems.append("没有完成首次加至少三次重复")
    if len({r["wsl_pid"] for r in outcomes}) != 1:
        problems.append("未在同一个 WSL 主进程中完成重复启动检查")
    after = [r["wsl_after_rss_mib"] for r in outcomes]
    if (
        after
        and after[-1] - after[0] > criteria["memory_total_growth_limit_mib"]
        and all(b > a for a, b in pairwise(after))
    ):
        problems.append("各次结束后 WSL 内存持续增长，需要进一步检查")
    report = {
        "passed": all(r["passed"] for r in runs) and not problems,
        "problems": problems,
        "criteria": criteria,
        "runs": runs,
        "series_outcomes": outcomes,
        "limitations": [
            "只证明本次多段62秒运行，不能保证任意长时间运行不泄漏。",
            "内存是各进程RSS/Windows私有内存，不包括所有操作系统缓存。",
            "电脑端延迟不含曝光和相机内部时间；内存对齐有开始标记传递误差。",
            "保持原分析有效性标准，运行速度不代表有效流速或闭环控制通过。",
        ],
    }
    write_json(output / "stability.json", report)
    with (output / "time_bins.csv").open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run",
                "start_s",
                "end_s",
                "processed",
                "analysis_hz",
                "age_median_ms",
                "age_p95_ms",
            ],
        )
        writer.writeheader()
        for number, run in enumerate(runs, 1):
            writer.writerows({"run": number, **row} for row in run.get("bins", []))
    with (output / "summary.csv").open("x", newline="", encoding="utf-8-sig") as file:
        fields = [
            "run",
            "passed",
            "received_frames",
            "saved_frames",
            "extra_lost_frames",
            "analysis_hz",
            "latency_median_ms",
            "latency_p95_ms",
            "wsl_rss_peak_mib",
            "receiver_rss_peak_mib",
            "ffmpeg_rss_peak_mib",
            "wsl_rss_tail_change_mib",
            "receiver_rss_tail_change_mib",
            "record_queue_peak_frames",
            "decode_queue_peak_frames",
            "problems",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for number, run in enumerate(runs, 1):
            row = {
                "run": number,
                "passed": run["passed"],
                "problems": "; ".join(run["problems"]),
            }
            if "analysis" in run:
                recording, analysis, memory = (
                    run["recording"],
                    run["analysis"],
                    run["memory"],
                )
                row.update(
                    received_frames=recording["received_window_frames"],
                    saved_frames=recording["saved_frames"],
                    extra_lost_frames=recording["recording_extra_lost"],
                    analysis_hz=analysis["analysis_hz"],
                    latency_median_ms=analysis["software_age_upper_bound_ms"]["median"],
                    latency_p95_ms=analysis["software_age_upper_bound_ms"]["p95"],
                    wsl_rss_peak_mib=memory["wsl_rss_mib"].get("max"),
                    receiver_rss_peak_mib=memory["receiver_rss_mib"].get("max"),
                    ffmpeg_rss_peak_mib=memory["ffmpeg_rss_mib"].get("max"),
                    wsl_rss_tail_change_mib=memory["wsl_rss_mib"].get("tail_change"),
                    receiver_rss_tail_change_mib=memory["receiver_rss_mib"].get(
                        "tail_change"
                    ),
                    record_queue_peak_frames=run["queues"]["record_queue"].get("max"),
                    decode_queue_peak_frames=run["queues"]["decode_queue"].get("max"),
                )
            writer.writerow(row)
    plot_runs(output, runs, criteria["seconds"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = create_report(args.series_dir, args.output_dir)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "problems": result["problems"],
                "runs": [
                    {"passed": r["passed"], "problems": r["problems"]}
                    for r in result["runs"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(0 if result["passed"] else 1)
