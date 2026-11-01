"""第二轮验收：逐帧核对分流前后数据，再比较同参数分析性能。"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from customer_workspace.streamcopy_recording import write_json


def check_branches(session: dict, probe: dict, recording: dict) -> dict:
    """同时检查 GStreamer 接收、录像队列及最终文件，不能只看文件能播放。"""
    problems = []
    frames = session["record_frames"]
    packets = [r for r in probe["packets_and_frames"] if r["type"] == "packet"]
    source_frames = []
    clock_error_ticks = None
    if not frames or not packets:
        problems.append("没有录像帧")
    else:
        source_frames = [
            r
            for r in session["input_frames"]
            if frames[0]["seq"] <= r["seq"] <= frames[-1]["seq"]
        ]
        source_ids = [r["seq"] for r in source_frames]
        selected_ids = [r["seq"] for r in frames]
        if source_ids != selected_ids:
            problems.append("录像窗口内接收到的画面没有全部进入录像支路")
        if not {5, 7, 8}.issubset(frames[0].get("nal_types", [])):
            problems.append("录像起点缺少完整解码头信息或 IDR 画面")
        if len(frames) != len(packets) or len(frames) != session["target_frames"]:
            problems.append("分流前、Submitted和文件的画面数量不一致")
        else:
            # MPEG-TS/MP4 使用 1/90000 秒单位，GStreamer 使用纳秒。
            # 只允许量化造成的 1 tick 差异；不改写任何输入时间。
            time_base = Fraction(probe["streams"][0]["time_base"])
            errors = []
            for source, packet in zip(frames, packets):
                source_delta = Fraction(source["pts_ns"] - frames[0]["pts_ns"], 10**9)
                expected = round(source_delta / time_base)
                actual = packet["pts"] - packets[0]["pts"]
                errors.append(abs(actual - expected))
            clock_error_ticks = max(errors)
            if clock_error_ticks > 1:
                problems.append("分流前与保存后的视频时间不能逐帧对应")
    if not recording["passed"]:
        problems.extend(recording["problems"])
    if session["errors"] or session["warnings"] or session["relay_errors"]:
        problems.append("接收器或控制连接存在错误/告警")
    if session["rtp_sequence_missing"]:
        problems.append("收到的 RTP 包编号存在缺口")
    if any(session["queue_overruns"].values()):
        problems.append("录像队列或解码队列曾达到容量上限")
    exchanges = session["rtsp_exchanges"]
    play_count = sum(row["method"] == "PLAY" for row in exchanges)
    expected_connections = 0 if session.get("input_kind") == "offline_video" else 1
    if play_count != expected_connections:
        problems.append("未能证明只使用一次相机连接")
    return {
        "passed": not problems,
        "problems": problems,
        "received_window_frames": len(source_frames),
        "record_branch_frames": len(frames),
        "saved_frames": len(packets),
        "recording_extra_lost": len(source_frames) - len(packets),
        "clock_comparison_max_error_ticks": clock_error_ticks,
        "camera_play_count": play_count,
        "rtp_sequence_missing": session["rtp_sequence_missing"],
        "queue_overruns": session["queue_overruns"],
        "queue_max_buffers": {
            name: max(
                (row[name + "_buffers"] for row in session["queue_levels"]), default=0
            )
            for name in ("record_queue", "decode_queue", "latest_queue")
        },
        "decoded_frames_total": len(session["decoded_frames"]),
    }


def compare_analysis(baseline: dict, parallel: dict, criteria: dict) -> dict:
    """验收门槛预先确定，不能为了让本次数据通过而临时改规则。"""
    problems = []
    baseline_rate, parallel_rate = baseline["analysis_hz"], parallel["analysis_hz"]
    if baseline.get("engine") != parallel.get("engine"):
        problems.append("对照与实验使用了不同分析程序，必须重新采集同配置对照")
    rate_ratio = parallel_rate / baseline_rate if baseline_rate else 0
    baseline_p95 = baseline["software_age_upper_bound_ms"]["p95"]
    parallel_p95 = parallel["software_age_upper_bound_ms"]["p95"]
    latency_limit = (
        max(baseline_p95 * 1.2, baseline_p95 + 20) if baseline_p95 is not None else None
    )
    if (
        parallel_rate < criteria["analysis_at_least_hz"]
        or rate_ratio < criteria["analysis_rate_ratio_to_baseline_min"]
    ):
        problems.append("分析速度低于预定要求")
    if parallel_p95 is None or latency_limit is None or parallel_p95 > latency_limit:
        problems.append("电脑端 P95 延迟增长超过预定要求")
    for name, result in (("baseline", baseline), ("parallel", parallel)):
        if (
            result["pipeline_error"]
            or result["recorder_dropped"]
            or not result["threads_closed"]
        ):
            problems.append(f"{name} 分析程序或结果保存存在异常")
        if (
            result.get("actual_tracking_results", result.get("actual_detector_and_piv_results", 0)) < 0.9 * result["processed"]
            or not result["processed"]
        ):
            problems.append(f"{name} 没有充分执行实际分析，不能用空转速度验收")
        gap = result["completion_interval_ms"]["max"]
        if gap is None or gap > criteria["maximum_analysis_completion_gap_ms"]:
            problems.append(f"{name} 分析出现超过预定上限的停顿")
    return {
        "passed": not problems,
        "problems": problems,
        "baseline_hz": baseline_rate,
        "parallel_hz": parallel_rate,
        "rate_ratio": rate_ratio,
        "p95_latency_limit_ms": latency_limit,
        "baseline": baseline,
        "parallel": parallel,
    }


def plot_comparison(output: Path, report: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    from customer_workspace.pzt_video_review import find_font

    font_manager.fontManager.addfont(str(find_font(None)))
    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    branches, analysis = report["recording"], report["analysis"]
    counts = [
        branches[k]
        for k in ("received_window_frames", "record_branch_frames", "saved_frames")
    ]
    axes[0].bar(["Received", "Submitted", "Saved"], counts, color="#388eae")
    axes[0].set_ylabel("Frames")
    axes[0].set_title("Recording integrity")
    for index, value in enumerate(counts):
        axes[0].text(index, value, str(value), ha="center", va="bottom")
    rates = [analysis["baseline_hz"], analysis["parallel_hz"]]
    axes[1].bar(["Analysis only", "Recording + analysis"], rates, color=["#999999", "#388eae"])
    axes[1].axhline(60 if analysis["parallel"].get("engine") else 7, color="orange", linestyle="--", label="Target rate")
    for index, value in enumerate(rates):
        axes[1].text(index, value, f"{value:.2f}", ha="center", va="bottom")
    axes[1].set_ylabel("Processed frames / s")
    axes[1].set_title("Analysis throughput")
    axes[1].legend()
    positions = np.arange(2)
    for shift, key, label, color in (
        (-0.18, "baseline", "Analysis only", "#999999"),
        (0.18, "parallel", "Recording + analysis", "#388eae"),
    ):
        values = [
            analysis[key]["software_age_upper_bound_ms"][k] for k in ("median", "p95")
        ]
        axes[2].bar(positions + shift, values, width=0.35, label=label, color=color)
        for x, value in zip(positions + shift, values):
            axes[2].text(x, value, f"{value:.1f}", ha="center", va="bottom")
    axes[2].set_xticks(positions, ["Median", "P95"])
    axes[2].set_ylabel("Software age upper bound (ms)")
    axes[2].set_title("Software latency")
    axes[2].legend()
    for axis in axes:
        axis.set_ylim(0, axis.get_ylim()[1] * 1.15)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Recording and analysis: " + ("Passed" if report["passed"] else "Not passed")
    )
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def create_report(
    baseline_dir: Path, parallel_dir: Path, criteria_path: Path, output: Path
):
    output.mkdir(parents=True, exist_ok=False)

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    recording = check_branches(
        read(parallel_dir / "gst_session.json"),
        read(parallel_dir / "recording/probe.json"),
        read(parallel_dir / "recording/recording_check.json"),
    )
    analysis = compare_analysis(
        read(baseline_dir / "analysis_check.json"),
        read(parallel_dir / "analysis_check.json"),
        read(criteria_path),
    )
    report = {
        "passed": recording["passed"] and analysis["passed"],
        "recording": recording,
        "analysis": analysis,
        "baseline_directory": str(baseline_dir.resolve()),
        "parallel_directory": str(parallel_dir.resolve()),
        "limitations": [
            "电脑端延迟不含无法测量的曝光及相机内部延迟。",
            "同一参数、相近时间各测试20秒；不将本次速度差直接归因于分流。",
            "分析有效性门槛保持原样；吞吐量不代表有效流速、检测准确率或闭环控制已经通过。",
        ],
    }
    write_json(output / "comparison.json", report)
    plot_comparison(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--parallel", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = create_report(args.baseline, args.parallel, args.criteria, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
