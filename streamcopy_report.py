"""离线核对独立录像：收到的压缩帧、实际保存帧以及原始播放时间。

FFmpeg 9.0.2 的 debug_ts 和总数相互核对。日志格式不认识时判为失败，
不能把少解析的记录当成零丢帧。网络之前的相机采集帧数不在此计数范围。
"""

import argparse
import csv
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from customer_workspace.streamcopy_recording import write_json

WARNING_PATTERN = re.compile(
    r"error|corrupt|missed|\blost\b|dropp|conceal|max delay|jitter buffer full|"
    r"non.monoton|invalid|timed out|overrun|could not|failed",
    re.IGNORECASE,
)


def read_log_packets(log: str) -> dict:
    """读取三处视频时间：原始输入、归零后的输入、交给文件封装的输出。"""
    groups = {"source": [], "received": [], "muxed": []}
    markers = {
        "demuxer ->": "source",
        "demuxer+ffmpeg ->": "received",
        "muxer <-": "muxed",
    }
    for line in log.splitlines():
        for marker, group in markers.items():
            if marker not in line:
                continue
            fields = dict(re.findall(r"(\w+):([^\s]+)", line.split(marker, 1)[1]))
            prefix = "" if group == "muxed" else "pkt_"
            try:
                if all(
                    fields[prefix + key] == "NOPTS"
                    for key in ("pts", "dts", "pts_time")
                ):
                    groups[group].append({"pts": None, "dts": None, "pts_s": None})
                    continue
                groups[group].append(
                    {
                        "pts": int(fields[prefix + "pts"]),
                        "dts": int(fields[prefix + "dts"]),
                        "pts_s": float(fields[prefix + "pts_time"]),
                    }
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"不能完整解析 {group} 时间记录：{line}") from exc
    for group, pattern in (
        ("source", r"Input stream #0:0 \(video\): (\d+) packets read"),
        ("muxed", r"Output stream #0:0 \(video\): (\d+) packets muxed"),
    ):
        counts = re.findall(pattern, log)
        if len(counts) != 1 or int(counts[0]) != len(groups[group]):
            raise ValueError(f"{group} 逐帧日志与 FFmpeg 汇总总数不一致或汇总缺失")
    if len(groups["received"]) != len(groups["source"]):
        raise ValueError("归零前后的输入日志数量不一致")
    offsets = set()
    for source, received in zip(groups["source"], groups["received"]):
        if source["pts"] is None and received["pts"] is None:
            continue
        if source["pts"] is None or received["pts"] is None:
            raise ValueError("原始时间缺失被改写，不能确认原始播放时间")
        offsets.add(source["pts"] - received["pts"])
        offsets.add(source["dts"] - received["dts"])
    if len(offsets) != 1:
        raise ValueError("输入时间发生非常量改写，不能确认原始播放间隔")
    return groups


def compare_packets(
    received: list[dict],
    muxed: list[dict],
    saved: list[dict],
    time_base: Fraction,
    seconds: float,
    output_seek_s: float = 0.0,
) -> dict:
    """按整数时间标识逐帧比较，单列边界外读入的帧；不只比较总数。"""
    if not muxed or not saved or muxed[0]["pts"] is None:
        raise ValueError("没有有效的首个保存帧")
    for row in received + muxed:
        if row["pts"] is None:
            continue
        if abs(float(row["pts"] * time_base) - row["pts_s"]) > 0.0001:
            raise ValueError("输入、封装与文件的时间单位不一致，不能直接核对")
    # 第一个可独立解码的保存帧才是录像零点。启动等待包不能算作录像帧，
    # 但仍逐项报告数量。MP4 会整体平移到零，核对时恢复这个常量偏移。
    seek_ticks = Fraction(str(output_seek_s)) / time_base
    if seek_ticks.denominator != 1:
        raise ValueError("启动偏移不能用视频时间单位精确表示")
    aligned_muxed = [
        {"pts": row["pts"] + int(seek_ticks), "dts": row["dts"] + int(seek_ticks)}
        for row in muxed
    ]
    start_pts = aligned_muxed[0]["pts"]
    first_index = next(
        (i for i, r in enumerate(received) if r["pts"] == start_pts), None
    )
    if first_index is None:
        raise ValueError("首个保存帧在接收记录中不存在")
    if any(r["pts"] is None for r in received[first_index:]):
        raise ValueError("录像开始后出现缺失时间的输入包")
    valid = [r for r in received if r["pts"] is not None]
    # 实测来源时钟可偏差一个 tick（1/90000 秒）。结束边界取最后一帧
    # 的真实结束时间；是否覆盖请求时长另行严格检查，不能把边界后读入
    # 的一帧因这 11 微秒的差异误判为丢失，也不能据此掩盖录像过短。
    window_duration = Fraction(str(seconds))
    if "duration" in saved[-1]:
        if int(saved[-1]["duration"]) <= 0:
            raise ValueError("末帧缺少有效显示时长，无法确定实际录像边界")
        window_duration = (
            saved[-1]["pts"] - saved[0]["pts"] + int(saved[-1]["duration"])
        ) * time_base
    stop = start_pts * time_base + window_duration
    inside = [r for r in valid if start_pts * time_base <= r["pts"] * time_base < stop]
    before = sum(r["pts"] < start_pts for r in valid)
    unknown = len(received) - len(valid)
    after = len(valid) - len(inside) - before
    file_offset = start_pts - saved[0]["pts"]
    aligned_saved = [
        {"pts": r["pts"] + file_offset, "dts": r["dts"] + file_offset} for r in saved
    ]

    def keys(rows):
        return Counter((int(row["pts"]), int(row["dts"])) for row in rows)

    expected, submitted, actual = keys(inside), keys(aligned_muxed), keys(aligned_saved)
    return {
        "received_total_packets": len(received),
        "received_window_frames": len(inside),
        "before_window_packets": before,
        "startup_untimed_packets": unknown,
        "after_window_packets": after,
        "first_mux_pts": start_pts,
        "output_seek_ticks": int(seek_ticks),
        "actual_window_duration_s": float(window_duration),
        "file_timestamp_offset_ticks": file_offset,
        "muxed_packets": len(muxed),
        "saved_packets": len(saved),
        "missing_in_saved": sum((expected - actual).values()),
        "unexpected_in_saved": sum((actual - expected).values()),
        "input_to_mux_missing": sum((expected - submitted).values()),
        "mux_to_file_missing": sum((submitted - actual).values()),
        "ordered_timestamps_match": (
            [(r["pts"], r["dts"]) for r in inside]
            == [(r["pts"], r["dts"]) for r in aligned_muxed]
            == [(r["pts"], r["dts"]) for r in aligned_saved]
        ),
    }


def check_timestamps(
    frames: list[dict], time_base: Fraction, seconds: float, nominal_fps: Fraction
) -> tuple[dict, list[dict]]:
    """检查视频实际播放时间，不按标称帧率补时间或补画面。"""
    if len(frames) < 2 or nominal_fps <= 0:
        raise ValueError("有效帧或标称帧率不足，无法检查时间")
    times = [float(int(frame["pts"]) * time_base) for frame in frames]
    intervals_ms = np.diff(times) * 1000
    expected_ms = 1000 / float(nominal_fps)
    gap_limit_ms = 1.5 * expected_ms
    final_duration = float(int(frames[-1].get("duration", 0)) * time_base)
    end_s = times[-1] + final_duration
    table = []
    for index, (frame, timestamp) in enumerate(zip(frames, times)):
        table.append(
            {
                "录像帧号": index + 1,
                "原始播放时间单位": int(frame["pts"]),
                "播放时间_秒": timestamp,
                "距上一帧_毫秒": None if index == 0 else float(intervals_ms[index - 1]),
                "关键帧": frame["key_frame"],
            }
        )
    result = {
        "nominal_fps": float(nominal_fps),
        "first_frame_s": times[0],
        "last_frame_s": times[-1],
        "first_to_last_s": times[-1] - times[0],
        "playback_end_s": end_s,
        "median_interval_ms": float(np.median(intervals_ms)),
        "p99_interval_ms": float(np.percentile(intervals_ms, 99)),
        "min_interval_ms": float(np.min(intervals_ms)),
        "max_interval_ms": float(np.max(intervals_ms)),
        "non_increasing": int(np.count_nonzero(intervals_ms <= 0)),
        "gap_limit_ms": gap_limit_ms,
        "gaps_over_1_5_frames": int(np.count_nonzero(intervals_ms > gap_limit_ms)),
        "gaps_over_100ms": int(np.count_nonzero(intervals_ms > 100)),
        "covers_requested_window": (
            abs(times[0]) <= float(time_base)
            and abs(end_s - seconds) <= 1.5 / float(nominal_fps)
        ),
    }
    return result, table


def analyze(output: Path) -> tuple[dict, list[dict]]:
    """证据缺失、命令失败、损坏、时间跳变或帧不符均不能判定通过。"""
    problems, table = [], []
    report = {
        "passed": False,
        "problems": problems,
        "scope": "Windows FFmpeg 收到的本录像窗口内压缩视频；非相机内部采集计数",
        "timestamp_meaning": "视频流播放时间，不是曝光时间或电脑收包时间",
    }
    try:
        session = json.loads(
            (output / "capture_session.json").read_text(encoding="utf-8")
        )
        report["requested_seconds"] = session["seconds"]
        report["capture_wall_seconds"] = session.get("capture_wall_seconds")
        for field in ("error", "stop_reason", "relay_errors", "relay_close_error"):
            if session.get(field):
                problems.append(f"{field}: {session[field]}")
        for field in ("capture_exit", "probe_exit", "decode_exit"):
            if session.get(field) != 0:
                problems.append(f"{field} 未成功：{session.get(field)}")
        if "rtsp_exchanges" in session and session.get("input_kind") != "offline_video":
            exchanges = session["rtsp_exchanges"]
            report["rtsp_teardown_confirmed"] = bool(
                exchanges
                and exchanges[-1]["method"] == "TEARDOWN"
                and exchanges[-1].get("response", "").startswith("RTSP/1.0 200 ")
            )
            if not report["rtsp_teardown_confirmed"] or any(
                not event.get("response", "").startswith("RTSP/1.0 200 ")
                for event in exchanges
            ):
                problems.append("相机控制交互未全部成功或未确认结束")
        log = (output / "ffmpeg.log").read_text(encoding="utf-8", errors="replace")
        report["capture_warning_lines"] = [
            line for line in log.splitlines() if WARNING_PATTERN.search(line)
        ]
        # 初次接入尚无 PPS/关键帧的提示单列，不将其藏在“零告警”里。
        # 只认可输入探测阶段的这两种明确提示；其他异常始终影响验收。
        report["startup_header_warning_lines"] = []
        probing = True
        for line in log.splitlines():
            if line.startswith("Input #0,"):
                probing = False
            if "non-existing PPS" in line or "no frame!" in line:
                if probing:
                    report["startup_header_warning_lines"].append(line)
                else:
                    report["capture_warning_lines"].append(line)
        if report["capture_warning_lines"]:
            problems.append("接收日志存在网络、数据或时间异常，见原始日志")
        for filename in ("probe.log", "decode.log"):
            text = (
                (output / filename)
                .read_text(encoding="utf-8", errors="replace")
                .strip()
            )
            report[filename] = text
            if text:
                problems.append(f"离线检查存在告警：{filename}")
        parsed = read_log_packets(log)
        probe = json.loads((output / "probe.json").read_text(encoding="utf-8"))
        streams = probe["streams"]
        if len(streams) != 1:
            raise ValueError("预期只有一条视频流")
        stream = streams[0]
        if stream["codec_name"] != "h264" or stream["has_b_frames"] != 0:
            raise ValueError("本轮逐帧核查只验证了不含 B 帧的 Chronos H.264 流")
        time_base = Fraction(stream["time_base"])
        packets = [r for r in probe["packets_and_frames"] if r["type"] == "packet"]
        frames = [r for r in probe["packets_and_frames"] if r["type"] == "frame"]
        counts = compare_packets(
            parsed["received"],
            parsed["muxed"],
            packets,
            time_base,
            session["seconds"],
            session.get("startup_skip_s", 0.0),
        )
        counts["decoded_frames"] = len(frames)
        report["counts"] = counts
        if not counts["ordered_timestamps_match"] or not packets:
            problems.append("录像窗口内的接收、封装和文件逐帧时间未完全对应")
        if (
            not packets
            or not frames
            or "K" not in packets[0]["flags"]
            or not frames[0]["key_frame"]
        ):
            problems.append("录像没有从可独立解码的关键帧开始")
        if (
            session.get("target_frames") is not None
            and len(frames) != session["target_frames"]
        ):
            problems.append("没有保存要求的完整画面数量")
        if len(frames) != len(packets):
            problems.append("压缩帧与解码画面数量不相等")
        if [r["pts"] for r in packets] != [r["pts"] for r in frames]:
            problems.append("压缩帧与解码画面播放时间未逐帧对应")
        if int(stream["nb_read_frames"]) != len(frames) or int(
            stream["nb_read_packets"]
        ) != len(packets):
            problems.append("文件检查的总数与逐帧记录不一致")
        report["video"] = {
            "codec": stream["codec_name"],
            "width": stream["width"],
            "height": stream["height"],
            "time_base": str(time_base),
            "duration_s": float(probe["format"]["duration"]),
            "size_bytes": int(probe["format"]["size"]),
        }
        timing, table = check_timestamps(
            frames,
            time_base,
            session["seconds"],
            Fraction(stream["r_frame_rate"]),
        )
        report["timing"] = timing
        if (
            session.get("expected_fps") is not None
            and abs(timing["nominal_fps"] - session["expected_fps"]) > 0.001
        ):
            problems.append("实际帧率与本次指定帧率不一致，不能确认请求时长")
        if timing["non_increasing"] or timing["gaps_over_1_5_frames"]:
            problems.append("发现播放时间重复、倒退或超过 1.5 帧的间隔")
        if not timing["covers_requested_window"]:
            problems.append("实际视频未完整覆盖请求的录像时间")
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        problems.append(f"证据不能完整核对：{type(exc).__name__}: {exc}")
    report["passed"] = not problems
    return report, table


def plot_timing(output: Path, report: dict, rows: list[dict]) -> None:
    """绘制文件中的播放间隔及保存进度；图表不假造网络收包时间。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    from customer_workspace.pzt_video_review import find_font

    font_manager.fontManager.addfont(str(find_font(None)))
    plt.rcParams["font.family"] = "Arial"
    times = [row["播放时间_秒"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(times[1:], [r["距上一帧_毫秒"] for r in rows[1:]], linewidth=0.8)
    axes[0].axhline(
        report["timing"]["gap_limit_ms"],
        color="red",
        linestyle="--",
        label="Gap threshold",
    )
    axes[0].set_ylabel("Frame interval (ms)")
    axes[0].legend()
    axes[1].plot(times, np.arange(1, len(rows) + 1))
    axes[1].set_ylabel("Saved frames")
    axes[1].set_xlabel("Video time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    counts = report["counts"]
    fig.suptitle(
        f"Received: {counts['received_window_frames']}  |  "
        f"Saved: {counts['decoded_frames']}  |  Missing: {counts['missing_in_saved']}"
    )
    fig.tight_layout()
    fig.savefig(output / "recording_check.png", dpi=160)
    plt.close(fig)


def make_report(output: Path, report_dir: Path | None = None) -> dict:
    """生成唯一的一份报告和逐帧时间表；重查请调用 analyze，不覆盖证据。"""
    output = Path(output)
    destination = output if report_dir is None else Path(report_dir)
    if report_dir is not None:
        destination.mkdir(parents=True, exist_ok=False)
    for name in ("recording_check.json", "frame_times.csv", "recording_check.png"):
        if (destination / name).exists():
            raise FileExistsError(f"不能覆盖已有报告：{destination / name}")
    report, rows = analyze(output)
    report["source_directory"] = str(output.resolve())
    write_json(destination / "recording_check.json", report)
    if rows:
        with (destination / "frame_times.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_timing(destination, report, rows)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report-dir", type=Path, help="复核时指定新目录，保留原报告")
    arguments = parser.parse_args()
    result = make_report(arguments.output_dir, arguments.report_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
