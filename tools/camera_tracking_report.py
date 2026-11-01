"""汇总相机接入的文件模拟、逐像素/逐表核对与性能限制；不操作设备。"""

import argparse
import csv
import json
from pathlib import Path

from customer_workspace.streamcopy_recording import write_json
from customer_workspace.streamcopy_report import analyze, plot_timing


def read(path):
    return json.loads(path.read_text())


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    names = ["A_01", "B_01", "A_02", "B_02", "long_A", "long_B", "current_long_A"]
    reports = []
    gaps = []
    for name in names:
        folder = source / name
        analysis = read(folder / "analysis_check.json")
        recording = read(folder / "recording_check.json")
        equivalent = read(folder / "equivalence.json")
        writer = read(folder / "tracking/writer_report.json")
        receiver = read(folder / "tracking_bridge.json")
        rows = list(csv.DictReader((folder / "analysis_frames.csv").open()))
        outcomes = list(csv.DictReader((folder / "frame_outcomes.csv").open()))
        analyzed = {int(row["video_frame"]): row for row in rows}
        decoded = {row["video_frame"]: row for row in receiver["decoded"]}
        missing = [row for row in outcomes if row["outcome"] != "processed"]
        previous = None
        for outcome in outcomes:
            frame = int(outcome["video_frame"])
            if frame in analyzed:
                previous = analyzed[frame]
                continue
            arrived = decoded.get(frame)
            following = decoded.get(frame + 1)
            gaps.append(
                {
                    "run": name,
                    "video_frame": frame,
                    "source_frame": int(outcome["source_frame"]),
                    "outcome": outcome["outcome"],
                    "previous_analyzed_frame": previous["video_frame"]
                    if previous
                    else None,
                    "previous_processing_ms": previous["processing_call_ms"]
                    if previous
                    else None,
                    "arrival_to_next_frame_ms": (
                        following["published_wsl_s"] - arrived["published_wsl_s"]
                    )
                    * 1000
                    if arrived and following
                    else None,
                }
            )
        assert len(missing) == analysis["analysis_skipped"] == receiver["overwritten"]
        assert analysis["processed"] == analysis["saved"] == writer["committed_frames"]
        assert writer["all_submitted_saved"] and writer["worker_finished"]
        assert not writer["pending_records"]
        assert recording["passed"] and equivalent["scientific_results_equal"]
        assert equivalent["different_video_frames"] == 0
        reports.append(
            {
                "run": name,
                "seconds": analysis["window"]["seconds"],
                "received": recording["received_window_frames"],
                "recorded": recording["saved_frames"],
                "processed": analysis["processed"],
                "saved": analysis["saved"],
                "skipped": analysis["analysis_skipped"],
                "rate_hz": analysis["analysis_hz"],
                "processing_ms": analysis["processing_ms"],
                "software_age_upper_bound_ms": analysis["software_age_upper_bound_ms"],
                "save_peak_pending": writer["peak_pending_records"],
                "pixels_and_scientific_results_equal": True,
            }
        )
        # 新的审阅图使用英文 Arial；先前自动生成的图保持原样，避免覆盖证据。
        figures = output / name
        figures.mkdir()
        timing, table = analyze(folder / "recording")
        plot_timing(figures, timing, table)
    with (output / "gap_details.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(gaps[0]))
        writer.writeheader()
        writer.writerows(gaps)
    local = {
        name: read(source / f"local_{name}/paced_report.json") for name in ("A", "B")
    }
    failures = read(source / "failures/checks.json")
    memory = read(source / "current_long_A/stability_assessment.json")
    summary = {
        "runs": reports,
        "total_recorded": sum(r["recorded"] for r in reports),
        "total_processed_and_saved": sum(r["saved"] for r in reports),
        "total_skipped": len(gaps),
        "recording_and_save_integrity_passed": True,
        "pixel_and_tracking_equivalence_passed": True,
        "every_frame_at_60_fps_passed": not gaps,
        "live_hardware_tested": False,
        "fault_checks": failures,
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# 新相机入口接入在线追踪：本轮结果",
        "",
        "**接入与结果正确性已通过；60 fps 每帧都完成分析的目标尚未全部达到。真实相机和 AFG 本轮均未连接、未操作。**",
        "",
        "## 实际改了什么",
        "",
        "- 工作区的录像检查、持续检查、PZT 联合入口均改用现有 OnlineProcessor；识别阈值、追踪规则未改变。工程原 GUI 不在本轮替换范围。",
        "- Windows 用一份压缩流分出录像和分析两路。录像直接保存；分析数据传到独立 WSL 进程，再用原 PyAV 和 OpenCV 转换灰度。",
        "- 解码后保留最新画面，三块共享空间明确区分接收、等待、分析，不能覆盖正在使用的图像。压缩帧不能随意丢弃，队列满时报告错误。",
        "- 复用原分析预热、工作区复用、开始前/保存后清理与异步保存。运行中不修改自动清理规则。",
        "- 同一段实验共享 experiment_id 和 stream_id；保存源编号、录像编号、视频时间与 Windows 接收时间。WSL 解码时间单列，禁止直接相减两套系统计时器。",
        "- Ctrl+C 先让当前帧完成，再停止接收、等待已提交数据保存；有 AFG 时沿用独立控制器先关断再退出。",
        "",
        "## 验证条件",
        "",
        "使用已有 expA.mp4、expB.mp4，1280×1024、60 fps，每段 1320 帧。背景按原 combine_backgrounds 方法用参考背景和首帧准备，文件及来源见 ../inputs.json。",
        "分析 OpenCV 24 线程，解码/颜色转换 1 线程，详细逐帧诊断关闭。未暂停用户程序，也未改变 Windows/WSL 内存配置。",
        "62 秒素材由原片压缩数据重复拼接而成，只用于负载检查；循环接缝不是新的物理实验，不能拿来作刺激响应结论。",
        "",
        "## 完整采集结果",
        "",
        "| 运行 | 时长 s | 录像收到/保存 | 分析/保存 | 分析跳帧 | 处理中位 ms | P99 ms | 最慢 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        t = r["processing_ms"]
        lines.append(
            f"| {r['run']} | {r['seconds']:g} | {r['received']}/{r['recorded']} | {r['processed']}/{r['saved']} | {r['skipped']} | {t['median']:.3f} | {t['p99']:.3f} | {t['max']:.3f} |"
        )
    lines += [
        "",
        f"合计录像 **{summary['total_recorded']} 帧**，全部正确保存；分析 **{summary['total_processed_and_saved']} 帧**，这些帧全部写入数据库。**{summary['total_skipped']} 帧未分析**，全部已解码，在最新帧等待区被后续画面替换，没有把它们算作已分析。",
        "",
        "分析跳帧位置与前一张画面的处理耗时见 [gap_details.csv](gap_details.csv)。每次运行的 frame_outcomes.csv 保存逐帧去向，tracking_bridge.json 保存解码与传递时间。",
        "",
        "## 与原离线回放比较",
        "",
        "| 原回放 | 处理/送入 | 中位 ms | P99 ms | 最慢 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, data in local.items():
        t = data["metrics"]["process_ms"]
        lines.append(
            f"| {name} | {data['processed_frames']}/{data['offered_frames']} | {t['median']:.3f} | {t['p99']:.3f} | {t['max']:.3f} |"
        )
    lines += [
        "",
        "两条路线日常计算均在约 9–12 ms，但新路线的部分运行慢帧更多，不能声称已完全保住每一帧。日志能确定丢失发生在分析等待区；本轮没有逐线程系统诊断，不把这些波动归咎于某个未经核实的后台程序。",
        "",
        "原回放的结果年龄从 WSL 计划送帧起算，新路线报告 Windows 收到压缩帧至分析完成的保守上界，二者起点不同，不能直接相减作延迟增量。新路线 P95 约 42–47 ms；它包含接收后的软件过程，不代表曝光至结果的延迟。",
        "",
        "## 正确性核对",
        "",
        "七次运行全部逐像素核对了保存视频与原输入，没有不同画面。随后用原离线解码再次处理本次实际分析过的同一批帧，数据库逐表一致：轮廓、编号、速度、轨迹、合并/分离等事件和输入时间都保留比较，仅排除计算耗时。证据分别在各运行的 equivalence.json。",
        "未分析的帧没有参与同批输入一致性比较；这不能证明跳帧对物理运动测量没有影响。数据库完整性及关联关系检查也通过。",
        "",
        "历史真实相机日志中的 DTS（解码排序时间）全部缺失，新入口将其保留为缺失而使用有效 PTS（播放时间）计算运动。另以真实压缩视频模拟缺失 DTS 和非零 PTS，30 帧解码/转换像素与原输入一致；见 ../live_timestamp_check.json。",
        "",
        "## 队列、内存及异常",
        "",
    ]
    for key, label in [
        ("wsl_rss_mib", "WSL 分析"),
        ("decoder_rss_mib", "WSL 解码"),
        ("receiver_rss_mib", "Windows 接收"),
        ("ffmpeg_rss_mib", "FFmpeg"),
    ]:
        d = memory["memory"][key]
        lines.append(
            f"- {label}：最后 62 秒复测的前/后 10 秒平均常驻内存为 {d['first_10s_mean']:.2f}/{d['last_10s_mean']:.2f} MiB。"
        )
    lines += [
        "- 保存等待队列结束时均清空，最大待存数量见 summary.json；没有发现随运行持续积压。",
        "- 最后长测录像队列启动时最多暂存 85 帧，后 10 秒平均约 0.03 帧；未达到队列容量，没有新增录像缺失。",
        "- 分析、解码内存在最后长测分别增加约 4.5、3.1 MiB，未出现失控增长；这是有限时长检查，不能承诺无限运行。",
        "- 实际 Windows 文件流测试了取消、短视频提前结束、背景尺寸错误：均报告原因并退出；已分析帧全部保存。取消测试由完整帧边界退出，见 ../failures/checks.json。",
        "",
        "## 如何使用",
        "",
        "从工程根目录运行，下面只模拟视频，不连接设备：",
        "",
        "```bash",
        ".venv/bin/python -m customer_workspace.parallel_video_check \\",
        "  --offline-video customer_workspace/microbubble_tracking/data/raw/expA.mp4 \\",
        "  --background customer_workspace/validation/camera_tracking_20261031_final/background_A.npy \\",
        "  --seconds 22",
        "```",
        "",
        "接相机后去掉 --offline-video，并明确指定该视野已准备好的背景。新入口不使用旧 --geometry；不能拿 A 视频背景直接套用不同现场视野。具体命令见工作区 README 第 7.3–7.5 节。",
        "",
        "## 尚未通过或尚未验证",
        "",
        "1. **60 fps 每帧都分析：未全部通过。** 原因是分析偶尔来不及取走等待画面；程序保留最新画面并准确记录缺口。正常处理速度接近原回放，不等于严格逐帧保证。",
        "2. **真实相机与网络：未测试。** 当前相机未连接；RTSP/UDP 传输质量、现场格式及背景适配需要接机确认。",
        "3. **与真实 AFG 同时运行：未测试。** 本轮只做视频模拟，保留原关断控制及时间关联，未发送任何 AFG 命令。",
        "4. **曝光与超声物理同步：没有测量。** 当前对应的是软件接收和动作记录时间。",
        "",
        "原始视频、旧结果和科学标注均保留；没有修改识别算法、没有部署外部 Windows 设备目录、没有 Git 推送。",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
