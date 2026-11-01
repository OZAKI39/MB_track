"""只读诊断记录，分解管道、解码、交接、分析与 Windows 同钟反馈延迟。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from customer_workspace.streamcopy_recording import write_json
from customer_workspace.tracking_latency import (
    feedback_components_ms,
    forward_bounds_ms,
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Missing or non-finite latency measurements")
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
    }


def indexed(rows, key="seq"):
    mapping = {row[key]: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return mapping


def save_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def live_run(folder, output):
    native = read_json(folder / "latency_windows.json")
    decoder = read_json(folder / "latency_wsl_decoder.json")
    analysis = read_json(folder / "latency_wsl_analysis.json")["frames"]
    sent_all = indexed(native["sent"])
    sent = {seq: row for seq, row in sent_all.items() if row["in_window"]}
    received = indexed(
        [r for r in native["acknowledgements"] if r["kind"] == "received"]
    )
    completed = indexed(
        [r for r in native["acknowledgements"] if r["kind"] == "complete"]
    )
    decoded = indexed(decoder["frames"])
    analyzed = indexed(analysis)
    video_frames = {
        r["seq"]: r["video_frame"]
        for r in read_json(folder / "gst_session.json")["record_frames"]
    }
    with (folder / "analysis_frames.csv").open(newline="") as file:
        original = {int(r["source_frame"]): r for r in csv.DictReader(file)}
    checks = {
        "each_sent_frame_acknowledged": set(sent_all) == set(received),
        "each_analyzed_frame_acknowledged": set(analyzed)
        == set(completed)
        == set(original),
        "all_window_packets_decoded": set(sent) == set(decoded),
        "no_ack_error": not decoder["ack_errors"],
    }
    pipe_rows, result_rows, decode_rows = [], [], []
    for seq, source in sent.items():
        ack = received[seq]
        lower, upper = forward_bounds_ms(source, ack)
        pipe_rows.append(
            {
                "seq": seq,
                "video_frame": video_frames[seq],
                "payload_bytes": source["payload_bytes"],
                "forward_lower_ms": lower,
                "forward_upper_ms": upper,
                "windows_wait_for_frame_ms": (
                    source["selected_windows_s"] - source["get_started_windows_s"]
                )
                * 1000,
                "windows_write_ms": (
                    source["write_end_windows_s"] - source["write_start_windows_s"]
                )
                * 1000,
            }
        )
        d = decoded[seq]
        decode_rows.append(
            {
                "seq": seq,
                "video_frame": video_frames[seq],
                "receive_ack_preparation_ms": (
                    d["decode_started_wsl_s"] - d["arrived_wsl_s"]
                )
                * 1000,
                "decode_ms": (d["decode_finished_wsl_s"] - d["decode_started_wsl_s"])
                * 1000,
                "image_conversion_ms": (
                    d["converted_wsl_s"] - d["decode_finished_wsl_s"]
                )
                * 1000,
                "shared_copy_ms": (d["shared_copy_ready_wsl_s"] - d["converted_wsl_s"])
                * 1000,
                "analyzed": seq in analyzed,
            }
        )
    for seq, row in analyzed.items():
        ack = completed[seq]
        result = {
            "seq": seq,
            "video_frame": row["video_frame"],
            **feedback_components_ms(sent[seq], ack),
        }
        result.update(
            {
                "receive_ack_preparation_ms": (
                    row["decode_started_wsl_s"] - row["arrived_wsl_s"]
                )
                * 1000,
                "decode_ms": (
                    row["decode_finished_wsl_s"] - row["decode_started_wsl_s"]
                )
                * 1000,
                "image_conversion_ms": (
                    row["converted_wsl_s"] - row["decode_finished_wsl_s"]
                )
                * 1000,
                "shared_copy_ms": (
                    row["shared_copy_ready_wsl_s"] - row["converted_wsl_s"]
                )
                * 1000,
                # 发布、唤醒、等待、取出小型元数据统一作为交接；不用两线程的竞争先后相减。
                "publish_wait_take_ms": (
                    row["analysis_started_wsl_s"] - row["shared_copy_ready_wsl_s"]
                )
                * 1000,
                "analysis_ms": (
                    row["analysis_completed_wsl_s"] - row["analysis_started_wsl_s"]
                )
                * 1000,
                "old_age_estimate_ms": float(
                    original[seq]["software_age_upper_bound_ms"]
                ),
                "request_including_next_frame_wait_ms": float(
                    original[seq]["request_ms"]
                ),
            }
        )
        result["image_handoff_ms"] = (
            result["shared_copy_ms"] + result["publish_wait_take_ms"]
        )
        result["old_minus_feedback_ms"] = (
            result["old_age_estimate_ms"] - result["feedback_total_ms"]
        )
        result_rows.append(result)
    checks["causal_bounds_valid"] = all(
        0 <= r["forward_lower_ms"] <= r["forward_upper_ms"] + 0.001 for r in pipe_rows
    )
    components = [
        "receive_ack_preparation_ms",
        "decode_ms",
        "image_conversion_ms",
        "shared_copy_ms",
        "publish_wait_take_ms",
        "analysis_ms",
    ]
    checks["wsl_components_add_up"] = all(
        abs(sum(r[k] for k in components) - r["wsl_arrival_to_result_ms"]) < 0.001
        for r in result_rows
    )
    checks["nonnegative_communication_residual"] = all(
        r["forward_plus_ack_return_ms"] >= -0.001 for r in result_rows
    )
    save_csv(output / f"{folder.name}_per_frame.csv", result_rows)
    save_csv(output / f"{folder.name}_pipe.csv", pipe_rows)
    save_csv(output / f"{folder.name}_all_decoded.csv", decode_rows)
    worst = max(result_rows, key=lambda r: r["feedback_total_ms"])
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "windows_clock": native["clock"],
        "wsl_clock": decoder["clock"],
        "analyzed": len(analysis),
        "received_ack_window": len(sent),
        "completed_ack": len(completed),
        "stages": {
            k: stats([r[k] for r in result_rows])
            for k in result_rows[0]
            if k.endswith("_ms")
        },
        "all_decoded": {
            k: stats([r[k] for r in decode_rows])
            for k in decode_rows[0]
            if k.endswith("_ms")
        },
        "pipe": {
            k: stats([r[k] for r in pipe_rows])
            for k in pipe_rows[0]
            if k.endswith("_ms") or k == "payload_bytes"
        },
        "worst_feedback_frame": worst,
    }


def table(title, labels, summaries):
    lines = [
        f"### {title}",
        "",
        "单位 ms；P95 表示 95% 的记录不超过该值。",
        "",
        "| 视频 | 项目 | 中位数 | P95 | P99 | 最大值 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in summaries.items():
        for key, label in labels.items():
            row = values[key]
            numbers = " | ".join(
                f"{row[k]:.3f}" for k in ["median", "p95", "p99", "max"]
            )
            lines.append(f"| {name} | {label} | {numbers} |")
    return lines + [""]


def build(root, output):
    output.mkdir(parents=True, exist_ok=False)
    live, ready, controls = {}, {}, {}
    for name in ("A", "B"):
        live[name] = live_run(root / f"{name}_on", output)
        isolated = read_json(root / f"pipe_{name}/wsl.json")
        rows = [r for r in isolated["frames"] if r["measured"]]
        ready[name] = {
            k: stats([r[k] for r in rows])
            for k in [
                "ready_transaction_ms",
                "forward_lower_ms",
                "forward_upper_ms",
                "payload_bytes",
            ]
        }
        save_csv(output / f"{name}_ready_pipe.csv", rows)
        controls[name] = {}
        for mode in ("off", "on"):
            folder = root / f"{name}_{mode}"
            analysis = read_json(folder / "analysis_check.json")
            recording = read_json(folder / "recording_check.json")
            controls[name][mode] = {
                **{
                    k: analysis[k]
                    for k in [
                        "processed",
                        "saved",
                        "analysis_skipped",
                        "processing_ms",
                        "pipeline_error",
                        "threads_closed",
                    ]
                },
                **{
                    k: recording[k]
                    for k in [
                        "saved_frames",
                        "recording_extra_lost",
                        "rtp_sequence_missing",
                        "passed",
                    ]
                },
            }
    integrity = read_json(root / "integrity_report/summary.json")
    result = {
        "live": live,
        "ready_pipe": ready,
        "diagnostic_control": controls,
        "integrity": integrity,
        "physical_devices_tested": False,
    }
    write_json(output / "summary.json", result)
    lines = [
        "# Windows→WSL 延迟实测",
        "",
        "已确认 AGENTS 文件，我将遵守要求进行开发。",
        "",
        "本轮使用 F 盘 A/B RTSP 模拟源，每组运行 22 秒。识别、追踪、解码与灰度转换算法保持不变。",
        "确认消息通过已有 stdin 控制管道返回；视频继续通过原 stdout 管道传递。诊断默认关闭。",
        "",
        "## 结论",
        "",
        f"完整反馈中位数：A {live['A']['stages']['feedback_total_ms']['median']:.2f} ms，B {live['B']['stages']['feedback_total_ms']['median']:.2f} ms。主要耗时在 WSL 解码、图像转换和微泡分析。",
        f"正式运行的单向传输上界中位数：A {live['A']['pipe']['forward_upper_ms']['median']:.3f} ms，B {live['B']['pipe']['forward_upper_ms']['median']:.3f} ms。原保守估计包含等待下一帧，不能视为跨系统通信耗时。是否值得改架构，应比较这些值与解码、分析耗时，而不是直接使用原估计。",
        "",
        "## 计时方法",
        "",
        "- 完整反馈：Windows 收到 H.264 帧 → Windows 收到对应帧的分析完成确认，两端均取 Windows 的同一个时钟。",
        "- 单向传输下界：WSL 收齐 12 字节前缀 → 收齐帧头和 H.264 内容。",
        "- 单向传输上界：Windows 开始写帧 → Windows 收到接收确认，再减去 WSL 收齐数据到准备发送确认的本机耗时。上界仍含回程，不能当成精确单向值。",
        "- 独立管道测试预先加载真实压缩帧，先预热 32 包，再测全部 1320 包。计时前无需等待下一视频帧；每包随后核对 SHA-256。该测试没有录像和分析负载，因此另列正式链路中的传输上界。",
        "- 完成确认由独立诊断线程立即发送，不等下一次 GET；单独记录它在 WSL 内部排队的时间。",
        "- WSL 内各段只相减 WSL 时间；跨系统仅比较各自的时长，不相减两端的绝对计数，也不使用往返除以二。",
        "",
    ]
    lines += table(
        "完整反馈与 WSL 各段",
        {
            "feedback_total_ms": "完整反馈（Windows 同钟）",
            "windows_before_send_ms": "Windows 收到帧至开始发出",
            "decode_ms": "H.264 解码",
            "image_conversion_ms": "图像格式及灰度转换",
            "image_handoff_ms": "共享内存复制、发布、等待及取帧",
            "analysis_ms": "微泡分析调用（含提交保存）",
            "receive_ack_preparation_ms": "接收确认准备及进入解码",
            "completion_ack_queue_ms": "完成确认在 WSL 内等待",
            "forward_plus_ack_return_ms": "正向发送＋完成确认回程",
        },
        {k: {**v["stages"], **v["all_decoded"]} for k, v in live.items()},
    )
    lines += [
        "各列是分别排序所得的统计，不能把各项中位数或 P99 直接相加。逐帧可加的明细保存在 `*_per_frame.csv`。",
        "解码、转换、共享复制及接收确认准备统计覆盖全部 1320 个解码帧，包括之后被跳过的帧；分析、交接至分析和完整反馈只统计确实完成分析的帧。",
        "“正向发送＋完成确认回程”包含封装、管道读写和线程调度，不等于纯内核传输时间。",
        "",
    ]
    lines += table(
        "无等待、无分析负载的独立管道",
        {
            "ready_transaction_ms": "请求现成帧至收齐（含请求去程）",
            "forward_lower_ms": "单向传输下界",
            "forward_upper_ms": "单向传输上界",
        },
        ready,
    )
    lines += table(
        "正式运行中的管道与等待",
        {
            "forward_lower_ms": "单向传输下界",
            "forward_upper_ms": "单向传输上界",
            "windows_wait_for_frame_ms": "Windows 等待有可发送的帧",
            "windows_write_ms": "Windows 写帧调用",
        },
        {k: v["pipe"] for k, v in live.items()},
    )
    lines += table(
        "原估计与本轮测量的区别",
        {
            "old_age_estimate_ms": "原 software_age_upper_bound_ms",
            "request_including_next_frame_wait_ms": "原取帧请求（包含等下一帧）",
            "feedback_total_ms": "实际完整反馈",
            "old_minus_feedback_ms": "原估计减实际反馈（逐帧）",
        },
        {k: v["stages"] for k, v in live.items()},
    )
    lines += [
        "原公式 = Windows 已持有该帧时间 + GET 至收齐的时间 + WSL 收齐至完成的时间。",
        "GET 往往在下一帧到达前发出，其等待不能当作这张尚未到达的画面的年龄。原字段可以作为保守估计，不能据此认为跨系统传输用了约 26 ms。",
        "",
        "## 数量、保存及计时影响",
        "",
        "| 视频 | 计时 | 录像 | 分析/保存 | 分析跳帧 | 分析中位数 ms | P99 ms | 最大 ms |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, modes in controls.items():
        for mode, row in modes.items():
            p = row["processing_ms"]
            lines.append(
                f"| {name} | {mode} | {row['saved_frames']} | {row['processed']}/{row['saved']} | {row['analysis_skipped']} | {p['median']:.3f} | {p['p99']:.3f} | {p['max']:.3f} |"
            )
    lines += [
        "",
        "on 为开启诊断，off 为关闭诊断；每种条件只跑一遍，直播接入点与系统调度也可能不同，不能把差值全部归因于诊断。",
        "完整反馈只针对已分析帧；没有分析的帧不会伪造完成确认，去向记录见各运行的 `frame_outcomes.csv`。",
        "视频逐帧像素核对、分析结果与相同输入的离线重算、数据库完整性、跳帧位置及原因记录见 `../integrity_report/`。",
        "",
        "## 最慢反馈对应哪一帧",
        "",
    ]
    for name, values in live.items():
        worst = values["worst_feedback_frame"]
        lines.append(
            f"- {name}：录像第 {worst['video_frame']} 帧（从 0 编号），源编号 {worst['seq']}；总反馈 {worst['feedback_total_ms']:.3f} ms，其中 WSL {worst['wsl_arrival_to_result_ms']:.3f} ms、Windows 发出前 {worst['windows_before_send_ms']:.3f} ms、确认在 WSL 排队 {worst['completion_ack_queue_ms']:.3f} ms、正向发送加回程 {worst['forward_plus_ack_return_ms']:.3f} ms。"
        )
    lines += [
        "",
        "## 边界与尚未验证的部分",
        "",
        "- 这是模拟源的 Windows 应用程序收到完整压缩帧以后的计时，不包含真实曝光、相机编码、相机到 Windows 的网络时间，也不包含 AFG 动作。",
        "- 接收时间沿用 parallel_camera.py 的 _input_buffer：在 H.264 拆包、解析之后，进入 tee 分流之前。此前 RTSP 接收缓冲不在本轮约 14 ms 反馈中；rtspsrc 的 latency=20 配置值不等于已经测得这段延迟恰好为 20 ms。",
        "- 分析完成指轮廓、编号等已计算并提交保存；数据库全部写完在结束时另行核对。",
        "- 在禁止混减两端时钟、且不假定往返对称的条件下，单向传输只能给出有依据的上下界。本轮没有声称精确的单向延迟。",
        "- `giolibproxy.dll` 警告保留在各组 Windows 日志中；未据此修改环境，也没有把警告判定为慢帧原因。",
        "- 本轮没有消除所有偶发慢帧或分析跳帧；已将其计入完整分布，没有删掉最大值。",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if (
        not all(v["passed"] for v in live.values())
        or not integrity["recording_and_frame_accounting_passed"]
    ):
        raise ValueError("Latency coverage or recording/data integrity checks failed")
    print(output / "REPORT.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)


if __name__ == "__main__":
    main()
