"""汇总 GPU 解码诊断：分开呈现公平短测、内部调用追踪和证据边界。"""

import argparse
import bisect
import sqlite3
from collections import defaultdict
from pathlib import Path

from customer_workspace.tools.gpu_decode_check import save_json
from customer_workspace.tools.gpu_decode_report import read_csv, read_json, stats


def trace_frames(database, timings):
    """按同一线程的包含关系关联调用；解析器、解码和映射是嵌套调用，不能重复累加。"""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        names = dict(connection.execute("SELECT id, value FROM StringIds"))
        video_calls = [
            (start, end, thread, names[name])
            for start, end, thread, name in connection.execute(
                "SELECT start, end, globalTid, nameId FROM NVVIDEO_DECODER_API ORDER BY start"
            )
        ]
        parsers = [r for r in video_calls if r[3] == "cuvidParseVideoData"]
        if len(parsers) != len(timings):
            raise ValueError("Parser count does not match source frames")
        threads = {r[2] for r in parsers}
        if len(threads) != 1:
            raise ValueError("Expected one decode thread")
        thread = threads.pop()
        # 使用已解析的实际调用栈名称；未采到栈的短调用明确保留 unknown。
        chain_names = {}
        for chain, symbol, module, depth in connection.execute(
            "SELECT id, symbol, module, stackDepth FROM OSRT_CALLCHAINS ORDER BY stackDepth"
        ):
            if "libdxcore" in names.get(module, ""):
                chain_names.setdefault(chain, names.get(symbol, "unknown"))
        ioctls = [
            (start, end, chain_names.get(chain, "unknown"), returned)
            for start, end, chain, returned in connection.execute(
                "SELECT a.start, a.end, a.callchainId, a.returnValue FROM OSRT_API a "
                "JOIN StringIds s ON a.nameId=s.id "
                "WHERE s.value='ioctl' AND a.globalTid=? ORDER BY a.start",
                (thread,),
            )
        ]
    starts = [r[0] for r in ioctls]
    submits = [r for r in video_calls if r[3] == "cuvidDecodePicture"]
    maps = [r for r in video_calls if r[3] == "cuvidMapVideoFrame64"]
    if len(submits) != len(parsers) or len(maps) != len(parsers):
        raise ValueError("Expected one submit and map per frame")
    results = []
    for index, (parser, submit, mapped, timing) in enumerate(
        zip(parsers, submits, maps, timings)
    ):
        start, end, _, _ = parser
        if not start <= submit[0] < submit[1] <= mapped[0] < mapped[1] <= end:
            raise ValueError("Unexpected nested call order")
        chosen = [
            r
            for r in ioctls[
                bisect.bisect_left(starts, start) : bisect.bisect_left(starts, end)
            ]
            if r[1] <= end
        ]
        by_name = defaultdict(float)
        calls = []
        for a, b, name, returned in chosen:
            duration = (b - a) / 1e6
            by_name[name] += duration
            calls.append(
                {
                    "name": name,
                    "duration_ms": duration,
                    "start_from_parser_ms": (a - start) / 1e6,
                    "return": returned,
                }
            )
        results.append(
            {
                "frame": index,
                "total_ms": float(timing["total_ms"]),
                "decode_ms": float(timing["decode_ms"]),
                "decode_cpu_ms": float(timing["decode_cpu_ms"]),
                "linux_runqueue_ms": float(timing["linux_runqueue_ms"]),
                "major_faults": int(timing["major_faults"]),
                "minor_faults": int(timing["minor_faults"]),
                "parser_ms": (end - start) / 1e6,
                "submit_ms": (submit[1] - submit[0]) / 1e6,
                "map_ms": (mapped[1] - mapped[0]) / 1e6,
                "ioctl_ms": sum(by_name.values()),
                "lock_ms": by_name.get("D3DKMTLock2", 0),
                "unlock_ms": by_name.get("D3DKMTUnlock2", 0),
                "frame_ready_wait_ms": by_name.get(
                    "D3DKMTWaitForSynchronizationObjectFromCpu", 0
                ),
                "calls": calls,
            }
        )
    return results


def build(root, output):
    output.mkdir(exist_ok=False)
    groups = defaultdict(list)
    status = {}
    for item in read_json(root / "commands.json"):
        name = item["name"]
        rows = read_csv(root / name / "frames.csv")
        if [int(r["frame"]) for r in rows] != list(range(300)):
            raise ValueError(f"Incomplete sequence: {name}")
        groups["_".join(name.split("_")[:2])].extend(rows)
        status[name] = read_csv(root / name / "gpu_status.csv")
    summary = {"controls": {}, "traces": {}, "slow_frames": [], "gpu_status": status}
    for name, rows in groups.items():
        summary["controls"][name] = {
            "frames": len(rows),
            **{
                key: stats([r[key] for r in rows])
                for key in (
                    "total_ms",
                    "decode_ms",
                    "decode_cpu_ms",
                    "linux_runqueue_ms",
                    "stream_sync_ms",
                )
            },
            "major_faults": sum(int(r["major_faults"]) for r in rows),
        }
    for name in ("hybrid", "spin", "gpu", "none", "full_A", "full_B"):
        frames = trace_frames(
            root / f"profile_{name}.sqlite",
            read_csv(root / f"traced_{name}/frames.csv"),
        )
        save_json(output / f"trace_{name}.json", frames)
        summary["traces"][name] = {
            "frames": len(frames),
            **{
                key: stats([r[key] for r in frames])
                for key in (
                    "parser_ms",
                    "submit_ms",
                    "map_ms",
                    "ioctl_ms",
                    "lock_ms",
                    "unlock_ms",
                    "frame_ready_wait_ms",
                )
            },
        }
        if name.startswith("full_"):
            for frame in sorted(frames, key=lambda r: -r["total_ms"])[:3]:
                summary["slow_frames"].append({"run": name, **frame})
    host_root = root / "host_trace"
    summary["host_windows"] = read_json(host_root / "host_windows/summary.json")
    summary["host_runs"] = {}
    for video in ("A", "B"):
        frames = trace_frames(
            host_root / f"profile_{video}.sqlite",
            read_csv(host_root / f"traced_{video}/frames.csv"),
        )
        save_json(output / f"host_trace_{video}.json", frames)
        summary["host_runs"][video] = {
            "frames": len(frames),
            "total_ms": stats([row["total_ms"] for row in frames]),
            "worst_frames": sorted(frames, key=lambda row: -row["total_ms"])[:3],
        }
    summary["protected_files"] = read_json(root / "protected_check.json")
    light_failure = root / "host_light/launch_failure.json"
    if light_failure.exists():
        summary["light_capture_attempt"] = read_json(light_failure)
    save_json(output / "summary.json", summary)
    lines = [
        "# GPU 解码延迟：内部请求与系统等待诊断",
        "",
        "主要额外耗时已定位到当前 WSL NVDEC 路径的驱动同步交接，特别是每帧资源锁定/释放请求；取回画面时另有等待解码完成的时间。不能把 Decode() 的总耗时全部当作显卡解码芯片的计算时间。",
        "",
        "具体到偶发十几毫秒等待：已捕获到等待的是哪个驱动请求；尚未完整捕获同一次异常在 Windows 内部的全过程，因此还不能断言是某个后台程序、驱动缺陷或硬件省电造成。下面区分已经证实的原因与尚未证实的更深层触发因素。",
        "",
        "## 对照条件",
        "",
        "A/B 各使用同样的前 300 帧，每种条件各两次，交错顺序，共 4800 帧。所有方案复用相同解码和图像示例、8 帧预热；只改变帧间等待方式。额外读数在解码计时以外。",
        "",
        "- hybrid：按源视频时刻，提前 1 ms 从睡眠切到忙等。",
        "- spin：同样的源视频时刻，间隔内 CPU 一直忙等。",
        "- gpu：同样的源视频时刻，间隔内执行短小 GPU 空工作，最后 2 ms 停止投递并忙等。这是因果对照，不是正式优化建议。",
        "- none：连续尽快送帧，只用来观察持续工作与间歇工作的差别，不能当作实时摄像机延迟。",
        "",
        "| 视频/条件 | 帧数 | 中位 ms | P95 ms | P99 ms | 最大 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(summary["controls"]):
        item = summary["controls"][name]
        t = item["total_ms"]
        lines.append(
            f"| {name} | {item['frames']} | {t['median']:.3f} | {t['p95']:.3f} | {t['p99']:.3f} | {t['max']:.3f} |"
        )
    lines += [
        "",
        "## 解码库内部发生什么",
        "",
        "Nsight 的 NVDEC/系统 API 追踪记录了实际调用栈。解析器内部先调用 cuvidDecodePicture，再调用 cuvidMapVideoFrame64。下表为独立追踪的 180 帧中位数；追踪结果与未追踪的对照分开，不把父调用和子调用重复相加。",
        "",
        "| 条件 | 提交解码 ms | 等待画面可用 ms |",
        "|---|---:|---:|",
    ]
    for name in ("hybrid", "spin", "gpu", "none"):
        t = summary["traces"][name]
        lines.append(
            f"| {name} | {t['submit_ms']['median']:.3f} | {t['map_ms']['median']:.3f} |"
        )
    lines += [
        "",
        "提交阶段的主要系统调用来自 libdxcore.so 的 D3DKMTLock2 / D3DKMTUnlock2，追踪中每帧约三组。它们是显卡资源的锁定/释放请求，不是 Python 线程之间的互斥锁。取回画面阶段的主要等待来自 D3DKMTWaitForSynchronizationObjectFromCpu。",
        "",
        "微软公开的 WSL 驱动实现显示，Lock2 / Unlock2 要通过虚拟机总线发送同步请求并等待 Windows 返回。本机实际调用栈与该路径吻合；公开源码只作为机制解释，不冒充本机闭源驱动的逐行源码。",
        "",
        "## 完整视频捕获的慢帧",
        "",
        "另对 A/B 各 1320 帧完整追踪，保留源帧号及原视频时间。下面的系统调用耗时是实际落在该帧 Decode 内部的记录：",
        "",
    ]
    for frame in summary["slow_frames"]:
        if frame["total_ms"] < 10:
            continue
        largest = sorted(frame["calls"], key=lambda r: -r["duration_ms"])[:3]
        lines.append(
            f"- {frame['run']}，源帧 {frame['frame']}：总计 {frame['total_ms']:.3f} ms；Linux 运行队列等待 {frame['linux_runqueue_ms']:.3f} ms；Linux 大缺页 {frame['major_faults']}。"
        )
        lines.append(
            "  主要请求："
            + "；".join(f"{r['name']} {r['duration_ms']:.3f} ms" for r in largest)
            + "。"
        )
    lines += [
        "",
        "## Windows 侧的独立核对",
        "",
        "又同时记录 Windows GPU、线程与内存活动，并完整解码 A/B 各 1320 帧。录制已正常结束，原始 ETL 已移入本轮 host_trace/host.etl，大小与 SHA256 见 etl_archive.json；没有继续在后台记录。",
        "",
        "选取 A 的 805、679 帧和 B 的 1215、633 帧附近窗口，读取到 120563 条系统事件，事件丢失数 0、字段解析错误数 0。Windows 实际执行资源锁定和释放的时间如下：",
        "",
        "| Windows 内部函数 | 样本数 | 中位 ms | P99 ms | 最大 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in summary["host_windows"]["host_functions"].items():
        lines.append(
            f"| {name} | {item['count']} | {item['median_ms']:.4f} | {item['p99_ms']:.4f} | {item['max_ms']:.4f} |"
        )
    lines += [
        "",
        "这里核对的是选定窗口内 System 进程的同类资源调用，不冒充所有调用，也不把两次不同运行的调用强行一一对应。Windows 处理锁定/释放通常远短于 WSL 等待整个请求返回的时间，说明额外开销主要发生在请求交接和等待回复这条路径上，而不是这些 Windows 函数本身持续计算数毫秒。",
        "",
        "时钟通过实际请求/回复区间核对；中点只用于近似选择对应窗口，保留区间宽度。四个窗口最大半区间为 0.68–1.33 ms，因此不宣称跨系统逐个亚毫秒调用已精确配对，也不把往返耗时除以二称为单向延迟。",
        "",
        "完整系统追踪会改变负载。这次没有再次捕获前面 12.736 ms / 10.875 ms 的同类超长锁定调用；因此不能凭这次较短的 Windows 函数耗时，断言前一次异常的 Windows 内部也一定很快。",
        "",
        "A 的 805 帧附近有另一进程的短硬缺页，但尚无证据把它与该解码调用连接起来；B 的 633 帧主要慢在最终 CUDA 流同步，不能仅凭附近有 GPU 内存映射事件，就称为 Windows 页文件读取。Linux 无缺页也不用于排除宿主内存等待。",
        "",
        "读取 ETL 时，xperf 在 CLR 元数据处失败；系统原生事件读取也超时。失败日志保留，最终使用微软 TraceEvent 2.0.74 读取成功，原始文件没有更改。依赖仅供独立诊断，不加入正式分析环境。",
        "",
        "后续准备了更小范围的 GpuLight 配置，wpr 已识别其格式；但这次 Windows 管理员启动等待 90 秒超时，没有开始录制，未产生可用于结论的新数据。host_light/launch_failure.json 保存失败原因；已创建阻止迟到启动的目录与停止标志，检查确认 WPR 不在录制。",
        "",
        "## 证据边界",
        "",
        "- 对照中普通等待也主要处于 P0、视频时钟 1995 MHz。GPU 轻负载时频率可升到 2160 MHz，不能把改善全部归因于频率，也不能用这些低频率采样排除更短的硬件门控变化。",
        "- Linux 运行队列等待很小，长锁定等待中可为零；它不等于 Windows 宿主没有调度或内存问题。Linux 无大缺页也不能排除 Windows 页文件读取。",
        "- 一个短测慢帧在最终 CUDA 流同步里耗时 11.373 ms，且调用线程 CPU 时间约同样长。这是另一种同步忙等，不能把全部尾部波动都归为 Lock2。",
        "- Nsight 2024.5 与当前驱动的 GPU 活动追踪不完全兼容；这里只使用已实际记录的 CPU API、调用栈和线程事件，没有宣称已测得纯硬件解码时间。",
        "- 现有证据支持“逐帧同步交接代价大”，还不足以给偶发最长等待指定唯一的 Windows 内部触发原因。完整系统记录改变了负载，原来的超长锁定等待没有同时复现。",
        "",
        "## 资料与代码",
        "",
        "本轮只新增独立诊断工具，并给已有 Windows 诊断脚本增加显式 GPU 诊断选项，原 Stalls 默认不变。12 个受保护文件的内容摘要核对一致，包括正式追踪、录像入口、源视频和设备配置。原始 commands.json、逐帧 CSV、GPU 状态和 .nsys-rep/.sqlite 均保留。",
        "",
        "诊断入口：tools/gpu_decode_diagnosis.py；汇总入口：tools/gpu_decode_root_report.py；Windows 联合诊断入口：tools/gpu_decode_host_trace.py；ETL 读取源码：tools/gpu_host_events.cs。host_trace/host_windows 中保留选择窗口、解析命令、字段结果和失败记录。",
        "",
        "结论用于定位问题，没有把 CPU 忙等或持续 GPU 空工作接入正式程序。短测证明这些干预会改变等待，但持续空工作增加资源消耗，不能据此声称已经获得可用于实验的优化方案。",
        "",
        "机制依据：[NVIDIA NVDEC 编程指南](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvdec-video-decoder-api-prog-guide/index.html)、[微软 WSL GPU 架构](https://devblogs.microsoft.com/directx/directx-heart-linux/)、[微软公开 Lock2/Unlock2 实现](https://github.com/microsoft/WSL2-Linux-Kernel/blob/linux-msft-wsl-6.6.y/drivers/hv/dxgkrnl/dxgvmbus.c#L2920)。",
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
