"""汇总独立 CPU/NVDEC 对照、像素核对和显存搬运证据；不访问设备。"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from customer_workspace.tools.gpu_decode_check import save_json


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Missing or non-finite timing values")
    return dict(
        zip(
            ("median", "p95", "p99", "max"),
            map(float, np.percentile(values, [50, 95, 99, 100])),
        )
    )


def copies(path):
    rows = read_csv(path)
    groups = defaultdict(lambda: {"calls": 0, "bytes": 0})
    unknown = 0
    for row in rows:
        group = groups[f"{row['source_type']}->{row['destination_type']}"]
        group["calls"] += 1
        group["bytes"] += int(row["bytes"])
        unknown += row["known"] != "1"
    return {"groups": dict(groups), "unknown_calls": unknown, "total_calls": len(rows)}


def charts(output, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    font = Path("/mnt/c/Windows/Fonts/arial.ttf")
    if not font.exists():
        raise FileNotFoundError("Actual Arial is required")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True)
    for video, ax in zip(("A", "B"), axes):
        for engine, color, label in (
            ("cpu", "#B75B66", "CPU"),
            ("gpu", "#4A897B", "NVDEC"),
        ):
            values = np.sort(
                [float(r["decode_ready_ms"]) for r in data[f"{video}_{engine}"]]
            )
            ax.plot(
                values,
                np.arange(1, len(values) + 1) * 100 / len(values),
                label=label,
                color=color,
                linewidth=1.8,
            )
        ax.set_title(f"Exp. {video}")
        ax.set_xlabel("Packet to decoded frame (ms)")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0, 101)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    axes[0].set_ylabel("Cumulative frames (%)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "decode_latency.png", dpi=240)
    fig.savefig(output / "decode_latency.pdf")
    plt.close(fig)


def build(root, output):
    output.mkdir(exist_ok=False)
    groups = defaultdict(list)
    runs = {}
    for item in read_json(root / "timing/commands.json"):
        name = item["name"]
        folder = root / "timing" / name
        rows = read_csv(folder / "frames.csv")
        state = read_json(folder / "run.json")
        source = read_json(folder / "input.json")
        if len(rows) != source["packets"] or state["frames"] != len(rows):
            raise ValueError(f"Incomplete frame count in {name}")
        if [int(row["frame"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"Missing or repeated frames in {name}")
        group = "_".join(name.split("_")[:2])
        groups[group].extend(rows)
        runs[name] = {
            "frames": len(rows),
            "preparation_ms": state["preparation_ms"],
            **{
                field: stats([row[field] for row in rows])
                for field in (
                    "decode_ready_ms",
                    "view_ms",
                    "image_ms",
                    "packet_to_image_ms",
                    "scheduled_to_image_ms",
                    "late_ms",
                )
            },
        }
    summary = {
        "runs": runs,
        "groups": {
            name: {
                "frames": len(rows),
                **{
                    field: stats([row[field] for row in rows])
                    for field in (
                        "decode_ready_ms",
                        "view_ms",
                        "image_ms",
                        "packet_to_image_ms",
                        "scheduled_to_image_ms",
                        "late_ms",
                    )
                },
                "decode_over_16_67": sum(
                    float(r["decode_ready_ms"]) > 1000 / 60 for r in rows
                ),
                "work_over_16_67": sum(
                    float(r["packet_to_image_ms"]) > 1000 / 60 for r in rows
                ),
            }
            for name, rows in groups.items()
        },
        "pixels": {
            video: read_json(root / f"verify_{video}_warmed/verification.json")
            for video in ("A", "B")
        },
        "copies": {
            video: copies(root / f"copy_{video}_checked/cuda_copies.csv")
            for video in ("A", "B")
        },
        "copy_positive_control": copies(root / "copy_control.csv"),
    }
    for video, pixels in summary["pixels"].items():
        if (
            pixels["native_different_pixels"]
            or pixels["mask_different_pixels"]
            or not np.isfinite(pixels["image_max_error"])
            or pixels["image_max_error"] > 1e-6
            or not pixels["all_addresses_shared"]
        ):
            raise ValueError(f"Image or device-address validation failed: {video}")
        audit = summary["copies"][video]
        if (
            audit["unknown_calls"]
            or set(audit["groups"]) != {"2->2"}
            or audit["total_calls"] != 240
        ):
            raise ValueError(f"Unexpected copy audit result: {video}")
    if set(summary["copy_positive_control"]["groups"]) != {"1->2", "2->1"}:
        raise ValueError("Copy observer did not detect the known host transfers")
    save_json(output / "summary.json", summary)
    charts(output, groups)
    lines = [
        "# WSL GPU 解码验证",
        "",
        "**已通过：WSL 可实际使用 RTX 4060 Laptop GPU 的 NVDEC 和 CUDA；H.264 解码结果可直接作为 CuPy 显存视图，继续执行 CUDA 图像操作，无须先下载整图。正式微泡追踪程序未修改。**",
        "",
        "**性能结论：本次按约 60 FPS 送帧，GPU 解码中位数与 P99 均高于 CPU，没有取得解码加速。加入简单图像示例后，中位总时间稍短，但较慢帧没有稳定改善；目前没有依据直接替换正式 CPU 解码。**",
        "",
        "## 测试范围与方法",
        "",
        "- 使用已有 A、B H.264 视频，各 1320 帧、1280×1024、约 60 FPS、无 B 帧。压缩包先读入内存；两种解码器使用同样的 Annex B 压缩数据，不把读盘时间混入解码。",
        "- CPU 使用正式环境的 PyAV 19.0.1、单解码线程；GPU 使用独立 `.venv_nvdec` 的 PyNvVideoCodec 2.2.3 / CuPy 13.6.0。CUDA toolkit 为原有 12.6，当前驱动支持 CUDA 13.4。没有安装或更换 Windows 驱动。",
        "- CPU/GPU 每段各重复三次，交错顺序，按源视频时间逐帧送包；每次先预热 8 帧，再从第 0 帧的关键帧重新开始。启动准备另计。没有预取未来画面，没有增加正式等待区。",
        "- 12 次计时运行共完成 15,840 帧，每轮帧号连续、数量完整。核对模式另行检查 A/B 全部 2,640 帧。",
        "- 每次等待 NVDEC 和 CUDA 流确实完成再停止计时，测到的不是异步提交命令的时间。所有正式计时帧都处理，超时只记录为晚完成，不主动跳帧。",
        "- 解码计时包含当前 Python/CUDA 接口调用与等待完成的成本，不是 NVDEC 芯片的纯计算耗时。GPU 较慢的具体来源未在本轮继续拆分，不能据此归因于某个驱动或系统组件。",
        "- 图像处理示例为原始亮度平面的 3×3 均值、归一化暗度、阈值；CPU/GPU 使用同样数学计算。它只用于证明显存内运算，不是正式微泡分割或追踪。",
        "- 单独的全帧核对显式下载结果；速度测试没有完整图像下载。内存搬运观察另跑 A/B 各 120 帧，未混入速度统计。",
        "",
        "## 解码耗时与延迟",
        "",
        "下表从本帧压缩包开始提交，计时到其原生解码画面确实可用。CPU 结果在主内存，GPU 结果在显存；均不含后面的图像示例。",
        "",
        "| 视频 / 路线 | 帧数 | 中位 ms | P95 ms | P99 ms | 最大 ms | >16.67 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("A_cpu", "A_gpu", "B_cpu", "B_gpu"):
        group = summary["groups"][name]
        t = group["decode_ready_ms"]
        lines.append(
            f"| {name} | {group['frames']} | {t['median']:.3f} | {t['p95']:.3f} | {t['p99']:.3f} | {t['max']:.3f} | {group['decode_over_16_67']} |"
        )
    lines += [
        "",
        "![Decode latency](decode_latency.png)",
        "",
        "图的横轴是完成解码的时间（对数刻度），纵轴是累计完成比例；曲线越靠左，用时越短。统计合并三轮，每轮数据仍保留在 summary.json。",
        "",
        "**计入相同图像示例后的时间：**",
        "",
        "| 视频 / 路线 | 送包到图像处理完成，中位 ms | P99 ms | 最大 ms | 计划时刻到完成 P99 ms | 送包晚到 P99 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("A_cpu", "A_gpu", "B_cpu", "B_gpu"):
        group = summary["groups"][name]
        t = group["packet_to_image_ms"]
        lines.append(
            f"| {name} | {t['median']:.3f} | {t['p99']:.3f} | {t['max']:.3f} | {group['scheduled_to_image_ms']['p99']:.3f} | {group['late_ms']['p99']:.3f} |"
        )
    lines += [
        "",
        "以上全部使用 WSL 的同一个时钟。没有连接 RTSP、Windows 视频接收或正式追踪，因此不能把这些数值称为相机曝光到反馈的整条链路延迟。",
        "",
        "## 像素和数据搬运核对",
        "",
        "| 视频 | 核对帧数 | 原生 YUV 不同像素 | GPU 图像示例最大误差 | 阈值图不同像素 | 显存地址全部相同 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for video, p in summary["pixels"].items():
        lines.append(
            f"| {video} | {p['frames']} | {p['native_different_pixels']} | {p['image_max_error']} | {p['mask_different_pixels']} | {p['all_addresses_shared']} |"
        )
    lines += [
        "",
        "每个 NV12 解码帧为 1,966,080 字节。DLPack/CuPy 视图与解码结果的显存地址逐帧相同，指针类型为 CUDA device memory。",
        "",
        "CUDA 拷贝观察在 A/B 各 120 帧中分别捕捉到 240 次复制：每帧亮度 1,310,720 字节、色度 655,360 字节，**全部为显存→显存**。没有 CPU 主内存→显存或显存→主内存的图像复制，也没有无法识别方向的复制。",
        "",
        "观察器已用已知的一次 1,310,720 字节上传和一次同尺寸下载做阳性检查，能准确记录两个方向。这里的结论是“没有整图经过 CPU 中转”，**不是 GPU 内部完全没有复制**。压缩码流仍从主内存输入解码器，驱动内部不可见操作不属于本探针完整覆盖的范围。",
        "",
        "旧 Nsight Systems 2024.5 能记录 NVDEC 调用，但警告当前 13.4 驱动超出支持版本，缺少完整 GPU 活动记录；没有把该缺失当作零拷贝证据。最终方向核对来自 CUDA API 参数观察及阳性检查。",
        "",
        "## 正式接入前仍需处理的限制",
        "",
        "1. **解码模式不能直接用默认值。** 初测 `LOW` 模式仍积压 16 帧（约 267 ms）；`ZERO` 才做到每个包立即输出一帧。本轮只对已核对无 B 帧、每包一帧的 A/B 视频使用该模式，不直接推广到其他视频。",
        "2. **当前库的 ZERO 输出时间戳不能直接使用。** 实测输出 PTS 始终为 0。测试保留原输入包的 PTS、帧号和计划时间，通过一包一帧及全帧像素顺序核对关联；不把库返回的 0 当作真实时间。正式接入仍需保留已有来源帧号和时间。",
        "3. **NV12 的亮度不是原追踪程序的灰度。** A/B 的原始 Y 平面与当前 YUV→BGR→灰度输入平均相差约 3.69/3.74 个灰度级，最大差 12。原始 YUV 解码完全一致，但不能将这个 Y 平面直接替换正式输入，否则可能改变识别边界。",
        "4. **现有识别与追踪仍在 CPU。** 若只换成 GPU 解码、后面继续调用当前 CPU 算法，仍需把图像下载回主内存。本轮证明的是“GPU 解码后可留在 GPU 做处理”，没有证明整套微泡分析已经 GPU 化，也没有证明先前 8–9 ms 的分析慢帧已经解决。",
        "5. **Windows→WSL 原本就传 H.264。** 本轮可避免的是解码后的 CPU/GPU 整图往返，不是消除一条原本就在跨系统传原图的链路。",
        "",
        "## 复现与证据",
        "",
        "入口与依赖见工作区 README 的 GPU 解码验证小节。timing/commands.json 保存每轮实际命令；每轮 frames.csv 保存原输入时间、实际提交/完成时间及耗时。",
        "",
        "原始资料：../verify_A_warmed/、../verify_B_warmed/；搬运资料：../copy_A_checked/、../copy_B_checked/、../copy_control.csv；早期试验失败记录保留，不计入最终性能统计。",
        "",
        "官方接口依据：[NVIDIA 解码接口](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-reference/decoder.html)、[NVIDIA 低延迟及 CuPy 互操作说明](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/using_pynvvideocodec_apis.html)。本地实测与文档存在差别的地方以上文实测限制为准。",
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
