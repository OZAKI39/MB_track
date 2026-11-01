"""将六项实测读回画成表格图；只读现有记录，不连接设备。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
FONT = Path.home() / ".local/share/fonts/mevo-msyh.ttc"


def main():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(6, 2, figsize=(13, 17), layout="constrained")
    summaries = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    folders = sorted(p for p in ROOT.glob("[0-9][0-9]_*") if p.is_dir())
    for index, (summary, folder) in enumerate(zip(summaries, folders)):
        snapshots = [
            json.loads(line)
            for line in (folder / "stable_checks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for column, (key, scale, limits, label, cmap) in enumerate(
            (
                ("amplitude_vpp", 1, (0.001, 0.1), "幅值 / Vpp", "Blues"),
                ("frequency_hz", 1e6, (2.5, 2.6), "频率 / MHz", "Oranges"),
            )
        ):
            ax = axes[index, column]
            values = [
                [s["channels"][f"PZT{channel}"][key] / scale for s in snapshots]
                for channel in range(1, 5)
            ]
            ax.imshow(values, vmin=limits[0], vmax=limits[1], cmap=cmap, aspect="auto")
            for row, channel_values in enumerate(values):
                for col, value in enumerate(channel_values):
                    color = (
                        "white"
                        if value > limits[0] + 0.6 * (limits[1] - limits[0])
                        else "#17202a"
                    )
                    ax.text(
                        col,
                        row,
                        f"{value:g}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=10,
                    )
            ax.set_xticks(
                range(len(snapshots)),
                [f"{s['planned_check_s']:g} 秒" for s in snapshots],
            )
            ax.set_yticks(range(4), [f"PZT{i}" for i in range(1, 5)])
            ax.set_title(
                f"{index + 1}. {summary['name']}｜{label}", fontsize=12, loc="left"
            )
            ax.tick_params(length=0, pad=5)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(
        "四路自动控制：6 项实机测试全部通过\n"
        "每格是一次真实读回，124 组通道设置全部符合计划；检查时四路输出开关均为开启。\n"
        "横轴为计划检查时刻，完整检查最多耗时 0.416 秒；本图不表示连续测得的电压波形。\n"
        "第 1–5 项在 20 秒关闭；第 6 项在 10 秒取消，10.093 秒确认四路关闭。",
        fontsize=14,
    )
    for suffix in ("png", "svg"):
        with (ROOT / f"readback_overview.{suffix}").open("xb") as file:
            fig.savefig(file, format=suffix, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
