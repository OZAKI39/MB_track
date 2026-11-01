"""汇总同脚本 Windows/WSL 解码对照；只读取保存记录，不启动 GPU 或设备。"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from customer_workspace.tools.gpu_decode_check import save_json
from customer_workspace.tools.gpu_decode_report import read_csv, read_json, stats


def collect(root, output):
    """保留每遍结果，同时用合并后的原始样本计算分位数，不平均各遍 P99。"""
    output.mkdir(exist_ok=False)
    groups = defaultdict(list)
    runs, sources = [], {}
    for directory in sorted((root / "timing").iterdir()):
        if not directory.is_dir():
            continue
        system, video, repeat = directory.name.split("_")
        metadata = read_json(directory / "input.json")
        summary = read_json(directory / "summary.json")
        closed = read_json(directory / "closed.json")
        frames = read_csv(directory / "frames.csv")
        if (
            not summary["completed"]
            or not closed["completed"]
            or summary["full_image_downloads"]
        ):
            raise ValueError(f"Incomplete or verification-only run: {directory}")
        if len(frames) != metadata["packets"] or [
            int(f["frame"]) for f in frames
        ] != list(range(len(frames))):
            raise ValueError(f"Missing or out-of-order frames: {directory}")
        source = sources.setdefault(video, metadata["sha256"])
        if source != metadata["sha256"]:
            raise ValueError(f"Different source video in {directory}")
        for frame in frames:
            frame["repeat"] = repeat
        groups[(system, video)].extend(frames)
        runs.append(
            {
                "system": system,
                "video": video,
                "repeat": repeat,
                "frames": len(frames),
                **stats([float(f["decode_ready_ms"]) for f in frames]),
                "over_16_67_ms": summary["over_16_67_ms"],
            }
        )
    if set(groups) != {
        (system, video) for system in ("native", "wsl") for video in ("A", "B")
    }:
        raise ValueError("Both systems and both videos are required")
    results = []
    for (system, video), frames in sorted(groups.items()):
        values = [float(f["decode_ready_ms"]) for f in frames]
        worst = max(frames, key=lambda f: float(f["decode_ready_ms"]))
        results.append(
            {
                "system": system,
                "video": video,
                "frames": len(frames),
                "runs": len({f["repeat"] for f in frames}),
                **stats(values),
                "over_16_67_ms": sum(v > 1000 / 60 for v in values),
                "worst_repeat": worst["repeat"],
                "worst_frame": int(worst["frame"]),
                "late_ms": stats([float(f["late_ms"]) for f in frames]),
                "stages": {
                    name: stats([float(f[name]) for f in frames])
                    for name in (
                        "prepare_ms",
                        "decode_call_ms",
                        "decoder_sync_ms",
                        "stream_sync_ms",
                    )
                },
            }
        )
    save_json(
        output / "summary.json",
        {"groups": results, "runs": runs, "video_sha256": sources},
    )
    for filename, rows in (
        ("runs.csv", runs),
        (
            "comparison.csv",
            [
                {k: v for k, v in r.items() if k not in ("late_ms", "stages")}
                for r in results
            ],
        ),
    ):
        with (output / filename).open("x", newline="") as file:
            writer = csv.DictWriter(file, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for result in results:
        print(
            result["system"],
            result["video"],
            {k: result[k] for k in ("frames", "median", "p99", "max", "over_16_67_ms")},
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
