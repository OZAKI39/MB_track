"""独立重算本批录像缺帧位置、导出图和抽查帧；不连接相机或 AFG。"""

import csv
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import av
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT / "capture_after_uuid_fix"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def main():
    session = read_json(CAPTURE / "capture_session.json")
    packets = session["packets"]
    rows = [
        json.loads(line)
        for line in (CAPTURE / "video_frames.jsonl").read_text().splitlines()
    ]
    report = read_json(CAPTURE / "recording_check.json")
    manifest = read_json(CAPTURE / "video_manifest.json")
    bridge = read_json(next(CAPTURE.glob("bridge_*/bridge_status.json")))
    first, last = packets[0]["source_frame_id"], packets[-1]["source_frame_id"]
    headers = [
        h
        for h in bridge["messages"]
        if h["status"] == "FRAME" and first <= h["seq"] <= last
    ]
    server_ids = {h["seq"] for h in headers}
    received_ids = {p["source_frame_id"] for p in packets}
    saved_ids = {p["source_frame_id"] for p in packets if p["accepted"]}
    assert len(server_ids) == len(headers) == len(received_ids) == len(packets) == 1185
    assert server_ids == received_ids
    assert saved_ids == {row["source_frame_id"] for row in rows}
    assert len(saved_ids) == len(rows) == 1011
    missing_before_transfer = sorted(set(range(first, last + 1)) - server_ids)
    rejected = sorted(received_ids - saved_ids)
    assert len(missing_before_transfer) == 47 and len(rejected) == 174
    assert not (set(missing_before_transfer) & set(rejected))
    assert report["recorder_dropped"] == len(rejected)
    assert report["upstream_sequence_skips_within_window"] == len(
        missing_before_transfer
    )
    assert not manifest["error"]
    assert all(
        d["reason"] == "QUEUE_FULL" and d["frame_id"] in rejected
        for d in manifest["dropped_frame_examples"]
    )
    assert all(
        item["passed"] and item["decoded_frames"] == 1011 for item in report["videos"]
    )
    with (CAPTURE / "frame_times.csv").open(encoding="utf-8-sig", newline="") as file:
        table = list(csv.DictReader(file))
    assert len(table) == len(rows)
    for row, csv_row in zip(rows, table):
        assert int(csv_row["来源帧号"]) == row["source_frame_id"]
        assert float(csv_row["录像时间_秒"]) == row["elapsed_since_first_frame_s"]
        assert (
            float(csv_row["Windows解码时间_秒"])
            == row["host_decode_timestamp"]["value"]
        )
        assert float(csv_row["WSL收到时间_秒"]) == row["arrival_monotonic"]
    origin = packets[0]["arrival_monotonic"]
    upstream_events = []
    for previous, current in pairwise(packets):
        count = current["source_frame_id"] - previous["source_frame_id"] - 1
        if count:
            upstream_events.append(
                {"noticed_at_s": current["arrival_monotonic"] - origin, "count": count}
            )
    rejection_times = [
        p["arrival_monotonic"] - origin for p in packets if not p["accepted"]
    ]
    result = {
        "verification_passed": True,
        "recording_complete": False,
        "source_sequence_first": first,
        "source_sequence_last": last,
        "decoded_source_frames_between_endpoints": last - first + 1,
        "windows_latest_frames_not_transferred": len(missing_before_transfer),
        "missing_before_transfer_ids": missing_before_transfer,
        "wsl_bridge_to_callback_missing": 0,
        "recorder_queue_full_rejections": len(rejected),
        "recorder_rejected_ids": rejected,
        "queue_first_rejection_s": min(rejection_times),
        "queue_last_rejection_s": max(rejection_times),
        "written_and_timestamps_verified": len(rows),
        "upstream_gap_noticed_times": upstream_events,
        "video_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in CAPTURE.glob("*.mp4")
        },
    }
    save_json(ROOT / "verification.json", result)

    # 仅提取原视频里的三帧供人工检查，不修改或替换原录像。
    samples = CAPTURE / "sample_frames"
    samples.mkdir(exist_ok=False)
    indices = {
        0,
        min(
            range(len(rows)),
            key=lambda i: abs(rows[i]["elapsed_since_first_frame_s"] - 10),
        ),
        len(rows) - 1,
    }
    extracted = []
    with av.open(str(CAPTURE / "experiment_raw.mp4")) as video:
        for index, frame in enumerate(video.decode(video=0)):
            if index in indices:
                path = samples / f"frame_{index + 1:04d}.png"
                frame.to_image().save(path)
                extracted.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "frame": index + 1,
                        "elapsed_s": rows[index]["elapsed_since_first_frame_s"],
                    }
                )
    save_json(ROOT / "sample_frames.json", extracted)

    font_manager.fontManager.addfont(
        str(Path.home() / ".local/share/fonts/mevo-msyh.ttc")
    )
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    elapsed = np.asarray([r["elapsed_since_first_frame_s"] for r in rows])
    axes[0].plot(elapsed[1:], np.diff(elapsed) * 1000, linewidth=0.7, color="#23628f")
    axes[0].axhline(
        100, color="#b03b30", linestyle="--", label="本次明显断档检查线：100 ms"
    )
    axes[0].set(
        ylabel="相邻保存画面的间隔 / 毫秒",
        title="已保存 1011 帧：最大间隔 55.740 ms；没有超过 100 ms 的间隔",
    )
    axes[0].legend(loc="upper right")
    bins = np.arange(21)
    upstream, _ = np.histogram(
        [e["noticed_at_s"] for e in upstream_events],
        bins=bins,
        weights=[e["count"] for e in upstream_events],
    )
    queue, _ = np.histogram(rejection_times, bins=bins)
    axes[1].bar(
        bins[:-1] + 0.5,
        upstream,
        width=0.8,
        label="Windows 最新帧未传出：47 帧",
        color="#e4a02b",
    )
    axes[1].bar(
        bins[:-1] + 0.5,
        queue,
        bottom=upstream,
        width=0.8,
        label="录像队列满时丢弃：174 帧",
        color="#b03b30",
    )
    axes[1].set(
        ylabel="每秒发现的缺帧数",
        xlabel="从收到第一帧开始 / 秒",
        title="文件能够完整读取，但已知有 221 帧没有保存",
    )
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.set_xlim(0, 20)
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("20 秒真实录像检查：完整性未通过", fontsize=17)
    with (ROOT / "recording_check.png").open("xb") as file:
        fig.savefig(file, format="png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {k: v for k, v in result.items() if not isinstance(v, (list, dict))},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
