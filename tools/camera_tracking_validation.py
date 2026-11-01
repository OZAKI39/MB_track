"""使用现有 A/B 原视频验证相机接入；只模拟输入，不连接相机或 AFG。"""

import argparse
import csv
import hashlib
import json
import sqlite3
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import av
import cv2
import numpy as np

from customer_workspace.microbubble_tracking.bubble_tracker import combine_backgrounds
from customer_workspace.microbubble_tracking.online_replay import file_hash
from customer_workspace.microbubble_tracking.online_tracking import (
    FrameStamp,
    OnlineProcessor,
)
from customer_workspace.microbubble_tracking.paced_replay import paced_replay
from customer_workspace.parallel_video_check import run_check
from customer_workspace.streamcopy_recording import write_json
from microbubble_experiment.configuration import ROOT

CODE = ROOT / "customer_workspace/microbubble_tracking"
TABLES = (
    "frames",
    "observations",
    "tracks",
    "events",
    "event_updates",
    "track_updates",
    "input_rejections",
)


def fingerprint(database):
    """仅排除处理用时；保留全部轮廓、编号、速度、事件和输入时间。"""
    result = {}
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Database integrity check failed")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Database foreign key check failed")
        for table in TABLES:
            columns = [
                r[1]
                for r in db.execute(f"PRAGMA table_info({table})")
                if r[1] != "processing_ms"
            ]
            order = "source_frame,track_id" if table == "observations" else columns[0]
            digest, count = hashlib.sha256(), 0
            for row in db.execute(
                f"SELECT {','.join(columns)} FROM {table} ORDER BY {order}"
            ):
                digest.update(repr(row).encode())
                count += 1
            result[table] = {"rows": count, "sha256": digest.hexdigest()}
    return result


def verify_run(folder, source_video):
    """用原离线解码重新分析同一批真实输入帧，严格比较数据库科学结果。"""
    with (folder / "analysis_frames.csv").open() as file:
        stamps = {int(row["video_frame"]): row for row in csv.DictReader(file)}
    session = json.loads((folder / "tracking/session.json").read_text())
    reference = folder / "verification_reference"
    processor = OnlineProcessor(
        experiment_id=session["experiment_id"],
        stream_id=session["stream_id"],
        source_clock=session["source_clock"],
        nominal_fps=60,
        source_time_resolution_s=1 / 90000,
        background=np.load(folder / "tracking/background_initial.npy"),
        output_dir=reference,
    )
    cv2.setNumThreads(24)
    pixel_frames = different = 0
    try:
        processor.start()
        with (
            av.open(str(source_video)) as original,
            av.open(str(folder / "recording/video.mp4")) as recorded,
        ):
            original.streams.video[0].thread_count = 1
            recorded.streams.video[0].thread_count = 1
            source = iter(original.decode(video=0))
            for index, saved_frame in enumerate(recorded.decode(video=0)):
                raw = next(source)
                bgr = raw.to_ndarray(format="bgr24")
                if not np.array_equal(bgr, saved_frame.to_ndarray(format="bgr24")):
                    different += 1
                pixel_frames += 1
                if index not in stamps:
                    continue
                row = stamps[index]
                values = {}
                for field in fields(FrameStamp):
                    value = row[field.name]
                    if field.name in ("experiment_id", "stream_id"):
                        values[field.name] = value
                    elif not value:
                        values[field.name] = None
                    elif field.name in ("source_frame", "video_frame"):
                        values[field.name] = int(value)
                    else:
                        values[field.name] = float(value)
                processor.process(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), FrameStamp(**values)
                )
        processor.close()
    except BaseException:
        processor.close("failed")
        raise
    actual = fingerprint(folder / "tracking/tracking.sqlite")
    expected = fingerprint(reference / "tracking.sqlite")
    result = {
        "source_video": str(source_video),
        "source_sha256": file_hash(source_video),
        "compared_video_frames": pixel_frames,
        "different_video_frames": different,
        "compared_analysis_frames": len(stamps),
        "scientific_results_equal": actual == expected,
        "actual": actual,
        "expected": expected,
    }
    write_json(folder / "equivalence.json", result)
    if different or actual != expected:
        raise ValueError(f"Input/result equivalence failed: {folder}")
    return result


def repeated_video(source, output):
    """只为持续负载测试拼接压缩视频，不编码、不冒充新的真实实验。"""
    with av.open(str(source)) as template, av.open(str(output), "w") as target:
        stream = template.streams.video[0]
        target_stream = target.add_stream_from_template(stream)
        duration = stream.duration
        for cycle in range(3):
            with av.open(str(source)) as container:
                for packet in container.demux(video=0):
                    if packet.pts is None:
                        continue
                    packet.pts += cycle * duration
                    packet.dts += cycle * duration
                    packet.stream = target_stream
                    target.mux(packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--long", action="store_true", help="增加 A/B 各一段 62 秒的重复素材负载测试"
    )
    parser.add_argument(
        "--verify-only", type=Path, help="核对已完成运行；须同时指定 --video"
    )
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_run(args.verify_only, args.video), indent=2))
        return
    if not 1 <= args.rounds <= 5:
        parser.error("rounds must be 1–5")
    args.output.mkdir(parents=True, exist_ok=False)
    reference = CODE / "results/B_offline/background.npy"
    sources, backgrounds = {}, {}
    for experiment in ("A", "B"):
        video = CODE / f"data/raw/exp{experiment}.mp4"
        sources[experiment] = video
        with av.open(str(video)) as container:
            gray = cv2.cvtColor(
                next(container.decode(video=0)).to_ndarray(format="bgr24"),
                cv2.COLOR_BGR2GRAY,
            )
        background = combine_backgrounds(
            [np.load(reference)], gray.astype(np.float32), verbose=False
        )
        backgrounds[experiment] = args.output / f"background_{experiment}.npy"
        np.save(backgrounds[experiment], background)
    write_json(
        args.output / "inputs.json",
        {
            "videos": {
                k: {"path": str(v), "sha256": file_hash(v)} for k, v in sources.items()
            },
            "reference": str(reference),
            "reference_sha256": file_hash(reference),
            "background_preparation": "Unchanged combine_backgrounds(reference, first frame)",
            "rounds": args.rounds,
            "long_runs": args.long,
            "timing_note": "New simulated session IDs; archived PZT actions never executed",
        },
    )
    runs = []
    for experiment in ("A", "B"):
        print(f"Local paced reference {experiment}", flush=True)
        paced_replay(
            sources[experiment],
            reference,
            CODE / f"data/pzt/exp{experiment}",
            args.output / f"local_{experiment}",
            opencv_threads=24,
            conversion_threads=1,
        )
    for number in range(1, args.rounds + 1):
        for experiment in ("A", "B"):
            folder = args.output / f"{experiment}_{number:02d}"
            print(f"Windows recording + tracking: {folder.name}", flush=True)
            report = run_check(
                SimpleNamespace(
                    output_dir=folder,
                    seconds=22,
                    mode="parallel",
                    opencv_threads=24,
                    offline_video=sources[experiment],
                    background=backgrounds[experiment],
                )
            )
            runs.append((folder, sources[experiment]))
            print(
                json.dumps(
                    {
                        k: report[k]
                        for k in (
                            "processed",
                            "saved",
                            "analysis_skipped",
                            "processing_ms",
                            "recording_passed",
                        )
                    }
                ),
                flush=True,
            )
    if args.long:
        for experiment in ("A", "B"):
            video = args.output / f"synthetic_{experiment}_repeated.mp4"
            repeated_video(sources[experiment], video)
            folder = args.output / f"long_{experiment}"
            print(f"62-second synthetic load: {experiment}", flush=True)
            run_check(
                SimpleNamespace(
                    output_dir=folder,
                    seconds=62,
                    mode="parallel",
                    opencv_threads=24,
                    offline_video=video,
                    background=backgrounds[experiment],
                )
            )
            runs.append((folder, video))
    # 核对计算在计时实验全部结束后运行，避免同时抢占资源影响性能数据。
    for folder, video in runs:
        print(f"Exact input/result verification: {folder.name}", flush=True)
        verify_run(folder, video)
    write_json(
        args.output / "completed.json", {"runs": [str(folder) for folder, _ in runs]}
    )


if __name__ == "__main__":
    main()
