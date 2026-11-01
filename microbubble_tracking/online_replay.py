"""用已保存的实验录像验证逐帧接口；只读历史 PZT 数据，不控制设备。"""

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime
from contextlib import nullcontext
from itertools import chain
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .bubble_tracker import combine_backgrounds
    from .online_store import export_tables
    from .online_tracking import TIMING_FIELDS, FrameStamp, OnlineProcessor
else:
    from bubble_tracker import combine_backgrounds
    from online_store import export_tables
    from online_tracking import TIMING_FIELDS, FrameStamp, OnlineProcessor


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validated_frames(frames, timestamps, experiment_id):
    """共用的录像帧/原始帧时间表核查；完整遍历时也检查时间表尾部。"""
    previous_frame = previous_time = None
    for index, frame in enumerate(frames):
        row = next(timestamps, None)
        if (
            row is None
            or row["experiment_id"] != experiment_id
            or int(row["frame_index"]) != index
        ):
            raise ValueError(f"Frame mapping mismatch at video frame {index}")
        if frame.pts is None:
            raise ValueError(f"Video frame {index} has no PTS")
        source_time = float(row["video_s"])
        source_frame = int(row["source_seq"])
        if previous_frame is not None and (
            source_frame <= previous_frame or source_time <= previous_time
        ):
            raise ValueError("Archived source frame/time does not increase")
        error = abs(float(frame.pts * frame.time_base) - source_time)
        if not np.isfinite(error) or error > float(frame.time_base) + 1e-9:
            raise ValueError(f"PTS disagrees with frame mapping at frame {index}")
        previous_frame, previous_time = source_frame, source_time
        yield index, frame, row, error
    if next(timestamps, None) is not None:
        raise ValueError("Video ended before the archived frame table")


def verify_inputs(video: Path, reference: Path, pzt: Path) -> dict:
    """确认视频和时间表来自同一次实验；不能仅凭相似文件名建立关联。"""
    manifest = json.loads((pzt.parent / "manifest.json").read_text(encoding="utf-8"))
    matches = [e for e in manifest["experiments"] if e["name"] == pzt.name]
    if len(matches) != 1:
        raise ValueError("PZT directory is missing or ambiguous in the manifest")
    entry = matches[0]
    if file_hash(video) != entry["video_sha256"]:
        raise ValueError("Video hash does not match the experiment manifest")
    hashes = {}
    for name in ("link_session.json", "video_frames.csv", "actions.csv"):
        expected = next(
            f["sha256"] for f in entry["files"] if f["path"] == f"{pzt.name}/{name}"
        )
        actual = file_hash(pzt / name)
        if actual != expected:
            raise ValueError(f"Archived synchronization file changed: {name}")
        hashes[name] = actual
    session = json.loads((pzt / "link_session.json").read_text(encoding="utf-8"))
    if session["experiment_id"] != entry["experiment_id"]:
        raise ValueError("Experiment identity differs between session and manifest")
    return dict(
        experiment_id=entry["experiment_id"],
        video=str(video.resolve()),
        video_sha256=entry["video_sha256"],
        reference=str(reference.resolve()),
        reference_sha256=file_hash(reference),
        pzt=str(pzt.resolve()),
        synchronization_hashes=hashes,
        expected_frames=entry["video_frame_rows"],
        expected_actions=entry["action_rows"],
        clock_margin_s=entry["clock_margin_s"],
        precision_notes=entry.get(
            "precision_notes", manifest.get("precision_notes", [])
        ),
    )


def link_actions(database: Path, actions: Path, output: Path) -> dict:
    """按原有视频范围查已处理帧，不把跳过的画面冒充已经识别的画面。"""
    db = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        experiment = json.loads(
            db.execute(
                "SELECT value FROM session WHERE key='experiment_id'"
            ).fetchone()[0]
        )
        stream = json.loads(
            db.execute("SELECT value FROM session WHERE key='stream_id'").fetchone()[0]
        )
        count = missing = 0
        with actions.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            extra = ["stream_id"]
            for kind in ("command", "confirmation"):
                extra.extend(
                    f"{kind}_{name}"
                    for name in (
                        "processed_source_frames",
                        "processed_video_frames",
                        "tracking_coverage",
                    )
                )
            with output.open("x", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, [*reader.fieldnames, *extra])
                writer.writeheader()
                for row in reader:
                    if row["experiment_id"] != experiment:
                        raise ValueError("PZT action belongs to a different experiment")
                    count += 1
                    row["stream_id"] = stream
                    for kind in ("command", "confirmation"):
                        prefix = kind + "_video_"
                        records = []
                        coverage = "unmapped"
                        if row[prefix + "mapped"].lower() == "true":
                            first, last = (
                                int(row[prefix + "first_frame"]),
                                int(row[prefix + "last_frame"]),
                            )
                            if first < 0 or last < first:
                                raise ValueError("Invalid action video range")
                            records = db.execute(
                                "SELECT source_frame,video_frame FROM frames "
                                "WHERE video_frame BETWEEN ? AND ? ORDER BY video_frame",
                                (first, last),
                            ).fetchall()
                            coverage = (
                                "complete"
                                if len(records) == last - first + 1
                                else "partial"
                                if records
                                else "none"
                            )
                        if coverage != "complete":
                            missing += 1
                        row[kind + "_processed_source_frames"] = json.dumps(
                            [r["source_frame"] for r in records]
                        )
                        row[kind + "_processed_video_frames"] = json.dumps(
                            [r["video_frame"] for r in records]
                        )
                        row[kind + "_tracking_coverage"] = coverage
                    writer.writerow(row)
        return dict(actions=count, incomplete_action_ranges=missing)
    finally:
        db.close()


def replay(
    video: Path,
    reference: Path,
    pzt: Path,
    output: Path,
    *,
    drop_every=0,
    measure_stages=False,
    save_mode="sync",
    save_settings=None,
) -> dict:
    """顺序解码原始 PTS；drop_every 只供跳帧验证，不改变原录像/时间表。"""
    import av  # 仅录像回放需要；核心接口不依赖视频文件或 PyAV。

    if drop_every < 0 or drop_every == 1:
        raise ValueError("drop_every must be 0 or at least 2")
    provenance = verify_inputs(video, reference, pzt)
    experiment = provenance["experiment_id"]
    stream_id = "recording:" + provenance["video_sha256"]
    max_pts_error = 0.0
    decoded_count = omitted = 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        frames = container.decode(stream)
        first = next(frames)
        first_gray = cv2.cvtColor(
            first.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        reference_image = np.load(reference, allow_pickle=False)
        if reference_image.shape != first_gray.shape:
            raise ValueError("Reference background and video sizes differ")
        background = combine_backgrounds([reference_image], first_gray)
        processor = OnlineProcessor(
            experiment_id=experiment,
            stream_id=stream_id,
            source_clock="recording video PTS",
            background=background,
            nominal_fps=float(stream.average_rate),
            source_time_resolution_s=float(stream.time_base),
            output_dir=output,
            provenance=provenance,
            measure_stages=measure_stages,
            save_mode=save_mode,
            save_settings=save_settings,
        )
        # 复制经过哈希核查的少量时间记录，保持输出可独立追溯。
        with processor:
            for name in ("actions.csv", "video_frames.csv"):
                shutil.copyfile(pzt / name, output / ("source_" + name))
            timing_log = (
                (output / "stage_timing.csv").open("x", encoding="utf-8", newline="")
                if measure_stages
                else nullcontext(None)
            )
            with (
                (pzt / "video_frames.csv").open(
                    encoding="utf-8-sig", newline=""
                ) as file,
                timing_log as timing_file,
            ):
                writer = None
                if timing_file is not None:
                    writer = csv.DictWriter(
                        timing_file, ("video_frame", "source_frame", *TIMING_FIELDS)
                    )
                    writer.writeheader()
                timestamps = csv.DictReader(file)
                for index, frame, row, error in validated_frames(
                    chain([first], frames), timestamps, experiment
                ):
                    source_time = float(row["video_s"])
                    source_frame = int(row["source_seq"])
                    max_pts_error = max(max_pts_error, error)
                    decoded_count += 1
                    if drop_every and (index + 1) % drop_every == 0:
                        omitted += 1
                        continue
                    stamp = FrameStamp(
                        experiment,
                        stream_id,
                        source_frame,
                        source_time,
                        video_frame=index,
                        video_time_s=source_time,
                        received_windows_s=float(row["received_windows_s"]),
                        decoded_windows_s=float(row["decoded_windows_s"]),
                    )
                    result = processor.process(frame.to_ndarray(format="bgr24"), stamp)
                    if writer is not None:
                        writer.writerow(
                            dict(
                                video_frame=index,
                                source_frame=source_frame,
                                **result["timings"],
                            )
                        )
                    if index % 300 == 0:
                        print(
                            f"Video frame {index}; active tracks {len(processor.tracker.tracks)}",
                            flush=True,
                        )
                if decoded_count != provenance["expected_frames"]:
                    raise ValueError("Decoded video and archived frame counts differ")
    links = link_actions(
        output / "tracking.sqlite",
        output / "source_actions.csv",
        output / "action_links.csv",
    )
    if links["actions"] != provenance["expected_actions"]:
        raise ValueError("PZT action count differs from manifest")
    export_tables(output / "tracking.sqlite", output / "tables")
    report = dict(
        **processor.summary,
        **links,
        decoded_frames=decoded_count,
        intentionally_omitted_frames=omitted,
        drop_every=drop_every,
        max_video_pts_error_s=max_pts_error,
        scope="Offline sequential replay, no hardware connection or real-time pacing",
        timing_limit="Video receive-time ranges only; exposure and acoustic onset remain unknown",
    )
    (output / "replay_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--save-mode", choices=("sync", "async"), default="sync")
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Previously prepared background.npy",
    )
    parser.add_argument(
        "--pzt",
        required=True,
        type=Path,
        help="Archived data/pzt/expA or expB directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; existing results are never overwritten",
    )
    parser.add_argument(
        "--drop-every",
        type=int,
        default=0,
        help="Test only: omit every Nth frame; 0 disables",
    )
    parser.add_argument(
        "--measure-stages",
        action="store_true",
        help="Record per-frame processing stages",
    )
    args = parser.parse_args()
    output = args.output_dir or Path(
        __file__
    ).resolve().parent.parent / "validation" / (
        "online_replay_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    report = replay(
        args.video,
        args.reference,
        args.pzt,
        output,
        drop_every=args.drop_every,
        measure_stages=args.measure_stages,
        save_mode=args.save_mode,
    )
    print(json.dumps(report, indent=2))
    print(f"Records: {output.resolve()}")


if __name__ == "__main__":
    main()
