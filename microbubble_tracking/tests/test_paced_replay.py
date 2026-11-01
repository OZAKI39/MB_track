"""第二轮的队列、帧映射、固定时刻等待和发送失败保护。"""

import csv
import json
import multiprocessing as mp
import sqlite3
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from customer_workspace.microbubble_tracking import paced_source
from customer_workspace.microbubble_tracking.online_replay import validated_frames
from customer_workspace.microbubble_tracking.paced_report import summarize


def metadata(index):
    return {
        "video_frame": index,
        "source_frame": 274 + index,
        "video_s": index / 60,
        "received_windows_s": 1000 + index / 60,
        "decoded_windows_s": 1001 + index / 60,
        "deadline_s": 10 + index / 60,
        "decode_ms": 0.5,
    }


def test_latest_slot_reports_every_overwritten_frame_and_copies_image():
    slot = paced_source.LatestFrame(mp.get_context("spawn"), (5, 6))
    image = np.full((5, 6), 100, np.uint8)
    first, replaced = slot.put(image, metadata(0))
    assert replaced is None
    _, replaced = slot.put(image, metadata(1))
    assert replaced == 0
    image[:] = 30
    copied, stamp = slot.take()
    assert np.all(copied == 100)
    assert stamp["source_frame"] == 275
    assert stamp["received_windows_s"] == 1000 + 1 / 60
    assert stamp["published_s"] >= first["published_s"]
    assert slot.take() is None
    assert not slot.available.is_set()
    _, replaced = slot.put(image, metadata(2))
    assert replaced is None
    assert np.all(copied == 100)


def test_latest_slot_rejects_size_or_dtype_mismatch():
    slot = paced_source.LatestFrame(mp.get_context("spawn"), (5, 6))
    for image in (np.zeros((4, 6), np.uint8), np.zeros((5, 6), np.float32)):
        with pytest.raises(ValueError):
            slot.put(image, metadata(0))
    assert slot.take() is None


def test_wait_uses_original_deadline_despite_late_previous_step(monkeypatch):
    now = [1.0]
    waits = []

    class Cancel:
        def is_set(self):
            now[0] += 0.0001
            return False

        def wait(self, duration):
            waits.append(duration)
            now[0] += duration + 0.0001

    monkeypatch.setattr(paced_source.time, "perf_counter", lambda: now[0])
    cancel = Cancel()
    assert paced_source.wait_for_deadline(1.02, cancel, 0.001)
    now[0] = 1.035  # 上一轮工作变慢；第二个期限仍是 1.04，不顺延。
    assert paced_source.wait_for_deadline(1.04, cancel, 0.001)
    assert waits[0] == pytest.approx(0.0189)
    assert waits[1] == pytest.approx(0.0039)
    assert 1.04 <= now[0] < 1.041
    assert paced_source.wait_for_deadline(1.03, cancel, 0.001)
    assert len(waits) == 2


def test_spin_wait_checks_cancel(monkeypatch):
    cancel = mp.get_context("spawn").Event()
    cancel.set()
    monkeypatch.setattr(paced_source.time, "perf_counter", lambda: 1.0)
    assert not paced_source.wait_for_deadline(1.0001, cancel, 0.001)


def mapped_row(index):
    return {
        "experiment_id": "exp",
        "frame_index": str(index),
        "source_seq": str(274 + index),
        "video_s": str(index / 60),
    }


def test_frame_mapping_preserves_original_indices_and_pts():
    frames = [
        SimpleNamespace(pts=i * 1500, time_base=Fraction(1, 90000)) for i in range(3)
    ]
    result = list(
        validated_frames(frames, iter([mapped_row(i) for i in range(3)]), "exp")
    )
    assert [int(item[2]["source_seq"]) for item in result] == [274, 275, 276]
    assert all(item[3] == 0 for item in result)


@pytest.mark.parametrize(
    "problem", ["experiment", "frame", "pts", "tail", "regression"]
)
def test_frame_mapping_rejects_inconsistent_recordings(problem):
    frames = [
        SimpleNamespace(pts=i * 1500, time_base=Fraction(1, 90000)) for i in range(2)
    ]
    rows = [mapped_row(i) for i in range(2)]
    if problem == "experiment":
        rows[0]["experiment_id"] = "other"
    elif problem == "frame":
        rows[1]["frame_index"] = "3"
    elif problem == "pts":
        frames[1].pts = 2000
    elif problem == "tail":
        rows.append(mapped_row(2))
    else:
        rows[1]["source_seq"] = rows[0]["source_seq"]
    with pytest.raises(ValueError):
        list(validated_frames(frames, iter(rows), "exp"))


def test_source_failure_wakes_consumer_and_keeps_reason(tmp_path, monkeypatch):
    context = mp.get_context("spawn")
    slot = paced_source.LatestFrame(context, (5, 6))
    ready, start, done, cancel = [context.Event() for _ in range(4)]
    monkeypatch.setattr(paced_source, "calibrate_wait", lambda *a: {"ahead_ms": 1})
    config = {"output": str(tmp_path), "video": str(tmp_path / "missing.mp4")}
    paced_source.send_video(
        config, slot, ready, start, done, cancel, context.RawValue("d", 0)
    )
    report = json.loads((tmp_path / "source_report.json").read_text())
    assert done.is_set() and slot.available.is_set()
    assert not ready.is_set()
    assert report["status"] == "failed"
    assert "missing.mp4" in report["error"]


@pytest.mark.parametrize("repeats", [1, 2])
def test_repeated_replay_has_explicit_original_mapping(tmp_path, monkeypatch, repeats):
    class Video:
        def __init__(self):
            self.streams = SimpleNamespace(video=[SimpleNamespace(average_rate=60)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, stream):
            return iter(
                SimpleNamespace(pts=i * 1500, time_base=Fraction(1, 90000))
                for i in range(3)
            )

    monkeypatch.setattr(paced_source.av, "open", lambda _: Video())
    timing = tmp_path / "times.csv"
    write_table(timing, [mapped_row(i) for i in range(3)])
    actual = list(
        paced_source.recording_frames(
            dict(
                video="unused",
                timestamps=str(timing),
                experiment_id="exp",
                expected_frames=3,
                repeats=repeats,
            )
        )
    )
    assert [r[0] for r in actual] == list(range(3 * repeats))
    assert [r[2]["original_source_frame"] for r in actual] == [274, 275, 276] * repeats
    if repeats == 1:
        assert [int(r[2]["source_seq"]) for r in actual] == [274, 275, 276]
    else:
        assert [int(r[2]["source_seq"]) for r in actual] == list(range(6))
        assert [r[2]["video_s"] for r in actual] == pytest.approx(
            [i / 60 for i in range(6)]
        )


def write_table(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("status", ["completed", "cancelled"])
def test_tail_frame_is_explicitly_accounted_or_rejected(tmp_path, status):
    """取消后的尾帧也要有去向；正常结束不能静默漏掉这张画面。"""
    (tmp_path / "pacing_settings.json").write_text(
        json.dumps({"nominal_fps": 60, "max_age_ms": 100})
    )
    (tmp_path / "run_state.json").write_text(
        json.dumps({"status": status, "error": None})
    )
    source = []
    for index in range(2):
        source.append(
            {
                "video_frame": index,
                "source_frame": 274 + index,
                "video_s": index / 60,
                "deadline_s": 10 + index / 60,
                "published_s": 10.001 + index / 60,
                "decode_ms": 1,
                "replaced_video_frame": "",
            }
        )
    write_table(tmp_path / "source_timing.csv", source)
    write_table(
        tmp_path / "consumer_timing.csv",
        [
            {
                "video_frame": 0,
                "source_frame": 274,
                "video_s": 0,
                "outcome": "processed",
                "queue_ms": 1,
                "process_ms": 5,
                "core_ms": 3,
                "age_ms": 7,
            }
        ],
    )
    with sqlite3.connect(tmp_path / "tracking.sqlite") as db:
        db.execute(
            "CREATE TABLE frames (video_frame INTEGER, source_frame INTEGER, source_time_s REAL)"
        )
        db.execute("INSERT INTO frames VALUES (0, 274, 0)")
    if status == "completed":
        with pytest.raises(ValueError, match="unaccounted"):
            summarize(tmp_path)
    else:
        report = summarize(tmp_path)
        assert report["abandoned_frames"] == 1
        assert report["all_offered_frames_accounted"]
        assert not report["all_frames_processed"]
        with (tmp_path / "frame_outcomes.csv").open() as file:
            rows = list(csv.DictReader(file))
        assert rows[-1]["outcome"] == "abandoned_on_stop"
