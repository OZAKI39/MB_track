"""逐帧接口的时间、生命周期、落盘和实验关联检查；不连接硬件。"""

import csv
import json
import sqlite3
from dataclasses import replace

import cv2
import numpy as np
import pytest

from customer_workspace.microbubble_tracking.bubble_tracker import Tracker, TrackParams
from customer_workspace.microbubble_tracking.online_replay import link_actions
from customer_workspace.microbubble_tracking.online_store import export_tables
from customer_workspace.microbubble_tracking.online_tracking import (
    TIMING_FIELDS,
    FrameStamp,
    OnlineProcessor,
)


def test_stage_measurements_are_additive_and_do_not_change_results(tmp_path):
    measured = processor(tmp_path / "measured", measure_stages=True)
    normal = processor(tmp_path / "normal")
    with measured, normal:
        for index in range(8):
            frame = image((20 + index, 20, 12, 12))
            actual = measured.process(frame, stamp(index))
            expected = normal.process(frame, stamp(index))
            assert actual["observations"] == expected["observations"]
            assert actual["events"] == expected["events"]
            assert actual["track_updates"] == expected["track_updates"]
            for (aid, ac), (eid, ec) in zip(actual["contours"], expected["contours"]):
                assert aid == eid and np.array_equal(ac, ec)
            timing = actual["timings"]
            assert set(timing) == set(TIMING_FIELDS)
            assert all(value >= 0 for value in timing.values())
            assert sum(timing[key] for key in TIMING_FIELDS[:11]) == pytest.approx(
                timing["process_total_ms"], abs=1e-8
            )
            assert (
                sum(timing[key] for key in TIMING_FIELDS if key.startswith("seg_"))
                <= timing["segment_ms"]
            )
            assert "timings" not in expected
        assert measured.store.db.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert measured.store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_local_pixel_collection_preserves_holes_nested_labels_and_border():
    labels = np.zeros((80, 80), np.uint16)
    labels[10:35, 10:35] = 1
    labels[15:30, 15:30] = 0
    labels[19:25, 19:25] = 2
    labels[:8, 55:68] = 3
    contours = [
        cv2.findContours(
            (labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )[0][0]
        for label in (1, 2, 3)
    ]
    tracker = Tracker(labels.shape, TrackParams())
    tracker.step(0, labels, contours)
    for label in (1, 2, 3):
        expected_y, expected_x = np.nonzero(labels == label)
        assert np.array_equal(tracker.tracks[label].ys, expected_y)
        assert np.array_equal(tracker.tracks[label].xs, expected_x)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 255.1])
def test_float_image_validation_is_preserved(tmp_path, invalid):
    with processor(tmp_path) as p:
        frame = image().astype(np.float32)
        frame[40, 40] = invalid
        with pytest.raises(ValueError, match="Image values"):
            p.process(frame, stamp(0))
        assert p.last_stamp is None


def image(*rectangles):
    frame = np.full((80, 80), 200, np.uint8)
    for x, y, width, height in rectangles:
        frame[y : y + height, x : x + width] = 40
    return frame


def stamp(frame, time_s=None, **kwargs):
    return FrameStamp(
        "experiment",
        "camera",
        frame,
        frame / 60 if time_s is None else time_s,
        **kwargs,
    )


def processor(tmp_path, **kwargs):
    kwargs.setdefault("save_mode", "sync")
    return OnlineProcessor(
        experiment_id="experiment",
        stream_id="camera",
        source_clock="synthetic test clock",
        background=np.full((80, 80), 200, np.float32),
        nominal_fps=60,
        output_dir=tmp_path / "run",
        **kwargs,
    )


def rows(path, query):
    with sqlite3.connect(path / "tracking.sqlite") as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(query)]


def test_real_frame_time_and_five_actual_observations(tmp_path):
    with processor(tmp_path, max_gap_s=0.5) as p:
        for index, (frame, time_s) in enumerate(
            [(100, 1.0), (102, 1.05), (105, 1.15), (107, 1.2), (109, 1.25)]
        ):
            result = p.process(
                image((20 + index * 2, 20, 12, 12)), stamp(frame, time_s)
            )
            measurement = result["observations"][0]
            assert measurement["track_id"] == 1
            assert measurement["status_at_frame"] == (
                "confirmed" if index == 4 else "tentative"
            )
            if index == 2:
                assert measurement["dt_s"] == pytest.approx(0.1)
                assert measurement["raw_vx_px_s"] == pytest.approx(20)
                assert measurement["vx_px_s"] == pytest.approx(20)
            # 第二个连接在 process 返回时就可读到数据，不必等 close。
            assert len(rows(p.output, "SELECT * FROM frames")) == index + 1
        assert p.background.k == 5
        assert p.skipped_frames == 5
        event = result["events"][0]
        assert event["anchor_frame"] == 100 and event["update_frame"] == 109
    tracks = rows(p.output, "SELECT * FROM tracks")
    assert tracks[0]["confirmed_frame"] == 109
    assert not p.tracker.tracks


@pytest.mark.parametrize(
    "change",
    [
        dict(source_frame=10),
        dict(source_frame=9),
        dict(source_frame=-1),
        dict(source_frame=10.5),
        dict(source_frame=True),
        dict(source_time_s=float("nan")),
        dict(source_time_s=float("inf")),
        dict(source_time_s=1.0),
        dict(source_time_s=0.9),
        dict(experiment_id="other"),
        dict(stream_id="restart"),
        dict(video_frame=2),
        dict(received_windows_s=float("nan")),
    ],
)
def test_invalid_input_is_logged_without_advancing(tmp_path, change):
    with processor(tmp_path) as p:
        p.process(image((20, 20, 12, 12)), stamp(10, 1.0))
        with pytest.raises(ValueError):
            p.process(image(), replace(stamp(11, 1.02), **change))
        assert p.last_stamp.source_frame == 10
        assert p.background.k == 1
        p.process(image((21, 20, 12, 12)), stamp(11, 1.02))
    assert p.summary["input_rejections"] == 1 and p.summary["frames"] == 2


def test_bad_image_and_video_mapping_rejected(tmp_path):
    with processor(tmp_path) as p:
        p.process(image(), stamp(10, 1, video_frame=0, video_time_s=0))
        with pytest.raises(ValueError):
            p.process(np.zeros((10, 10), np.uint8), stamp(11, 1.02))
        with pytest.raises(ValueError):
            p.process(image(), stamp(11, 1.02, video_frame=0, video_time_s=0))
        assert p.background.k == 1


@pytest.mark.parametrize("next_stamp", [stamp(109, 1.1), stamp(101, 2.0)])
def test_long_frame_or_time_gap_ends_old_id(tmp_path, next_stamp):
    with processor(tmp_path) as p:
        p.process(image((20, 20, 12, 12)), stamp(100, 1.0))
        result = p.process(image((20, 20, 12, 12)), next_stamp)
        assert result["observations"][0]["track_id"] == 2
        assert list(p.tracker.tracks) == [2]
        lost = [e for e in result["events"] if e["kind"] == "lost"][0]
        assert lost["anchor_frame"] == 100
        assert lost["reported_frame"] == next_stamp.source_frame
        assert lost["interval_from_s"] == 1.0
        assert lost["interval_to_s"] == next_stamp.source_time_s
        assert lost["update_status"] == "pending"
    assert p.summary["tracks"] == 2


def test_split_confirmation_then_merge(tmp_path):
    whole = image((20, 25, 40, 14))
    split = image((20, 25, 14, 14), (46, 25, 14, 14))
    with processor(tmp_path) as p:
        for index in range(5):
            p.process(whole, stamp(index))
        result = p.process(split, stamp(5))
        event = [e for e in result["events"] if e["kind"] == "split"][0]
        assert event["status"] == "pending"
        event_id = event["event_id"]
        for index in range(6, 10):
            result = p.process(split, stamp(index))
        confirmed = [e for e in result["events"] if e["event_id"] == event_id][0]
        assert confirmed["anchor_frame"] == 5
        assert confirmed["decided_frame"] == 9
        assert confirmed["update_status"] == "confirmed_tracks"
        result = p.process(whole, stamp(10))
        assert any(
            e["kind"] == "merge" and e["status"] == "confirmed_tracks"
            for e in result["events"]
        )
        assert len(p.tracker.tracks) == 1
    events = rows(p.output, "SELECT * FROM events")
    assert any(e["kind"] == "split" for e in events)
    assert any(e["kind"] == "merge" for e in events)


def test_short_lived_and_unfinished_targets_not_erased(tmp_path):
    with processor(tmp_path) as p:
        p.process(image((20, 20, 12, 12)), stamp(0))
        p.process(image(), stamp(9))
        p.process(image((20, 20, 12, 12)), stamp(10))
    states = rows(p.output, "SELECT * FROM events ORDER BY event_id")
    assert [e["status"] for e in states] == [
        "rejected_short_track",
        "rejected_short_track",
        "unresolved_at_end",
    ]
    assert len(rows(p.output, "SELECT * FROM observations")) == 2
    assert [t["track_id"] for t in rows(p.output, "SELECT * FROM tracks")] == [1, 2]


def test_known_clock_resolution_does_not_break_gap_boundary(tmp_path):
    with processor(tmp_path, source_time_resolution_s=1 / 90000) as p:
        p.process(image((20, 20, 12, 12)), stamp(994, 16.566644444444446))
        result = p.process(image((20, 20, 12, 12)), stamp(1002, 16.69998888888889))
        assert result["observations"][0]["track_id"] == 1
        # 余量只处理时钟舍入，不允许明显超时仍沿用编号。
        result = p.process(image((20, 20, 12, 12)), stamp(1010, 16.84))
        assert result["observations"][0]["track_id"] == 2


def test_history_grows_on_disk_not_in_tracker(tmp_path):
    with processor(tmp_path) as p:
        for cycle in range(200):
            p.process(image((20, 20, 12, 12)), stamp(cycle * 10))
            p.process(image(), stamp(cycle * 10 + 9))
            assert not p.tracker.tracks
            assert (
                not p.tracker.rows and not p.tracker.events and not p.tracker.contours
            )
        assert len(rows(p.output, "SELECT track_id FROM tracks")) == 200
        assert len(rows(p.output, "SELECT source_frame FROM frames")) == 400
        observation = rows(p.output, "SELECT contour_i32 FROM observations LIMIT 1")[0]
        assert (
            np.frombuffer(observation["contour_i32"], dtype="<i4")
            .reshape(-1, 2)
            .shape[0]
            > 10
        )
    assert p.summary["peak_active_tracks"] == 1
    export_tables(p.output / "tracking.sqlite", p.output / "csv")
    with (p.output / "csv/observations.csv").open() as file:
        exported = list(csv.DictReader(file))
    assert len(exported) == 200
    assert all(r["experiment_id"] == "experiment" for r in exported)


def test_transaction_failure_stops_processing_preserves_committed_frame(tmp_path):
    p = processor(tmp_path)
    p.process(image((20, 20, 12, 12)), stamp(0))
    p.store.db.execute("""CREATE TRIGGER fail_write BEFORE INSERT ON observations
                         BEGIN SELECT RAISE(ABORT,'test disk write failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="test disk write failure"):
        p.process(image((21, 20, 12, 12)), stamp(1))
    with pytest.raises(RuntimeError):
        p.process(image(), stamp(2))
    assert len(rows(p.output, "SELECT * FROM frames")) == 1
    p.close()
    assert p.summary["status"] == "failed"
    assert not (p.output / "background_final.npy").exists()


def test_cancel_retains_records_and_is_not_disappearance(tmp_path):
    p = processor(tmp_path)
    p.process(image((20, 20, 12, 12)), stamp(100))
    p.close("cancelled")
    assert p.summary["status"] == "cancelled"
    assert (
        rows(p.output, "SELECT end_reason FROM tracks")[0]["end_reason"]
        == "stream_cancelled"
    )
    assert [e["kind"] for e in rows(p.output, "SELECT kind FROM events")] == ["appear"]
    assert not p.tracker.tracks
    with pytest.raises(FileExistsError):
        processor(tmp_path)


def test_interrupt_in_middle_of_frame_does_not_save_uncommitted_background(
    tmp_path, monkeypatch
):
    p = processor(tmp_path)
    p.process(image((20, 20, 12, 12)), stamp(0))

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(p.background, "update", interrupt)
    with pytest.raises(KeyboardInterrupt):
        with p:
            p.process(image((22, 20, 12, 12)), stamp(1))
    assert p.summary["status"] == "cancelled"
    assert p.summary["frames"] == 1
    assert not p.tracker.tracks
    assert not (p.output / "background_final.npy").exists()


def test_time_based_prediction_agrees_at_uniform_intervals():
    legacy = Tracker((80, 80), TrackParams())
    online = Tracker(
        (80, 80), TrackParams(), retain_history=False, time_based=True, max_gap_s=8 / 60
    )
    for frame in range(40):
        mask = np.zeros((80, 80), np.uint8)
        mask[20:30, 10 + frame : 20 + frame] = 1
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        old = legacy.step(frame, mask.astype(np.uint16), contours)
        new = online.step(
            frame + 274, mask.astype(np.uint16), contours, timestamp_s=frame / 60
        )
        assert old["rows"][0]["id"] == new["rows"][0]["id"]
        assert new["rows"][0]["vx"] == pytest.approx(old["rows"][0]["vx"] * 60)
        assert np.array_equal(old["contours"][0][1], new["contours"][0][1])


def write_actions(path, experiment="experiment"):
    row = dict(action_id="001", experiment_id=experiment)
    for kind in ("command", "confirmation"):
        row.update(
            {
                kind + "_video_mapped": "True",
                kind + "_video_first_frame": 0,
                kind + "_video_last_frame": 2,
            }
        )
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, row.keys())
        writer.writeheader()
        writer.writerow(row)


def test_action_links_keep_distinct_source_and_video_ids(tmp_path):
    with processor(tmp_path) as p:
        for index in (0, 2):
            p.process(
                image(),
                stamp(
                    274 + index, index / 60, video_frame=index, video_time_s=index / 60
                ),
            )
    actions = tmp_path / "actions.csv"
    write_actions(actions)
    report = link_actions(
        p.output / "tracking.sqlite", actions, tmp_path / "linked.csv"
    )
    with (tmp_path / "linked.csv").open() as file:
        row = next(csv.DictReader(file))
    assert json.loads(row["command_processed_source_frames"]) == [274, 276]
    assert json.loads(row["command_processed_video_frames"]) == [0, 2]
    assert row["command_tracking_coverage"] == "partial"
    assert report["incomplete_action_ranges"] == 2
    write_actions(actions, experiment="wrong")
    with pytest.raises(ValueError, match="different experiment"):
        link_actions(p.output / "tracking.sqlite", actions, tmp_path / "wrong.csv")
