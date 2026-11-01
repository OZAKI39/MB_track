"""相机接入的所有权、真实编号、错误收尾与设备权限边界；不连接硬件。"""

import csv
import io
import json
import multiprocessing as mp
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from customer_workspace.parallel_camera import ParallelCameraStream
from customer_workspace.parallel_camera_bridge import (
    LatestFrames,
    decoder_packet,
    read_exact,
)
from customer_workspace.parallel_video_check import run_check


def test_latest_slot_never_overwrites_borrowed_image():
    frames = LatestFrames(mp.get_context("spawn"), (8, 12))
    first, image = frames.acquire()
    image.fill(17)
    frames.publish(first, {"seq": 1})
    borrowed, old, _metadata = frames.take()
    for seq in range(2, 30):
        index, new = frames.acquire()
        new.fill(seq)
        frames.publish(index, {"seq": seq})
        assert np.all(old == 17)
    latest, image, header = frames.take()
    assert header["seq"] == 29
    assert np.all(image == 29)
    assert frames.overwritten == 27
    frames.release(borrowed)
    frames.release(latest)
    with pytest.raises(RuntimeError, match="not owned"):
        frames.release(latest)


def test_read_into_handles_short_reads_and_eof():
    class ShortReader(io.BytesIO):
        def readinto(self, target):
            return super().readinto(target[:3])

    target = bytearray(7)
    read_exact(ShortReader(b"1234567"), target)
    assert target == b"1234567"
    with pytest.raises(EOFError):
        read_exact(ShortReader(b"123"), target)


def test_live_rtsp_missing_decode_time_is_preserved_as_missing():
    # 历史真实相机的 DTS 均为 GST_CLOCK_TIME_NONE；只有 PTS 可用于运动时间。
    packet = decoder_packet(
        b"\x00\x00\x01\x65", {"pts_ns": 5183333334, "dts_ns": 2**64 - 1}
    )
    assert packet.pts == 5183333334
    assert packet.dts is None
    assert float(packet.time_base) == pytest.approx(1e-9)
    packet = decoder_packet(b"x", {"pts_ns": 16666666, "dts_ns": 16666666})
    assert packet.dts == packet.pts


def test_archived_video_can_never_start_afg(tmp_path):
    class Controller:
        def start(self, output):
            pytest.fail("AFG must not start during video simulation")

    with pytest.raises(ValueError, match="never"):
        run_check(SimpleNamespace(offline_video=Path("archive.mp4")), Controller())


@pytest.mark.parametrize(
    "failure", [None, RuntimeError("broken stream"), KeyboardInterrupt()]
)
@pytest.mark.parametrize("diagnose_analysis", [False, True])
def test_real_stamps_and_saved_records_survive_shutdown(
    tmp_path, monkeypatch, failure, diagnose_analysis
):
    background = tmp_path / "background.npy"
    np.save(background, np.full((64, 64), 180, np.float32))
    stopped = []

    class Bridge:
        def __init__(self, output, seconds, shape, **kwargs):
            self.output = output
            self.error = None
            self.ready = True
            self.window = {"frames": 3, "first_seq": 10, "last_seq": 12, "seconds": 1}
            self.process = SimpleNamespace(is_alive=lambda: False)
            self.done = SimpleNamespace(is_set=lambda: True)
            self.frames = SimpleNamespace(
                take=self.take,
                release=lambda index: None,
                overwritten=0,
                pending=SimpleNamespace(value=-1),
            )
            self.count = 0

        def start(self):
            assert (self.output / "tracking/analysis_ready.json").is_file()
            return self

        def poll(self):
            pass

        def take(self):
            if self.count == 1 and failure is not None:
                raise failure
            if self.count >= 2:
                return None
            seq = (10, 12)[self.count]
            self.count += 1
            session = json.loads((self.output / "link_session.json").read_text())
            now = time.perf_counter()
            return (
                0,
                np.full((64, 64), 180, np.uint8),
                {
                    **session,
                    "seq": seq,
                    "source_time_s": seq / 60,
                    "video_frame": seq - 10,
                    "video_time_s": (seq - 10) / 60,
                    "received_windows_mono": 9876 + seq / 60,
                    "arrived_wsl_s": now,
                    "decoded_wsl_s": now,
                    "published_wsl_s": now,
                    "request_ms": 1,
                    "sender_frame_age_ms": 1,
                    "latency": {},
                },
            )

        def stop(self):
            stopped.append(True)

        def acknowledge(self, header, completed):
            assert completed >= header["published_wsl_s"]

    monkeypatch.setattr(
        "customer_workspace.parallel_video_check.TrackingWindowsBridge", Bridge
    )
    args = SimpleNamespace(
        background=background,
        output_dir=tmp_path / "run",
        seconds=1,
        mode="baseline",
        opencv_threads=1,
        diagnose_analysis=diagnose_analysis,
    )
    if failure is None:
        report = run_check(args)
        assert report["processed"] == report["saved"] == 2
        assert report["analysis_skipped"] == 1
    else:
        with pytest.raises(RuntimeError):
            run_check(args)
    assert stopped == [True]
    with sqlite3.connect(args.output_dir / "tracking/tracking.sqlite") as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        rows = db.execute(
            "SELECT source_frame,gap_frames,delta_time_s FROM frames ORDER BY source_frame"
        ).fetchall()
        assert rows[0][0] == 10
        if failure is None:
            assert rows[1][0:2] == (12, 1)
            assert rows[1][2] == pytest.approx(2 / 60)
        else:
            assert len(rows) == 1
    summary = json.loads((args.output_dir / "tracking/summary.json").read_text())
    expected = (
        "completed"
        if failure is None
        else "cancelled"
        if isinstance(failure, KeyboardInterrupt)
        else "failed"
    )
    assert summary["status"] == expected
    assert summary["persistence"]["pending_records"] == 0
    with (args.output_dir / "analysis_frames.csv").open() as file:
        assert len(list(csv.DictReader(file))) == len(rows)
    diagnostic_file = args.output_dir / "consumer_stalls.csv"
    assert diagnostic_file.exists() is diagnose_analysis
    if diagnose_analysis:
        with diagnostic_file.open() as file:
            records = list(csv.DictReader(file))
        assert {int(row["frame"]) for row in records} == {row[0] for row in rows}
        assert (args.output_dir / "analysis_finished.json").is_file()


def test_compressed_gate_warms_before_window_and_stops_after_window():
    stream = ParallelCameraStream.__new__(ParallelCameraStream)
    stream.Gst = SimpleNamespace(PadProbeReturn=SimpleNamespace(OK="OK", DROP="DROP"))
    stream.record_closed = stream.analysis_started = False
    stream.by_pts = {
        1: {"nal_types": [1]},
        2: {"nal_types": [5, 7, 8]},
        3: {"video_frame": 0},
        4: {},
    }
    info = lambda pts: SimpleNamespace(get_buffer=lambda: SimpleNamespace(pts=pts))
    assert stream._tracking_gate(None, info(1)) == "DROP"
    assert stream._tracking_gate(None, info(2)) == "OK"
    stream.record_closed = True
    assert stream._tracking_gate(None, info(3)) == "OK"
    assert stream._tracking_gate(None, info(4)) == "DROP"
