"""普通运行不创建逐帧探针，仍记录真实时间、缺帧、拒绝原因及保存确认。"""

import csv
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from customer_workspace.microbubble_tracking import (
    paced_replay,
    paced_source,
    stall_probe,
)
from customer_workspace.microbubble_tracking.memory_pressure_check import (
    build_replay_command,
)
from customer_workspace.microbubble_tracking.memory_pressure_report import summarize_run
from customer_workspace.microbubble_tracking.tests.test_online_tracking import (
    image,
    processor,
    rows,
    stamp,
)
from customer_workspace.microbubble_tracking.tests.test_source_warmup import (
    source_inputs,
)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], (False, False, False)),
        (["--diagnostics"], (True, True, True)),
        (["--measure-stages"], (True, False, False)),
        (["--diagnose-stalls"], (False, True, False)),
        (["--diagnose-stalls", "--diagnose-details"], (False, True, True)),
    ],
)
def test_replay_cli_requires_explicit_diagnostics(
    tmp_path, monkeypatch, flags, expected
):
    called = {}

    def record(*args, **kwargs):
        called.update(kwargs)
        return {}

    monkeypatch.setattr(paced_replay, "paced_replay", record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paced_replay.py",
            "--video",
            "video.mp4",
            "--reference",
            "background.npy",
            "--pzt",
            "pzt",
            "--output-dir",
            str(tmp_path / "run"),
            *flags,
        ],
    )
    paced_replay.main()
    assert (
        tuple(
            called[name]
            for name in (
                "measure_stages",
                "diagnose_stalls",
                "diagnose_details",
            )
        )
        == expected
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_memory_observer_does_not_silently_enable_frame_probes(tmp_path, enabled):
    args = SimpleNamespace(
        video=Path("video.mp4"),
        reference=Path("background.npy"),
        pzt=Path("pzt"),
        repeats=3,
        diagnostics=enabled,
    )
    command = build_replay_command(args, tmp_path)
    assert ("--diagnostics" in command) is enabled
    assert "--diagnose-stalls" not in command
    assert "--measure-stages" not in command
    del args.diagnostics  # 旧的 Python 调用方式也应保持普通运行。
    assert "--diagnostics" not in build_replay_command(args, tmp_path)


def test_normal_source_and_tracker_keep_records_without_creating_probes(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Normal operation must not create per-frame probes")

    monkeypatch.setattr(stall_probe.StallProbe, "__init__", forbidden)
    monkeypatch.setattr(stall_probe.HandoffTrace, "__init__", forbidden)
    callbacks = list(gc.callbacks)
    with processor(tmp_path / "tracker", save_mode="async") as tracker:
        for index in (274, 275, 278, 279):
            result = tracker.process(image((20, 20, 12, 12)), stamp(index))
            assert "timings" not in result
            assert gc.callbacks == callbacks
        with pytest.raises(ValueError):
            tracker.process(image(), stamp(279))
    saved = rows(tracker.output, "SELECT * FROM frames ORDER BY source_frame")
    assert [r["source_frame"] for r in saved] == [274, 275, 278, 279]
    assert [r["source_time_s"] for r in saved] == pytest.approx(
        [i / 60 for i in (274, 275, 278, 279)]
    )
    assert [r["gap_frames"] for r in saved] == [0, 0, 2, 0]
    assert len(rows(tracker.output, "SELECT * FROM input_rejections")) == 1
    report = json.loads((tracker.output / "writer_report.json").read_text())
    assert report["all_submitted_saved"] and report["committed_frames"] == 4
    assert not list(tracker.output.glob("*stalls*"))

    source = tmp_path / "source"
    source.mkdir()
    _, _, config, slot, ready, start, done, cancel = source_inputs(source, monkeypatch)
    paced_source.send_video(
        config, slot, ready, start, done, cancel, SimpleNamespace(value=10.0)
    )
    assert (
        json.loads((source / "source_report.json").read_text())["status"] == "completed"
    )
    with (source / "source_timing.csv").open() as file:
        sent = list(csv.DictReader(file))
    assert [int(r["source_frame"]) for r in sent] == [274, 275, 276]
    assert all(r["published_s"] and "replaced_video_frame" in r for r in sent)
    assert not list(source.glob("*stalls*"))
    assert gc.callbacks == callbacks


def test_missing_conversion_details_are_not_reported_as_zero_stalls(tmp_path):
    replay = tmp_path / "replay"
    replay.mkdir()
    settings = replay / "pacing_settings.json"
    settings.write_text(json.dumps({"diagnose_stalls": False}))
    (tmp_path / "run.json").write_text(json.dumps({"status": "completed"}))
    for name in ("linux_memory.jsonl", "windows_memory.jsonl"):
        (tmp_path / name).write_text("")
    result = summarize_run(tmp_path)
    assert result["conversion_timing_recorded"] is False
    assert result["conversion_ms"]["max"] is None
    assert result["conversion_over_10_ms"] is None
    settings.write_text(json.dumps({"diagnose_stalls": True}))
    with pytest.raises(FileNotFoundError):
        summarize_run(tmp_path)
