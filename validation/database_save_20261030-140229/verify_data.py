"""Run from project root: .venv/bin/python - < this_file.py."""

import json
import sqlite3
import subprocess
from pathlib import Path

import cv2
import numpy as np

from customer_workspace.microbubble_tracking.async_store import SaveSettings
from customer_workspace.microbubble_tracking.online_replay import replay

ROOT = Path.cwd()
CODE = ROOT / "customer_workspace/microbubble_tracking"
OUT = ROOT / "customer_workspace/validation/database_save_20261030-140229"
commands = []
for policy in ("auto", "deferred"):
    command = [
        str(ROOT / "customer_workspace/dependencies/strace/installed/usr/bin/strace"),
        "--seccomp-bpf", "-f", "-ttt", "-T", "-yy",
        "-e", "trace=fdatasync,fsync", "-o", str(OUT / f"trace_{policy}_syscalls.log"),
        str(ROOT / ".venv/bin/python"), str(CODE / "paced_replay.py"),
        "--video", str(CODE / "data/raw/expB.mp4"),
        "--reference", str(CODE / "results/A_offline/background.npy"),
        "--pzt", str(CODE / "data/pzt/expB"),
        "--output-dir", str(OUT / f"trace_{policy}"),
        "--opencv-threads", "24", "--conversion-threads", "1",
        "--max-frames", "600", "--save-checkpoint-policy", policy,
        "--diagnose-stalls", "--measure-stages",
    ]
    commands.append(command)
    with (OUT / f"trace_{policy}.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    print("Completed syscall trace", policy, flush=True)
with (OUT / "trace_commands.json").open("x") as file:
    json.dump(commands, file, indent=2)

cv2.setNumThreads(24)
for policy in ("auto", "deferred"):
    report = replay(
        CODE / "data/raw/expB.mp4", CODE / "results/A_offline/background.npy",
        CODE / "data/pzt/expB", OUT / f"identity_{policy}",
        save_mode="async", save_settings=SaveSettings(checkpoint_policy=policy),
    )
    print("Completed identical-input replay", policy, report["frames"], flush=True)

counts = {}
connections = [sqlite3.connect(f"file:{OUT}/identity_{policy}/tracking.sqlite?mode=ro", uri=True)
               for policy in ("auto", "deferred")]
try:
    for table in ("frames", "observations", "tracks", "events", "event_updates",
                  "track_updates", "input_rejections"):
        columns = [row[1] for row in connections[0].execute(f"PRAGMA table_info({table})")
                   if row[1] != "processing_ms"]
        query = f"SELECT {','.join(columns)} FROM {table} ORDER BY rowid"
        left, right = [connection.execute(query).fetchall() for connection in connections]
        assert left == right, table
        counts[table] = len(left)
    assert counts["frames"] == 1320
    for connection in connections:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
finally:
    for connection in connections:
        connection.close()
assert np.array_equal(
    np.load(OUT / "identity_auto/background_final.npy"),
    np.load(OUT / "identity_deferred/background_final.npy"),
)
with (OUT / "identity_check.json").open("x") as file:
    json.dump({"equal_tables": counts, "background_equal": True,
               "excluded_columns": ["frames.processing_ms"],
               "input": str(CODE / "data/raw/expB.mp4")}, file, indent=2)
print("All saved scientific values match", counts, flush=True)
