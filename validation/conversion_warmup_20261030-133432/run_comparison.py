"""Alternate independent cold and warmed startups with identical replay inputs."""

import csv
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = ROOT / "customer_workspace/microbubble_tracking"
PYTHON = ROOT / ".venv/bin/python"


def main():
    commands = []
    results = []
    for pair in range(1, 13):
        order = (0, 8) if pair % 2 else (8, 0)
        for iterations in order:
            label = "cold" if iterations == 0 else "warm"
            name = f"pair_{pair:02d}_{label}"
            output = HERE / name
            command = [
                str(PYTHON), str(CODE / "paced_replay.py"),
                "--video", str(CODE / "data/raw/expB.mp4"),
                "--reference", str(CODE / "results/A_offline/background.npy"),
                "--pzt", str(CODE / "data/pzt/expB"),
                "--output-dir", str(output), "--max-frames", "60",
                "--opencv-threads", "24", "--conversion-threads", "1",
                "--conversion-warmup-iterations", str(iterations),
                "--measure-stages", "--diagnose-stalls",
            ]
            commands.append(command)
            (HERE / "commands.json").write_text(json.dumps(commands, indent=2))
            started = time.perf_counter()
            with (HERE / f"{name}.log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=30)
            with (output / "source_stalls.csv").open(newline="") as file:
                rows = [row for row in csv.DictReader(file) if row["stage"] == "convert_bgr"]
            first = [row for row in rows if int(row["frame"]) < 5]
            warmup = json.loads((output / "source_warmup.json").read_text())
            result = {
                "name": name, "iterations": iterations,
                "elapsed_s": time.perf_counter() - started,
                "first5_max_ms": max(float(row["wall_ms"]) for row in first),
                "first60_max_ms": max(float(row["wall_ms"]) for row in rows),
                "first5_minor_faults": sum(int(row["minor_faults"]) for row in first),
                "warmup_ms": warmup["duration_ms"],
            }
            results.append(result)
            (HERE / "progress.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
    print("Completed all 24 startups", flush=True)


if __name__ == "__main__":
    main()
