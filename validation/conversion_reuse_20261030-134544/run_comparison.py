"""Same-video comparison: existing startup warm-up versus warm-up plus reuse."""

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
    cases = []
    for pair in range(1, 13):
        modes = ("allocate", "reuse") if pair % 2 else ("reuse", "allocate")
        for mode in modes:
            cases.append((f"pair_{pair:02d}_{mode}", mode, "B", 60, 1))
    cases.extend([
        ("full_A_reuse", "reuse", "A", 0, 1),
        ("long_B_reuse", "reuse", "B", 0, 3),
    ])
    commands, results = [], []
    for name, mode, experiment, limit, repeats in cases:
        other = "A" if experiment == "B" else "B"
        command = [
            str(PYTHON), str(CODE / "paced_replay.py"),
            "--video", str(CODE / f"data/raw/exp{experiment}.mp4"),
            "--reference", str(CODE / f"results/{other}_offline/background.npy"),
            "--pzt", str(CODE / f"data/pzt/exp{experiment}"),
            "--output-dir", str(HERE / name),
            "--max-frames", str(limit), "--repeats", str(repeats),
            "--opencv-threads", "24", "--conversion-threads", "1",
            "--conversion-warmup-iterations", "8", "--conversion-memory", mode,
            "--measure-stages", "--diagnose-stalls",
        ]
        commands.append(command)
        (HERE / "commands.json").write_text(json.dumps(commands, indent=2))
        started = time.perf_counter()
        with (HERE / f"{name}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=150)
        with (HERE / name / "source_stalls.csv").open(newline="") as file:
            rows = [r for r in csv.DictReader(file) if r["stage"] == "convert_bgr"]
        paced = json.loads((HERE / name / "paced_report.json").read_text())
        result = {
            "name": name, "mode": mode, "elapsed_s": time.perf_counter() - started,
            "frames": len(rows),
            "first5_max_ms": max(float(r["wall_ms"]) for r in rows[:5]),
            "max_ms": max(float(r["wall_ms"]) for r in rows),
            "first5_minor_faults": sum(int(r["minor_faults"]) for r in rows[:5]),
            "minor_faults": sum(int(r["minor_faults"]) for r in rows),
            "processed": paced["processed_frames"],
            "replaced": paced["replaced_frames"],
            "expired": paced["expired_frames"],
        }
        results.append(result)
        (HERE / "progress.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
