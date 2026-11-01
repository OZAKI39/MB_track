"""Untraced, alternating same-video SQLite policy comparison."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = ROOT / "customer_workspace/microbubble_tracking"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="")
    prefix = parser.parse_args().prefix
    cases = []
    for pair in range(1, 5):
        experiment = "A" if pair in (1, 4) else "B"
        policies = ("auto", "deferred") if pair % 2 else ("deferred", "auto")
        for policy in policies:
            cases.append((f"pair_{pair}_{experiment}_{policy}", experiment, policy, 1, 4))
    cases.extend([
        ("long_B_deferred", "B", "deferred", 3, 4),
        ("B_batch16", "B", "deferred", 1, 16),
    ])
    commands, progress = [], []
    for name, experiment, policy, repeats, batch in cases:
        name = prefix + name
        reference = "B" if experiment == "A" else "A"
        command = [
            str(ROOT / ".venv/bin/python"), str(CODE / "paced_replay.py"),
            "--video", str(CODE / f"data/raw/exp{experiment}.mp4"),
            "--reference", str(CODE / f"results/{reference}_offline/background.npy"),
            "--pzt", str(CODE / f"data/pzt/exp{experiment}"),
            "--output-dir", str(HERE / name),
            "--opencv-threads", "24", "--conversion-threads", "1",
            "--save-checkpoint-policy", policy, "--save-batch-frames", str(batch),
            "--repeats", str(repeats), "--measure-stages", "--diagnose-stalls",
        ]
        commands.append(command)
        (HERE / f"{prefix}commands.json").write_text(json.dumps(commands, indent=2))
        with (HERE / f"{name}.log").open("x") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=150)
        with (HERE / name / "writer_timing.csv").open() as file:
            rows = list(csv.DictReader(file))
        tx = {row["transaction_id"]: row for row in rows}.values()
        result = {
            "name": name, "saved_frames": len(rows), "transactions": len(tx),
            "max_commit_ms": max(float(r["commit_ms"]) for r in tx),
            "max_durable_ms": max(float(r["durable_ms"]) for r in rows),
        }
        progress.append(result)
        (HERE / f"{prefix}progress.json").write_text(json.dumps(progress, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
