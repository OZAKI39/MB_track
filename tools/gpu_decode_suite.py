"""按交错顺序重复 CPU/NVDEC 解码测试；只读取 A/B 录像，不连接设备。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from customer_workspace.tools.gpu_decode_check import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 5:
        parser.error("Use 1–5 short repetitions")
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    commands = []
    for repeat in range(1, args.rounds + 1):
        for video in ("A", "B"):
            engines = (
                ("cpu", "gpu") if (repeat + (video == "B")) % 2 else ("gpu", "cpu")
            )
            for engine in engines:
                name = f"{video}_{engine}_r{repeat}"
                python = (
                    sys.executable
                    if engine == "cpu"
                    else str(root / "customer_workspace/.venv_nvdec/bin/python")
                )
                command = [
                    python,
                    "-u",
                    "-B",
                    "-m",
                    "customer_workspace.tools.gpu_decode_check",
                    "bench",
                    "--video",
                    str(
                        root
                        / f"customer_workspace/microbubble_tracking/data/raw/exp{video}.mp4"
                    ),
                    "--engine",
                    engine,
                    "--paced",
                    "--output-dir",
                    str(output / name),
                ]
                commands.append({"name": name, "command": command})
    save_json(output / "commands.json", commands)
    for item in commands:
        name = item["name"]
        print("START", name, flush=True)
        with (output / f"{name}.log").open("x") as file:
            subprocess.run(
                item["command"],
                stdout=file,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=root,
                timeout=120,
            )
        result = json.loads((output / name / "run.json").read_text())
        print("DONE", name, result["frames"], flush=True)


if __name__ == "__main__":
    main()
