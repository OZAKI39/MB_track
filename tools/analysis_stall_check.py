"""使用 F 盘模拟相机定位分析内部慢帧；默认参数和识别算法保持不变。"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from customer_workspace.microbubble_tracking.online_replay import file_hash
from customer_workspace.streamcopy_recording import write_json
from customer_workspace.tools.frame_waiting_check import read_json, verify
from microbubble_experiment.configuration import ROOT


def capture(root, phase, simulator):
    plan = read_json(root / "plan.json")
    ready = read_json(simulator / "ready.json")
    if ready["status"] != "ready" or (simulator / "shutdown.json").exists():
        raise ValueError("The isolated simulator is not running")
    if not (root / "inputs.json").exists():
        inputs = {
            "simulation": True,
            "physical_hardware": False,
            "simulator_run": str(simulator),
            "runs": {},
        }
        for name, video, diagnosis in plan["runs"]:
            source = (
                ROOT
                / f"customer_workspace/microbubble_tracking/data/raw/exp{video}.mp4"
            )
            background = (
                ROOT
                / f"customer_workspace/validation/camera_tracking_20261031_final/background_{video}.npy"
            )
            config = ROOT / f"customer_workspace/configs/rtsp_sim_{video}.json"
            if read_json(config)["rtsp_url"] != f"rtsp://127.0.0.1:8554/exp{video}":
                raise ValueError("Not the approved local RTSP simulator")
            inputs["runs"][name] = {
                "experiment": video,
                "diagnose_analysis": diagnosis,
                "original_video": str(source),
                "original_sha256": file_hash(source),
                "background": str(background),
                "background_sha256": file_hash(background),
                "camera_config": str(config),
                "waiting_frames": 1,
            }
        write_json(root / "inputs.json", inputs)
    inputs = read_json(root / "inputs.json")
    for name, row in inputs["runs"].items():
        native = name.endswith("_r3")
        if native != (phase == "native"):
            continue
        folder = root / name
        if folder.exists():
            raise FileExistsError(folder)
        command = [
            sys.executable,
            "-B",
            "-m",
            "customer_workspace.parallel_video_check",
            "--camera-config",
            row["camera_config"],
            "--background",
            row["background"],
            "--mode",
            "parallel",
            "--seconds",
            str(plan["seconds"]),
            "--measure-latency",
            "--waiting-frames",
            "1",
            "--output-dir",
            str(folder),
        ]
        if row["diagnose_analysis"]:
            command.append("--diagnose-analysis")
        write_json(root / f"{name}.command.json", command)
        observer = None
        log = None
        if native:
            perf_root = (
                ROOT / "customer_workspace/dependencies/native_perf/installed/usr/lib"
            )
            observer_command = [
                "/mnt/c/Windows/System32/wsl.exe",
                "-d",
                "Ubuntu",
                "-u",
                "root",
                "--",
                sys.executable,
                str(
                    ROOT
                    / "customer_workspace/microbubble_tracking/native_stall_watch.py"
                ),
                str(folder),
                str(root / f"{name}_native.jsonl"),
                "--duration",
                "90",
                "--interval",
                ".005",
                "--perf",
                str(perf_root / "linux-tools-6.8.0-147/perf"),
                "--perf-lib",
                str(perf_root / "x86_64-linux-gnu"),
                "--perf-targets",
                "consumer",
                "--frequency",
                "999",
                "--stop-file",
                str(folder / "analysis_finished.json"),
            ]
            write_json(root / f"{name}.observer.command.json", observer_command)
            log = (root / f"{name}.observer.log").open("xb")
            observer = subprocess.Popen(
                observer_command, stdout=log, stderr=log, cwd=ROOT
            )
        print(f"START {name}", flush=True)
        try:
            with (root / f"{name}.console.log").open("xb") as output:
                subprocess.run(
                    command, stdout=output, stderr=output, cwd=ROOT, check=True
                )
        finally:
            if observer is not None:
                try:
                    observer.wait(timeout=100)
                finally:
                    log.close()
                if observer.returncode:
                    raise RuntimeError(f"Native observation failed: {name}")
                # perf 由 root 读取内核，输出文件交回当前用户，供普通报告程序读取。
                subprocess.run(
                    [
                        "/mnt/c/Windows/System32/wsl.exe",
                        "-d",
                        "Ubuntu",
                        "-u",
                        "root",
                        "--",
                        "chown",
                        f"{os.getuid()}:{os.getgid()}",
                        str(root / f"{name}_native.data"),
                    ],
                    check=True,
                )
        result = read_json(folder / "analysis_check.json")
        print(
            json.dumps(
                {
                    "run": name,
                    "processed": result["processed"],
                    "skipped": result["analysis_skipped"],
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("basic", "native", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--simulator-run", type=Path)
    args = parser.parse_args()
    if args.phase == "verify":
        verify(args.root.resolve())
    else:
        if args.simulator_run is None:
            parser.error("Capture requires --simulator-run")
        capture(args.root.resolve(), args.phase, args.simulator_run.resolve())


if __name__ == "__main__":
    main()
