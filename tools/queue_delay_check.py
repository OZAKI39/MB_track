"""短时诊断一帧/两帧等待；先从同一帧播放文件，再运行本机 RTSP 模拟源。"""

import argparse
import subprocess
import sys
from pathlib import Path

from customer_workspace.microbubble_tracking.online_replay import file_hash
from customer_workspace.streamcopy_recording import write_json
from customer_workspace.tools.frame_waiting_check import read_json, verify
from microbubble_experiment.configuration import ROOT


def capture(root, mode, simulator_run=None):
    """全部计时采集完成后，另行 verify，避免离线核对争用分析资源。"""
    plan = read_json(root / "plan.json")
    path = root / "inputs.json"
    if not path.exists():
        inputs = {"simulation": True, "physical_hardware": False, "runs": {}}
        for name, video, capacity, source_mode in plan["runs"]:
            source = (
                ROOT
                / f"customer_workspace/microbubble_tracking/data/raw/exp{video}.mp4"
            )
            background = (
                ROOT
                / f"customer_workspace/validation/camera_tracking_20261031_final/background_{video}.npy"
            )
            inputs["runs"][name] = {
                "experiment": video,
                "waiting_frames": capacity,
                "mode": source_mode,
                "original_video": str(source),
                "original_sha256": file_hash(source),
                "background": str(background),
                "background_sha256": file_hash(background),
            }
        write_json(path, inputs)
    inputs = read_json(path)
    if mode == "rtsp":
        if simulator_run is None:
            raise ValueError("RTSP capture requires the running isolated simulator")
        ready = read_json(simulator_run / "ready.json")
        if ready["status"] != "ready" or (simulator_run / "shutdown.json").exists():
            raise ValueError("Simulator not running")
        write_json(root / "simulator.json", {"run": str(simulator_run), "ready": ready})
    for name, row in inputs["runs"].items():
        if row["mode"] != mode:
            continue
        if (root / name).exists():
            raise FileExistsError(root / name)
        video = row["experiment"]
        if mode == "file":
            source_copy = Path(f"/mnt/f/MB_camera_sim/videos/exp{video}.mp4")
            if file_hash(source_copy) != row["original_sha256"]:
                raise ValueError("Windows video copy differs from original")
            source_args = ["--offline-video", str(source_copy)]
        else:
            config = ROOT / f"customer_workspace/configs/rtsp_sim_{video}.json"
            if read_json(config)["rtsp_url"] != f"rtsp://127.0.0.1:8554/exp{video}":
                raise ValueError("Only the approved local simulator is allowed")
            source_args = ["--camera-config", str(config)]
        command = [
            sys.executable,
            "-B",
            "-m",
            "customer_workspace.parallel_video_check",
            *source_args,
            "--background",
            row["background"],
            "--mode",
            "parallel",
            "--seconds",
            str(plan["seconds"]),
            "--diagnose-queue",
            "--waiting-frames",
            str(row["waiting_frames"]),
            "--output-dir",
            str(root / name),
        ]
        write_json(root / f"{name}.command.json", command)
        print(f"START {name}", flush=True)
        with (root / f"{name}.console.log").open("xb") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, check=True)
        result = read_json(root / name / "analysis_check.json")
        print(
            f"DONE {name}: processed={result['processed']}, skipped={result['analysis_skipped']}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("file", "rtsp", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--simulator-run", type=Path)
    args = parser.parse_args()
    if args.action == "verify":
        verify(args.root.resolve())
    else:
        capture(args.root.resolve(), args.action, args.simulator_run)


if __name__ == "__main__":
    main()
