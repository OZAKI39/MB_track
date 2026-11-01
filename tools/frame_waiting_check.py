"""用已有 F 盘 RTSP 模拟源交替检查一帧/两帧等待区；不控制真实设备。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from customer_workspace.microbubble_tracking.online_replay import file_hash
from customer_workspace.parallel_video_check import distribution
from customer_workspace.streamcopy_recording import write_json
from customer_workspace.tools.camera_tracking_validation import verify_run
from customer_workspace.tools.rtsp_tracking_report import build_report
from customer_workspace.tracking_latency import feedback_components_ms
from microbubble_experiment.configuration import ROOT


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def capture(root, simulator_run):
    """只执行计划内的新目录；全部计时采集结束后才做离线重算。"""
    plan = read_json(root / "test_plan.json")
    if (root / "inputs.json").exists():
        raise FileExistsError("Capture already has inputs.json; keep the previous run")
    ready = read_json(simulator_run / "ready.json")
    if ready["status"] != "ready" or (simulator_run / "shutdown.json").exists():
        raise ValueError("The isolated Windows simulator is not running")
    inputs = {
        "simulation": True,
        "physical_hardware": False,
        "simulator_run": str(simulator_run),
        "runs": {},
    }
    for name, video, capacity, repeat in plan["run_order"]:
        if (root / name).exists():
            raise FileExistsError(root / name)
        source = (
            ROOT / f"customer_workspace/microbubble_tracking/data/raw/exp{video}.mp4"
        )
        background = (
            ROOT
            / f"customer_workspace/validation/camera_tracking_20261031_final/background_{video}.npy"
        )
        config = ROOT / f"customer_workspace/configs/rtsp_sim_{video}.json"
        camera = read_json(config)
        # 运行器只允许已经准备好的本机模拟地址；不能误用正式相机配置。
        if camera.get("rtsp_url") != f"rtsp://127.0.0.1:8554/exp{video}":
            raise ValueError("Not the approved local RTSP simulator")
        inputs["runs"][name] = {
            "experiment": video,
            "waiting_frames": capacity,
            "repeat": repeat,
            "original_video": str(source),
            "original_sha256": file_hash(source),
            "background": str(background),
            "background_sha256": file_hash(background),
            "camera_config": str(config),
            "camera_configuration": camera,
        }
    write_json(root / "inputs.json", inputs)
    for name, row in inputs["runs"].items():
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
            str(row["waiting_frames"]),
            "--output-dir",
            str(root / name),
        ]
        write_json(root / f"{name}.command.json", command)
        print(f"START {name}", flush=True)
        with (root / f"{name}.console.log").open("xb") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, check=True)
        report = read_json(root / name / "analysis_check.json")
        native = read_json(root / name / "latency_windows.json")
        sent = {r["seq"]: r for r in native["sent"]}
        values = [
            feedback_components_ms(sent[r["seq"]], r)["feedback_total_ms"]
            for r in native["acknowledgements"]
            if r["kind"] == "complete"
        ]
        print(
            json.dumps(
                {
                    "run": name,
                    "processed": report["processed"],
                    "skipped": report["analysis_skipped"],
                    "feedback_ms": distribution(values),
                    "error": report["pipeline_error"],
                }
            ),
            flush=True,
        )


def verify(root):
    """逐组重算完全相同的输入，再核对原视频像素、帧去向与数据库。"""
    inputs = read_json(root / "inputs.json")
    for name in inputs["runs"]:
        folder = root / name
        print(f"VERIFY {name}", flush=True)
        result = verify_run(folder, folder / "recording/video.mp4")
        print(
            json.dumps(
                {
                    "run": name,
                    "scientific_results_equal": result["scientific_results_equal"],
                }
            ),
            flush=True,
        )
    build_report(root, root / "integrity_report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--simulator-run", type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        if args.simulator_run is None:
            parser.error("capture requires --simulator-run")
        capture(args.root.resolve(), args.simulator_run)
    else:
        verify(args.root.resolve())


if __name__ == "__main__":
    main()
