"""真实 Windows 文件输入下检查取消、短片结束及错误尺寸；不连接硬件。"""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np

from customer_workspace.streamcopy_recording import write_json
from microbubble_experiment.configuration import ROOT


def command(video, background, output, seconds):
    return [
        sys.executable,
        "-m",
        "customer_workspace.parallel_video_check",
        "--offline-video",
        str(video),
        "--background",
        str(background),
        "--output-dir",
        str(output),
        "--seconds",
        str(seconds),
    ]


def check(folder, expected_text):
    summary = json.loads((folder / "analysis_check.json").read_text())
    saved = json.loads((folder / "tracking/writer_report.json").read_text())
    assert expected_text in str(summary["pipeline_error"]), summary
    assert summary["processed"] == summary["saved"]
    assert summary["threads_closed"]
    assert saved["all_submitted_saved"] and saved["worker_finished"]
    assert not saved["pending_records"]
    return {
        "errors": summary["pipeline_error"],
        "processed": summary["processed"],
        "saved": summary["saved"],
        "status": summary["tracking_summary"]["status"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    video = ROOT / "customer_workspace/microbubble_tracking/data/raw/expA.mp4"
    results = {}
    folder = args.output / "cancel"
    with (args.output / "cancel.log").open("x") as log:
        process = subprocess.Popen(
            command(video, args.background, folder, 22),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 45
        while not (folder / "recording_started.json").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Cancel test did not reach the recording window")
            time.sleep(0.05)
        time.sleep(3)
        process.send_signal(signal.SIGINT)
        process.wait(timeout=60)
        assert process.returncode != 0
    results["cancel"] = check(folder, "KeyboardInterrupt")
    short = args.output / "short_input.mp4"
    with av.open(str(video)) as source, av.open(str(short), "w") as target:
        stream = target.add_stream_from_template(source.streams.video[0])
        for index, packet in enumerate(source.demux(video=0)):
            if packet.pts is None or index >= 30:
                break
            packet.stream = stream
            target.mux(packet)
    for name, source, background, message in [
        ("early_end", short, args.background, "Input ended"),
        ("wrong_shape", video, args.output / "wrong_background.npy", "dimensions"),
    ]:
        if name == "wrong_shape":
            np.save(background, np.full((64, 64), 180, np.float32))
        folder = args.output / name
        with (args.output / f"{name}.log").open("x") as log:
            result = subprocess.run(
                command(source, background, folder, 2),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
        assert result.returncode != 0
        results[name] = check(folder, message)
    write_json(args.output / "checks.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
