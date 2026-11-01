"""第三轮：同一个 WSL 进程连续运行多次完整录像与分析实验段，不连接 AFG。"""

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import psutil

from customer_workspace.parallel_video_check import add_tracking_arguments, run_check
from customer_workspace.streamcopy_recording import write_json
from microbubble_experiment.configuration import ROOT

DEFAULT_CRITERIA = ROOT / "customer_workspace/configs/video_stability.json"


def run_series(output: Path, criteria: dict, background: Path, **source_options) -> list[dict]:
    """逐次收尾再开始下一次；失败保留原始文件和原因，不用重试覆盖失败。"""
    seconds, runs = criteria["seconds"], criteria["runs"]
    if not 60 <= seconds <= 90 or not 4 <= runs <= 10:
        raise ValueError("稳定性测试每组限 60–90 秒，至少首次加三次重复，最多十组")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "acceptance_criteria.json", criteria)
    process = psutil.Process()
    records = []
    for number in range(1, runs + 1):
        folder = output / f"run_{number:02d}"
        record = {
            "run": number,
            "directory": str(folder.resolve()),
            "started_at": datetime.now().astimezone().isoformat(),
            "wsl_pid": process.pid,
            "wsl_before_rss_mib": process.memory_info().rss / 2**20,
        }
        print(f"开始第 {number}/{runs} 次，录像 {seconds:g} 秒。", flush=True)
        started = time.monotonic()
        try:
            report = run_check(
                SimpleNamespace(
                    output_dir=folder,
                    background=background,
                    mode="parallel",
                    seconds=seconds,
                    **source_options,
                )
            )
            record["capture_completed"] = True
            record["recording_passed"] = report["recording_passed"]
            record["analysis_hz"] = report["analysis_hz"]
        except Exception:  # noqa: BLE001 -- 批次边界必须保存任意一轮的失败证据。
            # 仅在整次测试边界记录失败，后续独立测试仍保留自己的结果。
            record["capture_completed"] = False
            record["error"] = traceback.format_exc()
        record["wall_seconds"] = time.monotonic() - started
        record["wsl_after_rss_mib"] = process.memory_info().rss / 2**20
        records.append(record)
        write_json(output / f"run_{number:02d}_outcome.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    write_json(output / "series.json", records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    add_tracking_arguments(parser)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        stamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d-%H%M%S")
        args.output_dir = ROOT / "customer_workspace/validation" / f"stability_{stamp}"
    criteria = json.loads(args.criteria.read_text(encoding="utf-8"))
    records = run_series(args.output_dir, criteria, args.background,
                         camera_config=args.camera_config, offline_video=args.offline_video,
                         opencv_threads=args.opencv_threads)
    return int(any(not row.get("recording_passed", False) for row in records))


if __name__ == "__main__":
    raise SystemExit(main())
