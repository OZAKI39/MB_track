"""实测 Event.wait 的晚醒时间；不连接仪器、不调整系统定时器。"""

import argparse
import json
import math
import platform
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import get_clock_info, perf_counter

from customer_workspace.benchmark_report import distribution
from customer_workspace.timing_benchmark import save_results


def choose_advance_ms(lateness_by_duration: dict[float, list[float]]) -> float:
    """取各等待长度实测 P99 的最大值，加 1 ms 后向上取整。"""
    if not lateness_by_duration or any(
        not values or any(not math.isfinite(v) or v < 0 for v in values)
        for values in lateness_by_duration.values()
    ):
        raise ValueError("需要非空、非负且有限的晚醒记录")
    return float(
        math.ceil(
            max(distribution(v)["p99"] for v in lateness_by_duration.values()) + 1
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-per-duration", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.count_per_duration < 100:
        parser.error("每种等待长度至少测量 100 次")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    durations_ms = (1, 5, 10, 20, 40, 60)
    event = threading.Event()
    samples = []
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "os": platform.platform(),
        "python": sys.version,
        "clocks": {
            name: vars(get_clock_info(name)) for name in ("monotonic", "perf_counter")
        },
        "requested_per_duration": args.count_per_duration,
        "durations_ms": durations_ms,
        "status": "running",
    }
    try:
        for _ in range(args.count_per_duration):
            for duration_ms in durations_ms:
                started = perf_counter()
                event.wait(duration_ms / 1000)
                returned = perf_counter()
                samples.append(
                    {
                        "index": len(samples) + 1,
                        "requested_ms": duration_ms,
                        "actual_ms": (returned - started) * 1000,
                        "lateness_ms": max(
                            0.0, (returned - started) * 1000 - duration_ms
                        ),
                    }
                )
        by_duration = {
            duration: [
                r["lateness_ms"] for r in samples if r["requested_ms"] == duration
            ]
            for duration in durations_ms
        }
        summary["pooled_lateness_ms"] = distribution(
            [r["lateness_ms"] for r in samples]
        )
        summary["by_duration_ms"] = {
            str(k): distribution(v) for k, v in by_duration.items()
        }
        summary["selected_advance_ms"] = choose_advance_ms(by_duration)
        summary["selection_rule"] = "ceil(max(per-duration P99) + 1 ms)"
        summary["status"] = "completed"
    finally:
        summary["recorded_waits"] = len(samples)
        summary["finished_utc"] = datetime.now(UTC).isoformat()
        save_results(args.output_dir, samples, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
