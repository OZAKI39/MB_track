"""Isolate Gaussian output allocation on a real frame; do not change production."""

import csv
import gc
import json
import resource
import time
from pathlib import Path

import av
import cv2
import numpy as np


def main():
    output = Path(__file__).resolve().parent
    root = output.parents[2]
    video = root / "customer_workspace/microbubble_tracking/data/raw/expA.mp4"
    cv2.setNumThreads(24)
    with av.open(str(video)) as container:
        frame = next(container.decode(video=0))
        gray = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY).astype(np.float32)
    target = np.empty_like(gray)
    target.fill(0)
    expected = cv2.GaussianBlur(gray, (0, 0), 1.0)
    settings = gc.get_threshold(), gc.isenabled()
    rows, summaries = [], []
    for repeat in range(1, 4):
        for mode in ("allocate", "reuse"):
            gc.collect(2)
            for index in range(500):
                before = resource.getrusage(resource.RUSAGE_THREAD)
                started, cpu = time.perf_counter(), time.thread_time()
                if mode == "allocate":
                    result = cv2.GaussianBlur(gray, (0, 0), 1.0)
                else:
                    result = cv2.GaussianBlur(gray, (0, 0), 1.0, dst=target)
                finished, cpu_end = time.perf_counter(), time.thread_time()
                after = resource.getrusage(resource.RUSAGE_THREAD)
                if index in (0, 499):
                    assert np.array_equal(result, expected)
                    if mode == "reuse":
                        assert np.shares_memory(result, target)
                rows.append({
                    "repeat": repeat, "mode": mode, "iteration": index,
                    "wall_ms": (finished - started) * 1000,
                    "cpu_ms": (cpu_end - cpu) * 1000,
                    "minor_faults": after.ru_minflt - before.ru_minflt,
                    "major_faults": after.ru_majflt - before.ru_majflt,
                })
                del result
            part = rows[-500:]
            values = [row["wall_ms"] for row in part]
            summaries.append({
                "repeat": repeat, "mode": mode,
                "median_p99_max_ms": np.percentile(values, [50, 99, 100]).tolist(),
                "minor_faults": sum(row["minor_faults"] for row in part),
                "major_faults": sum(row["major_faults"] for row in part),
            })
    assert (gc.get_threshold(), gc.isenabled()) == settings
    with (output / "blur_buffer_samples.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "scope": "Same real first-frame input; only Gaussian output buffer differs; not full-pipeline speed",
        "input": str(video), "shape": list(gray.shape), "buffer_bytes": target.nbytes,
        "outputs_exact": True, "summaries": summaries,
    }
    (output / "blur_buffer_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
