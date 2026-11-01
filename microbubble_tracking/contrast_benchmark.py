"""仅比较相对暗度计算；候选公式留在诊断中，不替换正式识别规则。"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .bubble_tracker import (
        OnlineBackground,
        SegParams,
        combine_backgrounds,
        iter_frames,
        segment,
    )
    from .online_replay import verify_inputs
    from .paced_replay import environment
else:
    from bubble_tracker import (
        OnlineBackground,
        SegParams,
        combine_backgrounds,
        iter_frames,
        segment,
    )
    from online_replay import verify_inputs
    from paced_replay import environment


METHODS = (
    "original",
    "numpy_inplace",
    "numpy_buffers",
    "numpy_cached_divide",
    "numpy_cached_reciprocal",
    "opencv_buffers",
    "opencv_cached_divide",
    "numpy_tiles_32",
    "numpy_tiles_64",
    "numpy_tiles_128",
)


class Candidate:
    """诊断用的有限计算空间；初始化不计入逐帧耗时，缓存刷新计入。"""

    def __init__(self, method, background):
        self.method = method
        self.denominator = background + np.float32(1)
        self.reciprocal = np.reciprocal(self.denominator)
        self.out = np.empty_like(background)

    def calculate(self, fb, background, band):
        method = self.method
        if method == "original":
            darkness = 1.0 - fb / (background + 1.0)
        elif method == "numpy_inplace":
            darkness = fb / (background + 1.0)
            np.subtract(1.0, darkness, out=darkness)
        elif method.startswith("numpy_tiles_"):
            height = int(method.rsplit("_", 1)[1])
            darkness = self.out
            for y0 in range(0, fb.shape[0], height):
                rows = slice(y0, y0 + height)
                np.add(background[rows], 1.0, out=self.denominator[rows])
                np.divide(fb[rows], self.denominator[rows], out=darkness[rows])
                np.subtract(1.0, darkness[rows], out=darkness[rows])
        else:
            darkness = self.out
            if "cached" in method:
                rows = slice(band * fb.shape[0] // 10, (band + 1) * fb.shape[0] // 10)
                np.add(background[rows], 1.0, out=self.denominator[rows])
            elif method == "opencv_buffers":
                cv2.add(background, 1.0, dst=self.denominator)
            else:
                np.add(background, 1.0, out=self.denominator)
            if method == "numpy_cached_reciprocal":
                np.reciprocal(self.denominator[rows], out=self.reciprocal[rows])
                np.multiply(fb, self.reciprocal, out=darkness)
            elif method.startswith("opencv"):
                cv2.divide(fb, self.denominator, dst=darkness)
            else:
                np.divide(fb, self.denominator, out=darkness)
            if method.startswith("opencv"):
                cv2.subtract(1.0, darkness, dst=darkness)
            else:
                np.subtract(1.0, darkness, out=darkness)
        darkness -= np.median(darkness[::8, ::8])
        return darkness


def run(output, repeats=24):
    """顺序重建真实背景，每段取 12 帧；交错测候选，逐像素核对，不比较误差容限。"""
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261029)
    checks = {
        method: {
            "different_pixels": 0,
            "threshold_differences": 0,
            "max_abs_error": 0.0,
        }
        for method in METHODS
    }
    measured = {method: [] for method in METHODS}
    inputs = {}
    with (output / "kernel_timing.csv").open("x", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(["experiment", "video_frame", "repeat", "method", "time_ms"])
        for experiment, ref in (("A", "B"), ("B", "A")):
            video = Path(f"data/raw/exp{experiment}.mp4")
            reference = Path(f"results/{ref}_offline/background.npy")
            inputs[experiment] = verify_inputs(
                video, reference, Path(f"data/pzt/exp{experiment}")
            )
            selected = set(
                np.linspace(0, inputs[experiment]["expected_frames"] - 1, 12, dtype=int)
            )
            parameters = SegParams()
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (parameters.close, parameters.close)
            )
            background = None
            for index, gray in iter_frames(str(video)):
                if background is None:
                    background = OnlineBackground(
                        combine_backgrounds(
                            [np.load(reference, allow_pickle=False)],
                            gray,
                            verbose=False,
                        )
                    )
                labels, _, blurred = segment(gray, background.B, parameters, kernel)
                if index in selected:
                    candidates = {
                        name: Candidate(name, background.B) for name in METHODS
                    }
                    expected = candidates["original"].calculate(
                        blurred, background.B, 0
                    )
                    for name, candidate in candidates.items():
                        actual = candidate.calculate(blurred, background.B, 0)
                        checks[name]["different_pixels"] += int(
                            np.count_nonzero(actual != expected)
                        )
                        checks[name]["max_abs_error"] = max(
                            checks[name]["max_abs_error"],
                            float(np.max(np.abs(actual - expected))),
                        )
                        for threshold in (parameters.lo, parameters.hi):
                            checks[name]["threshold_differences"] += int(
                                np.count_nonzero(
                                    (actual > threshold) != (expected > threshold)
                                )
                            )
                    # 丢弃预热，避免把新数组第一次触碰页面的成本全算给某个候选。
                    for repetition in range(repeats + 5):
                        for name in rng.permutation(METHODS):
                            started = time.perf_counter()
                            candidates[name].calculate(
                                blurred, background.B, repetition % 10
                            )
                            elapsed_ms = (time.perf_counter() - started) * 1000
                            if repetition >= 5:
                                writer.writerow(
                                    [
                                        experiment,
                                        index,
                                        repetition - 5,
                                        name,
                                        elapsed_ms,
                                    ]
                                )
                                measured[name].append(elapsed_ms)
                    print(f"Compared {experiment} frame {index}", flush=True)
                background.update(blurred, labels)
    report = {
        "environment": environment(),
        "inputs": inputs,
        "checks": checks,
        "samples_per_video": 12,
        "repetitions": repeats,
        "scope": "Kernel-only diagnostic; final choice requires full-frame timing",
    }
    report["timing_ms"] = {
        name: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }
        for name, values in measured.items()
    }
    (output / "kernel_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"timing_ms": report["timing_ms"], "checks": checks}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument(
        "--verify-stream",
        action="store_true",
        help="Verify every optimized darkness pixel and cache update; not a timing test",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.verify_stream:
        verify_stream(args.output_dir)
    else:
        run(args.output_dir, args.repeats)


def verify_stream(output):
    """在整段真实输入上逐像素核对位模式，以及每次背景更新后的缓存。"""
    output.mkdir(parents=True, exist_ok=False)
    report = {}
    for experiment, ref in (("A", "B"), ("B", "A")):
        video = Path(f"data/raw/exp{experiment}.mp4")
        reference = Path(f"results/{ref}_offline/background.npy")
        provenance = verify_inputs(video, reference, Path(f"data/pzt/exp{experiment}"))
        params = SegParams()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params.close, params.close)
        )
        background = None
        count = pixels = 0
        with (output / f"pixels_{experiment}.csv").open(
            "x", newline="", encoding="utf-8"
        ) as log:
            writer = csv.writer(log)
            writer.writerow(
                ["video_frame", "pixels", "different_bits", "stale_cache_pixels"]
            )
            for index, gray in iter_frames(str(video)):
                if background is None:
                    background = OnlineBackground(
                        combine_backgrounds(
                            [np.load(reference, allow_pickle=False)],
                            gray,
                            verbose=False,
                        )
                    )
                labels, _, blurred = segment(
                    gray, background.B, params, kernel, workspace=background.workspace
                )
                expected = 1.0 - blurred / (background.B + 1.0)
                expected -= np.median(expected[::8, ::8])
                actual = background.workspace._darkness
                differences = int(
                    np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32))
                )
                background.update(blurred, labels)
                stale = int(
                    np.count_nonzero(
                        background.workspace._denominator != background.B + 1.0
                    )
                )
                writer.writerow([index, gray.size, differences, stale])
                if differences or stale:
                    raise ValueError(
                        f"Darkness or cache changed at {experiment}:{index}"
                    )
                count += 1
                pixels += gray.size
        if count != provenance["expected_frames"]:
            raise ValueError("Incomplete verification video")
        report[experiment] = {
            "frames": count,
            "pixels": pixels,
            "different_bits": 0,
            "stale_cache_pixels": 0,
            "provenance": provenance,
        }
        print(
            experiment,
            count,
            "frames: all darkness bits and cache values identical",
            flush=True,
        )
    (output / "pixel_verification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
