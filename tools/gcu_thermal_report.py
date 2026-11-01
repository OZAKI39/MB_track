"""整理 GCU 暂停对照的温度预检；只读取记录，不控制服务或显卡。

准入条件是短时诊断的保守工作边界，不是 CPU/GPU 的厂商温度极限。
历史样本通过也不能代替实际暂停前的重新检查，更不构成暂停授权。
"""

import argparse
import json
import math
from pathlib import Path


def assess(samples):
    """所有温度、风扇、读取耗时均完整有效时，才评估记录中的启动条件。"""
    reasons = []
    if len(samples) < 10:
        reasons.append("Fewer than 10 sensor samples")
    fields = ("cpu_c", "gpu_c", "cpu_fan_rpm", "gpu_fan_rpm", "read_ms")
    complete = [
        row
        for row in samples
        if row.get("ok") is True
        and all(
            type(row.get(key)) in (int, float) and math.isfinite(row[key])
            for key in fields
        )
    ]
    if len(complete) != len(samples):
        reasons.append("Missing or failed sensor readings")
    ranges = {}
    if complete:
        for key in fields:
            values = [row[key] for row in complete]
            ranges[key] = {"min": min(values), "max": max(values)}
        if ranges["cpu_c"]["min"] <= 0 or ranges["cpu_c"]["max"] >= 80:
            reasons.append("CPU temperature is outside the 0–80 C start range")
        if ranges["gpu_c"]["min"] <= 0 or ranges["gpu_c"]["max"] >= 70:
            reasons.append("GPU temperature is outside the 0–70 C start range")
        for key in ("cpu_fan_rpm", "gpu_fan_rpm"):
            if ranges[key]["min"] < 1000 or ranges[key]["max"] > 15000:
                reasons.append(f"{key} is outside the 1000–15000 RPM check range")
        if ranges["read_ms"]["max"] >= 1500:
            reasons.append("Sensor collection takes at least 1500 ms")
    return {
        "samples": len(samples),
        "complete_samples": len(complete),
        "ranges": ranges,
        "recorded_start_conditions_pass": not reasons,
        "reasons": reasons,
        "authorization": "No service-pause authorization is implied by this report",
        "live_recheck_required": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.samples.read_text().splitlines()]
    report = assess(samples)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
