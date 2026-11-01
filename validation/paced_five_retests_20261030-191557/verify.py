"""核对五组原速复测；保留缺帧运行，完整输入才与前次科学结果比较。"""

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def fingerprint(folder):
    """只排除计算耗时；轮廓原始字节、事件、编号和输入时间全部参与核对。"""
    result = {}
    with sqlite3.connect(f"file:{folder / 'tracking.sqlite'}?mode=ro", uri=True) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
        for table in ("frames", "observations", "tracks", "events", "track_updates",
                      "event_updates", "input_rejections"):
            columns = [r[1] for r in db.execute(f"PRAGMA table_info({table})")
                       if r[1] != "processing_ms"]
            order = "source_frame, track_id" if table == "observations" else columns[0]
            digest, count = hashlib.sha256(), 0
            for row in db.execute(f"SELECT {','.join(columns)} FROM {table} ORDER BY {order}"):
                digest.update(repr(row).encode())
                digest.update(b"\n")
                count += 1
            result[table] = {"rows": count, "sha256": digest.hexdigest()}
    for name in ("background_initial.npy", "background_final.npy", "source_actions.csv",
                 "source_video_frames.csv", "action_links.csv"):
        result[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    return result


def main():
    plan = load(ROOT / "plan.json")
    commands = load(ROOT / "commands.json")
    assert len(commands) == 10
    reference = {label: fingerprint(Path(path)) for label, path in plan["reference_runs"].items()}
    summary = {"runs": [], "aggregates": {}, "failures": []}
    measured = {label: [] for label in ("A", "B")}
    for command in commands:
        name = command["name"]
        if command["exit_code"]:
            summary["failures"].append(command)
            continue
        folder = ROOT / name
        label, repetition = name[3], int(name.rsplit("_", 1)[1])
        paced = load(folder / "paced_report.json")
        writer = load(folder / "writer_report.json")
        timing = read_csv(folder / "consumer_timing.csv")
        outcomes = read_csv(folder / "frame_outcomes.csv")
        missing = [row for row in outcomes if row["outcome"] != "processed"]
        assert paced["status"] == "completed" and paced["all_offered_frames_accounted"]
        assert writer["all_submitted_saved"] and writer["worker_finished"]
        assert not writer["uncommitted"] and writer["pending_records"] == 0
        actual = fingerprint(folder)
        assert actual["frames"]["rows"] == paced["processed_frames"] == writer["committed_frames"]
        assert actual["frames"]["rows"] + len(missing) == paced["offered_frames"] == 1320
        equal = actual == reference[label] if not missing else None
        if not missing:
            assert equal, name
        values = [float(row["process_ms"]) for row in timing]
        measured[label].extend(values)
        record = {
            "run": name, "video": label, "repetition": repetition,
            "offered": paced["offered_frames"], "processed": paced["processed_frames"],
            "saved": writer["committed_frames"], "missing": missing,
            "first_frame_ms": float(timing[0]["process_ms"]),
            "process_ms": paced["metrics"]["process_ms"],
            "over_frame_budget": sum(v > paced["frame_budget_ms"] for v in values),
            "age_ms": paced["metrics"]["age_ms"],
            "analysis_rss_mib": paced["memory_sampled_mb"]["consumer_rss_mb"],
            "save_queue_peak": writer["peak_pending_records"],
            "complete_input_scientific_results_equal": equal,
            "previous_gap_region": [row for row in timing if 1142 <= int(row["video_frame"]) <= 1146] if label == "A" else [],
        }
        if missing:
            indices = {int(row["video_frame"]) for row in missing}
            window = [row for row in timing if any(abs(int(row["video_frame"]) - i) <= 5 for i in indices)]
            (folder / "gap_window.json").write_text(json.dumps(window, indent=2) + "\n")
        summary["runs"].append(record)
    code = ROOT.parent.parent / "microbubble_tracking"
    assert all(hashlib.sha256((code / name).read_bytes()).hexdigest() == value
               for name, value in plan["source_before_sha256"].items())
    summary["code_unchanged"] = True
    for label, values in measured.items():
        rows = [r for r in summary["runs"] if r["video"] == label]
        summary["aggregates"][label] = {
            "completed_runs": len(rows), "processed": sum(r["processed"] for r in rows),
            "saved": sum(r["saved"] for r in rows), "missing": sum(len(r["missing"]) for r in rows),
            "process_ms": dict(zip(("median", "p99", "max"), map(float, np.percentile(values, (50, 99, 100))))) if values else None,
            "first_frame_range_ms": [min(r["first_frame_ms"] for r in rows), max(r["first_frame_ms"] for r in rows)] if rows else None,
            "rss_peak_mib": max(r["analysis_rss_mib"]["max"] for r in rows) if rows else None,
        }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (ROOT / "runs.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "processed", "saved", "missing", "first_ms", "median_ms", "p99_ms", "max_ms", "over_frame_budget", "rss_peak_mib"])
        for row in summary["runs"]:
            writer.writerow([row["run"], row["processed"], row["saved"], len(row["missing"]),
                             row["first_frame_ms"], row["process_ms"]["median"], row["process_ms"]["p99"],
                             row["process_ms"]["max"], row["over_frame_budget"], row["analysis_rss_mib"]["max"]])
    lines = [
        "# A、B 各五遍原速复测", "",
        "本轮按 A1、B1、A2、B2……A5、B5 顺序运行，共十次独立启动。每次处理完整原视频，",
        "没有用一次进程内循环冒充五次启动，也没有改变识别、追踪或保存代码。", "",
        "沿用上一轮设置：原速约 60 帧/秒，OpenCV 24 线程、送帧转换 1 线程，",
        "启用分析预热、平滑结果复用和异步保存；保留阶段计时，关闭重型逐帧诊断。",
        "两段参考背景沿用 results/B_offline/background.npy，对应 PZT 资料仅用于读取原有时间映射。",
        "本轮没有连接真实相机或 AFG，没有修改 Windows 部署。", "",
        "## 每次结果", "",
        "| 运行 | 处理 / 保存 | 未处理 | 首帧 ms | 中位 ms | P99 ms | 最大 ms | 超过 16.67 ms 的帧数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary["runs"]:
        m = r["process_ms"]
        lines.append(f"| {r['run']} | {r['processed']} / {r['saved']} | {len(r['missing'])} | "
                     f"{r['first_frame_ms']:.3f} | {m['median']:.3f} | {m['p99']:.3f} | "
                     f"{m['max']:.3f} | {r['over_frame_budget']} |")
    lines += ["", "P99 表示 99% 的帧不超过这个耗时。个别帧超过 16.67 ms 不一定立即跳帧；",
              "实际是否跳帧以 frame_outcomes.csv 为准。这里测的是本机回放处理时间，不是曝光时间。", "",
              "## 合并统计", "",
              "| 视频 | 完成遍数 | 处理 / 保存 | 未处理 | 中位 ms | P99 ms | 最大 ms | 分析内存峰值 MiB |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, r in summary["aggregates"].items():
        if r["process_ms"] is None:
            continue
        m = r["process_ms"]
        lines.append(f"| {label} | {r['completed_runs']} | {r['processed']} / {r['saved']} | {r['missing']} | "
                     f"{m['median']:.3f} | {m['p99']:.3f} | {m['max']:.3f} | {r['rss_peak_mib']:.3f} |")
    equals = sum(r["complete_input_scientific_results_equal"] is True for r in summary["runs"])
    budget = sum(r["over_frame_budget"] for r in summary["runs"])
    lines += ["", "## 核对及限制", "",
              f"- 共 {len(summary['runs'])} 次完成、{len(summary['failures'])} 次进程失败；全部命令和状态保存在 commands.json。",
              "- 完成运行的数据库完整性和关联检查通过；已分析的帧全部保存，保存线程正常结束，收尾无积压。",
              f"- {equals} 次完整输入的科学结果与上一轮对应结果一致：包括原始轮廓、编号、事件及修订、输入时间、背景和动作对应表。",
              "  计算耗时和每次运行身份不要求相同；缺帧运行保留原记录，不冒充完整输入一致。",
              f"- 全部完成运行中，共 {budget} 帧的处理时间超过 16.67 ms。没有跳帧也不等于没有偶发慢帧。",
              "- A 旧问题位置的第 1142–1146 帧耗时保存在 summary.json 的 previous_gap_region；新缺口会单独保存 gap_window.json。",
              "- 各次为约 22 秒的独立回放。内存记录只反映这些运行，不能据此证明单进程连续任意长时间稳定。",
              "- 生产代码指纹在复测前后相同；本轮仅复测，未重跑单元测试、未调整算法或系统配置。",
              "- 本轮通过不能证明此前的偶发问题已经彻底消失。", "",
              "机器可读汇总：summary.json；逐次简表：runs.csv；核对脚本：verify.py；日志：verification.log。", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary["aggregates"], indent=2))
    print("Failed processes:", len(summary["failures"]))


if __name__ == "__main__":
    main()
