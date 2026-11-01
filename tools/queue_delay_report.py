"""只读逐帧记录，区分排队、交接成本与样本选择；不把跨系统时钟直接相减。"""

import argparse
from pathlib import Path

from customer_workspace.streamcopy_recording import write_json
from customer_workspace.tools.rtsp_tracking_report import read_json, read_rows
from customer_workspace.tools.tracking_latency_report import live_run, save_csv, stats


def overlap_ms(start, end, other_start, other_end):
    return max(0.0, min(end, other_end) - max(start, other_start)) * 1000


def numeric_rows(path):
    return [
        {key: float(value) for key, value in row.items()} for row in read_rows(path)
    ]


def source_map(root, name):
    rows = read_rows(root / "integrity_report" / f"{name}_source_mapping.csv")
    return {int(row["source_frame"]): int(row["original_frame"]) for row in rows}


def matched_comparison(left, right, map_left, map_right):
    """同一原始画面配对；独立运行的系统抖动和跳帧造成的历史差异仍须保留。"""
    one = {map_left[int(row["seq"])]: row for row in left}
    two = {map_right[int(row["seq"])]: row for row in right}
    if len(one) != len(left) or len(two) != len(right):
        raise ValueError("A repeated original frame needs an explicit cycle index")
    common = sorted(one.keys() & two.keys())
    paired = []
    for frame in common:
        a, b = one[frame], two[frame]
        row = {"original_frame": frame}
        for key in (
            "feedback_total_ms",
            "analysis_ms",
            "publish_wait_take_ms",
            "queue_residence_ms",
        ):
            if key not in a or key not in b:
                continue
            row[f"q1_{key}"] = a[key]
            row[f"q2_{key}"] = b[key]
            row[f"difference_{key}"] = b[key] - a[key]
        paired.append(row)
    summary = {
        "common_count": len(common),
        "q1_only": sorted(one.keys() - two.keys()),
        "q2_only": sorted(two.keys() - one.keys()),
        "q2_only_feedback_ms": [
            two[key]["feedback_total_ms"] for key in sorted(two.keys() - one.keys())
        ],
        "q1_all": stats([row["feedback_total_ms"] for row in left]),
        "q2_all": stats([row["feedback_total_ms"] for row in right]),
        "q1_common": stats([one[key]["feedback_total_ms"] for key in common]),
        "q2_common": stats([two[key]["feedback_total_ms"] for key in common]),
        "paired_duration_difference": stats(
            [row["difference_feedback_total_ms"] for row in paired]
        ),
    }
    return summary, paired


def historical(root, output):
    inputs = read_json(root / "inputs.json")
    all_rows, maps, slow, neighborhoods = {}, {}, [], []
    for name in inputs["runs"]:
        rows = numeric_rows(root / "report" / f"{name}_per_frame.csv")
        all_rows[name], maps[name] = rows, source_map(root, name)
        if "_q2_" not in name:
            continue
        timings = read_json(root / name / "latency_wsl_analysis.json")["frames"]
        frame_rows = {
            int(r["source_frame"]): r
            for r in read_rows(root / name / "analysis_frames.csv")
        }
        native = read_json(root / name / "latency_windows.json")
        sent = {row["seq"]: row for row in native["sent"]}
        by_seq = {row["seq"]: row for row in timings}
        neighbor_seqs = set()
        for row in timings:
            seq = row["seq"]
            old_wait = float(frame_rows[seq]["queue_wait_ms"])
            if old_wait <= 1000 / 60:
                continue
            neighbor_seqs.update(range(seq - 4, seq + 3))
            end = row["analysis_started_wsl_s"]
            start = end - old_wait / 1000
            preceding, occupied = [], 0.0
            for earlier in timings:
                if earlier["seq"] >= seq:
                    continue
                part = overlap_ms(
                    start,
                    end,
                    earlier["analysis_started_wsl_s"],
                    earlier["analysis_completed_wsl_s"],
                )
                if part:
                    occupied += part
                    preceding.append(
                        {
                            "seq": earlier["seq"],
                            "video_frame": earlier["video_frame"],
                            "analysis_ms": (
                                earlier["analysis_completed_wsl_s"]
                                - earlier["analysis_started_wsl_s"]
                            )
                            * 1000,
                            "overlap_ms": part,
                        }
                    )
            slow.append(
                {
                    "run": name,
                    "seq": seq,
                    "video_frame": row["video_frame"],
                    "original_frame": maps[name][seq],
                    "wait_ms": old_wait,
                    "prior_analysis_overlap_ms": occupied,
                    "remainder_ms": old_wait - occupied,
                    "prior_analysis_percent": occupied / old_wait * 100,
                    "preceding": preceding,
                }
            )
        for seq in sorted(neighbor_seqs):
            if seq not in by_seq or seq - 1 not in by_seq:
                continue
            row, previous = by_seq[seq], by_seq[seq - 1]
            neighborhoods.append(
                {
                    "run": name,
                    "seq": seq,
                    "video_frame": row["video_frame"],
                    "original_frame": maps[name][seq],
                    "windows_arrival_interval_ms": (
                        sent[seq]["received_windows_s"]
                        - sent[seq - 1]["received_windows_s"]
                    )
                    * 1000,
                    "wsl_ready_interval_ms": (
                        row["shared_copy_ready_wsl_s"]
                        - previous["shared_copy_ready_wsl_s"]
                    )
                    * 1000,
                    "wait_ms": float(frame_rows[seq]["queue_wait_ms"]),
                    "analysis_ms": (
                        row["analysis_completed_wsl_s"] - row["analysis_started_wsl_s"]
                    )
                    * 1000,
                }
            )
    comparisons, pairs = {}, []
    for video in ("A", "B"):
        left, right, common_left, common_right, extra = [], [], [], [], []
        without_handoff_left, without_handoff_right = [], []
        for repeat in (1, 2, 3):
            a, b = f"{video}_q1_r{repeat}", f"{video}_q2_r{repeat}"
            comparison, matched = matched_comparison(
                all_rows[a], all_rows[b], maps[a], maps[b]
            )
            left.extend(row["feedback_total_ms"] for row in all_rows[a])
            right.extend(row["feedback_total_ms"] for row in all_rows[b])
            common_left.extend(row["q1_feedback_total_ms"] for row in matched)
            common_right.extend(row["q2_feedback_total_ms"] for row in matched)
            # 仅对已有时间做账面扣除，不是假装重跑了一个没有等待的系统。
            without_handoff_left.extend(
                row["q1_feedback_total_ms"] - row["q1_publish_wait_take_ms"]
                for row in matched
            )
            without_handoff_right.extend(
                row["q2_feedback_total_ms"] - row["q2_publish_wait_take_ms"]
                for row in matched
            )
            extra.extend(comparison["q2_only_feedback_ms"])
            pairs.extend({"video": video, "repeat": repeat, **row} for row in matched)
        comparisons[video] = {
            "q1_all": stats(left),
            "q2_all": stats(right),
            "q1_common": stats(common_left),
            "q2_common": stats(common_right),
            "q2_only_feedback_ms": extra,
            "q1_common_excluding_handoff": stats(without_handoff_left),
            "q2_common_excluding_handoff": stats(without_handoff_right),
        }
        tails = {}
        for capacity in (1, 2):
            cohort = [
                row
                for repeat in (1, 2, 3)
                for row in all_rows[f"{video}_q{capacity}_r{repeat}"]
            ]
            cutoff = stats([row["feedback_total_ms"] for row in cohort])["p99"]
            tail = [row for row in cohort if row["feedback_total_ms"] >= cutoff]
            tails[f"q{capacity}"] = {
                "count": len(tail),
                "mean_ms": {
                    key: sum(row[key] for row in tail) / len(tail)
                    for key in (
                        "feedback_total_ms",
                        "windows_before_send_ms",
                        "receive_ack_preparation_ms",
                        "decode_ms",
                        "image_conversion_ms",
                        "shared_copy_ms",
                        "publish_wait_take_ms",
                        "analysis_ms",
                        "completion_ack_queue_ms",
                        "forward_plus_ack_return_ms",
                    )
                },
            }
        comparisons[video]["slowest_one_percent"] = tails
    save_csv(output / "historical_matched.csv", pairs)
    save_csv(output / "historical_neighborhoods.csv", neighborhoods)
    save_csv(
        output / "historical_five.csv",
        [
            {key: value for key, value in row.items() if key != "preceding"}
            for row in slow
        ],
    )
    return {"slow_frames": slow, "matched": comparisons}


def trace_run(folder, output):
    basic = live_run(folder, output)
    existing = numeric_rows(output / f"{folder.name}_per_frame.csv")
    analysis = read_json(folder / "latency_wsl_analysis.json")["frames"]
    indexed = {row["seq"]: row for row in analysis}
    native = read_json(folder / "latency_windows.json")
    acks = {
        row["seq"]: row
        for row in native["acknowledgements"]
        if row["kind"] == "complete"
    }
    producer = read_json(folder / "queue_producer.json")
    consumer = read_json(folder / "queue_consumer.json")
    loops = read_json(folder / "queue_loop.json")["events"]
    rows, costs, slow = [], [], []
    for event in producer["events"] + consumer["events"]:
        lock_requested = event.get("lock_requested_wsl_s", event["requested_wsl_s"])
        costs.append(
            {
                "seq": event["seq"],
                "operation": event["operation"],
                "elapsed_ms": (event["finished_wsl_s"] - event["requested_wsl_s"])
                * 1000,
                "lock_wait_ms": (event["locked_wsl_s"] - lock_requested) * 1000,
                "before_lock_ms": (lock_requested - event["requested_wsl_s"]) * 1000,
                "after_lock_ms": (event["finished_wsl_s"] - event["locked_wsl_s"])
                * 1000,
                "thread_cpu_ms": event["thread_cpu_ms"],
            }
        )
    for source in existing:
        seq = int(source["seq"])
        row, ack = indexed[seq], acks[seq]
        enter, take = row["entered_wsl_s"], row["taken_wsl_s"]
        part = {
            **source,
            "entered_wsl_s": enter,
            "taken_wsl_s": take,
            "analysis_started_wsl_s": row["processor_started_wsl_s"],
            "analysis_completed_wsl_s": row["analysis_completed_wsl_s"],
            "ack_sent_wsl_s": ack["ack_sent_wsl_s"],
            "ack_received_windows_s": ack["ack_received_windows_s"],
            "queue_residence_ms": (take - enter) * 1000,
            "before_enqueue_ms": (enter - row["shared_copy_ready_wsl_s"]) * 1000,
            "dequeue_to_call_ms": (row["processor_started_wsl_s"] - take) * 1000,
            "processor_ms": (
                row["analysis_completed_wsl_s"] - row["processor_started_wsl_s"]
            )
            * 1000,
            "processor_thread_cpu_ms": row["processor_thread_cpu_ms"],
            "processor_runnable_wait_ms": row["processor_counters"]["runnable_wait_s"]
            * 1000,
            **{
                key: value
                for key, value in row["processor_counters"].items()
                if key != "runnable_wait_s"
            },
        }
        if part["queue_residence_ms"] > 1000 / 60:
            occupied, preceding = 0.0, []
            for previous in analysis:
                amount = overlap_ms(
                    enter,
                    take,
                    previous["processor_started_wsl_s"],
                    previous["analysis_completed_wsl_s"],
                )
                if amount:
                    occupied += amount
                    preceding.append(
                        {
                            "seq": previous["seq"],
                            "overlap_ms": amount,
                            "processor_ms": (
                                previous["analysis_completed_wsl_s"]
                                - previous["processor_started_wsl_s"]
                            )
                            * 1000,
                            "thread_cpu_ms": previous["processor_thread_cpu_ms"],
                            "counters": previous["processor_counters"],
                        }
                    )
            loop_parts = {}
            for event in loops:
                amount = overlap_ms(
                    enter, take, event["started_wsl_s"], event["finished_wsl_s"]
                )
                if amount:
                    loop_parts[event["stage"]] = (
                        loop_parts.get(event["stage"], 0.0) + amount
                    )
            slow.append(
                {
                    "seq": seq,
                    "video_frame": part["video_frame"],
                    "queue_ms": part["queue_residence_ms"],
                    "prior_analysis_ms": occupied,
                    "loop_parts_ms": loop_parts,
                    "unassigned_ms": part["queue_residence_ms"]
                    - occupied
                    - sum(loop_parts.values()),
                    "preceding": preceding,
                }
            )
        rows.append(part)
    save_csv(output / f"{folder.name}_detailed.csv", rows)
    save_csv(output / f"{folder.name}_operations.csv", costs)
    copy_rows = producer["copies"]
    copy_stats = stats(
        [(r["finished_wsl_s"] - r["started_wsl_s"]) * 1000 for r in copy_rows]
    )
    operations = {}
    for operation in ("acquire", "publish", "take", "release"):
        selected = [r for r in costs if r["operation"] == operation]
        operations[operation] = {
            key: stats([r[key] for r in selected])
            for key in (
                "elapsed_ms",
                "lock_wait_ms",
                "thread_cpu_ms",
                "before_lock_ms",
                "after_lock_ms",
            )
        }
    detail_stats = {
        key: stats([r[key] for r in rows]) for key in rows[0] if key.endswith("_ms")
    }
    checks = {
        **basic["checks"],
        "one_copy_per_decoded_frame": len(copy_rows)
        == basic["received_ack_window"]
        == len({r["seq"] for r in copy_rows}),
        "timestamps_ordered": all(
            r["entered_wsl_s"]
            <= r["taken_wsl_s"]
            <= r["analysis_started_wsl_s"]
            <= r["analysis_completed_wsl_s"]
            <= r["ack_sent_wsl_s"]
            for r in rows
        ),
        "wsl_detailed_components_add_up": all(
            abs(
                r["before_enqueue_ms"]
                + r["queue_residence_ms"]
                + r["dequeue_to_call_ms"]
                + r["processor_ms"]
                - r["publish_wait_take_ms"]
                - r["analysis_ms"]
            )
            < 0.001
            for r in rows
        ),
    }
    return {
        "checks": checks,
        "stages": detail_stats,
        "operations": operations,
        "copy_ms": copy_stats,
        "copy_bytes": sorted({r["bytes"] for r in copy_rows}),
        "slow_wait_frames": slow,
        "processed": len(rows),
        "loop_stages": {
            stage: stats(
                [
                    (r["finished_wsl_s"] - r["started_wsl_s"]) * 1000
                    for r in loops
                    if r["stage"] == stage
                ]
            )
            for stage in sorted({r["stage"] for r in loops})
        },
    }, rows


def slow_frame_lines(summaries):
    """直接由本轮记录生成慢帧解释，不将一次运行的数值写成下次的结论。"""
    lines = [
        "## 本轮慢帧和调度检查",
        "",
        "下表是本轮真正排队超过一帧时间的记录。分析后整理包括确认提交、诊断读数、记录整理和归还；其余包括状态检查、取帧和分析前准备。单位 ms。",
        "",
        "| 组 | 源帧 | 排队 | 前序分析 | 分析后整理 | 其余 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in summaries.items():
        for row in summary["slow_wait_frames"]:
            post = row["loop_parts_ms"].get("post_analysis_and_release", 0)
            rest = row["queue_ms"] - row["prior_analysis_ms"] - post
            lines.append(
                f"| {name} | {row['seq']} | {row['queue_ms']:.3f} | {row['prior_analysis_ms']:.3f} | {post:.3f} | {rest:.3f} |"
            )
    post_stats = [
        row["loop_stages"]["post_analysis_and_release"] for row in summaries.values()
    ]
    lines += [
        "",
        f"分析后整理阶段（包含诊断）在各组的中位耗时范围为 {min(s['median'] for s in post_stats):.3f}–{max(s['median'] for s in post_stats):.3f} ms，P99 范围为 {min(s['p99'] for s in post_stats):.3f}–{max(s['p99'] for s in post_stats):.3f} ms。这部分包含原有工作，不能全部当作新增诊断的成本。",
        "两帧机制没有增加分析进程或线程。原始记录还包含主线程 CPU 时间、等待 CPU 计数及内存缺页计数，详见 summary.json 的 preceding 字段；不能把主线程以外的全部时间都叫作系统调度，也不能用本轮计数倒推上一轮的操作系统状态。",
        "诊断读数自身会影响短测，因此不能把本轮绝对耗时视为正式默认运行的无干扰结果。一帧等待也会排队，只是满后更早替换旧画面。",
        "",
    ]
    return lines


def build(root, output):
    output.mkdir(parents=True, exist_ok=False)
    history = historical(Path(read_json(root / "plan.json")["historical_root"]), output)
    inputs = read_json(root / "inputs.json")
    summaries, rows, comparisons = {}, {}, {}
    for name in inputs["runs"]:
        summaries[name], rows[name] = trace_run(root / name, output)
    for mode in ("file", "rtsp"):
        for video in ("A", "B"):
            a, b = f"{mode}_{video}_q1", f"{mode}_{video}_q2"
            comparison, pairs = matched_comparison(
                rows[a], rows[b], source_map(root, a), source_map(root, b)
            )
            comparisons[f"{mode}_{video}"] = comparison
            save_csv(output / f"{mode}_{video}_matched.csv", pairs)
    report = {"historical": history, "runs": summaries, "matched": comparisons}
    write_json(output / "summary.json", report)
    lines = [
        "# 一帧 / 两帧等待的延迟来源",
        "",
        "**结论：上一轮较慢的尾部反馈，最大的增加来自排队等前面的分析；没有发现两帧机制新增整图复制或毫秒级锁等待。仅多保留若干画面造成的统计变化很小。解码、转换等运行波动也贡献了差异，因此不能把 1.3–2.2 ms 当成固定成本。**",
        "",
        "诊断仅使用原 A/B 视频和 Windows F 盘模拟源；算法、默认一帧设置和设备配置保持不变。每组约 22 秒，没有长期压力测试。",
        "",
        "## 上一轮的 5 帧",
        "",
        "旧记录的等待包括发布、唤醒和取帧。本表逐段与前面画面的分析调用对齐，帧号从 0 开始。",
        "",
        "| 组 | 源帧 / 保存帧 / 原视频帧 | 等待 ms | 等前面分析 ms | 其余 ms |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in history["slow_frames"]:
        lines.append(
            f"| {row['run']} | {row['seq']} / {row['video_frame']} / {row['original_frame']} | {row['wait_ms']:.3f} | {row['prior_analysis_overlap_ms']:.3f} | {row['remainder_ms']:.3f} |"
        )
    lines += [
        "",
        "前面画面的分析占用了约 97%–99% 的等待区间。A 的积压起于 25.021 ms 的分析，B 起于 29.694 ms 的分析。旧记录没有线程运行计数，不能倒推这两次分析究竟有多少时间在计算或等待 CPU。",
        "A 中源帧 191→192 到达 Windows 的间隔只有 0.133 ms；WSL 准备好源帧 192→193 的间隔只有 6.459 ms。B 中源帧 1237→1238 到达间隔为 9.030 ms，WSL 准备间隔为 8.336 ms。因此是到达不均匀与前序分析变慢叠加，而不是等待区刻意凑足两张才开始分析。完整邻近记录见 historical_neighborhoods.csv。",
        "",
        "## 样本是否造成 P99 增加",
        "",
        "按原视频逐帧像素确认画面位置后，只比较两方案都处理的画面。",
        "",
        "| 上一轮 | 全部 q1 P99 | 全部 q2 P99 | 相同画面 q1 P99 | 相同画面 q2 P99 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for video, row in history["matched"].items():
        lines.append(
            f"| {video} | {row['q1_all']['p99']:.3f} | {row['q2_all']['p99']:.3f} | {row['q1_common']['p99']:.3f} | {row['q2_common']['p99']:.3f} |"
        )
    lines += [
        "",
        "分别取各组反馈最慢的 1%（各 40 帧）检查时间构成，得到下表。这里是最慢 1% 的平均值，不是 P99，也不是把不同步骤的 P99 相加。两组慢帧名单不同，此表只解释已有尾部耗时的组成。",
        "",
        "| 视频 | 最慢 1% 平均反馈 q1→q2 | 其中交接等待 q1→q2 |",
        "| --- | --- | --- |",
    ]
    for video, row in history["matched"].items():
        a, b = [row["slowest_one_percent"][key]["mean_ms"] for key in ("q1", "q2")]
        lines.append(
            f"| {video} | {a['feedback_total_ms']:.3f} → {b['feedback_total_ms']:.3f} | {a['publish_wait_take_ms']:.3f} → {b['publish_wait_take_ms']:.3f} |"
        )
    lines += [
        "",
        "交接等待是这部分增加中最大的来源；解码、图像转换和 Windows 发出前等待也会随运行波动。把交接等待从每条记录账面扣除后，两帧组仍有约 0.6 ms（A）、1.3 ms（B）的 P99 差异，不能把全部增量都归到容量变化。账面扣除不是重新运行无等待系统，不可当作优化收益。",
        "移除两帧方案额外分析的画面后，P99 改变不足 0.006 ms。因此上一轮约 1.3–2.2 ms 的增加并非主要来自多统计了几帧。独立 RTSP 运行的到达节奏和系统抖动不同，不能将所有差值视为等待区容量的固定代价。",
        "",
        "## 本轮短测",
        "",
        "file 从同一原视频第 0 帧播放；rtsp 使用实时模拟源。每种模式、每段视频、每种容量各一遍，不以短测推断长期保证。",
        "",
        "| 组 | 已分析 | 反馈 P95 | 反馈 P99 | 真正排队 P99 | 排队最大 | >16.67 ms 帧数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in summaries.items():
        feedback, waiting = (
            row["stages"]["feedback_total_ms"],
            row["stages"]["queue_residence_ms"],
        )
        lines.append(
            f"| {name} | {row['processed']} | {feedback['p95']:.3f} | {feedback['p99']:.3f} | {waiting['p99']:.3f} | {waiting['max']:.3f} | {len(row['slow_wait_frames'])} |"
        )
    lines += [
        "",
        "## 相同画面配对",
        "",
        "| 组 | 共同画面 | 全部 q1/q2 P99 | 共同画面 q1/q2 P99 | q1/q2 独有画面 |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for name, row in comparisons.items():
        lines.append(
            f"| {name} | {row['common_count']} | {row['q1_all']['p99']:.3f} / {row['q2_all']['p99']:.3f} | {row['q1_common']['p99']:.3f} / {row['q2_common']['p99']:.3f} | {len(row['q1_only'])} / {len(row['q2_only'])} |"
        )
    lines += [
        "",
        "## 复制和锁",
        "",
        "每帧仅执行原有的一次灰度图复制；取帧返回共享数组的视图，没有因为两帧等待增加整图复制。多出的空间在启动时分配。",
        "",
        "表内为各项 P99，单位 ms；各项的 P99 不可直接相加。",
        "",
        "| 组 | 复制 | 申请空位 | 发布 | 取出 | 归还 | 最慢的锁等待 P99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in summaries.items():
        ops = row["operations"]
        numbers = [row["copy_ms"]["p99"]] + [
            ops[k]["elapsed_ms"]["p99"]
            for k in ("acquire", "publish", "take", "release")
        ]
        numbers += [max(op["lock_wait_ms"]["p99"] for op in ops.values())]
        lines.append(
            f"| {name} | " + " | ".join(f"{value:.3f}" for value in numbers) + " |"
        )
    lines += [
        "",
        *slow_frame_lines(summaries),
        "## 时间解释及边界",
        "",
        "- 真正排队：共享帧发布完成到分析端将它取出；所有时间来自 WSL 同一单调时钟。",
        "- 完整反馈：Windows 收到压缩帧到 Windows 收到分析完成确认；使用 Windows 同一个时钟。没有直接相减 Windows 与 WSL 时间，也没有用往返时间除以二。",
        "- processor_thread_cpu_ms 仅为分析主线程的 CPU 时间，不能代替 OpenCV 所有工作线程的总时间；runnable_wait 是该线程已可运行但等 CPU 的累计时间。计数窗口稍微包括紧随分析后的确认提交和计数读取。",
        "- 细分日志仅在 --diagnose-queue 时启用，短测结束再写盘；读计数和记录自身也有小成本，默认运行不启用这些诊断。",
        "- 排队时前一帧在运行，属于顺序处理的正常等待，并不等同于锁争用。具体前序帧及每段重叠时间见 summary.json。",
        "- 本轮没有改变等待策略，也没有处理导致分析偶发变慢的算法或操作系统问题。",
        "",
        "## 保存核对",
        "",
        "逐帧录像像素、帧数去向、数据库保存及同输入离线重算结果见 ../integrity_report/summary.json。每条诊断记录的时间顺序、确认完整性与时间分段求和见 summary.json 中 checks。",
        "",
        "详细文件：*_detailed.csv 为逐帧时间和耗时，*_operations.csv 为复制之外的队列操作，*_matched.csv 为相同原画面的配对比较，historical_five.csv 为上一轮 5 帧。原始复制记录在每组 queue_producer.json。",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "REPORT.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
