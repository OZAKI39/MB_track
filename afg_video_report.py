"""把真实 AFG 发送/读回时间对应到视频接收时间；不推断曝光或声波生效时间。"""

import argparse
import bisect
import csv
import json
import math
import re
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from customer_workspace.experiment_runner import (
    CHANNEL_NAMES,
    IDLE_AMPLITUDE_VPP,
    ExperimentStep,
)
from customer_workspace.parallel_video_report import check_branches, compare_analysis
from customer_workspace.streamcopy_recording import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame_range(frames: list[dict], windows_s: float, clock_margin_s=0.0) -> dict:
    """用动作前后实际收到的帧夹住时刻；超出录像就明确无对应，绝不外推。"""
    times = [r["received_windows_s"] for r in frames]
    if not times or any(b < a for a, b in pairwise(times)):
        raise ValueError("视频接收时间缺失或倒退")
    low, high = windows_s - clock_margin_s, windows_s + clock_margin_s
    if low < times[0] or high > times[-1]:
        return {"mapped": False, "reason": "outside_recorded_receive_times"}
    first = max(0, bisect.bisect_right(times, low) - 1)
    last = min(len(times) - 1, bisect.bisect_left(times, high))
    return {
        "mapped": True,
        "first_frame": frames[first]["frame_index"],
        "last_frame": frames[last]["frame_index"],
        "video_from_s": frames[first]["video_s"],
        "video_to_s": frames[last]["video_s"],
        "receive_bracket_ms": (times[last] - times[first]) * 1000,
        "video_bracket_ms": (frames[last]["video_s"] - frames[first]["video_s"]) * 1000,
    }


def linked_frames(session, probe, experiment_id):
    packets = [r for r in probe["packets_and_frames"] if r["type"] == "packet"]
    recorded = session["record_frames"]
    if len(packets) != len(recorded):
        raise ValueError("录像入口与最终文件帧数不等，不能逐帧对应")
    base = Fraction(probe["streams"][0]["time_base"])
    decoded = {r["seq"]: r["decoded_windows_mono"] for r in session["decoded_frames"]}
    return [
        {
            "experiment_id": experiment_id,
            "frame_index": index,
            "source_seq": frame["seq"],
            "video_s": float(packet["pts"] * base),
            "received_windows_s": frame["received_windows_mono"],
            "decoded_windows_s": decoded.get(frame["seq"]),
        }
        for index, (frame, packet) in enumerate(zip(recorded, packets))
    ]


def amplitude_confirmation(event, afg, config, origin):
    """关联同一计划步骤内该通道的真实幅值回复，不能用最后一次总核查代替。"""
    channel = event["channel"]
    checks = [
        c for c in afg["checks"] if c["step_at_s"] == event["planned_s"] and c["passed"]
    ]
    if len(checks) != 1:
        return None
    check = checks[0]
    queries = [
        q
        for q in check["queries"]
        if q["target"] == channel and q["command"].upper().endswith(":AMPLITUDE?")
    ]
    if len(queries) != 1:
        return None
    query = queries[0]
    replies = [
        r
        for r in afg["communication"]
        if r["event"] == "reply"
        and r["device"] == config["channels"][channel]["device"]
        and r["command"] == query["command"]
        and origin + query["started_s"]
        <= r["received_windows_s"]
        <= origin + query["finished_s"]
    ]
    if len(replies) != 1:
        return None
    reply = replies[0]
    if not math.isclose(
        float(reply["reply"]), event["amplitude_requested_vpp"], abs_tol=0.00051
    ):
        return None
    return reply


def collect_actions(events, afg, config, frames, margin):
    origin = afg["origin"]["origin_windows_s"]
    actions = []
    for event in events:
        if (
            event.get("event") != "action_finished"
            or event.get("action") != "send_amplitude"
        ):
            continue
        timing = event.get("amplitude_timing", {})
        sent = timing.get("command_sent_s")
        if sent is None:
            continue
        reply = amplitude_confirmation(event, afg, config, origin)
        actions.append(
            {
                "kind": "amplitude",
                "channel": event["channel"],
                "state": event["state_requested"],
                "amplitude_vpp": event["amplitude_requested_vpp"],
                "planned_s": event["planned_s"],
                "command_windows_s": origin + sent,
                "write_returned_windows_s": origin + timing["command_write_returned_s"]
                if timing.get("command_write_returned_s") is not None
                else None,
                "confirmation_windows_s": reply["received_windows_s"]
                if reply
                else None,
                "readback": reply["reply"] if reply else None,
                "status": event["status"],
            }
        )
    mapping = {
        (c["device"], c["channel"]): name for name, c in config["channels"].items()
    }
    stopped = set()
    for event in afg["communication"]:
        if event["event"] != "write" or event["started_windows_s"] < origin:
            continue
        match = re.fullmatch(r"OUTPUT([12]):STATE OFF", event["command"].upper())
        if match is None:
            continue
        channel = mapping[event["device"], int(match.group(1))]
        query = f"OUTPut{match.group(1)}:STATe?"
        replies = [
            r
            for r in afg["communication"]
            if r["event"] == "reply"
            and r["device"] == event["device"]
            and r["command"] == query
            and r["received_windows_s"] >= event["finished_windows_s"]
        ]
        reply = replies[0] if replies else None
        confirmed = reply is not None and reply["reply"].strip().upper() in ("0", "OFF")
        actions.append(
            {
                "kind": "stop_repeat" if channel in stopped else "stop",
                "channel": channel,
                "state": "STOP",
                "amplitude_vpp": None,
                "planned_s": 20
                if afg["validation_mode"] == "normal" and channel not in stopped
                else None,
                "command_windows_s": event["started_windows_s"],
                "write_returned_windows_s": event["finished_windows_s"],
                "confirmation_windows_s": reply["received_windows_s"]
                if confirmed
                else None,
                "readback": reply["reply"] if reply else None,
                "status": "confirmed" if confirmed else "unconfirmed",
            }
        )
        stopped.add(channel)
    actions.sort(key=lambda r: r["command_windows_s"])
    for index, action in enumerate(actions, 1):
        action["action_id"] = f"action_{index:03d}"
        action["experiment_id"] = afg["experiment_id"]
        action["command_elapsed_s"] = action["command_windows_s"] - origin
        action["confirmation_elapsed_s"] = (
            action["confirmation_windows_s"] - origin
            if action["confirmation_windows_s"] is not None
            else None
        )
        action["command_video"] = frame_range(
            frames, action["command_windows_s"], margin
        )
        action["confirmation_video"] = (
            frame_range(frames, action["confirmation_windows_s"], margin)
            if action["confirmation_windows_s"] is not None
            else {"mapped": False, "reason": "no_readback_confirmation"}
        )
        if action["command_video"]["mapped"] and action["confirmation_video"]["mapped"]:
            action["software_confirmation_window_s"] = [
                action["command_video"]["video_from_s"],
                action["confirmation_video"]["video_to_s"],
            ]
    return actions


def plot_timeline(output, actions, frames, mode):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch

    from customer_workspace.pzt_video_review import find_font

    font_manager.fontManager.addfont(str(find_font(None)))
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(13, 5))
    end = frames[-1]["video_s"]
    ax.broken_barh([(0, end)], (4.65, 0.6), facecolors="#d5e1e7")
    ax.vlines(
        [r["video_s"] for r in frames[::5]], 4.7, 5.2, color="#819cac", linewidth=0.4
    )
    for index, channel in enumerate(("PZT1", "PZT2", "PZT3", "PZT4")):
        y, active = 3 - index, None
        for action in [
            r
            for r in actions
            if r["channel"] == channel and r["command_video"]["mapped"]
        ]:
            at = (
                action["command_video"]["video_from_s"]
                + action["command_video"]["video_to_s"]
            ) / 2
            if action["state"] == "ACTIVE":
                active = at
            elif active is not None:
                ax.broken_barh(
                    [(active, at - active)], (y - 0.28, 0.56), facecolors="#368fbe"
                )
                active = None
            if action["confirmation_video"]["mapped"]:
                confirmed = action["confirmation_video"]["video_to_s"]
                ax.broken_barh(
                    [(at, max(0, confirmed - at))],
                    (y - 0.35, 0.12),
                    facecolors="#df9b35",
                )
            if action["kind"] == "stop":
                ax.plot(at, y, "rx", markersize=8)
    ax.axvline(20, linestyle="--", color="grey", linewidth=1)
    ax.set_yticks([5, 3, 2, 1, 0], ["Video", "PZT1", "PZT2", "PZT3", "PZT4"])
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.7, 5.7)
    ax.set_xticks(range(0, 23, 1))
    ax.set_xlabel("Video time (s)")
    ax.set_title(f"PZT commands and received video: {mode}")
    ax.legend(
        handles=[
            Patch(color="#368fbe", label="Commanded active state"),
            Patch(color="#df9b35", label="Command to readback"),
        ],
        loc="upper right",
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "timeline.png", dpi=160)
    plt.close(fig)


def check_plan_execution(plan: dict, actions: list[dict], checks: list[dict]) -> dict:
    """按实际保存的时间表核对通道、顺序、幅值和检查阶段，不固定要求12条动作。"""
    expected = []
    states = dict.fromkeys(CHANNEL_NAMES, "IDLE")
    expected_checks = []
    for row in plan["steps"]:
        step = ExperimentStep(**row)
        if step.state == "STOP":
            continue
        expected_checks.append(step.at_s)
        for channel, state in step.changes_from(states):
            amplitude = (
                plan["settings"][channel]["amplitude_vpp"]
                if state == "ACTIVE"
                else IDLE_AMPLITUDE_VPP
            )
            expected.append(
                {
                    "planned_s": step.at_s,
                    "channel": channel,
                    "state": state,
                    "amplitude_vpp": amplitude,
                }
            )
            states[channel] = state
    actual = [
        {key: action[key] for key in ("planned_s", "channel", "state", "amplitude_vpp")}
        for action in actions
        if action["kind"] == "amplitude"
    ]
    actual_checks = [check["step_at_s"] for check in checks]
    problems = []
    if actual != expected:
        problems.append("实际调幅的通道、顺序、幅值或计划时刻与时间表不符")
    if actual_checks != expected_checks or not all(check["passed"] for check in checks):
        problems.append("稳定核查阶段缺失、重复或未通过")
    return {
        "passed": not problems,
        "problems": problems,
        "expected_actions": expected,
        "actual_actions": actual,
        "expected_check_steps_s": expected_checks,
        "actual_check_steps_s": actual_checks,
    }


def create_report(folder, baseline, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(folder / "link_session.json")
    afg = read(folder / "afg_link_summary.json")
    camera_clock = read(folder / "camera_clock.json")
    experiment = read(folder / "afg_experiment/plan.json")
    session, probe = (
        read(folder / "gst_session.json"),
        read(folder / "recording/probe.json"),
    )
    for item in (
        afg,
        camera_clock,
        afg["clock_before"]["identity"],
        afg["clock_after"]["identity"],
    ):
        if item["experiment_id"] != manifest["experiment_id"]:
            raise ValueError("跨实验的文件不能关联")
    with (folder / "afg_experiment/events.jsonl").open(encoding="utf-8") as file:
        events = [json.loads(line) for line in file]
    frames = linked_frames(session, probe, manifest["experiment_id"])
    margin = max(
        abs(v)
        for name in ("clock_before", "clock_after")
        for v in afg[name]["offset_interval_s"]
    )
    actions = collect_actions(events, afg, experiment["afg_config"], frames, margin)
    recording = check_branches(
        session, probe, read(folder / "recording/recording_check.json")
    )
    current_analysis = read(folder / "analysis_check.json")
    uses_tracking = current_analysis.get("engine") == "microbubble_tracking.OnlineProcessor"
    analysis = compare_analysis(
        read(baseline / "analysis_check.json"),
        current_analysis,
        {
            "analysis_at_least_hz": 60 if uses_tracking else 5.6,
            "analysis_rate_ratio_to_baseline_min": 0.8,
            "maximum_analysis_completion_gap_ms": 100 if uses_tracking else 1000,
        },
    )
    problems = []
    if not afg["passed"]:
        problems.append("AFG 执行、稳定读回或安全关断验收未通过")
    if not actions or any(
        not a["command_video"]["mapped"] or not a["confirmation_video"]["mapped"]
        for a in actions
    ):
        problems.append("有动作缺少确认或不能对应到录像")
    parameter_actions = [r for r in actions if r["kind"] == "amplitude"]
    plan_check = None
    if manifest["mode"] == "normal":
        plan_check = check_plan_execution(experiment["plan"], actions, afg["checks"])
        problems.extend(plan_check["problems"])
    if {r["channel"] for r in actions if r["kind"] == "stop"} != {
        "PZT1",
        "PZT2",
        "PZT3",
        "PZT4",
    }:
        problems.append("没有记录到四路分别关闭")
    requests = afg["requests"]
    if manifest["mode"] != "normal":
        selected = [r for r in requests if r["kind"] == manifest["mode"]]
        if len(selected) != 1 or any(
            r["command_windows_s"] >= selected[0]["windows_s"]
            for r in parameter_actions
        ):
            problems.append("取消/异常请求缺失，或请求后仍发送了调幅命令")
    report = {
        "passed": recording["passed"] and analysis["passed"] and not problems,
        "experiment_id": manifest["experiment_id"],
        "mode": manifest["mode"],
        "problems": problems,
        "recording": recording,
        "analysis": analysis,
        "afg_passed": afg["passed"],
        "afg_status": afg["actual_status"],
        "plan_check": plan_check,
        "clock_margin_s": margin,
        "actions": actions,
        "max_command_receive_bracket_ms": max(
            (r["command_video"].get("receive_bracket_ms", 0) for r in actions),
            default=None,
        ),
        "max_command_video_bracket_ms": max(
            (r["command_video"].get("video_bracket_ms", 0) for r in actions),
            default=None,
        ),
        "precision_notes": [
            "两端为同一Windows主机的公共monotonic/QPC时钟，实验前后双向核对，不使用WSL计时器做差。",
            "对应范围表示动作调用前后接收端收到的帧；接收时间取压缩帧解析后，解码时间另存。",
            "AFG确认是切换后约1秒的稳定读回；发送到确认之间是软件确认区间，不是模拟电压变化区间。",
            "时钟核对包络和帧范围不能覆盖未知的曝光、相机编码/网络传输和AFG模拟输出延迟；物理同步误差未知。",
            "视频帧编号从0开始；工作段按真实命令发送推算，不表示已测得声波强弱。",
        ],
    }
    write_json(output / "link_report.json", report)
    with (output / "video_frames.csv").open(
        "x", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(frames[0]))
        writer.writeheader()
        writer.writerows(frames)
    rows = []
    for action in actions:
        row = {k: v for k, v in action.items() if not isinstance(v, (dict, list))}
        for label in ("command_video", "confirmation_video"):
            row.update({label + "_" + k: v for k, v in action[label].items()})
        rows.append(row)
    if rows:
        with (output / "actions.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
            )
            writer.writeheader()
            writer.writerows(rows)
        database = folder / "tracking/tracking.sqlite"
        if database.is_file():
            from customer_workspace.microbubble_tracking.online_replay import (
                link_actions,
            )

            tracking_link = link_actions(
                database, output / "actions.csv", output / "actions_tracking.csv"
            )
            write_json(output / "tracking_link.json", tracking_link)
    plot_timeline(output, actions, frames, manifest["mode"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = create_report(args.folder, args.baseline, args.output_dir)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "problems": result["problems"],
                "recording_problems": result["recording"]["problems"],
                "analysis_problems": result["analysis"]["problems"],
                "max_receive_bracket_ms": result["max_command_receive_bracket_ms"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(0 if result["passed"] else 1)
