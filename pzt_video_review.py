"""离线合成原始视频与动态 PZT 时间轴；只读实验记录，不连接设备。"""

import argparse
import csv
import hashlib
import json
import math
import time
import traceback
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import av
from PIL import Image, ImageDraw, ImageFont

from customer_workspace.afg_video_report import collect_actions, linked_frames, read
from customer_workspace.streamcopy_recording import write_json

CHANNELS = ("PZT1", "PZT2", "PZT3", "PZT4")
COLORS = {"ACTIVE": "#368fbe", "IDLE": "#dfe7ed", "STOP": "#71818f"}
PANEL_WIDTH = 800
MIN_HEIGHT = 1024


def sha256(path: Path) -> str:
    """分块核对输入文件，避免大视频一次读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evidence(experiment: Path, report_dir: Path) -> dict:
    """复用已有时间关联，核对实验身份及原始动作，拒绝混用不同实验。"""
    paths = {
        "video": experiment / "recording/video.mp4",
        "manifest": experiment / "link_session.json",
        "afg": experiment / "afg_link_summary.json",
        "plan": experiment / "afg_experiment/plan.json",
        "events": experiment / "afg_experiment/events.jsonl",
        "session": experiment / "gst_session.json",
        "probe": experiment / "recording/probe.json",
        "report": report_dir / "link_report.json",
        "ready": experiment / "afg_ready.json",
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    manifest, afg, report = (
        read(paths[name]) for name in ("manifest", "afg", "report")
    )
    experiment_id = manifest["experiment_id"]
    ready = read(paths["ready"])
    if any(item["experiment_id"] != experiment_id for item in (afg, report, ready)):
        raise ValueError("视频与动作报告不属于同一次实验")
    if not math.isclose(ready["idle_vpp"], manifest["idle_vpp"], abs_tol=1e-9):
        raise ValueError("准备记录与待机幅值不一致")
    frames = linked_frames(read(paths["session"]), read(paths["probe"]), experiment_id)
    with paths["events"].open(encoding="utf-8") as file:
        events = [json.loads(line) for line in file]
    actions = collect_actions(
        events, afg, read(paths["plan"])["afg_config"], frames, report["clock_margin_s"]
    )
    if actions != report["actions"]:
        raise ValueError("动作报告与原始记录不一致，不能合成误导性的时间轴")
    if not actions or any(not row["command_video"]["mapped"] for row in actions):
        raise ValueError("存在无法对应到录像的动作，请先检查时间关联报告")
    return {
        "paths": paths,
        "hashes": hashes,
        "manifest": manifest,
        "report": report,
        "frames": frames,
        "actions": actions,
    }


def status_at(actions: list[dict], video_s: float) -> tuple[dict | None, str]:
    """依据当前播放时间显示状态；不提前使用未来的读回确认。"""
    current = None
    for action in actions:
        bounds = action["command_video"]
        if video_s < bounds["video_from_s"]:
            break
        if video_s < bounds["video_to_s"]:
            return action, "Switching"
        current = action
    if current is None:
        return None, "Preset idle"
    reply = current["confirmation_video"]
    if not reply["mapped"]:
        return current, "Unconfirmed"
    if video_s < reply["video_to_s"]:
        return current, "Awaiting readback"
    return current, "Readback received"


def find_font(requested: Path | None) -> Path:
    """Find actual Arial and reject substitutes, including explicitly supplied fonts."""
    candidates = (
        [requested]
        if requested
        else [
            Path("/mnt/c/Windows/Fonts/arial.ttf"),
            Path.home() / ".local/share/fonts/arial.ttf",
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"),
        ]
    )
    for path in candidates:
        if path.is_file():
            family, _ = ImageFont.truetype(str(path), 20).getname()
            if family != "Arial":
                raise ValueError(f"Arial is required; found {family!r}: {path}")
            return path
    raise FileNotFoundError(
        "Arial is required. Supply an installed Arial TTF with --font."
    )


class TimelinePanel:
    """缓存静态时间轴，每帧仅绘制播放指针、时间和四路当前记录状态。"""

    def __init__(self, evidence: dict, duration_s: float, height: int, font: Path):
        self.evidence, self.duration_s = evidence, duration_s
        font = find_font(font)
        self.fonts = {
            size: ImageFont.truetype(str(font), size) for size in (16, 18, 20, 24, 30)
        }
        self.actions = {
            channel: [
                a
                for a in evidence["actions"]
                if a["channel"] == channel and a["kind"] != "stop_repeat"
            ]
            for channel in CHANNELS
        }
        self.left, self.right = 214, PANEL_WIDTH - 30
        self.background = Image.new("RGB", (PANEL_WIDTH, height), "#f5f8fb")
        self._draw_background()

    def x(self, video_s: float) -> int:
        return round(
            self.left
            + min(max(video_s, 0), self.duration_s)
            / self.duration_s
            * (self.right - self.left)
        )

    def text(self, draw, position, content, size=20, fill="#233747"):
        draw.text(position, content, font=self.fonts[size], fill=fill)

    def _draw_background(self):
        draw = ImageDraw.Draw(self.background)
        self.text(draw, (26, 22), "PZT timeline", 30)
        self.text(draw, (26, 210), "Command state", 20)
        self.text(draw, (self.left, 210), "Video time (s)", 20)
        tick = max(2, math.ceil(self.duration_s / 11))
        for second in range(0, math.ceil(self.duration_s) + 1, tick):
            if second > self.duration_s:
                break
            x = self.x(second)
            draw.line((x, 266, x, 696), fill="#d9e2e8", width=1)
            draw.text(
                (x, 242),
                f"{second:g}",
                font=self.fonts[16],
                fill="#526575",
                anchor="mt",
            )
        for index, channel in enumerate(CHANNELS):
            y = 288 + 100 * index
            start, state = 0.0, "IDLE"
            for action in self.actions[channel]:
                bounds = action["command_video"]
                middle = (bounds["video_from_s"] + bounds["video_to_s"]) / 2
                draw.rectangle(
                    (self.x(start), y, self.x(middle), y + 36), fill=COLORS[state]
                )
                start, state = middle, action["state"]
            draw.rectangle((self.x(start), y, self.right, y + 36), fill=COLORS[state])
            for action in self.actions[channel]:
                bounds, confirmed = (
                    action["command_video"],
                    action["confirmation_video"],
                )
                left, right = (
                    self.x(bounds["video_from_s"]),
                    self.x(bounds["video_to_s"]),
                )
                # 范围在总览中可能不足一个像素，至少画 2 像素以提醒这里不是精确切换瞬间。
                draw.rectangle(
                    (left, y - 3, max(left + 1, right), y + 39),
                    outline="#df9b35",
                    width=2,
                )
                if confirmed["mapped"]:
                    end = self.x(confirmed["video_to_s"])
                    draw.line((left, y + 48, end, y + 48), fill="#df9b35", width=5)
                    draw.line((end, y + 43, end, y + 53), fill="#328562", width=3)
                else:
                    self.text(draw, (left, y + 42), "?", 18, "#b13d35")
        for x, color, label in (
            (26, COLORS["ACTIVE"], "Active"),
            (226, COLORS["IDLE"], "Idle"),
            (426, COLORS["STOP"], "Off"),
        ):
            draw.rectangle((x, 726, x + 24, 747), fill=color)
            self.text(draw, (x + 34, 722), label, 18)
        notes = [
            "Orange: command window / scheduled readback",
            "Green: readback received",
            "Software timing; exposure time unknown",
        ]
        for index, note in enumerate(notes):
            self.text(draw, (26, 772 + index * 30), note, 18)

    def render(self, video_s: float, index: int, frame_count: int) -> Image.Image:
        panel = self.background.copy()
        draw = ImageDraw.Draw(panel)
        self.text(draw, (26, 90), f"t = {video_s:.3f} s", 24)
        for row, channel in enumerate(CHANNELS):
            y = 280 + 100 * row
            action, confirmation = status_at(self.actions[channel], video_s)
            state = action["state"] if action else "IDLE"
            if state == "STOP":
                label = "Off"
            else:
                vpp = (
                    action["amplitude_vpp"]
                    if action
                    else self.evidence["manifest"]["idle_vpp"]
                )
                label = (
                    f"{'Active' if state == 'ACTIVE' else 'Idle'} {vpp * 1000:g} mVpp"
                )
            self.text(draw, (26, y), channel, 24)
            self.text(draw, (26, y + 34), label, 18)
            self.text(
                draw,
                (26, y + 63),
                confirmation,
                16,
                "#328562" if confirmation == "Readback received" else "#9d6820",
            )
        cursor = self.x(video_s)
        draw.line((cursor, 263, cursor, 696), fill="#cf3c3c", width=3)
        draw.polygon(
            ((cursor - 6, 260), (cursor + 6, 260), (cursor, 269)), fill="#cf3c3c"
        )
        return panel


def check_frame_time(frame, expected: dict) -> Fraction:
    """使用原文件播放时间，绝不按帧号/假定帧率重造时间轴。"""
    if frame.pts is None or frame.time_base is None:
        raise ValueError("原视频存在没有播放时间的帧")
    timestamp = frame.pts * frame.time_base
    if abs(float(timestamp) - expected["video_s"]) > 1e-9:
        raise ValueError(f"第 {expected['frame_index']} 帧播放时间与关联记录不符")
    return timestamp


def verify_video(
    path: Path,
    expected_times: list[Fraction],
    expected_end: Fraction,
    size: tuple[int, int],
    preview: Path,
) -> dict:
    """完整解码审查视频，逐帧核对时间；缩略图也从实际编码后的文件提取。"""
    times = []
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        if (stream.width, stream.height) != size:
            raise ValueError("合成视频尺寸与原画面保留范围不一致")
        end = (
            stream.start_time or 0
        ) * stream.time_base + stream.duration * stream.time_base
        for index, frame in enumerate(video.decode(stream)):
            timestamp = frame.pts * frame.time_base
            if index >= len(expected_times) or timestamp != expected_times[index]:
                raise ValueError(f"合成视频第 {index} 帧时间改变或增加了帧")
            times.append(timestamp)
            if index == int(len(expected_times) * 0.75):
                frame.to_image().save(preview)
        if len(times) != len(expected_times) or end != expected_end:
            raise ValueError("合成视频帧数或总播放时间与原视频不同")
    return {
        "decoded_frames": len(times),
        "all_frame_times_identical": True,
        "duration_s": float(end - times[0]),
        "size": list(size),
    }


def render_review(experiment: Path, report_dir: Path, output: Path, font: Path) -> dict:
    """逐帧合成并检查新 MP4；独占创建结果目录，失败时保存原因与已有文件。"""
    font = find_font(font)
    evidence = load_evidence(experiment, report_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {
        "experiment_id": evidence["manifest"]["experiment_id"],
        "status": "running",
        "inputs": {k: str(v) for k, v in evidence["paths"].items()},
        "input_sha256": evidence["hashes"],
        "font": str(font),
        "font_family": "Arial",
        "font_sha256": sha256(font),
        "display_language": "en",
        "encoding": {
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "crf": 18,
            "preset": "fast",
            "threads": 4,
        },
        "meaning": "离线审查副本；原画面不裁剪不缩放，重新编码；不表示物理同步",
    }
    try:
        expected_times = []
        with av.open(str(evidence["paths"]["video"])) as source:
            if len(source.streams.video) != 1 or source.streams.audio:
                raise ValueError(
                    "本脚本用于单路无声 Chronos 视频，不能静默删除其他音视频轨"
                )
            stream = source.streams.video[0]
            if stream.start_time not in (0, None) or stream.duration is None:
                raise ValueError("需要从零开始且有完整时长的原始录像")
            end = stream.duration * stream.time_base
            height = max(MIN_HEIGHT, stream.height)
            height += height % 2
            width = stream.width + PANEL_WIDTH
            width += width % 2
            panel = TimelinePanel(evidence, float(end), height, font)
            frames = evidence["frames"]
            result["original_image_region"] = [0, 0, stream.width, stream.height]
            with av.open(
                str(output / "review.mp4"), "w", options={"movflags": "+faststart"}
            ) as target:
                encoded = target.add_stream("libx264", rate=stream.average_rate)
                encoded.width, encoded.height = width, height
                encoded.pix_fmt = "yuv420p"
                encoded.time_base = stream.time_base
                encoded.codec_context.time_base = stream.time_base
                encoded.codec_context.thread_count = 4
                encoded.codec_context.max_b_frames = 0
                encoded.options = {"crf": "18", "preset": "fast"}
                target.metadata["comment"] = (
                    f"PZT software timing review; experiment={result['experiment_id']}; not physical synchronization"
                )
                for index, frame in enumerate(source.decode(stream)):
                    if index >= len(frames):
                        raise ValueError("原视频帧数多于关联记录")
                    timestamp = check_frame_time(frame, frames[index])
                    expected_times.append(timestamp)
                    canvas = Image.new("RGB", (width, height), "black")
                    canvas.paste(frame.to_image(), (0, 0))
                    canvas.paste(
                        panel.render(float(timestamp), index, len(frames)),
                        (stream.width, 0),
                    )
                    combined = av.VideoFrame.from_image(canvas)
                    combined.pts, combined.time_base = frame.pts, frame.time_base
                    combined.duration = frame.duration
                    for packet in encoded.encode(combined):
                        target.mux(packet)
                    if index % 300 == 0:
                        print(f"已合成 {index + 1}/{len(frames)} 帧", flush=True)
                if len(expected_times) != len(frames):
                    raise ValueError("原视频缺少关联记录中的帧")
                for packet in encoded.encode():
                    target.mux(packet)
        result["verification"] = verify_video(
            output / "review.mp4",
            expected_times,
            end,
            (width, height),
            output / "preview.png",
        )
        after = {name: sha256(path) for name, path in evidence["paths"].items()}
        if after != evidence["hashes"]:
            raise ValueError("输入文件在合成期间发生变化，请检查原始数据")
        with (output / "frame_mapping.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "experiment_id",
                    "frame_index",
                    "source_seq",
                    "source_video_s",
                    "review_video_s",
                    "received_windows_s",
                ]
            )
            for frame, timestamp in zip(frames, expected_times):
                writer.writerow(
                    [
                        result["experiment_id"],
                        frame["frame_index"],
                        frame["source_seq"],
                        frame["video_s"],
                        float(timestamp),
                        frame["received_windows_s"],
                    ]
                )
        result.update(
            status="completed",
            inputs_unchanged=True,
            output_sha256=sha256(output / "review.mp4"),
        )
    except BaseException as exc:
        result.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["elapsed_s"] = time.perf_counter() - started
        write_json(output / "review_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment", type=Path, help="本次实验目录，内含 recording/video.mp4"
    )
    parser.add_argument(
        "--report-dir", type=Path, required=True, help="同次实验的时间关联报告目录"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="必须不存在的新目录；默认实验目录/review_年月日-时分秒",
    )
    parser.add_argument(
        "--font", type=Path, help="Optional path to an installed Arial TTF"
    )
    args = parser.parse_args()
    font = find_font(args.font)
    if args.output_dir is None:
        stamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d-%H%M%S")
        args.output_dir = args.experiment / f"review_{stamp}"
    result = render_review(
        args.experiment.resolve(),
        args.report_dir.resolve(),
        args.output_dir.resolve(),
        font,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "video": str(args.output_dir / "review.mp4"),
                "verification": result["verification"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
