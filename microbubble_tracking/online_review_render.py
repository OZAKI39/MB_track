"""第 1 轮人工审查视频的绘制与编码；只使用已存画面和记录。"""

import csv
import html
import json
from collections import deque
from fractions import Fraction

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
TEAL, AMBER, INK = "#087F8C", "#D59316", "#233443"
STATE_COLORS = {
    "ACTIVE": TEAL,
    "IDLE": "#E8EDF1",
    "STOP": "#94A1AC",
    "SWITCHING": AMBER,
}
STATUS_NAMES = {
    "pending": "Pending",
    "confirmed_tracks": "Tracks confirmed",
    "rejected_short_track": "Short track: rejected",
    "unresolved_at_end": "Unresolved at end",
}


class Drawing:
    def __init__(self, font):
        self.fonts = {
            n: ImageFont.truetype(str(font), n) for n in (17, 19, 20, 22, 26, 30)
        }

    def text(self, image, xy, text, size=22, color=INK, anchor=None):
        ImageDraw.Draw(image).text(
            xy, text, font=self.fonts[size], fill=color, anchor=anchor
        )


def event_label(event):
    ids = json.loads(event["participants"])
    if event["kind"] == "merge":
        return f"Merge #{ids[0]} into #{ids[1]}"
    if event["kind"] == "split":
        return f"Split #{ids[0]} -> #{ids[1]}"
    return event["kind"].capitalize() + " " + ", ".join(f"#{i}" for i in ids)


def overlay_view(raw, roi, observations, history, time_s, source_frame, drawing):
    """轮廓按当帧状态绘制；轨迹线不跨缺失观测，不补画未来位置。"""
    x, y, w, h = roi
    view = raw.crop((x, y, x + w, y + h)).resize((900, 720), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(view)
    scale = 900 / w
    for tid in list(history):
        while history[tid] and time_s - history[tid][0][0] > 0.5:
            history[tid].popleft()
        if not history[tid]:
            del history[tid]
    for row in observations:
        tid = row["track_id"]
        history.setdefault(tid, deque(maxlen=40)).append(
            (time_s, source_frame, row["cx"], row["cy"])
        )
        tail = list(history[tid])
        for p, q in zip(tail, tail[1:]):
            if q[1] - p[1] == 1:
                draw.line(
                    (
                        (p[2] - x) * scale,
                        (p[3] - y) * scale,
                        (q[2] - x) * scale,
                        (q[3] - y) * scale,
                    ),
                    fill="#B0D9DC",
                    width=2,
                )
        points = np.frombuffer(row["contour_i32"], dtype="<i4").reshape(-1, 2)
        points = (points - (x, y)) * scale
        if (
            points[:, 0].max() < 0
            or points[:, 0].min() >= 900
            or points[:, 1].max() < 0
            or points[:, 1].min() >= 720
        ):
            continue
        points = [tuple(p) for p in points]
        confirmed = row["status_at_frame"] == "confirmed"
        color = TEAL if confirmed else AMBER
        if confirmed:
            draw.line(points + points[:1], fill=color, width=2)
        else:
            for j in range(0, len(points), 8):
                if len(points[j : j + 5]) >= 2:
                    draw.line(points[j : j + 5], fill=color, width=2)
        px, py = (row["cx"] - x) * scale, (row["cy"] - y) * scale
        if 0 <= px < 900 and 0 <= py < 720:
            text = f"{tid}" if confirmed else f"{tid}?"
            draw.text(
                (min(px + 8, 846), max(4, min(py - 21, 692))),
                text,
                font=drawing.fonts[20],
                fill=color,
                stroke_width=2,
                stroke_fill="white",
            )
    return view


def legend(image, xy, drawing):
    # 仅图例半透明；画布和图表均为白底，原图不增加白色滤镜。
    panel = Image.new("RGBA", (476, 38), (255, 255, 255, 205))
    draw = ImageDraw.Draw(panel)
    draw.line((12, 19, 44, 19), fill=TEAL, width=3)
    draw.text((53, 8), "Confirmed track", font=drawing.fonts[19], fill=INK)
    for x in (236, 248, 260):
        draw.line((x, 19, x + 6, 19), fill=AMBER, width=3)
    draw.text((280, 8), "Provisional (?)", font=drawing.fonts[19], fill=INK)
    image.paste(panel, xy, panel)


def timeline_image(run, drawing):
    panel = Image.new("RGB", (900, 183), "white")
    draw = ImageDraw.Draw(panel)
    left, width = 78, 800
    for channel, low, high, state in run.pzt_intervals():
        y = 10 + (int(channel[-1]) - 1) * 31
        draw.rectangle(
            (
                left + low / run.duration * width,
                y,
                left + high / run.duration * width,
                y + 20,
            ),
            fill=STATE_COLORS[state],
        )
    for n in range(4):
        drawing.text(panel, (0, 9 + n * 31), f"PZT{n + 1}", 19)
    for t in [0, 5, 10, 15, 20, 22]:
        x = left + t / run.duration * width
        drawing.text(panel, (x, 136), str(t), 17, anchor="mt")
    drawing.text(panel, (left + width / 2, 159), "Video time (s)", 17, anchor="mt")
    return panel


def render_video(
    run,
    output,
    stem,
    font,
    *,
    title,
    start_s,
    end_s,
    speed=1.0,
    roi=(0, 0, 1280, 1024),
    comparison=None,
    event=None,
):
    """每张源画面均写入输出；慢放只改播放时间，不插帧、不预测缺失轮廓。"""
    drawing = Drawing(font)
    timeline = timeline_image(run, drawing)
    histories = [{}, {}]
    path = output / "videos" / (stem + ".mp4")
    mapping_path = output / "data" / (stem + ".frames.csv")
    first_index = int(np.searchsorted(run.times, start_s))
    stop_index = int(np.searchsorted(run.times, end_s))
    first_time = run.times[first_index]
    nominal_rate = Fraction(round(run.metadata["nominal_fps"] * speed))
    time_base = Fraction(1, 90000)
    preview_frame = min(stop_index - 1, first_index + 60)
    if event is not None:
        preview_frame = run.source[event["reported_frame"]]["video_frame"]
    written = 0
    with (
        av.open(str(run.video)) as source,
        av.open(str(path), "w", options={"movflags": "+faststart"}) as target,
    ):
        stream = target.add_stream("libx264", rate=nominal_rate)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.codec_context.thread_count = 2
        stream.options = {"crf": "20", "preset": "veryfast"}
        with mapping_path.open("x", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "output_frame",
                    "output_time_s",
                    "video_frame",
                    "source_frame",
                    "video_time_s",
                    "run_id",
                    "comparison_run_id",
                    "processed",
                    "comparison_processed",
                    "event_status",
                ]
            )
            for index, decoded in enumerate(source.decode(video=0)):
                if index < first_index:
                    continue
                if index >= stop_index:
                    break
                t = run.times[index]
                if (
                    decoded.pts is None
                    or abs(float(decoded.pts * decoded.time_base) - t)
                    > float(decoded.time_base) + 1e-9
                ):
                    raise ValueError(f"Video PTS mismatch at frame {index}")
                source_frame = int(run.recording[index]["source_seq"])
                raw = decoded.to_image()
                if raw.size != (1280, 1024):
                    raise ValueError(
                        "Review layout expects the archived 1280 x 1024 field"
                    )
                canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
                drawing.text(canvas, (30, 16), title, 30)
                drawing.text(
                    canvas,
                    (1890, 24),
                    f"Video {index}  |  Source {source_frame}  |  {t:.3f} s  |  {speed:g}x",
                    22,
                    anchor="ra",
                )
                observations = run.observations(index)
                if comparison is None:
                    x, y, w, h = roi
                    left = raw.crop((x, y, x + w, y + h)).resize(
                        (900, 720), Image.Resampling.LANCZOS
                    )
                    right = overlay_view(
                        raw, roi, observations, histories[1], t, source_frame, drawing
                    )
                    titles = ("Raw video", "Online records")
                    other_observations = None
                else:
                    left = overlay_view(
                        raw, roi, observations, histories[0], t, source_frame, drawing
                    )
                    other_observations = comparison.observations(index)
                    right = overlay_view(
                        raw,
                        roi,
                        other_observations,
                        histories[1],
                        t,
                        source_frame,
                        drawing,
                    )
                    right_title = (
                        "Every third frame withheld"
                        if index in comparison.frames
                        else "Not processed: no contour or ID"
                    )
                    titles = ("All frames processed", right_title)
                canvas.paste(left, (30, 100))
                canvas.paste(right, (990, 100))
                drawing.text(canvas, (30, 64), titles[0], 26)
                drawing.text(canvas, (990, 64), titles[1], 26)
                legend(canvas, (1395, 58), drawing)
                drawing.text(canvas, (30, 839), "PZT commands", 20)
                for x, state, label in [
                    (212, "ACTIVE", "Active"),
                    (338, "IDLE", "Standby"),
                    (491, "STOP", "Off"),
                    (597, "SWITCHING", "Switch range"),
                ]:
                    ImageDraw.Draw(canvas).rectangle(
                        (x, 845, x + 15, 860), fill=STATE_COLORS[state]
                    )
                    drawing.text(canvas, (x + 23, 839), label, 19)
                canvas.paste(timeline, (30, 872))
                cursor_x = 30 + 78 + t / run.duration * 800
                ImageDraw.Draw(canvas).line(
                    (cursor_x, 880, cursor_x, 1004), fill=INK, width=2
                )
                provisional = sum(
                    r["status_at_frame"] == "tentative" for r in observations
                )
                count_label = "Full input" if comparison is not None else "Whole field"
                drawing.text(
                    canvas,
                    (990, 839),
                    f"{count_label}: {len(observations)} regions   |   Provisional: {provisional}",
                    22,
                )
                event_status = ""
                if comparison is not None:
                    text = (
                        f"Reduced input: {len(other_observations)} regions"
                        if index in comparison.frames
                        else "Reduced input: frame withheld"
                    )
                    drawing.text(canvas, (990, 888), text, 22)
                    drawing.text(
                        canvas,
                        (990, 927),
                        "IDs are independent between the two runs",
                        20,
                    )
                    drawing.text(
                        canvas, (990, 966), "Missing frames are not filled in", 20
                    )
                elif event is not None:
                    event_status = run.event_status(event["event_id"], source_frame)
                    if event_status is None:
                        drawing.text(canvas, (990, 888), "No event reported yet", 22)
                    else:
                        drawing.text(canvas, (990, 888), event_label(event), 22)
                        drawing.text(
                            canvas,
                            (990, 927),
                            STATUS_NAMES[event_status],
                            22,
                            TEAL if event_status == "confirmed_tracks" else AMBER,
                        )
                        anchor = run.source[event["anchor_frame"]]["video_frame"]
                        reported = run.source[event["reported_frame"]]["video_frame"]
                        drawing.text(
                            canvas,
                            (990, 966),
                            f"Anchor frame: {anchor}   |   Reported frame: {reported}",
                            20,
                        )
                        decision = event["decided_frame"]
                        if decision is not None and decision <= source_frame:
                            drawing.text(
                                canvas,
                                (990, 1003),
                                f"Decision frame: {run.source[decision]['video_frame']}",
                                20,
                            )
                else:
                    recent = []
                    for e in run.events.values():
                        if e["kind"] not in ("merge", "split", "lost", "exit"):
                            continue
                        if (
                            0
                            <= t - run.source[e["reported_frame"]]["video_time_s"]
                            <= 0.8
                        ):
                            recent.append(e)
                    for n, e in enumerate(recent[-3:]):
                        status = run.event_status(e["event_id"], source_frame)
                        drawing.text(
                            canvas,
                            (990, 888 + n * 39),
                            event_label(e) + "  |  " + STATUS_NAMES[status],
                            20,
                        )
                    if not recent:
                        drawing.text(
                            canvas,
                            (990, 888),
                            "No recent merge, split or loss report",
                            20,
                        )
                if index == preview_frame:
                    canvas.save(output / "previews" / (stem + ".png"))
                frame = av.VideoFrame.from_image(canvas)
                output_pts = round((t - first_time) / speed / float(time_base))
                frame.pts = output_pts
                frame.time_base = time_base
                for packet in stream.encode(frame):
                    target.mux(packet)
                writer.writerow(
                    [
                        written,
                        output_pts * float(time_base),
                        index,
                        source_frame,
                        t,
                        run.metadata["run_id"],
                        comparison.metadata["run_id"] if comparison else "",
                        index in run.frames,
                        index in comparison.frames if comparison else "",
                        event_status or "",
                    ]
                )
                written += 1
                if written % 300 == 0:
                    print(f"{stem}: {written} frames", flush=True)
            for packet in stream.encode():
                target.mux(packet)
    if written != stop_index - first_index:
        raise ValueError("Video ended before the selected review interval")
    count, last_pts = 0, None
    with av.open(str(path)) as verify, mapping_path.open() as audit:
        expected = csv.DictReader(audit)
        for frame in verify.decode(video=0):
            if frame.width != WIDTH or frame.height != HEIGHT or frame.pts is None:
                raise ValueError("Invalid review video frame")
            if last_pts is not None and frame.pts <= last_pts:
                raise ValueError("Review video timestamps do not increase")
            row = next(expected, None)
            if (
                row is None
                or abs(float(frame.pts * frame.time_base) - float(row["output_time_s"]))
                > float(time_base) + 1e-9
            ):
                raise ValueError(
                    "Review timing differs from the exported frame mapping"
                )
            last_pts = frame.pts
            count += 1
    if count != written:
        raise ValueError("Encoded video lost frames")
    report = dict(
        file=f"videos/{stem}.mp4",
        title=title,
        frames=written,
        source_video_from=first_index,
        source_video_to=stop_index - 1,
        video_from_s=float(first_time),
        video_to_s=float(run.times[stop_index - 1]),
        playback_speed=speed,
        dimensions=[WIDTH, HEIGHT],
        event_id=event["event_id"] if event else None,
        preview=f"previews/{stem}.png",
        decoded_validation="passed",
    )
    print(f"Verified {stem}: {written} frames", flush=True)
    return report


def write_gallery(output, videos):
    """本地 HTML 索引：浏览器播放、暂停和拖动，不部署服务器。"""
    guide = output / "REVIEW_GUIDE.md"
    if not guide.exists():
        links = "\n".join(f"- [{v['title']}]({v['file']})" for v in videos)
        guide.write_text(
            "# 第 1 轮人工审核\n\n先打开 index.html，再按下面顺序检查视频。\n\n"
            + links
            + "\n\n左侧原图、右侧记录：检查轮廓是否贴合深色区域，编号是否被错换。"
            "青色实线表示当时已经确认的轨迹；黄色虚线和问号表示还在观察。"
            "算法确认不等于人工认定正确。\n\n"
            "跳帧对照中的左右编号来自独立运行，不能按数值逐个对应。"
            "未处理的画面不补轮廓。慢放只改变播放速度，原视频帧号不变。\n\n"
            "合并、分离、跟丢视频均为待审候选。请对照原图判断，"
            "尤其不能把跟丢直接解释为微泡真的消失。事件编号和选择范围见 manifest.json。\n\n"
            "图 1 看真实画面间隔和源编号缺口；图 2 每行是一个编号，白色表示没有观测；"
            "图 3 对比活跃目标数与已写盘观测数，不是 MB 内存测量；"
            "图 4 对应命令发送和读回范围，不表示曝光或声波实际生效时间。\n\n"
            "发现问题请在 review_notes.csv 填写实验、Video 帧号、编号和问题。"
            "技术解码检查不能代替人工审核。\n",
            encoding="utf-8",
        )
    notes = output / "review_notes.csv"
    if not notes.exists():
        notes.write_text(
            "experiment,video_frame,track_ids,category,observation,expected_correction,reviewer\n",
            encoding="utf-8",
        )
    sections = []
    for video in videos:
        path, title, preview = (
            html.escape(str(video[k])) for k in ("file", "title", "preview")
        )
        sections.append(
            f'<section><h2>{title}</h2><video controls preload="none" poster="{preview}" src="{path}"></video>'
            f'<p><a href="{path}">Open video</a> &nbsp; {video["playback_speed"]:g}x playback</p></section>'
        )
    charts = []
    for path in sorted((output / "charts").glob("*.png")):
        name = html.escape(path.stem.replace("_", " "))
        charts.append(
            f'<section><h2>{name}</h2><a href="charts/{path.name}"><img src="charts/{path.name}"></a></section>'
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Round 1 review</title>
<style>@font-face{font-family:Arial;src:url('assets/arial.ttf')}*{box-sizing:border-box}
body{font-family:Arial;background:white;color:#233443;max-width:1280px;margin:32px auto;padding:0 24px}
h1{font-size:30px}h2{font-size:22px;font-weight:normal}p{line-height:1.5}video,img{width:100%;height:auto}
section{margin:38px 0;border-top:1px solid #dce3e8;padding-top:16px}a{color:#087F8C}
</style><h1>Round 1: frame-by-frame tracking</h1>
<p>Solid teal: confirmed track. Dashed amber and ?: provisional. Confirmation is algorithmic, not human review.</p>
<p>PZT bands show command-to-video mapping, not measured acoustic onset. <a href="REVIEW_GUIDE.md">Review guide</a></p>
"""
    (output / "index.html").write_text(
        page + "".join(sections + charts) + "</html>\n", encoding="utf-8"
    )
