"""Export data-based stimulus–response visuals without changing scientific inputs.

Run from the project root with ``.venv/bin/python -m
customer_workspace.stimulus_response_ppt``. The existing response detector is
called in memory. A presentation fraction uses the same resting population in
its numerator and denominator; the original module's broader numerator is also
saved for audit. No hardware is accessed and no original results are rewritten.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from PIL import Image, ImageDraw

from customer_workspace.analysis_ppt import (
    BOLD_FONT, FONT, PROJECT, ROOT, Design, VideoSource, digest, dot,
    encode_movie, line, paste_view, translucent_panel, verify_movie, write_csv,
)
from customer_workspace.microbubble_tracking.analysis import analyze, pzt_analysis


SIZE = (1920, 1080)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
BLACK = "#202124"
GREY = "#73777C"
ACTION_LABELS = (
    "PZT1 on", "PZT1 → PZT2", "PZT2 → PZT3", "PZT3 → PZT4",
    "PZT4 → PZT1 + PZT3", "All to standby", "Output off",
)


class ResponseEvidence:
    """Validate the saved measurements and apply the existing response rules."""

    def __init__(self, experiment="A"):
        self.experiment = experiment
        self.action_labels = ACTION_LABELS if experiment == "A" else (
            "PZT1 on", "PZT1 → PZT3", "PZT3 → PZT1", "All to standby", "Output off",
        )
        self.result = PROJECT / f"results/{experiment}_offline"
        self.analysis = self.result / "analysis"
        self.video = PROJECT / f"data/raw/exp{experiment}.mp4"
        self.pzt_dir = PROJECT / f"data/pzt/exp{experiment}"
        paths = [self.video, PROJECT / "data/pzt/manifest.json"]
        paths += [p for p in self.result.rglob("*") if p.is_file()]
        paths += [p for p in self.pzt_dir.rglob("*") if p.is_file()]
        self.preserved = {str(p): digest(p) for p in paths}
        self.tables = pzt_analysis.Tables(str(self.analysis))
        report = self.tables.report
        self.data = analyze.Data(
            str(self.result), report["parameters"], video=str(self.video)
        )
        if self.data.input_sha1 != report["inputs"]["sha1"]:
            raise ValueError("The saved analysis no longer matches its reviewed inputs")
        self.fps = self.data.fps
        self.n_frames = self.data.n_frames
        self.validate_measurements()
        self.record = pzt_analysis.PztRecord(str(self.pzt_dir))
        manifest = self.record.manifest_entry(f"exp{experiment}")
        if manifest["video_sha256"] != self.preserved[str(self.video)]:
            raise ValueError("The video does not match the archived PZT experiment")
        for item in manifest["files"]:
            path = self.pzt_dir.parent / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError(f"An archived PZT input changed: {path}")
        if len(self.record.video_s) != self.n_frames:
            raise ValueError("The PZT video-frame record has a different length")
        self.parameters = dict(pzt_analysis.DEFAULTS)
        self.commands, self.actions = pzt_analysis.build_actions(
            self.record, self.parameters
        )
        if len(self.actions) - 1 != len(self.action_labels):
            raise ValueError("The action count differs from the selected experiment")
        rate, responded, total = pzt_analysis.placebo_rate(
            self.tables, self.actions, self.n_frames, self.fps, self.parameters
        )
        self.control = dict(fraction=rate, responded=responded, resting_samples=total)
        events = defaultdict(list)
        for event in self.tables.events:
            for object_id in event.get("object_ids", "").split():
                events[int(object_id)].append(
                    dict(frame=int(event["frame"]), type=event["type"])
                )
        self.responses, self.summary = [], []
        for index, action in enumerate(self.actions[1:], 1):
            windows = pzt_analysis.windows(
                self.actions, index, self.n_frames, self.fps, self.parameters
            )
            rows = []
            for object_id in sorted(self.tables.obj):
                row = pzt_analysis.object_response(
                    self.tables, object_id, self.actions, index, windows,
                    self.fps, self.parameters, events,
                )
                if row is not None:
                    rows.append(row)
            self.responses.extend(rows)
            resting = [r for r in rows if r["at_rest_before"]]
            responding = [r for r in resting if r["responded_within_window"]]
            original = [r for r in rows if r["responded_within_window"]]
            excluded = [r["object_id"] for r in original if not r["at_rest_before"]]
            onsets = Counter(r["first_change_frame"] for r in responding)
            self.summary.append(dict(
                action_no=index, action=self.action_labels[index - 1],
                original_action=action["label"], planned_s=action["planned_s"],
                software_reference_s=action["switch_video_s"],
                resting_objects=len(resting), responding_resting_objects=len(responding),
                responding_fraction=len(responding) / len(resting) if resting else None,
                original_module_responding_count=len(original),
                original_numerator_outside_resting_population=excluded,
                responding_object_ids=[r["object_id"] for r in responding],
                resting_object_ids=[r["object_id"] for r in resting],
                onset_frames=dict(sorted(onsets.items())),
            ))

    def validate_measurements(self):
        """Recompute trusted positions/speeds in memory, then check saved rounding."""
        data = self.data
        static = analyze.static_structure(
            data.bg, data.P["wall_darkness"], data.P["wall_dilate_px"]
        )
        edge, border = analyze.per_row_geometry(data, static)
        segments, _, _, _, _ = analyze.build_segments(data, border)
        motion, shape = analyze.motion_and_shape(data, segments, edge, border)
        if len(segments) != self.tables.report["trusted_segments"]:
            raise ValueError("Trusted segment count changed")
        for saved, computed, columns in (
            (self.tables.motion, motion, ("x_px", "y_px", "speed_px_s")),
            (self.tables.shape, shape, ("area_px2", "aspect_ratio")),
        ):
            expected = {(int(r["frame"]), int(r["id"])): r for r in computed}
            if len(saved) != len(expected):
                raise ValueError("Saved and recomputed analysis lengths differ")
            for row in saved:
                current = expected[(int(row["frame"]), int(row["id"]))]
                if int(row["segment"]) != current["segment"]:
                    raise ValueError("A trusted segment changed")
                for key in columns:
                    if not np.isclose(pzt_analysis.fnum(row[key]), current[key], atol=0.00051,
                                      rtol=0, equal_nan=True):
                        raise ValueError(f"Saved {key} does not match current analysis")

    def example(self):
        """Use a complete, unflagged real response with a visible rest baseline."""
        object_id = 10 if self.experiment == "A" else 8
        row = next(r for r in self.responses
                   if r["action_no"] == 1 and r["object_id"] == object_id)
        if not (row["at_rest_before"] and row["responded_within_window"]
                and row["quality"] == "good"):
            raise ValueError("The selected example is no longer a reliable response")
        obj = self.tables.obj[object_id]
        action = self.actions[1]
        windows = pzt_analysis.windows(
            self.actions, 1, self.n_frames, self.fps, self.parameters
        )
        departure = pzt_analysis.departure(
            obj, windows, action, self.fps, self.parameters
        )
        pre = pzt_analysis.in_win(
            obj["frame"], (windows["switch"] - round(0.5*self.fps), windows["switch"]-1)
        )
        segment = obj["segment"][np.nonzero(pre)[0][-1]]
        pre &= obj["segment"] == segment
        centre = np.array([np.median(obj["x"][pre]), np.median(obj["y"][pre])])
        distance = np.hypot(obj["x"] - centre[0], obj["y"] - centre[1])
        frames = (np.arange(150, 241) if self.experiment == "A" else
                  np.arange(windows["switch"] - 35, windows["switch"] + 56))
        indices = np.searchsorted(obj["frame"], frames)
        if not np.array_equal(obj["frame"][indices], frames):
            raise ValueError("The example has missing measured frames")
        if not np.all(obj["segment"][indices] == segment):
            raise ValueError("The example crosses a trusted segment boundary")
        post = pzt_analysis.in_win(
            obj["frame"], (departure["frame"], departure["frame"] + round(0.1*self.fps)-1)
        )
        speed_threshold = max(self.parameters["response_min_speed_px_s"],
                              self.parameters["response_speed_ratio"] * np.median(obj["speed"][pre]))
        metadata = dict(
            object_id=object_id, segment=int(segment), departure=departure,
            rest_centroid_px=centre.tolist(), rest_max_distance_px=float(distance[pre].max()),
            rest_median_speed_px_s=float(np.median(obj["speed"][pre])),
            response_median_speed_px_s=float(np.median(obj["speed"][post])),
            speed_threshold_px_s=float(speed_threshold),
            software_command_reference_s=action["switch_video_s"],
            first_frame=int(frames[0]), last_frame=int(frames[-1]),
            playback_speed=0.25, hold_final_s=1.5,
            roi_xywh=([736, 446, 160, 160] if self.experiment == "A" else
                      [int(np.clip(centre[0]-80, 0, self.data.W-160)),
                       int(np.clip(centre[1]-80, 0, self.data.H-160)), 160, 160]),
        )
        trace = [dict(
            frame=int(frame), video_time_s=frame/self.fps,
            relative_time_s=frame/self.fps-action["switch_video_s"],
            object_id=object_id, track_id=int(obj["id"][index]),
            segment=int(segment), x_px=float(obj["x"][index]), y_px=float(obj["y"][index]),
            displacement_px=float(distance[index]), speed_px_s=float(obj["speed"][index]),
        ) for frame, index in zip(frames, indices)]
        return metadata, trace


def style_axes(axes, size=13):
    """Apply explicit Arial and conventional scientific axes, with no grid."""
    for spine in axes.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(0.9)
    axes.tick_params(which="both", direction="in", top=True, right=True,
                     colors=BLACK, width=0.9, labelsize=size)
    axes.tick_params(which="major", length=5, pad=8)
    axes.tick_params(which="minor", length=2.5)
    axes.xaxis.set_minor_locator(AutoMinorLocator(2))
    axes.yaxis.set_minor_locator(AutoMinorLocator(2))
    for axis in (axes.xaxis, axes.yaxis):
        axis.offsetText.set_fontproperties(FontProperties(fname=str(FONT), size=size))
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            for text in (tick.label1, tick.label2):
                text.set_fontproperties(FontProperties(fname=str(FONT), size=size))


def audit_figure(figure, design):
    """Reject font substitution and record visible labels for the export audit."""
    for text in figure.findobj(Text):
        if not text.get_text():
            continue
        if text.get_fontproperties().get_file() not in (str(FONT), str(BOLD_FONT)):
            raise ValueError(f"A figure label does not use actual Arial: {text.get_text()}")
        if any("\u4e00" <= c <= "\u9fff" for c in text.get_text()):
            raise ValueError("Non-English figure label")
        design.labels.add(text.get_text())


def response_chart(evidence, design, output):
    """Export one static result chart as PNG, outlined SVG and embedded-font PDF."""
    figure = Figure(figsize=(14.4, 8.1), dpi=250, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((.10, .22, .87, .61), facecolor="white")
    regular = FontProperties(fname=str(FONT), size=16)
    bold = FontProperties(fname=str(BOLD_FONT), size=16)
    title = FontProperties(fname=str(BOLD_FONT), size=24)
    figure.text(.10, .94, f"Stimulus–Response Analysis (Exp. {evidence.experiment})", fontproperties=title)
    figure.text(.10, .885, "Resting objects responding within 0.5 s", fontproperties=regular)
    colors = ((BLUE, ORANGE, GREEN, PURPLE, BLUE, GREY, BLACK)
              if evidence.experiment == "A" else (BLUE, GREEN, BLUE, GREY, BLACK))
    values = [100*r["responding_fraction"] if r["responding_fraction"] is not None else 0
              for r in evidence.summary]
    for index, (row, value, color) in enumerate(zip(evidence.summary, values, colors)):
        if evidence.experiment == "A" and index == 4:
            axes.add_patch(Rectangle((index-.32, 0), .32, value, facecolor=BLUE))
            axes.add_patch(Rectangle((index, 0), .32, value, facecolor=GREEN))
        else:
            axes.bar(index, value, width=.64, color=color, zorder=2)
        if value == 0:
            axes.plot([index-.32, index+.32], [0, 0], color=color, lw=2, clip_on=False)
        axes.text(index, value+3 if value else 7, (f"{row['responding_resting_objects']}/{row['resting_objects']}"
                   if row["resting_objects"] else "N/A"),
                  ha="center", va="bottom", fontproperties=bold)
    control = 100*evidence.control["fraction"]
    axes.axhline(control, color=GREY, lw=1.3, linestyle=(0, (5, 3)), zorder=3)
    axes.text(.98, .96,
              f"No-switch control: {control:.1f}% ({evidence.control['responded']}/{evidence.control['resting_samples']})",
              transform=axes.transAxes, ha="right", va="top", fontproperties=regular, color=GREY)
    labels = ("PZT1\non", "PZT1\n→ PZT2", "PZT2\n→ PZT3", "PZT3\n→ PZT4",
              "PZT4 →\nPZT1 + PZT3", "All to\nstandby", "Output\noff")
    if evidence.experiment == "B":
        labels = ("PZT1\non", "PZT1\n→ PZT3", "PZT3\n→ PZT1", "All to\nstandby", "Output\noff")
    axes.set_xticks(range(len(labels)), labels)
    axes.set_xlim(-.7, len(labels)-.3)
    axes.set_ylim(0, 110)
    axes.set_yticks(np.arange(0, 101, 20))
    axes.set_ylabel("Responding fraction (%)", fontproperties=regular, labelpad=13)
    style_axes(axes, size=14)
    axes.xaxis.set_minor_locator(AutoMinorLocator(1))
    axes.tick_params(axis="x", which="both", top=False, bottom=False, pad=14)
    figure.text(.10, .063, "Counts: responding / resting objects", fontproperties=regular)
    figure.text(.97, .063, "Single experiment; repeated trials needed.",
                ha="right", fontproperties=FontProperties(fname=str(FONT), size=13), color=GREY)
    canvas.draw()
    audit_figure(figure, design)
    with rc_context({"svg.fonttype": "path", "pdf.fonttype": 42}):
        for extension in ("png", "svg", "pdf"):
            figure.savefig(output / f"02_response_summary.{extension}", facecolor="white")


def response_plot(metadata, trace, design):
    """Prepare the real displacement curve, rest band and detection threshold."""
    figure = Figure(figsize=(1080/150, 650/150), dpi=150, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((.14, .16, .82, .74), facecolor="white")
    regular = FontProperties(fname=str(FONT), size=13)
    times = np.array([r["relative_time_s"] for r in trace])
    distances = np.array([r["displacement_px"] for r in trace])
    axes.axhspan(0, metadata["rest_max_distance_px"], color="#E4E8EB", lw=0)
    threshold = metadata["departure"]["threshold_px"]
    threshold_line = axes.axhline(threshold, color=PURPLE, linestyle=(0, (3, 3)), lw=1.3)
    axes.axvline(0, color=GREY, linestyle=(0, (4, 3)), lw=1.2)
    axes.text(.015, 1.045, "PZT1 command reference: 0 s", transform=axes.transAxes,
              fontproperties=regular, color=GREY)
    curve, = axes.plot([], [], color=BLUE, linewidth=2)
    point, = axes.plot([], [], color=BLUE, marker="o", markersize=6)
    onset, = axes.plot([], [], color=ORANGE, marker="o", markersize=7)
    axes.legend(
        [curve, Patch(facecolor="#E4E8EB"), threshold_line],
        ["Displacement", "Rest range", f"Threshold: {threshold:.2f} px"],
        loc="upper left", frameon=False, prop=FontProperties(fname=str(FONT), size=11),
    )
    axes.set_xlim(-.6, .95)
    upper = max(11, np.ceil(distances.max() * 1.12 / 2) * 2)
    axes.set_ylim(0, upper)
    axes.set_xticks(np.arange(-.5, 1, .25))
    axes.set_yticks(np.arange(0, upper, max(2, round(upper/6))))
    axes.set_xlabel("Time from command reference (s)", fontproperties=regular, labelpad=10)
    axes.set_ylabel("Displacement from rest (px)", fontproperties=regular, labelpad=9)
    style_axes(axes, size=12)
    for artist in (curve, point, onset):
        artist.set_animated(True)
    canvas.draw()
    audit_figure(figure, design)
    background = canvas.copy_from_bbox(figure.bbox)
    return canvas, axes, background, (curve, point, onset), times, distances


def response_frames(evidence, metadata, trace, design):
    """Show each real source frame at quarter speed, with its measured curve."""
    canvas, axes, background, artists, times, distances = response_plot(metadata, trace, design)
    curve, point, onset = artists
    source = VideoSource(evidence.video)
    start_frame = metadata["departure"]["frame"]
    start_index = next(i for i, row in enumerate(trace) if row["frame"] == start_frame)
    try:
        for index, row in enumerate(trace):
            image = Image.new("RGB", SIZE, "white")
            design.text(image, (70, 45), f"Stimulus–Response Analysis (Exp. {evidence.experiment})", 48, BLACK, bold=True)
            design.text(image, (72, 119), f"Object {metadata['object_id']}  |  PZT1 on  |  0.25× speed", 29, GREY)
            transform, _ = paste_view(image, source.get(row["frame"]), metadata["roi_xywh"],
                                      (70, 220, 650, 650))
            contour = evidence.data.contour[(row["frame"], row["track_id"])]
            line(image, transform(contour), BLUE, 3, closed=True)
            centre = transform([[row["x_px"], row["y_px"]]])[0]
            dot(image, centre, ORANGE, radius=5)
            translucent_panel(image, (85, 235, 330, 322))
            ImageDraw.Draw(image).line((103, 258, 129, 258), fill=BLUE, width=3)
            design.text(image, (142, 241), "Region outline", 25, BLACK)
            dot(image, (116, 297), ORANGE, radius=5)
            design.text(image, (142, 280), "Centroid", 25, BLACK)
            design.text(image, (70, 894), f"Video: {row['video_time_s']:.3f} s", 28, BLACK)
            design.text(image, (720, 894), f"Frame {row['frame']}", 28, GREY, anchor="ra")
            canvas.restore_region(background)
            curve.set_data(times[:index+1], distances[:index+1])
            point.set_data(times[index:index+1], distances[index:index+1])
            if row["frame"] >= start_frame:
                onset.set_data(times[start_index:start_index+1], distances[start_index:start_index+1])
            else:
                onset.set_data([], [])
            for artist in artists:
                axes.draw_artist(artist)
            image.paste(Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert("RGB"), (770, 200))
            design.text(image, (865, 860), "Displacement > 1 px, > 4× jitter, > 1.5× rest maximum", 24, BLACK)
            design.text(image, (865, 902), "Held 3 frames; speed ≥ 15 px/s and ≥ 3× rest speed", 24, BLACK)
            if row["frame"] >= start_frame:
                dot(image, (881, 963), ORANGE, radius=6)
                design.text(image, (900, 946), "Detected onset", 25, ORANGE)
            design.text(image, (70, 1023), "Software time alignment; not a physical latency measurement.", 24, GREY)
            mapping = dict(source_frame=row["frame"], source_video_s=row["video_time_s"],
                           object_id=metadata["object_id"], displacement_px=row["displacement_px"])
            yield image, 2, mapping
        yield image, 45, mapping
    finally:
        source.close()


def write_report(output, evidence, metadata):
    """Keep detailed provenance and interpretation outside the presentation."""
    report = {
        "created": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
        "experiment": f"exp{evidence.experiment}", "parameters": evidence.parameters,
        "input_sha256": evidence.preserved, "control": evidence.control,
        "summary": evidence.summary, "example": metadata,
        "font_files": {str(p): digest(p) for p in (FONT, BOLD_FONT)},
        "code_sha256": {str(p): digest(p) for p in (
            Path(__file__), ROOT / "customer_workspace/analysis_ppt.py",
            Path(pzt_analysis.__file__), Path(analyze.__file__),
        )},
        "population_note": "The figure numerator is the intersection of responded_within_window and at_rest_before. "
                           "Original field_response sums all responded_within_window, including objects outside its resting "
                           "denominator. Excluded IDs are recorded per action. The detector is unchanged.",
        "control_note": "Existing placebo_rate is reused unchanged: 0.5 s resting baseline and made-up switch times "
                        "inside constant PZT states. Action at_rest_before uses a 1 s baseline; departure uses 0.5 s. "
                        "Control is a descriptive reference, not an independent randomized control group.",
        "timing_note": "The current pzt_analysis software time mapping is preserved for reproducibility. "
                       "Its k_lo/earliest-frame terminology does not establish actual exposure, analog output onset "
                       "or a calibrated physical delay. The plotted 0 s is labelled command reference only.",
        "limitations": [
            "A single experiment; objects and repeated control samples are not independent repeated trials.",
            "Only trusted motion rows are used. Existing per-action quality flags remain in responses.csv.",
            "Displacement is measured from the median centroid in the last 0.5 s before the switch frame.",
            "Threshold = max(1 px, 4 x robust rest jitter, 1.5 x maximum rest displacement).",
            "Three consecutive frames above threshold; the next 0.1 s median speed must be at least max(15 px/s, "
            "3 x resting speed). First qualifying departure must be within 0.5 s of the software reference.",
            "Speed uses centred frame differences; its rise can precede the visible displacement. No physical delay is inferred.",
            "Zero observed responses does not establish equivalence to the descriptive control rate.",
        ],
    }
    (output / "provenance.json").write_text(
        json.dumps(pzt_analysis.json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = """# 刺激—响应展示素材

- `01_response_example.mp4`：实验 A 的目标 10，原始视频第 150–240 帧，四分之一速度。左侧是真实裁剪画面，右侧为同帧质心相对静止位置的位移。末尾停留 1.5 秒。没有生成中间帧，没有平滑测量值。
- `02_response_summary.png`：七次动作的响应比例；SVG/PDF 是同一张图的矢量版本。白色背景，文字使用实际 Arial；SVG 文字转成路径，避免打开时换字体，PDF 嵌入字体。
- `summary.csv`、`responses.csv` 和 `example_trace.csv`：图中数字、逐目标判定和动画曲线的来源。

当前代码的原始汇总在第二次切换中统计了 12 个响应目标，但其中目标 11 不属于同一行的 18 个静止目标。因此展示图采用同一批静止目标计算的 11/18。原始 12 的计数和被排除的目标编号也已保留，未修改分析算法或原始结果。

其余依次为 17/19、3/11、10/13、4/13、0/12、0/14。未切换时的对照为 2/57，即 3.5%。对照使用原程序的判定方法，但其静止筛选窗口为 0.5 秒，动作汇总的筛选窗口为 1 秒，因此这里只作描述性比较；不是独立的对照实验，也未作显著性检验。

截图只列出了部分判定条件。实际代码还要求位移超过静止时最大偏移的 1.5 倍，且速度超过静止速度的 3 倍。图与动画沿用这些规则，完整参数见 provenance.json。

图中的时间零点沿用当前软件对命令与视频的关联方法。它不是已校准的曝光或超声生效时刻，不能据此宣称测得了真实物理延迟。没有响应不等于证明与对照完全相同。只有一段实验，仍需重复实验才能推广结论。

所有科学输入只读，导出前后检查文件哈希。验证报告保存在 verification.json。未实际打开 PowerPoint 播放。
"""
    if evidence.experiment == "B":
        counts = ", ".join(f"{r['action']}: {r['responding_resting_objects']}/{r['resting_objects']}"
                           for r in evidence.summary)
        text = f"""# Stimulus–Response Analysis (Exp. B)

`01_response_example.mp4` shows object {metadata['object_id']} in actual source frames
{metadata['first_frame']}–{metadata['last_frame']}, at quarter speed. No interpolated frames or smoothed measurements.
`02_response_summary.png` (also PDF/SVG) uses the same resting population in the numerator and denominator.

{counts}.

The original broader counts and excluded IDs remain in summary.csv. The existing response rules,
quality flags and descriptive no-switch control are unchanged. The command reference is a software
video mapping, not calibrated exposure or ultrasound onset. A single experiment cannot establish
causality or physical response latency. Full input hashes and limitations: provenance.json.
All visible text uses actual Arial. PowerPoint playback has not been tested.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment", choices=("A", "B"), default="A")
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir or ROOT / f"customer_workspace/validation/stimulus_response_slides_{stamp}"
    if output.exists():
        raise FileExistsError(f"Preserve previous exports: {output}")
    design = Design()
    evidence = ResponseEvidence(args.experiment)
    metadata, trace = evidence.example()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "responses.csv", evidence.responses)
    write_csv(output / "summary.csv", evidence.summary)
    write_csv(output / "example_trace.csv", trace)
    response_chart(evidence, design, output)
    print("Verified data; result chart exported", flush=True)
    movie = output / "01_response_example.mp4"
    frame_count = encode_movie(movie, response_frames(evidence, metadata, trace, design))
    verification = verify_movie(movie, frame_count, size=SIZE)
    changed = [p for p, value in evidence.preserved.items() if digest(p) != value]
    if changed:
        raise ValueError(f"Scientific inputs changed during rendering: {changed}")
    verification.update(scientific_inputs_unchanged=True, saved_measurements_reproduced=True,
                        actual_arial_verified=True, ppt_playback_tested=False,
                        original_algorithms_unchanged=True)
    write_report(output, evidence, metadata)
    (output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (output / "visible_labels.json").write_text(json.dumps(sorted(design.labels), indent=2), encoding="utf-8")
    with zipfile.ZipFile(output / "slide8_assets.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix != ".zip":
                archive.write(path, path.name)
    print(output, flush=True)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
