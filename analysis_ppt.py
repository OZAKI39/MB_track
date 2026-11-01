"""Create five data-based analysis animations for PowerPoint.

Run from the project root with ``.venv/bin/python -m
customer_workspace.analysis_ppt``. Existing analysis functions are called only
in memory; original videos, annotations and result tables are never rewritten.
Arial is loaded from explicit font files. Slow motion repeats real frames;
neither image interpolation nor smoothing of scientific values is used.
Use ``--step 1-3`` to export the first three animations, or a single step number.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import zipfile

import av
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from customer_workspace.microbubble_tracking.analysis import analyze
from customer_workspace.pzt_video_review import status_at


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "customer_workspace/microbubble_tracking"
FONT = Path("/mnt/c/Windows/Fonts/arial.ttf")
BOLD_FONT = Path("/mnt/c/Windows/Fonts/arialbd.ttf")
WIDTH, HEIGHT, OUTPUT_FPS = 1920, 1080, 30
COMPACT_SIZE = (1280, 960)
PANEL_ALPHA = 153
INK = "#172b40"
MUTED = "#576a7c"
GRID = "#dfe6ec"
PALE = "#f3f6f9"
BLUE = "#0078b8"
TEAL = "#009b87"
ORANGE = "#db7e00"
RED = "#cc3552"
PURPLE = "#7c50ad"
WHITE = "#ffffff"


def digest(path, algorithm="sha256"):
    """Hash a file without loading it all into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture_validation(data, motion, shape, report, video):
    """Observe the original raw-video check and retain its actual sample data.

    A temporary trace reads this function's local values just before each
    displacement comparison is appended. No matching algorithm is copied or
    changed. The previous tracing callback is restored even on failure.
    """
    function = analyze.video_motion_check
    lines, first_line = inspect.getsourcelines(function)
    append_line = first_line + next(
        i for i, line in enumerate(lines) if "diffs.append(" in line
    )
    samples = []

    def trace(frame, event, argument):
        if frame.f_code is not function.__code__:
            return None
        if event == "line" and frame.f_lineno == append_line:
            local = frame.f_locals
            row, later = local["r"], local["r2"]
            sample = {
                "sample": len(samples) + 1,
                "frame": row["frame"],
                "later_frame": later["frame"],
                "id": row["id"],
                "segment": row["segment"],
                "x_px": float(row["x_px"]),
                "y_px": float(row["y_px"]),
            }
            for key in ("X", "Y", "h", "x0", "y0", "x1", "y1"):
                sample[key] = int(local[key])
            for key in ("cdx", "cdy", "tdx", "tdy", "score"):
                sample[key] = float(local[key])
            sample["difference_px"] = float(np.hypot(
                sample["tdx"] - sample["cdx"],
                sample["tdy"] - sample["cdy"],
            ))
            samples.append(sample)
        return trace

    previous_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        summary = function(
            data, str(video), motion, shape, report["main_objects"]
        )
    finally:
        sys.settrace(previous_trace)
    for key, expected in report["video_motion_check"].items():
        if not np.isclose(summary[key], expected, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Video check changed: {key}: {summary[key]}")
    if len(samples) != summary["samples"]:
        raise ValueError("Missing captured template-matching samples")
    return summary, samples


class Evidence:
    """Read and verify the selected reviewed experiment results used in every animation."""

    def __init__(self, experiment="A"):
        self.experiment = experiment
        self.video = PROJECT / f"data/raw/exp{experiment}.mp4"
        self.result = PROJECT / f"results/{experiment}_offline"
        self.analysis = self.result / "analysis"
        self.report = json.loads((self.analysis / "report.json").read_text())
        self.data = analyze.Data(
            str(self.result), self.report["parameters"], video=str(self.video)
        )
        for name, expected in self.report["inputs"]["sha1"].items():
            if self.data.input_sha1.get(name) != expected:
                raise ValueError(f"Analysis input no longer matches: {name}")
        static = analyze.static_structure(
            self.data.bg,
            self.data.P["wall_darkness"],
            self.data.P["wall_dilate_px"],
        )
        edge, border = analyze.per_row_geometry(self.data, static)
        segments, self.segment_of, _, _, _ = analyze.build_segments(
            self.data, border
        )
        motion, shape = analyze.motion_and_shape(
            self.data, segments, edge, border
        )
        if len(segments) != self.report["trusted_segments"]:
            raise ValueError("Trusted segment count differs from saved analysis")
        self.segments = {int(row["segment"]): row for row in segments}
        self.motion = {(r["frame"], r["id"]): r for r in motion}
        self.shape = {(r["frame"], r["id"]): r for r in shape}
        # Compare regenerated measurements with the saved, rounded CSV values.
        for filename, computed, fields in (
            ("motion.csv", self.motion, ("x_px", "y_px", "speed_px_s")),
            ("shape.csv", self.shape, ("aspect_ratio", "circularity", "solidity")),
        ):
            saved = read_csv(self.analysis / filename)
            if len(saved) != len(computed):
                raise ValueError(f"Changed row count: {filename}")
            for row in saved:
                actual = computed[(int(row["frame"]), int(row["id"]))]
                for field in fields:
                    if not np.isclose(
                        float(row[field]) if row[field] else np.nan,
                        actual[field], atol=0.000501,
                        rtol=0, equal_nan=True,
                    ):
                        raise ValueError(f"Changed measurement: {filename} {field}")
        self.rows = {
            (int(f), int(i)): k
            for k, (f, i) in enumerate(zip(self.data.F, self.data.I))
        }
        self.collective = read_csv(self.analysis / "collective.csv")
        self.validation, self.samples = capture_validation(
            self.data, motion, shape, self.report, self.video
        )
        self.preserved = {
            str(path.relative_to(ROOT)): digest(path)
            for path in [self.video, *self.result.glob("*"), *self.analysis.glob("*")]
            if path.is_file()
        }

    def roi(self, frames, ids, padding=35, minimum=170):
        """Return one fixed square crop containing the selected real contours."""
        contours = [
            self.data.contour[(f, i)] for f in frames for i in ids
            if (f, i) in self.data.contour
        ]
        points = np.vstack(contours)
        low, high = points.min(0), points.max(0)
        size = int(max(minimum, np.max(high - low) + 2 * padding))
        size = min(size, self.data.W, self.data.H)
        center = (low + high) / 2
        x = int(np.clip(center[0] - size / 2, 0, self.data.W - size))
        y = int(np.clip(center[1] - size / 2, 0, self.data.H - size))
        return x, y, size, size


class VideoSource:
    """Read only required frames, retaining one decoded frame at a time."""

    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"Cannot read {path}")
        self.index = -1
        self.image = None

    def get(self, frame):
        if frame == self.index:
            return self.image.copy()
        if frame < self.index or frame - self.index > 10:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.index = frame - 1
        while self.index < frame:
            success, decoded = self.capture.read()
            if not success:
                raise ValueError(f"Cannot decode source frame {frame}")
            self.index += 1
        self.image = Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        return self.image.copy()

    def close(self):
        self.capture.release()


class Design:
    """Arial-only drawing helpers shared by the analysis animations."""

    def __init__(self):
        self.fonts = {}
        self.labels = set()
        for path in (FONT, BOLD_FONT):
            if ImageFont.truetype(str(path), 32).getname()[0] != "Arial":
                raise ValueError(f"Actual Arial is required: {path}")

    def text(self, image, xy, value, size=30, color=INK, bold=False, anchor=None):
        value = str(value)
        if any("\u4e00" <= char <= "\u9fff" for char in value):
            raise ValueError("Non-English label in the animation")
        self.labels.add(value)
        key = (size, bold)
        if key not in self.fonts:
            path = BOLD_FONT if bold else FONT
            self.fonts[key] = ImageFont.truetype(str(path), size)
        ImageDraw.Draw(image).text(
            xy, value, font=self.fonts[key], fill=color, anchor=anchor
        )

    def base(self, title, subtitle, footer):
        image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.text(image, (64, 42), title, 52, bold=True)
        self.text(image, (66, 112), subtitle, 29, MUTED)
        ImageDraw.Draw(image).line((64, 169, 1856, 169), fill=GRID, width=2)
        ImageDraw.Draw(image).line((64, 1013, 1856, 1013), fill=GRID, width=2)
        self.text(image, (66, 1031), footer, 24, MUTED)
        return image

    def chart(self, image, box, xs, ys, xlabel, ylabel, color, cursor=None,
              ylim=None, xlim=None):
        """Plot measured samples with a cursor; no curve fitting or smoothing."""
        left, top, width, height = box
        xs, ys = np.asarray(xs), np.asarray(ys)
        xlo, xhi = xlim or (float(xs.min()), float(xs.max()))
        finite = ys[np.isfinite(ys)]
        if ylim is None:
            span = max(float(np.ptp(finite)), 1)
            ylim = (max(0, float(finite.min()) - span * .12),
                    float(finite.max()) + span * .15)
        ylo, yhi = ylim
        x0, y0, x1, y1 = left + 76, top + 50, left + width - 15, top + height - 55

        def point(x, y):
            return (x0 + (x - xlo) / (xhi - xlo) * (x1 - x0),
                    y1 - (y - ylo) / (yhi - ylo) * (y1 - y0))

        draw = ImageDraw.Draw(image)
        self.text(image, (left, top), ylabel, 29, bold=True)
        for y in np.linspace(ylo, yhi, 3):
            _, py = point(xlo, y)
            draw.line((x0, py, x1, py), fill=GRID, width=1)
            self.text(image, (x0 - 12, py), f"{y:.0f}", 23, MUTED, anchor="rm")
        for x in np.linspace(xlo, xhi, 3):
            px, _ = point(x, ylo)
            self.text(image, (px, y1 + 10), f"{x:.2f}", 23, MUTED, anchor="mt")
        self.text(image, ((x0 + x1) / 2, top + height - 12), xlabel, 22,
                  MUTED, anchor="mt")
        points = [point(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        if cursor is not None:
            px, _ = point(cursor[0], ylo)
            draw.line((px, y0, px, y1), fill=INK, width=2)
            px, py = point(*cursor)
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)
        return point


def paste_view(image, raw, roi, box):
    """Paste a fixed raw-video crop and return its pixel-to-slide transform."""
    rx, ry, rw, rh = roi
    x, y, width, height = box
    scale = min(width / rw, height / rh)
    rendered = raw.crop((rx, ry, rx + rw, ry + rh)).resize(
        (round(rw * scale), round(rh * scale)), Image.Resampling.LANCZOS
    )
    ox = x + (width - rendered.width) // 2
    oy = y + (height - rendered.height) // 2
    image.paste(rendered, (ox, oy))

    def transform(points):
        values = np.asarray(points, dtype=float)
        return (values - (rx, ry)) * scale + (ox, oy)

    return transform, scale


def translucent_panel(image, box):
    """Composite a 60%-opaque neutral panel over the actual video pixels."""
    x0, y0, x1, y1 = map(int, box)
    panel = Image.new("RGBA", (x1-x0, y1-y0), (250, 250, 250, PANEL_ALPHA))
    ImageDraw.Draw(panel).rectangle(
        (0, 0, panel.width-1, panel.height-1), outline=(65, 75, 85, 110), width=1
    )
    image.paste(panel, (x0, y0), panel)


def video_stamp(image, design, text):
    """Keep time and playback information in a small translucent caption."""
    key = (20, False)
    if key not in design.fonts:
        design.fonts[key] = ImageFont.truetype(str(FONT), 20)
    width = int(design.fonts[key].getlength(text)) + 24
    translucent_panel(image, (16, image.height-48, 16+width, image.height-16))
    design.text(image, (28, image.height-32), text, 20, INK, anchor="lm")


def compact_roi(evidence, square_roi):
    """Use an exact 4:3 crop with the measured region left of the information box."""
    x, y, side, _ = square_roi
    height = int(np.ceil(side / 3) * 3)
    width = height * 4 // 3
    if width > evidence.data.W or height > evidence.data.H:
        raise ValueError("The compact crop would extend beyond the raw video")
    left = int(np.clip(x + side/2 - .36*width, 0, evidence.data.W-width))
    top = int(np.clip(y + (side-height)/2, 0, evidence.data.H-height))
    return left, top, width, height


class PztTimeline:
    """Display archived command-to-video ranges for the selected recording.

    The original status_at() rule is reused. Orange transition intervals retain
    the saved receive-time bracket; they are not physical output measurements.
    """

    def __init__(self, evidence):
        folder = PROJECT / f"data/pzt/exp{evidence.experiment}"
        manifest_path = folder.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        entry = next(item for item in manifest["experiments"] if item["name"] == f"exp{evidence.experiment}")
        if digest(evidence.video) != entry["video_sha256"]:
            raise ValueError("PZT archive belongs to a different raw video")
        expected = {item["path"]: item["sha256"] for item in entry["files"]}
        names = ("link_report.json", "actions.csv", "video_frames.csv",
                 "link_session.json", "afg_ready.json")
        self.hashes = {str(manifest_path.relative_to(ROOT)): digest(manifest_path)}
        for name in names:
            path = folder / name
            actual = digest(path)
            if actual != expected[f"exp{evidence.experiment}/{name}"]:
                raise ValueError(f"Changed PZT archive file: {name}")
            self.hashes[str(path.relative_to(ROOT))] = actual
        report = json.loads((folder / "link_report.json").read_text())
        session = json.loads((folder / "link_session.json").read_text())
        ready = json.loads((folder / "afg_ready.json").read_text())
        self.experiment_id = entry["experiment_id"]
        if any(item["experiment_id"] != self.experiment_id for item in (report, session, ready)):
            raise ValueError("PZT archive contains mixed experiment identities")
        if not report["passed"] or ready["idle_vpp"] != session["idle_vpp"]:
            raise ValueError("PZT archive does not confirm its preset idle state")
        frames = read_csv(folder / "video_frames.csv")
        if [int(r["frame_index"]) for r in frames] != list(range(evidence.data.n_frames)):
            raise ValueError("PZT video-frame mapping is incomplete")
        if any(r["experiment_id"] != self.experiment_id for r in frames):
            raise ValueError("PZT video-frame identities differ")
        self.video_s = np.array([float(r["video_s"]) for r in frames])
        # Encoded PTS can alternate by one time-base tick. Compare the archive
        # with actual video timestamps, not an idealized constant-rate clock.
        with av.open(str(evidence.video)) as container:
            decoded_s = np.array([float(frame.pts * frame.time_base)
                                  for frame in container.decode(video=0)])
        decoded_s -= decoded_s[0]
        if len(decoded_s) != len(frames) or not np.allclose(
            self.video_s, decoded_s, atol=1e-9, rtol=0
        ):
            raise ValueError("PZT video times differ from the analysed frame sequence")
        self.duration_s = len(frames) / evidence.data.fps
        self.actions = {}
        self.intervals = []
        for channel in ("PZT1", "PZT2", "PZT3", "PZT4"):
            actions = [a for a in report["actions"]
                       if a["channel"] == channel and a["kind"] != "stop_repeat"]
            actions.sort(key=lambda a: a["command_windows_s"])
            self.actions[channel] = actions
            start, state = 0.0, "IDLE"
            for action in actions:
                bounds = action["command_video"]
                if not bounds["mapped"] or action["experiment_id"] != self.experiment_id:
                    raise ValueError("Unmapped or mismatched PZT command")
                low, high = bounds["video_from_s"], bounds["video_to_s"]
                if not start <= low <= high <= self.duration_s:
                    raise ValueError("PZT command ranges overlap or leave the video")
                self.intervals.extend([
                    {"channel": channel, "from_s": start, "to_s": low, "state": state},
                    {"channel": channel, "from_s": low, "to_s": high, "state": "SWITCHING"},
                ])
                start, state = high, action["state"]
            self.intervals.append({"channel": channel, "from_s": start,
                                   "to_s": self.duration_s, "state": state})
        evidence.preserved.update(self.hashes)
        evidence.pzt_timeline = self

    def draw(self, image, design, video_s):
        box = (image.width-416, 16, image.width-16, 282)
        translucent_panel(image, box)
        draw = ImageDraw.Draw(image)
        left, _, right, _ = box
        x0, x1 = left+86, right-18
        colors = {"ACTIVE": BLUE, "IDLE": "#c2ccd3", "STOP": "#657786",
                  "SWITCHING": ORANGE}

        def x(time_s):
            return x0 + time_s/self.duration_s*(x1-x0)

        design.text(image, (left+16, 29), "PZT commands", 22, bold=True)
        states = {}
        for index, (channel, actions) in enumerate(self.actions.items()):
            action, phase = status_at(actions, video_s)
            state = "IDLE" if action is None else action["state"]
            if phase == "Switching":
                state = "SWITCHING"
            states[channel] = state
            y = 69 + index*32
            design.text(image, (left+16, y+9), channel, 20,
                        BLUE if state == "ACTIVE" else INK, anchor="lm")
            for interval in self.intervals:
                if interval["channel"] == channel:
                    xa, xb = x(interval["from_s"]), x(interval["to_s"])
                    draw.rectangle((xa, y, xb, y+18), fill=colors[interval["state"]])
            # Paint narrow brackets last so the next state cannot cover them.
            for interval in self.intervals:
                if interval["channel"] == channel and interval["state"] == "SWITCHING":
                    xa = int(np.floor(x(interval["from_s"])))
                    xb = max(xa, int(np.ceil(x(interval["to_s"]))) - 1)
                    draw.rectangle((xa, y, xb, y+18), fill=colors["SWITCHING"])
        cursor = x(video_s)
        draw.line((cursor, 62, cursor, 188), fill=INK, width=2)
        for second in (0, 5, 10, 15, 20):
            design.text(image, (x(second), 194), str(second), 18, INK, anchor="mt")
        design.text(image, ((x0+x1)/2, 218), "Video time (s)", 18, INK, anchor="mt")
        labels = (("ACTIVE", "Active"), ("IDLE", "Idle"),
                  ("STOP", "Off"), ("SWITCHING", "Switch"))
        for index, (state, label) in enumerate(labels):
            px = left+18+index*94
            draw.rectangle((px, 253, px+16, 265), fill=colors[state])
            design.text(image, (px+23, 259), label, 18, INK, anchor="lm")
        return states


def line(image, points, color, width=3, closed=False):
    points = [tuple(p) for p in points]
    if closed and points:
        points.append(points[0])
    if len(points) >= 2:
        ImageDraw.Draw(image).line(points, fill=color, width=width, joint="curve")


def dot(image, point, color, radius=6):
    x, y = point
    ImageDraw.Draw(image).ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)


def arrow(image, start, end, color, width=4):
    start, end = np.asarray(start), np.asarray(end)
    line(image, [start, end], color, width)
    delta = end - start
    length = np.linalg.norm(delta)
    if length > 1:
        unit = delta / length
        side = np.array([-unit[1], unit[0]])
        head = [end, end - unit * 15 + side * 7, end - unit * 15 - side * 7]
        ImageDraw.Draw(image).polygon([tuple(p) for p in head], fill=color)


def outline(image, evidence, frame, track, transform, color, width=3):
    contour = evidence.data.contour.get((frame, track))
    if contour is not None:
        line(image, transform(contour), color, width, closed=True)


def segmentation_frames(evidence, source, design):
    """Show the entire recording and field, with separate trusted trail pieces.

    Trails show the last second of measured centroids. They never cross segment
    boundaries or missing observations. Cut records remain in the exported CSV;
    no cut rings are drawn on the video.
    """
    data = evidence.data
    palette = ("#00bddb", "#efad00", "#a17bea", "#00c192", "#f36d9a", "#488cfa")
    trail_frames = round(data.fps)
    timeline = PztTimeline(evidence)
    cut_events = []
    boundaries = {
        (int(data.F[s["rows"][0]]), s["track"])
        for s in evidence.segments.values()
    }
    for event in data.events:
        if event["type"] not in ("merge", "split"):
            continue
        if any((event["frame"], i) in boundaries
               for i in [*event["ids"], event["result"]]):
            cut_events.append({
                "frame": event["frame"], "type": event["type"].title(),
                "x_px": event["x"], "y_px": event["y"],
            })
    for rows in data.rows_by_track.values():
        for previous, current in zip(rows[:-1], rows[1:]):
            if data.status[previous] == "trusted" and data.status[current] != "trusted":
                cut_events.append({
                    "frame": int(data.F[current]), "type": "Untrusted",
                    "x_px": float(data.cx[current]), "y_px": float(data.cy[current]),
                })
    cut_events.sort(key=lambda event: event["frame"])
    evidence.global_cut_events = cut_events
    seen_segments = set()
    for frame in range(data.n_frames):
        image = Image.new("RGB", (data.W, data.H))
        transform, _ = paste_view(
            image, source.get(frame), (0, 0, data.W, data.H),
            (0, 0, data.W, data.H),
        )
        # Keep completed pieces visible for one second, but never join pieces.
        for segment, record in evidence.segments.items():
            rows = record["rows"]
            frames = data.F[rows]
            first = np.searchsorted(frames, frame - trail_frames + 1)
            last = np.searchsorted(frames, frame, side="right")
            selected = rows[first:last]
            if not len(selected):
                continue
            if not np.all(evidence.segment_of[selected] == segment):
                raise ValueError("A displayed trail crosses a segment boundary")
            if not np.all(data.status[selected] == "trusted"):
                raise ValueError("An untrusted row entered a displayed trail")
            breaks = np.flatnonzero(np.diff(data.F[selected]) > 1) + 1
            color = palette[(segment - 1) % len(palette)]
            for piece in np.split(selected, breaks):
                points = np.c_[data.cx[piece], data.cy[piece]]
                line(image, transform(points), color, 3)
        active_segments = []
        occupied = []
        draw = ImageDraw.Draw(image)
        for k in data.rows_by_frame[frame]:
            track = int(data.I[k])
            segment = int(evidence.segment_of[k])
            if not segment:
                outline(image, evidence, frame, track, transform, "#a5a5a5", 2)
                continue
            active_segments.append(segment)
            seen_segments.add(segment)
            color = palette[(segment - 1) % len(palette)]
            outline(image, evidence, frame, track, transform, color, 2)
            center = transform([[data.cx[k], data.cy[k]]])[0]
            dot(image, center, color, 4)
            # Place compact segment labels beside regions, avoiding other labels.
            label = f"S{segment}"
            label_width = 12 * len(label) + 8
            for dx, dy in ((9, -26), (9, 5), (-label_width-9, -26),
                           (-label_width-9, 5), (9, -51), (9, 30)):
                x = float(np.clip(center[0] + dx, 4, data.W-label_width-4))
                y = float(np.clip(center[1] + dy, 4, data.H-29))
                box = (x, y, x + label_width, y + 25)
                if not any(box[0] < b[2] and box[2] > b[0]
                           and box[1] < b[3] and box[3] > b[1] for b in occupied):
                    break
            occupied.append(box)
            draw.rounded_rectangle(box, radius=3, fill=INK)
            design.text(image, (x+4, y+1), label, 20, WHITE)
        translucent_panel(image, (16, 16, 270, 115))
        design.text(image, (32, 29), "Segments seen", 24, bold=True)
        design.text(image, (32, 66),
                    f"{len(seen_segments)} / {len(evidence.segments)}", 34, BLUE, bold=True)
        states = timeline.draw(image, design, timeline.video_s[frame])
        video_stamp(image, design,
                    f"t = {frame / data.fps:.3f} s  |  S: segment ID  |  Trails: 1 s")
        yield image, 1, {
            "source_frame": frame, "scene": "full-field trajectory segmentation",
            "active_segments": " ".join(map(str, active_segments)),
            "segments_seen": len(seen_segments),
            "pzt_command_states": "; ".join(f"{key}={value}" for key, value in states.items()),
        }
    if len(seen_segments) != len(evidence.segments):
        raise ValueError("The global animation did not display every trusted segment")


def kinematics_plot(times, speeds, design):
    """Prepare an Arial plot with fixed axes and a moving measured-data marker.

    The axes use regular major/minor ticks, inward ticks and explicit units.
    No smoothing, fitted curve or invented uncertainty is added to the data.
    """
    dpi = 160
    figure = Figure(figsize=(440 / dpi, 260 / dpi), dpi=dpi, facecolor=(0, 0, 0, 0))
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0.17, 0.23, 0.79, 0.72))
    axes.patch.set_alpha(0)
    font = FontProperties(fname=str(FONT), size=9)
    tick_font = FontProperties(fname=str(FONT), size=8)
    axes.plot(times, speeds, color=BLUE, linewidth=1.3)
    cursor = axes.axvline(times[0], color="0.35", linewidth=0.75,
                         linestyle=(0, (3, 3)))
    marker, = axes.plot([times[0]], [speeds[0]], marker="o", markersize=5.5,
                        color=BLUE, markeredgecolor=WHITE, markeredgewidth=0.7)
    axes.set_xlim(np.floor(times[0]), np.ceil(times[-1]))
    axes.set_ylim(0, np.ceil(max(speeds) / 200) * 200)
    axes.set_xticks(np.arange(np.floor(times[0]), np.ceil(times[-1]) + .01, .2))
    axes.set_yticks(np.arange(0, axes.get_ylim()[1] + 1, 200))
    axes.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    axes.xaxis.set_minor_locator(AutoMinorLocator(2))
    axes.yaxis.set_minor_locator(AutoMinorLocator(2))
    axes.set_xlabel("Time (s)", fontproperties=font, labelpad=4)
    axes.set_ylabel("Speed (px/s)", fontproperties=font, labelpad=5)
    axes.tick_params(which="both", direction="in", top=True, right=True,
                     labelsize=8, colors="black", width=0.7)
    axes.tick_params(which="major", length=3.5, pad=4)
    axes.tick_params(which="minor", length=2)
    for spine in axes.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("black")
    ticks = axes.get_xticklabels() + axes.get_yticklabels()
    for label in ticks:
        label.set_fontproperties(tick_font)
    canvas.draw()
    for label in [axes.xaxis.label, axes.yaxis.label, *ticks]:
        if label.get_fontproperties().get_file() != str(FONT):
            raise ValueError("A plot label does not use the requested Arial file")
        design.labels.add(label.get_text())
    return canvas, cursor, marker


def kinematics_frames(evidence, source, design):
    """Show motion with a point legend and a speed plot inside the raw view."""
    first, track = (720, 11) if evidence.experiment == "A" else (246, 2)
    frames = range(first, first + 60)
    source_fps = evidence.data.fps
    square_roi = evidence.roi(range(first - 3, first + 63), [track], padding=45, minimum=240)
    video_box = (0, 0, *COMPACT_SIZE)
    roi = compact_roi(evidence, square_roi)
    times = np.array(list(frames)) / source_fps
    speeds = [evidence.motion[(f, track)]["speed_px_s"] for f in frames]
    canvas, cursor, marker = kinematics_plot(times, speeds, design)
    panel_box = (800, 16, 1264, 460)
    plot_origin = (812, 94)
    evidence.kinematics_display = {
        "source_frames": [frames.start, frames.stop - 1],
        "source_roi_xywh": list(roi), "video_box_xywh": list(video_box),
        "combined_panel_xyxy": list(panel_box),
        "speed_plot_box_xyxy": [*plot_origin, plot_origin[0] + 440,
                                plot_origin[1] + 260],
        "panel_opacity": PANEL_ALPHA / 255,
        "velocity_window_frames": [-3, 3], "playback_speed": 0.125,
        "plot_x_limits_s": list(canvas.figure.axes[0].get_xlim()),
        "plot_y_limits_px_s": list(canvas.figure.axes[0].get_ylim()),
        "smoothing": False, "target_clear_of_overlays": True,
    }
    for frame in frames:
        row = evidence.motion[(frame, track)]
        before, after = [evidence.motion[(frame + delta, track)] for delta in (-3, 3)]
        if not before["segment"] == row["segment"] == after["segment"]:
            raise ValueError("Velocity window crosses a segment boundary")
        image = Image.new("RGB", COMPACT_SIZE)
        transform, _ = paste_view(image, source.get(frame), roi, video_box)
        outline(image, evidence, frame, track, transform, BLUE, 4)
        trail = [evidence.motion[(f, track)] for f in range(first - 3, frame+1)]
        points = transform([(r["x_px"], r["y_px"]) for r in trail])
        line(image, points, BLUE, 3)
        positions = transform([(r["x_px"], r["y_px"]) for r in (before, row, after)])
        arrow(image, positions[0], positions[2], ORANGE, 5)
        draw = ImageDraw.Draw(image)
        for point, color in zip(positions, (PURPLE, BLUE, TEAL)):
            x, y = point
            draw.ellipse((x-8, y-8, x+8, y+8), fill=color, outline=INK, width=1)
        # These overlays lie entirely inside the raw view, away from the target.
        target = transform(evidence.data.contour[(frame, track)])
        target_min, target_max = target.min(0), target.max(0)
        if (target_min[0] < panel_box[2] and target_max[0] > panel_box[0]
                and target_min[1] < panel_box[3] and target_max[1] > panel_box[1]):
            raise ValueError("The kinematics panel would cover the measured region")
        translucent_panel(image, panel_box)
        draw = ImageDraw.Draw(image)
        for index, (label, color) in enumerate(
                (("f − 3", PURPLE), ("f", BLUE), ("f + 3", TEAL))):
            x, y = 830 + index*135, 42
            draw.ellipse((x-7, y-7, x+7, y+7), fill=color, outline=INK, width=1)
            design.text(image, (x+16, y), label, 23, INK, anchor="lm")
        design.text(image, (823, 65), "Δt = 0.10 s", 21)
        cursor.set_xdata([frame / source_fps, frame / source_fps])
        marker.set_data([frame / source_fps], [row["speed_px_s"]])
        canvas.draw()
        plot_image = Image.fromarray(np.asarray(canvas.buffer_rgba()).copy())
        image.paste(plot_image, plot_origin, plot_image)
        distance = np.hypot(after["x_px"]-before["x_px"], after["y_px"]-before["y_px"])
        design.text(image, (823, 375), f"Δposition = {distance:.2f} px", 24)
        design.text(image, (823, 415), f"Speed = {row['speed_px_s']:.1f} px/s", 24, BLUE)
        video_stamp(image, design,
                    f"t = {frame/source_fps:.3f} s  |  ID {track}  |  0.125×")
        yield image, 4, {
            "source_frame": frame, "scene": "centroid velocity", "id": track,
            "speed_px_s": float(row["speed_px_s"]),
            "window_displacement_px": float(distance),
        }


def morphology_frames(evidence, source, design):
    """Show shape geometry with its legend and measurements inside the video."""
    first, track = (180, 11) if evidence.experiment == "A" else (234, 2)
    frames = range(first, first + 21)
    square_roi = evidence.roi(frames, [track], padding=30, minimum=145)
    video_box = (0, 0, *COMPACT_SIZE)
    roi = compact_roi(evidence, square_roi)
    panel_box = (824, 16, 1264, 344)
    evidence.morphology_display = {
        "source_frames": [frames.start, frames.stop - 1],
        "source_roi_xywh": list(roi), "video_box_xywh": list(video_box),
        "combined_panel_xyxy": list(panel_box),
        "panel_opacity": PANEL_ALPHA / 255,
        "playback_speed": 0.0625, "target_clear_of_overlays": True,
        "measurements": ["area_px2", "aspect_ratio", "circularity", "solidity"],
    }
    for frame in frames:
        row = evidence.shape[(frame, track)]
        if not row["shape_ok"]:
            raise ValueError("Morphology example is not marked reliable")
        image = Image.new("RGB", COMPACT_SIZE)
        transform, _ = paste_view(image, source.get(frame), roi, video_box)
        contour = evidence.data.contour[(frame, track)]
        line(image, transform(cv2.convexHull(contour).reshape(-1, 2)), ORANGE, 4, True)
        outline(image, evidence, frame, track, transform, BLUE, 4)
        k = evidence.rows[(frame, track)]
        theta = np.deg2rad(evidence.data.angle[k])
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        center = np.array([evidence.data.cx[k], evidence.data.cy[k]])
        radii = np.array([row["major_px"], row["minor_px"]]) / 2
        angles = np.linspace(0, 2*np.pi, 120)
        ellipse = np.c_[np.cos(angles), np.sin(angles)] * radii
        ellipse_points = ellipse @ rotation.T + center
        line(image, transform(ellipse_points), TEAL, 3, True)
        for axis in range(2):
            vector = rotation[:, axis] * radii[axis]
            line(image, transform([center-vector, center+vector]), TEAL, 2)
        geometry = transform(np.vstack([contour, ellipse_points]))
        target_min, target_max = geometry.min(0), geometry.max(0)
        if (target_min[0] < panel_box[2] and target_max[0] > panel_box[0]
                and target_min[1] < panel_box[3] and target_max[1] > panel_box[1]):
            raise ValueError("The morphology panel would cover the shape geometry")
        translucent_panel(image, panel_box)
        draw = ImageDraw.Draw(image)
        for index, (label, color) in enumerate(
                (("Contour", BLUE), ("Moment ellipse", TEAL), ("Convex hull", ORANGE))):
            y = 46 + index*35
            draw.line((844, y, 884, y), fill=color, width=4)
            design.text(image, (899, y), label, 24, INK, anchor="lm")
        draw.line((844, 145, 1244, 145), fill=MUTED, width=1)
        measures = [
            ("Area", f"{row['area_px2']:.0f} px²"),
            ("Aspect ratio", f"{row['aspect_ratio']:.2f}"),
            ("Circularity", f"{row['circularity']:.2f}"),
            ("Solidity", f"{row['solidity']:.2f}"),
        ]
        for index, (label, value) in enumerate(measures):
            y = 178 + index*44
            design.text(image, (844, y), label, 23, INK, anchor="lm")
            design.text(image, (1244, y), value, 27, BLUE, anchor="rm")
        video_stamp(image, design,
                    f"t = {frame/evidence.data.fps:.3f} s  |  ID {track}  |  0.0625×")
        yield image, 8, {
            "source_frame": frame, "scene": "shape measurements", "id": 11,
            "area_px2": float(row["area_px2"]),
            "aspect_ratio": float(row["aspect_ratio"]),
            "circularity": float(row["circularity"]),
            "solidity": float(row["solidity"]),
        }


def collective_plots(times, series, design):
    """Prepare three aligned scientific plots without changing measured values."""
    dpi = 160
    figure = Figure(figsize=(690/dpi, 780/dpi), dpi=dpi, facecolor=WHITE)
    canvas = FigureCanvasAgg(figure)
    font = FontProperties(fname=str(FONT), size=10.5)
    tick_font = FontProperties(fname=str(FONT), size=9.5)
    value_font = FontProperties(fname=str(BOLD_FONT), size=10.5)
    specs = (
        ("n_regions", "Region count", BLUE, 30, 10),
        ("cluster_area_share", "Area in clusters (%)", ORANGE, 100, 25),
        ("spread_px", "Spread (px)", TEAL, 500, 100),
    )
    artists = []
    for index, (key, label, color, maximum, tick_step) in enumerate(specs):
        if np.nanmax(series[key]) > maximum:
            raise ValueError(f"Collective plot would clip {key}")
        bottom = (780 - (index*260 + 190)) / 780
        axes = figure.add_axes((.18, bottom, .78, 148/780), facecolor=WHITE)
        axes.plot(times, series[key], color=color, linewidth=1.25)
        cursor = axes.axvline(times[0], color="0.35", linewidth=.7,
                             linestyle=(0, (3, 3)))
        marker, = axes.plot([times[0]], [series[key][0]], marker="o", markersize=5,
                            color=color, markeredgecolor=WHITE, markeredgewidth=.6)
        value = axes.text(1, 1.06, "", ha="right", va="bottom", transform=axes.transAxes,
                          fontproperties=value_font, color=color)
        for artist in (cursor, marker, value):
            artist.set_animated(True)
        axes.set_xlim(0, 22)
        axes.set_ylim(0, maximum)
        axes.set_xticks(np.arange(0, 21, 5))
        axes.set_yticks(np.arange(0, maximum+1, tick_step))
        axes.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        axes.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        axes.xaxis.set_minor_locator(AutoMinorLocator(2))
        axes.yaxis.set_minor_locator(AutoMinorLocator(2))
        axes.set_xlabel("Video time (s)", fontproperties=font, labelpad=3)
        axes.set_ylabel(label, fontproperties=font, labelpad=7)
        axes.tick_params(which="both", direction="in", top=True, right=True,
                         colors="black", width=.7)
        axes.tick_params(which="major", length=3.5, pad=4)
        axes.tick_params(which="minor", length=2)
        for spine in axes.spines.values():
            spine.set_linewidth(.7)
            spine.set_color("black")
        for tick in axes.get_xticklabels() + axes.get_yticklabels():
            tick.set_fontproperties(tick_font)
        artists.append((key, cursor, marker, value))
    canvas.draw()
    for axes in figure.axes:
        for text in [axes.xaxis.label, axes.yaxis.label,
                     *axes.get_xticklabels(), *axes.get_yticklabels(), *axes.texts]:
            if text.get_fontproperties().get_file() not in (str(FONT), str(BOLD_FONT)):
                raise ValueError("A collective plot label does not use actual Arial")
            design.labels.add(text.get_text())
    # Only the cursor, current point and number change between video frames.
    # Cache the unchanging curves and axes instead of rebuilding them each time.
    background = canvas.copy_from_bbox(figure.bbox)
    return canvas, artists, background


def collective_frames(evidence, source, design):
    """Show full-field metrics with an inset legend and aligned right-hand plots."""
    data = evidence.data
    times = np.arange(data.n_frames) / data.fps
    series = {
        key: np.array([float(r[key]) for r in evidence.collective])
        for key in ("n_regions", "cluster_area_share", "spread_px")
    }
    series["cluster_area_share"] *= 100
    canvas, artists, plot_background = collective_plots(times, series, design)
    legend_box = (130, 227, 455, 455)
    plot_origin = (1150, 206)
    evidence.collective_display = {
        "video_box_xywh": [64, 211, 1050, 760],
        "legend_box_xyxy": list(legend_box),
        "plot_box_xyxy": [1150, 206, 1840, 986],
        "legend_opacity": PANEL_ALPHA/255,
        "plot_background_opacity": 1.0,
        "canvas_background": WHITE,
        "x_limits_s": [0, 22], "x_major_tick_s": 5,
        "y_limits": {"n_regions": [0, 30], "cluster_area_percent": [0, 100],
                     "spread_px": [0, 500]},
        "smoothing": False,
        "cluster_threshold_px2": evidence.data.P["cluster_area_px"],
        "included_roles": ["confirmed", "uncertain"],
        "excluded_role": "excluded",
        "definitions": {
            "n_regions": "Count of included segmented regions, not individual bubbles.",
            "cluster_area_share": "Sum of included region areas >= 300 px2 / sum of all included region areas.",
            "weighted_centre": "Region-centroid coordinates weighted by each region's area.",
            "spread_px": "Square root of the area-weighted mean squared distance of region centroids from the weighted centre.",
            "circle": "Radius equals spread_px; not the field boundary or an enclosing circle.",
        },
    }
    for frame in range(0, data.n_frames, 2):
        row = evidence.collective[frame]
        # Keep a white canvas and white plots. Only the in-video legend blends
        # with raw pixels, as requested; plot locations remain unchanged.
        image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        design.text(image, (64, 42), "4  Collective metrics", 52, bold=True)
        design.text(image, (66, 112), "Whole field  •  Clusters: area ≥ 300 px²", 29, MUTED)
        transform, scale = paste_view(
            image, source.get(frame), (0, 0, data.W, data.H), (64, 211, 1050, 760)
        )
        for k in data.rows_by_frame[frame]:
            if data.role[k] == "excluded":
                continue
            color = ORANGE if data.area[k] >= 300 else BLUE
            if data.role[k] == "uncertain":
                color = RED
            outline(image, evidence, frame, int(data.I[k]), transform, color, 2)
        center = transform([[float(row["centroid_x_px"]), float(row["centroid_y_px"])]])[0]
        radius = float(row["spread_px"]) * scale
        draw = ImageDraw.Draw(image)
        # RMS distance of region centroids, not a containment boundary.
        for start in range(0, 360, 15):
            draw.arc((center[0]-radius, center[1]-radius,
                      center[0]+radius, center[1]+radius), start, start+8,
                     fill=WHITE, width=2)
        draw.line((center[0]-12, center[1], center[0]+12, center[1]), fill=WHITE, width=3)
        draw.line((center[0], center[1]-12, center[0], center[1]+12), fill=WHITE, width=3)
        translucent_panel(image, legend_box)
        draw = ImageDraw.Draw(image)
        for index, (label, color) in enumerate(
                (("Region < 300 px²", BLUE), ("Cluster ≥ 300 px²", ORANGE),
                 ("Uncertain", RED))):
            y = 255 + index*40
            draw.line((148, y, 178, y), fill=color, width=4)
            design.text(image, (191, y), label, 23, INK, anchor="lm")
        for color, width in ((INK, 5), (WHITE, 3)):
            draw.line((153, 375, 173, 375), fill=color, width=width)
            draw.line((163, 365, 163, 385), fill=color, width=width)
        design.text(image, (191, 375), "Weighted centre", 23, INK, anchor="lm")
        for start in range(0, 360, 60):
            draw.arc((151, 403, 175, 427), start, start+35, fill=INK, width=4)
            draw.arc((151, 403, 175, 427), start, start+35, fill=WHITE, width=2)
        design.text(image, (191, 415), "RMS spread", 23, INK, anchor="lm")
        canvas.restore_region(plot_background)
        for key, cursor, marker, label in artists:
            value = series[key][frame]
            cursor.set_xdata([frame/data.fps, frame/data.fps])
            marker.set_data([frame/data.fps], [value])
            if key == "n_regions":
                text = f"{value:.0f}"
            elif key == "cluster_area_share":
                text = f"{value:.1f}%"
            else:
                text = f"{value:.1f} px"
            label.set_text(text)
            design.labels.add(text)
            for artist in (cursor, marker, label):
                artist.axes.draw_artist(artist)
        plot = Image.fromarray(np.asarray(canvas.buffer_rgba()).copy())
        image.paste(plot, plot_origin, plot)
        design.text(image, (64, 1034), f"exp{evidence.experiment}  |  t = {frame/data.fps:.3f} s  |  1×", 24, MUTED)
        yield image, 1, {
            "source_frame": frame, "scene": "whole field", "id": "",
            "n_regions": int(row["n_regions"]),
            "cluster_area_percent": float(series["cluster_area_share"][frame]),
            "spread_px": float(row["spread_px"]),
        }


def validation_frames(evidence, source, design):
    samples = evidence.samples
    differences = np.array([s["difference_px"] for s in samples])
    # Show representative quantiles, including the largest discrepancy.
    order = np.argsort(differences)
    chosen = [samples[order[round(q*(len(order)-1))]]
              for q in (0.1, 0.25, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0)]
    centroid = np.array([np.hypot(s["cdx"], s["cdy"]) for s in samples])
    template = np.array([np.hypot(s["tdx"], s["tdy"]) for s in samples])
    limit = float(np.ceil(max(centroid.max(), template.max()) / 10) * 10)
    for number, sample in enumerate(chosen, 1):
        f = sample["frame"]
        image = design.base(
            "Output  Raw-video motion check",
            "Template matching on raw frames  •  0.10 s displacement",
            f"exp{evidence.experiment}  |  Sample {sample['sample']} / 100  |  ID {sample['id']}  |  Frame {f} → {f+6}",
        )
        x0, y0 = sample["x0"], sample["y0"]
        rw, rh = sample["x1"]-x0, sample["y1"]-y0
        roi = (x0, y0, rw, rh)
        for index, offset in enumerate((0, 6)):
            x = 65 + index * 520
            design.text(image, (x, 215), "Reference" if not offset else "+0.10 s", 32, bold=True)
            transform, _ = paste_view(image, source.get(f+offset), roi, (x, 273, 495, 495))
            center = np.array([sample["X"], sample["Y"]], float)
            if offset:
                center += (sample["tdx"], sample["tdy"])
            half = sample["h"]
            box = center + np.array([[-half, -half], [half, -half],
                                     [half, half], [-half, half]])
            line(image, transform(box), TEAL if offset else BLUE, 3, True)
            dot(image, transform([center])[0], TEAL if offset else BLUE, 5)
        d_centroid = np.hypot(sample["cdx"], sample["cdy"])
        d_template = np.hypot(sample["tdx"], sample["tdy"])
        design.text(image, (65, 800), f"Centroid  {d_centroid:.2f} px", 32, BLUE)
        design.text(image, (585, 800), f"Template  {d_template:.2f} px", 32, TEAL)
        design.text(image, (65, 873), f"Vector difference  {sample['difference_px']:.2f} px", 38, bold=True)
        design.text(image, (65, 935), f"Match score  {sample['score']:.3f}", 28, MUTED)
        left, top, size = 1230, 270, 540

        def point(x, y):
            return left + x/limit*size, top + size - y/limit*size

        draw = ImageDraw.Draw(image)
        design.text(image, (1165, 207), "100 real comparisons", 34, bold=True)
        for tick in np.linspace(0, limit, 5):
            px, py = point(tick, tick)
            draw.line((left, py, left+size, py), fill=GRID, width=1)
            draw.line((px, top, px, top+size), fill=GRID, width=1)
            design.text(image, (left-15, py), f"{tick:.0f}", 23, MUTED, anchor="rm")
            design.text(image, (px, top+size+15), f"{tick:.0f}", 23, MUTED, anchor="mt")
        line(image, [point(0, 0), point(limit, limit)], MUTED, 2)
        design.text(image, (left+size-15, top+15), "1:1", 25, MUTED, anchor="rt")
        for dx, dy in zip(centroid, template):
            dot(image, point(dx, dy), "#8499ad", 4)
        dot(image, point(d_centroid, d_template), TEAL, 9)
        design.text(image, (left+size/2, 864), "Centroid displacement (px)", 26,
                    MUTED, anchor="mt")
        # Horizontal axis labels avoid rotated/rasterised font substitutes.
        design.text(image, (left, 239), "Template displacement (px)", 25, MUTED)
        median = evidence.validation["median_diff_px"]
        design.text(image, (1165, 931), f"Median difference  {median:.2f} px", 38, TEAL, bold=True)
        yield image, 48, {
            "source_frame": f, "later_frame": f+6,
            "scene": f"validation pair {number}", "id": sample["id"],
            "sample": sample["sample"],
        }


def encode_movie(path, frames, fps=OUTPUT_FPS):
    """Encode H.264 MP4 and retain a map from every slide frame to source data."""
    mapping = []
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
        stream = None
        index = 0
        for image, repetitions, evidence in frames:
            if stream is None:
                if image.width % 2 or image.height % 2:
                    raise ValueError("H.264 yuv420p dimensions must be even")
                stream = container.add_stream("libx264", rate=fps)
                stream.width, stream.height = image.size
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18", "preset": "fast", "threads": "4"}
            if image.size != (stream.width, stream.height):
                raise ValueError("Video dimensions changed during export")
            rgb = np.asarray(image)
            for _ in range(repetitions):
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                mapping.append({"output_frame": index,
                                "output_time_s": index/fps, **evidence})
                index += 1
        if stream is None:
            raise ValueError("No animation frames were generated")
        for packet in stream.encode():
            container.mux(packet)
    write_csv(path.with_suffix(".frames.csv"), mapping)
    return len(mapping)


def verify_movie(path, expected_frames, fps=OUTPUT_FPS, size=(WIDTH, HEIGHT)):
    """Decode every encoded frame and verify dimensions, count and timestamps."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        times = []
        for frame in container.decode(stream):
            if (frame.width, frame.height) != tuple(size):
                raise ValueError(f"Wrong video dimensions: {path}")
            times.append(float(frame.pts * frame.time_base))
        if len(times) != expected_frames:
            raise ValueError(f"Incomplete movie: {path}")
        if not np.allclose(times, np.arange(expected_frames)/fps, atol=1e-6):
            raise ValueError(f"Unexpected movie timing: {path}")
        return {"frames": len(times), "duration_s": len(times)/fps,
                "codec": stream.codec_context.name, "pixel_format": stream.pix_fmt,
                "size": list(size), "fps": fps,
                "all_frames_decoded": True, "sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment", choices=("A", "B"), default="A")
    parser.add_argument("--step", choices=("all", "1-3", "1", "2", "3", "4", "5"), default="all",
                        help="Export all steps, or only the selected step")
    args = parser.parse_args()
    timestamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir or ROOT / f"customer_workspace/validation/analysis_slides_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    design = Design()
    print("Verifying saved measurements and reproducing the raw-video check...", flush=True)
    evidence = Evidence(args.experiment)
    print(f"Verified {len(evidence.segments)} segments, {len(evidence.motion)} trusted rows; "
          f"median difference {evidence.validation['median_diff_px']:.9f} px", flush=True)
    write_csv(output / "validation_samples.csv", evidence.samples)
    movies = {}
    renderers = [
        ("01_trajectory_segmentation", segmentation_frames),
        ("02_kinematics", kinematics_frames),
        ("03_morphology", morphology_frames),
        ("04_collective_metrics", collective_frames),
        ("05_raw_video_validation", validation_frames),
    ]
    if args.step == "1-3":
        renderers = renderers[:3]
    elif args.step != "all":
        renderers = [renderers[int(args.step) - 1]]
    for name, renderer in renderers:
        source = VideoSource(evidence.video)
        path = output / f"{name}.mp4"
        print(f"Rendering {path.name}...", flush=True)
        fps = round(evidence.data.fps) if renderer is segmentation_frames else OUTPUT_FPS
        if renderer is segmentation_frames:
            size = (evidence.data.W, evidence.data.H)
        elif renderer in (kinematics_frames, morphology_frames):
            size = COMPACT_SIZE
        else:
            size = (WIDTH, HEIGHT)
        try:
            count = encode_movie(path, renderer(evidence, source, design), fps=fps)
        finally:
            source.close()
        movies[path.name] = verify_movie(path, count, fps=fps, size=size)
        print(f"Decoded all {count} frames: {path.name}", flush=True)
    if hasattr(evidence, "global_cut_events"):
        write_csv(output / "trajectory_cut_events.csv", evidence.global_cut_events)
    if hasattr(evidence, "pzt_timeline"):
        write_csv(output / "pzt_command_intervals.csv", evidence.pzt_timeline.intervals)
    changed = [p for p, value in evidence.preserved.items() if digest(ROOT/p) != value]
    if changed:
        raise ValueError(f"Original inputs changed: {changed}")
    provenance = {
        "created": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
        "source_video": str(evidence.video), "source_result": str(evidence.result),
        "source_fps": evidence.data.fps,
        "source_frames": evidence.data.n_frames,
        "trusted_segments": len(evidence.segments),
        "trusted_rows": len(evidence.motion),
        "exported_steps": args.step,
        "font_files": {str(p): digest(p) for p in (FONT, BOLD_FONT)},
        "visible_font": "Arial", "visible_language": "English",
        "overlay_background_opacity": PANEL_ALPHA / 255,
        "source_sha256": evidence.preserved, "original_files_unchanged": True,
        "script_sha256": digest(Path(__file__)),
        "video_motion_check": evidence.validation, "movies": movies,
        "limitations": [
            "The raw-video validation value is a median difference between centroid and template displacements, not absolute tracking accuracy.",
            "The original 100-sample selection is reproduced with seed 1; eight displayed pairs span discrepancy quantiles including the maximum.",
            "Template matching uses raw pixels; the existing check derives its initial crop and search radius from tracking geometry.",
            "Saved outlines are simplified contours. Metrics come from original analysis values, not from the display polygons.",
            "The ellipse uses region moments, not cv2.fitEllipse. Shape examples are marked good by the existing analysis.",
            "Whole-field metrics include uncertain regions and exclude confirmed false targets. Entry and exit can change these metrics.",
            "The spread circle is area-weighted RMS distance of region centroids, not an enclosing boundary.",
            "Step 1 displays all source frames at 60 fps. Step 4 displays every second source frame at 30 fps. Both play at 1x. Other slow-motion clips repeat real frames.",
            "No exposure timing, physical synchronization, hardware performance or micrometre calibration is inferred.",
            "PZT bars show archived command-to-video ranges. They do not establish physical voltage or acoustic onset times.",
            "Translucency is composited over video pixels. The H.264 MP4 itself has no alpha channel.",
        ],
    }
    if hasattr(evidence, "global_cut_events"):
        provenance["global_trajectory_display"] = {
            "source_frame_stride": 1, "playback_speed": 1,
            "trail_history_s": 1, "cut_markers_drawn": False,
            "labels": "trusted segment IDs, not object IDs",
            "cut_events_file": "trajectory_cut_events.csv",
        }
    if hasattr(evidence, "pzt_timeline"):
        provenance["pzt_timeline"] = {
            "experiment_id": evidence.pzt_timeline.experiment_id,
            "video_sha256_matches_archive": True,
            "file": "pzt_command_intervals.csv",
            "basis": "archived command_video ranges; not physical output onset",
            "switching_range_display": "orange, at least one display pixel",
            "source_sha256": evidence.pzt_timeline.hashes,
        }
    if hasattr(evidence, "kinematics_display"):
        provenance["kinematics_display"] = evidence.kinematics_display
    if hasattr(evidence, "morphology_display"):
        provenance["morphology_display"] = evidence.morphology_display
    if hasattr(evidence, "collective_display"):
        provenance["collective_display"] = evidence.collective_display
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output / "visible_labels.json").write_text(json.dumps(sorted(design.labels), indent=2), encoding="utf-8")
    archive = output / "slide6_animations.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(output.iterdir()):
            if path != archive:
                bundle.write(path, arcname=path.name)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
