"""Run existing algorithms and export matched detection/tracking/event movies.

All experiment inputs are read only. Work is split into prepare, masks,
baseline and render phases so the CSRT/MobileSAM environment stays isolated.
Run as a module from the authoritative project root. This wrapper does not
change either detector, tracker, identity assignment or event definitions.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import platform
import subprocess
import time
import zipfile
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image, ImageDraw

from customer_workspace.analysis_ppt import (
    BOLD_FONT, FONT, PROJECT, ROOT, Design, VideoSource, digest, dot,
    encode_movie, line, paste_view, read_csv, translucent_panel, verify_movie,
    write_csv,
)
from customer_workspace.microbubble_tracking import bubble_tracker as bt
from customer_workspace.microbubble_tracking.analysis import analyze
from customer_workspace.microbubble_tracking.baseline import channel_mask as cm
from customer_workspace.microbubble_tracking.baseline import medany_baseline as mb


VIDEO = PROJECT / "data/raw/expA.mp4"
REFERENCE = PROJECT / "results/A_offline"
MOBILE_SAM = ROOT / "customer_workspace/dependencies/MobileSAM"
THRESHOLDS = (100, 130)
SIZE = (1600, 1000)
BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
INK = "#202124"
GREY = "#62676D"
PANELS = ((50, 165, 730, 730), (820, 165, 730, 730))
EVENT_CLIPS = (
    dict(name="Merger", first=170, last=230, event_frame=201,
         event_type="merge", target=8, ids=[8, 11], message="Merger: 8 + 11 → 11"),
    dict(name="Separation", first=590, last=650, event_frame=617,
         event_type="split", target=11, ids=[11, 67], message="Separation: 11 → 11 + 67"),
    dict(name="Disappearance", first=804, last=894, event_frame=834,
         event_type="exit", target=11, ids=[11], message="Exit: ID 11"),
)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_run(output):
    """Read new runs while retaining compatibility with archived A exports."""
    metadata = json.loads((output / "run.json").read_text())
    metadata.setdefault("reference", str(REFERENCE))
    metadata.setdefault("experiment", "A")
    metadata.setdefault("event_clips", EVENT_CLIPS)
    metadata.setdefault("tracking_focus", [
        dict(frames=[0, 359], seed_target=11),
        dict(frames=[360, 539], seed_target=15),
        dict(frames=[540, 1319], seed_target=10),
    ])
    return metadata


def reference_data(reference=REFERENCE, video=VIDEO):
    report = json.loads((reference / "analysis/report.json").read_text())
    data = analyze.Data(str(reference), report["parameters"], video=str(video))
    if data.input_sha1 != report["inputs"]["sha1"]:
        raise ValueError("Reviewed inputs no longer match the saved analysis")
    return data, report


def prepare(output, experiment="A"):
    """Recompute our full automatic run and preserve all previous inputs."""
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT {output}", flush=True)
    video = PROJECT / f"data/raw/exp{experiment}.mp4"
    reference = PROJECT / f"results/{experiment}_offline"
    _, report = reference_data(reference, video)
    paths = [video, FONT, BOLD_FONT]
    paths += [reference / name for name in (
        "objects.csv", "events.csv", "contours.npz", "background.npy", "stats.json",
        "review.json", "analysis/report.json", "analysis/field_events.csv",
    )]
    paths += [Path(module.__file__) for module in (bt, mb, cm)]
    locked = {str(path): digest(path) for path in paths}
    metadata = dict(
        created=datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
        video=str(video), reference=str(reference), experiment=experiment, source_fps=report["fps"], source_frames=report["frames"],
        input_sha256=locked, thresholds=list(THRESHOLDS), roi_xywh=list(cm.ROI),
        target_object_ids=[o["object_id"] for o in report["main_objects"]],
        threshold_note="100 is the baseline default; 130 comes from the existing README, not a new optimization.",
        reference_note="Reviewed labels only choose initial boxes and interpret outputs. Our displayed IDs and events are automatic.",
    )
    metadata["event_clips"] = list(EVENT_CLIPS)
    metadata["tracking_focus"] = [dict(frames=[0, 359], seed_target=11),
                                  dict(frames=[360, 539], seed_target=15),
                                  dict(frames=[540, 1319], seed_target=10)]
    if experiment == "B":
        metadata["target_object_ids"] = [7, 2, 5, 23]
        metadata["event_clips"] = [
            dict(name="Merger", first=223, last=283, event_frame=253,
                 event_type="merge", target=5, ids=[5, 7], message="Merger: 5 + 7 → 7"),
            dict(name="Separation", first=117, last=177, event_frame=147,
                 event_type="split", target=7, ids=[7, 14], message="Separation: 7 → 7 + 14"),
            dict(name="Disappearance", first=299, last=372, event_frame=312,
                 event_type="lost", target=23, ids=[23], message="Tracking loss: ID 23"),
        ]
        metadata["tracking_focus"] = [dict(frames=[0, 359], seed_target=5),
                                      dict(frames=[360, 779], seed_target=2),
                                      dict(frames=[780, 1319], seed_target=7)]
    save_json(output / "run.json", metadata)
    run_ours(output)


def run_ours(output):
    """Replay the detector and tracker with the frozen experiment background."""
    metadata = load_run(output)
    reference = Path(metadata["reference"])
    video = Path(metadata["video"])
    own_name = "ours_fixed_background" if (output / "ours").exists() else "ours"
    own = output / own_name
    own.mkdir()
    cv2.setNumThreads(4)
    background = np.load(reference / "background.npy")
    parameters = bt.SegParams()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (parameters.close, parameters.close))
    tracker = bt.Tracker(background.shape, bt.TrackParams())
    timings = []
    for frame, gray in bt.iter_frames(str(video)):
        started = time.perf_counter()
        labels, contours, _ = bt.segment(gray, background, parameters, kernel)
        tracker.step(frame, labels, contours)
        timings.append(time.perf_counter()-started)
        if frame % 300 == 0:
            print(f"Our frozen-background replay: frame {frame}", flush=True)
    rows, events, contours, good, total = tracker.finalize()
    for row in rows:
        row["t"] = row["frame"] / metadata["source_fps"]
    for event in events:
        event["t"] = event["frame"] / metadata["source_fps"]
    stats = dict(frames=len(timings), fps=metadata["source_fps"], confirmed_ids=good,
                 raw_ids=total, ms_per_frame_mean=1000*float(np.mean(timings[1:])),
                 background=f"Frozen {reference.name}/background.npy", reviewed_labels_applied=False)
    bt.save_tables(rows, events, str(own))
    bt.save_contours(contours, str(own / "contours.npz"))
    np.save(own / "background.npy", background)
    save_json(own / "stats.json", stats)
    # Existing review IDs can only be used for seed boxes if the automatic replay agrees.
    for filename in ("objects.csv", "events.csv"):
        if read_csv(own / filename) != read_csv(reference / filename):
            raise ValueError(f"Automatic replay differs from the archived {filename}; do not reuse its IDs")
    if not np.array_equal(background, np.load(reference / "background.npy")):
        raise ValueError("Recomputed background differs from the archived run")
    metadata["our_automatic_replay_matches_archive"] = True
    metadata["ours_stats"] = stats
    metadata["ours_directory"] = own_name
    if own_name == "ours_fixed_background":
        metadata["background_rebuild_diagnostic"] = dict(
            directory="ours", used_in_animations=False,
            note="Rebuilding the combined background differed by up to 0.17953491 gray levels "
                 "and changed some detections. Preserved separately; animation uses the frozen background replay.",
        )
    save_json(output / "run.json", metadata)
    print("Automatic run reproduced all archived rows and events", flush=True)


def make_masks(output):
    """Run the existing MobileSAM prompt and refinement procedure once on CPU."""
    metadata = load_run(output)
    reference = Path(metadata["reference"])
    video = Path(metadata["video"])
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    cv2.setNumThreads(4)
    directory = output / "masks"
    directory.mkdir(exist_ok=False)
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Cannot read the first source frame")
    x, y, width, height = cm.ROI
    crop = frame[y:y+height, x:x+width].copy()
    foreground, background = cm.to_roi(cm.FOREGROUND), cm.to_roi(cm.BACKGROUND)
    thresholded = cm.threshold_image(crop)
    cm.flood(thresholded, background, (0, 0, 0))
    cm.flood(thresholded, foreground, (255, 255, 255))
    mask = cm.sam_mask(thresholded, foreground, background, str(MOBILE_SAM))
    ideal = cm.ideal_mask(str(reference / "background.npy"), cm.FOREGROUND)
    cv2.imwrite(str(directory / "first_frame.png"), crop)
    cv2.imwrite(str(directory / "prompt_image.png"), thresholded)
    cv2.imwrite(str(directory / "mask_mobilesam.png"), mask * 255)
    cv2.imwrite(str(directory / "mask_ideal.png"), ideal * 255)
    save_json(directory / "provenance.json", dict(
        roi=cm.ROI, foreground=cm.FOREGROUND, background=cm.BACKGROUND,
        model="MobileSAM vit_t", device="cpu", refinement_loops=25,
        source="https://github.com/ChaoningZhang/MobileSAM",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MOBILE_SAM, text=True).strip(),
        weight_sha256=digest(MOBILE_SAM / "weights/mobile_sam.pt"),
        pixels_inside=int(mask.sum()), ideal_pixels_inside=int(ideal.sum()),
        iou_with_background_mask=float(np.logical_and(mask, ideal).sum()/np.logical_or(mask, ideal).sum()),
        note="MobileSAM substitutes for original SAM ViT-H, as in the current local baseline implementation.",
    ))
    print("Channel masks generated", flush=True)


def baseline_worker(task):
    """Run one independent single-target tracker at one threshold, frame by frame."""
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    output = Path(task["output"])
    mask = cv2.imread(str(output / "masks/mask_mobilesam.png"), cv2.IMREAD_GRAYSCALE) > 0
    tracker = mb.MedanyTracker(mask, threshold=task["threshold"])
    cap = cv2.VideoCapture(task["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, task["first_frame"])
    x0, y0, width, height = cm.ROI
    rows = []
    for frame in range(task["first_frame"], task["stop_frame"]+1):
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode baseline source frame {frame}")
        crop = np.ascontiguousarray(image[y0:y0+height, x0:x0+width])
        started = time.perf_counter()
        if frame == task["first_frame"]:
            result = tracker.start(crop, task["seed_box"])
        else:
            result = tracker.step(crop)
        elapsed = time.perf_counter() - started
        box = result["bbox"]
        rows.append(dict(
            frame=frame, video_s=frame/task["fps"], target_id=task["object_id"],
            threshold=task["threshold"], ok=result["ok"], reinitialized=result["reinit"],
            x_px=result["x"]+x0 if result["ok"] else None,
            y_px=result["y"]+y0 if result["ok"] else None,
            blue_area_px=result["area"], box_x=box[0]+x0 if box else None,
            box_y=box[1]+y0 if box else None, box_width=box[2] if box else None,
            box_height=box[3] if box else None, processing_ms=1000*elapsed,
        ))
    cap.release()
    name = f"target_{task['object_id']:03d}_threshold_{task['threshold']}.csv"
    write_csv(output / "baseline" / name, rows)
    return dict(file=name, target_id=task["object_id"], threshold=task["threshold"],
                frames=len(rows), reported=sum(r["ok"] for r in rows),
                reinitializations=sum(r["reinitialized"] for r in rows))


def run_baseline(output, workers):
    """Run selected reference-seeded targets independently, without shared state."""
    if not hasattr(cv2, "TrackerCSRT_create"):
        raise RuntimeError("Use the isolated OpenCV contrib environment for this phase")
    (output / "baseline").mkdir(exist_ok=False)
    metadata = load_run(output)
    data, report = reference_data(Path(metadata["reference"]), Path(metadata["video"]))
    tasks = []
    for object_id in metadata["target_object_ids"]:
        rows = np.nonzero((data.object_id == object_id) & (data.status == "trusted"))[0]
        first = rows[np.argmin(data.F[rows])]
        last_frame = int(data.F[rows].max())
        contour = data.contour[(int(data.F[first]), int(data.I[first]))] - np.array(cm.ROI[:2])
        box = cv2.boundingRect(contour.astype(np.int32).reshape(-1, 1, 2))
        for threshold in THRESHOLDS:
            tasks.append(dict(output=str(output), video=metadata["video"], object_id=object_id, threshold=threshold,
                              first_frame=int(data.F[first]), stop_frame=min(data.n_frames-1, last_frame+120),
                              seed_box=box, fps=data.fps, last_reference_frame=last_frame))
    save_json(output / "baseline/initialization.json", tasks)
    summaries = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(baseline_worker, task) for task in tasks]
        for future in as_completed(futures):
            summaries.append(future.result())
            print(f"Baseline {len(summaries)}/{len(tasks)}: {summaries[-1]['file']}", flush=True)
    save_json(output / "baseline/summary.json", dict(
        runs=summaries, python=platform.python_version(), opencv=cv2.__version__,
        workers=workers, opencv_threads_per_worker=1,
        note="Independent single-target runs; their labels indicate the requested seed, not proven identity. "
             "Parallel execution times are not a controlled speed comparison.",
    ))


class AnimationData:
    """Load completed outputs; review labels are never substituted for automatic IDs."""

    def __init__(self, output):
        self.output = output
        self.metadata = load_run(output)
        if not self.metadata.get("our_automatic_replay_matches_archive"):
            raise ValueError("The automatic replay has not been verified")
        own = output / self.metadata["ours_directory"]
        self.background = np.load(own / "background.npy")
        self.parameters = bt.SegParams()
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.parameters.close, self.parameters.close)
        )
        self.rows = defaultdict(list)
        self.track_rows = {}
        for row in read_csv(own / "objects.csv"):
            item = {key: float(value) for key, value in row.items()}
            item["frame"], item["id"] = int(item["frame"]), int(item["id"])
            self.rows[item["frame"]].append(item)
            self.track_rows[(item["frame"], item["id"])] = item
        self.contours = {}
        with np.load(own / "contours.npz") as arrays:
            frames, tracks = arrays["frame"], arrays["id"]
            offsets, coordinates = arrays["offset"], arrays["xy"]
            for index, (frame, track) in enumerate(zip(frames, tracks)):
                lo, hi = offsets[index:index+2]
                self.contours[(int(frame), int(track))] = coordinates[lo:hi].copy()
        self.events = read_csv(own / "events.csv")
        for clip in self.metadata["event_clips"]:
            found = any(int(e["frame"]) == clip["event_frame"] and e["type"] == clip["event_type"]
                        and list(map(int, e["ids"].split())) == clip["ids"] for e in self.events)
            if not found:
                raise ValueError(f"Requested event absent in this actual run: {clip}")
        self.baseline = {}
        summaries = json.loads((output / "baseline/summary.json").read_text())
        if len(summaries["runs"]) != len(self.metadata["target_object_ids"]) * len(THRESHOLDS):
            raise ValueError("Some independent baseline runs are missing")
        for summary in summaries["runs"]:
            values = {}
            for row in read_csv(output / "baseline" / summary["file"]):
                item = {k: (float(v) if v else None) for k, v in row.items()
                        if k not in ("ok", "reinitialized")}
                item["ok"] = row["ok"] == "True"
                item["reinitialized"] = row["reinitialized"] == "True"
                item["frame"] = int(item["frame"])
                values[item["frame"]] = item
            self.baseline[(summary["target_id"], summary["threshold"])] = values
        self.mask = cv2.imread(str(output / "masks/mask_mobilesam.png"), cv2.IMREAD_GRAYSCALE) > 0
        self.reference, _ = reference_data(Path(self.metadata["reference"]), Path(self.metadata["video"]))

    def reference_contour(self, frame, target):
        data = self.reference
        for row in data.rows_by_frame.get(frame, []):
            if data.object_id[row] == target and data.status[row] == "trusted":
                return data.contour.get((frame, int(data.I[row])))
        return None


def canvas(design, title, method, panel_titles, frame, playback, metadata):
    image = Image.new("RGB", SIZE, "white")
    design.text(image, (50, 26), title, 40, INK, bold=True)
    design.text(image, (50, 86), method, 25, GREY)
    for box, label in zip(PANELS, panel_titles):
        design.text(image, (box[0], 130), label, 25, INK)
    design.text(image, (50, 944), f"Exp. {metadata['experiment']}  |  {frame/metadata['source_fps']:.3f} s  |  Frame {frame}", 25, INK)
    design.text(image, (1550, 944), playback, 25, GREY, anchor="ra")
    return image


def paste_roi(image, raw, panel):
    """Use the same square source ROI and scale in all six result videos."""
    return paste_view(image, raw, cm.ROI, PANELS[panel])[0]


def colored_detection(raw_roi, mask, color):
    """Show exactly detected pixels; color mixing is display-only."""
    rgb = np.asarray(raw_roi).copy()
    shade = np.asarray(tuple(bytes.fromhex(color.lstrip("#"))), dtype=float)
    rgb[mask] = np.round(.55*rgb[mask] + .45*shade).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, tuple(int(v) for v in shade), 1, cv2.LINE_AA)
    return Image.fromarray(rgb)


def detection_frames(data, design, method):
    source = VideoSource(Path(data.metadata["video"]))
    x, y, width, height = cm.ROI
    title = "Microbubble detection"
    try:
        for frame in range(data.metadata["source_frames"]):
            raw = source.get(frame)
            crop = raw.crop((x, y, x+width, y+height))
            labels = ("Threshold 100", "Threshold 130") if method == "baseline" else ("Raw video", "Our detection")
            description = "Baseline · fixed intensity threshold" if method == "baseline" else "Our method · relative darkness"
            image = canvas(design, title, description, labels, frame, "1× speed", data.metadata)
            if method == "baseline":
                bgr = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
                for panel, threshold in enumerate(THRESHOLDS):
                    mask = mb.blue_mask(data.mask, bgr, threshold)
                    overlay = colored_detection(crop, mask, BLUE)
                    image.paste(overlay.resize((730, 730), Image.Resampling.LANCZOS), PANELS[panel][:2])
            else:
                gray = cv2.cvtColor(np.asarray(raw), cv2.COLOR_RGB2GRAY).astype(np.float32)
                regions, _, _ = bt.segment(gray, data.background, data.parameters, data.kernel)
                mask = regions[y:y+height, x:x+width] > 0
                paste_roi(image, raw, 0)
                overlay = colored_detection(crop, mask, TEAL)
                image.paste(overlay.resize((730, 730), Image.Resampling.LANCZOS), PANELS[1][:2])
            if frame % 300 == 0:
                print(f"Render detection {method}: {frame}", flush=True)
            yield image, 1, dict(source_frame=frame, source_video_s=frame/data.metadata["source_fps"], section="detection")
    finally:
        source.close()


def track_color(track):
    b, g, r = bt._color(track)
    return (r, g, b)


def draw_our_tracks(image, data, design, frame, transform, emphasis=()):
    """Draw only measured rows; never bridge a missing-frame gap with a line."""
    for row in data.rows.get(frame, []):
        track = row["id"]
        color = ORANGE if track in emphasis else track_color(track)
        contour = data.contours[(frame, track)]
        points = transform(contour)
        # Pixel clipping is inherited from the fixed display crop only.
        if not np.any((contour[:, 0] >= cm.ROI[0]) & (contour[:, 0] < cm.ROI[0]+cm.ROI[2])):
            continue
        line(image, points, color, 2, closed=True)
        trail = []
        for previous in range(max(0, frame-29), frame+1):
            old = data.track_rows.get((previous, track))
            if old is None:
                if len(trail) > 1:
                    line(image, transform(trail), color, 2)
                trail = []
            else:
                trail.append((old["cx"], old["cy"]))
        if len(trail) > 1:
            line(image, transform(trail), color, 2)
        px, py = transform([[row["cx"], row["cy"]]])[0]
        if 820 <= px < 1550 and 165 <= py < 895:
            dot(image, (px, py), color, 3)
            design.text(image, (min(1510, px+8), max(167, py-22)), str(track), 21, color, bold=True)


def reference_overlay(image, data, design, frame, target, transform, panel):
    contour = data.reference_contour(frame, target)
    if contour is not None:
        line(image, transform(contour), "white", 2, closed=True)
    x, y, _, _ = PANELS[panel]
    translucent_panel(image, (x+12, y+12, x+310, y+57))
    ImageDraw.Draw(image).line((x+26, y+34, x+57, y+34), fill="white", width=3)
    design.text(image, (x+70, y+19), "Reviewed reference", 22, INK)


def baseline_overlay(image, data, design, frame, target, threshold, transform, panel):
    row = data.baseline[(target, threshold)].get(frame)
    if row is None:
        raise ValueError(f"Missing baseline frame: {target}, {threshold}, {frame}")
    x, y, _, _ = PANELS[panel]
    if row["ok"]:
        left, top, width, height = [row[k] for k in ("box_x", "box_y", "box_width", "box_height")]
        corners = transform([[left, top], [left+width, top+height]])
        # Clip the displayed box to the viewport, never alter stored tracking data.
        lo = np.maximum(corners[0], (x, y))
        hi = np.minimum(corners[1], (x+730, y+730))
        if np.all(hi >= lo):
            ImageDraw.Draw(image).rectangle((*lo, *hi), outline=BLUE, width=3)
        px, py = transform([[row["x_px"], row["y_px"]]])[0]
        if x <= px < x+730 and y <= py < y+730:
            dot(image, (px, py), BLUE, 5)
        state = "Reinitialized" if row["reinitialized"] else "Position reported"
    else:
        state = "Lost"
    translucent_panel(image, (x+12, y+63, x+310, y+107))
    design.text(image, (x+26, y+70), state, 22, BLUE if row["ok"] else ORANGE)


def tracking_frames(data, design, method):
    """Keep a common full 22-s timeline; show three independent baseline runs."""
    source = VideoSource(Path(data.metadata["video"]))
    try:
        for frame in range(data.metadata["source_frames"]):
            target = next(item["seed_target"] for item in data.metadata["tracking_focus"]
                          if item["frames"][0] <= frame <= item["frames"][1])
            raw = source.get(frame)
            labels = ("Threshold 100", "Threshold 130") if method == "baseline" else ("Reviewed reference", "Automatic tracks")
            description = (f"Baseline · independent single-target runs · seed {target}" if method == "baseline"
                           else f"Our method · all automatic IDs · reference object {target}")
            image = canvas(design, "Tracking", description, labels, frame, "1× speed", data.metadata)
            for panel in (0, 1):
                transform = paste_roi(image, raw, panel)
                if method == "baseline":
                    reference_overlay(image, data, design, frame, target, transform, panel)
                    baseline_overlay(image, data, design, frame, target, THRESHOLDS[panel], transform, panel)
                elif panel == 0:
                    reference_overlay(image, data, design, frame, target, transform, panel)
                else:
                    draw_our_tracks(image, data, design, frame, transform)
            if frame % 300 == 0:
                print(f"Render tracking {method}: {frame}", flush=True)
            yield image, 1, dict(source_frame=frame, source_video_s=frame/data.metadata["source_fps"], section="tracking", seed_target=target)
    finally:
        source.close()


def event_frames(data, design, method):
    """Display identical event excerpts; the baseline supplies no semantic labels."""
    source = VideoSource(Path(data.metadata["video"]))
    try:
        for clip in data.metadata["event_clips"]:
            for frame in range(clip["first"], clip["last"]+1):
                raw = source.get(frame)
                labels = ("Threshold 100", "Threshold 130") if method == "baseline" else ("Raw video", "Automatic tracks and events")
                description = (f"Baseline · reference scene: {clip['name'].lower()} · seed {clip['target']}" if method == "baseline"
                               else f"Our method · {clip['name']}")
                image = canvas(design, "Merger, separation, disappearance", description, labels, frame, "0.25× speed", data.metadata)
                for panel in (0, 1):
                    transform = paste_roi(image, raw, panel)
                    if method == "baseline":
                        reference_overlay(image, data, design, frame, clip["target"], transform, panel)
                        baseline_overlay(image, data, design, frame, clip["target"], THRESHOLDS[panel], transform, panel)
                    elif panel == 1:
                        draw_our_tracks(image, data, design, frame, transform, emphasis=clip["ids"])
                reported_frame = clip["event_frame"]
                if clip["event_type"] in ("exit", "lost"):
                    # Tracker emits timeout events nine frames after the last detection,
                    # but stores the last detection frame as the event's frame number.
                    reported_frame += bt.TrackParams().max_gap + 1
                if method == "baseline":
                    note = "No merger, separation or exit labels"
                elif frame >= reported_frame:
                    note = clip["message"]
                else:
                    note = "Awaiting event"
                design.text(image, (820, 906), note, 25, ORANGE if method == "ours" and frame >= reported_frame else GREY)
                yield image, 2, dict(source_frame=frame, source_video_s=frame/data.metadata["source_fps"],
                                     section=clip["name"], seed_target=clip["target"],
                                     event_frame=clip["event_frame"], report_frame=reported_frame)
    finally:
        source.close()


def render(output):
    """Export six movies, verify every frame, then package only reviewable results."""
    data = AnimationData(output)
    design = Design()
    cv2.setNumThreads(4)
    videos = output / "videos"
    videos.mkdir(exist_ok=False)
    checks = output / "checks"
    checks.mkdir(exist_ok=False)
    manifest = []
    definitions = (
        ("01_detection", detection_frames, 60),
        ("02_tracking", tracking_frames, 60),
        ("03_events", event_frames, 30),
    )
    for category, generate, fps in definitions:
        for method in ("baseline", "ours"):
            name = f"{category}_{method}.mp4"
            movie = videos / name
            count = encode_movie(movie, generate(data, design, method), fps=fps)
            result = verify_movie(movie, count, fps=fps, size=SIZE)
            result["file"] = f"videos/{name}"
            manifest.append(result)
            save_json(output / "video_verification.json", manifest)
            print(f"Verified {name}: {count} frames", flush=True)
    unchanged = all(digest(Path(path)) == value for path, value in data.metadata["input_sha256"].items())
    if not unchanged:
        raise ValueError("An original scientific input changed during the task")
    save_json(output / "animation_manifest.json", dict(
        videos=manifest, inputs_unchanged=unchanged, source_roi=cm.ROI,
        event_clips=data.metadata["event_clips"], frame_interpolation=False,
        full_videos_preserve_every_source_frame=True, actual_arial=True,
        fonts={str(p): digest(p) for p in (FONT, BOLD_FONT)},
        renderer_sha256=digest(Path(__file__)),
        baseline_tracking_focus=data.metadata["tracking_focus"],
        ppt_playback_tested=False,
    ))
    save_json(output / "visible_labels.json", sorted(design.labels))
    notes = """# Baseline and our method: result animations

Six videos use the same expA source and fixed square ROI (128, 0, 1024, 1024).
All visible text uses actual Arial. White captions/axes surround the original video.
The legend panels alone are translucent. Inputs and original results are preserved.

## Videos

- 01_detection_baseline: fixed threshold 100 (left) and 130 (right), 22 s, all 1320 source frames.
- 01_detection_ours: raw video (left), freshly recomputed segmentation (right), identical frame times.
- 02_tracking_baseline: threshold 100 / 130; independent single-target runs seeded as object 11 during 0–6 s,
  object 15 during 6–9 s, and object 10 during 9–22 s. Each tracker was actually initialized at its first trusted
  source frame and run continuously; switching the displayed run does not reinitialize it. The white outline
  is the reviewed reference; blue is the actual CSRT box/position. "Position reported" does not mean identity is correct.
- 02_tracking_ours: reviewed reference video (left), all automatic track IDs and recent 0.5-s trails (right).
  Human-linked IDs are not applied to automatic outputs. Missing-frame gaps are not interpolated.
- 03_events_baseline / 03_events_ours: the same merger (frame 201), separation (617) and exit (834) excerpts,
  played at quarter speed by repeating actual frames. The baseline scene name is context, not a baseline event.
  The baseline has no merger/split/exit classifier. Our exit label appears at frame 843, when the 8-frame
  waiting period is exceeded; its stored event frame 834 identifies the last observed position.

## Actual computation

The existing bubble_tracker.segment and Tracker were rerun over every source frame using the frozen
A_offline/background.npy. All resulting rows, IDs and events reproduce the archive exactly. The initial
attempt to rebuild the combined background changed it slightly and is retained under ours/ as a diagnostic;
it is not used in the delivered movies. The accepted outputs are under ours_fixed_background/ for this run.

The existing medany_baseline.MedanyTracker was run independently for all ten reviewed main objects at both
thresholds: 20 complete runs, not a native multi-object baseline. All per-frame outputs, including failed
tracking and reinitializations, are in baseline/. The mask was freshly generated with the existing MobileSAM
25-refinement procedure. Its prompts, model version and weight hash are in masks/provenance.json.
Threshold 130 comes from the existing project documentation; it was not retuned in this task.

Detection colors display detected pixels, not proof that every colored region is a microbubble. The default
fixed threshold, initial prompt boxes and reviewed reference all have the limitations described by the project.
These are offline result visualizations, not independent accuracy scores or controlled speed benchmarks.
No camera or PZT was operated. PowerPoint playback has not been tested in PowerPoint itself.

Reproduce from the project root using customer_workspace.baseline_animations: prepare with the project
Python, masks and baseline with customer_workspace/.venv_baseline312/bin/python, then render with the project
Python. Later phases require --output-dir pointing to the new prepare output. Dependency code is pinned to
the MobileSAM commit in masks/provenance.json; no existing project Python environment was changed.
"""
    if data.metadata["experiment"] == "B":
        notes = """# Experiment B: baseline and our method

Six movies use the same original expB video. All labels use Arial.
The baseline shows thresholds 100 and 130, kept fixed from experiment A.
Four independently seeded targets (7, 2, 5, 23) were run at both thresholds.
White contours are the reviewed reference; blue boxes are baseline outputs.
Our movies show original automatic IDs, not manually linked IDs.

Event excerpts show merge 5 + 7 at frame 253, split 7 + 14 at frame 147,
and loss of ID 23 after its last detection at frame 312. Loss is reported
at frame 321; it does not establish bubble destruction or departure from view.
The baseline does not classify these events. Both exit events in B were
excluded from the illustration because the reviewed targets are artifacts.

Results use the frozen B background and reproduce all archived detector/tracker
rows and events. The MobileSAM channel mask was generated from B's first frame.
These development examples are not independent accuracy scores or speed benchmarks.
No hardware was used. PowerPoint playback has not been tested.
"""
    (output / "README.md").write_text(notes, encoding="utf-8")
    with zipfile.ZipFile(output / "result_animations.zip", "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(videos.iterdir()):
            archive.write(path, f"videos/{path.name}")
        for name in ("README.md", "run.json", "animation_manifest.json", "video_verification.json"):
            archive.write(output / name, name)
    print(f"Completed {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "ours", "masks", "baseline", "render"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--experiment", choices=("A", "B"), default="A")
    args = parser.parse_args()
    if args.output_dir is None and args.phase != "prepare":
        parser.error("--output-dir is required after prepare")
    stamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir or ROOT / f"customer_workspace/validation/baseline_animations_{stamp}"
    if args.phase == "prepare":
        prepare(output, args.experiment)
    elif args.phase == "ours":
        run_ours(output)
    elif args.phase == "masks":
        make_masks(output)
    elif args.phase == "baseline":
        run_baseline(output, args.workers)
    else:
        render(output)


if __name__ == "__main__":
    main()
