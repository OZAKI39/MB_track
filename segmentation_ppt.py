"""Export data-based segmentation figures for slides without changing the tracker.

Run from the project root with ``.venv/bin/python -m
customer_workspace.segmentation_ppt``. Intermediate arrays are captured from the
existing segment() call, so the figures use the production algorithm itself.
Use ``--background-curve`` for the step-1 pixel time series. The default mode
processes only the explicitly selected frame. Original results stay intact.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from PIL import Image, ImageFont

from customer_workspace.microbubble_tracking import bubble_tracker


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "customer_workspace/microbubble_tracking"
INK = "#18283b"
MUTED = "#5c6978"
CYAN = np.array([37, 179, 201], dtype=np.uint8)
AMBER = np.array([250, 183, 61], dtype=np.uint8)
DARK = np.array([16, 25, 39], dtype=np.uint8)
DETAIL_ROI = (460, 390, 380, 380)
CLEANUP_ROIS = ((582, 218, 64, 40), (582, 218, 64, 40), (570, 560, 64, 40))


def sha256(path):
    """Read a file incrementally for provenance and input preservation checks."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture_stages(gray, background, parameters):
    """Read intermediate arrays at segment() return without copying its logic.

    A temporary Python profiling callback observes this one synchronous call.
    The previous callback is restored even on failure. No tracker code, input
    array, tracking state, or scientific parameter is modified.
    """
    captured = {}

    def record(frame, event, argument):
        if event == "return" and frame.f_code is bubble_tracker.segment.__code__:
            for name in ("D", "lo_m", "m", "labels", "fb"):
                captured[name] = frame.f_locals[name].copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (parameters.close, parameters.close)
    )
    previous = sys.getprofile()
    sys.setprofile(record)
    try:
        labels, contours, _ = bubble_tracker.segment(
            gray, background, parameters, kernel
        )
    finally:
        sys.setprofile(previous)

    # A normal call must return exactly the same labels as the observed call.
    plain_labels, _, _ = bubble_tracker.segment(gray, background, parameters, kernel)
    if not np.array_equal(labels, plain_labels):
        raise ValueError("Observed segmentation differs from a normal segment() call")
    stages = {
        "darkness": captured["D"],
        "smoothed_frame": captured["fb"],
        "candidates": captured["D"] > parameters.lo,
        "cores": captured["D"] > parameters.hi,
        "closed_candidates": captured["lo_m"] > 0,
        "core_selected": captured["m"] > 0,
        "final_mask": labels > 0,
    }
    return stages, len(contours)


def crop(array, roi):
    """Use an identical, integer-pixel crop for every panel in a comparison."""
    x, y, width, height = roi
    return array[y:y + height, x:x + width]


def colored_mask(mask, cores=None):
    """Color pixels only for explanation; the exported binary mask stays 0/255."""
    image = np.empty((*mask.shape, 3), dtype=np.uint8)
    image[:] = DARK
    image[mask] = CYAN
    if cores is not None:
        image[mask & cores] = AMBER
    return image


class SlideFigures:
    """Shared Arial typography and minimal layout for the exported images."""

    def __init__(self, output, regular_font, bold_font, source_label):
        self.output = output
        self.source_label = source_label
        self.regular = FontProperties(fname=str(regular_font))
        self.bold = FontProperties(fname=str(bold_font))
        self.saved = []
        self.text_audit = []

    def text(self, figure, x, y, text, size=18, bold=False, **kwargs):
        figure.text(
            x, y, text, fontsize=size, color=INK,
            fontproperties=self.bold if bold else self.regular, **kwargs,
        )

    def new(self, title, subtitle=None):
        figure = plt.figure(figsize=(12.8, 7.2), dpi=250, facecolor="white")
        self.text(figure, 0.035, 0.925, title, 29, True)
        self.text(
            figure, 0.965, 0.94, self.source_label, 11,
            ha="right", va="bottom",
        )
        if subtitle:
            self.text(figure, 0.035, 0.85, subtitle, 16)
        return figure

    def panel(self, figure, bounds, pixels, title=None, **kwargs):
        axes = figure.add_axes(bounds)
        image = axes.imshow(pixels, interpolation="nearest", **kwargs)
        axes.set_axis_off()
        if title:
            self.text(
                figure, bounds[0] + bounds[2] / 2,
                bounds[1] + bounds[3] + 0.035, title, 18, ha="center",
            )
        return image

    def save(self, figure, name, dpi=250, formats=("png", "pdf")):
        figure.canvas.draw()
        labels = []
        for item in figure.findobj(match=Text):
            if item.get_visible() and item.get_text():
                path = font_manager.findfont(item.get_fontproperties())
                if ImageFont.truetype(path, 16).getname()[0] != "Arial":
                    raise ValueError(f"Non-Arial text: {item.get_text()!r}")
                if any("\u4e00" <= ch <= "\u9fff" for ch in item.get_text()):
                    raise ValueError("Chinese text found in an English-only figure")
                labels.append(item.get_text())
        # Outlined SVG text and embedded PDF fonts prevent Arial substitution.
        with matplotlib.rc_context({"svg.fonttype": "path", "pdf.fonttype": 42}):
            for extension in formats:
                path = self.output / f"{name}.{extension}"
                figure.savefig(path, dpi=dpi, facecolor="white")
                self.saved.append(path.name)
        self.text_audit.append({"figure": name, "font_family": "Arial", "labels": labels})
        plt.close(figure)


def render_figures(style, rgb, background, stages, detail_roi=DETAIL_ROI,
                   cleanup_rois=CLEANUP_ROIS):
    """Illustrate the real processing order with full-field and local views."""
    bounds = [(0.035 + 0.32 * i, 0.18, 0.29, 0.52) for i in range(3)]
    figure = style.new(
        "2  Relative darkness",
        "D = 1 - I_s / (B + 1), corrected for overall brightness",
    )
    style.panel(figure, bounds[0], crop(rgb, detail_roi), "Input frame")
    style.panel(
        figure, bounds[1], crop(background, detail_roi), "Background",
        cmap="gray", vmin=0, vmax=255,
    )
    image = style.panel(
        figure, bounds[2], crop(stages["darkness"], detail_roi), "Relative darkness",
        cmap="magma", vmin=0, vmax=0.4,
    )
    color_axes = figure.add_axes((0.675, 0.12, 0.29, 0.022))
    colorbar = figure.colorbar(image, cax=color_axes, orientation="horizontal")
    colorbar.set_ticks([0, 0.06, 0.12, 0.4], labels=["0", "0.06", "0.12", "0.40"])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=12, length=3, color=MUTED)
    for label in colorbar.ax.get_xticklabels():
        label.set_fontproperties(style.regular)
        label.set_fontsize(12)
    style.save(figure, "step_2_relative_darkness")

    figure = style.new("3  Hysteresis thresholding")
    masks = (
        colored_mask(stages["candidates"]),
        colored_mask(stages["closed_candidates"], stages["cores"]),
        colored_mask(stages["core_selected"]),
    )
    titles = ("Candidates > 6%", "Closing + cores > 12%", "Keep regions with a core")
    for box, pixels, title in zip(bounds, masks, titles):
        style.panel(figure, box, crop(pixels, detail_roi), title)
    style.text(figure, 0.5, 0.1, "Cyan: candidates     Gold: deep cores", 15, ha="center")
    style.save(figure, "step_3_hysteresis")

    figure = style.new(
        "4  Closing & cleanup",
        "Closing → core selection → hole filling → area filter",
    )
    before = (stages["candidates"], stages["core_selected"], stages["core_selected"])
    after = (stages["closed_candidates"], stages["final_mask"], stages["final_mask"])
    titles = ("5 × 5 closing", "Fill inner holes", "Remove regions < 12 px")
    for i, (roi, title, first, last) in enumerate(zip(cleanup_rois, titles, before, after)):
        x = 0.08 + 0.305 * i
        style.text(figure, x + 0.13, 0.755, title, 17, ha="center")
        for y, pixels in ((0.445, first), (0.105, last)):
            style.panel(
                figure, (x, y, 0.26, 0.285), crop(pixels, roi),
                cmap="gray", vmin=0, vmax=1,
            )
        if i == 1 and not np.any(crop(last & ~first, roi)):
            style.text(figure, x + 0.13, 0.055, "No holes in this frame", 13, ha="center")
        if i == 2 and not np.any(crop(first & ~last, roi)):
            style.text(figure, x + 0.13, 0.055, "No small regions removed", 13, ha="center")
    style.text(figure, 0.023, 0.59, "Before", 15, rotation=90, va="center")
    style.text(figure, 0.023, 0.25, "After", 15, rotation=90, va="center")
    style.save(figure, "step_4_closing_cleanup")

    figure = style.new("Output  One binary mask per frame")
    style.panel(figure, (0.045, 0.14, 0.43, 0.61), rgb, "Original frame")
    style.panel(
        figure, (0.525, 0.14, 0.43, 0.61), stages["final_mask"], "Binary mask",
        cmap="gray", vmin=0, vmax=1,
    )
    style.save(figure, "output_binary_mask")


def export_background_curve(output, font_dir, experiment="A", pixel=(520, 640)):
    """Plot an observed pixel using the estimator's actual temporal samples.

    Locations were selected from real frames with and without a dark cluster.
    These are illustrative examples, not accuracy benchmarks.
    Spatial blur matches the estimator; no temporal smoothing is applied.
    """
    x, y = pixel
    source = {
        "video": PROJECT / f"data/raw/exp{experiment}.mp4",
        "algorithm": Path(bubble_tracker.__file__).resolve(),
        "renderer": Path(__file__).resolve(),
        "arial_regular": font_dir / "arial.ttf",
        "arial_bold": font_dir / "arialbd.ttf",
    }
    for key in ("arial_regular", "arial_bold"):
        if ImageFont.truetype(str(source[key]), 16).getname()[0] != "Arial":
            raise ValueError(f"Actual Arial required: {source[key]}")
        font_manager.fontManager.addfont(source[key])
    matplotlib.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    inputs = {
        key: {"path": str(path), "sha256": sha256(path)}
        for key, path in source.items()
    }
    fps, n_frames = bubble_tracker.video_info(str(source["video"]))
    radius = 8  # Larger than the sigma=1 Gaussian support; center is unaffected.
    roi = (x - radius, y - radius, 2 * radius + 1, 2 * radius + 1)
    frame_indices, raw_values, blurred_values = [], [], []
    for index, patch in bubble_tracker.iter_frames(str(source["video"]), crop=roi):
        smoothed = cv2.GaussianBlur(patch, (0, 0), 1.0)
        frame_indices.append(index)
        raw_values.append(float(patch[radius, radius]))
        blurred_values.append(float(smoothed[radius, radius]))
    if len(frame_indices) != n_frames:
        raise ValueError("Decoded frame count differs from the video metadata")
    frame_indices = np.asarray(frame_indices)
    blurred_values = np.asarray(blurred_values, dtype=np.float32)
    max_samples = 330
    sample_step = max(1, n_frames // max_samples)
    sampled = frame_indices % sample_step == 0
    # Preserve the estimator's 1/16-gray-level quantization for exact comparison.
    quantized = np.clip(blurred_values * 16 + 0.5, 0, 65535).astype(np.uint16) / 16
    time_s = frame_indices[sampled] / fps
    brightness = quantized[sampled]
    background = bubble_tracker.percentile_background(
        str(source["video"]), crop=roi, percentile=90, max_samples=max_samples,
    )
    background_value = float(background[radius, radius])
    percentile_index = round(0.9 * (len(brightness) - 1))
    if background_value != np.sort(brightness)[percentile_index]:
        raise ValueError("Curve samples disagree with the existing background estimator")

    output.mkdir(parents=True, exist_ok=False)
    style = SlideFigures(
        output, source["arial_regular"], source["arial_bold"],
        f"exp{experiment} · Pixel ({x}, {y})",
    )
    # Publication layout: a 180-mm-wide figure with text sized for final print.
    figure = plt.figure(figsize=(7.1, 4.1), dpi=150, facecolor="white")
    axes = figure.add_axes((0.13, 0.16, 0.84, 0.73), facecolor="white")
    axes.plot(
        time_s, brightness, color="#0072B2", linewidth=1.0, zorder=3,
        label="Pixel brightness",
    )
    axes.axhline(
        background_value, color="#222222", linewidth=1.0,
        linestyle=(0, (5, 3)), zorder=4,
        label="90th percentile = background",
    )
    legend_font = style.regular.copy()
    legend_font.set_size(9)
    axes.legend(
        loc="upper center", ncol=2, frameon=False, prop=legend_font,
        handlelength=3, columnspacing=1.8, borderaxespad=0.7,
    )
    axes.set_title(
        f"Exp. {experiment} · Pixel ({x}, {y})", loc="right", fontsize=9,
        fontproperties=style.regular, color="#222222", pad=8,
    )
    passage_window = time_s < 8 if experiment == "A" else (time_s > 10) & (time_s < 13)
    passage_index = np.argmin(np.where(passage_window, brightness, np.inf))
    axes.annotate(
        "Microbubble passage",
        xy=(time_s[passage_index], brightness[passage_index]),
        xytext=(5.9, 105) if experiment == "A" else
        (float(time_s[passage_index]) + 2, float(brightness[passage_index]) + 12),
        fontsize=9, color="#222222",
        fontproperties=style.regular,
        arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 0.8},
    )
    axes.set_xlim(0, n_frames / fps)
    if experiment == "A":
        axes.set_ylim(80, 165)
    else:
        axes.set_ylim(np.floor(brightness.min() / 20) * 20 - 5,
                      np.ceil(brightness.max() / 20) * 20 + 5)
    axes.set_xticks([0, 5, 10, 15, 20])
    if experiment == "A":
        axes.set_yticks([80, 100, 120, 140, 160])
    axes.xaxis.set_minor_locator(AutoMinorLocator(2))
    axes.yaxis.set_minor_locator(AutoMinorLocator(2))
    axes.set_xlabel("Time (s)", fontsize=10, fontproperties=style.regular, labelpad=7)
    axes.set_ylabel(
        "Pixel brightness (gray level)", fontsize=10,
        fontproperties=style.regular, labelpad=8,
    )
    for spine in axes.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    axes.tick_params(
        which="both", direction="in", top=True, right=True,
        colors="black", width=0.8, pad=6,
    )
    axes.tick_params(which="major", length=4)
    axes.tick_params(which="minor", length=2)
    for axis in (axes.xaxis, axes.yaxis):
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            for label in (tick.label1, tick.label2):
                label.set_fontproperties(style.regular)
                label.set_fontsize(9)
    style.save(
        figure, "step_1_background_estimation", dpi=600,
        formats=("png", "pdf", "svg"),
    )

    with (output / "pixel_brightness.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "frame", "video_time_s", "raw_gray", "smoothed_gray",
            "quantized_gray", "used_for_background_and_curve",
        ])
        for index in frame_indices:
            writer.writerow([
                index, index / fps, raw_values[index], float(blurred_values[index]),
                float(quantized[index]), int(sampled[index]),
            ])
    for key, path in source.items():
        if sha256(path) != inputs[key]["sha256"]:
            raise ValueError(f"Source changed while rendering: {path}")
    report = {
        "inputs": inputs, "pixel_xy": [x, y], "pixel_origin": "zero-based top-left",
        "fps": fps, "n_frames": n_frames, "sample_step_frames": sample_step,
        "sample_count": int(sampled.sum()), "percentile": 90,
        "percentile_index": percentile_index, "background_gray": background_value,
        "checks": {"matches_existing_percentile_background": True,
                   "all_visible_labels_Arial": True, "inputs_unchanged": True},
        "figures": style.saved, "text_audit": style.text_audit,
        "figure_style": {
            "width_mm": 180.34, "height_mm": 104.14, "raster_dpi": 600,
            "font": "Arial", "axis_label_pt": 10, "tick_label_pt": 9,
            "curve_linewidth_pt": 1.0, "background": "white",
            "pdf_fonts_embedded": True, "svg_text_as_paths": True,
        },
        "notes": [
            f"Real exp{experiment} pixel ({x}, {y}); an illustrative passage, not an accuracy benchmark.",
            "Curve uses the exact 330 time samples and 1/16-gray-level quantization of the estimator.",
            "Spatial Gaussian sigma=1 preprocessing matches the existing algorithm; no temporal smoothing.",
            "Shows the single-video percentile stage, before reference alignment and maximum combination.",
            f"The line is not asserted to equal the final multi-reference {experiment}_offline/background.npy pixel.",
            "Times are frame index / video FPS, not measured exposure or AFG timing.",
        ],
    }
    report["outputs"] = {
        path.name: sha256(path) for path in output.iterdir() if path.is_file()
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "background_gray": background_value,
                      "samples": int(sampled.sum()), "checks": report["checks"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--experiment", choices=("A", "B"), default="A")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--font-dir", type=Path, default=Path("/mnt/c/Windows/Fonts"))
    parser.add_argument(
        "--background-curve", action="store_true",
        help="Export the step-1 curve from the selected experiment's measured pixel",
    )
    args = parser.parse_args()
    if args.experiment == "A" and args.frame != 0:
        parser.error("These verified demonstration crops are specific to frame 0.")
    timestamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir or ROOT / "customer_workspace/validation" / f"segmentation_slides_{timestamp}"
    if args.background_curve:
        pixel = (520, 640) if args.experiment == "A" else (438, 780)
        export_background_curve(output, args.font_dir, args.experiment, pixel)
        return
    source = {
        "video": PROJECT / f"data/raw/exp{args.experiment}.mp4",
        "background": PROJECT / f"results/{args.experiment}_offline/background.npy",
        "stats": PROJECT / f"results/{args.experiment}_offline/stats.json",
        "algorithm": Path(bubble_tracker.__file__).resolve(),
        "renderer": Path(__file__).resolve(),
        "arial_regular": args.font_dir / "arial.ttf",
        "arial_bold": args.font_dir / "arialbd.ttf",
    }
    for key in ("arial_regular", "arial_bold"):
        if ImageFont.truetype(str(source[key]), 16).getname()[0] != "Arial":
            raise ValueError(f"Actual Arial required: {source[key]}")
        font_manager.fontManager.addfont(source[key])
    matplotlib.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    inputs = {key: {"path": str(path), "sha256": sha256(path)} for key, path in source.items()}
    stats = json.loads(source["stats"].read_text())
    if stats["video"] != f"exp{args.experiment}.mp4" or stats["mode"] != "offline" or stats["params"]["crop"]:
        raise ValueError("Expected the selected experiment's full-frame offline background")
    parameters = bubble_tracker.SegParams(
        lo=stats["params"]["lo"], hi=stats["params"]["hi"],
        min_area=stats["params"]["min_area"],
    )
    _, bgr = next(bubble_tracker.iter_frames(
        str(source["video"]), start=args.frame, stop=args.frame+1, color=True
    ))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = np.load(source["background"], allow_pickle=False)
    if background.shape != gray.shape:
        raise ValueError("Background and frame dimensions differ")
    stages, regions = capture_stages(gray, background, parameters)
    changes = (
        stages["closed_candidates"] & ~stages["candidates"],
        stages["final_mask"] & ~stages["core_selected"],
        stages["core_selected"] & ~stages["final_mask"],
    )
    detail_roi, cleanup_rois = DETAIL_ROI, CLEANUP_ROIS
    if args.experiment == "B":
        detail_roi = (300, 600, 380, 380)
        cleanup_rois = []
        for changed in changes:
            if not np.any(changed):
                cleanup_rois.append(cleanup_rois[0])
                continue
            _, _, components, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8))
            x, y, width, height, _ = components[1 + np.argmax(components[1:, 4])]
            height = max(40, int(max(width, height) + 24))
            width = int(np.ceil(height * 1.6))
            cleanup_rois.append((int(np.clip(x-width//2, 0, gray.shape[1]-width)),
                                 int(np.clip(y-height//2, 0, gray.shape[0]-height)), width, height))
    crop_changes = [int(crop(mask, roi).sum()) for mask, roi in zip(changes, cleanup_rois)]
    if args.experiment == "A" and not all(crop_changes):
        raise ValueError("A selected crop no longer demonstrates its stated operation")
    output.mkdir(parents=True, exist_ok=False)
    style = SlideFigures(output, source["arial_regular"], source["arial_bold"],
                         f"exp{args.experiment} · Frame {args.frame}")
    render_figures(style, rgb, background, stages, detail_roi, cleanup_rois)
    data = output / "data"
    data.mkdir()
    Image.fromarray(rgb).save(data / f"frame_{args.frame:04d}.png")
    mask_pixels = stages["final_mask"].astype(np.uint8) * 255
    Image.fromarray(mask_pixels).save(data / f"mask_{args.frame:04d}.png")
    np.savez_compressed(data / f"stages_{args.frame:04d}.npz", **stages)
    if not np.array_equal(np.asarray(Image.open(data / f"mask_{args.frame:04d}.png")), mask_pixels):
        raise ValueError("Saved binary mask differs from segment() output")
    for key, path in source.items():
        if sha256(path) != inputs[key]["sha256"]:
            raise ValueError(f"Source changed while rendering: {path}")
    report = {
        "inputs": inputs, "frame_index": args.frame, "video_time_s": args.frame/stats["fps"],
        "frame_shape": list(gray.shape), "parameters": asdict(parameters),
        "detail_roi_xywh": detail_roi, "cleanup_rois_xywh": cleanup_rois,
        "changed_pixels_in_cleanup_crops": crop_changes,
        "segmentation_regions": regions,
        "checks": {"exact_match_to_normal_segment_call": True,
                   "binary_png_matches_segment_output": True,
                   "all_visible_labels_Arial": True, "inputs_unchanged": True},
        "pixel_counts": {name: int(array.sum()) for name, array in stages.items() if array.dtype == bool},
        "figures": style.saved, "text_audit": style.text_audit,
        "notes": [
            "Actual segment() locals captured; no duplicated segmentation implementation.",
            "Blur sigma and closing size use current SegParams defaults; thresholds and minimum area come from saved stats.",
            "D = 1 - GaussianBlur(gray)/(B+1); subtract median(D[::8, ::8]).",
            "Closing precedes deep-core selection in the actual implementation.",
            "External contours fill holes; a contour-area prefilter precedes the final pixel-area filter.",
            "Step 4 uses actual changed pixels. If this frame has no holes, its unchanged hole-filling panel is labelled explicitly.",
            "Step 2 color scale is clipped to [0, 0.4] for display only; all numerical arrays remain unchanged.",
            "This is a pre-tracking segmentation mask, not manually reviewed ground truth.",
            "No physical length calibration is added. Crops are enlarged with nearest-neighbor display.",
        ],
    }
    report["outputs"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "figures": style.saved, "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
