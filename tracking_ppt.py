"""Create tracking-method slide figures from an exact offline experiment replay.

The existing segmenter and Tracker are called directly. Original videos,
results, annotations and algorithms are read only. Run from the project root
using ``.venv/bin/python -m customer_workspace.tracking_ppt``.
"""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import zipfile
from zoneinfo import ZoneInfo

import cv2
import matplotlib
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
import numpy as np
from PIL import ImageFont

from customer_workspace.microbubble_tracking import bubble_tracker
from customer_workspace.segmentation_ppt import PROJECT, ROOT, SlideFigures, sha256


SNAPSHOT_FRAMES = (195, 198, 200, 201, 250, 260, 428, 617, 730, 731, 745, 995)
COLORS = {1: "#a56bcc", 5: "#cb9d14", 6: "#20a986", 8: "#ee9c35",
          11: "#00a9cf", 15: "#df648b", 21: "#7773d9", 67: "#ef9843"}


def track_color(track_id):
    """Keep established colors while supporting IDs from other recordings."""
    palette = ("#24a5bd", "#ee9c35", "#20a986", "#a56bcc", "#cb9d14")
    return COLORS.get(track_id, palette[track_id % len(palette)])


def figure_examples(evidence):
    """Choose actual archived events; never reuse experiment A IDs for B."""
    if evidence.get("experiment", "A") == "A":
        return dict(prediction=(745, 11), matrix_frame=201, matrix_ids=[6, 8, 11],
                    matrix_hosts=[6, 11], merge=(201, [8, 11], [11]),
                    split=(617, [11], [11, 67]), distance=(731, 1),
                    output_frame=250, output_ids=[5, 6, 11, 15, 21])
    choices = []
    for frame, snap in evidence["snapshots"].items():
        for region, owner in snap["owner"].items():
            if owner not in snap["active_ids"] or owner not in evidence["remap"]:
                continue
            row = snap["active_ids"].index(owner)
            if not snap["sig"][row, region]:
                point = np.array([snap["feats"][region]["cx"], snap["feats"][region]["cy"]])
                distance = float(np.linalg.norm(snap["pred"][row] - point))
                choices.append((distance, frame, evidence["remap"][owner]))
    if not choices:
        raise ValueError("No measured distance-match example in selected frames")
    _, frame, track = max(choices)
    return dict(prediction=(260, 2), matrix_frame=253, matrix_ids=[4, 5, 7, 15],
                matrix_hosts=[4, 7], merge=(253, [5, 7, 15], [7]),
                split=(147, [7], [7, 14]), distance=(frame, track),
                output_frame=260, output_ids=[2, 7, 4, 11, 8])


def raw_id(evidence, display_id):
    """Map the saved CSV ID back to the live ID before short-track filtering."""
    return next(old for old, new in evidence["remap"].items() if new == display_id)


def previous_mask(evidence, snapshot, display_id, predicted=False):
    """Display the original mask or the exact integer shift used by Tracker."""
    track = snapshot["previous"][raw_id(evidence, display_id)]
    yy, xx = track["ys"], track["xs"]
    shape = snapshot["labels"].shape
    if predicted:
        gap = snapshot["frame"] - track["last"]
        yy = np.clip(np.round(yy + track["vy"] * gap).astype(int), 0, shape[0] - 1)
        xx = np.clip(np.round(xx + track["vx"] * gap).astype(int), 0, shape[1] - 1)
    mask = np.zeros(shape, bool)
    mask[yy, xx] = True
    return mask


def current_mask(evidence, snapshot, display_id):
    original_id = raw_id(evidence, display_id)
    region = next(j for j, owner in snapshot["owner"].items() if owner == original_id)
    return snapshot["labels"] == region


def mask_roi(masks, padding=22):
    """Square crop shared by before/after panels; no image stretching."""
    union = np.logical_or.reduce(masks)
    ys, xs = np.nonzero(union)
    side = max(int(np.ptp(xs)), int(np.ptp(ys))) + 2 * padding + 1
    x = max(0, min(union.shape[1] - side, int((xs.min() + xs.max() - side) / 2)))
    y = max(0, min(union.shape[0] - side, int((ys.min() + ys.max() - side) / 2)))
    return x, y, side, side


def image_axes(style, figure, bounds, rgb, roi, title):
    x, y, width, height = roi
    axes = figure.add_axes(bounds)
    axes.imshow(rgb, interpolation="nearest")
    axes.set_xlim(x - 0.5, x + width - 0.5)
    axes.set_ylim(y + height - 0.5, y - 0.5)
    axes.set_axis_off()
    style.text(figure, bounds[0] + bounds[2] / 2,
               bounds[1] + bounds[3] + 0.025, title, 16, ha="center")
    return axes


def draw_mask(style, axes, mask, label, color, offset=(0, 0), dashed=False, fill=True):
    """Overlay measured pixels; every label is rendered by Matplotlib in Arial."""
    if fill:
        rgba = np.zeros((*mask.shape, 4), np.float32)
        rgba[mask] = to_rgba(color, 0.18)
        axes.imshow(rgba, interpolation="nearest")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        points = np.vstack((contour[:, 0], contour[0, 0]))
        axes.plot(points[:, 0], points[:, 1], color=color, linewidth=1.4,
                  linestyle="--" if dashed else "-")
    if label:
        ys, xs = np.nonzero(mask)
        axes.annotate(
            label, xy=(xs.mean(), ys.mean()),
            xytext=(xs.mean() + offset[0], ys.mean() + offset[1]),
            ha="center", va="center", fontsize=14, fontproperties=style.bold,
            color=color, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
            arrowprops={"arrowstyle": "-", "color": color} if offset != (0, 0) else None,
        )


def matrix_example(evidence):
    """Extract an independent track/region block from the real full matrix."""
    example = evidence["examples"]
    snapshot = evidence["snapshots"][example["matrix_frame"]]
    ids = example["matrix_ids"]
    rows = [snapshot["active_ids"].index(raw_id(evidence, i)) for i in ids]
    columns = [next(j for j, owner in snapshot["owner"].items()
                    if owner == raw_id(evidence, i)) for i in example["matrix_hosts"]]
    matrix = snapshot["O"][np.ix_(rows, columns)]
    other_rows = [a for a in range(len(snapshot["active_ids"])) if a not in rows]
    other_columns = [j for j in range(snapshot["O"].shape[1]) if j not in columns]
    if snapshot["sig"][np.ix_(rows, other_columns)].any() or snapshot["sig"][np.ix_(other_rows, columns)].any():
        raise ValueError("Chosen matrix block has eligible matches outside the displayed example")
    # Reuse the same solver and cost matrix as the production implementation.
    rr, cc = bubble_tracker.linear_sum_assignment(
        np.where(snapshot["sig"], -snapshot["O"].astype(float), 1e9)
    )
    chosen = [(rows.index(int(a)), columns.index(int(j))) for a, j in zip(rr, cc)
              if a in rows and j in columns and snapshot["sig"][a, j]]
    if evidence.get("experiment", "A") == "A" and chosen != [(0, 0), (2, 1)]:
        raise ValueError("The verified matching example has changed")
    for a in rows:
        for j in columns:
            display_id = evidence["remap"][snapshot["active_ids"][a]]
            predicted = previous_mask(evidence, snapshot, display_id, predicted=True)
            if np.count_nonzero(predicted & (snapshot["labels"] == j)) != snapshot["O"][a, j]:
                raise ValueError("Displayed predicted mask does not reproduce recorded overlap")
    return snapshot, ids, columns, matrix, chosen


def heatmap(style, figure, bounds, matrix, ids=(6, 8, 11), chosen=()):
    axes = figure.add_axes(bounds)
    axes.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max(), aspect="auto")
    axes.set_xticks(range(matrix.shape[1]), [f"R{i+1}" for i in range(matrix.shape[1])])
    axes.set_yticks(range(len(ids)), [f"ID {i}" for i in ids])
    axes.xaxis.tick_top()
    axes.tick_params(length=0, pad=9)
    for text in axes.get_xticklabels() + axes.get_yticklabels():
        text.set_fontproperties(style.regular)
        text.set_fontsize(16)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            axes.text(c, r, str(int(matrix[r, c])), ha="center", va="center",
                      fontsize=21, fontproperties=style.bold,
                      color="white" if matrix[r, c] > matrix.max() * 0.6 else "#18283b")
    for spine in axes.spines.values():
        spine.set_visible(False)
    if chosen:
        for r, c in chosen:
            axes.add_patch(Rectangle((c - 0.47, r - 0.47), 0.94, 0.94,
                                    fill=False, edgecolor="#ef9d26", linewidth=3))
    return axes


def render_prediction(style, evidence):
    frame, target = evidence["examples"]["prediction"]
    snapshot = evidence["snapshots"][frame]
    track = snapshot["previous"][raw_id(evidence, target)]
    before = previous_mask(evidence, snapshot, target)
    predicted = previous_mask(evidence, snapshot, target, predicted=True)
    current = current_mask(evidence, snapshot, target)
    roi = mask_roi([before, predicted, current])
    style.source_label = f"exp{evidence['experiment']} · Frames {frame-1}–{frame}"
    figure = style.new("1  Motion prediction")
    panels = [(0.035 + 0.32 * i, 0.23, 0.29, 0.52) for i in range(3)]
    axes = image_axes(style, figure, panels[0], snapshot["previous_rgb"], roi, f"Previous mask · ID {target}")
    draw_mask(style, axes, before, "", "#00a9cf")
    axes = image_axes(style, figure, panels[1], snapshot["rgb"], roi, "Shifted mask")
    draw_mask(style, axes, before, "", "#f8f8f8", dashed=True, fill=False)
    draw_mask(style, axes, predicted, "", "#00a9cf")
    axes.annotate("", xy=(track["cx"] + track["vx"], track["cy"] + track["vy"]),
                  xytext=(track["cx"], track["cy"]),
                  arrowprops={"arrowstyle": "->", "color": "#ee9c35", "lw": 2})
    axes = image_axes(style, figure, panels[2], snapshot["rgb"], roi, "Prediction vs. new region")
    draw_mask(style, axes, current, "", "#ee9c35")
    draw_mask(style, axes, predicted, "", "#00a9cf", dashed=True, fill=False)
    style.text(figure, 0.5, 0.14,
               f"Shift: dx = {track['vx']:+.2f} px, dy = {track['vy']:+.2f} px", 17, ha="center")
    style.text(figure, 0.5, 0.08, "Cyan: predicted mask     Gold: current region", 14, ha="center")
    style.save(figure, "step_1_motion_prediction")
    return {"frame": frame, "id": target, "dx_px": track["vx"], "dy_px": track["vy"], "roi_xywh": roi}


def render_overlap(style, evidence, example):
    snapshot, ids, columns, matrix, _ = example
    predicted = [previous_mask(evidence, snapshot, i, predicted=True) for i in ids]
    current = [snapshot["labels"] == j for j in columns]
    roi = mask_roi(predicted + current, padding=20)
    style.source_label = f"exp{evidence['experiment']} · Frames {snapshot['frame']-1}–{snapshot['frame']}"
    figure = style.new("2  Overlap matrix")
    boxes = [(0.035, 0.22, 0.285, 0.51), (0.35, 0.22, 0.285, 0.51)]
    axes = image_axes(style, figure, boxes[0], snapshot["rgb"], roi, "Predicted tracks")
    for i, mask in zip(ids, predicted):
        offset = (0, 0)
        if evidence["experiment"] == "B":
            offset = {5: (-18, -12), 7: (0, -25), 15: (24, 0)}.get(i, offset)
        draw_mask(style, axes, mask, f"ID {i}", track_color(i), offset=offset)
    axes = image_axes(style, figure, boxes[1], snapshot["rgb"], roi, "Current regions")
    for i, mask in enumerate(current):
        draw_mask(style, axes, mask, f"R{i + 1}", track_color(ids[min(i, len(ids)-1)]))
    heatmap(style, figure, (0.755, 0.25, 0.205, 0.43), matrix, ids=ids)
    style.text(figure, 0.84, 0.755, "Overlap (pixels)", 16, ha="center")
    style.save(figure, "step_2_overlap_matrix")
    return roi


def render_assignment(style, example, evidence):
    snapshot, ids, _, matrix, chosen = example
    style.source_label = f"exp{evidence['experiment']} · Frames {snapshot['frame']-1}–{snapshot['frame']}"
    figure = style.new("3  Hungarian assignment", "One-to-one pairing with maximum total overlap")
    heatmap(style, figure, (0.13, 0.23, 0.30, 0.48), matrix, ids=ids, chosen=chosen)
    axes = figure.add_axes((0.51, 0.20, 0.46, 0.54))
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_axis_off()
    positions = np.linspace(.86, .14, len(ids))
    for i, y in zip(ids, positions):
        axes.text(0.06, y, f"ID {i}", fontsize=21, color=track_color(i),
                  fontproperties=style.bold, ha="center", va="center")
    for r, c in chosen:
        y = positions[r]
        axes.annotate("", xy=(0.56, y), xytext=(0.22, y),
                      arrowprops={"arrowstyle": "->", "color": track_color(ids[r]), "lw": 2.2})
        axes.text(0.79, y, f"R{c + 1} → ID {ids[r]}", fontsize=20,
                  fontproperties=style.bold, ha="center", va="center", color="#18283b")
    for r in range(len(ids)):
        if r not in [pair[0] for pair in chosen]:
            axes.text(0.42, positions[r], "Unassigned", fontsize=15, color="#747e88",
                      fontproperties=style.regular, va="center")
    total = int(sum(matrix[r, c] for r, c in chosen))
    style.text(figure, 0.5, 0.11, f"Shown matches: {total:,} overlapping pixels", 17, ha="center")
    style.save(figure, "step_3_hungarian_assignment")


def render_events(style, evidence):
    style.source_label = f"exp{evidence['experiment']}"
    figure = style.new("4  Merge, split & distance matching")
    distance_frame, distance_track = evidence["examples"]["distance"]
    distance_snapshot = evidence["snapshots"][distance_frame]
    distance_raw_id = raw_id(evidence, distance_track)
    a = distance_snapshot["active_ids"].index(distance_raw_id)
    j = next(j for j, owner in distance_snapshot["owner"].items() if owner == distance_raw_id)
    feature = distance_snapshot["feats"][j]
    distance = float(np.linalg.norm(distance_snapshot["pred"][a] - [feature["cx"], feature["cy"]]))
    overlap = int(distance_snapshot["O"][a, j])
    if distance_snapshot["sig"][a, j]:
        raise ValueError("The distance example now passes the overlap gate")
    merge_frame, old_merge, new_merge = evidence["examples"]["merge"]
    split_frame, old_split, new_split = evidence["examples"]["split"]
    cases = [(merge_frame, old_merge, new_merge, "Merge", f"Keep matched ID {new_merge[0]}"),
             (split_frame, old_split, new_split, "Split", f"New ID {new_split[-1]} · parent {old_split[0]}"),
             (distance_frame, [distance_track], [distance_track], "Distance match",
              f"{overlap} px overlap · {distance:.2f} px distance")]
    rois = []
    for column, (frame, old_ids, new_ids, title, caption) in enumerate(cases):
        snapshot = evidence["snapshots"][frame]
        is_distance = column == 2
        old_masks = [previous_mask(evidence, snapshot, i, predicted=is_distance) for i in old_ids]
        new_masks = [current_mask(evidence, snapshot, i) for i in new_ids]
        roi = mask_roi(old_masks + new_masks, padding=13 if is_distance else 15)
        rois.append(roi)
        x = 0.055 + 0.315 * column
        style.text(figure, x + 0.13, 0.835, title, 19, True, ha="center")
        for row, (masks, ids) in enumerate(((old_masks, old_ids), (new_masks, new_ids))):
            box = (x, 0.475 if row == 0 else 0.125, 0.26, 0.27)
            picture = snapshot["rgb"] if row or is_distance else snapshot["previous_rgb"]
            label = f"Predicted · {frame}" if is_distance and not row else f"Frame {frame - 1 + row}"
            axes = image_axes(style, figure, box, picture, roi, label)
            for i, mask in zip(ids, masks):
                offset = (20, -16) if i == 67 else (0, 0)
                draw_mask(style, axes, mask, str(i), track_color(i), offset=offset,
                          dashed=is_distance and not row)
        style.text(figure, x + 0.13, 0.055, caption, 14, ha="center")
    style.save(figure, "step_4_merge_split_distance")
    return {"rois_xywh": rois, "distance_frame": distance_frame, "distance_track_id": distance_track,
            "overlap_px": overlap, "predicted_to_detected_distance_px": distance}


def render_output(style, evidence):
    frame = evidence["examples"]["output_frame"]
    snapshot = evidence["snapshots"][frame]
    style.source_label = f"exp{evidence['experiment']} · Frame {frame}"
    figure = style.new("Output  IDs, trajectories & track filtering")
    axes = image_axes(style, figure, (0.035, 0.14, 0.49, 0.66), snapshot["rgb"],
                      (0, 0, 1280, 1024), "Tracked regions")
    for i in evidence["examples"]["output_ids"]:
        mask = current_mask(evidence, snapshot, i)
        draw_mask(style, axes, mask, str(i), track_color(i), offset=(30, -25))
        tail = [r for r in evidence["rows"] if r["id"] == i and frame-70 <= r["frame"] <= frame]
        axes.plot([r["cx"] for r in tail], [r["cy"] for r in tail], color=track_color(i), linewidth=1.8)
    style.text(figure, 0.59, 0.76, "Max matching gap: 8 frames", 19)
    style.text(figure, 0.59, 0.685, "Keep ≥ 5 observations", 19)
    discarded = evidence["raw_tracks"] - evidence["kept_tracks"]
    style.text(figure, 0.59, 0.61,
               f"{evidence['kept_tracks']} tracks kept · {discarded} short tracks removed", 14)
    style.text(figure, 0.59, 0.49, "Recorded processing time", 19, True)
    stats = evidence["stats"]
    plot = figure.add_axes((0.65, 0.19, 0.29, 0.23))
    times = [stats["ms_per_frame_mean"], stats["ms_per_frame_p95"]]
    plot.barh([1, 0], times, color=["#24a5bd", "#78bdcc"], height=0.45)
    plot.set_yticks([1, 0], ["Mean", "P95"])
    plot.set_xticks([0, 5, 10, 15, 20])
    plot.set_xlim(0, 20)
    plot.set_ylim(-0.5, 1.65)
    for y, value in zip((1, 0), times):
        plot.text(value - 0.35, y, f"{value:.2f}", ha="right", va="center",
                  fontsize=13, fontproperties=style.bold, color="#163347")
    interval = 1000 / stats["fps"]
    plot.axvline(interval, ls="--", color="#b98220", linewidth=1.4)
    plot.text(interval, 1.54, "60 fps: 16.67 ms", fontsize=12, ha="center",
              color="#94660f", fontproperties=style.regular, clip_on=False)
    plot.set_xlabel("ms / frame", fontproperties=style.regular, fontsize=14)
    plot.tick_params(length=0, pad=7)
    for text in plot.get_xticklabels() + plot.get_yticklabels():
        text.set_fontproperties(style.regular)
        text.set_fontsize(13)
    for spine in plot.spines.values():
        spine.set_visible(False)
    style.text(figure, 0.775, 0.09, "Segmentation + tracking only", 13, ha="center")
    style.save(figure, "output_tracking")


def export_figures(evidence, output, font_dir=Path("/mnt/c/Windows/Fonts")):
    """Render verified replay data and save the exact arrays behind the figures."""
    for key, path in evidence["source_paths"].items():
        if sha256(Path(path)) != evidence["source_hashes"][key]:
            raise ValueError(f"Evidence no longer matches source: {path}")
    regular, bold = font_dir / "arial.ttf", font_dir / "arialbd.ttf"
    for path in (regular, bold):
        if ImageFont.truetype(str(path), 16).getname()[0] != "Arial":
            raise ValueError(f"Actual Arial required: {path}")
        font_manager.fontManager.addfont(path)
    matplotlib.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    evidence["examples"] = figure_examples(evidence)
    ex = evidence["examples"]
    required_events = [(ex["merge"][0], "merge", ex["merge"][1][:1] + ex["merge"][2]),
                       (ex["split"][0], "split", ex["split"][2])]
    for frame, kind, ids in required_events:
        if not any(e["frame"] == frame and e["type"] == kind and e["ids"] == ids
                   for e in evidence["events"]):
            raise ValueError(f"Required recorded event is absent: {frame}, {kind}")
    example = matrix_example(evidence)
    output.mkdir(parents=True, exist_ok=False)
    data = output / "data"
    data.mkdir()
    style = SlideFigures(output, regular, bold, f"exp{evidence['experiment']}")
    prediction = render_prediction(style, evidence)
    overlap_roi = render_overlap(style, evidence, example)
    render_assignment(style, example, evidence)
    event_details = render_events(style, evidence)
    render_output(style, evidence)

    snapshot, ids, columns, matrix, chosen = example
    arrays = {"overlap_matrix_full": snapshot["O"], "eligible_full": snapshot["sig"],
              "overlap_matrix_shown": matrix, "predicted_centroids": snapshot["pred"]}
    for frame in {ex["matrix_frame"], ex["split"][0], ex["distance"][0], ex["prediction"][0]}:
        arrays[f"current_labels_{frame}"] = evidence["snapshots"][frame]["labels"]
    for i in ids:
        arrays[f"predicted_mask_{ex['matrix_frame']}_id_{i}"] = previous_mask(evidence, snapshot, i, predicted=True)
    motion_frame, motion_track = ex["prediction"]
    motion = evidence["snapshots"][motion_frame]
    arrays["previous_motion_mask"] = previous_mask(evidence, motion, motion_track)
    arrays["predicted_motion_mask"] = previous_mask(evidence, motion, motion_track, predicted=True)
    np.savez_compressed(data / "tracking_arrays.npz", **arrays)
    with (data / "overlap_matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["track_id", "R1", "R2"])
        for i, row in zip(ids, matrix):
            writer.writerow([i, *map(int, row)])
    with (data / "trajectory_excerpt.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        keys = ["frame", "id", "cx", "cy", "vx", "vy", "area"]
        writer.writerow(keys)
        for row in evidence["rows"]:
            if (row["id"] in ex["output_ids"] and ex["output_frame"]-70 <= row["frame"] <= ex["output_frame"]) or (
                row["id"] == motion_track and row["frame"] in (motion_frame-1, motion_frame)
            ):
                writer.writerow([row[key] for key in keys])

    for key, path in evidence["source_paths"].items():
        if sha256(Path(path)) != evidence["source_hashes"][key]:
            raise ValueError(f"Source changed while rendering: {path}")
    report = {
        "inputs": {key: {"path": path, "sha256": evidence["source_hashes"][key]}
                   for key, path in evidence["source_paths"].items()},
        "renderer_sha256": sha256(Path(__file__)),
        "shared_renderer_sha256": sha256(ROOT / "customer_workspace/segmentation_ppt.py"),
        "fonts": {path.name: {"path": str(path), "sha256": sha256(path)} for path in (regular, bold)},
        "motion_prediction": prediction,
        "overlap_example": {
            "frame": ex["matrix_frame"], "track_ids": ids, "original_region_indices": columns,
            "displayed_region_names": ["R1", "R2"], "overlap_pixels": matrix.tolist(),
            "assigned_cells_row_column": chosen, "roi_xywh": overlap_roi,
            "isolated_from_other_eligible_pairs": True,
            "global_solver_assignments": [(ids[r], columns[c]) for r, c in chosen],
        },
        "events": [e for e in evidence["events"] if e["frame"] in (ex["merge"][0], ex["split"][0])],
        "distance_matching": event_details,
        "track_parameters": evidence["track_parameters"],
        "kept_track_count": evidence["kept_tracks"],
        "discarded_short_track_count": evidence["raw_tracks"] - evidence["kept_tracks"],
        "archived_runtime_ms": {"mean": evidence["stats"]["ms_per_frame_mean"],
                                "p95": evidence["stats"]["ms_per_frame_p95"]},
        "checks": {
            "all_archived_rows_match_to_saved_precision": evidence["all_archived_rows_match"],
            "all_archived_event_frames_types_and_ids_match": evidence["all_archived_events_match"],
            "displayed_overlap_matches_predicted_mask_pixels": True,
            "all_visible_labels_Arial": True, "sources_unchanged": True,
        },
        "figures": style.saved, "text_audit": style.text_audit,
        "notes": [
            "All replay arrays come from the current segment() and Tracker.step(); neither algorithm was edited.",
            "Saved IDs are obtained using finalize() short-track removal and ID remapping.",
            "R1 and R2 name current-frame detections, not track IDs; their original region indices are recorded.",
            "The merge keeps the Hungarian-assigned host, not necessarily the larger numeric ID or larger area.",
            "These are algorithmic region merge/split events, not proof of physical bubble coalescence.",
            "Distance matching is allowed when overlap fails the significance gate; it need not be exactly zero.",
            "max_gap=8 is current_frame-last_observed_frame <= 8, allowing seven intervening missing frames.",
            "min_len=5 counts observed rows, not total elapsed lifetime including gaps.",
            "Runtime is read from archived stats.json and covers segmentation/tracking only, excluding video decoding.",
            "The 60 fps line is the nominal frame interval, not a new end-to-end real-time validation.",
            "The output image shows five illustrative tracks with their observed centroid paths; other tracks are omitted.",
        ],
    }
    report["outputs"] = {
        str(path.relative_to(output)): sha256(path)
        for path in output.rglob("*") if path.is_file()
    }
    (output / "provenance.json").write_text(
        json.dumps(report, indent=2, default=lambda value: value.item()) + "\n"
    )
    with zipfile.ZipFile(output / "tracking_figures.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in style.saved + ["provenance.json", "data/overlap_matrix.csv", "data/trajectory_excerpt.csv"]:
            archive.write(output / name, arcname=name)
    print(json.dumps({"output": str(output), "checks": report["checks"]}, indent=2))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def collect_evidence(experiment="A", snapshot_frames=SNAPSHOT_FRAMES):
    """Replay the saved background and verify all rows against archived results.

    Selected step() calls are observed with a temporary profiling callback.
    B additionally searches for actual distance-only matches for illustration.
    It captures the real overlap matrix and ownership results. Pre-update track
    masks and velocities are copied before the call, since step() mutates them.
    """
    result = PROJECT / f"results/{experiment}_offline"
    paths = {name: result / name for name in (
        "background.npy", "objects.csv", "events.csv", "stats.json", "contours.npz",
    )}
    paths["video"] = PROJECT / f"data/raw/exp{experiment}.mp4"
    paths["algorithm"] = Path(bubble_tracker.__file__).resolve()
    hashes = {key: sha256(path) for key, path in paths.items()}
    stats = json.loads(paths["stats.json"].read_text())
    config = stats["params"]
    if stats["mode"] != "offline" or config["crop"]:
        raise ValueError("These examples require a full-frame offline result")
    background = np.load(paths["background.npy"], allow_pickle=False)
    parameters = bubble_tracker.SegParams(
        lo=config["lo"], hi=config["hi"], min_area=config["min_area"],
    )
    track_parameters = bubble_tracker.TrackParams()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (parameters.close, parameters.close)
    )
    tracker = bubble_tracker.Tracker(background.shape, track_parameters)
    snapshots = {}
    previous_rgb = None
    distance_snapshots = 0
    for index, bgr in bubble_tracker.iter_frames(str(paths["video"]), color=True):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        labels, contours, _ = bubble_tracker.segment(gray, background, parameters, kernel)
        seek_distance = experiment == "B" and distance_snapshots < 4 and index > 0
        if index in snapshot_frames or seek_distance:
            previous = {}
            for track in tracker.tracks.values():
                if not track.alive or index - track.last > track_parameters.max_gap:
                    continue
                previous[track.id] = {
                    "ys": track.ys.copy(), "xs": track.xs.copy(), "cx": track.cx,
                    "cy": track.cy, "vx": track.vx, "vy": track.vy,
                    "last": track.last, "area": track.area,
                }
            snapshot = {
                "frame": index, "previous": previous, "labels": labels.copy(),
                "rgb": rgb, "previous_rgb": previous_rgb, "contours": contours,
            }

            def capture(frame, event, argument):
                if event == "return" and frame.f_code is bubble_tracker.Tracker.step.__code__:
                    values = frame.f_locals
                    active_ids = [track.id for track in values.get("active", [])]
                    owners = values.get("owner", {})
                    significant = values.get("sig")
                    distance_match = significant is not None and any(
                        track.id in active_ids and not significant[active_ids.index(track.id), region]
                        for region, track in owners.items()
                    )
                    if index not in snapshot_frames and not distance_match:
                        return
                    for name in ("O", "pred", "sig"):
                        snapshot[name] = values[name].copy()
                    snapshot["active_ids"] = [track.id for track in values["active"]]
                    snapshot["owner"] = {int(j): track.id for j, track in values["owner"].items()}
                    snapshot["feats"] = values["feats"]

            previous_profiler = sys.getprofile()
            sys.setprofile(capture)
            try:
                tracker.step(index, labels, contours)
            finally:
                sys.setprofile(previous_profiler)
            if "O" in snapshot:
                snapshots[index] = snapshot
                if index not in snapshot_frames:
                    distance_snapshots += 1
        else:
            tracker.step(index, labels, contours)
        previous_rgb = rgb
        if index % 300 == 0:
            print(f"Replayed frame {index}", flush=True)

    rows, events, _, kept_count, raw_count = tracker.finalize()
    archived = read_csv(paths["objects.csv"])
    if len(rows) != len(archived):
        raise ValueError("Replay row count differs from saved objects.csv")
    numeric_keys = (
        "cx", "cy", "vx", "vy", "area", "eq_diam", "major", "minor",
        "angle", "circularity", "solidity",
    )
    for row, saved in zip(rows, archived):
        if row["frame"] != int(saved["frame"]) or row["id"] != int(saved["id"]):
            raise ValueError("Replayed frame/ID differs from the saved result")
        for key in numeric_keys:
            if f"{row[key]:.3f}" != saved[key]:
                raise ValueError(f"Saved precision mismatch: frame {row['frame']}, {key}")
    expected_events = [
        (int(event["frame"]), event["type"], event["ids"], int(event["result_id"]))
        for event in read_csv(paths["events.csv"])
    ]
    replayed_events = [
        (event["frame"], event["type"], " ".join(map(str, event["ids"])), event["result"])
        for event in sorted(events, key=lambda event: (event["frame"], event["type"]))
    ]
    if replayed_events != expected_events:
        raise ValueError("Replay events differ from saved events.csv")
    counts = Counter(row["id"] for row in tracker.rows)
    first = {}
    for row in tracker.rows:
        first.setdefault(row["id"], row["frame"])
    kept = [track_id for track_id, count in counts.items() if count >= track_parameters.min_len]
    remap = {
        old: new for new, old in enumerate(sorted(kept, key=lambda i: (first[i], i)), 1)
    }
    for key, path in paths.items():
        if sha256(path) != hashes[key]:
            raise ValueError(f"Source changed while replaying: {path}")
    print(f"Verified {len(rows)} rows and {len(events)} events against saved results", flush=True)
    return {
        "experiment": experiment,
        "snapshots": snapshots, "remap": remap, "rows": rows, "events": events,
        "raw_rows": tracker.rows, "raw_counts": dict(counts), "stats": stats,
        "source_paths": {key: str(path) for key, path in paths.items()},
        "source_hashes": hashes, "track_parameters": asdict(track_parameters),
        "kept_tracks": kept_count, "raw_tracks": raw_count,
        "all_archived_rows_match": True, "all_archived_events_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment", choices=("A", "B"), default="A")
    parser.add_argument("--font-dir", type=Path, default=Path("/mnt/c/Windows/Fonts"))
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir or ROOT / "customer_workspace/validation" / f"tracking_slides_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    frames = SNAPSHOT_FRAMES if args.experiment == "A" else (
        147, 182, 253, 260, 280, 296, 299, 329, 744, 881, 883, 1170, 1225,
    )
    export_figures(collect_evidence(args.experiment, frames), output, args.font_dir)


if __name__ == "__main__":
    main()
