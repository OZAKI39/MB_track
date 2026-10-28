#!/usr/bin/env python3
"""
channel_mask.py - the channel mask of the Medany et al. pipeline, made once per set-up.

Original procedure (environments/ARSL_env_camera_dreamer.py : _naive_thresholding, _seg_anything):
    1. adaptive mean threshold of the first ROI frame (block 91, C = 2), median blur 5
       (utils/preprocessing.py : ThresholdMaskGenerator.threshold)
    2. flood fill black at hand-picked background points and white at foreground points, tolerance 25
    3. SAM ViT-H on that thresholded image, the same points as prompts, 25 rounds feeding the mask back,
       single mask (utils/image_postprocessing.py : create_binary_bitmap)
    4. morphological closing, 3 x 3 kernel, 3 iterations
Here step 3 uses MobileSAM (same prompt interface), because the SAM ViT-H weights could not be downloaded.

Also writes an "ideal" mask for a best-case baseline: the channel interior bounded by the walls found in
our background image (holes filled, so fixed debris inside the channel stays inside, as in a real mask).

Usage (baseline environment, see baseline/README section):
    python baseline/channel_mask.py --video data/raw/expA.mp4 --background results/A_offline/background.npy \
        --mobilesam /path/to/MobileSAM --out results/baseline_comparison/masks
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Same field of view in experiments A and B (offset < 1 px). Points in full-frame pixels (x, y).
ROI = (128, 0, 1024, 1024)            # square, as the original requires
FOREGROUND = [(680, 80), (680, 250), (690, 420), (660, 540), (720, 560),          # main channel and junction
              (560, 700), (480, 800), (410, 900), (345, 990),                       # left branch
              (780, 700), (835, 800), (885, 900), (930, 990)]                       # right branch
BACKGROUND = [(250, 250), (300, 600), (170, 880), (420, 150),                       # left of the channel
              (1000, 250), (1100, 650), (900, 150), (1120, 950),                    # right of the channel
              (660, 860), (650, 990), (700, 960)]                                   # wedge between the branches


def to_roi(points, roi=ROI):
    return [(int(x - roi[0]), int(y - roi[1])) for x, y in points]


def threshold_image(crop, c=2):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 91, c)
    thr = cv2.medianBlur(thr, 5)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR).astype(np.uint8)


def flood(img, points, color):
    for p in points:
        cv2.floodFill(img, None, p, color, (25,) * 3, (25,) * 3)
    return img


def sam_mask(thr_img, fg, bg, mobilesam_dir, loops=25):
    sys.path.insert(0, mobilesam_dir)
    import warnings
    warnings.filterwarnings('ignore')
    import torch
    from mobile_sam import sam_model_registry, SamPredictor
    model = sam_model_registry['vit_t'](checkpoint=os.path.join(mobilesam_dir, 'weights', 'mobile_sam.pt'))
    model.eval()
    pred = SamPredictor(model)
    pts = np.array(fg + bg)
    lab = np.array([1] * len(fg) + [0] * len(bg))
    with torch.no_grad():
        pred.set_image(thr_img)
        mask_in = None
        for _ in range(loops):
            masks, _, logits = pred.predict(point_coords=pts, point_labels=lab, multimask_output=False, mask_input=mask_in)
            mask_in = logits
    m = masks[0].astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)


def ideal_mask(background, fg_points, roi=ROI):
    """Channel interior bounded by the walls of our background image (best case for the baseline)."""
    sys.path.insert(0, ROOT)
    import bubble_tracker as bt
    B = np.load(background).astype(np.float32)
    walls = (1 - B / (bt.smooth_illumination(B) + 1)) > 0.06
    free = (~walls).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    keep = {lab[y, x] for x, y in fg_points if lab[y, x] > 0}
    inside = np.isin(lab, list(keep)).astype(np.uint8)
    # fill holes (fixed debris inside the channel), close the 1-2 px gaps along the wall edge
    cs, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(inside)
    cv2.drawContours(filled, cs, -1, 1, -1)
    x, y, w, h = roi
    return filled[y:y + h, x:x + w]


def main():
    ap = argparse.ArgumentParser(description='Channel mask of the Medany et al. pipeline')
    ap.add_argument('--video', required=True)
    ap.add_argument('--background', help='our background.npy, for the ideal mask')
    ap.add_argument('--mobilesam', help='MobileSAM checkout (with weights/mobile_sam.pt)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    ok, frame = cap.read()
    cap.release()
    x, y, w, h = ROI
    crop = frame[y:y + h, x:x + w].copy()
    cv2.imwrite(os.path.join(a.out, 'first_frame_roi.png'), crop)
    fg, bg = to_roi(FOREGROUND), to_roi(BACKGROUND)
    thr = threshold_image(crop)
    flood(thr, bg, (0, 0, 0))
    flood(thr, fg, (255, 255, 255))
    cv2.imwrite(os.path.join(a.out, 'thresholded_prompt_image.png'), thr)
    info = dict(roi=ROI, foreground_points=FOREGROUND, background_points=BACKGROUND, adaptive_threshold_c=2,
                sam='MobileSAM vit_t (stand-in for SAM ViT-H)', loops=25, closing='3x3, 3 iterations')
    if a.mobilesam:
        m = sam_mask(thr, fg, bg, a.mobilesam)
        cv2.imwrite(os.path.join(a.out, 'mask_mobilesam.png'), m * 255)
        info['mask_mobilesam_area_px'] = int(m.sum())
        print('MobileSAM mask:', int(m.sum()), 'px inside')
    if a.background:
        m = ideal_mask(a.background, FOREGROUND)
        cv2.imwrite(os.path.join(a.out, 'mask_ideal.png'), m * 255)
        info['mask_ideal_area_px'] = int(m.sum())
        print('ideal mask:', int(m.sum()), 'px inside')
    json.dump(info, open(os.path.join(a.out, 'masks.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
