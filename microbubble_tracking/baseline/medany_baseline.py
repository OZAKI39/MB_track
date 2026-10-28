#!/usr/bin/env python3
"""
medany_baseline.py - the microbubble detection and tracking of Medany et al., re-implemented for an
offline comparison on our recordings.

    Medany, M., Piglia, L., Achenbach, L. et al. Model-based reinforcement learning for ultrasound-driven
    autonomous microrobots. Nature Machine Intelligence 7, 1076-1090 (2025).
    Code: https://github.com/M-Medany/Model-Based-Reinforcement-Learning-for-Ultrasound-Driven-Autonomous-Microrobots
    (GPL-3.0; this file follows that code closely and is shared under the same licence)

What the original does, and where (file : function)
    1. Square region of interest, chosen by hand          environments/ARSL_env_camera_dreamer.py : __init__,
                                                          utils/image_postprocessing.py : resize_and_crop_frame
    2. Channel mask, made once per set-up:                ARSL_env_camera_dreamer.py : _naive_thresholding, _seg_anything
       adaptive mean threshold (block 91, C = 2) + median 5  utils/preprocessing.py : ThresholdMaskGenerator.threshold
       flood fill at hand-picked background / foreground  ThresholdMaskGenerator.set_backround_points / set_foreground_points
       points (tolerance 25)
       SAM ViT-H with those points as prompts on the     utils/image_postprocessing.py : create_binary_bitmap
       thresholded image, 25 refinement loops, closing    (kernel 3x3, 3 iterations in _seg_anything)
    3. Every frame: inside the channel mask, pixels       utils/image_postprocessing.py : plot_cluster_on_image_blue
       darker than THRESHOLD (100) become "blue"          scripts/config.yaml : CSRT_Tracker_settings.THRESHOLD
    4. One target, tracked with OpenCV CSRT on the        utils/tracking_CSRT.py : CSRT_tracker.track
       blue image; position = mean of the blue pixels in
       the box, size = number of blue pixels in the box
    5. When CSRT fails or the box holds no blue pixel:   ARSL_env_camera_dreamer.py : _reinitialize_tracker_autonomously,
       new tracker on the largest blue region, box        utils/image_postprocessing.py : detect_largest_cluster,
       enlarged by BOX_PADDING (0.6) on every side;       config.yaml : BOX_PADDING 0.6, MIN_SIZE_BEFORE_ABORT 0.1
       accepted only if it holds >= 0.1 x the first size

Differences in this offline version (and why)
    - SAM ViT-H weights could not be downloaded here, so the channel mask is made with MobileSAM, which takes
      the same point prompts (channel_mask.py). A mask made with the original SAM can be given instead.
    - Hand steps are replaced by fixed inputs: the ROI, the prompt points, and the first box (taken from the
      reference outline of the target, enlarged by the original BOX_PADDING, as the original's own automatic
      re-initialisation does).
    - When re-detection fails, the original asks the operator and switches on a PZT; offline the frame is
      recorded as "lost" and re-detection is tried again on the next frame (the original also moves on to
      the next camera frame).
    - The original has a quirk: np.logical_and(a, b, c) uses c as the output array, so only the first two
      colour channels are compared with the threshold. It is kept (our frames are grey, so it makes no
      difference).
Needs OpenCV with the contrib trackers (cv2.TrackerCSRT_create), e.g. opencv-contrib-python 4.x.
"""
import numpy as np
import cv2

THRESHOLD = 100            # scripts/config.yaml
BOX_PADDING = 0.6          # scripts/config.yaml
MIN_SIZE_BEFORE_ABORT = 0.1
INITIAL_PADDING = 0.1      # first box: tight, like a careful hand-drawn box (not in the original config)
BLUE = (255, 0, 0)


def crop_roi(frame, roi):
    """resize_and_crop_frame: the square region of interest (x, y, w, h)."""
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def blue_image(channel_mask, image, threshold=THRESHOLD):
    """plot_cluster_on_image_blue: white channel, black outside, blue = dark pixels inside the channel."""
    dark = blue_mask(np.ones(image.shape[:2], np.uint8), image, threshold)
    img = np.zeros(image.shape, np.uint8)
    img[channel_mask > 0] = (255, 255, 255)
    img[dark & (channel_mask > 0)] = BLUE
    return img


def blue_mask(channel_mask, image, threshold=THRESHOLD):
    """Only the blue pixels (same rule as blue_image), as a boolean array."""
    out = image[:, :, 2] < threshold          # the original passes this array as `out` (third argument)
    np.logical_and(image[:, :, 0] < threshold, image[:, :, 1] < threshold, out)
    return out & (channel_mask > 0)


def is_blue(img):
    return (img[:, :, 0] == 255) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)


def detect_largest_cluster(img):
    """detect_largest_cluster: bounding box of the largest blue contour, or zeros."""
    m = is_blue(img).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 0, 0
    c = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(c)


class CSRTTracker:
    """utils/tracking_CSRT.py : CSRT_tracker (autonomous initialisation)."""

    def __init__(self, img, x, y, w, h, box_padding=BOX_PADDING):
        self._tracker = cv2.TrackerCSRT_create()      # parameters='rgb_on_rest_default' -> default CSRT
        pw, ph = w * box_padding, h * box_padding
        box = (x - pw, y - ph, w + pw * 2, h + ph * 2)
        size = img.shape[0]
        self.init_box = tuple(int(v) for v in np.clip(tuple(int(v) for v in box), 0, size))
        self._tracker.init(img, self.init_box)
        bx, by, bw, bh = self.init_box
        self.blue_area = int(is_blue(img[by:by + bh, bx:bx + bw]).sum())
        self.center = (bx + bw // 2, by + bh // 2)
        self.bbox = self.init_box

    def track(self, img):
        ok, bbox = self._tracker.update(img)
        if ok:
            x, y, w, h = [int(i) for i in bbox]
            sub = is_blue(img[max(0, y):y + h, max(0, x):x + w])
            n = int(sub.sum())
            if n > 0:
                yy, xx = np.nonzero(sub)
                self.center = (float(xx.mean()) + max(0, x), float(yy.mean()) + max(0, y))
                self.blue_area = n
                self.bbox = (x, y, w, h)
                return True
        self.blue_area = 0
        return False


class MedanyTracker:
    """One target, as in ARSL_env_camera_dreamer._get_obs with automatic re-initialisation."""

    def __init__(self, channel_mask, threshold=THRESHOLD, box_padding=BOX_PADDING):
        self.mask, self.threshold, self.padding = channel_mask, threshold, box_padding
        self.tracker = None
        self.initial_area = None
        self.reinits = 0

    def start(self, crop, box, padding=INITIAL_PADDING):
        """First frame: the target box (x, y, w, h in ROI pixels), standing in for the box the operator draws
        by hand in the original (cv2.selectROI). A careful hand-drawn box is tight, so only a small margin
        (INITIAL_PADDING) is added; later re-initialisations use the original BOX_PADDING."""
        img = blue_image(self.mask, crop, self.threshold)
        self.tracker = CSRTTracker(img, *box, box_padding=padding)
        self.initial_area = self.tracker.blue_area
        self.threshold_area = self.initial_area * MIN_SIZE_BEFORE_ABORT
        ok = self.tracker.track(img)
        return self._result(ok, reinit=False)

    def step(self, crop):
        img = blue_image(self.mask, crop, self.threshold)
        ok = self.tracker.track(img)
        if ok:
            return self._result(True, reinit=False)
        # _reinitialize_tracker_autonomously: largest blue region, padded box, size check
        x, y, w, h = detect_largest_cluster(img)
        if (x, y, w, h) == (0, 0, 0, 0):
            return self._result(False, reinit=False)
        cand = CSRTTracker(img, x, y, w, h, box_padding=self.padding)
        if cand.blue_area < self.threshold_area:
            return self._result(False, reinit=False)
        self.tracker = cand
        self.reinits += 1
        ok = self.tracker.track(img)
        return self._result(ok, reinit=True)

    def _result(self, ok, reinit):
        t = self.tracker
        return dict(ok=bool(ok), x=t.center[0] if ok else None, y=t.center[1] if ok else None,
                    area=t.blue_area if ok else 0, bbox=t.bbox if ok else None, reinit=reinit)
