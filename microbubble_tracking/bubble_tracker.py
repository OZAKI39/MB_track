#!/usr/bin/env python3
"""
bubble_tracker.py — 明场显微视频中微泡 / 微泡团的分割与跟踪（v2）

核心假设：同一次装片（注射完成、芯片不再挪动）期间，视野是静止的。
于是背景可以从录像本身估计，不需要空白帧。

背景（--ref 可给多个视频或保存的 .npy 背景）：
  * 每段参考视频逐像素取第 90 百分位（微泡是暗的，取偏亮的分位数）。
  * 参考背景与当前视频用相位相关做亚像素对齐，多个参考逐像素取最大值：
    只要某个像素在任意一段录像里露出过背景，就能得到正确背景。
  * offline 模式默认把当前视频自己也作为参考；online（实时）模式只用 --ref，
    不读未来帧。没有 --ref 时退化为“第一帧 + 修补粗团块”，可能误伤粗的通道壁结构。

逐帧处理（因果，可实时）：
  1. D = 1 - I / B，并减去全帧中位数（抵消整体亮度波动）。D 越大越暗。
  2. 滞后阈值 lo / hi + 闭运算 + 只取外轮廓（等价于填孔）+ 面积过滤。
  3. 上一帧掩膜按速度平移后与当前连通域求重叠：重叠图上用匈牙利算法定“延续”，
     其余重叠关系判为融合 / 分裂；无重叠的小目标再按预测位置距离匹配。
  4. 融合时面积最大的一方保留编号，其余记为并入；分裂时最大的子块保留编号。
  5. 少于 min_len 帧的轨迹视为分割闪烁，连同相关事件一起丢弃（约 80 ms 延迟）。

输出（都在 --out 目录下）：
  objects.csv     每帧每个目标：位置、速度、面积、等效直径、椭圆、圆度、凸度
  events.csv      融合 / 分裂 / 出现 / 消失 / 进出画面
  background.npy  本次使用的背景（可作为同一次装片后续实验的 --ref）
  stats.json      帧数、耗时、编号数、事件计数、所用参数
  overlay.mp4     叠加视频（加 --render 时）
  contours.npz    每帧每个目标的轮廓（加 --save-contours 时，人工审查工具需要）

示例：
  # 离线分析，同一次装片的另一段实验一起参与估计背景
  python bubble_tracker.py expB.mp4 --ref expA.mp4 --out resB --render
  # 实时模式（只用参考背景，不看未来帧）
  python bubble_tracker.py expB.mp4 --bg online --ref resA/background.npy --out resB_online
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
import colorsys
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================================
# 读帧
# ============================================================================
def iter_frames(path, crop=None, start=0, stop=None, color=False):
    cap = cv2.VideoCapture(path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    while stop is None or i < stop:
        ok, fr = cap.read()
        if not ok:
            break
        if crop:
            x, y, w, h = crop
            fr = fr[y:y + h, x:x + w]
        if color:
            yield i, fr if fr.ndim == 3 else cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        else:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
            yield i, g.astype(np.float32)
        i += 1
    cap.release()


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def first_frame(path, crop=None):
    return next(iter_frames(path, crop))[1]


# ============================================================================
# 背景
# ============================================================================
def percentile_background(path, crop=None, percentile=90, max_samples=330, start=0, stop=None):
    """逐像素高分位数。取第 90 而不是第 75 百分位：停留时间不到 90% 的目标都不会被吸收。"""
    _, n = video_info(path)
    stop = n if stop is None or stop > n else stop
    step = max(1, (stop - start) // max_samples)
    # 以 uint16（灰度×16）存储并原地 partition，内存约为 float 方案的 1/4
    stack = np.stack([np.clip(cv2.GaussianBlur(f, (0, 0), 1.0) * 16 + 0.5, 0, 65535).astype(np.uint16)
                      for i, f in iter_frames(path, crop, start, stop) if (i - start) % step == 0])
    k = int(round(percentile / 100 * (len(stack) - 1)))
    out = np.empty(stack.shape[1:], np.float32)
    for y in range(0, stack.shape[1], 64):
        blk = stack[:, y:y + 64]
        blk.partition(k, axis=0)
        out[y:y + 64] = blk[k] / 16.0
    return out


def _grad(img):
    x = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))


def register_shift(ref, target):
    """返回把 ref 对齐到 target 所需的平移 (dx, dy)，用通道壁的梯度做相位相关。"""
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(_grad(ref), _grad(target), win)
    return dx, dy, resp


def shift_image(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, img.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def load_reference(src, crop=None, percentile=90):
    if src.endswith('.npy'):
        return np.load(src).astype(np.float32)
    return percentile_background(src, crop, percentile)


def combine_backgrounds(bgs, target_frame, verbose=True):
    """把多个背景对齐到当前视频后逐像素取最大值。"""
    out = None
    for k, B in enumerate(bgs):
        dx, dy, resp = register_shift(B, target_frame)
        if verbose:
            print(f'  reference {k}: shift dx={dx:+.2f} dy={dy:+.2f} px (response {resp:.2f})')
        if abs(dx) > 0.05 or abs(dy) > 0.05:
            B = shift_image(B, dx, dy)
        out = B if out is None else np.maximum(out, B)
    return out


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def smooth_illumination(B):
    small = cv2.resize(B, (B.shape[1] // 4, B.shape[0] // 4), interpolation=cv2.INTER_AREA)
    med = cv2.medianBlur(np.clip(small, 0, 255).astype(np.uint8), 25).astype(np.float32)
    return cv2.resize(cv2.GaussianBlur(med, (0, 0), 3), B.shape[::-1], interpolation=cv2.INTER_LINEAR)


def remove_thick_blobs(B, thick=14, lo=0.06, open_r=7, ring=4):
    """后备方案：没有参考背景时，用第一帧当背景，并把其中“粗”的深色块（团）修补掉。
    通道壁是细线，团是粗块；但壁上的缺陷或贴壁团可能被误判，所以只作为后备。"""
    Bb = cv2.GaussianBlur(B, (0, 0), 1)
    D = 1 - Bb / (smooth_illumination(Bb) + 1)
    m = (D > lo).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    op = cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(open_r))
    n, lab = cv2.connectedComponents(op, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[np.unique(lab[(dist >= thick) & (op > 0)])] = 1
    keep[0] = 0
    region = cv2.dilate(cv2.dilate(keep[lab], _disk(ring)) & m, _disk(2))
    if not region.any():
        return B.copy()
    inp = cv2.inpaint(np.clip(B, 0, 255).astype(np.uint8), region, 7, cv2.INPAINT_TELEA).astype(np.float32)
    out = B.copy()
    out[region > 0] = inp[region > 0]
    return out


class OnlineBackground:
    """非对称慢速更新：背景像素变亮时较快跟上，变暗时极慢；前景及其周边不更新，
    所以停住的团不会被吸收。画面按行分成 bands 条，每帧只更新一条（轮流），
    每个像素每 bands 帧更新一次，计算量摊到每帧上。"""

    def __init__(self, B0, a_up=0.2, a_down=0.002, bands=10, fg_margin=6):
        self.B = B0.astype(np.float32).copy()
        self.a_up, self.a_down, self.bands = np.float32(a_up), np.float32(a_down), bands
        self.K = _disk(fg_margin)
        self.m = fg_margin + 2
        self.k = 0

    def update(self, fb, labels):
        H = self.B.shape[0]
        k = self.k % self.bands
        self.k += 1
        y0, y1 = k * H // self.bands, (k + 1) * H // self.bands
        m0, m1 = max(0, y0 - self.m), min(H, y1 + self.m)
        fg = cv2.dilate((labels[m0:m1] > 0).astype(np.uint8), self.K)[y0 - m0:y1 - m0]
        Bs = self.B[y0:y1]
        diff = fb[y0:y1] - Bs
        a = np.where(diff > 0, self.a_up, self.a_down)
        a[fg > 0] = 0
        Bs += a * diff


# ============================================================================
# 分割
# ============================================================================
@dataclass
class SegParams:
    lo: float = 0.06        # 比背景暗 6% 以上算候选（背景噪声 σ≈0.004；更低会引入壁边缘碎点）
    hi: float = 0.12        # 连通域中至少有像素暗 12% 以上才保留
    close: int = 5          # 闭运算核直径（约一个单泡直径）
    min_area: int = 12      # 像素
    blur: float = 1.0


def segment(f, B, p: SegParams, K):
    """B 为已模糊的背景。返回 labels(uint16)、外轮廓列表、模糊后的帧。"""
    fb = cv2.GaussianBlur(f, (0, 0), p.blur)
    D = 1.0 - fb / (B + 1.0)
    D -= np.median(D[::8, ::8])
    lo_m = cv2.morphologyEx((D > p.lo).astype(np.uint8), cv2.MORPH_CLOSE, K)
    n, lab, st, _ = cv2.connectedComponentsWithStats(lo_m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[(D > p.hi) & (lo_m > 0)])] = True
    keep[0] = False
    m = np.zeros(f.shape, np.uint8)
    for k in np.nonzero(keep)[0]:
        x, y, w, h = st[k, :4]
        m[y:y + h, x:x + w] |= (lab[y:y + h, x:x + w] == k).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    labels = np.zeros(f.shape, np.uint16)
    kept = []
    for c in contours:
        if cv2.contourArea(c) < p.min_area * 0.5:     # 粗筛，精确面积在下面按像素数算
            continue
        kept.append(c)
        cv2.drawContours(labels, [c], -1, len(kept), thickness=-1)
    if kept:
        areas = np.bincount(labels[labels > 0], minlength=len(kept) + 1)
        good = [j for j in range(1, len(kept) + 1) if areas[j] >= p.min_area]
        if len(good) < len(kept):
            remap = np.zeros(len(kept) + 1, np.uint16)
            for new, old in enumerate(good, start=1):
                remap[old] = new
            labels = remap[labels]
            kept = [kept[j - 1] for j in good]
    return labels, kept, fb


def shape_features(ys, xs, contour):
    area = len(xs)
    cx, cy = xs.mean(), ys.mean()
    eqd = 2.0 * np.sqrt(area / np.pi)
    if area >= 5:
        ev, evec = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
        ev = np.clip(ev, 1e-6, None)
        major, minor = 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0])
        ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180
    else:
        major = minor = eqd
        ang = 0.0
    per = cv2.arcLength(contour, True) if len(contour) > 2 else np.pi * eqd
    circ = min(1.0, 4 * np.pi * area / max(per, 1e-6) ** 2)
    ha = cv2.contourArea(cv2.convexHull(contour))
    solidity = min(1.0, area / ha) if ha > 0 else 1.0
    return dict(cx=cx, cy=cy, area=area, eq_diam=eqd, major=major, minor=minor,
                angle=ang, circularity=circ, solidity=solidity)


# ============================================================================
# 跟踪
# ============================================================================
@dataclass
class Track:
    id: int
    ys: np.ndarray
    xs: np.ndarray
    cx: float
    cy: float
    area: int
    vx: float = 0.0
    vy: float = 0.0
    first: int = 0
    last: int = 0
    alive: bool = True
    parents: list = field(default_factory=list)


@dataclass
class TrackParams:
    min_overlap_px: int = 3
    min_overlap_frac: float = 0.15   # 相对较小一方的面积
    gate_px: float = 15.0            # 无重叠时的距离门限（另加速度项）
    max_gap: int = 8                 # 允许丢失的帧数
    min_len: int = 5                 # 确认所需帧数
    border: int = 8
    vel_alpha: float = 0.5


class Tracker:
    def __init__(self, shape, tp: TrackParams):
        self.H, self.W = shape
        self.tp = tp
        self.tracks = {}
        self.next_id = 1
        self.events, self.rows = [], []
        self.contours = {}

    def _new(self, frame, ys, xs, feat, parents=()):
        t = Track(self.next_id, ys, xs, feat['cx'], feat['cy'], feat['area'],
                  first=frame, last=frame, parents=list(parents))
        self.tracks[t.id] = t
        self.next_id += 1
        return t

    def _near_border(self, x, y):
        b = self.tp.border
        return x < b or y < b or x > self.W - b or y > self.H - b

    def _event(self, frame, kind, ids, result, f):
        self.events.append(dict(frame=frame, type=kind, ids=list(ids), result=result, x=f['cx'], y=f['cy']))

    def step(self, frame, labels, contours):
        tp = self.tp
        n_det = len(contours)
        pix = [None] * (n_det + 1)
        if n_det:
            ys, xs = np.nonzero(labels)
            vals = labels[ys, xs]
            o = np.argsort(vals, kind='stable')
            ys, xs, vals = ys[o], xs[o], vals[o]
            b = np.searchsorted(vals, np.arange(1, n_det + 2))
            for j in range(1, n_det + 1):
                pix[j] = (ys[b[j - 1]:b[j]], xs[b[j - 1]:b[j]])
        feats = [None] + [shape_features(pix[j][0], pix[j][1], contours[j - 1]) for j in range(1, n_det + 1)]
        det_area = np.array([0] + [f['area'] for f in feats[1:]])

        active = [t for t in self.tracks.values() if t.alive and frame - t.last <= tp.max_gap]
        nT = len(active)
        O = np.zeros((nT, n_det + 1), np.int64)
        pred = np.zeros((nT, 2))
        for a, t in enumerate(active):
            gap = frame - t.last
            dx, dy = t.vx * gap, t.vy * gap
            pred[a] = (t.cx + dx, t.cy + dy)
            yy = np.clip(np.round(t.ys + dy).astype(int), 0, self.H - 1)
            xx = np.clip(np.round(t.xs + dx).astype(int), 0, self.W - 1)
            O[a] = np.bincount(labels[yy, xx], minlength=n_det + 1)[:n_det + 1]
        O[:, 0] = 0
        sig = np.zeros_like(O, bool)
        if nT and n_det:
            tr_area = np.array([t.area for t in active])
            mins = np.minimum(tr_area[:, None], det_area[None, :])
            sig = (O >= tp.min_overlap_px) & (O >= tp.min_overlap_frac * mins)
            sig[:, 0] = False

        owner, matched = {}, set()
        # 1. 有重叠：匈牙利定延续，其余重叠关系为融合 / 分裂
        if sig.any():
            r, c = linear_sum_assignment(np.where(sig, -O.astype(float), 1e9))
            for a, j in zip(r, c):
                if sig[a, j]:
                    owner[j] = active[a]
                    matched.add(a)
            for a in range(nT):
                if a in matched or not sig[a].any():
                    continue
                j = int(np.argmax(np.where(sig[a], O[a], -1)))
                if j in owner:
                    host = owner[j]
                    active[a].alive = False
                    matched.add(a)
                    self._event(frame, 'merge', [active[a].id, host.id], host.id, feats[j])
            for j in range(1, n_det + 1):
                if j in owner or not sig[:, j].any():
                    continue
                a = int(np.argmax(np.where(sig[:, j], O[:, j], -1)))
                if a not in matched:
                    continue
                parent = active[a]
                t = self._new(frame, *pix[j], feats[j], parents=[parent.id])
                t.vx, t.vy = parent.vx, parent.vy
                owner[j] = t
                self._event(frame, 'split', [parent.id, t.id], parent.id, feats[j])

        # 2. 无重叠：按预测位置距离匹配
        free_tr = [a for a in range(nT) if a not in matched]
        free_det = [j for j in range(1, n_det + 1) if j not in owner and not sig[:, j].any()]
        if free_tr and free_det:
            C = np.full((len(free_tr), len(free_det)), 1e9)
            for u, a in enumerate(free_tr):
                t = active[a]
                gate = tp.gate_px + 2.0 * np.hypot(t.vx, t.vy) * (frame - t.last)
                for w, j in enumerate(free_det):
                    d = np.hypot(feats[j]['cx'] - pred[a, 0], feats[j]['cy'] - pred[a, 1])
                    ratio = max(t.area, det_area[j]) / max(1, min(t.area, det_area[j]))
                    if d < gate and ratio < 4:
                        C[u, w] = d
            r, c = linear_sum_assignment(C)
            for u, w in zip(r, c):
                if C[u, w] < 1e9:
                    owner[free_det[w]] = active[free_tr[u]]
                    matched.add(free_tr[u])

        # 3. 更新 / 新建
        for j in range(1, n_det + 1):
            f = feats[j]
            if j in owner:
                t = owner[j]
                if t.first != frame:
                    gap = max(1, frame - t.last)
                    al = tp.vel_alpha
                    t.vx = al * (f['cx'] - t.cx) / gap + (1 - al) * t.vx
                    t.vy = al * (f['cy'] - t.cy) / gap + (1 - al) * t.vy
            else:
                t = self._new(frame, *pix[j], f)
                self._event(frame, 'enter' if self._near_border(f['cx'], f['cy']) else 'appear', [t.id], t.id, f)
            t.ys, t.xs = pix[j]
            t.cx, t.cy, t.area, t.last = f['cx'], f['cy'], f['area'], frame
            self.rows.append(dict(frame=frame, id=t.id, vx=t.vx, vy=t.vy, **f))
            self.contours.setdefault(frame, []).append((t.id, contours[j - 1]))

        # 4. 超时结束
        for t in self.tracks.values():
            if t.alive and frame - t.last > tp.max_gap:
                t.alive = False
                self.events.append(dict(frame=t.last, type='exit' if self._near_border(t.cx, t.cy) else 'lost',
                                        ids=[t.id], result=t.id, x=t.cx, y=t.cy))

    def finalize(self):
        """丢弃短命轨迹（分割闪烁）及相关事件，按出现顺序重新编号。"""
        counts, first = defaultdict(int), {}
        for r in self.rows:
            counts[r['id']] += 1
            first.setdefault(r['id'], r['frame'])
        good = {i for i, c in counts.items() if c >= self.tp.min_len}
        remap = {old: new for new, old in enumerate(sorted(good, key=lambda i: (first[i], i)), start=1)}
        rows = [dict(r, id=remap[r['id']]) for r in self.rows if r['id'] in good]
        events = [dict(e, ids=[remap[i] for i in e['ids']], result=remap.get(e['result'], e['result']))
                  for e in self.events if all(i in good for i in e['ids'])]
        contours = {f: [(remap[i], c) for i, c in lst if i in good] for f, lst in self.contours.items()}
        return rows, events, contours, len(good), len(counts)


# ============================================================================
# 渲染叠加视频
# ============================================================================
def _color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.9, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def render(path, crop, rows, events, contours, out_path, fps, tail=60):
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r['frame']].append(r)
    ev = defaultdict(list)
    for e in events:
        if e['type'] in ('merge', 'split'):
            ev[e['frame']].append(e)
    writer = None
    tails = defaultdict(lambda: deque(maxlen=tail))
    recent, seen = deque(), set()
    for i, fr in iter_frames(path, crop, color=True):
        H, W = fr.shape[:2]
        if writer is None:
            if shutil.which('ffmpeg'):
                writer = subprocess.Popen(
                    ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H + 60}',
                     '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22',
                     '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out_path], stdin=subprocess.PIPE)
            else:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H + 60))
        cur = by_frame.get(i, [])
        ids = {r['id'] for r in cur}
        for r in cur:
            tails[r['id']].append((int(round(r['cx'])), int(round(r['cy']))))
            seen.add(r['id'])
        for k in [k for k in tails if k not in ids]:
            del tails[k]
        for k, pts in tails.items():
            if len(pts) > 1:
                cv2.polylines(fr, [np.array(pts, np.int32)], False, _color(k), 1, cv2.LINE_AA)
        for k, c in contours.get(i, []):
            cv2.drawContours(fr, [c], -1, _color(k), 2, cv2.LINE_AA)
        for r in cur:
            x, y = int(r['cx'] + r['eq_diam'] / 2 + 3), int(r['cy'] - r['eq_diam'] / 2)
            cv2.putText(fr, f"#{r['id']}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(fr, f"#{r['id']}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _color(r['id']), 1, cv2.LINE_AA)
        for e in ev.get(i, []):
            recent.append((i, e))
        while recent and i - recent[0][0] > int(1.5 * fps):
            recent.popleft()
        head = np.full((60, W, 3), (43, 34, 27), np.uint8)
        cv2.putText(head, f't = {i / fps:6.3f} s   objects {len(cur)}   IDs so far {len(seen)}', (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        if recent:
            txt = '   '.join(
                (f"merge {'+'.join('#%d' % k for k in e['ids'])} -> #{e['result']}" if e['type'] == 'merge'
                 else f"split #{e['ids'][0]} -> #{e['ids'][0]} + #{e['ids'][1]}") for _, e in list(recent)[-3:])
            cv2.putText(head, txt, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 205, 255), 1, cv2.LINE_AA)
        frame = np.vstack([head, fr])
        if isinstance(writer, subprocess.Popen):
            writer.stdin.write(frame.tobytes())
        else:
            writer.write(frame)
    if isinstance(writer, subprocess.Popen):
        writer.stdin.close()
        writer.wait()
    elif writer is not None:
        writer.release()


# ============================================================================
# 主程序
# ============================================================================
def build_background(video, crop, mode, refs, percentile, verbose=True):
    f0 = first_frame(video, crop)
    sources = [load_reference(r, crop, percentile) for r in refs]
    if mode == 'offline':
        sources.append(percentile_background(video, crop, percentile))
    if sources:
        label = ('self+ref' if refs else 'self') if mode == 'offline' else 'ref'
        return combine_backgrounds(sources, f0, verbose), label
    if verbose:
        print('  警告：online 模式没有 --ref，退化为“第一帧 + 修补粗团块”，可能误伤粗的通道壁结构')
    return remove_thick_blobs(cv2.GaussianBlur(f0, (0, 0), 1.0)), 'first-frame'


def run(video, crop=None, mode='offline', refs=(), seg=SegParams(), tp=TrackParams(),
        percentile=90, start=0, stop=None, fps=None, verbose=True):
    fps = fps or video_info(video)[0]
    if verbose:
        print(f'[{os.path.basename(video)}] background ({mode})')
    B, bg_src = build_background(video, crop, mode, list(refs), percentile, verbose)
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seg.close, seg.close))
    online = OnlineBackground(B) if mode == 'online' else None
    tracker, times = None, []
    for i, f in iter_frames(video, crop, start, stop):
        t0 = time.perf_counter()
        Bcur = online.B if online is not None else B
        labels, contours, fb = segment(f, Bcur, seg, K)
        if tracker is None:
            tracker = Tracker(f.shape, tp)
        tracker.step(i, labels, contours)
        if online is not None:
            online.update(fb, labels)
        times.append(time.perf_counter() - t0)
        if verbose and i % 300 == 0:
            print(f'  frame {i}: {len(contours)} objects, {np.mean(times[-300:]) * 1000:.1f} ms/frame')
    rows, events, contours, n_good, n_raw = tracker.finalize()
    for r in rows:
        r['t'] = r['frame'] / fps
    for e in events:
        e['t'] = e['frame'] / fps
    t = np.array(times[1:]) * 1000
    stats = dict(video=os.path.basename(video), mode=mode, background=bg_src, frames=len(times), fps=fps,
                 ms_per_frame_mean=round(float(t.mean()), 2), ms_per_frame_p95=round(float(np.percentile(t, 95)), 2),
                 confirmed_ids=n_good, raw_ids=n_raw,
                 events={k: sum(e['type'] == k for e in events) for k in ('merge', 'split', 'appear', 'lost', 'enter', 'exit')})
    final_bg = online.B if online is not None else B
    return rows, events, contours, stats, final_bg


def save_contours(contours, path, eps=0.5):
    """保存每帧每个目标的外轮廓（供 review/review_tool.py 高亮显示）。
    只是额外输出，不影响任何计算结果。轮廓用 approxPolyDP 轻度简化（误差 ≤ 0.5 像素）。"""
    frames, ids, offsets, pts = [], [], [0], []
    for f in sorted(contours):
        for i, c in contours[f]:
            ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2) if len(c) > 3 else c.reshape(-1, 2)
            frames.append(f)
            ids.append(i)
            pts.append(ap.astype(np.int16))
            offsets.append(offsets[-1] + len(ap))
    xy = np.concatenate(pts) if pts else np.zeros((0, 2), np.int16)
    np.savez_compressed(path, frame=np.array(frames, np.int32), id=np.array(ids, np.int32),
                        offset=np.array(offsets, np.int64), xy=xy)


def save_tables(rows, events, out_dir):
    cols = ['frame', 't', 'id', 'cx', 'cy', 'vx', 'vy', 'area', 'eq_diam', 'major', 'minor',
            'angle', 'circularity', 'solidity']
    with open(os.path.join(out_dir, 'objects.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r['frame'], f"{r['t']:.4f}", r['id']] + [f'{r[c]:.3f}' for c in cols[3:]])
    with open(os.path.join(out_dir, 'events.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frame', 't', 'type', 'ids', 'result_id', 'x', 'y'])
        for e in sorted(events, key=lambda e: (e['frame'], e['type'])):
            w.writerow([e['frame'], f"{e['t']:.4f}", e['type'], ' '.join(map(str, e['ids'])),
                        e['result'], f"{e['x']:.1f}", f"{e['y']:.1f}"])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('--bg', choices=['offline', 'online'], default='offline')
    ap.add_argument('--ref', nargs='*', default=[], help='同一次装片的其他视频或保存的 background.npy')
    ap.add_argument('--crop', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    ap.add_argument('--lo', type=float, default=0.06)
    ap.add_argument('--hi', type=float, default=0.12)
    ap.add_argument('--min-area', type=int, default=12)
    ap.add_argument('--percentile', type=float, default=90)
    ap.add_argument('--fps', type=float, help='视频文件里的帧率不准时手动指定')
    ap.add_argument('--out', default='result', help='输出目录（自动创建）')
    ap.add_argument('--render', action='store_true', help='另存叠加视频 <out>/overlay.mp4')
    ap.add_argument('--save-contours', action='store_true', help='另存每帧轮廓 <out>/contours.npz（人工审查工具需要）')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seg = SegParams(lo=a.lo, hi=a.hi, min_area=a.min_area)
    rows, events, contours, stats, B = run(a.video, a.crop, a.bg, a.ref, seg, percentile=a.percentile, fps=a.fps)
    save_tables(rows, events, a.out)
    np.save(os.path.join(a.out, 'background.npy'), B)
    if a.render:
        render(a.video, a.crop, rows, events, contours, os.path.join(a.out, 'overlay.mp4'), stats['fps'])
    if a.save_contours:
        save_contours(contours, os.path.join(a.out, 'contours.npz'))
    stats['params'] = dict(bg=a.bg, ref=[os.path.basename(r) for r in a.ref], crop=a.crop, lo=a.lo, hi=a.hi,
                           min_area=a.min_area, percentile=a.percentile)
    with open(os.path.join(a.out, 'stats.json'), 'w', encoding='utf-8') as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
