#!/usr/bin/env python3
"""
dataset.py - stage 2: turn automatic tracking results into world-model samples

One sample ("graph") = one anchor frame f of one experiment:
    nodes      every tracked region present at f (automatic tracking, no manual review used for training)
    history    each node's centroid 0.1, 0.2, 0.3 s before f (inside the same piece of track)
    future     each node's centroid, log area and aspect ratio 0.1 ... 1.0 s after f (same piece of track)
    actions    the four PZT amplitudes during each 0.1 s step before and after f
    context    whole-field numbers at f (region count, total area, spread)

A "piece of track" (here: run) is a stretch of one track ID that is not interrupted by a merge / split
event; future and history are never taken across such a cut, because
the region before and after is not the same thing.

Rule-based quality flags (no learned trust model, by design):
    short run            runs shorter than 0.5 s are mostly flicker; kept as context, never used in the loss
    image edge           region cut by the image border: centroid is not the real centre; no loss

The manual review (review.json) is read only to label test samples for the evaluation ("trusted" subset).

Inputs per experiment (read only):
    <result>/objects.csv, events.csv, background.npy, contours.npz, review.json (optional)
    <result>/analysis/pzt_analysis/aligned_frames.csv   (round 3; PZT state of every frame)
"""
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, 'analysis')]
import analyze  # noqa: E402

N_PZT = 4
DEFAULTS = dict(
    step_frames=6,              # one model step = 0.1 s at 60 fps
    n_history=3,                # 0.3 s of history
    n_future=10,                # predict up to 1.0 s
    horizons=(1, 5, 10),        # reported horizons in steps: 0.1, 0.5, 1.0 s
    max_gap_frames=8,           # the tracker bridges up to 8 missing frames; interpolate inside a run
    min_run_s=0.5,              # shorter runs: context only
    wall_darkness=0.06, wall_dilate_px=2,
    full_vpp=0.8,               # amplitude scale: 0.8 Vpp (the driving amplitude of both experiments) = 1
    active_vpp=0.1,
    tsc_cap_s=3.0,              # time since the last PZT change, capped
)


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


class Experiment:
    """Everything the models need from one recording."""

    def __init__(self, name, result_dir, aligned_csv=None, P=None):
        self.name, self.dir = name, result_dir
        self.P = P = dict(DEFAULTS, **(P or {}))
        A = dict(analyze.DEFAULTS)
        D = analyze.Data(result_dir, A)
        self.D = D
        self.fps, self.W, self.H, self.n_frames = D.fps, D.W, D.H, D.n_frames
        static = analyze.static_structure(D.bg, P['wall_darkness'], P['wall_dilate_px'])
        self.static = static
        edge, border = analyze.per_row_geometry(D, static)
        self.edge_on_wall, self.border = np.nan_to_num(edge, nan=0.0), border
        # distance to the nearest wall / fixed debris pixel, and the direction pointing away from it
        free = (~static).astype(np.uint8) if static is not None else np.ones((self.H, self.W), np.uint8)
        self.wall_dist = cv2.distanceTransform(free, cv2.DIST_L2, 5).astype(np.float32)
        gx = cv2.Sobel(self.wall_dist, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(self.wall_dist, cv2.CV_32F, 0, 1, ksize=5)
        nrm = np.hypot(gx, gy) + 1e-6
        self.wall_dir = np.stack([gx / nrm, gy / nrm], -1)
        self._actions(aligned_csv or os.path.join(result_dir, 'analysis', 'pzt_analysis', 'aligned_frames.csv'))
        self._runs()
        self._field()

    # ------------------------------------------------------------------ PZT state per frame
    def _actions(self, path):
        P = self.P
        rows = read_csv(path)
        amp = np.zeros((self.n_frames, N_PZT), np.float32)
        trans = np.zeros(self.n_frames, bool)
        for r in rows:
            f = int(r['frame'])
            if f >= self.n_frames:
                continue
            for c in range(N_PZT):
                on = r[f'PZT{c + 1}_output'] == 'on'
                v = float(r[f'PZT{c + 1}_vpp'] or 0)
                amp[f, c] = v / P['full_vpp'] if on and v >= P['active_vpp'] else 0.0
            trans[f] = r.get('time_quality') == 'transition'
        self.amp, self.transition = amp, trans
        change = np.r_[True, np.any(np.abs(np.diff(amp, axis=0)) > 1e-6, axis=1)]
        last = np.maximum.accumulate(np.where(change, np.arange(self.n_frames), 0))
        self.tsc = np.minimum((np.arange(self.n_frames) - last) / self.fps, P['tsc_cap_s']).astype(np.float32)
        self.patterns = sorted({tuple((a > 0.5).astype(int)) for a in amp})

    def step_action(self, f0):
        """Mean amplitudes over frames f0 ... f0 + step - 1 (fraction of the step each PZT drove)."""
        s = self.P['step_frames']
        a, b = max(0, f0), min(self.n_frames, f0 + s)
        if b <= a:
            return np.zeros(N_PZT, np.float32)
        return self.amp[a:b].mean(0)

    # ------------------------------------------------------------------ runs (pieces of track)
    def _runs(self):
        D, P = self.D, self.P
        cuts, _, _ = analyze.track_ends(D, self.border)
        self.run_of = np.full(D.n, -1, int)
        self.runs = []
        for t in sorted(D.rows_by_track):
            ks = D.rows_by_track[t]
            f = D.F[ks]
            cut_frames = set(cuts[t].keys()) if t in cuts else set()
            cur = []
            for j, k in enumerate(ks):
                if cur:
                    prev = ks[j - 1]
                    gap = f[j] - D.F[prev]
                    crossed = any(D.F[prev] < c <= f[j] for c in cut_frames)
                    if gap > P['max_gap_frames'] + 1 or crossed:
                        self._close(t, cur)
                        cur = []
                cur.append(int(k))
            if cur:
                self._close(t, cur)
        # frame -> rows
        self.rows_at = defaultdict(list)
        for k in np.nonzero(self.run_of >= 0)[0]:
            self.rows_at[int(D.F[k])].append(int(k))

    def _close(self, track, ks):
        D, P = self.D, self.P
        ks = np.array(ks)
        f0, f1 = int(D.F[ks[0]]), int(D.F[ks[-1]])
        fr = np.arange(f0, f1 + 1)
        def interp(v):
            return np.interp(fr, D.F[ks], v).astype(np.float32)
        ar = D.major[ks] / np.maximum(D.minor[ks], 1e-6)
        run = dict(track=int(track), f0=f0, f1=f1, rows=ks,
                   x=interp(D.cx[ks]), y=interp(D.cy[ks]),
                   logA=interp(np.log(np.maximum(D.area[ks], 1.0))), aspect=interp(ar),
                   border=np.interp(fr, D.F[ks], self.border[ks].astype(float)) > 0.5,
                   ok=(f1 - f0) / self.fps >= P['min_run_s'])
        self.run_of[ks] = len(self.runs)
        self.runs.append(run)

    # ------------------------------------------------------------------ whole field at each frame
    def _field(self):
        D = self.D
        self.field = np.zeros((self.n_frames, 3), np.float32)   # n regions, total area, spread
        for f in range(self.n_frames):
            ks = self.rows_at.get(f, [])
            if not ks:
                continue
            a = D.area[ks]
            x, y = D.cx[ks], D.cy[ks]
            cx, cy = np.average(x, weights=a), np.average(y, weights=a)
            spread = np.sqrt(np.average((x - cx) ** 2 + (y - cy) ** 2, weights=a))
            self.field[f] = (len(ks), a.sum(), spread)

    # ------------------------------------------------------------------ lookups used during roll-outs
    def wall_features(self, x, y):
        """Distance to the nearest wall / fixed debris (px) and unit vector pointing away from it."""
        xi = np.clip(np.round(x).astype(int), 0, self.W - 1)
        yi = np.clip(np.round(y).astype(int), 0, self.H - 1)
        return self.wall_dist[yi, xi], self.wall_dir[yi, xi]

    # ------------------------------------------------------------------ samples
    def graph(self, f):
        """Sample at anchor frame f (None if no region)."""
        D, P = self.D, self.P
        s, nh, nf = P['step_frames'], P['n_history'], P['n_future']
        ks = self.rows_at.get(f, [])
        if not ks:
            return None
        n = len(ks)
        pos = np.zeros((n, 2), np.float32)
        hist = np.zeros((n, nh, 2), np.float32)           # displacement of step -j  (pos[-j+1] - pos[-j])
        hmask = np.zeros((n, nh), np.float32)
        fut = np.full((n, nf, 2), np.nan, np.float32)      # absolute centroid after step 1 ... nf
        fut_logA = np.full((n, nf), np.nan, np.float32)
        fut_ar = np.full((n, nf), np.nan, np.float32)
        fut_edge = np.zeros((n, nf), bool)
        shape = np.zeros((n, 6), np.float32)               # logA, aspect, circularity, solidity, sin2θ, cos2θ
        flags = np.zeros((n, 3), np.float32)               # at image edge, run ok, edge_on_wall share
        status = np.empty(n, dtype=object)
        track = np.zeros(n, int)
        for i, k in enumerate(ks):
            r = self.runs[self.run_of[k]]
            j0 = f - r['f0']
            pos[i] = r['x'][j0], r['y'][j0]
            prev = pos[i]
            for h in range(1, nh + 1):
                jj = j0 - h * s
                if jj < 0:
                    break
                p = np.array([r['x'][jj], r['y'][jj]], np.float32)
                hist[i, h - 1] = prev - p
                hmask[i, h - 1] = 1
                prev = p
            for h in range(1, nf + 1):
                jj = j0 + h * s
                if jj > r['f1'] - r['f0']:
                    break
                fut[i, h - 1] = r['x'][jj], r['y'][jj]
                fut_logA[i, h - 1] = r['logA'][jj]
                fut_ar[i, h - 1] = r['aspect'][jj]
                fut_edge[i, h - 1] = r['border'][jj]
            th = np.radians(D.angle[k]) * 2
            shape[i] = (np.log(max(D.area[k], 1.0)), D.major[k] / max(D.minor[k], 1e-6), D.circularity[k],
                        D.solidity[k], np.sin(th), np.cos(th))
            flags[i] = (float(self.border[k]), float(r['ok']), self.edge_on_wall[k])
            status[i] = D.status[k]
            track[i] = D.I[k]
        act = np.stack([self.step_action(f + j * s) for j in range(-1, nf)])   # step -1 (just before) ... nf-1
        return dict(exp=self.name, frame=int(f), rows=np.array(ks), track=track, pos=pos, hist=hist, hmask=hmask,
                    shape=shape, flags=flags, status=status, fut=fut, fut_logA=fut_logA, fut_ar=fut_ar,
                    fut_edge=fut_edge, act=act.astype(np.float32), tsc=float(self.tsc[f]),
                    field=self.field[f].copy(), transition=bool(self.transition[f]))

    def graphs(self, stride=1, need_future=1):
        out = []
        for f in range(0, self.n_frames - need_future * self.P['step_frames'], stride):
            g = self.graph(f)
            if g is not None:
                out.append(g)
        return out

    def summary(self):
        ok = [r for r in self.runs if r['ok']]
        return dict(experiment=self.name, frames=self.n_frames, fps=self.fps, regions_rows=int(self.D.n),
                    runs=len(self.runs), runs_used_in_loss=len(ok),
                    rows_in_runs_used=int(sum(len(r['rows']) for r in ok)),
                    action_patterns=[''.join(map(str, p)) for p in self.patterns])


def pattern_name(p):
    on = [f'PZT{i + 1}' for i, v in enumerate(p) if v]
    return '+'.join(on) if on else 'idle'
