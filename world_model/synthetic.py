#!/usr/bin/env python3
"""
synthetic.py - stage 2 sanity check: can the models learn PZT effects when the data contain enough switches?

A simple simulated "chip" replaces the camera. Regions are pushed by each PZT pattern towards the nearest
node of that pattern's (made-up) standing wave, stop there, and jitter a little; idle = no push. Every
session also has its own slow drift (0 ... 4 px/s, like the drift seen in experiment A). The PZT pattern
changes at random every 0.5 ... 3 s. The samples have exactly the format of dataset.py, so the same models
and the same scoring are used.

    python world_model/synthetic.py --out results/world_model/synthetic

Trains on 1, 4 and 12 simulated sessions of 30 s (about 12, 50 and 150 switches) and tests on 4 other
sessions. This is not a model of the real acoustics; it only checks that the training code can learn an
action-dependent motion when the data hold it, and roughly how many switches that takes.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset as ds  # noqa: E402
import models as M    # noqa: E402

W, H, FPS = 1280, 1024, 60.0
PATTERNS = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]


def push(p, x, y):
    """Velocity (px/s) of a region at (x, y) under pattern p: towards the nearest node of a standing wave."""
    vx, vy = np.zeros_like(x), np.zeros_like(y)
    lam, vmax = 240.0, 250.0
    if p[0]:   # nodes on vertical lines: push along x
        vx += -vmax * np.sin(2 * np.pi * x / lam)
    if p[1]:   # nodes on horizontal lines: push along y
        vy += -vmax * np.sin(2 * np.pi * y / lam)
    if p[2]:   # nodes on diagonal lines
        u = (x + y) / np.sqrt(2)
        s = -vmax * np.sin(2 * np.pi * u / lam)
        vx += s / np.sqrt(2)
        vy += s / np.sqrt(2)
    if p[3]:   # towards the centre, slowing down near it
        dx, dy = W / 2 - x, H / 2 - y
        r = np.hypot(dx, dy) + 1e-6
        sp = vmax * np.tanh(r / 150.0)
        vx += sp * dx / r
        vy += sp * dy / r
    return vx, vy


class SyntheticExperiment:
    """Looks like dataset.Experiment to the models."""

    def __init__(self, name, seconds=30.0, n_regions=14, seed=0):
        rng = np.random.default_rng(seed)
        self.name, self.W, self.H, self.fps = name, W, H, FPS
        self.P = dict(ds.DEFAULTS)
        self.n_frames = n = int(seconds * FPS)
        self.wall_dist = np.full((H, W), 200.0, np.float32)
        self.wall_dir = np.zeros((H, W, 2), np.float32)
        # PZT schedule
        self.amp = np.zeros((n, 4), np.float32)
        f = 0
        pat = PATTERNS[0]
        while f < n:
            d = int(rng.uniform(0.5, 3.0) * FPS)
            self.amp[f:f + d] = pat
            f += d
            pat = PATTERNS[rng.integers(len(PATTERNS))]
        change = np.r_[True, np.any(np.abs(np.diff(self.amp, axis=0)) > 0, axis=1)]
        last = np.maximum.accumulate(np.where(change, np.arange(n), 0))
        self.tsc = np.minimum((np.arange(n) - last) / FPS, self.P['tsc_cap_s']).astype(np.float32)
        self.switches = int(change.sum() - 1)
        self.patterns = sorted({tuple(a.astype(int)) for a in self.amp})
        # regions
        self.area = np.exp(rng.uniform(np.log(20), np.log(2000), n_regions)).astype(np.float32)
        size_k = np.clip((self.area / 300.0) ** 0.3, 0.4, 2.0)
        drift = rng.uniform(0, 4.0) * np.array([np.cos(a := rng.uniform(0, 2 * np.pi)), np.sin(a)])
        x = rng.uniform(80, W - 80, n_regions)
        y = rng.uniform(80, H - 80, n_regions)
        self.x = np.zeros((n, n_regions), np.float32)
        self.y = np.zeros((n, n_regions), np.float32)
        for k in range(n):
            vx, vy = push(tuple(self.amp[k].astype(int)), x, y)
            x = np.clip(x + (vx * size_k + drift[0]) / FPS + rng.normal(0, 0.15, n_regions), 20, W - 20)
            y = np.clip(y + (vy * size_k + drift[1]) / FPS + rng.normal(0, 0.15, n_regions), 20, H - 20)
            self.x[k], self.y[k] = x, y

    def wall_features(self, x, y):
        return np.full(len(x), 200.0, np.float32), np.zeros((len(x), 2), np.float32)

    def step_action(self, f0):
        s = self.P['step_frames']
        a, b = max(0, f0), min(self.n_frames, f0 + s)
        return self.amp[a:b].mean(0) if b > a else np.zeros(4, np.float32)

    def graph(self, f):
        P = self.P
        s, nh, nf = P['step_frames'], P['n_history'], P['n_future']
        n = self.x.shape[1]
        pos = np.c_[self.x[f], self.y[f]].astype(np.float32)
        hist = np.zeros((n, nh, 2), np.float32)
        hmask = np.zeros((n, nh), np.float32)
        prev = pos
        for h in range(1, nh + 1):
            if f - h * s < 0:
                break
            p = np.c_[self.x[f - h * s], self.y[f - h * s]]
            hist[:, h - 1] = prev - p
            hmask[:, h - 1] = 1
            prev = p
        fut = np.full((n, nf, 2), np.nan, np.float32)
        for h in range(1, nf + 1):
            if f + h * s < self.n_frames:
                fut[:, h - 1] = np.c_[self.x[f + h * s], self.y[f + h * s]]
        la = np.log(self.area)
        shape = np.c_[la, np.full(n, 1.2), np.full(n, 0.8), np.full(n, 0.95), np.zeros(n), np.ones(n)].astype(np.float32)
        act = np.stack([self.step_action(f + j * s) for j in range(-1, nf)]).astype(np.float32)
        return dict(exp=self.name, frame=int(f), rows=np.arange(n), track=np.arange(n), pos=pos, hist=hist,
                    hmask=hmask, shape=shape, flags=np.c_[np.zeros(n), np.ones(n), np.zeros(n)].astype(np.float32),
                    status=np.array(['trusted'] * n, dtype=object), fut=fut,
                    fut_logA=np.repeat(la[:, None], nf, 1).astype(np.float32),
                    fut_ar=np.full((n, nf), 1.2, np.float32), fut_edge=np.zeros((n, nf), bool), act=act,
                    tsc=float(self.tsc[f]), field=np.zeros(3, np.float32), transition=False)

    def graphs(self, stride=1):
        return [self.graph(f) for f in range(3 * 6, self.n_frames - 60, stride)]


def evaluate(models, tests):
    rows = []
    for m in models:
        for h in (5, 10):
            e, es = [], []
            for te in tests:
                gs = te.graphs(stride=6)
                for g, o in zip(gs, m.predict(te, gs)):
                    ok = np.isfinite(g['fut'][:, h - 1, 0])
                    err = np.hypot(*(o['mu'][ok, h - 1] - g['fut'][ok, h - 1]).T)
                    e += list(err)
                    if M.switch_within(g, h):
                        es += list(err)
            rows.append(dict(model=m.name, horizon_s=h / 10, mean_error_px=float(np.mean(e)),
                             mean_error_at_switch_px=float(np.mean(es))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--sessions', type=int, nargs='*', default=[1, 4, 12])
    ap.add_argument("--members", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tests = [SyntheticExperiment(f'test{k}', seed=1000 + k) for k in range(4)]
    pool = [SyntheticExperiment(f'train{k}', seed=k) for k in range(max(a.sessions))]
    results = []
    for n in a.sessions:
        train = pool[:n]
        graphs = [g for e in train for g in e.graphs(stride=2)]
        switches = sum(e.switches for e in train)
        print(f'== {n} training session(s), {switches} switches, {len(graphs)} anchors', flush=True)
        models = [M.Persistence(), M.ResponseMap(),
                  M.GraphWorldModel(members=a.members, label='Graph world model', verbose=False).fit(train, graphs)]
        models[1].fit(train, graphs)
        for r in evaluate(models, tests):
            r.update(training_sessions=n, training_switches=switches)
            results.append(r)
            print(f"   {r['model']:20s} {r['horizon_s']:.1f} s  all {r['mean_error_px']:6.2f}  at switch {r['mean_error_at_switch_px']:6.2f}",
                  flush=True)
    json.dump(dict(note=__doc__.strip().splitlines()[2:9], test_sessions=len(tests),
                   test_switches=sum(t.switches for t in tests), results=results, runtime_s=time.time() - t0),
              open(os.path.join(a.out, 'synthetic.json'), 'w'), indent=1)
    print('written', a.out, f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
