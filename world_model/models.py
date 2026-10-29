#!/usr/bin/env python3
"""
models.py - stage 2: models that predict what the tracked regions do in the next 0.1 ... 1.0 s

Every model answers the same question: given the regions at anchor frame f (positions, recent motion,
shape, neighbours, walls) and the PZT amplitudes for the next 10 steps of 0.1 s, where is each region
after each step (mean), and how sure is the model (variance, if it has one).

    Persistence        nothing moves                                       (reference)
    ConstantVelocity   keeps the last 0.1 s displacement                    (reference)
    ResponseMap        "action-response table": for each PZT pattern, the average velocity at each position
                       of the channel, measured in the training recordings, early (< 0.5 s after a switch)
                       and late separately; rolled out step by step        (non-AI reference)
    GradientBoosting   gradient-boosted trees on the same inputs, one model per horizon (learned, no uncertainty)
    GraphWorldModel    the stage-2 AI model: a graph network in the style of the "learning to simulate"
                       graph network simulators (encode - process - decode, message passing between
                       neighbouring regions, PZT state as a global input), rolled out in 0.1 s steps,
                       trained on the 1 s roll-out with a beta-weighted Gaussian likelihood (beta = 1, so the
                       mean is learned like a least-squares fit and the spread alongside it); position also
                       enters as sin / cos at several frequencies; during training the recent motion of half of
                       the regions is hidden so that the model cannot just "keep going"; an ensemble of 5 networks
                       gives the uncertainty (spread between networks + each network's own variance).
                       GraphWorldModel(interactions=False) is the same network without neighbours.
                       context='motion' (default): the global input includes how the whole field moved in the
                       last 0.1 s (mean displacement and mean speed of all regions), updated during the roll-out;
                       this is how the model sees a slow background flow of the session.
                       context='field' (version 1): instead the region count, total area and spread.

Predictions are arrays per graph: mu (n, n_future, 2) absolute centroid after each step, var (n, n_future)
per-axis variance in px^2 (or None), logA and aspect (n, n_future) (or None).
"""
import copy
import math
import time

import numpy as np

import dataset as ds

N_PZT = ds.N_PZT


# =====================================================================================================
# helpers shared by the models
# =====================================================================================================
def tsc_rollout(act, tsc0, dt, cap):
    """Time since the last PZT change at the start of each step, from the step amplitudes act[0..nf]
    (act[0] = the step just before the anchor)."""
    out = []
    t = tsc0
    for j in range(1, len(act)):
        if j > 1:
            t = 0.0 if np.max(np.abs(act[j] - act[j - 1])) > 0.25 else min(t + dt, cap)
        out.append(t)
    return np.array(out, np.float32)


def pattern_of(a):
    return tuple((np.asarray(a) > 0.5).astype(int))


def hold_actions(g):
    """Counterfactual: the PZT state just before the anchor is kept for the whole horizon (no switch)."""
    act = g['act'].copy()
    act[1:] = act[0]
    return act


def switch_within(g, h):
    """Does the PZT state change during the first h steps (compared with the step just before the anchor)?"""
    return float(np.max(np.abs(g['act'][1:h + 1] - g['act'][0]))) > 0.25


# =====================================================================================================
# reference models
# =====================================================================================================
class Persistence:
    name = 'Persistence'
    has_var = False

    def fit(self, exps, graphs):
        return self

    def predict(self, exp, graphs, act_override=None):
        out = []
        for g in graphs:
            nf = g['fut'].shape[1]
            out.append(dict(mu=np.repeat(g['pos'][:, None], nf, 1), var=None,
                            logA=np.repeat(g['shape'][:, None, 0], nf, 1), aspect=np.repeat(g['shape'][:, None, 1], nf, 1)))
        return out


class ConstantVelocity(Persistence):
    name = 'Constant velocity'

    def predict(self, exp, graphs, act_override=None):
        out = super().predict(exp, graphs)
        for g, o in zip(graphs, out):
            v = g['hist'][:, 0] * g['hmask'][:, :1]
            nf = g['fut'].shape[1]
            o['mu'] = g['pos'][:, None] + v[:, None] * np.arange(1, nf + 1)[None, :, None]
        return out


class ResponseMap:
    """Average velocity field per PZT pattern (and early / late after a switch), kernel-smoothed over the
    channel. No learning beyond averaging: this is the 'action-response table' of the round plan."""
    name = 'Response map'
    has_var = True

    def __init__(self, sigma_px=40.0, grid_px=8, early_s=0.5, min_weight=3.0):
        self.sigma, self.grid, self.early, self.min_w = sigma_px, grid_px, early_s, min_weight

    def fit(self, exps, graphs):
        import cv2
        e0 = exps[0]
        self.W, self.H = e0.W, e0.H
        gw, gh = int(math.ceil(self.W / self.grid)), int(math.ceil(self.H / self.grid))
        acc = {}
        for g in graphs:
            ok = (g['flags'][:, 1] > 0) & (g['flags'][:, 0] == 0) & np.isfinite(g['fut'][:, 0, 0])
            if not ok.any():
                continue
            key = (pattern_of(g['act'][1]), g['tsc'] < self.early)
            v = g['fut'][ok, 0] - g['pos'][ok]
            p = g['pos'][ok]
            a = acc.setdefault(key, np.zeros((5, gh, gw), np.float64))   # w, w*vx, w*vy, w*vx^2, w*vy^2
            xi = np.clip((p[:, 0] / self.grid).astype(int), 0, gw - 1)
            yi = np.clip((p[:, 1] / self.grid).astype(int), 0, gh - 1)
            for c, val in enumerate((np.ones(len(v)), v[:, 0], v[:, 1], v[:, 0] ** 2, v[:, 1] ** 2)):
                np.add.at(a[c], (yi, xi), val)
        s = self.sigma / self.grid
        self.maps = {}
        for key, a in acc.items():
            b = np.stack([cv2.GaussianBlur(a[c].astype(np.float32), (0, 0), s) for c in range(5)])
            w = b[0] * (2 * math.pi * s * s)          # effective number of samples nearby
            with np.errstate(invalid='ignore', divide='ignore'):
                mx, my = b[1] / b[0], b[2] / b[0]
                var = np.maximum((b[3] / b[0] - mx ** 2 + b[4] / b[0] - my ** 2) / 2, 0)
            self.maps[key] = (w.astype(np.float32), np.nan_to_num(mx), np.nan_to_num(my), np.nan_to_num(var))
        self.patterns = sorted({k[0] for k in self.maps})
        return self

    def _velocity(self, pat, early, x, y):
        gh, gw = next(iter(self.maps.values()))[0].shape
        xi = np.clip((x / self.grid).astype(int), 0, gw - 1)
        yi = np.clip((y / self.grid).astype(int), 0, gh - 1)
        vx, vy, var = np.zeros(len(x)), np.zeros(len(x)), np.full(len(x), np.nan)
        for key in ((pat, early), (pat, not early)):
            m = self.maps.get(key)
            if m is None:
                continue
            w = m[0][yi, xi]
            use = (w >= self.min_w) & np.isnan(var)
            vx[use], vy[use], var[use] = m[1][yi, xi][use], m[2][yi, xi][use], m[3][yi, xi][use]
        unknown = np.isnan(var)
        var[unknown] = 4.0                 # unseen pattern / place: assume no motion, 2 px per step spread
        return vx, vy, var + 0.01

    def predict(self, exp, graphs, act_override=None):
        out = []
        P = exp.P
        for gi, g in enumerate(graphs):
            act = act_override[gi] if act_override is not None else g['act']
            nf = g['fut'].shape[1]
            tsc = tsc_rollout(act, g['tsc'], P['step_frames'] / exp.fps, P['tsc_cap_s'])
            x, y = g['pos'][:, 0].astype(float).copy(), g['pos'][:, 1].astype(float).copy()
            mu = np.zeros((len(x), nf, 2), np.float32)
            var = np.zeros((len(x), nf), np.float32)
            v_acc = np.zeros(len(x))
            for j in range(nf):
                vx, vy, vv = self._velocity(pattern_of(act[j + 1]), tsc[j] < self.early, x, y)
                x, y = x + vx, y + vy
                v_acc = v_acc + vv
                mu[:, j] = np.stack([x, y], 1)
                var[:, j] = v_acc
            o = Persistence().predict(exp, [g])[0]
            o.update(mu=mu, var=var)
            out.append(o)
        return out


# =====================================================================================================
# tabular features (gradient boosting)
# =====================================================================================================
def node_table(exp, g, act=None, horizon=None, R=150.0):
    """One row per region: the region, its neighbours, the field, the PZT state now and over the horizon."""
    act = g['act'] if act is None else act
    pos, n = g['pos'], len(g['pos'])
    d = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1])
    np.fill_diagonal(d, np.inf)
    r = np.sqrt(np.exp(g['shape'][:, 0]) / math.pi)
    gap = d - r[:, None] - r[None, :]
    near = np.argmin(d, 1) if n > 1 else np.zeros(n, int)
    within = d < R
    area = np.exp(g['shape'][:, 0])
    wd, wdir = exp.wall_features(pos[:, 0], pos[:, 1])
    cols = [pos[:, 0] / exp.W, pos[:, 1] / exp.H,
            g['hist'].reshape(n, -1), g['hmask'], g['shape'], g['flags'][:, :1],
            (wd - r)[:, None], wdir,
            within.sum(1)[:, None],
            np.where(n > 1, gap[np.arange(n), near], R)[:, None] if n > 1 else np.full((n, 1), R),
            (pos[near] - pos) if n > 1 else np.zeros((n, 2)),
            (within * area[None, :]).sum(1)[:, None],
            np.repeat(field_motion(g)[None], n, 0), np.full((n, 1), g['tsc']),
            np.repeat(act[:2].reshape(1, -1), n, 0)]
    if horizon is not None:
        mean_act = act[1:horizon + 1].mean(0)
        changed = float(np.max(np.abs(act[1:horizon + 1] - act[0])) > 0.25)
        cols += [np.repeat(mean_act[None], n, 0), np.full((n, 1), changed)]
    return np.concatenate([np.asarray(c, np.float32).reshape(n, -1) for c in cols], 1)


def field_motion(g):
    """How the whole field moved in the last 0.1 s: mean dx, mean dy, mean distance (px), all regions."""
    m = g['hmask'][:, 0] > 0
    if not m.any():
        return np.zeros(3, np.float32)
    d = g['hist'][m, 0]
    return np.array([d[:, 0].mean(), d[:, 1].mean(), np.hypot(d[:, 0], d[:, 1]).mean()], np.float32)


def loss_mask(g):
    """Rule-based: runs of at least 0.5 s, not cut by the image edge."""
    return (g['flags'][:, 1] > 0) & (g['flags'][:, 0] == 0)


class GradientBoosting:
    name = 'Gradient boosting'
    has_var = False

    def __init__(self, horizons=(1, 5, 10), seed=0):
        self.horizons, self.seed = horizons, seed

    def fit(self, exps, graphs, exp_of=None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.models = {}
        by_name = {e.name: e for e in exps}
        for h in self.horizons:
            X, Y = [], []
            for g in graphs:
                e = by_name[g['exp']]
                ok = loss_mask(g) & np.isfinite(g['fut'][:, h - 1, 0]) & ~g['fut_edge'][:, h - 1]
                if not ok.any():
                    continue
                X.append(node_table(e, g, horizon=h)[ok])
                Y.append(np.c_[g['fut'][ok, h - 1] - g['pos'][ok],
                               g['fut_logA'][ok, h - 1] - g['shape'][ok, 0],
                               g['fut_ar'][ok, h - 1] - g['shape'][ok, 1]])
            X, Y = np.concatenate(X), np.concatenate(Y)
            self.models[h] = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                            l2_regularization=1.0, random_state=self.seed).fit(X, Y[:, c])
                              for c in range(4)]
        return self

    def predict(self, exp, graphs, act_override=None):
        out = []
        for gi, g in enumerate(graphs):
            act = act_override[gi] if act_override is not None else g['act']
            nf = g['fut'].shape[1]
            n = len(g['pos'])
            mu = np.full((n, nf, 2), np.nan, np.float32)
            la = np.full((n, nf), np.nan, np.float32)
            ar = np.full((n, nf), np.nan, np.float32)
            for h, ms in self.models.items():
                X = node_table(exp, g, act=act, horizon=h)
                p = [m.predict(X) for m in ms]
                mu[:, h - 1] = g['pos'] + np.c_[p[0], p[1]]
                la[:, h - 1] = g['shape'][:, 0] + p[2]
                ar[:, h - 1] = g['shape'][:, 1] + p[3]
            out.append(dict(mu=mu, var=None, logA=la, aspect=ar))
        return out


# =====================================================================================================
# the graph world model (PyTorch)
# =====================================================================================================
def _torch():
    import torch
    return torch


POS_SCALE = 5.0        # network outputs displacements in units of 5 px per 0.1 s step
NODE_IN = 21
FOURIER_CYCLES = (1, 2, 4, 8, 16, 32)   # position encoding: sin / cos at these many cycles across the image
EDGE_IN = 5
GLOBAL_IN = 12


def build_net(latent=64, steps=3, interactions=True, node_in=NODE_IN):
    torch = _torch()
    nn = torch.nn

    def mlp(i, o, h=latent):
        return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.interactions = interactions
            self.node_enc = mlp(node_in, latent)
            self.glob_enc = mlp(GLOBAL_IN, latent)
            if interactions:
                self.edge_enc = mlp(EDGE_IN, latent)
                self.edge_upd = nn.ModuleList([mlp(4 * latent, latent) for _ in range(steps)])
                self.node_upd = nn.ModuleList([mlp(3 * latent, latent) for _ in range(steps)])
            else:
                self.node_upd = nn.ModuleList([mlp(2 * latent, latent) for _ in range(steps)])
            self.dec = mlp(2 * latent, 8)

        def forward(self, xn, xg, batch, src, dst, xe):
            h = self.node_enc(xn)
            g = self.glob_enc(xg)[batch]
            if self.interactions:
                e = self.edge_enc(xe) if len(src) else xe.new_zeros((0, h.shape[1]))
                for eu, nu in zip(self.edge_upd, self.node_upd):
                    if len(src):
                        e = e + eu(torch.cat([e, h[src], h[dst], g[dst]], 1))
                        agg = torch.zeros_like(h).index_add_(0, dst, e)
                    else:
                        agg = torch.zeros_like(h)
                    h = h + nu(torch.cat([h, agg, g], 1))
            else:
                for nu in self.node_upd:
                    h = h + nu(torch.cat([h, g], 1))
            return self.dec(torch.cat([h, g], 1))

    return Net()


class Batch:
    """Several anchor graphs packed together (tensors), ready for a roll-out."""

    def __init__(self, exp_index, graphs, exps, stats, act_override=None, device='cpu'):
        torch = _torch()
        T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        n_per = [len(g['pos']) for g in graphs]
        self.batch = T(np.repeat(np.arange(len(graphs)), n_per), torch.long)
        cat = lambda k: np.concatenate([g[k] for g in graphs])
        self.pos = T(cat('pos'))
        self.hist = T(cat('hist'))
        self.hmask = T(cat('hmask'))
        self.shape = T(cat('shape'))
        self.edge_flag = T(cat('flags')[:, :1])
        self.fut = T(np.nan_to_num(cat('fut')))
        self.fut_ok = T(np.isfinite(cat('fut')[..., 0]) & ~cat('fut_edge'), torch.bool)
        self.fut_logA = T(np.nan_to_num(cat('fut_logA')))
        self.fut_ar = T(np.nan_to_num(cat('fut_ar')))
        self.loss_ok = T(np.concatenate([loss_mask(g) for g in graphs]), torch.bool)
        acts = [act_override[i] if act_override is not None else g['act'] for i, g in enumerate(graphs)]
        self.act = T(np.stack(acts))                                       # (G, nf+1, 4)
        P = exps[0].P
        self.tsc = T(np.stack([tsc_rollout(a, g['tsc'], P['step_frames'] / exps[0].fps, P['tsc_cap_s'])
                               for a, g in zip(acts, graphs)]))           # (G, nf)
        f = np.stack([g['field'] for g in graphs])
        self.field = T(np.c_[f[:, 0] / 20.0, f[:, 1] / 1e4, f[:, 2] / 300.0])
        ei = np.repeat([exp_index[g['exp']] for g in graphs], n_per)
        self.exp_idx = T(ei, torch.long)
        self.W, self.H = exps[0].W, exps[0].H
        self.n_graphs = len(graphs)
        self.n_per = n_per
        self.stats = stats


class GraphWorldModel:
    has_var = True

    def __init__(self, interactions=True, members=5, radius_px=150.0, latent=64, steps=3, epochs=150,
                 patience=25, lr=1e-3, batch_graphs=32, train_stride=3, block_s=2.0, val_share=0.2,
                 shape_weight=0.3, seed=0, threads=None, verbose=True, label=None, context='motion',
                 fourier=True, beta=1.0, history_dropout=0.5):
        self.interactions, self.members, self.context = interactions, members, context
        self.fourier, self.beta, self.history_dropout = fourier, beta, history_dropout
        self.node_in = NODE_IN + (4 * len(FOURIER_CYCLES) if fourier else 0)
        self.R, self.latent, self.steps = radius_px, latent, steps
        self.epochs, self.patience, self.lr, self.bg = epochs, patience, lr, batch_graphs
        self.train_stride, self.block_s, self.val_share = train_stride, block_s, val_share
        self.shape_weight, self.seed, self.verbose = shape_weight, seed, verbose
        self.threads = threads
        self.name = label or ('Graph world model' if interactions else 'Graph world model, no neighbours')
        self.history = []

    # ------------------------------------------------------------------ maps and statistics
    def _prepare(self, exps):
        torch = _torch()
        if self.threads:
            torch.set_num_threads(self.threads)
        self.exp_index = {e.name: i for i, e in enumerate(exps)}
        self.wall_dist = torch.as_tensor(np.stack([e.wall_dist for e in exps]))
        self.wall_dir = torch.as_tensor(np.stack([e.wall_dir for e in exps]))

    def _maps_for(self, exps):
        """Wall maps for the experiments being predicted (may differ from the training ones)."""
        torch = _torch()
        return ({e.name: i for i, e in enumerate(exps)}, torch.as_tensor(np.stack([e.wall_dist for e in exps])),
                torch.as_tensor(np.stack([e.wall_dir for e in exps])))

    def _fit_stats(self, graphs):
        la = np.concatenate([g['shape'][:, 0] for g in graphs])
        self.stats = dict(logA_mean=float(la.mean()), logA_std=float(la.std() + 1e-6))

    # ------------------------------------------------------------------ one roll-out
    def _features(self, b, pos, hist, hmask, logA, aspect, j, maps):
        torch = _torch()
        exp_index, wd_map, wdir_map = maps
        W, H, R = b.W, b.H, self.R
        xi = pos[:, 0].round().long().clamp(0, W - 1)
        yi = pos[:, 1].round().long().clamp(0, H - 1)
        wd = wd_map[b.exp_idx, yi, xi]
        wdir = wdir_map[b.exp_idx, yi, xi]
        r = torch.sqrt(torch.exp(logA) / math.pi)
        st = b.stats
        xn = torch.cat([pos[:, :1] / W * 2 - 1, pos[:, 1:] / H * 2 - 1,
                        (hist / POS_SCALE).reshape(len(pos), -1), hmask,
                        ((logA - st['logA_mean']) / st['logA_std'])[:, None], (aspect - 1).clamp(0, 5)[:, None],
                        b.shape[:, 2:6], ((wd - r) / 20.0).clamp(-2, 5)[:, None], wdir, b.edge_flag], 1)
        if self.fourier:
            k = torch.as_tensor(FOURIER_CYCLES, dtype=torch.float32) * 2 * math.pi
            fx, fy = pos[:, :1] / W * k, pos[:, 1:] / H * k
            xn = torch.cat([xn, torch.sin(fx), torch.cos(fx), torch.sin(fy), torch.cos(fy)], 1)
        a_now = b.act[:, j + 1]
        a_prev = b.act[:, j]
        if self.context == 'motion':
            last = hist[:, 0] * hmask[:, :1]
            cnt = torch.zeros(b.n_graphs).index_add_(0, b.batch, hmask[:, 0]).clamp(min=1)
            mx = torch.zeros(b.n_graphs).index_add_(0, b.batch, last[:, 0]) / cnt
            my = torch.zeros(b.n_graphs).index_add_(0, b.batch, last[:, 1]) / cnt
            ms = torch.zeros(b.n_graphs).index_add_(0, b.batch, torch.linalg.norm(last, dim=1)) / cnt
            ctx = torch.stack([mx, my, ms], 1) / POS_SCALE
        else:
            ctx = b.field
        xg = torch.cat([a_now, a_prev, (b.tsc[:, j] / 3.0)[:, None], ctx], 1)
        if self.interactions:
            same = b.batch[:, None] == b.batch[None, :]
            d = torch.cdist(pos, pos)
            m = same & (d < R)
            m.fill_diagonal_(False)
            src, dst = torch.nonzero(m, as_tuple=True)
            rel = (pos[src] - pos[dst]) / R
            dd = d[src, dst][:, None] / R
            gap = (d[src, dst] - r[src] - r[dst])[:, None] / R
            xe = torch.cat([rel, dd, gap, (logA[src] - logA[dst])[:, None]], 1)
        else:
            src = dst = torch.zeros(0, dtype=torch.long)
            xe = torch.zeros((0, EDGE_IN))
        return xn, xg, src, dst, xe

    def _rollout(self, net, b, n_steps, maps, train=False):
        torch = _torch()
        pos = b.pos.clone()
        hist, hmask = b.hist.clone(), b.hmask.clone()
        if train and self.history_dropout:
            # training only: hide the recent motion of some regions, so the model cannot rely on "keep going"
            # alone and has to use position, neighbours and the PZT state (like the input noise of graph simulators)
            keep = (torch.rand(len(pos)) > self.history_dropout).float()[:, None]
            hmask = hmask * keep
            hist = hist * keep[:, :, None]
        logA, aspect = b.shape[:, 0].clone(), b.shape[:, 1].clone()
        var_xy = torch.zeros_like(pos)
        var_la = torch.zeros_like(logA)
        var_ar = torch.zeros_like(logA)
        mus, vxy, las, vla, ars, var_ = [], [], [], [], [], []
        for j in range(n_steps):
            xn, xg, src, dst, xe = self._features(b, pos, hist, hmask, logA, aspect, j, maps)
            o = net(xn, xg, b.batch, src, dst, xe)
            d = o[:, 0:2] * POS_SCALE
            lv = o[:, 2:4].clamp(-6, 6)
            pos = pos + d
            var_xy = var_xy + torch.exp(lv) * POS_SCALE ** 2
            logA = logA + o[:, 4] * 0.1
            var_la = var_la + torch.exp(o[:, 5].clamp(-8, 4)) * 0.01
            aspect = aspect + o[:, 6] * 0.1
            var_ar = var_ar + torch.exp(o[:, 7].clamp(-8, 4)) * 0.01
            hist = torch.cat([d[:, None], hist[:, :-1]], 1)
            hmask = torch.cat([torch.ones_like(hmask[:, :1]), hmask[:, :-1]], 1)
            mus.append(pos)
            vxy.append(var_xy)
            las.append(logA)
            vla.append(var_la)
            ars.append(aspect)
            var_.append(var_ar)
        st = lambda L: torch.stack(L, 1)
        return st(mus), st(vxy), st(las), st(vla), st(ars), st(var_)

    def _loss(self, b, out):
        torch = _torch()
        mu, vxy, la, vla, ar, var_ar = out
        nf = mu.shape[1]
        ok = b.fut_ok[:, :nf] & b.loss_ok[:, None]
        if ok.sum() == 0:
            return mu.sum() * 0
        s2 = POS_SCALE ** 2
        e2 = ((mu - b.fut[:, :nf]) ** 2) / s2
        nll = 0.5 * (e2 / (vxy / s2) + torch.log(vxy / s2))
        if self.beta:      # beta-NLL (Seitzer et al. 2022): keeps the mean learning where the variance is large
            nll = nll * (vxy / s2).detach() ** self.beta
        nll = nll.sum(-1)
        loss = nll[ok].mean()
        sh = ok & torch.isfinite(b.fut_logA[:, :nf])
        nll_la = 0.5 * ((la - b.fut_logA[:, :nf]) ** 2 / vla + torch.log(vla))
        nll_ar = 0.5 * ((ar - b.fut_ar[:, :nf]) ** 2 / var_ar + torch.log(var_ar))
        if self.beta:
            nll_la = nll_la * vla.detach() ** self.beta
            nll_ar = nll_ar * var_ar.detach() ** self.beta
        return loss + self.shape_weight * (nll_la[sh].mean() + nll_ar[sh].mean())

    # ------------------------------------------------------------------ training
    def fit(self, exps, graphs):
        torch = _torch()
        self._prepare(exps)
        self._fit_stats(graphs)
        maps = (self.exp_index, self.wall_dist, self.wall_dir)
        nf = graphs[0]['fut'].shape[1]
        fps = exps[0].fps
        # blocks of block_s seconds for bootstrap / early-stopping validation (never mixes experiments)
        block = np.array([self.exp_index[g['exp']] * 100000 + int(g['frame'] / fps // self.block_s) for g in graphs])
        ublocks = np.unique(block)
        self.nets = []
        t0 = time.time()
        for m in range(self.members):
            rng = np.random.default_rng(self.seed + 1000 * m)
            torch.manual_seed(self.seed + 1000 * m)
            if len(exps) >= 3:
                # several recordings: validate on whole held-out recordings, which is what the test measures
                n_val = max(1, int(round(len(exps) * self.val_share)))
                val_e = set(rng.choice([self.exp_index[e.name] for e in exps], n_val, replace=False))
                val_b = {u for u in ublocks if u // 100000 in val_e}
            else:
                val_b = set(rng.choice(ublocks, max(1, int(len(ublocks) * self.val_share)), replace=False))
            tr_b = [u for u in ublocks if u not in val_b]
            boot = rng.choice(tr_b, len(tr_b), replace=True)          # bootstrap over blocks
            idx_of = {u: np.nonzero(block == u)[0] for u in ublocks}
            tr = np.concatenate([idx_of[u][::self.train_stride] for u in boot])
            va = np.concatenate([idx_of[u][::self.train_stride] for u in val_b])
            net = build_net(self.latent, self.steps, self.interactions, self.node_in)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
            vbatches = [Batch(self.exp_index, [graphs[i] for i in va[k:k + 64]], exps, self.stats)
                        for k in range(0, len(va), 64)]
            best, best_state, bad = np.inf, None, 0
            for ep in range(self.epochs):
                net.train()
                order = rng.permutation(tr)
                tl = []
                for k in range(0, len(order), self.bg):
                    b = Batch(self.exp_index, [graphs[i] for i in order[k:k + self.bg]], exps, self.stats)
                    loss = self._loss(b, self._rollout(net, b, nf, maps, train=True))
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    tl.append(float(loss))
                net.eval()
                with torch.no_grad():
                    vl = float(np.mean([float(self._loss(vb, self._rollout(net, vb, nf, maps))) for vb in vbatches]))
                self.history.append(dict(member=m, epoch=ep, train=float(np.mean(tl)), val=vl))
                if vl < best - 1e-4:
                    best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
            if self.verbose:
                print(f'    {self.name}: member {m + 1}/{self.members} stopped at epoch {ep + 1}, '
                      f'best validation {best:.3f}  ({time.time() - t0:.0f} s)', flush=True)
        return self

    # ------------------------------------------------------------------ prediction
    def predict(self, exp, graphs, act_override=None, chunk=64):
        torch = _torch()
        maps = self._maps_for([exp])
        nf = graphs[0]['fut'].shape[1]
        out = []
        with torch.no_grad():
            for k in range(0, len(graphs), chunk):
                gs = graphs[k:k + chunk]
                ao = act_override[k:k + chunk] if act_override is not None else None
                b = Batch(maps[0], gs, [exp], self.stats, act_override=ao)
                res = [self._rollout(net, b, nf, maps) for net in self.nets]
                mu = torch.stack([r[0] for r in res])                  # (M, n, nf, 2)
                m_mu = mu.mean(0)
                alea = torch.stack([r[1] for r in res]).mean(0).mean(-1)
                epi = mu.var(0, unbiased=False).mean(-1) if len(res) > 1 else torch.zeros_like(alea)
                la = torch.stack([r[2] for r in res]).mean(0)
                ar = torch.stack([r[4] for r in res]).mean(0)
                start = 0
                for n in b.n_per:
                    sl = slice(start, start + n)
                    out.append(dict(mu=m_mu[sl].numpy(), var=(alea + epi)[sl].numpy(), var_epistemic=epi[sl].numpy(),
                                    logA=la[sl].numpy(), aspect=ar[sl].numpy()))
                    start += n
        return out

    # ------------------------------------------------------------------ save / load (for stage 3)
    def state(self):
        return dict(config=dict(interactions=self.interactions, members=self.members, radius_px=self.R,
                                latent=self.latent, steps=self.steps, context=self.context, fourier=self.fourier,
                                beta=self.beta), stats=self.stats,
                    nets=[n.state_dict() for n in self.nets])

    @classmethod
    def from_state(cls, st):
        m = cls(**{k: st['config'][k] for k in ('interactions', 'members', 'radius_px', 'latent', 'steps', 'context',
                                                 'fourier', 'beta') if k in st['config']})
        m.stats = st['stats']
        m.nets = []
        for sd in st['nets']:
            net = build_net(m.latent, m.steps, m.interactions, m.node_in)
            net.load_state_dict(sd)
            net.eval()
            m.nets.append(net)
        return m
