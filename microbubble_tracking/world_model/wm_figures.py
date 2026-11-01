#!/usr/bin/env python3
"""
wm_figures.py - stage 2 figures (English labels)

    errors.png          mean error 0.1 / 0.5 / 1.0 s ahead, every model, all regions / at a PZT switch / moved
    action_check.png    the same samples predicted with the real PZT switch and with "no switch"
    uncertainty.png     predicted spread vs. real error (is the model less sure when it is wrong?)
    examples.png        out-of-sample predictions at PZT switches, drawn on the video frame
    imagined.png        the graph world model "imagining" every PZT pattern from the same moment
    synthetic.png       simulated chip: error on unseen sessions vs. number of PZT switches in the training data
    events.png          merge / split within 0.5 and 1 s: ranking (AUC) and accuracy of the chance (Brier skill)
    field_map.png       whole field 1 s ahead: merge chances of every region, real and predicted density grid
"""
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), 'analysis')]
import figures as F  # noqa: E402  (analysis/figures.py: shared style)

plt = F.plt
import dataset as ds  # noqa: E402
import models as M  # noqa: E402

COL = {'Persistence': F.MUTED, 'Constant velocity': F.OTHER, 'Response map': '#eb6834',
       'Gradient boosting': '#eda100', 'Graph world model, no neighbours': '#e87ba4',
       'Graph world model': '#6250d6', 'Graph world model, trained on reviewed data': '#2a78d6',
       'Graph world model v1 (field size instead of field motion)': F.INK2,
       'Graph world model, without darkness / density inputs': '#9b8fe8', 'Base rate': F.MUTED, 'Gap rule': '#008300'}
OBS = '#1baf7a'
SHORT = {'Graph world model, no neighbours': 'Graph model,\nno\nneighbours',
         'Graph world model, trained on reviewed data': 'Graph model,\nreviewed\ndata',
         'Graph world model': 'Graph world\nmodel', 'Gradient boosting': 'Gradient\nboosting',
         'Constant velocity': 'Constant\nvelocity', 'Response map': 'Response\nmap',
         'Graph world model v1 (field size instead of field motion)': 'Graph model v1\n(field size input)',
         'Graph world model, without darkness / density inputs': 'Graph model,\nno darkness\n/ density',
         'Base rate': 'Base\nrate', 'Gap rule': 'Gap\nrule'}


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def short_fold(f):
    return (f.replace('same recordings, alternating 4 s blocks ', 'same recordings, 4 s blocks ')
             .replace('(even→odd)', 'even→odd').replace('(odd→even)', 'odd→even'))


def model_order(names):
    order = list(COL)
    return [m for m in order if m in names] + [m for m in names if m not in order]


# =====================================================================================================
def fig_errors(out):
    rows = read(os.path.join(out, 'metrics.csv'))
    if not rows:
        return None
    folds = list(dict.fromkeys(r['fold'] for r in rows))
    subs = ['all regions', 'PZT switch within horizon', 'moved ≥ 2 px']
    models = model_order(list(dict.fromkeys(r['model'] for r in rows)))
    fig, axs = plt.subplots(len(folds), len(subs), figsize=(15, 3.3 * len(folds)), squeeze=False)
    for i, fo in enumerate(folds):
        for j, s in enumerate(subs):
            ax = axs[i, j]
            hs = sorted({r['horizon_s'] for r in rows if r['fold'] == fo and r['subset'] == s})
            w = 0.8 / len(models)
            n = None
            for k, m in enumerate(models):
                v = [next((r['mean_error_px'] for r in rows if r['fold'] == fo and r['subset'] == s
                           and r['horizon_s'] == h and r['model'] == m), np.nan) for h in hs]
                ax.bar(np.arange(len(hs)) + (k - (len(models) - 1) / 2) * w, v, w * 0.92, color=COL.get(m, F.INK2),
                       label=m)
                n = next((int(r['n']) for r in rows if r['fold'] == fo and r['subset'] == s and r['horizon_s'] == hs[-1]), n)
            ax.set_xticks(range(len(hs)))
            ax.set_xticklabels([f'{h:g} s ahead' for h in hs])
            ax.set_title(f'{short_fold(fo)}\n{s}' + (f'  (n = {n} at 1 s)' if n else ''), fontsize=8.5)
            if j == 0:
                ax.set_ylabel('mean position error (px)')
            ax.grid(axis='x', visible=False)
    axs[0, -1].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    fig.tight_layout()
    return F.save(fig, os.path.join(out, 'figures', 'errors.png'))


def fig_action_check(out, horizon=1.0):
    rows = [r for r in read(os.path.join(out, 'action_check.csv')) if r['horizon_s'] == horizon]
    if not rows:
        return None
    folds = list(dict.fromkeys(r['fold'] for r in rows))
    models = model_order(list(dict.fromkeys(r['model'] for r in rows)))
    nc = 2
    nr = int(np.ceil(len(folds) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(12, 3.6 * nr), squeeze=False)
    for ax, fo in zip(axs.ravel(), folds):
        ms = [m for m in models if any(r['fold'] == fo and r['model'] == m for r in rows)]
        for x, m in enumerate(ms):
            r = next(r for r in rows if r['fold'] == fo and r['model'] == m)
            ax.bar(x - 0.2, r['error_real_action_px'], 0.38, color=COL.get(m, F.INK2))
            ax.bar(x + 0.2, r['error_no_switch_px'], 0.38, color=COL.get(m, F.INK2), alpha=0.35, hatch='//',
                   edgecolor='white')
            ax.text(x, max(r['error_real_action_px'], r['error_no_switch_px']) * 1.03,
                    f"{r['real_action_closer_share'] * 100:.0f}%", ha='center', va='bottom', fontsize=8, color=F.INK2)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([SHORT.get(m, m) for m in ms], fontsize=7.5)
        ax.set_ylabel(f'mean position error {horizon:g} s ahead (px)')
        ax.set_title(short_fold(fo), fontsize=9)
        ax.grid(axis='x', visible=False)
        ax.margins(y=0.15)
    for ax in axs.ravel()[len(folds):]:
        ax.axis('off')
    fig.suptitle('Samples whose next 1 s contains a PZT switch. Solid: predicted with the real switch; hatched: as if '
                 'there were no switch.\n% = share of samples where the real switch gives the closer prediction '
                 '(50% = the model gains nothing from knowing the action).', fontsize=9, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return F.save(fig, os.path.join(out, 'figures', 'action_check.png'))


def fig_uncertainty(out, horizon=1.0, models=('Response map', 'Graph world model')):
    bins = [r for r in read(os.path.join(out, 'uncertainty_bins.csv')) if r['horizon_s'] == horizon]
    cov = read(os.path.join(out, 'uncertainty.csv'))
    if not bins:
        return None
    folds = list(dict.fromkeys(r['fold'] for r in bins))
    nc = 2
    nr = int(np.ceil(len(folds) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(11, 4 * nr), squeeze=False)
    for ax, fo in zip(axs.ravel(), folds):
        mx = 1
        for m in models:
            rs = sorted([r for r in bins if r['fold'] == fo and r['model'] == m], key=lambda r: r['quintile'])
            if not rs:
                continue
            x = [r['predicted_spread_px'] for r in rs]
            y = [r['mean_error_px'] for r in rs]
            c = next((r for r in cov if r['fold'] == fo and r['model'] == m and r['horizon_s'] == horizon
                      and r['subset'] == 'all regions'), None)
            lab = m + (f": 90% range holds {c['inside_90pct_range'] * 100:.0f}% of samples, "
                       f"rank correlation {c['spearman_spread_vs_error']:.2f}" if c else '')
            ax.plot(x, y, '-', marker='o', ms=4, color=COL.get(m, F.INK2), label=lab)
            mx = max(mx, max(x), max(y))
        ax.plot([0.1, mx], [0.125, 1.25 * mx], ':', color=F.MUTED, lw=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f'predicted spread {horizon:g} s ahead (px); samples split into fifths')
        ax.set_ylabel('real mean error (px)')
        ax.set_title(short_fold(fo), fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc='upper left')
    for ax in axs.ravel()[len(folds):]:
        ax.axis('off')
    fig.suptitle('Is the model less sure when it is wrong? Dotted line: the error a correct spread would give. '
                 'Above it: too confident; below it: too cautious.', fontsize=9, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return F.save(fig, os.path.join(out, 'figures', 'uncertainty.png'))


# =====================================================================================================
def frame_image(exp, f, video):
    if video and os.path.exists(video):
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        cap.release()
        if ok:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bg = np.clip(exp.D.bg, 0, 255).astype(np.uint8)
    return np.dstack([bg] * 3)


def switch_anchors(exp, graphs, k=3, before_steps=1):
    """Anchor 0.1 s before each PZT switch, ranked by how much the regions then moved in 1 s."""
    sw = [f for f in range(1, exp.n_frames) if np.max(np.abs(exp.amp[f] - exp.amp[f - 1])) > 0.25]
    by_f = {g['frame']: g for g in graphs}
    cands = []
    for f in sw:
        a = f - before_steps * exp.P['step_frames']
        g = by_f.get(a)
        if g is None:
            continue
        ok = M.loss_mask(g) & np.isfinite(g['fut'][:, -1, 0])
        moved = np.nansum(np.hypot(*(g['fut'][ok, -1] - g['pos'][ok]).T))
        cands.append((moved, a, f))
    cands.sort(reverse=True)
    return [(a, f) for _, a, f in cands[:k]]


def crop_box(points, W, H, margin=60, min_size=320):
    p = np.concatenate(points)
    p = p[np.isfinite(p).all(1)]
    x0, y0 = p.min(0) - margin
    x1, y1 = p.max(0) + margin
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0, min_size)
    x0, y0 = int(np.clip(cx - s / 2, 0, max(0, W - s))), int(np.clip(cy - s / 2, 0, max(0, H - s)))
    return x0, y0, int(min(s, W)), int(min(s, H))


def fig_examples(out, exps, graphs, videos, fold_models):
    panels = []
    for test, ms in (fold_models or {}).items():
        exp = exps[test]
        for a, f in switch_anchors(exp, graphs[test], k=2):
            panels.append((test, a, f, ms))
    if not panels:
        return None
    fig, axs = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.2), squeeze=False)
    for ax, (test, a, f, ms) in zip(axs[0], panels):
        exp = exps[test]
        g = next(g for g in graphs[test] if g['frame'] == a)
        ok = M.loss_mask(g)
        preds = {name: m.predict(exp, [g])[0] for name, m in ms.items()
                 if name in ('Graph world model', 'Response map')}
        obs = np.concatenate([g['pos'][:, None], g['fut']], 1)
        pts = [obs[ok].reshape(-1, 2)] + [np.concatenate([g['pos'][ok, None], p['mu'][ok]], 1).reshape(-1, 2)
                                          for p in preds.values()]
        moving = ok & (np.nan_to_num(np.hypot(*(g['fut'][:, -1] - g['pos']).T)) >= 3)
        if moving.any():
            pts = [obs[moving].reshape(-1, 2)] + [np.concatenate([g['pos'][moving, None], p['mu'][moving]], 1).reshape(-1, 2)
                                                  for p in preds.values()]
        x0, y0, w, h = crop_box(pts, exp.W, exp.H)
        img = frame_image(exp, a, videos.get(test))
        ax.imshow(img[y0:y0 + h, x0:x0 + w], cmap='gray', extent=(x0, x0 + w, y0 + h, y0))
        for i in np.nonzero(ok)[0]:
            o = obs[i]
            o = o[np.isfinite(o).all(1)]
            ax.plot(o[:, 0], o[:, 1], '-', color=OBS, lw=2.2, alpha=0.95)
            for name, p in preds.items():
                q = np.concatenate([g['pos'][i:i + 1], p['mu'][i]])
                ax.plot(q[:, 0], q[:, 1], '-', color=COL[name], lw=1.4)
                if p.get('var') is not None and name == 'Graph world model':
                    sd = float(np.sqrt(p['var'][i, -1]))
                    ax.add_patch(plt.Circle(q[-1], sd, fill=False, color=COL[name], lw=0.8, alpha=0.8))
            ax.plot(*g['pos'][i], 'o', ms=2.5, color='white', mec=F.INK, mew=0.4)
        ax.set_xlim(x0, x0 + w)
        ax.set_ylim(y0 + h, y0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        before = ds.pattern_name(M.pattern_of(exp.amp[f - 1]))
        after = ds.pattern_name(M.pattern_of(exp.amp[f]))
        ax.set_title(f'test on {test}, model trained without {test}\n{a / exp.fps:.2f} s; switch {before} → {after} '
                     f'0.1 s later', fontsize=8.5)
    handles = [plt.Line2D([], [], color=OBS, lw=2.2, label='what happened (next 1 s)'),
               plt.Line2D([], [], color=COL['Graph world model'], lw=1.4, label='graph world model (circle: 1 spread at 1 s)'),
               plt.Line2D([], [], color=COL['Response map'], lw=1.4, label='response map')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return F.save(fig, os.path.join(out, 'figures', 'examples.png'))


def fig_imagined(out, exps, graphs, videos, fold_models):
    """One moment, every PZT pattern: what the graph world model expects in the next 1 s."""
    if not fold_models:
        return None
    # the fold whose model was trained on the experiment with the most patterns
    test = max(fold_models, key=lambda t: len({p for n, e in exps.items() if n != t for p in e.patterns}))
    model = fold_models[test].get('Graph world model')
    if model is None:
        return None
    exp = exps[test]
    sw = switch_anchors(exp, graphs[test], k=1)
    if not sw:
        return None
    a, f = sw[0]
    g = next(g for g in graphs[test] if g['frame'] == a)
    pats = sorted({p for n, e in exps.items() if n != test for p in e.patterns}, key=lambda p: (sum(p), p[::-1]))
    cand = []
    for p in pats:
        act = np.repeat(np.array(p, np.float32)[None], g['act'].shape[0], 0)
        act[0] = g['act'][0]
        cand.append(act)
    preds = model.predict(exp, [g] * len(cand), act_override=cand)
    ok = M.loss_mask(g)
    allp = [g['pos'][ok]] + [p['mu'][ok, -1] for p in preds] + [g['fut'][ok, -1]]
    x0, y0, w, h = crop_box(allp, exp.W, exp.H, margin=80, min_size=420)
    img = frame_image(exp, a, videos.get(test))
    cols = 4
    rows = int(np.ceil((len(pats) + 1) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows), squeeze=False)
    items = [('what really happened\n(' + ds.pattern_name(M.pattern_of(exp.amp[f])) + ')', None)] + \
            [(f'imagined: {ds.pattern_name(p)} for 1 s', pr) for p, pr in zip(pats, preds)]
    for ax, (title, pr) in zip(axs.ravel(), items):
        ax.imshow(img[y0:y0 + h, x0:x0 + w], cmap='gray', extent=(x0, x0 + w, y0 + h, y0))
        for i in np.nonzero(ok)[0]:
            if pr is None:
                e = g['fut'][i, -1]
                c = OBS
            else:
                e = pr['mu'][i, -1]
                c = COL['Graph world model']
            if not np.isfinite(e).all():
                continue
            d = e - g['pos'][i]
            if np.hypot(*d) >= 1.0:
                ax.annotate('', xy=e, xytext=g['pos'][i], arrowprops=dict(arrowstyle='-|>', color=c, lw=1.3,
                                                                          mutation_scale=8))
            ax.plot(*g['pos'][i], 'o', ms=2.5, color='white', mec=F.INK, mew=0.4)
        ax.set_xlim(x0, x0 + w)
        ax.set_ylim(y0 + h, y0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, fontsize=9)
    for ax in axs.ravel()[len(items):]:
        ax.axis('off')
    fig.suptitle(f'{test} at {a / exp.fps:.2f} s, model trained without {test}: arrows = displacement over the next 1 s',
                 fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    return F.save(fig, os.path.join(out, 'figures', 'imagined.png'))


def fig_synthetic(out):
    import json
    p = os.path.join(out, 'synthetic', 'synthetic.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    rows = [r for r in d['results'] if r['horizon_s'] == 1.0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in ((axs[0], 'mean_error_px', 'all regions'),
                           (axs[1], 'mean_error_at_switch_px', 'next 1 s contains a PZT switch')):
        for m in ('Persistence', 'Response map', 'Graph world model'):
            rs = sorted([r for r in rows if r['model'] == m], key=lambda r: r['training_switches'])
            x = [r['training_switches'] for r in rs]
            ax.plot(x, [r[key] for r in rs], '-o', color=COL[m], label=m, lw=1.8, ms=5)
        ax.set_xscale('log')
        ax.set_xticks([r['training_switches'] for r in rows if r['model'] == 'Persistence'])
        ax.set_xticklabels([f"{r['training_switches']}\n({r['training_sessions']} session{'s' if r['training_sessions'] > 1 else ''})"
                            for r in rows if r['model'] == 'Persistence'])
        ax.set_xlabel('PZT switches in the training data')
        ax.set_ylabel('mean position error 1 s ahead (px)')
        ax.set_title(f'Simulated chip, {d["test_sessions"]} unseen test sessions: {title}', fontsize=9)
        ax.minorticks_off()
    axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return F.save(fig, os.path.join(out, 'figures', 'synthetic.png'))


def fig_events(out, kind='merge'):
    rows = [r for r in read(os.path.join(out, 'events.csv')) if r['event'] == kind and r['horizon_s'] in (0.5, 1.0)]
    if not rows:
        return None
    folds = list(dict.fromkeys(r['fold'] for r in rows))
    first = ['Base rate', 'Gap rule', 'Gradient boosting']
    present = list(dict.fromkeys(r['model'] for r in rows))
    models = [m for m in first if m in present] + [m for m in model_order(present) if m not in first]
    n_m = [sum(any(r['fold'] == fo and r['model'] == m for r in rows) for m in models) for fo in folds]
    fig, axs = plt.subplots(2, len(folds), figsize=(0.85 * sum(n_m) + 1.2 * len(folds), 7), squeeze=False,
                            gridspec_kw=dict(width_ratios=n_m))
    for j, fo in enumerate(folds):
        ms = [m for m in models if any(r['fold'] == fo and r['model'] == m for r in rows)]
        for i, (key, lab) in enumerate((('auc', 'AUC (0.5 = guessing, 1 = perfect ranking)'),
                                        ('brier_skill_vs_base_rate', 'Brier skill vs. base rate (> 0 = better)'))):
            ax = axs[i, j]
            for x, m in enumerate(ms):
                for k, (h, alpha) in enumerate(((0.5, 0.45), (1.0, 1.0))):
                    r = next((r for r in rows if r['fold'] == fo and r['model'] == m and r['horizon_s'] == h), None)
                    if r is None or not np.isfinite(r[key]):
                        continue
                    ax.bar(x + (k - 0.5) * 0.38, r[key], 0.36, color=COL.get(m, F.INK2), alpha=alpha)
            ax.axhline(0.5 if key == 'auc' else 0.0, color=F.MUTED, lw=0.8, ls=':')
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels([SHORT.get(m, m) for m in ms], fontsize=6, rotation=0)
            ax.set_ylabel(lab, fontsize=8)
            if key == 'auc':
                ax.set_ylim(0.3, 1.0)
            else:
                lo = min([r[key] for r in rows if r['fold'] == fo and np.isfinite(r[key])] + [-0.1])
                ax.set_ylim(max(lo * 1.1, -1.0), 0.6)
            ax.grid(axis='x', visible=False)
            if i == 0:
                n = next((int(r['positives']) for r in rows if r['fold'] == fo and r['horizon_s'] == 1.0), 0)
                ax.set_title(f'{short_fold(fo)}\n{n} region-anchors that {kind} within 1 s', fontsize=8.5)
    fig.suptitle(f'Will a region {kind} with another within 0.5 s (light) / 1 s (dark)?', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return F.save(fig, os.path.join(out, 'figures', 'events.png'))


def fig_field_map(out, exps, graphs, videos, fold_models):
    """Whole field 1 s ahead, on a recording the model never saw."""
    import run as R
    if not fold_models:
        return None
    test = next(iter(fold_models))
    model = fold_models[test].get('Graph world model')
    exp = exps[test]
    if model is None or not getattr(exp, 'has_darkness', False):
        return None
    # the moment with the most merges coming up within 1 s
    best = max(graphs[test], key=lambda g: (np.nansum(g['ev_merge']), -g['frame']))
    g = best
    p = model.predict(exp, [g])[0]
    f2 = g['frame'] + 10 * exp.P['step_frames']
    obs = exp.grid_area[f2]
    pred = R.render_grid(exp, p['mu'][:, -1], np.exp(p['logA'][:, -1]))
    pers = R.render_grid(exp, g['pos'], np.exp(g['shape'][:, 0]))
    img = frame_image(exp, g['frame'], videos.get(test))
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
    ext = (0, exp.W, exp.H, 0)
    ax = axs[0]
    ax.imshow(img, extent=ext)
    pm = p['p_merge'][:, -1]
    lab = R.M.event_label(g, 'merge', 10)
    sc = ax.scatter(g['pos'][:, 0], g['pos'][:, 1], c=pm, cmap='plasma', vmin=0, vmax=1, s=28, edgecolors='white',
                    linewidths=0.5)
    hit = lab == 1
    ax.scatter(g['pos'][hit, 0], g['pos'][hit, 1], s=120, facecolors='none', edgecolors=OBS, linewidths=1.6)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01).set_label('predicted chance of merging within 1 s', fontsize=7)
    ax.set_title(f'{test} at {g["frame"] / exp.fps:.2f} s (model trained without {test})\n'
                 'green ring: really merged within 1 s', fontsize=8.5)
    vmax = max(obs.max(), pred.max(), pers.max(), 1)
    for ax, grid, title in ((axs[1], pers, 'nothing moves (regions as discs)'),
                            (axs[2], pred, 'graph world model (regions as discs)'),
                            (axs[3], obs, 'what really happened')):
        ax.imshow(img, extent=ext, alpha=0.35)
        ax.imshow(np.ma.masked_less(grid, 1), extent=ext, cmap='magma_r', vmin=0, vmax=vmax, alpha=0.85,
                  interpolation='nearest')
        ms = R.misplaced_share(grid, obs) if grid is not obs else None
        ax.set_title(f'density grid 1 s later: {title}' + (f'\narea in the wrong place: {ms * 100:.0f}%' if ms is not None
                                                        else '\n(real outlines)'), fontsize=8.5)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    return F.save(fig, os.path.join(out, 'figures', 'field_map.png'))


def make_all(out, exps, graphs, videos, fold_models=None, final=None):
    if fold_models is None:
        fold_models = load_fold_models(out, exps, graphs)
    files = [fig_errors(out), fig_action_check(out), fig_uncertainty(out),
             fig_examples(out, exps, graphs, videos, fold_models), fig_imagined(out, exps, graphs, videos, fold_models),
             fig_synthetic(out), fig_events(out), fig_field_map(out, exps, graphs, videos, fold_models)]
    return [os.path.relpath(f, out) for f in files if f]


def save_fold_models(out, fold_models):
    import torch
    d = os.path.join(out, 'fold_models')
    os.makedirs(d, exist_ok=True)
    for test, ms in fold_models.items():
        m = ms.get('Graph world model')
        if m is not None:
            torch.save(m.state(), os.path.join(d, f'graph_world_model_without_{test}.pt'))


def load_fold_models(out, exps, graphs):
    import torch
    res = {}
    for test in exps:
        p = os.path.join(out, 'fold_models', f'graph_world_model_without_{test}.pt')
        if not os.path.exists(p):
            continue
        gm = M.GraphWorldModel.from_state(torch.load(p))
        train = [e for n, e in exps.items() if n != test]
        rm = M.ResponseMap().fit(train, [g for n in exps if n != test for g in graphs[n]])
        res[test] = {'Graph world model': gm, 'Response map': rm}
    return res
