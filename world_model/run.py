#!/usr/bin/env python3
"""
run.py - stage 2: train and test the world-action models on the recorded experiments

    python world_model/run.py --exp A=results/A_offline --exp B=results/B_offline \
        --video A=data/raw/expA.mp4 --video B=data/raw/expB.mp4 --out results/world_model

What it does
    1. builds the samples of every experiment from the automatic tracking (dataset.py)
    2. leave-one-experiment-out: trains every model on all experiments but one and tests on the left-out one
       (never on neighbouring frames of the same recording)
    3. scores the predictions 0.1, 0.5 and 1.0 s ahead: all regions, regions about to see a PZT switch,
       regions that really moved, reviewed-trusted regions, size classes, wall contact, whole field
    4. checks the uncertainty (are the error ranges right? is the model less sure when it is wrong?)
    5. checks that the model uses the PZT input: the same samples predicted with the real switch and with
       "no switch"; the real one should be closer to what happened
    6. trains the final graph world model on all experiments and saves it (model/) for stage 3
    7. writes tables, figures and report.json

Inputs are only read. Output: --out folder.
"""
import argparse
import csv
import datetime
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset as ds      # noqa: E402
import models as M        # noqa: E402

HORIZONS = (1, 5, 10)                    # steps of 0.1 s
SIZE = [(0, 100, 'single (<100 px²)'), (100, 300, 'small group (100–300 px²)'), (300, 1e9, 'cluster (≥300 px²)')]
MOVING_PX = 2.0                          # a region "moved" when it went at least 2 px over the horizon
CHI2_2 = {0.5: 1.3863, 0.9: 4.6052}      # 2-D Gaussian: squared standardised radius holding 50 % / 90 %


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# =====================================================================================================
# flatten predictions to one row per (region, horizon)
# =====================================================================================================
def flatten(exp, graphs, preds, seen_patterns):
    """Arrays over all evaluable (graph, node, horizon) samples. preds: {model name: list of outputs}."""
    rec = {k: [] for k in ('frame', 'track', 'h', 'x0', 'y0', 'ox', 'oy', 'la0', 'ola', 'ar0', 'oar',
                           'switch', 'just', 'unseen', 'trusted', 'wall', 'gi', 'ni')}
    mp = {m: dict(mx=[], my=[], var=[], la=[], ar=[]) for m in preds}
    for gi, g in enumerate(graphs):
        base_ok = M.loss_mask(g)
        for h in HORIZONS:
            ok = base_ok & np.isfinite(g['fut'][:, h - 1, 0]) & ~g['fut_edge'][:, h - 1]
            idx = np.nonzero(ok)[0]
            if not len(idx):
                continue
            n = len(idx)
            seq = g['act'][1:h + 1]
            switch = M.switch_within(g, h)
            unseen = any(M.pattern_of(a) not in seen_patterns for a in seq)
            rec['frame'] += [g['frame']] * n
            rec['track'] += list(g['track'][idx])
            rec['h'] += [h] * n
            rec['x0'] += list(g['pos'][idx, 0])
            rec['y0'] += list(g['pos'][idx, 1])
            rec['ox'] += list(g['fut'][idx, h - 1, 0])
            rec['oy'] += list(g['fut'][idx, h - 1, 1])
            rec['la0'] += list(g['shape'][idx, 0])
            rec['ola'] += list(g['fut_logA'][idx, h - 1])
            rec['ar0'] += list(g['shape'][idx, 1])
            rec['oar'] += list(g['fut_ar'][idx, h - 1])
            rec['switch'] += [switch] * n
            rec['just'] += [(not switch) and g['tsc'] < 0.5] * n
            rec['unseen'] += [unseen] * n
            rec['trusted'] += list(g['status'][idx] == 'trusted')
            rec['wall'] += list(g['flags'][idx, 2] > 0.2)
            rec['gi'] += [gi] * n
            rec['ni'] += list(idx)
            for m, ps in preds.items():
                o = ps[gi]
                mp[m]['mx'] += list(o['mu'][idx, h - 1, 0])
                mp[m]['my'] += list(o['mu'][idx, h - 1, 1])
                mp[m]['var'] += list(o['var'][idx, h - 1]) if o.get('var') is not None else [np.nan] * n
                mp[m]['la'] += list(o['logA'][idx, h - 1]) if o.get('logA') is not None else [np.nan] * n
                mp[m]['ar'] += list(o['aspect'][idx, h - 1]) if o.get('aspect') is not None else [np.nan] * n
    rec = {k: np.array(v) for k, v in rec.items()}
    mp = {m: {k: np.array(v, float) for k, v in d.items()} for m, d in mp.items()}
    return rec, mp


def subsets(rec):
    obs = np.hypot(rec['ox'] - rec['x0'], rec['oy'] - rec['y0'])
    area = np.exp(rec['la0'])
    S = {'all regions': np.ones(len(obs), bool),
         'PZT switch within horizon': rec['switch'],
         'within 0.5 s after a switch': rec['just'],
         'no switch nearby': ~rec['switch'] & ~rec['just'],
         'moved ≥ 2 px': obs >= MOVING_PX,
         'reviewed: trusted': rec['trusted'],
         'wall contact': rec['wall'],
         'action never seen in training': rec['unseen']}
    for lo, hi, name in SIZE:
        S[name] = (area >= lo) & (area < hi)
    return S, obs


def score(fold, rec, mp, ref='Persistence'):
    S, obs = subsets(rec)
    rows = []
    err = {m: np.hypot(d['mx'] - rec['ox'], d['my'] - rec['oy']) for m, d in mp.items()}
    for sname, sm in S.items():
        for h in HORIZONS:
            sel = sm & (rec['h'] == h)
            if sel.sum() == 0:
                continue
            for m, d in mp.items():
                e = err[m][sel]
                if not np.isfinite(e).all():
                    e = e[np.isfinite(e)]
                if not len(e):
                    continue
                r = dict(fold=fold, subset=sname, horizon_s=h / 10, model=m, n=int(sel.sum()),
                         mean_error_px=float(np.mean(e)), median_error_px=float(np.median(e)),
                         p90_error_px=float(np.percentile(e, 90)))
                r['skill_vs_persistence'] = 1 - r['mean_error_px'] / max(float(np.mean(err[ref][sel])), 1e-9)
                mv = sel & (obs >= MOVING_PX)
                if mv.sum():
                    pdx, pdy = d['mx'][mv] - rec['x0'][mv], d['my'][mv] - rec['y0'][mv]
                    odx, ody = rec['ox'][mv] - rec['x0'][mv], rec['oy'][mv] - rec['y0'][mv]
                    cosang = (pdx * odx + pdy * ody) / (np.hypot(pdx, pdy) * np.hypot(odx, ody) + 1e-9)
                    r['direction_within_45deg'] = float(np.mean(cosang > np.cos(np.radians(45))))
                # shape: log-area for groups and clusters, aspect ratio for clusters off the wall
                big = sel & (np.exp(rec['la0']) >= 100) & np.isfinite(d['la'])
                if big.sum():
                    r['logarea_abs_error'] = float(np.mean(np.abs(d['la'][big] - rec['ola'][big])))
                cl = sel & (np.exp(rec['la0']) >= 200) & ~rec['wall'] & np.isfinite(d['ar'])
                if cl.sum():
                    r['aspect_abs_error'] = float(np.mean(np.abs(d['ar'][cl] - rec['oar'][cl])))
                rows.append(r)
    return rows


def uncertainty(fold, rec, mp):
    rows, bins = [], []
    from scipy.stats import spearmanr
    S, obs = subsets(rec)
    for m, d in mp.items():
        if not np.isfinite(d['var']).any():
            continue
        e = np.hypot(d['mx'] - rec['ox'], d['my'] - rec['oy'])
        z2 = ((d['mx'] - rec['ox']) ** 2 + (d['my'] - rec['oy']) ** 2) / np.maximum(d['var'], 1e-9)
        sd = np.sqrt(d['var'])
        for sname in ('all regions', 'PZT switch within horizon', 'moved ≥ 2 px'):
            for h in HORIZONS:
                sel = S[sname] & (rec['h'] == h) & np.isfinite(z2)
                if sel.sum() < 10:
                    continue
                rho = spearmanr(sd[sel], e[sel]).correlation
                rows.append(dict(fold=fold, model=m, subset=sname, horizon_s=h / 10, n=int(sel.sum()),
                                 inside_50pct_range=float(np.mean(z2[sel] <= CHI2_2[0.5])),
                                 inside_90pct_range=float(np.mean(z2[sel] <= CHI2_2[0.9])),
                                 spearman_spread_vs_error=float(rho)))
                if sname == 'all regions':
                    q = np.quantile(sd[sel], [0, .2, .4, .6, .8, 1])
                    for k in range(5):
                        b = sel & (sd >= q[k]) & (sd <= q[k + 1])
                        bins.append(dict(fold=fold, model=m, horizon_s=h / 10, quintile=k + 1,
                                         predicted_spread_px=float(np.median(sd[b])),
                                         mean_error_px=float(np.mean(e[b])), n=int(b.sum())))
    return rows, bins


def field_scores(fold, exp, graphs, preds):
    """Whole field (regions present at the anchor and still tracked): area-weighted centre and spread."""
    rows = []
    for m, ps in preds.items():
        for h in HORIZONS:
            ce, se, cp, sp = [], [], [], []
            for g, o in zip(graphs, ps):
                ok = M.loss_mask(g) & np.isfinite(g['fut'][:, h - 1, 0]) & ~g['fut_edge'][:, h - 1]
                if ok.sum() < 3:
                    continue
                w = np.exp(g['shape'][ok, 0])

                def cs(p):
                    c = np.average(p, axis=0, weights=w)
                    return c, np.sqrt(np.average(((p - c) ** 2).sum(1), weights=w))
                c_obs, s_obs = cs(g['fut'][ok, h - 1])
                c_p, s_p = cs(o['mu'][ok, h - 1])
                c_0, s_0 = cs(g['pos'][ok])
                ce.append(np.hypot(*(c_p - c_obs)))
                se.append(abs(s_p - s_obs))
                cp.append(np.hypot(*(c_0 - c_obs)))
                sp.append(abs(s_0 - s_obs))
            if ce:
                rows.append(dict(fold=fold, model=m, horizon_s=h / 10, anchors=len(ce),
                                 centre_error_px=float(np.mean(ce)), spread_error_px=float(np.mean(se)),
                                 centre_error_persistence_px=float(np.mean(cp)),
                                 spread_error_persistence_px=float(np.mean(sp))))
    return rows


def action_check(fold, exp, graphs, models, seen_patterns):
    """Same samples, real PZT sequence vs. 'no switch': does knowing the action make the prediction better?"""
    idx = [i for i, g in enumerate(graphs) if M.switch_within(g, len(g['act']) - 1)]
    if not idx:
        return []
    gs = [graphs[i] for i in idx]
    hold = [M.hold_actions(g) for g in gs]
    rows = []
    for m in models:
        if m.name in ('Persistence', 'Constant velocity'):
            continue
        real = m.predict(exp, gs)
        cf = m.predict(exp, gs, act_override=hold)
        for h in HORIZONS:
            er, ec, eff, better = [], [], [], []
            for g, a, b in zip(gs, real, cf):
                if not M.switch_within(g, h):
                    continue
                ok = M.loss_mask(g) & np.isfinite(g['fut'][:, h - 1, 0]) & ~g['fut_edge'][:, h - 1]
                if not ok.any():
                    continue
                o = g['fut'][ok, h - 1]
                e1 = np.hypot(*(a['mu'][ok, h - 1] - o).T)
                e2 = np.hypot(*(b['mu'][ok, h - 1] - o).T)
                er += list(e1)
                ec += list(e2)
                eff += list(np.hypot(*(a['mu'][ok, h - 1] - b['mu'][ok, h - 1]).T))
                better += list(e1 < e2)
            if er:
                rows.append(dict(fold=fold, model=m.name, horizon_s=h / 10, n=len(er),
                                 error_real_action_px=float(np.mean(er)), error_no_switch_px=float(np.mean(ec)),
                                 prediction_change_px=float(np.mean(eff)),
                                 real_action_closer_share=float(np.mean(better))))
    return rows


def merge_action_rows(rows):
    """Combine the action check of several test recordings of one fold (weighted by sample count)."""
    out = {}
    for r in rows:
        k = (r['fold'], r['model'], r['horizon_s'])
        if k not in out:
            out[k] = dict(r)
            continue
        o = out[k]
        n1, n2 = o['n'], r['n']
        for f in ('error_real_action_px', 'error_no_switch_px', 'prediction_change_px', 'real_action_closer_share'):
            o[f] = (o[f] * n1 + r[f] * n2) / (n1 + n2)
        o['n'] = n1 + n2
    return list(out.values())


def block_split(exp, graphs, parity, block_s=4.0, before_s=0.3, after_s=1.0):
    """Anchors of the even (parity 0) or odd (1) blocks of block_s seconds whose history (before_s) and horizon
    (after_s) stay inside the same block, so training and test windows never overlap."""
    out = []
    for g in graphs:
        t = g['frame'] / exp.fps
        b = int(t // block_s)
        if b % 2 == parity and int((t - before_s) // block_s) == b and int((t + after_s) // block_s) == b:
            out.append(g)
    return out


# =====================================================================================================
def make_models(a, label_suffix=''):
    common = dict(members=a.members, epochs=a.epochs, patience=a.patience, threads=a.threads)
    return [M.Persistence(), M.ConstantVelocity(), M.ResponseMap(), M.GradientBoosting(),
            M.GraphWorldModel(interactions=False, **common),
            M.GraphWorldModel(interactions=True, **common)]


def main():
    ap = argparse.ArgumentParser(description='Stage 2: world-action models on the recorded experiments')
    ap.add_argument('--exp', action='append', required=True, help='NAME=result_dir (at least two)')
    ap.add_argument('--video', action='append', default=[], help='NAME=video (figures only)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--members', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--no-review-variant', action='store_true',
                    help='skip the extra graph model trained only on manually trusted regions')
    ap.add_argument('--figures-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_all = time.time()
    exps = {}
    for s in a.exp:
        name, d = s.split('=', 1)
        exps[name] = ds.Experiment(name, d)
        print(f'{name}: {exps[name].summary()}', flush=True)
    videos = dict(s.split('=', 1) for s in a.video)
    graphs = {n: e.graphs() for n, e in exps.items()}

    if a.figures_only:
        import wm_figures
        print('figures:', wm_figures.make_all(a.out, exps, graphs, videos))
        return

    metrics, unc_rows, unc_bins, field_rows, act_rows, folds, pred_rows = [], [], [], [], [], [], []
    examples = {}
    plan = []
    # (1) across recordings: train on the other recording(s), test on one; the fold whose training set has
    #     the most PZT patterns comes first (every test pattern is then seen in training)
    order = sorted(exps, key=lambda t: -len({p for n, e in exps.items() if n != t for p in e.patterns}))
    for k, test in enumerate(order):
        train = [n for n in exps if n != test]
        plan.append(dict(fold=f'train {"+".join(train)} → test {test}', kind='across recordings',
                         train_exps=[exps[n] for n in train], train_graphs=[g for n in train for g in graphs[n]],
                         tests=[(exps[test], graphs[test])], variants=(k == 0), example=test))
    # (2) within recordings (optimistic): alternating 4 s blocks of every recording, both ways
    for parity in (0, 1):
        tr = [g for n in exps for g in block_split(exps[n], graphs[n], parity)]
        plan.append(dict(fold=f'same recordings, alternating 4 s blocks ({"even→odd" if parity == 0 else "odd→even"})',
                         kind='within recordings', train_exps=list(exps.values()), train_graphs=tr,
                         tests=[(exps[n], block_split(exps[n], graphs[n], 1 - parity)) for n in exps],
                         variants=False, example=None))

    for fd in plan:
        fold = fd['fold']
        print(f'\n== {fold}', flush=True)
        tr_exps, tr_graphs = fd['train_exps'], fd['train_graphs']
        seen = {M.pattern_of(x) for g in tr_graphs for x in g['act']}
        if fd['kind'] == 'across recordings':
            assert all(g['exp'] not in {e.name for e, _ in fd['tests']} for g in tr_graphs)
        else:
            tr_keys = {(g['exp'], g['frame']) for g in tr_graphs}
            assert not any((g['exp'], g['frame']) in tr_keys for _, gs in fd['tests'] for g in gs)
        models = make_models(a)
        preds = {}
        timing = {}
        for m in models:
            t0 = time.time()
            m.fit(tr_exps, tr_graphs)
            t1 = time.time()
            preds[m.name] = [m.predict(e, gs) for e, gs in fd['tests']]
            n_anchor = sum(len(gs) for _, gs in fd['tests'])
            timing[m.name] = dict(fit_s=t1 - t0, predict_ms_per_anchor=(time.time() - t1) / n_anchor * 1000)
            print(f'  {m.name}: fit {t1 - t0:.0f} s, predict {timing[m.name]["predict_ms_per_anchor"]:.1f} ms per anchor',
                  flush=True)
        if fd['variants'] and not a.no_review_variant:
            # same graph model, but the loss only uses regions a person marked trusted
            rv = M.GraphWorldModel(interactions=True, members=a.members, epochs=a.epochs, patience=a.patience,
                                   threads=a.threads, label='Graph world model, trained on reviewed data')
            rg = [dict(g, flags=np.c_[g['flags'][:, :1], g['flags'][:, 1:2] * (g['status'] == 'trusted')[:, None],
                                      g['flags'][:, 2:]]) for g in tr_graphs]
            t0 = time.time()
            rv.fit(tr_exps, rg)
            preds[rv.name] = [rv.predict(e, gs) for e, gs in fd['tests']]
            timing[rv.name] = dict(fit_s=time.time() - t0)
            print(f'  {rv.name}: fit {time.time() - t0:.0f} s', flush=True)
            models.append(rv)
        recs, mps, fold_act = [], [], []
        for ti, (te, te_graphs) in enumerate(fd['tests']):
            p_t = {m: ps[ti] for m, ps in preds.items()}
            rec, mp = flatten(te, te_graphs, p_t, seen)
            rec['exp'] = np.array([te.name] * len(rec['frame']))
            recs.append(rec)
            mps.append(mp)
            field_rows += field_scores(fold if len(fd['tests']) == 1 else f'{fold}, test {te.name}', te, te_graphs, p_t)
            fold_act += action_check(fold, te, te_graphs, models, seen)
        act_rows += merge_action_rows(fold_act)
        rec = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
        mp = {m: {k: np.concatenate([x[m][k] for x in mps]) for k in mps[0][m]} for m in mps[0]}
        metrics += [dict(r, kind=fd['kind']) for r in score(fold, rec, mp)]
        u, b = uncertainty(fold, rec, mp)
        unc_rows += u
        unc_bins += b
        folds.append(dict(fold=fold, kind=fd['kind'], train_anchors=len(tr_graphs),
                          test_anchors=sum(len(gs) for _, gs in fd['tests']),
                          tests=[e.name for e, _ in fd['tests']],
                          patterns_seen_in_training=sorted(ds.pattern_name(p) for p in seen),
                          patterns_in_test=sorted({ds.pattern_name(M.pattern_of(x)) for _, gs in fd['tests']
                                                   for g in gs for x in g['act']}),
                          timing=timing))
        # per-sample predictions (every 0.1 s anchor) for later inspection
        keep = rec['frame'] % 6 == 0
        for i in np.nonzero(keep)[0]:
            r = dict(fold=fold, experiment=str(rec['exp'][i]), frame=int(rec['frame'][i]), track=int(rec['track'][i]),
                     horizon_s=rec['h'][i] / 10, x_now_px=rec['x0'][i], y_now_px=rec['y0'][i],
                     x_observed_px=rec['ox'][i], y_observed_px=rec['oy'][i], switch_in_horizon=int(rec['switch'][i]))
            for m, d in mp.items():
                r[f'{m}: x'] = d['mx'][i]
                r[f'{m}: y'] = d['my'][i]
                if np.isfinite(d['var'][i]):
                    r[f'{m}: spread_px'] = float(np.sqrt(d['var'][i]))
            pred_rows.append(r)
        if fd['example']:
            examples[fd['example']] = {m.name: m for m in models
                                       if m.name in ('Graph world model', 'Response map')}

    import wm_figures
    wm_figures.save_fold_models(a.out, examples)
    write_csv(os.path.join(a.out, 'metrics.csv'), metrics)
    write_csv(os.path.join(a.out, 'uncertainty.csv'), unc_rows)
    write_csv(os.path.join(a.out, 'uncertainty_bins.csv'), unc_bins)
    write_csv(os.path.join(a.out, 'field.csv'), field_rows)
    write_csv(os.path.join(a.out, 'action_check.csv'), act_rows)
    write_csv(os.path.join(a.out, 'predictions.csv'), pred_rows)

    # ------------------------------------------------------------------ final model on every experiment
    import torch
    print('\n== final graph world model on all experiments', flush=True)
    final = M.GraphWorldModel(interactions=True, members=a.members, epochs=a.epochs, patience=a.patience,
                              threads=a.threads)
    all_exps = list(exps.values())
    final.fit(all_exps, [g for n in exps for g in graphs[n]])
    md = os.path.join(a.out, 'model')
    os.makedirs(md, exist_ok=True)
    torch.save(final.state(), os.path.join(md, 'graph_world_model.pt'))
    reloaded = M.GraphWorldModel.from_state(torch.load(os.path.join(md, 'graph_world_model.pt')))
    e0 = all_exps[0]
    gtest = graphs[e0.name][::50]
    p1 = final.predict(e0, gtest)
    p2 = reloaded.predict(e0, gtest)
    reload_ok = all(np.allclose(x['mu'], y['mu']) for x, y in zip(p1, p2))
    # planning speed: one anchor, every action pattern seen, 1 s roll-out
    g = graphs[e0.name][len(graphs[e0.name]) // 2]
    pats = sorted({p for e in all_exps for p in e.patterns})
    cand = [np.repeat(np.array(p, np.float32)[None], g['act'].shape[0], 0) for p in pats]
    for c in cand:
        c[0] = g['act'][0]
    t0 = time.time()
    for _ in range(5):
        final.predict(e0, [g] * len(cand), act_override=cand)
    plan_ms = (time.time() - t0) / 5 * 1000
    json.dump(json_safe(dict(created=now(), experiments=list(exps), patterns=[ds.pattern_name(p) for p in pats],
                             regions_in_example=len(g['pos']), dataset=ds.DEFAULTS,
                             note='load with models.GraphWorldModel.from_state(torch.load(path))')),
              open(os.path.join(md, 'model.json'), 'w'), indent=1)
    write_csv(os.path.join(a.out, 'training_history.csv'),
              [dict(model='final', **h) for h in final.history])

    # ------------------------------------------------------------------ figures and report
    figs = wm_figures.make_all(a.out, exps, graphs, videos, fold_models=examples, final=final)
    checks = dict(test_never_in_training=True,
                  same_samples_for_every_model=all(len({r['n'] for r in metrics if r['fold'] == f['fold']
                                                        and r['subset'] == 'all regions' and r['horizon_s'] == h / 10}) == 1
                                                   for f in folds for h in HORIZONS),
                  within_recording_train_and_test_windows_disjoint=True,
                  saved_model_reloads_identically=bool(reload_ok))
    report = dict(created=now(), experiments={n: e.summary() for n, e in exps.items()}, parameters=dict(
        dataset=ds.DEFAULTS, horizons_s=[h / 10 for h in HORIZONS], moving_px=MOVING_PX,
        graph_model=dict(members=a.members, epochs=a.epochs, patience=a.patience, radius_px=150, latent=64,
                         message_passing_steps=3)),
        folds=folds, checks=checks, all_checks_passed=all(checks.values()),
        planning_speed=dict(candidates=len(cand), regions=len(g['pos']), horizon_s=1.0,
                            ms_for_all_candidates=plan_ms, ensemble_members=a.members),
        headline=[r for r in metrics if r['subset'] in ('all regions', 'PZT switch within horizon', 'moved ≥ 2 px')],
        figures=figs, runtime_s=time.time() - t_all)
    json.dump(json_safe(report), open(os.path.join(a.out, 'report.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)
    print(f'\nwritten to {a.out}  ({time.time() - t_all:.0f} s)')


if __name__ == '__main__':
    main()
