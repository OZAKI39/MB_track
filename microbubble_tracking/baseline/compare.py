#!/usr/bin/env python3
"""
compare.py - compare the Medany et al. detection / tracking (baseline) with our method on one experiment.

Reference: our tracking result after the round-1 manual review (review.json). Regions a person marked
"Not a bubble" are false targets; every other region is a real bubble region. For the single-target part
the reference is the human-checked (trusted) trajectory of each main object. Because the reference comes
from our own (reviewed) output, our method cannot be scored against it; it is described by what the
review found instead (share of trusted frames, false targets, IDs that had to be linked by hand) and, like
the baseline, by an independent check on the raw video (template matching, no segmentation involved).

Parts
    1. detection: in every 5th frame, which reference regions the baseline's "dark pixels inside the
       channel mask" find, and what else it finds (walls, debris, other), for thresholds 80-150 and for
       two channel masks (MobileSAM as in the original procedure, and an ideal mask)
    2. single-target tracking: the baseline tracker started on each main object, as in the original
       (CSRT on the blue image, automatic re-initialisation on the largest blue region)
    3. speed of both, and an independent displacement check on the raw video

Usage (baseline environment with opencv-contrib 4.x, see README):
    python baseline/compare.py --result results/A_offline --video data/raw/expA.mp4 \
        --masks results/baseline_comparison/masks --out results/baseline_comparison/A
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'analysis')]
import medany_baseline as MB  # noqa: E402
import channel_mask as CM  # noqa: E402
import analyze  # noqa: E402

DEFAULTS = dict(
    thresholds=list(range(80, 155, 5)),   # 100 is the value in the original config
    sweep_every=5,            # detection is scored on every 5th frame
    detect_min_px=3,          # a reference region counts as found if at least 3 of its pixels are "blue"
    blob_min_px=5,            # blue regions smaller than this are ignored as noise
    match_margin_px=2,        # a blue region belongs to a reference region if it touches it (2 px tolerance)
    wall_dilate_px=3,         # blue region touching a wall / fixed structure within 3 px = "wall"
    on_target_margin_px=5,    # tracker position within 5 px of the target outline = on target
    after_end_s=2.0,          # keep running 2 s after the target's last reference frame
    video_check_n=60,         # frames per target for the independent displacement check
    small_px=100, cluster_px=300,
)


def size_class(a, P):
    return 'single (<100 px²)' if a < P['small_px'] else ('small group (100-300 px²)' if a < P['cluster_px'] else 'cluster (≥300 px²)')


class Reference:
    """Reference regions of one frame, in ROI coordinates."""

    def __init__(self, D, f, roi, P):
        x0, y0, w, h = roi
        self.rows = list(D.rows_by_frame.get(f, []))
        self.lab = np.zeros((h, w), np.int32)
        self.cont = {}
        for k in self.rows:
            c = D.contour.get((f, int(D.I[k])))
            if c is None:
                continue
            c = (c.reshape(-1, 2) - [x0, y0]).astype(np.int32)
            self.cont[k] = c
            cv2.fillPoly(self.lab, [c.reshape(-1, 1, 2)], int(k) + 1)
        d = 2 * P['match_margin_px'] + 1
        real = np.isin(self.lab - 1, [k for k in self.rows if D.role[k] != 'excluded']) & (self.lab > 0)
        excl = np.isin(self.lab - 1, [k for k in self.rows if D.role[k] == 'excluded']) & (self.lab > 0)
        self.real_d = cv2.dilate(real.astype(np.uint8), np.ones((d, d), np.uint8)) > 0
        self.excl_d = cv2.dilate(excl.astype(np.uint8), np.ones((d, d), np.uint8)) > 0

    def where(self, x, y, margin, D):
        """Which reference row the point (ROI coordinates) lies on, or None."""
        best, bd = None, -1e9
        for k, c in self.cont.items():
            d = cv2.pointPolygonTest(c.reshape(-1, 1, 2).astype(np.float32), (float(x), float(y)), True)
            if d >= -margin and d > bd:
                best, bd = k, d
        return best


# ---------------------------------------------------------------------------
# 1. detection
# ---------------------------------------------------------------------------
def detection_sweep(D, video, roi, masks, walls_roi, P, log):
    acc = defaultdict(lambda: defaultdict(float))
    cap = cv2.VideoCapture(video)
    f = 0
    x0, y0, w, h = roi
    n_eval = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if f % P['sweep_every']:
            f += 1
            continue
        crop = fr[y0:y0 + h, x0:x0 + w]
        ref = Reference(D, f, roi, P)
        real_rows = [k for k in ref.rows if D.role[k] != 'excluded' and k in ref.cont]
        for mname, mask in masks.items():
            for thr in P['thresholds']:
                bm = MB.blue_mask(mask, crop, thr)
                a = acc[(mname, thr)]
                cnt = np.bincount(ref.lab[bm], minlength=len(D.F) + 1) if bm.any() else np.zeros(len(D.F) + 1, int)
                for k in real_rows:
                    sc = size_class(D.area[k], P)
                    hit = cnt[k + 1] >= P['detect_min_px']
                    a[f'n|{sc}'] += 1
                    a[f'hit|{sc}'] += hit
                    if D.role[k] == 'confirmed':
                        a['n|trusted'] += 1
                        a['hit|trusted'] += hit
                    a['area_cover_sum|' + sc] += min(1.0, cnt[k + 1] / max(1.0, D.area[k]))
                nl, cl, st, _ = cv2.connectedComponentsWithStats(bm.astype(np.uint8), connectivity=8)
                big = st[1:, cv2.CC_STAT_AREA] >= P['blob_min_px']
                ids = np.nonzero(big)[0] + 1
                if len(ids):
                    on_real = np.bincount(cl[ref.real_d], minlength=nl)[ids] > 0
                    on_excl = np.bincount(cl[ref.excl_d], minlength=nl)[ids] > 0
                    on_wall = np.bincount(cl[walls_roi], minlength=nl)[ids] > 0
                    area = st[ids, cv2.CC_STAT_AREA]
                    cat = np.where(on_real, 'matched', np.where(on_excl, 'false target', np.where(on_wall, 'wall', 'other')))
                    for c_ in ('matched', 'false target', 'wall', 'other'):
                        a[f'blobs|{c_}'] += int((cat == c_).sum())
                        a[f'blob_px|{c_}'] += int(area[cat == c_].sum())
                a['frames'] += 1
        n_eval += 1
        if n_eval % 50 == 0:
            log(f'  detection: {n_eval} frames scored')
        f += 1
    cap.release()
    rows = []
    for (mname, thr), a in sorted(acc.items()):
        fr_ = max(1, a['frames'])
        r = dict(mask=mname, threshold=thr, frames=int(fr_))
        for sc in ('single (<100 px²)', 'small group (100-300 px²)', 'cluster (≥300 px²)'):
            short = sc.split(' (')[0].replace(' ', '_')
            r[f'recall_{short}'] = a[f'hit|{sc}'] / a[f'n|{sc}'] if a[f'n|{sc}'] else np.nan
            r[f'area_covered_{short}'] = a[f'area_cover_sum|{sc}'] / a[f'n|{sc}'] if a[f'n|{sc}'] else np.nan
        n_all = sum(a[f'n|{sc}'] for sc in ('single (<100 px²)', 'small group (100-300 px²)', 'cluster (≥300 px²)'))
        h_all = sum(a[f'hit|{sc}'] for sc in ('single (<100 px²)', 'small group (100-300 px²)', 'cluster (≥300 px²)'))
        r['recall_all'] = h_all / n_all if n_all else np.nan
        r['recall_trusted'] = a['hit|trusted'] / a['n|trusted'] if a['n|trusted'] else np.nan
        for c_ in ('matched', 'false target', 'wall', 'other'):
            k = c_.replace(' ', '_')
            r[f'blobs_per_frame_{k}'] = a[f'blobs|{c_}'] / fr_
            r[f'blob_px_per_frame_{k}'] = a[f'blob_px|{c_}'] / fr_
        tot = sum(a[f'blobs|{c_}'] for c_ in ('matched', 'false target', 'wall', 'other'))
        r['precision_blobs'] = a['blobs|matched'] / tot if tot else np.nan
        p, rc = r['precision_blobs'], r['recall_all']
        r['f1'] = 2 * p * rc / (p + rc) if p + rc > 0 else 0.0
        rows.append(r)
    return rows


def detection_examples(D, video, roi, mask, walls_roi, thresholds, frames, P):
    """Blue pixels and their category on a few frames (for the figure)."""
    x0, y0, w, h = roi
    out = {}
    imgs = analyze_frames(video, frames)
    for f in frames:
        crop = imgs[f][y0:y0 + h, x0:x0 + w]
        ref = Reference(D, f, roi, P)
        for thr in thresholds:
            bm = MB.blue_mask(mask, crop, thr)
            nl, cl, st, _ = cv2.connectedComponentsWithStats(bm.astype(np.uint8), connectivity=8)
            ids = np.nonzero(st[1:, cv2.CC_STAT_AREA] >= P['blob_min_px'])[0] + 1
            cat = {}
            if len(ids):
                on_real = np.bincount(cl[ref.real_d], minlength=nl)[ids] > 0
                on_excl = np.bincount(cl[ref.excl_d], minlength=nl)[ids] > 0
                on_wall = np.bincount(cl[walls_roi], minlength=nl)[ids] > 0
                c = np.where(on_real, 'matched', np.where(on_excl, 'false target', np.where(on_wall, 'wall', 'other')))
                cat = dict(zip(ids.tolist(), c.tolist()))
            out[(thr, f)] = (crop.copy(), bm, cl, cat)
    return out


def analyze_frames(video, frames):
    cap = cv2.VideoCapture(video)
    out = {}
    for f in sorted(set(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if ok:
            out[f] = im
    cap.release()
    return out


# ---------------------------------------------------------------------------
# 2. single-target tracking
# ---------------------------------------------------------------------------
def object_rows(D, oid):
    ks = np.nonzero(D.object_id == oid)[0]
    return ks[np.argsort(D.F[ks])]


def track_targets(D, video, roi, masks, configs, targets, P, log):
    x0, y0, w, h = roi
    fps = D.fps
    plan = []
    for oid in targets:
        ks = object_rows(D, oid)
        tr = ks[D.status[ks] == 'trusted']
        if not len(tr):
            continue
        f0, f1 = int(D.F[tr[0]]), int(D.F[tr[-1]])
        for cfg in configs:
            plan.append(dict(oid=oid, cfg=cfg, f0=f0, f1=f1, f_stop=min(D.n_frames - 1, f1 + int(P['after_end_s'] * fps)),
                             first_row=int(tr[0]), tracker=None, out=[], secs=0.0))
    cap = cv2.VideoCapture(video)
    f = 0
    last = max(p['f_stop'] for p in plan) if plan else -1
    while f <= last:
        ok, fr = cap.read()
        if not ok:
            break
        crop = np.ascontiguousarray(fr[y0:y0 + h, x0:x0 + w])
        active = [p for p in plan if p['f0'] <= f <= p['f_stop']]
        ref = Reference(D, f, roi, P) if active else None
        for p in active:
            mname, thr = p['cfg']
            t0 = time.perf_counter()
            if f == p['f0']:
                c = ref.cont.get(p['first_row'])
                bx, by, bw, bh = cv2.boundingRect(c.reshape(-1, 1, 2))
                p['tracker'] = MB.MedanyTracker(masks[mname], threshold=thr)
                res = p['tracker'].start(crop, (bx, by, bw, bh))
            else:
                res = p['tracker'].step(crop)
            p['secs'] += time.perf_counter() - t0
            # what is the tracker on?
            tk = [k for k in ref.rows if D.object_id[k] == p['oid'] and D.status[k] == 'trusted' and k in ref.cont]
            target = tk[0] if tk else None
            if not res['ok']:
                state = 'lost'
                where = None
            else:
                where = ref.where(res['x'], res['y'], P['on_target_margin_px'], D)
                if target is not None and where is not None and D.object_id[where] == p['oid']:
                    state = 'on target'
                elif where is not None and D.role[where] == 'excluded':
                    state = 'false target'
                elif where is not None:
                    state = 'other object'
                else:
                    state = 'not on a region'
            err = None
            if target is not None and res['ok']:
                err = float(np.hypot(res['x'] + x0 - D.cx[target], res['y'] + y0 - D.cy[target]))
            p['out'].append(dict(
                object_id=p['oid'], mask=mname, threshold=thr, frame=f, t_s=f / fps,
                phase='target present' if target is not None else ('after target' if f > p['f1'] else 'target not trusted here'),
                ok=res['ok'], x_px=res['x'] + x0 if res['ok'] else None, y_px=res['y'] + y0 if res['ok'] else None,
                blue_area_px2=res['area'], reinit=res['reinit'], state=state,
                on_object=int(D.object_id[where]) if where is not None else None,
                ref_x_px=float(D.cx[target]) if target is not None else None,
                ref_y_px=float(D.cy[target]) if target is not None else None,
                ref_area_px2=float(D.area[target]) if target is not None else None, error_px=err))
        f += 1
        if f % 200 == 0:
            log(f'  tracking: frame {f}/{last + 1}')
    cap.release()
    return plan


def summarize_tracking(D, plan, P):
    rows = []
    for p in plan:
        o = p['out']
        pres = [r for r in o if r['phase'] == 'target present']
        aft = [r for r in o if r['phase'] == 'after target']
        n = len(pres)
        cnt = defaultdict(int)
        for r in pres:
            cnt[r['state']] += 1
        on = [r for r in pres if r['state'] == 'on target']
        errs = np.array([r['error_px'] for r in on])
        switches = sum(1 for a, b in zip(pres, pres[1:]) if a['state'] == 'on target' and b['state'] in ('other object', 'false target'))
        ar = np.array([(r['blue_area_px2'], r['ref_area_px2']) for r in on if r['ref_area_px2']])
        rows.append(dict(
            object_id=p['oid'], method='baseline', mask=p['cfg'][0], threshold=p['cfg'][1],
            object_end=end_reason(D, p['oid']),
            frames_with_reference=n, on_target=cnt['on target'] / n if n else np.nan,
            on_other_object=cnt['other object'] / n if n else np.nan,
            on_false_target=cnt['false target'] / n if n else np.nan,
            on_background_or_wall=cnt['not on a region'] / n if n else np.nan,
            lost=cnt['lost'] / n if n else np.nan,
            switches_to_other_object=switches, reinitialisations=sum(r['reinit'] for r in o),
            first_frame_off_target=next((r['frame'] for r in pres if r['state'] != 'on target'), None),
            center_error_median_px=float(np.median(errs)) if len(errs) else np.nan,
            center_error_p90_px=float(np.percentile(errs, 90)) if len(errs) else np.nan,
            area_ratio_median=float(np.median(ar[:, 0] / ar[:, 1])) if len(ar) else np.nan,
            area_correlation=float(np.corrcoef(ar[:, 0], ar[:, 1])[0, 1]) if len(ar) > 5 and ar[:, 0].std() > 0 else np.nan,
            frames_after_target=len(aft), reported_after_target=sum(1 for r in aft if r['ok']),
            ms_per_frame=1000 * p['secs'] / max(1, len(o))))
    return rows


def end_reason(D, oid):
    """How the object's last trusted segment ends (round-2 segments.csv)."""
    p = os.path.join(D.dir, 'analysis', 'segments.csv')
    if not os.path.exists(p):
        return ''
    with open(p, newline='', encoding='utf-8') as fh:
        segs = [r for r in csv.DictReader(fh) if int(r['object_id']) == oid]
    return max(segs, key=lambda r: int(r['end_frame']))['end_reason'] if segs else ''


def ours_on_targets(D, targets):
    """What the review found for our method on the same objects."""
    rows = []
    for oid in targets:
        ks = object_rows(D, oid)
        if not len(ks):
            continue
        f0, f1 = int(D.F[ks[0]]), int(D.F[ks[-1]])
        life = f1 - f0 + 1
        st = D.status[ks]
        rows.append(dict(object_id=oid, method='ours (after review)', frames_in_lifetime=life,
                         trusted_share=float((st == 'trusted').sum() / life),
                         untrusted_share=float((st == 'untrusted').sum() / life),
                         not_detected_share=float((life - len(ks)) / life),
                         track_ids=len(set(D.I[ks].tolist())),
                         reported_after_target=0))
    return rows


# ---------------------------------------------------------------------------
# 3. independent displacement check on the raw video
# ---------------------------------------------------------------------------
def video_check(D, video, plan, P, seed=0):
    lag = max(1, int(round(0.1 * D.fps)))
    rng = np.random.default_rng(seed)
    cap = cv2.VideoCapture(video)
    cache = {}

    def gray(f):
        if f not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            cache[f] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if ok else None
        return cache[f]

    out = []
    for p in plan:
        o = {r['frame']: r for r in p['out']}
        cand = [f for f, r in o.items() if r['state'] == 'on target' and (f + lag) in o
                and o[f + lag]['state'] == 'on target']
        if not cand:
            continue
        pick = rng.choice(cand, min(len(cand), P['video_check_n']), replace=False)
        d_base, d_ours = [], []
        for f in sorted(int(v) for v in pick):
            r0, r1 = o[f], o[f + lag]
            A, B = gray(f), gray(f + lag)
            if A is None or B is None:
                continue
            X, Y = int(round(r0['ref_x_px'])), int(round(r0['ref_y_px']))
            hh = int(max(12, 0.8 * np.sqrt(r0['ref_area_px2'])))
            H, W = A.shape
            if X - hh < 0 or Y - hh < 0 or X + hh >= W or Y + hh >= H:
                continue
            tpl = A[Y - hh:Y + hh + 1, X - hh:X + hh + 1]
            odx, ody = r1['ref_x_px'] - r0['ref_x_px'], r1['ref_y_px'] - r0['ref_y_px']
            bdx, bdy = r1['x_px'] - r0['x_px'], r1['y_px'] - r0['y_px']
            S = int(hh + max(np.hypot(odx, ody), np.hypot(bdx, bdy)) + 15)
            x0, y0, x1, y1 = max(0, X - S), max(0, Y - S), min(W, X + S + 1), min(H, Y + S + 1)
            res = cv2.matchTemplate(B[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
            _, _, _, (px, py) = cv2.minMaxLoc(res)
            tdx, tdy = x0 + px + hh - X, y0 + py + hh - Y
            d_base.append(np.hypot(bdx - tdx, bdy - tdy))
            d_ours.append(np.hypot(odx - tdx, ody - tdy))
        if d_base:
            out.append(dict(object_id=p['oid'], mask=p['cfg'][0], threshold=p['cfg'][1], samples=len(d_base),
                            baseline_median_px=float(np.median(d_base)), baseline_p90_px=float(np.percentile(d_base, 90)),
                            ours_median_px=float(np.median(d_ours)), ours_p90_px=float(np.percentile(d_ours, 90))))
    cap.release()
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    analyze.write_csv(path, keys, rows)


def main():
    ap = argparse.ArgumentParser(description='Compare the Medany et al. baseline with our method')
    ap.add_argument('--result', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--masks', required=True, help='folder with mask_mobilesam.png and mask_ideal.png')
    ap.add_argument('--out', required=True)
    ap.add_argument('--threshold', type=int, help='tuned threshold (default: best F1 in the sweep, MobileSAM mask)')
    ap.add_argument('--targets', type=int, nargs='*', help='object IDs (default: main objects of round 2)')
    ap.add_argument('--figures-only', action='store_true', help='redraw the figures from the saved tables')
    a = ap.parse_args()
    P = dict(DEFAULTS)
    log = print
    os.makedirs(a.out, exist_ok=True)
    rep2 = json.load(open(os.path.join(a.result, 'analysis', 'report.json'), encoding='utf-8'))
    D = analyze.Data(a.result, dict(analyze.DEFAULTS, **rep2.get('parameters', {})), video=a.video)
    roi = CM.ROI
    masks = {n: (cv2.imread(os.path.join(a.masks, f'mask_{n}.png'), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
             for n in ('mobilesam', 'ideal')}
    walls = analyze.static_structure(D.bg, 0.06, P['wall_dilate_px'])
    x0, y0, w, h = roi
    walls_roi = walls[y0:y0 + h, x0:x0 + w]

    t0 = time.time()
    if a.figures_only:
        redraw(D, a, roi, masks, walls_roi, P)
        return
    log('1. detection sweep ...')
    det = detection_sweep(D, a.video, roi, masks, walls_roi, P, log)
    write_csv(os.path.join(a.out, 'detection_sweep.csv'), det)
    ms = [r for r in det if r['mask'] == 'mobilesam']
    tuned = a.threshold or max(ms, key=lambda r: r['f1'])['threshold']
    log(f'   tuned threshold (best F1, MobileSAM mask): {tuned}   ({time.time() - t0:.0f} s)')
    examples = detection_examples(D, a.video, roi, masks['mobilesam'], walls_roi, sorted({100, tuned}), [200, 800], P)

    targets = a.targets or [o['object_id'] for o in rep2.get('main_objects', [])]
    configs = [('mobilesam', 100), ('mobilesam', tuned), ('ideal', tuned)]
    configs = list(dict.fromkeys(configs))
    log(f'2. single-target tracking: objects {targets}, configs {configs} ...')
    plan = track_targets(D, a.video, roi, masks, configs, targets, P, log)
    trk = summarize_tracking(D, plan, P)
    write_csv(os.path.join(a.out, 'tracking_summary.csv'), trk)
    write_csv(os.path.join(a.out, 'tracking_frames.csv'), [r for p in plan for r in p['out']])
    ours = ours_on_targets(D, targets)
    write_csv(os.path.join(a.out, 'ours_on_targets.csv'), ours)
    log(f'   done ({time.time() - t0:.0f} s)')
    log('3. independent check on the raw video ...')
    vc = video_check(D, a.video, plan, P)
    write_csv(os.path.join(a.out, 'video_check.csv'), vc)

    rows = D.n
    excl = int(np.sum(D.role == 'excluded'))
    summary = dict(
        experiment=os.path.basename(os.path.normpath(a.result)), roi=roi, tuned_threshold=tuned,
        configs=configs, targets=targets, parameters=P,
        ours=dict(region_frames=rows, false_target_share=excl / rows,
                  untrusted_share=float(np.mean(D.status == 'untrusted')),
                  linked_groups=len((D.review or {}).get('links', [])),
                  ms_per_frame_all_objects=(json.load(open(os.path.join(a.result, 'stats.json'))) or {}).get('ms_per_frame_mean')),
        detection={f"{r['mask']}@{r['threshold']}": r for r in det if (r['mask'], r['threshold']) in
                   {('mobilesam', 100), ('mobilesam', tuned), ('ideal', 100), ('ideal', tuned)}},
        tracking=trk, ours_on_targets=ours, video_check=vc)
    json.dump(analyze_safe(summary), open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    import compare_figures
    compare_figures.make_all(D, a.video, roi, masks, det, tuned, plan, trk, ours, vc, examples, a.out, P,
                             a.masks)
    log(f'written to {a.out}  ({time.time() - t0:.0f} s)')


def read_rows(path):
    def conv(v):
        if v == '':
            return None
        try:
            x = float(v)
            return int(x) if x.is_integer() and '.' not in v else x
        except ValueError:
            return v
    with open(path, newline='', encoding='utf-8') as fh:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def redraw(D, a, roi, masks, walls_roi, P):
    import compare_figures
    det = read_rows(os.path.join(a.out, 'detection_sweep.csv'))
    summ = json.load(open(os.path.join(a.out, 'summary.json'), encoding='utf-8'))
    tuned = summ['tuned_threshold']
    plan = {}
    for r in read_rows(os.path.join(a.out, 'tracking_frames.csv')):
        r['ok'] = bool(r['ok'])
        r['reinit'] = bool(r['reinit'])
        key = (r['object_id'], (r['mask'], r['threshold']))
        plan.setdefault(key, dict(oid=r['object_id'], cfg=(r['mask'], r['threshold']), out=[]))['out'].append(r)
    plan = list(plan.values())
    trk = read_rows(os.path.join(a.out, 'tracking_summary.csv'))
    ours = read_rows(os.path.join(a.out, 'ours_on_targets.csv'))
    vc = read_rows(os.path.join(a.out, 'video_check.csv'))
    examples = detection_examples(D, a.video, roi, masks['mobilesam'], walls_roi, sorted({100, tuned}), [200, 800], P)
    compare_figures.make_all(D, a.video, roi, masks, det, tuned, plan, trk, ours, vc, examples, a.out, P,
                             a.masks)
    print('figures redrawn in', os.path.join(a.out, 'figures'))


def analyze_safe(o):
    if isinstance(o, dict):
        return {str(k): analyze_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [analyze_safe(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


if __name__ == '__main__':
    main()
