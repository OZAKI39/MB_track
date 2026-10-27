#!/usr/bin/env python3
"""
analyze.py - round 2: motion, shape and whole-field aggregation of microbubble regions

Reads a result folder made by bubble_tracker.py and reviewed with review/review_tool.py.
Nothing in that folder is modified; everything is written to <result>/analysis/.

Inputs (read only)
    objects.csv      per-frame regions from the tracker
    events.csv       appear / lost / enter / exit / merge / split
    review.json      round-1 manual review (trusted / untrusted / unreviewed ranges, same-object links)
    stats.json       frame rate and crop
    contours.npz     outlines (optional: wall contact, edge-to-edge gaps, spread)
    background.npy   background (optional: where the channel walls and fixed debris are)

Outputs in <result>/analysis/
    segments.csv     task 1: one row per trusted segment - where and why it starts and ends,
                     plus a motion and shape summary
    motion.csv       task 2: every frame of every trusted segment - position, displacement, velocity
    shape.csv        task 3: every frame of every trusted segment - area, aspect ratio, circularity,
                     orientation, solidity, and whether the shape can be trusted in that frame
    collective.csv   task 4: one row per frame for the whole field - number of regions, total area,
                     spread, how much of the area sits in clusters, gaps between regions, events, data quality
    field_events.csv regions that connect or separate in the image (tracker merge / split), and regions that
                     enter or leave the field of view
    state.csv        unified record: every region in every frame, with review status, motion, shape
                     and whole-field values side by side (input for the round-3 PZT alignment)
    onsets.csv       moments when a region at rest starts to move; several regions starting together are
                     listed in report.json (joint_motion_starts) - candidates to compare with the PZT log
    figures/         task 5: PNG figures (figures.py)
    checks/          side-by-side video frames and curves at automatically chosen key moments
    report.json      parameters, data-quality notes and self-checks

Usage
    python analysis/analyze.py --result results/A_offline --video data/raw/expA.mp4
    python analysis/analyze.py --result results/A_offline --um-per-px 0.65     # adds micrometre columns
    python analysis/analyze.py --result results/A_offline --fps 1000           # if the camera recorded faster
                                                                             # than the file plays back
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------
# Parameters (all written to report.json)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # untrusted rows with these reasons are confirmed false targets and are left out of the whole field;
    # all other untrusted rows and all unreviewed rows are kept but counted as "uncertain"
    exclude_reasons=['Not a bubble (debris or artifact)'],
    velocity_half_window_s=0.05,      # velocity = position change over t-0.05 s ... t+0.05 s
    min_shape_area_px=200,            # below this (about 16 px across) the shape numbers are too coarse
    max_edge_on_wall=0.20,            # more than 20 % of the outline on a wall / fixed debris: shape not reliable
    min_aspect_for_orientation=1.2,   # a nearly round region has no meaningful direction
    border_px=2,                      # outline this close to the image edge: cut off by the field of view
    wall_darkness=0.06,               # static structure = background at least 6 % darker than its surroundings
    wall_dilate_px=2,
    cluster_area_px=300,              # regions this large are counted as clusters (same as compare_runs.py)
    field_uncertain_share=0.05,       # field values are marked uncertain when uncertain regions hold > 5 % of the area
    main_min_area_px=300,             # main objects (own figure): reach cluster size while trusted ...
    main_min_duration_s=1.0,          # ... and are trusted for at least 1 s in total
    rest_speed_px_s=10.0,             # a region "starts moving" at the first frame above 10 px/s after
    rest_s=0.3,                       # staying below it for at least 0.3 s, if it then exceeds
    onset_speed_px_s=30.0,            # 30 px/s within 0.2 s
    onset_group_s=0.1,                # starts within 0.1 s of each other belong to one group
    onset_group_min=3,                # groups with at least 3 objects are reported as joint starts
)


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fmt(v, nd=3):
    if v is None:
        return ''
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return '' if not np.isfinite(v) else round(float(v), nd)
    return v


def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(r.get(k)) for k in header])


def add_um_columns(header, rows, um):
    """Add micrometre copies of every *_px, *_px2 and *_px_s column."""
    if not um:
        return header
    new = []
    for k in header:
        new.append(k)
        for suf, rep, f in (('_px_s', '_um_s', um), ('_px2', '_um2', um * um), ('_px', '_um', um)):
            if k.endswith(suf):
                nk = k[: -len(suf)] + rep
                new.append(nk)
                for r in rows:
                    v = r.get(k)
                    r[nk] = v * f if isinstance(v, (int, float, np.floating, np.integer)) and v is not None else None
                break
    return new


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class Data:
    """Everything read from a result folder, plus per-row review status."""

    def __init__(self, result_dir, P, fps=None, video=None):
        self.dir, self.P = result_dir, P
        p = lambda n: os.path.join(result_dir, n)
        for n in ('objects.csv', 'events.csv'):
            if not os.path.exists(p(n)):
                sys.exit(f'Cannot find {p(n)}')
        self.stats = json.load(open(p('stats.json'), encoding='utf-8')) if os.path.exists(p('stats.json')) else {}
        self.fps_file = float(self.stats.get('fps') or 60.0)
        self.fps = float(fps or self.fps_file)
        self.input_sha1 = {n: sha1(p(n)) for n in ('objects.csv', 'events.csv', 'review.json', 'contours.npz')
                           if os.path.exists(p(n))}

        with open(p('objects.csv'), newline='', encoding='utf-8') as fh:
            rd = csv.DictReader(fh)
            rows = list(rd)
        num = lambda k, t=float: np.array([t(r[k]) for r in rows])
        self.F, self.I = num('frame', int), num('id', int)
        self.n = len(self.F)
        self.t = self.F / self.fps
        for k in ('cx', 'cy', 'vx', 'vy', 'area', 'eq_diam', 'major', 'minor', 'angle', 'circularity', 'solidity'):
            setattr(self, k, num(k))

        self.events = []
        with open(p('events.csv'), newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                self.events.append(dict(frame=int(r['frame']), type=r['type'], ids=[int(x) for x in r['ids'].split()],
                                        result=int(r['result_id']), x=float(r['x']), y=float(r['y'])))

        # image size: background > crop > video
        self.bg = np.load(p('background.npy')).astype(np.float32) if os.path.exists(p('background.npy')) else None
        crop = (self.stats.get('params') or {}).get('crop')
        if self.bg is not None:
            self.H, self.W = self.bg.shape
        elif crop:
            self.W, self.H = int(crop[2]), int(crop[3])
        elif video:
            cap = cv2.VideoCapture(video)
            self.W, self.H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            self.W, self.H = int(self.cx.max()) + 10, int(self.cy.max()) + 10
        self.n_frames = int(self.stats.get('frames') or (self.F.max() + 1))

        self.rows_by_track = defaultdict(list)
        for k in np.lexsort((self.F, self.I)):
            self.rows_by_track[int(self.I[k])].append(int(k))
        self.rows_by_track = {t: np.array(ks) for t, ks in self.rows_by_track.items()}
        self.rows_by_frame = defaultdict(list)
        for k in range(self.n):
            self.rows_by_frame[int(self.F[k])].append(k)

        self.contour = {}
        if os.path.exists(p('contours.npz')):
            z = np.load(p('contours.npz'))
            off, xy = z['offset'], z['xy']
            for k, (f, i) in enumerate(zip(z['frame'], z['id'])):
                self.contour[(int(f), int(i))] = xy[off[k]:off[k + 1]].astype(np.int32)

        self._load_review(p('review.json'))

    def _load_review(self, path):
        self.review = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else None
        self.status = np.full(self.n, 'unreviewed', dtype=object)
        self.reason = np.full(self.n, '', dtype=object)
        self.note = np.full(self.n, '', dtype=object)
        group = {}
        if self.review:
            for s in self.review.get('segments', []):
                ks = self.rows_by_track.get(int(s['track']))
                if ks is None:
                    continue
                m = ks[(self.F[ks] >= s['start']) & (self.F[ks] <= s['end'])]
                self.status[m] = s['status']
                self.reason[m] = s.get('reason', '') or ''
                self.note[m] = s.get('note', '') or ''
            for g in self.review.get('links', []):
                ids = [int(i) for i in (g['ids'] if isinstance(g, dict) else g)]
                for i in ids:
                    group[i] = min(ids)
        self.object_id = np.array([group.get(int(i), int(i)) for i in self.I])
        excl = set(self.P['exclude_reasons'])
        self.role = np.where(self.status == 'trusted', 'confirmed',
                             np.where((self.status == 'untrusted') & np.isin(self.reason, list(excl)),
                                      'excluded', 'uncertain')).astype(object)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def static_structure(bg, darkness, dilate_px):
    """Pixels of channel walls and fixed debris: clearly darker than the smoothed background."""
    if bg is None:
        return None
    try:
        import bubble_tracker as bt
        smooth = bt.smooth_illumination(bg)
    except Exception:  # bubble_tracker not importable: same idea with a plain blur
        smooth = cv2.GaussianBlur(cv2.medianBlur(np.clip(bg, 0, 255).astype(np.uint8), 31).astype(np.float32), (0, 0), 12)
    static = ((1 - bg / (smooth + 1)) > darkness).astype(np.uint8)
    k = 2 * dilate_px + 1
    return cv2.dilate(static, np.ones((k, k), np.uint8)) > 0


def outline_on_wall(c, static, W, H, border):
    """Share of the outline lying on a wall / fixed debris, and whether the outline touches the image edge."""
    c = c.reshape(-1, 2)
    x0, y0 = c.min(0)
    x1, y1 = c.max(0)
    at_border = bool(x0 <= border or y0 <= border or x1 >= W - 1 - border or y1 >= H - 1 - border)
    if static is None:
        return np.nan, at_border
    sub = static[y0:y1 + 1, x0:x1 + 1]
    if not sub.any():
        return 0.0, at_border
    m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.polylines(m, [(c - [x0, y0]).astype(np.int32)], True, 1, 1)
    on = m > 0
    return float(np.count_nonzero(on & sub)) / max(1, np.count_nonzero(on)), at_border


def densify(c, step=1.0):
    """Points along a closed outline, at most `step` px apart (for edge-to-edge distances)."""
    c = c.reshape(-1, 2).astype(np.float64)
    if len(c) < 2:
        return c
    seg = np.roll(c, -1, 0) - c
    n = np.maximum(1, np.ceil(np.hypot(seg[:, 0], seg[:, 1]) / step).astype(int))
    idx = np.repeat(np.arange(len(c)), n)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    frac = (np.arange(n.sum()) - start[idx]) / n[idx]
    return c[idx] + seg[idx] * frac[:, None]


def axial_unwrap(a):
    """Unwrap an orientation in degrees that is only defined modulo 180."""
    out = np.array(a, float)
    for i in range(1, len(out)):
        d = (out[i] - out[i - 1] + 90) % 180 - 90
        out[i] = out[i - 1] + d
    return out


# ---------------------------------------------------------------------------
# Task 1: trusted segments
# ---------------------------------------------------------------------------
def track_ends(D, border):
    """How each track begins and ends, and the connect / separate cuts inside it.

    'entered view' / 'left view' use the outline: a region whose first (last) outline touches the image edge
    came in from (went out of) the field of view. That is not the bubble appearing or vanishing.
    """
    cuts = defaultdict(lambda: defaultdict(list))   # track -> frame -> [(kind, other id)]
    start_why, end_why = {}, {}
    for e in D.events:
        typ, ids, f = e['type'], e['ids'], e['frame']
        if typ == 'merge' and len(ids) == 2:
            absorbed, host = ids
            cuts[host][f].append(('connect', absorbed))
            end_why[absorbed] = f'connected into #{host}'
        elif typ == 'split' and len(ids) == 2:
            parent, child = ids
            cuts[parent][f].append(('separate', child))
            start_why[child] = f'separated from #{parent}'
    last_frame = D.n_frames - 1
    for t, ks in D.rows_by_track.items():
        f0, f1 = int(D.F[ks[0]]), int(D.F[ks[-1]])
        if t not in start_why:
            start_why[t] = ('video start' if f0 == 0 else
                            'entered view' if border[ks[0]] or crosses_edge(D, ks[:4], True) else 'appeared')
        if t not in end_why:
            end_why[t] = ('video end' if f1 >= last_frame else
                          'left view' if border[ks[-1]] or crosses_edge(D, ks[-4:], False) else 'disappeared')
    return cuts, start_why, end_why


def crosses_edge(D, ks, at_start, frames=3):
    """Would the region's outline reach the image edge within `frames` frames before its first row
    (at_start) or after its last row, moving at its initial / final velocity? Catches fast regions whose
    first or last detection is already inside the image (ks: the first or last few rows, in time order)."""
    if len(ks) < 2 or D.F[ks[-1]] == D.F[ks[0]]:
        return False
    vx = (D.cx[ks[-1]] - D.cx[ks[0]]) / (D.F[ks[-1]] - D.F[ks[0]])
    vy = (D.cy[ks[-1]] - D.cy[ks[0]]) / (D.F[ks[-1]] - D.F[ks[0]])
    k0, sgn = (ks[0], -1) if at_start else (ks[-1], 1)
    c = D.contour.get((int(D.F[k0]), int(D.I[k0])))
    if c is not None and len(c):
        x0, y0 = c.min(0)
        x1, y1 = c.max(0)
    else:
        r = D.major[k0] / 2
        x0, y0, x1, y1 = D.cx[k0] - r, D.cy[k0] - r, D.cx[k0] + r, D.cy[k0] + r
    b = D.P['border_px']
    for j in range(1, frames + 1):
        dx, dy = sgn * vx * j, sgn * vy * j
        if x0 + dx <= b or y0 + dy <= b or x1 + dx >= D.W - 1 - b or y1 + dy >= D.H - 1 - b:
            return True
    return False


def build_segments(D, border):
    """Cut every track into trusted segments.

    A segment is a run of consecutive trusted rows of ONE track ID. It also ends where the region connects
    with another region or another region separates from it (tracker merge / split), because the region
    before and after such an event is not the same thing. Segments are never joined to each other, not even
    across linked IDs; object_id only says which segments a person judged to belong to one object.
    """
    cuts, start_why, end_why = track_ends(D, border)

    def words(evs, before):
        parts = []
        for kind, other in evs:
            if kind == 'connect':
                parts.append(f'{"before connecting" if before else "connected"} with #{other}')
            else:
                parts.append(f'{"before" if before else "after"} #{other} separated')
        return '; '.join(parts)

    seg_of = np.full(D.n, 0, int)
    segs = []
    for t in sorted(D.rows_by_track):
        ks = D.rows_by_track[t]
        cf = sorted(cuts[t]) if t in cuts else []
        cur = None
        for j, k in enumerate(ks):
            f = int(D.F[k])
            prev_f = int(D.F[ks[j - 1]]) if j else -1
            crossed = [ev for c in cf if prev_f < c <= f for ev in cuts[t][c]]
            if D.status[k] != 'trusted':
                if cur is not None:
                    cur['end_reason'] = f'{D.status[k]} part follows'
                    segs.append(cur)
                    cur = None
                continue
            if cur is not None and crossed:
                cur['end_reason'] = words(crossed, True)
                segs.append(cur)
                cur = None
            if cur is None:
                if j == 0:
                    why = start_why[t]
                elif crossed:
                    why = words(crossed, False)
                else:
                    why = f'after {D.status[ks[j - 1]]} part'
                cur = dict(track=t, rows=[], start_reason=why, end_reason='')
            cur['rows'].append(int(k))
        if cur is not None:
            cur['end_reason'] = end_why[t]
            segs.append(cur)
    segs.sort(key=lambda s: (D.F[s['rows'][0]], s['track']))
    for n, s in enumerate(segs, start=1):
        s['segment'] = n
        s['rows'] = np.array(s['rows'])
        seg_of[s['rows']] = n
    return segs, seg_of, cuts, start_why, end_why


# ---------------------------------------------------------------------------
# Tasks 2 and 3: motion and shape per trusted segment
# ---------------------------------------------------------------------------
def per_row_geometry(D, static):
    edge = np.full(D.n, np.nan)
    border = np.zeros(D.n, bool)
    b = D.P['border_px']
    for k in range(D.n):
        c = D.contour.get((int(D.F[k]), int(D.I[k])))
        if c is not None and len(c):
            edge[k], border[k] = outline_on_wall(c, static, D.W, D.H, b)
        else:   # no outline: approximate with the ellipse size
            r = D.major[k] / 2
            border[k] = (D.cx[k] - r <= b or D.cy[k] - r <= b or D.cx[k] + r >= D.W - 1 - b
                         or D.cy[k] + r >= D.H - 1 - b)
    return edge, border


def motion_and_shape(D, segs, edge, border):
    P, fps = D.P, D.fps
    hw = max(1, int(round(P['velocity_half_window_s'] * fps)))
    motion, shape = [], []
    for s in segs:
        ks = s['rows']
        f, x, y = D.F[ks], D.cx[ks], D.cy[ks]
        bd = border[ks]
        t = f / fps
        n = len(ks)
        j1 = np.searchsorted(f, f - hw, 'left')
        j2 = np.searchsorted(f, f + hw, 'right') - 1
        dt = (f[j2] - f[j1]) / fps
        with np.errstate(invalid='ignore', divide='ignore'):
            vx = np.where(dt > 0, (x[j2] - x[j1]) / dt, np.nan)
            vy = np.where(dt > 0, (y[j2] - y[j1]) / dt, np.nan)
        speed = np.hypot(vx, vy)
        step = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
        gap = np.r_[0, np.diff(f) - 1]
        path = np.r_[0, np.nancumsum(step[1:])]
        dx, dy = x - x[0], y - y[0]

        area, major, minor = D.area[ks], D.major[ks], D.minor[ks]
        ar = major / np.maximum(minor, 1e-6)
        orient = (180.0 - D.angle[ks]) % 180.0          # counter-clockwise on screen from horizontal
        e = edge[ks]
        small = area < P['min_shape_area_px']
        on_wall = np.nan_to_num(e, nan=0.0) > P['max_edge_on_wall']
        shape_ok = ~small & ~on_wall & ~bd
        quality = np.where(small | bd, 'poor', np.where(on_wall, 'wall contact', 'good'))
        orient_ok = shape_ok & (ar >= P['min_aspect_for_orientation'])
        orient_u = np.full(n, np.nan)
        okk = np.nonzero(orient_ok)[0]
        if len(okk):
            orient_u[okk] = axial_unwrap(orient[okk])
        for i, k in enumerate(ks):
            why = []
            if small[i]:
                why.append('small')
            if on_wall[i]:
                why.append('on wall/debris')
            if bd[i]:
                why.append('cut by image edge')
            base = dict(segment=s['segment'], frame=int(f[i]), t_s=t[i], id=s['track'],
                        object_id=int(D.object_id[k]))
            motion.append(dict(base, x_px=x[i], y_px=y[i], dx_px=dx[i], dy_px=dy[i],
                               disp_px=np.hypot(dx[i], dy[i]), path_px=path[i], step_px=step[i],
                               frames_missing_before=int(gap[i]), vx_px_s=vx[i], vy_px_s=vy[i],
                               speed_px_s=speed[i], at_image_edge=bool(bd[i]),
                               direction_deg=np.degrees(np.arctan2(-vy[i], vx[i])) if np.isfinite(speed[i]) and speed[i] > 0 else np.nan))
            shape.append(dict(base, area_px2=area[i], eq_diam_px=D.eq_diam[k], major_px=major[i], minor_px=minor[i],
                              aspect_ratio=ar[i], circularity=D.circularity[k], solidity=D.solidity[k],
                              orientation_deg=orient[i], orientation_unwrapped_deg=orient_u[i],
                              area_rel_start=area[i] / area[0], edge_on_wall=e[i], at_image_edge=bool(bd[i]),
                              shape_quality=quality[i], shape_ok=bool(shape_ok[i]), orientation_ok=bool(orient_ok[i]),
                              not_ok_because=', '.join(why)))
        # segment summary
        dur = (f[-1] - f[0]) / fps
        net = float(np.hypot(dx[-1], dy[-1]))
        s.update(object_id=int(D.object_id[ks[0]]), start_frame=int(f[0]), end_frame=int(f[-1]),
                 t_start_s=t[0], t_end_s=t[-1], duration_s=dur, n_frames=n,
                 frames_missing=int(gap.sum()),
                 x_start_px=x[0], y_start_px=y[0], x_end_px=x[-1], y_end_px=y[-1],
                 net_disp_px=net, path_px=float(path[-1]),
                 straightness=net / path[-1] if path[-1] > 0 else np.nan,
                 mean_speed_px_s=net / dur if dur > 0 else np.nan,
                 median_speed_px_s=float(np.nanmedian(speed)) if np.isfinite(speed).any() else np.nan,
                 max_speed_px_s=float(np.nanmax(speed)) if np.isfinite(speed).any() else np.nan,
                 net_direction_deg=np.degrees(np.arctan2(-dy[-1], dx[-1])) if net > 0 else np.nan,
                 area_start_px2=area[0], area_end_px2=area[-1], area_median_px2=float(np.median(area)),
                 area_min_px2=float(area.min()), area_max_px2=float(area.max()),
                 area_end_over_start=area[-1] / area[0],
                 aspect_start=ar[0], aspect_end=ar[-1], aspect_median=float(np.median(ar)),
                 circularity_median=float(np.median(D.circularity[ks])),
                 rotation_deg=float(orient_u[okk[-1]] - orient_u[okk[0]]) if len(okk) > 1 else np.nan,
                 shape_ok_share=float(shape_ok.mean()), wall_share=float((np.nan_to_num(e) > 0).mean()),
                 is_cluster=bool(area.max() >= P['cluster_area_px']))
    return motion, shape


# ---------------------------------------------------------------------------
# Task 4: whole field
# ---------------------------------------------------------------------------
def event_rows(D, start_why, end_why):
    """Field events: regions connecting / separating in the image (tracker merge / split), and regions
    entering / leaving the field of view, with the review status of the regions involved."""
    out = []
    for t, ks in D.rows_by_track.items():
        for k, why, typ in ((ks[0], start_why[t], 'enter view'), (ks[-1], end_why[t], 'leave view')):
            if why not in ('entered view', 'left view'):
                continue
            out.append(dict(frame=int(D.F[k]), t_s=D.F[k] / D.fps, type=typ, ids=str(t),
                            object_ids=str(int(D.object_id[k])), result_id=t, x_px=D.cx[k], y_px=D.cy[k],
                            area_px2=D.area[k], region_max_area_px2=float(D.area[ks].max()), review_status=D.status[k],
                            involves_false_target=bool(D.role[k] == 'excluded'),
                            counted_in_field=bool(D.role[k] != 'excluded')))
    for e in D.events:
        if e['type'] not in ('merge', 'split'):
            continue
        roles, sts = [], []
        for i in e['ids']:
            ks = D.rows_by_track.get(i)
            if ks is None:
                continue
            j = np.argmin(np.abs(D.F[ks] - e['frame']))
            roles.append(D.role[ks[j]])
            sts.append(D.status[ks[j]])
        st = 'untrusted' if 'untrusted' in sts else ('trusted' if sts and all(s == 'trusted' for s in sts) else 'unreviewed')
        objs = []
        for i in e['ids']:
            ks = D.rows_by_track.get(i)
            objs.append(int(D.object_id[ks[0]]) if ks is not None else i)
        res = D.rows_by_track.get(e['result'])
        a = D.area[res[np.argmin(np.abs(D.F[res] - e['frame']))]] if res is not None else np.nan
        out.append(dict(frame=e['frame'], t_s=e['frame'] / D.fps,
                        type='connect' if e['type'] == 'merge' else 'separate',
                        ids=' '.join(map(str, e['ids'])), object_ids=' '.join(map(str, objs)),
                        result_id=e['result'], x_px=e['x'], y_px=e['y'], area_px2=a,
                        region_max_area_px2=float(D.area[res].max()) if res is not None else np.nan, review_status=st,
                        involves_false_target=bool('excluded' in roles),
                        counted_in_field=bool('excluded' not in roles)))
    out.sort(key=lambda r: (r['frame'], r['type']))
    return out


def nearest_gaps(pts):
    """Edge-to-edge distance from every region to its nearest other region (pts: list of Nx2 arrays)."""
    n = len(pts)
    if n < 2:
        return np.full(n, np.nan)
    lo = np.array([p.min(0) for p in pts])
    hi = np.array([p.max(0) for p in pts])
    # distance between bounding boxes is a lower bound of the true gap
    dx = np.maximum(0, np.maximum(lo[:, None, 0] - hi[None, :, 0], lo[None, :, 0] - hi[:, None, 0]))
    dy = np.maximum(0, np.maximum(lo[:, None, 1] - hi[None, :, 1], lo[None, :, 1] - hi[:, None, 1]))
    bb = np.hypot(dx, dy)
    np.fill_diagonal(bb, np.inf)
    trees = [None] * n
    best = np.full(n, np.inf)
    exact = {}
    for i in range(n):
        for j in np.argsort(bb[i]):
            if j == i or bb[i, j] >= best[i]:
                break
            key = (min(i, j), max(i, j))
            if key not in exact:
                a, b = key
                if trees[b] is None:
                    trees[b] = cKDTree(pts[b])
                exact[key] = float(trees[b].query(pts[a])[0].min())
            best[i] = min(best[i], exact[key])
    return best


def collective(D, connections, border):
    P = D.P
    kinds = ['connect', 'separate', 'enter view', 'leave view']
    ev_by_frame = defaultdict(lambda: [0, 0, 0, 0])
    for c in connections:
        if c['counted_in_field']:
            ev_by_frame[c['frame']][kinds.index(c['type'])] += 1
    out = []
    for f in range(D.n_frames):
        ks = np.array(D.rows_by_frame.get(f, []), int)
        role = D.role[ks] if len(ks) else np.array([], object)
        inc = ks[role != 'excluded'] if len(ks) else ks
        conf = ks[role == 'confirmed'] if len(ks) else ks
        unc = ks[role == 'uncertain'] if len(ks) else ks
        row = dict(frame=f, t_s=f / D.fps, n_regions=len(inc), n_confirmed=len(conf), n_uncertain=len(unc),
                   n_excluded=int(len(ks) - len(inc)))
        a = D.area[inc]
        tot = float(a.sum())
        row.update(total_area_px2=tot, total_area_confirmed_px2=float(D.area[conf].sum()),
                   uncertain_area_share=float(D.area[unc].sum()) / tot if tot > 0 else np.nan)
        if len(inc):
            x, y = D.cx[inc], D.cy[inc]
            mx, my = float((a * x).sum() / tot), float((a * y).sum() / tot)
            spread = float(np.sqrt((a * ((x - mx) ** 2 + (y - my) ** 2)).sum() / tot))
            big = a >= P['cluster_area_px']
            jmax = int(np.argmax(a))
            pts = []
            for k in inc:
                c = D.contour.get((f, int(D.I[k])))
                if c is not None and len(c) >= 1:
                    pts.append(densify(c))
                else:   # no outline: a circle of the same area
                    r = D.eq_diam[k] / 2
                    th = np.linspace(0, 2 * np.pi, max(8, int(2 * np.pi * r)), endpoint=False)
                    pts.append(np.c_[D.cx[k] + r * np.cos(th), D.cy[k] + r * np.sin(th)])
            allp = np.vstack(pts).astype(np.float32)
            hull = float(cv2.contourArea(cv2.convexHull(allp))) if len(allp) >= 3 else 0.0
            gaps = nearest_gaps(pts)
            if len(conf):
                ac = D.area[conf]
                cx_, cy_ = (ac * D.cx[conf]).sum() / ac.sum(), (ac * D.cy[conf]).sum() / ac.sum()
                spread_c = float(np.sqrt((ac * ((D.cx[conf] - cx_) ** 2 + (D.cy[conf] - cy_) ** 2)).sum() / ac.sum()))
            else:
                spread_c = np.nan
            pair = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
            iu = np.triu_indices(len(inc), 1)
            row.update(centroid_x_px=mx, centroid_y_px=my, spread_px=spread, spread_confirmed_px=spread_c,
                       hull_area_px2=hull, n_clusters=int(big.sum()),
                       cluster_area_share=float(a[big].sum()) / tot,
                       largest_area_px2=float(a[jmax]), largest_share=float(a[jmax]) / tot,
                       largest_object_id=int(D.object_id[inc[jmax]]),
                       nn_gap_median_px=float(np.median(gaps)) if np.isfinite(gaps).any() else np.nan,
                       mean_center_dist_px=float(pair[iu].mean()) if len(iu[0]) else np.nan)
        e = ev_by_frame.get(f, [0, 0, 0, 0])
        row.update(n_at_image_edge=int(border[inc].sum()) if len(inc) else 0,
                   connect_events=e[0], separate_events=e[1], enter_view_events=e[2], leave_view_events=e[3],
                   field_quality='good' if (row['uncertain_area_share'] or 0) <= P['field_uncertain_share'] else 'uncertain')
        out.append(row)
    return out


def motion_onsets(D, segs, motion):
    """Moments when a region at rest starts to move, and groups of regions that start together.

    Several regions starting at the same moment usually means the driving changed (for example a PZT
    switch); these times are meant to be compared with the PZT log in round 3.
    """
    P = D.P
    rest = max(1, int(round(P['rest_s'] * D.fps)))
    ahead = max(1, int(round(0.2 * D.fps)))
    by_seg = defaultdict(list)
    for r in motion:
        by_seg[r['segment']].append(r)
    out = []
    for sg, rows in by_seg.items():
        sp = np.array([r['speed_px_s'] for r in rows], float)
        for i in range(rest, len(rows)):
            # the start is the first frame above the rest speed, after a rest, that leads to real motion
            if (sp[i] >= P['rest_speed_px_s'] and np.all(np.nan_to_num(sp[i - rest:i], nan=np.inf) < P['rest_speed_px_s'])
                    and np.nanmax(sp[i:i + ahead]) > P['onset_speed_px_s']):
                r = rows[i]
                out.append(dict(frame=r['frame'], t_s=r['t_s'], object_id=r['object_id'], id=r['id'], segment=sg,
                                x_px=r['x_px'], y_px=r['y_px'], area_px2=float(D.area[np.nonzero((D.F == r['frame']) & (D.I == r['id']))[0][0]])))
                break   # one start per segment: the first one after a rest
    out.sort(key=lambda r: r['t_s'])
    groups, cur = [], []
    for r in out:
        if cur and r['t_s'] - cur[0]['t_s'] > P['onset_group_s']:
            groups.append(cur)
            cur = []
        cur.append(r)
    if cur:
        groups.append(cur)
    joint = []
    for g in groups:
        objs = sorted({r['object_id'] for r in g})
        if len(objs) >= P['onset_group_min']:
            joint.append(dict(t_first_s=round(g[0]['t_s'], 3), t_last_s=round(g[-1]['t_s'], 3),
                              frame_first=g[0]['frame'], frame_last=g[-1]['frame'], n_objects=len(objs),
                              object_ids=objs))
    for r in out:
        r['joint_start'] = any(j['frame_first'] <= r['frame'] <= j['frame_last'] for j in joint)
    return out, joint


# ---------------------------------------------------------------------------
# Unified state record
# ---------------------------------------------------------------------------
def state_rows(D, seg_of, motion, shape, coll, edge, border):
    mo = {(r['frame'], r['id']): r for r in motion}
    sh = {(r['frame'], r['id']): r for r in shape}
    co = {r['frame']: r for r in coll}
    out = []
    for k in np.lexsort((D.I, D.F)):
        f, i = int(D.F[k]), int(D.I[k])
        m, s, c = mo.get((f, i), {}), sh.get((f, i), {}), co.get(f, {})
        out.append(dict(frame=f, t_s=f / D.fps, id=i, object_id=int(D.object_id[k]),
                        review_status=D.status[k], review_reason=D.reason[k], field_role=D.role[k],
                        segment=int(seg_of[k]) if seg_of[k] else None,
                        x_px=D.cx[k], y_px=D.cy[k], area_px2=D.area[k],
                        edge_on_wall=edge[k], at_image_edge=bool(border[k]),
                        vx_px_s=m.get('vx_px_s'), vy_px_s=m.get('vy_px_s'), speed_px_s=m.get('speed_px_s'),
                        disp_px=m.get('disp_px'),
                        aspect_ratio=s.get('aspect_ratio'), circularity=s.get('circularity'),
                        orientation_deg=s.get('orientation_deg'), solidity=s.get('solidity'),
                        shape_quality=s.get('shape_quality'), shape_ok=s.get('shape_ok'),
                        field_n_regions=c.get('n_regions'), field_total_area_px2=c.get('total_area_px2'),
                        field_spread_px=c.get('spread_px'), field_cluster_area_share=c.get('cluster_area_share'),
                        field_nn_gap_median_px=c.get('nn_gap_median_px'), field_quality=c.get('field_quality')))
    return out


# ---------------------------------------------------------------------------
# Main objects (get their own figure)
# ---------------------------------------------------------------------------
def main_objects(D, segs):
    P = D.P
    by_obj = defaultdict(list)
    for s in segs:
        by_obj[s['object_id']].append(s)
    out = []
    for o, ss in by_obj.items():
        n = sum(s['n_frames'] for s in ss)
        amax = max(s['area_max_px2'] for s in ss)
        if amax >= P['main_min_area_px'] and n / D.fps >= P['main_min_duration_s']:
            ids = sorted({s['track'] for s in ss})
            out.append(dict(object_id=o, ids=ids, segments=[s['segment'] for s in ss], trusted_s=n / D.fps,
                            area_max_px2=amax, area_time=sum(float(D.area[s['rows']].sum()) for s in ss) / D.fps,
                            t_first_s=min(s['t_start_s'] for s in ss), t_last_s=max(s['t_end_s'] for s in ss)))
    out.sort(key=lambda o: -o['area_time'])
    return out


# ---------------------------------------------------------------------------
# Self-checks and data-quality notes
# ---------------------------------------------------------------------------
def checks(D, segs, seg_of, motion, shape, coll, cuts, result_dir):
    res = []

    def add(name, ok, detail=''):
        res.append(dict(check=name, passed=bool(ok), detail=detail))

    trusted = np.nonzero(D.status == 'trusted')[0]
    add('motion and shape use trusted rows only',
        all(D.status[k] == 'trusted' for s in segs for k in s['rows'])
        and len(motion) == len(shape) == len(trusted),
        f'{len(motion)} motion rows, {len(shape)} shape rows, {len(trusted)} trusted rows in objects.csv')
    add('every trusted row is in exactly one segment',
        np.all(seg_of[trusted] > 0) and sum(len(s['rows']) for s in segs) == len(trusted)
        and len(np.unique(np.concatenate([s['rows'] for s in segs]))) == len(trusted) if segs else len(trusted) == 0)
    bad = [s['segment'] for s in segs
           if any(D.F[s['rows'][0]] < c <= D.F[s['rows'][-1]] for c in cuts.get(s['track'], {}))]
    add('no segment runs through a connect / separate event', not bad, f'segments: {bad}' if bad else '')
    bad = [s['segment'] for s in segs if len({int(D.I[k]) for k in s['rows']}) != 1]
    add('every segment holds a single track ID', not bad)

    # velocity agrees with the tracker's own (smoothed) estimate
    sp = np.array([m['speed_px_s'] for m in motion], float)
    tv = np.array([np.hypot(D.vx[k], D.vy[k]) * D.fps_file for s in segs for k in s['rows']], float)
    m = np.isfinite(sp) & (np.maximum(sp, tv) > 30)
    r = float(np.corrcoef(sp[m], tv[m])[0, 1]) if m.sum() > 10 else np.nan
    add('recomputed speed agrees with the tracker speed (r > 0.8 where moving > 30 px/s)',
        not np.isfinite(r) or r > 0.8, f'r = {r:.3f} over {int(m.sum())} rows' if np.isfinite(r) else 'too few moving rows')

    n_inc = np.array([c['n_regions'] for c in coll])
    cnt = np.bincount(D.F[D.role != 'excluded'], minlength=D.n_frames)[:D.n_frames]
    add('whole field counts every region except confirmed false targets', np.array_equal(n_inc, cnt))

    if D.review and D.review.get('objects_sha1'):
        add('review.json was made for this objects.csv', D.review['objects_sha1'] == D.input_sha1['objects.csv'])
    p = os.path.join(result_dir, 'objects_reviewed.csv')
    if os.path.exists(p):
        with open(p, newline='', encoding='utf-8') as fh:
            ex = {(int(r['frame']), int(r['id'])): r['review_status'] for r in csv.DictReader(fh)}
        mism = sum(ex.get((int(D.F[k]), int(D.I[k]))) != D.status[k] for k in range(D.n))
        add('review status matches the round-1 export objects_reviewed.csv', mism == 0, f'{mism} rows differ')
    after = {n: sha1(os.path.join(result_dir, n)) for n in D.input_sha1}
    add('input files unchanged by the analysis', after == D.input_sha1)
    return res


def video_motion_check(D, video, motion, shape, mains, n=100, lag_s=0.1, seed=1):
    """Independent check of the motion curves against the raw video.

    For random trusted frames of the main objects, the region's displacement over `lag_s` is measured a
    second way: the image patch around the region is searched for in the later frame (template matching,
    no segmentation involved). Returns a summary and whether the two agree.
    """
    lag = max(1, int(round(lag_s * D.fps)))
    ids = {o['object_id'] for o in mains[:8]}
    idx = {(r['frame'], r['id']): (r, s) for r, s in zip(motion, shape) if r['object_id'] in ids}
    cands = [(r, s, idx[(r['frame'] + lag, r['id'])][0]) for (f, i), (r, s) in idx.items()
             if (f + lag, i) in idx and not r['at_image_edge'] and not idx[(f + lag, i)][0]['at_image_edge']
             and idx[(f + lag, i)][0]['segment'] == r['segment']]
    if len(cands) < 10:
        return None
    rng = np.random.default_rng(seed)
    moving = [c for c in cands if np.isfinite(c[0]['speed_px_s']) and c[0]['speed_px_s'] > 30]
    pick = list(rng.choice(len(moving), min(len(moving), n // 2), replace=False)) if moving else []
    sample = [moving[k] for k in pick] + [cands[k] for k in rng.choice(len(cands), min(len(cands), n - len(pick)), replace=False)]
    crop = (D.stats.get('params') or {}).get('crop')
    cap = cv2.VideoCapture(video)
    cache = {}

    def gray(f):
        if f not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if not ok:
                return None
            if crop:
                x, y, w, h = crop
                im = im[y:y + h, x:x + w]
            cache[f] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return cache[f]

    def peak(v):
        d = v[0] - 2 * v[1] + v[2]
        return 0.0 if d == 0 else 0.5 * (v[0] - v[2]) / d

    diffs, a_c, a_t = [], [], []
    for r, s, r2 in sample:
        A, B = gray(r['frame']), gray(r['frame'] + lag)
        if A is None or B is None:
            continue
        H, W = A.shape
        X, Y = int(round(r['x_px'])), int(round(r['y_px']))
        h = int(max(12, 0.8 * np.sqrt(s['area_px2'])))
        if X - h < 0 or Y - h < 0 or X + h >= W or Y + h >= H:
            continue
        tpl = A[Y - h:Y + h + 1, X - h:X + h + 1]
        cdx, cdy = r2['x_px'] - r['x_px'], r2['y_px'] - r['y_px']
        S = int(h + np.hypot(cdx, cdy) + 15)
        x0, y0, x1, y1 = max(0, X - S), max(0, Y - S), min(W, X + S + 1), min(H, Y + S + 1)
        if x1 - x0 <= tpl.shape[1] or y1 - y0 <= tpl.shape[0]:
            continue
        res = cv2.matchTemplate(B[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (px, py) = cv2.minMaxLoc(res)
        sx = peak(res[py, px - 1:px + 2]) if 0 < px < res.shape[1] - 1 else 0.0
        sy = peak(res[py - 1:py + 2, px]) if 0 < py < res.shape[0] - 1 else 0.0
        tdx, tdy = x0 + px + sx + h - X, y0 + py + sy + h - Y
        diffs.append(float(np.hypot(tdx - cdx, tdy - cdy)))
        a_c.append(float(np.hypot(cdx, cdy)))
        a_t.append(float(np.hypot(tdx, tdy)))
    cap.release()
    if len(diffs) < 10:
        return None
    d = np.array(diffs)
    r_ = float(np.corrcoef(a_c, a_t)[0, 1]) if np.std(a_c) > 0 and np.std(a_t) > 0 else float('nan')
    return dict(samples=len(d), lag_s=lag / D.fps, median_diff_px=float(np.median(d)),
                p90_diff_px=float(np.percentile(d, 90)), corr_displacement=r_,
                max_displacement_px=float(max(a_c)),
                passed=bool(np.median(d) < 1.0 and (not np.isfinite(r_) or r_ > 0.9)))


def quality_notes(D, segs, shape):
    notes = []
    if not D.review:
        notes.append('No review.json: nothing is trusted yet, so motion.csv and shape.csv are empty and every '
                      'region counts as uncertain in the whole field. Review this result with review/review_tool.py first.')
    else:
        bare = [s for s in D.review.get('segments', []) if s['status'] == 'untrusted' and not s.get('reason')]
        if bare:
            notes.append('Untrusted ranges without a reason are counted as uncertain (kept) in the whole field: '
                         + ', '.join(f"#{s['track']} frames {s['start']}-{s['end']}" for s in bare)
                         + '. If they are not bubbles, give them the reason "Not a bubble" in the review tool and rerun.')
    if shape:
        ok = np.array([r['shape_ok'] for r in shape])
        why = defaultdict(int)
        for r in shape:
            for w in filter(None, r['not_ok_because'].split(', ')):
                why[w] += 1
        notes.append(f'Shape quality over all trusted frames: good {ok.mean() * 100:.0f} %. Reasons for the rest '
                     '(a frame can have several): '
                     + ', '.join(f'{k} {v} frames' for k, v in sorted(why.items(), key=lambda kv: -kv[1])) + '.')
        cl = [r for r in shape if r['area_px2'] >= D.P['cluster_area_px']]
        if cl:
            q = defaultdict(int)
            for r in cl:
                q[r['shape_quality']] += 1
            notes.append(f'Clusters (area >= {D.P["cluster_area_px"]} px), {len(cl)} trusted frames: '
                         + ', '.join(f'{k} {v / len(cl) * 100:.0f} %' for k, v in sorted(q.items()))
                         + '. "wall contact" = more than '
                         f'{D.P["max_edge_on_wall"] * 100:.0f} % of the outline lies on a channel wall or fixed debris: '
                         'the region is real and its position is usable, but area, aspect ratio and orientation are '
                         'shaped by the wall on that side.')
    ends = defaultdict(int)
    for s in segs:
        k = s['end_reason']
        k = 'connect / separate event' if ('connect' in k or 'separated' in k) else k
        ends[k] += 1
    if segs:
        notes.append('Why trusted segments end: ' + ', '.join(f'{k} {v}' for k, v in sorted(ends.items(), key=lambda kv: -kv[1])) + '.')
    notes.append('Positions are region centroids. A centroid also moves when part of a region is hidden by a wall, '
                 'or when a region connects or separates; segments are cut at connect / separate events, '
                 'but wall contact is only flagged (edge_on_wall), not corrected.')
    notes.append('Region speed is the speed of the visible region in the image, not the speed of the liquid.')
    notes.append('Whole-field values also change when a region enters or leaves the field of view; these moments '
                 'are listed in field_events.csv ("enter view" / "leave view") and marked in figures/collective.png. '
                 'A jump that coincides with one of them is a field-of-view effect, not aggregation or dispersion.')
    return notes


# ---------------------------------------------------------------------------
def run(result_dir, video=None, fps=None, um_per_px=None, figures=True, checks_n=8, verbose=True, **over):
    P = dict(DEFAULTS, **{k: v for k, v in over.items() if v is not None})
    log = print if verbose else (lambda *a, **k: None)
    D = Data(result_dir, P, fps=fps, video=video)
    out = os.path.join(result_dir, 'analysis')
    os.makedirs(out, exist_ok=True)
    log(f'{D.n} regions in {D.n_frames} frames, {len(D.rows_by_track)} track IDs, '
        f'{len(set(D.object_id.tolist()))} objects; trusted {np.sum(D.status == "trusted")}, '
        f'untrusted {np.sum(D.status == "untrusted")}, unreviewed {np.sum(D.status == "unreviewed")}')

    static = static_structure(D.bg, P['wall_darkness'], P['wall_dilate_px'])
    edge, border = per_row_geometry(D, static)
    segs, seg_of, cuts, start_why, end_why = build_segments(D, border)
    motion, shape = motion_and_shape(D, segs, edge, border)
    log(f'task 1-3: {len(segs)} trusted segments, {len(motion)} frames of motion and shape')
    conns = event_rows(D, start_why, end_why)
    coll = collective(D, conns, border)
    log(f'task 4: whole field over {len(coll)} frames, {len(conns)} field events')
    state = state_rows(D, seg_of, motion, shape, coll, edge, border)
    mains = main_objects(D, segs)
    onsets, joint = motion_onsets(D, segs, motion)
    if joint:
        log('joint starts of motion: ' + ', '.join(f"{j['t_first_s']:.2f} s ({j['n_objects']} objects)" for j in joint))

    # ---- tables ----
    seg_cols = ['segment', 'id', 'object_id', 'start_frame', 'end_frame', 't_start_s', 't_end_s', 'duration_s',
                'n_frames', 'frames_missing', 'start_reason', 'end_reason', 'x_start_px', 'y_start_px', 'x_end_px',
                'y_end_px', 'net_disp_px', 'path_px', 'straightness', 'mean_speed_px_s', 'median_speed_px_s',
                'max_speed_px_s', 'net_direction_deg', 'area_start_px2', 'area_end_px2', 'area_median_px2',
                'area_min_px2', 'area_max_px2', 'area_end_over_start', 'aspect_start', 'aspect_end', 'aspect_median',
                'circularity_median', 'rotation_deg', 'shape_ok_share', 'wall_share', 'is_cluster']
    seg_rows = [dict(s, id=s['track']) for s in segs]
    tables = {
        'segments.csv': (seg_cols, seg_rows),
        'motion.csv': (list(motion[0].keys()) if motion else
                       ['segment', 'frame', 't_s', 'id', 'object_id', 'x_px', 'y_px', 'dx_px', 'dy_px', 'disp_px',
                        'path_px', 'step_px', 'frames_missing_before', 'vx_px_s', 'vy_px_s', 'speed_px_s',
                        'at_image_edge', 'direction_deg'], motion),
        'shape.csv': (list(shape[0].keys()) if shape else
                      ['segment', 'frame', 't_s', 'id', 'object_id', 'area_px2', 'eq_diam_px', 'major_px', 'minor_px',
                       'aspect_ratio', 'circularity', 'solidity', 'orientation_deg', 'orientation_unwrapped_deg',
                       'area_rel_start', 'edge_on_wall', 'at_image_edge', 'shape_quality', 'shape_ok', 'orientation_ok',
                       'not_ok_because'], shape),
        'collective.csv': (['frame', 't_s', 'n_regions', 'n_confirmed', 'n_uncertain', 'n_excluded', 'total_area_px2',
                            'total_area_confirmed_px2', 'uncertain_area_share', 'centroid_x_px', 'centroid_y_px',
                            'spread_px', 'spread_confirmed_px', 'hull_area_px2', 'n_clusters', 'cluster_area_share',
                            'largest_area_px2', 'largest_share', 'largest_object_id', 'nn_gap_median_px',
                            'mean_center_dist_px', 'n_at_image_edge', 'connect_events', 'separate_events',
                            'enter_view_events', 'leave_view_events', 'field_quality'], coll),
        'field_events.csv': (['frame', 't_s', 'type', 'ids', 'object_ids', 'result_id', 'x_px', 'y_px', 'area_px2',
                              'region_max_area_px2',
                              'review_status', 'involves_false_target', 'counted_in_field'], conns),
        'state.csv': (list(state[0].keys()) if state else [], state),
        'onsets.csv': (['frame', 't_s', 'object_id', 'id', 'segment', 'x_px', 'y_px', 'area_px2', 'joint_start'], onsets),
    }
    for name, (cols, rows) in tables.items():
        cols = add_um_columns(cols, rows, um_per_px)
        write_csv(os.path.join(out, name), cols, rows)
    log('tables written: ' + ', '.join(tables))

    # ---- figures and key-moment checks ----
    fig_files, check_items = [], []
    if figures:
        import figures as figs
        fig_files = figs.make_all(D, out, segs, motion, shape, coll, conns, mains, static, joint)
        log(f'task 5: {len(fig_files)} figures')
        if video and checks_n:
            check_items = figs.key_moment_checks(D, video, out, segs, motion, shape, coll, conns, mains, n=checks_n)
            log(f'{len(check_items)} key-moment check sheets')

    res = checks(D, segs, seg_of, motion, shape, coll, cuts, result_dir)
    vm = video_motion_check(D, video, motion, shape, mains) if video and motion else None
    if vm:
        res.append(dict(check='motion agrees with the raw video (template matching, no segmentation): '
                              'median difference < 1 px over 0.1 s',
                        passed=vm['passed'],
                        detail=f"{vm['samples']} random frames of main objects: median {vm['median_diff_px']:.2f} px, "
                               f"90 % {vm['p90_diff_px']:.2f} px, r = {vm['corr_displacement']:.3f} "
                               f"(displacements up to {vm['max_displacement_px']:.0f} px)"))
    report = dict(
        created=now(), result_dir=result_dir, video=video,
        frames=D.n_frames, fps=D.fps, fps_in_stats=D.fps_file, image_size_px=[D.W, D.H],
        units=dict(length='px' + (f' (and um, {um_per_px} um per px)' if um_per_px else ''), time='s',
                   direction='degrees, 0 = right, 90 = up on screen, 180 = left, -90 = down',
                   orientation='degrees, 0-180, counter-clockwise on screen from horizontal'),
        um_per_px=um_per_px, parameters=P,
        inputs=dict(sha1=D.input_sha1, review_present=bool(D.review),
                    review_updated=(D.review or {}).get('updated')),
        rows=dict(total=D.n, trusted=int(np.sum(D.status == 'trusted')), untrusted=int(np.sum(D.status == 'untrusted')),
                  unreviewed=int(np.sum(D.status == 'unreviewed')),
                  field_confirmed=int(np.sum(D.role == 'confirmed')), field_uncertain=int(np.sum(D.role == 'uncertain')),
                  field_excluded=int(np.sum(D.role == 'excluded'))),
        tracks=len(D.rows_by_track), objects=len(set(D.object_id.tolist())), trusted_segments=len(segs),
        main_objects=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()} for o in mains],
        field_summary=field_summary(coll, D.fps),
        joint_motion_starts=joint,
        data_quality=quality_notes(D, segs, shape),
        checks=res, all_checks_passed=all(c['passed'] for c in res), video_motion_check=vm,
        key_moment_checks=check_items,
        files={'segments.csv': 'task 1: one row per trusted segment (start/end and why, motion and shape summary)',
               'motion.csv': 'task 2: per frame of each trusted segment: position, displacement, velocity',
               'shape.csv': 'task 3: per frame of each trusted segment: area, aspect ratio, circularity, '
                            'orientation, solidity, reliability',
               'collective.csv': 'task 4: per frame, whole field (all regions except confirmed false targets)',
               'field_events.csv': 'regions connecting / separating in the image (tracker merge / split) and '
                                   'entering / leaving the field of view',
               'state.csv': 'every region in every frame with review status, motion, shape and field values',
               'onsets.csv': 'moments when a trusted region at rest starts to move; joint_start = together with '
                             'at least two other objects (compare with the PZT log in round 3)',
               'figures/': 'task 5: figures', 'checks/': 'video frames next to the curves at key moments'},
    )
    with open(os.path.join(out, 'report.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    for c in res:
        log(('  ok    ' if c['passed'] else '  FAIL  ') + c['check'] + (f'  ({c["detail"]})' if c['detail'] else ''))
    for nline in report['data_quality']:
        log('  note: ' + nline)
    log(f'written to {out}')
    return report


def field_summary(coll, fps):
    def at(rows, k):
        v = [r.get(k) for r in rows if r.get(k) is not None and np.isfinite(r.get(k))]
        return round(float(np.median(v)), 3) if v else None
    n = len(coll)
    first, last = coll[:max(1, int(fps))], coll[-max(1, int(fps)):]
    keys = ['n_regions', 'total_area_px2', 'spread_px', 'cluster_area_share', 'largest_share', 'nn_gap_median_px']
    return dict(first_second={k: at(first, k) for k in keys}, last_second={k: at(last, k) for k in keys},
                connect_events=int(sum(r['connect_events'] for r in coll)),
                separate_events=int(sum(r['separate_events'] for r in coll)),
                enter_view_events=int(sum(r['enter_view_events'] for r in coll)),
                leave_view_events=int(sum(r['leave_view_events'] for r in coll)),
                frames_uncertain=int(sum(r['field_quality'] == 'uncertain' for r in coll)), frames=n)


def main():
    ap = argparse.ArgumentParser(description='Round 2: motion, shape and whole-field aggregation from reviewed tracks')
    ap.add_argument('--result', required=True, help='result folder (objects.csv, events.csv, review.json, ...)')
    ap.add_argument('--video', help='original video; needed only for the key-moment check sheets')
    ap.add_argument('--fps', type=float, help='real recording frame rate, if different from stats.json')
    ap.add_argument('--um-per-px', type=float, help='micrometres per pixel; adds _um columns')
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--checks', type=int, default=8, help='number of key-moment check sheets (0 = none)')
    ap.add_argument('--exclude-reason', action='append',
                    help='untrusted reason that marks a confirmed false target (repeatable; '
                         'default: "Not a bubble (debris or artifact)")')
    a = ap.parse_args()
    rep = run(a.result, video=a.video, fps=a.fps, um_per_px=a.um_per_px, figures=not a.no_figures,
              checks_n=a.checks, exclude_reasons=a.exclude_reason)
    sys.exit(0 if rep['all_checks_passed'] else 1)


if __name__ == '__main__':
    main()
