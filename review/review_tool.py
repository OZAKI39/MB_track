#!/usr/bin/env python3
"""
review_tool.py — manual track review and data cleaning

Review the automatically tracked trajectories in a browser, mark a whole track or
part of it as trusted / untrusted / unreviewed, link IDs that belong to the same
physical object, and export the three classes separately. The original
objects.csv and events.csv are never modified.

Preparation (once per result folder): the result folder needs contours.npz so the
selected track can be outlined.
    python bubble_tracker.py <video> ... --out <result folder> --save-contours
Without contours.npz the tool still works but draws ellipses instead of outlines.

Start:
    python review/review_tool.py --video data/raw/expA.mp4 --result results/A_offline
The browser opens http://127.0.0.1:8800 . Every change is saved immediately:
    review.json            all decisions (segments, same-object links, full history, undo)
    trusted.csv            trusted object-frames
    untrusted.csv          untrusted object-frames (with reason)
    unreviewed.csv         object-frames not reviewed yet
    objects_reviewed.csv   all object-frames with review columns
    events_reviewed.csv    events with review status and object IDs
    review_report.txt      export check report
Every exported row keeps the original objects.csv columns and adds:
    review_status, review_reason, review_note, touches_static, object_id

Export only (no browser), plus a colour-coded check video review_overlay.mp4:
    python review/review_tool.py --video ... --result ... --export --render
"""
import argparse
import bisect
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
try:
    import bubble_tracker as bt
except ImportError:  # the tool still runs on its own, only without the wall-contact hint
    bt = None

STATUSES = ('trusted', 'untrusted', 'unreviewed')
REASONS = ['ID switched to another object', 'Contour includes channel wall or debris',
           'Contour incomplete or cut off', 'False merge (not actually touching)', 'False split',
           'Not a bubble (debris or artifact)', 'Occluded or out of focus, cannot judge', 'Other']
SUGGEST_MAX_GAP = 300      # frames: how long an object may vanish and still be suggested as the same one
SUGGEST_MAX_DIST = 12.0    # px between where one ID ends and the next one starts (or its predicted position)
SUGGEST_MAX_AREA_RATIO = 2.5


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Store:
    def __init__(self, video, result_dir, max_width=1280, quality=85, load_frames=True, verbose=True):
        self.video, self.dir = video, result_dir
        self.lock = threading.Lock()
        p = lambda n: os.path.join(result_dir, n)
        for n in ('objects.csv', 'events.csv'):
            if not os.path.exists(p(n)):
                sys.exit(f'Cannot find {p(n)}')
        stats = json.load(open(p('stats.json'), encoding='utf-8')) if os.path.exists(p('stats.json')) else {}
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            sys.exit(f'Cannot open video {video}')
        self.fps = float(stats.get('fps') or cap.get(cv2.CAP_PROP_FPS) or 60.0)
        self.W, self.H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        crop = (stats.get('params') or {}).get('crop')
        self.crop = tuple(crop) if crop else None
        if self.crop:
            self.W, self.H = self.crop[2], self.crop[3]

        # ---- objects.csv: keep the original text so exports reproduce it exactly ----
        with open(p('objects.csv'), newline='') as fh:
            rd = csv.reader(fh)
            self.obj_header = next(rd)
            self.obj_rows = [r for r in rd]
        ix = {k: i for i, k in enumerate(self.obj_header)}
        self.ix = ix
        col = lambda k, f=float: np.array([f(r[ix[k]]) for r in self.obj_rows])
        self.F, self.I = col('frame', int).astype(np.int64), col('id', int).astype(np.int64)
        self.cx, self.cy, self.area = col('cx'), col('cy'), col('area')
        self.major, self.minor, self.angle = col('major'), col('minor'), col('angle')
        self.objects_sha1 = sha1(p('objects.csv'))
        self.n_frames = max(n_video, int(self.F.max()) + 1 if len(self.F) else 0)

        # ---- events.csv ----
        with open(p('events.csv'), newline='') as fh:
            rd = csv.reader(fh)
            self.ev_header = next(rd)
            self.ev_rows = [r for r in rd]
        eix = {k: i for i, k in enumerate(self.ev_header)}
        self.events = [dict(frame=int(r[eix['frame']]), type=r[eix['type']],
                            ids=[int(x) for x in r[eix['ids']].split()], result=int(r[eix['result_id']]))
                       for r in self.ev_rows]

        # ---- contours ----
        self.contour = {}
        cpath = p('contours.npz')
        self.has_contours = os.path.exists(cpath)
        if self.has_contours:
            z = np.load(cpath)
            off, xy = z['offset'], z['xy']
            for k, (f, i) in enumerate(zip(z['frame'], z['id'])):
                self.contour[(int(f), int(i))] = xy[off[k]:off[k + 1]]
        elif verbose:
            print('Note: no contours.npz in the result folder; objects will be drawn as ellipses. '
                  'Create it with: bubble_tracker.py ... --save-contours')

        # ---- indexes ----
        self.rows_by_frame = defaultdict(list)
        self.rows_by_track = defaultdict(list)
        for k in np.lexsort((self.F, self.I)):
            self.rows_by_track[int(self.I[k])].append(int(k))
        for k in range(len(self.F)):
            self.rows_by_frame[int(self.F[k])].append(k)
        self.frameset = {t: set(self.F[ks].tolist()) for t, ks in self.rows_by_track.items()}
        self.wall = self._wall_contact(p('background.npy'))
        self.tracks = self._track_summaries()
        self.ev_by_track = defaultdict(list)
        for e in self.events:
            for i in e['ids']:
                self.ev_by_track[i].append(e)

        # ---- review record ----
        self.review_path = p('review.json')
        if os.path.exists(self.review_path):
            self.review = json.load(open(self.review_path, encoding='utf-8'))
            if self.review.get('objects_sha1') != self.objects_sha1 and verbose:
                print('WARNING: objects.csv differs from the version this review.json was made for.')
        else:
            self.review = dict(version=2, video=os.path.basename(video), result_dir=os.path.abspath(result_dir),
                               objects_sha1=self.objects_sha1, fps=self.fps, created=now(),
                               segments=[], links=[], dismissed=[], history=[])
        self.review.setdefault('links', [])
        self.review.setdefault('dismissed', [])
        self.review.setdefault('history', [])
        self.segs = defaultdict(list)
        for s in self.review['segments']:
            self.segs[int(s['track'])].append(s)
        for t in self.segs:
            self.segs[t].sort(key=lambda s: s['start'])
        self.groups = [sorted(int(i) for i in g['ids']) for g in self.review['links']]
        self.dismissed = {tuple(sorted(map(int, d))) for d in self.review['dismissed']}
        self._index_groups()

        self.frames, self.scale = [], 1.0
        if load_frames:
            self._load_frames(max_width, quality, verbose)

    # ---------------- loading ----------------
    def _load_frames(self, max_width, quality, verbose):
        cap = cv2.VideoCapture(self.video)
        self.scale = min(1.0, max_width / self.W)
        t0, i = time.time(), 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if self.crop:
                x, y, w, h = self.crop
                fr = fr[y:y + h, x:x + w]
            if self.scale < 1:
                fr = cv2.resize(fr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            self.frames.append(cv2.imencode('.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes())
            i += 1
            if verbose and i % 300 == 0:
                print(f'  loaded {i} frames ...', flush=True)
        cap.release()
        self.n_frames = max(self.n_frames, len(self.frames))
        if verbose:
            mb = sum(len(b) for b in self.frames) / 1e6
            print(f'  {len(self.frames)} frames, {mb:.0f} MB cached, {time.time() - t0:.1f} s')

    def _wall_contact(self, bg_path):
        """Does the object touch a static dark structure of the background (channel wall, fixed debris)?"""
        wall = np.zeros(len(self.F), np.uint8)
        if bt is None or not os.path.exists(bg_path) or not self.has_contours:
            return wall
        B = np.load(bg_path).astype(np.float32)
        static = (1 - B / (bt.smooth_illumination(B) + 1)) > 0.06
        static = cv2.dilate(static.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        for k in range(len(self.F)):
            c = self.contour.get((int(self.F[k]), int(self.I[k])))
            if c is None or len(c) < 3:
                continue
            x, y, w, h = cv2.boundingRect(c.astype(np.int32))
            sub = static[y:y + h, x:x + w]
            if not sub.any():
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [c.astype(np.int32) - [x, y]], -1, 1, -1)
            wall[k] = int(np.count_nonzero(sub & (m[:sub.shape[0], :sub.shape[1]] > 0)) >= 3)
        return wall

    def _track_summaries(self):
        out = {}
        for t, ks in self.rows_by_track.items():
            ks = np.array(ks)
            f, a, x, y = self.F[ks], self.area[ks], self.cx[ks], self.cy[ks]
            out[t] = dict(id=t, start=int(f[0]), end=int(f[-1]), n=int(len(ks)),
                          area_med=float(np.median(a)), area_max=float(a.max()),
                          path=float(np.hypot(np.diff(x), np.diff(y)).sum()), wall_frac=float(self.wall[ks].mean()))
        return out

    # ---------------- same-object groups ----------------
    def _index_groups(self):
        self.groups = [g for g in self.groups if len(g) > 1]
        self.group_index = {t: k for k, g in enumerate(self.groups) for t in g}

    def group_of(self, t):
        k = self.group_index.get(t)
        return list(self.groups[k]) if k is not None else [t]

    def obj_id(self, t):
        return min(self.group_of(t))

    def _overlap(self, ids):
        ids = sorted(ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if self.frameset[a] & self.frameset[b]:
                    return a, b
        return None

    def _save_links(self):
        self.review['links'] = [dict(ids=g, object_id=min(g)) for g in self.groups]
        self.review['dismissed'] = [list(d) for d in sorted(self.dismissed)]

    def link(self, ids):
        ids = [int(i) for i in ids]
        for i in ids:
            if i not in self.tracks:
                raise ValueError(f'There is no track #{i}')
        members = sorted(set(m for i in ids for m in self.group_of(i)))
        if len(members) < 2:
            raise ValueError('Choose at least two different tracks to link')
        clash = self._overlap(members)
        if clash:
            raise ValueError(f'#{clash[0]} and #{clash[1]} exist in the same frames, so they cannot be the same object')
        before = [list(g) for g in self.groups]
        self.groups = [g for g in self.groups if not set(g) & set(members)] + [members]
        self._index_groups()
        self.review['history'].append(dict(time=now(), action='link', ids=members, before_links=before))
        self._persist()
        return members

    def unlink(self, t):
        t = int(t)
        g = self.group_of(t)
        if len(g) < 2:
            raise ValueError(f'#{t} is not linked to any other track')
        before = [list(x) for x in self.groups]
        k = self.group_index[t]
        self.groups[k] = [i for i in g if i != t]
        self._index_groups()
        self.review['history'].append(dict(time=now(), action='unlink', id=t, before_links=before))
        self._persist()

    def dismiss(self, a, b):
        pair = tuple(sorted((int(a), int(b))))
        before = [list(d) for d in sorted(self.dismissed)]
        self.dismissed.add(pair)
        self.review['history'].append(dict(time=now(), action='dismiss', pair=list(pair), before_dismissed=before))
        self._persist()

    def suggestions(self):
        """Pairs of IDs that probably belong to one object: one ends, the other starts later where it ended."""
        info = {}
        for t, ks in self.rows_by_track.items():
            ks = np.array(ks)
            f, x, y, a = self.F[ks], self.cx[ks], self.cy[ks], self.area[ks]
            tail = slice(max(0, len(ks) - 6), len(ks))
            v = ((x[tail][-1] - x[tail][0]) / max(1, f[tail][-1] - f[tail][0]),
                 (y[tail][-1] - y[tail][0]) / max(1, f[tail][-1] - f[tail][0])) if len(ks) > 1 else (0.0, 0.0)
            info[t] = dict(s=int(f[0]), e=int(f[-1]), xs=x[0], ys=y[0], xe=x[-1], ye=y[-1], v=v,
                           a_s=float(np.median(a[:5])), a_e=float(np.median(a[-5:])))
        cand = []
        ids = sorted(info)
        # a linked object can only be extended at its ends: after its last ID or before its first ID
        tail = {a for a in ids if a == max(self.group_of(a), key=lambda t: info[t]['e'])}
        head = {b for b in ids if b == min(self.group_of(b), key=lambda t: info[t]['s'])}
        for a in ids:
            if a not in tail:
                continue
            A = info[a]
            for b in ids:
                if b == a or b not in head:
                    continue
                Bi = info[b]
                gap = Bi['s'] - A['e']
                if gap < 1 or gap > SUGGEST_MAX_GAP:
                    continue
                if self.obj_id(a) == self.obj_id(b) or tuple(sorted((a, b))) in self.dismissed:
                    continue
                d_end = float(np.hypot(Bi['xs'] - A['xe'], Bi['ys'] - A['ye']))
                px, py = A['xe'] + A['v'][0] * min(gap, 30), A['ye'] + A['v'][1] * min(gap, 30)
                d = min(d_end, float(np.hypot(Bi['xs'] - px, Bi['ys'] - py)))
                ratio = max(A['a_e'], Bi['a_s']) / max(1.0, min(A['a_e'], Bi['a_s']))
                if d <= SUGGEST_MAX_DIST and ratio <= SUGGEST_MAX_AREA_RATIO:
                    cand.append((d / SUGGEST_MAX_DIST + gap / SUGGEST_MAX_GAP, a, b, gap, d, A['a_e'], Bi['a_s']))
        cand.sort()
        used_a, used_b, out = set(), set(), []
        for score, a, b, gap, d, aa, ab in cand:
            if a in used_a or b in used_b:
                continue
            if self._overlap(set(self.group_of(a)) | set(self.group_of(b))):
                continue
            used_a.add(a)
            used_b.add(b)
            out.append(dict(a=a, b=b, gap=gap, dist=round(d, 1), area_a=round(aa), area_b=round(ab),
                            score=round(score, 3)))
        return out

    # ---------------- review segments ----------------
    def status_at(self, track, frame):
        segs = self.segs.get(track)
        if not segs:
            return None
        k = bisect.bisect_right([s['start'] for s in segs], frame) - 1
        if k >= 0 and segs[k]['start'] <= frame <= segs[k]['end']:
            return segs[k]
        return None

    def _set_one(self, track, start, end, status, reason, note):
        before = [dict(s) for s in self.segs.get(track, [])]
        new = []
        for s in before:
            if s['end'] < start or s['start'] > end:
                new.append(dict(s))
                continue
            if s['start'] < start:
                new.append(dict(s, end=start - 1))
            if s['end'] > end:
                new.append(dict(s, start=end + 1))
        if status != 'unreviewed':
            new.append(dict(track=track, start=start, end=end, status=status,
                            reason=reason if status == 'untrusted' else '', note=note, updated=now()))
        new.sort(key=lambda s: s['start'])
        merged = []
        for s in new:
            if merged and merged[-1]['end'] + 1 == s['start'] and all(
                    merged[-1][k] == s[k] for k in ('status', 'reason', 'note')):
                merged[-1]['end'] = s['end']
            else:
                merged.append(s)
        self.segs[track] = merged
        return before

    def set_range(self, track, start, end, status, reason='', note='', whole_object=False):
        track = int(track)
        if track not in self.tracks:
            raise ValueError(f'There is no track #{track}')
        if status not in STATUSES:
            raise ValueError(f'Unknown status {status}')
        members = self.group_of(track) if whole_object else [track]
        items = []
        for t in members:
            tr = self.tracks[t]
            s, e = max(int(start), tr['start']), min(int(end), tr['end'])
            if s <= e:
                items.append(dict(track=t, start=s, end=e, before=self._set_one(t, s, e, status, reason, note)))
        if not items:
            raise ValueError('The selected range does not overlap this track')
        self.review['history'].append(dict(time=now(), action='set', status=status, reason=reason, note=note,
                                           items=items))
        self._persist()
        return [i['track'] for i in items]

    def undo(self):
        """Undo the most recent change that has not been undone. Returns the affected track IDs."""
        for k in range(len(self.review['history']) - 1, -1, -1):
            h = self.review['history'][k]
            if h.get('undone') or h.get('action') not in ('set', 'link', 'unlink', 'dismiss'):
                continue
            touched = []
            if h['action'] == 'set':
                items = h.get('items') or [dict(track=h['track'], before=h['before'])]   # old format
                for it in items:
                    self.segs[int(it['track'])] = [dict(s) for s in it['before']]
                    touched.append(int(it['track']))
            elif h['action'] in ('link', 'unlink'):
                old = set(t for g in self.groups for t in g)
                self.groups = [list(g) for g in h['before_links']]
                self._index_groups()
                touched = sorted(old | set(t for g in self.groups for t in g))
            else:
                self.dismissed = {tuple(d) for d in h['before_dismissed']}
                touched = list(h['pair'])
            h['undone'] = True
            self.review['history'].append(dict(time=now(), action='undo', undo_of=k))
            self._persist()
            return dict(action=h['action'], tracks=touched)
        return None

    def can_undo(self):
        return any(not h.get('undone') and h.get('action') in ('set', 'link', 'unlink', 'dismiss')
                   for h in self.review['history'])

    def progress(self):
        cnt = {s: 0 for s in STATUSES}
        for t, ks in self.rows_by_track.items():
            fr = self.F[ks]
            covered = np.zeros(len(ks), bool)
            for s in self.segs.get(t, []):
                m = (fr >= s['start']) & (fr <= s['end'])
                cnt[s['status']] += int(m.sum())
                covered |= m
            cnt['unreviewed'] += int((~covered).sum())
        done = sum(1 for t in self.tracks if self.track_unreviewed(t) == 0)
        return dict(total_rows=len(self.F), **cnt, tracks_total=len(self.tracks), tracks_done=done,
                    objects_total=len(set(self.obj_id(t) for t in self.tracks)), linked_groups=len(self.groups))

    def track_unreviewed(self, t):
        tr = self.tracks[t]
        return (tr['end'] - tr['start'] + 1) - sum(s['end'] - s['start'] + 1 for s in self.segs.get(t, []))

    # ---------------- saving and export ----------------
    def _persist(self):
        self.review['segments'] = [s for t in sorted(self.segs) for s in self.segs[t]]
        self._save_links()
        self.review['version'] = 2
        self.review['updated'] = now()
        self.review['objects_sha1'] = self.objects_sha1
        tmp = self.review_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(self.review, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.review_path)
        self.export_tables()

    def row_status(self):
        st = np.array(['unreviewed'] * len(self.F), dtype=object)
        rs = np.array([''] * len(self.F), dtype=object)
        nt = np.array([''] * len(self.F), dtype=object)
        for t, ks in self.rows_by_track.items():
            ks = np.array(ks)
            fr = self.F[ks]
            for s in self.segs.get(t, []):
                m = ks[(fr >= s['start']) & (fr <= s['end'])]
                st[m], rs[m], nt[m] = s['status'], s.get('reason', ''), s.get('note', '')
        return st, rs, nt

    def export_tables(self):
        st, rs, nt = self.row_status()
        p = lambda n: os.path.join(self.dir, n)
        head = self.obj_header + ['review_status', 'review_reason', 'review_note', 'touches_static', 'object_id']
        files = {s: open(p(f'{s}.csv'), 'w', newline='', encoding='utf-8') for s in STATUSES}
        allf = open(p('objects_reviewed.csv'), 'w', newline='', encoding='utf-8')
        writers = {s: csv.writer(fh) for s, fh in files.items()}
        all_w = csv.writer(allf)
        for w in list(writers.values()) + [all_w]:
            w.writerow(head)
        counts = {s: 0 for s in STATUSES}
        for k, r in enumerate(self.obj_rows):
            row = r + [st[k], rs[k], nt[k], int(self.wall[k]), self.obj_id(int(self.I[k]))]
            writers[st[k]].writerow(row)
            all_w.writerow(row)
            counts[st[k]] += 1
        for fh in list(files.values()) + [allf]:
            fh.close()
        with open(p('events_reviewed.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(self.ev_header + ['review_status', 'object_ids'])
            for r, e in zip(self.ev_rows, self.events):
                sts = []
                for i in e['ids']:
                    if i not in self.tracks:
                        continue
                    tr = self.tracks[i]
                    fr = min(max(e['frame'], tr['start']), tr['end'])
                    s = self.status_at(i, fr) or self.status_at(i, e['frame'] - 1)
                    sts.append(s['status'] if s else 'unreviewed')
                es = 'untrusted' if 'untrusted' in sts else ('trusted' if sts and all(x == 'trusted' for x in sts)
                                                               else 'unreviewed')
                w.writerow(r + [es, ' '.join(str(self.obj_id(i)) if i in self.tracks else str(i) for i in e['ids'])])
        return counts

    def verify(self):
        lines, ok = [], True
        orig = {(r[self.ix['frame']], r[self.ix['id']]): r for r in self.obj_rows}
        seen = {}
        nbase = len(self.obj_header)
        for s in STATUSES:
            with open(os.path.join(self.dir, f'{s}.csv'), newline='', encoding='utf-8') as fh:
                rd = csv.reader(fh)
                next(rd)
                n = 0
                for r in rd:
                    key = (r[self.ix['frame']], r[self.ix['id']])
                    if orig.get(key) != r[:nbase]:
                        ok = False
                        lines.append(f'  [ERROR] {s}.csv line {n + 2} differs from objects.csv (frame {key[0]}, ID {key[1]})')
                    if key in seen:
                        ok = False
                        lines.append(f'  [ERROR] frame {key[0]} ID {key[1]} is in both {seen[key]} and {s}')
                    seen[key] = s
                    if r[nbase] != s:
                        ok = False
                        lines.append(f'  [ERROR] {s}.csv contains a row with status {r[nbase]}')
                    n += 1
                lines.append(f'  {s}.csv: {n} rows')
        if len(seen) != len(orig):
            ok = False
            lines.append(f'  [ERROR] the three files hold {len(seen)} rows, objects.csv has {len(orig)}')
        if sha1(os.path.join(self.dir, 'objects.csv')) != self.objects_sha1:
            ok = False
            lines.append('  [ERROR] objects.csv was modified during the review')
        for t, segs in self.segs.items():
            tr = self.tracks.get(t)
            for s in segs:
                if tr is None or s['start'] < tr['start'] or s['end'] > tr['end']:
                    ok = False
                    lines.append(f'  [ERROR] track #{t}: segment {s["start"]}-{s["end"]} lies outside the track')
        for g in self.groups:
            clash = self._overlap(g)
            if clash:
                ok = False
                lines.append(f'  [ERROR] linked IDs #{clash[0]} and #{clash[1]} exist in the same frames')
        lines.append(f'  same-object links: {len(self.groups)} groups covering '
                     f'{sum(len(g) for g in self.groups)} IDs; {len(self.tracks)} IDs -> '
                     f'{len(set(self.obj_id(t) for t in self.tracks))} objects')
        head = [f'Checked: {now()}', f'Video: {self.video}', f'Result folder: {os.path.abspath(self.dir)}',
                f'Original objects.csv: {len(orig)} rows, SHA1 {self.objects_sha1}',
                'Result: ' + ('ALL CHECKS PASSED' if ok else 'PROBLEMS FOUND, see below')]
        report = '\n'.join(head + lines) + '\n'
        open(os.path.join(self.dir, 'review_report.txt'), 'w', encoding='utf-8').write(report)
        return ok, report

    def render_overlay(self, out_path, verbose=True):
        """Check video: trusted = green, untrusted = red, unreviewed = grey; label = object ID (track ID)."""
        st, _, _ = self.row_status()
        col = {'trusted': (70, 200, 60), 'untrusted': (60, 60, 235), 'unreviewed': (175, 175, 175)}
        cap = cv2.VideoCapture(self.video)
        W, H = self.W, self.H
        use_ff = shutil.which('ffmpeg') is not None
        if use_ff:
            proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                                     '-s', f'{W}x{H + 56}', '-r', str(self.fps), '-i', '-', '-c:v', 'libx264',
                                     '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p',
                                     '-movflags', '+faststart', out_path], stdin=subprocess.PIPE)
            write = lambda im: proc.stdin.write(im.tobytes())
        else:
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (W, H + 56))
            write = vw.write
        f = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if self.crop:
                x, y, w, h = self.crop
                fr = fr[y:y + h, x:x + w].copy()
            for k in self.rows_by_frame.get(f, []):
                t = int(self.I[k])
                c = self.contour.get((f, t))
                color = col[st[k]]
                if c is not None and len(c) >= 2:
                    cv2.polylines(fr, [c.astype(np.int32)], True, color, 2, cv2.LINE_AA)
                else:
                    cv2.ellipse(fr, (int(self.cx[k]), int(self.cy[k])),
                                (max(1, int(self.major[k] / 2)), max(1, int(self.minor[k] / 2))),
                                -self.angle[k], 0, 360, color, 2, cv2.LINE_AA)
                o = self.obj_id(t)
                txt = f'#{t}' if o == t else f'#{o}({t})'
                tx, ty = int(self.cx[k] + 6), int(self.cy[k] - 6)
                cv2.putText(fr, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(fr, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            head = np.full((56, W, 3), (40, 34, 30), np.uint8)
            cv2.putText(head, f'frame {f}   t = {f / self.fps:7.3f} s   label: #object(track)', (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
            x = 12
            for s in STATUSES:
                cv2.rectangle(head, (x, 34), (x + 22, 46), col[s], -1)
                cv2.putText(head, s, (x + 30, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                x += 150
            write(np.vstack([head, fr]))
            f += 1
        cap.release()
        if use_ff:
            proc.stdin.close()
            proc.wait()
        else:
            vw.release()
        if verbose:
            print(f'  wrote {out_path}')

    # ---------------- data for the browser ----------------
    def meta(self):
        sugg = self.suggestions()
        return dict(video=os.path.basename(self.video), result=os.path.abspath(self.dir), fps=self.fps,
                    n_frames=self.n_frames, width=self.W, height=self.H, scale=self.scale,
                    has_contours=self.has_contours, reasons=REASONS,
                    tracks=[dict(t, segs=[dict(start=s['start'], end=s['end'], status=s['status'])
                                          for s in self.segs.get(t['id'], [])],
                                 obj=self.obj_id(t['id']), gsize=len(self.group_of(t['id'])))
                            for t in sorted(self.tracks.values(), key=lambda d: d['id'])],
                    suggestions=sugg, progress=self.progress(), can_undo=self.can_undo())

    def objects_at(self, f):
        out = []
        for k in self.rows_by_frame.get(f, []):
            i = int(self.I[k])
            c = self.contour.get((f, i))
            s = self.status_at(i, f)
            out.append(dict(id=i, g=self.obj_id(i), cx=round(float(self.cx[k]), 1), cy=round(float(self.cy[k]), 1),
                            a=float(self.area[k]), maj=float(self.major[k]), mi=float(self.minor[k]),
                            ang=float(self.angle[k]), w=int(self.wall[k]), s=s['status'] if s else 'unreviewed',
                            c=c.reshape(-1).tolist() if c is not None else None))
        return out

    def _path(self, t):
        ks = np.array(self.rows_by_track[t])
        return self.F[ks], self.cx[ks], self.cy[ks], self.area[ks], ks

    def track_detail(self, t):
        f, x, y, a, ks = self._path(t)
        flags = []
        if len(ks) > 2:
            d = np.hypot(np.diff(x), np.diff(y))
            gap = np.diff(f)
            med = float(np.median(d)) if len(d) else 0
            for k in np.nonzero((gap == 1) & (d > max(8.0, 5 * med)))[0]:
                flags.append(dict(frame=int(f[k + 1]), kind='jump', text=f'Position jump {d[k]:.0f} px'))
            r = np.abs(np.log(np.maximum(a[1:], 1) / np.maximum(a[:-1], 1)))
            for k in np.nonzero((gap == 1) & (r > 0.3) & (np.maximum(a[1:], a[:-1]) >= 50))[0]:
                flags.append(dict(frame=int(f[k + 1]), kind='area', text=f'Area change {a[k]:.0f} -> {a[k + 1]:.0f} px'))
            for k in np.nonzero(gap > 1)[0]:
                flags.append(dict(frame=int(f[k + 1]), kind='gap', text=f'Re-acquired after {int(gap[k]) - 1} missing frames'))
        w = self.wall[ks].astype(int)
        wall_spans, k = [], 0
        while k < len(w):
            if w[k]:
                j = k
                while j + 1 < len(w) and w[j + 1] and f[j + 1] == f[j] + 1:
                    j += 1
                wall_spans.append([int(f[k]), int(f[j])])
                k = j + 1
            else:
                k += 1
        group = []
        for m in sorted(self.group_of(t), key=lambda i: self.tracks[i]['start']):
            gf, gx, gy, _, _ = self._path(m)
            group.append(dict(id=m, start=self.tracks[m]['start'], end=self.tracks[m]['end'],
                              segments=self.segs.get(m, []), frames=gf.tolist(),
                              cx=np.round(gx, 1).tolist(), cy=np.round(gy, 1).tolist()))
        sugg = [s for s in self.suggestions() if s['a'] in self.group_of(t) or s['b'] in self.group_of(t)]
        ev = [dict(frame=e['frame'], type=e['type'], ids=e['ids'], result=e['result']) for e in self.ev_by_track[t]]
        return dict(id=t, obj=self.obj_id(t), summary=self.tracks[t], frames=f.tolist(),
                    cx=np.round(x, 1).tolist(), cy=np.round(y, 1).tolist(), area=a.tolist(),
                    flags=sorted(flags, key=lambda d: d['frame']), wall_spans=wall_spans, events=ev,
                    segments=self.segs.get(t, []), group=group, suggestions=sugg)


# ============================================================================
# local web server
# ============================================================================
def make_handler(store):
    static = os.path.join(HERE, 'static')

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype='application/json; charset=utf-8', cache=False):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode('utf-8')
            elif isinstance(body, str):
                body = body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'max-age=3600' if cache else 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _state(self, extra=None):
            d = dict(progress=store.progress(), can_undo=store.can_undo(), suggestions=store.suggestions(),
                     tracks={t: dict(segs=[dict(start=s['start'], end=s['end'], status=s['status'])
                                           for s in store.segs.get(t, [])],
                                     obj=store.obj_id(t), gsize=len(store.group_of(t))) for t in store.tracks})
            d.update(extra or {})
            return d

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path in ('/', '/index.html'):
                    return self._send(200, open(os.path.join(static, 'index.html'), 'rb').read(), 'text/html; charset=utf-8')
                if path.startswith('/static/'):
                    name = os.path.basename(path)
                    fp = os.path.join(static, name)
                    if not os.path.isfile(fp):
                        return self._send(404, {'error': 'not found'})
                    ctype = {'.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8'}.get(
                        os.path.splitext(name)[1], 'application/octet-stream')
                    return self._send(200, open(fp, 'rb').read(), ctype)
                if path == '/api/meta':
                    with store.lock:
                        return self._send(200, store.meta())
                if path.startswith('/api/frame/'):
                    f = int(path.rsplit('/', 1)[1])
                    if 0 <= f < len(store.frames):
                        return self._send(200, store.frames[f], 'image/jpeg', cache=True)
                    return self._send(404, {'error': 'frame out of range'})
                if path.startswith('/api/objects/'):
                    with store.lock:
                        return self._send(200, store.objects_at(int(path.rsplit('/', 1)[1])))
                if path.startswith('/api/track/'):
                    t = int(path.rsplit('/', 1)[1])
                    with store.lock:
                        if t not in store.tracks:
                            return self._send(404, {'error': f'There is no track #{t}'})
                        return self._send(200, store.track_detail(t))
                return self._send(404, {'error': 'not found'})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa
                return self._send(500, {'error': str(e)})

        def do_POST(self):
            path = urlparse(self.path).path
            n = int(self.headers.get('Content-Length') or 0)
            try:
                body = json.loads(self.rfile.read(n) or b'{}')
                with store.lock:
                    if path == '/api/set':
                        touched = store.set_range(body['track'], body['start'], body['end'], body['status'],
                                                  body.get('reason', ''), body.get('note', ''),
                                                  bool(body.get('whole_object')))
                        return self._send(200, self._state(dict(touched=touched)))
                    if path == '/api/link':
                        members = store.link(body['ids'])
                        return self._send(200, self._state(dict(touched=members)))
                    if path == '/api/unlink':
                        g = store.group_of(int(body['id']))
                        store.unlink(body['id'])
                        return self._send(200, self._state(dict(touched=g)))
                    if path == '/api/dismiss':
                        store.dismiss(body['a'], body['b'])
                        return self._send(200, self._state(dict(touched=[int(body['a']), int(body['b'])])))
                    if path == '/api/undo':
                        r = store.undo()
                        return self._send(200, self._state(dict(undone=r)))
                    if path == '/api/export':
                        store.export_tables()
                        ok, report = store.verify()
                        out = None
                        if body.get('video'):
                            out = os.path.join(store.dir, 'review_overlay.mp4')
                            store.render_overlay(out, verbose=False)
                        return self._send(200, dict(ok=ok, report=report, video=out))
                return self._send(404, {'error': 'not found'})
            except (ValueError, KeyError) as e:
                return self._send(400, {'error': str(e)})
            except Exception as e:  # noqa
                return self._send(500, {'error': str(e)})

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True, help='original video (not overlay.mp4)')
    ap.add_argument('--result', required=True, help='output folder of bubble_tracker.py')
    ap.add_argument('--port', type=int, default=8800)
    ap.add_argument('--max-width', type=int, default=1280, help='maximum display width in pixels')
    ap.add_argument('--no-browser', action='store_true', help='do not open the browser automatically')
    ap.add_argument('--export', action='store_true', help='only export the three classes and the check report')
    ap.add_argument('--render', action='store_true', help='with --export: also write review_overlay.mp4')
    a = ap.parse_args()
    if a.export:
        store = Store(a.video, a.result, load_frames=False)
        store.export_tables()
        ok, report = store.verify()
        print(report)
        if a.render:
            store.render_overlay(os.path.join(a.result, 'review_overlay.mp4'))
        sys.exit(0 if ok else 1)
    print('Loading ...')
    store = Store(a.video, a.result, max_width=a.max_width)
    store.export_tables()
    url = f'http://127.0.0.1:{a.port}/'
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(store))
    print(f'Review page: {url}\nChanges are saved to {os.path.abspath(a.result)}\nPress Ctrl+C to quit.')
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
