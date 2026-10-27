#!/usr/bin/env python3
"""
viewer.py - check the round-2 curves against the video (task 5)

Shows the video next to the motion, shape and whole-field curves. Click anywhere on a curve to jump to
that frame; the frame shows the outlines (green = trusted, orange = uncertain, red = false target) and the
selected object in yellow. Nothing is written; the result folder is only read.

    python analysis/viewer.py --video data/raw/expA.mp4 --result results/A_offline

The browser opens http://127.0.0.1:8810 . If <result>/analysis/ does not exist yet, analyze.py is run first.
"""
import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import csv
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import analyze  # noqa: E402

SERIES_DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']


def read_table(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def num(v):
    if v in ('', None):
        return None
    try:
        x = float(v)
    except ValueError:
        return v
    return int(x) if x.is_integer() and 'e' not in v and '.' not in v else round(x, 3)


class Store:
    def __init__(self, video, result_dir, max_width=1280, quality=85):
        self.video, self.dir = video, result_dir
        an = os.path.join(result_dir, 'analysis')
        if not os.path.exists(os.path.join(an, 'report.json')):
            print('No analysis yet; running analyze.py first ...')
            analyze.run(result_dir, video=video)
        self.report = json.load(open(os.path.join(an, 'report.json'), encoding='utf-8'))
        fps = self.report['fps'] if self.report['fps'] != self.report.get('fps_in_stats') else None
        self.D = analyze.Data(result_dir, dict(analyze.DEFAULTS, **self.report.get('parameters', {})), fps=fps,
                              video=video)
        self.fps = self.D.fps
        tab = lambda n: read_table(os.path.join(an, n))
        self.segments = tab('segments.csv')
        self.motion = tab('motion.csv')
        self.shape = tab('shape.csv')
        self.coll = tab('collective.csv')
        ev = os.path.join(an, 'field_events.csv')
        self.events = read_table(ev) if os.path.exists(ev) else []
        self.mains = [o['object_id'] for o in self.report.get('main_objects', [])]
        self.color = {o: SERIES_DARK[k] for k, o in enumerate(self.mains[:len(SERIES_DARK)])}
        self.seg_by_obj = defaultdict(list)
        for s in self.segments:
            self.seg_by_obj[int(s['object_id'])].append(s)
        self.mo_by_obj, self.sh_by_obj = defaultdict(list), defaultdict(list)
        for r in self.motion:
            self.mo_by_obj[int(r['object_id'])].append(r)
        for r in self.shape:
            self.sh_by_obj[int(r['object_id'])].append(r)
        self._load_frames(max_width, quality)

    def _load_frames(self, max_width, quality):
        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            sys.exit(f'Cannot open video {self.video}')
        crop = (self.D.stats.get('params') or {}).get('crop')
        self.frames = []
        self.scale = min(1.0, max_width / self.D.W)
        t0 = time.time()
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if crop:
                x, y, w, h = crop
                fr = fr[y:y + h, x:x + w]
            if self.scale < 1:
                fr = cv2.resize(fr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            self.frames.append(cv2.imencode('.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes())
            if len(self.frames) % 300 == 0:
                print(f'  loaded {len(self.frames)} frames ...', flush=True)
        cap.release()
        print(f'  {len(self.frames)} frames cached in {time.time() - t0:.1f} s')

    # ---------------- API data ----------------
    def meta(self):
        objs = []
        for o, ss in self.seg_by_obj.items():
            n = sum(int(s['n_frames']) for s in ss)
            objs.append(dict(object_id=o, ids=sorted({int(s['id']) for s in ss}), segments=len(ss),
                             trusted_s=round(n / self.fps, 2), area_max=max(float(s['area_max_px2']) for s in ss),
                             t0=min(float(s['t_start_s']) for s in ss), t1=max(float(s['t_end_s']) for s in ss),
                             main=o in self.mains, color=self.color.get(o)))
        objs.sort(key=lambda o: (not o['main'], self.mains.index(o['object_id']) if o['main'] else 0, o['t0']))
        rep = self.report
        return dict(video=os.path.basename(self.video), result=os.path.abspath(self.dir), fps=self.fps,
                    n_frames=len(self.frames) or self.D.n_frames, W=self.D.W, H=self.D.H, scale=self.scale,
                    objects=objs, notes=rep.get('data_quality', []), checks=rep.get('key_moment_checks', []),
                    all_checks_passed=rep.get('all_checks_passed'), um_per_px=rep.get('um_per_px'),
                    params=rep.get('parameters', {}))

    def regions(self, f):
        D = self.D
        out = []
        for k in D.rows_by_frame.get(f, []):
            i = int(D.I[k])
            c = D.contour.get((f, i))
            out.append(dict(id=i, obj=int(D.object_id[k]), role=D.role[k], status=D.status[k],
                            x=round(float(D.cx[k]), 1), y=round(float(D.cy[k]), 1), a=float(D.area[k]),
                            pts=c.reshape(-1).tolist() if c is not None else None))
        return out

    def object(self, oid):
        keep_m = ['segment', 'frame', 'id', 'x_px', 'y_px', 'speed_px_s', 'at_image_edge']
        keep_s = ['area_px2', 'aspect_ratio', 'circularity', 'solidity', 'orientation_deg', 'shape_quality',
                  'orientation_ok', 'edge_on_wall']
        rows = []
        for m, s in zip(self.mo_by_obj.get(oid, []), self.sh_by_obj.get(oid, [])):
            r = {k: num(m[k]) for k in keep_m}
            r.update({k: num(s[k]) if k != 'shape_quality' else s[k] for k in keep_s})
            rows.append(r)
        segs = [dict(segment=int(s['segment']), id=int(s['id']), f0=int(s['start_frame']), f1=int(s['end_frame']),
                     start=s['start_reason'], end=s['end_reason']) for s in self.seg_by_obj.get(oid, [])]
        return dict(object_id=oid, color=self.color.get(oid), rows=rows, segments=segs)

    def field(self):
        keys = ['frame', 'n_regions', 'n_confirmed', 'total_area_px2', 'spread_px', 'cluster_area_share',
                'largest_share', 'nn_gap_median_px', 'uncertain_area_share', 'field_quality']
        rows = [{k: (num(r[k]) if k != 'field_quality' else r[k]) for k in keys} for r in self.coll]
        ev = [dict(frame=int(e['frame']), type=e['type'], ids=e['ids'], counted=e['counted_in_field'] in ('1', 'True'),
                   size=num(e.get('region_max_area_px2') or e.get('area_px2') or '0') or 0) for e in self.events]
        return dict(rows=rows, events=ev)


def make_handler(store):
    static_dir = os.path.join(HERE, 'static')
    cache = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype='application/json', code=200):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, separators=(',', ':')).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store' if ctype == 'application/json' else 'max-age=3600')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path in ('/', '/index.html'):
                    return self.send(open(os.path.join(static_dir, 'viewer.html'), 'rb').read(), 'text/html; charset=utf-8')
                if path.startswith('/static/'):
                    name = os.path.basename(path)
                    fp = os.path.join(static_dir, name)
                    if not os.path.exists(fp):
                        return self.send({'error': 'not found'}, code=404)
                    ctype = {'.js': 'text/javascript', '.css': 'text/css'}.get(os.path.splitext(name)[1], 'application/octet-stream')
                    return self.send(open(fp, 'rb').read(), ctype + '; charset=utf-8')
                if path == '/api/meta':
                    return self.send(store.meta())
                if path.startswith('/api/frame/'):
                    f = int(path.rsplit('/', 1)[1])
                    if not 0 <= f < len(store.frames):
                        return self.send({'error': 'frame out of range'}, code=404)
                    return self.send(store.frames[f], 'image/jpeg')
                if path.startswith('/api/regions/'):
                    return self.send(store.regions(int(path.rsplit('/', 1)[1])))
                if path.startswith('/api/object/'):
                    oid = int(path.rsplit('/', 1)[1])
                    if oid not in cache:
                        cache[oid] = json.dumps(store.object(oid), separators=(',', ':')).encode()
                    return self.send(cache[oid])
                if path == '/api/field':
                    if 'field' not in cache:
                        cache['field'] = json.dumps(store.field(), separators=(',', ':')).encode()
                    return self.send(cache['field'])
                return self.send({'error': 'not found'}, code=404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # report to the page instead of dropping the connection
                return self.send({'error': str(e)}, code=500)

    return H


def main():
    ap = argparse.ArgumentParser(description='Check the round-2 curves against the video')
    ap.add_argument('--video', required=True, help='original video (not overlay.mp4)')
    ap.add_argument('--result', required=True, help='result folder that contains analysis/')
    ap.add_argument('--port', type=int, default=8810)
    ap.add_argument('--max-width', type=int, default=1280)
    ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()
    print('Loading ...')
    store = Store(a.video, a.result, max_width=a.max_width)
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(store))
    url = f'http://127.0.0.1:{a.port}'
    print(f'Open {url}   (Ctrl+C to stop)')
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
