#!/usr/bin/env python3
"""
pzt_analysis.py - round 3: link the PZT (ultrasound) actions to the microbubble changes

Reads (never modifies)
    <result>/analysis/...        round-2 tables (motion, shape, collective, field_events, onsets, state);
                                 analyze.py is run first if they are missing
    <pzt folder>/actions.csv     PZT commands with their software times and video-frame associations
    <pzt folder>/video_frames.csv  receive time of every saved video frame (same Windows clock as the commands)
    <pzt folder>/afg_experiment/plan.json, afg_link_summary.json, link_session.json   settings and readbacks
Writes <result>/analysis/pzt_analysis/
    commands.csv       task 1: every channel command (incl. the repeated safety stop) with times and frames
    actions.csv        task 1: one row per change of the 4-PZT state, with the full state of all four PZTs after it
    aligned_frames.csv task 2: every video frame with the PZT state at that frame and how certain that is
    responses.csv      task 3: for every action, what each trusted object and the whole field did before / after
    model_samples.csv  task 5: one sample per action and object (and the field): state before, action, state after
    model_steps.csv    task 5: fixed 0.1 s steps: state now, PZT state, state 0.1 s later
    figures/           timeline, responses aligned to each action, timing precision
    checks/            video frames before / after every action
    report.json        time-alignment precision, verification, per-action summary, data-quality notes

Usage
    python analysis/pzt_analysis.py --result results/A_offline --pzt data/pzt/expA --video data/raw/expA.mp4

How the frames are aligned (details in README):
    The commands and the frame receive times share one Windows clock. A frame is received some time after it
    was exposed. The camera's own frame clock is regular (video_s), so the frame that arrived with the
    smallest delay fixes the earliest possible exposure time of every frame. From that, each command gets
    'k_lo', the earliest video frame (fractional) at which it can show. The unknown smallest camera delay
    can only move the true frame later. A short 'transition' window after k_lo covers this unknown delay,
    the AFG output and the response; frames in it are flagged.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import analyze  # noqa: E402

DEFAULTS = dict(
    active_vpp=0.1,              # a channel counts as "on" (driving) at or above this amplitude
    transition_s=0.1,            # frames within 0.1 s after the earliest possible switch frame are 'transition'
    pre_s=1.0,                   # 'before' window: the last 1 s before the switch
    early_s=0.5,                 # 'early' window: the first 0.5 s after the transition
    late_s=0.5,                  # 'late' window: the last 0.5 s before the next action (state reached)
    rest_speed_px_s=10.0,        # below this a region is at rest (same as round 2)
    moving_speed_px_s=30.0,      # above this a region is clearly moving (same as round 2)
    departure_min_px=1.0,        # first visible change: the centroid leaves its rest position by > 1 px ...
    departure_jitter_k=4.0,      # ... and by > 4 x its own jitter at rest ...
    departure_hold_frames=2,     # ... and stays away for the next 2 frames as well
    departure_search_s=1.0,      # look for it up to 1 s after the switch
    response_window_s=0.5,       # "responded to the action" = first change within 0.5 s of the switch ...
    response_min_speed_px_s=15.0,  # ... and in the 0.1 s after it the region moves at least 15 px/s ...
    response_speed_ratio=3.0,    # ... and at least 3 x as fast as at rest (slow drift is not a response)
    area_change=0.15,            # deformed: visible area changed by 15 % or more ...
    aspect_change=0.2,           # ... or aspect ratio by 0.2 or more (shape not 'poor' in both windows)
    step_dt_s=0.1,               # model_steps.csv: state now -> state 0.1 s later
)
PZTS = ['PZT1', 'PZT2', 'PZT3', 'PZT4']


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def fnum(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_json(path, default=None):
    return json.load(open(path, encoding='utf-8-sig')) if os.path.exists(path) else default


def json_safe(o):
    """numpy types to Python, NaN / inf to null, so report.json is strict JSON."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def median(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.median(v)) if len(v) else np.nan


# ---------------------------------------------------------------------------
# PZT records
# ---------------------------------------------------------------------------
class PztRecord:
    def __init__(self, folder):
        self.dir = folder
        p = lambda *n: os.path.join(folder, *n)
        for n in ('actions.csv', 'video_frames.csv'):
            if not os.path.exists(p(n)):
                sys.exit(f'Cannot find {p(n)}')
        self.raw_actions = read_csv(p('actions.csv'))
        vf = read_csv(p('video_frames.csv'))
        vf.sort(key=lambda r: int(r['frame_index']))
        self.frame_index = np.array([int(r['frame_index']) for r in vf])
        self.video_s = np.array([float(r['video_s']) for r in vf])
        self.recv = np.array([float(r['received_windows_s']) for r in vf])
        self.experiment_id = vf[0].get('experiment_id', '') if vf else ''
        plan = load_json(p('afg_experiment', 'plan.json'), {}) or {}
        self.plan = plan.get('plan', {})
        self.afg_config = plan.get('afg_config', {})
        self.summary = load_json(p('afg_experiment', 'summary.json'), {}) or {}
        self.link = load_json(p('afg_link_summary.json'), {}) or {}
        self.link_report = load_json(p('link_report.json'), {}) or {}
        self.session = load_json(p('link_session.json'), {}) or {}
        self.origin = load_json(p('afg_origin.json'), {}) or {}
        self.manifest = load_json(os.path.join(os.path.dirname(os.path.abspath(folder)), 'manifest.json'), None)
        dt = np.diff(self.video_s)
        self.fps = 1.0 / float(np.median(dt)) if len(dt) else 60.0
        # earliest possible exposure clock: E_k >= recv_k - excess_k ; offset = min(recv - video_s)
        self.excess = self.recv - self.video_s
        self.offset = float(self.excess.min()) if len(self.excess) else 0.0
        self.min_delay_frame = int(np.argmin(self.excess)) if len(self.excess) else 0

    def k_lo(self, t_windows):
        """Earliest video frame (fractional) at which something happening at t_windows can be exposed."""
        return (t_windows - self.offset) * self.fps

    def frame_received_after(self, t_windows):
        i = int(np.searchsorted(self.recv, t_windows, side='right'))
        return i if i < len(self.recv) else None

    def manifest_entry(self, name):
        if not self.manifest:
            return None
        for e in self.manifest.get('experiments', []):
            if e.get('name') == name or e.get('experiment_id') == self.experiment_id:
                return e
        return None


def initial_channels(rec):
    """Full channel settings at the start of the video: plan / precheck settings, outputs on at idle level."""
    pre = ((rec.summary.get('precheck') or {}).get('channels') or {})
    settings = rec.plan.get('settings', {})
    idle = float(rec.session.get('idle_vpp') or rec.summary.get('idle_amplitude_vpp') or 0.001)
    out = {}
    for ch in PZTS:
        s, q = settings.get(ch, {}), pre.get(ch, {})
        mapping = (rec.afg_config.get('channels') or {}).get(ch, {})
        out[ch] = dict(output=True, vpp=idle, freq_hz=float(q.get('frequency_hz') or s.get('frequency_hz') or np.nan),
                       phase_deg=float(q.get('phase_deg', s.get('phase_deg', 0.0)) or 0.0),
                       mode=q.get('frequency_mode') or s.get('mode') or '', waveform=q.get('waveform', ''),
                       device=f"{mapping.get('device', '')}{mapping.get('channel', '')}")
    return out


def active_set(state, active_vpp):
    return [ch for ch in PZTS if state[ch]['output'] and state[ch]['vpp'] >= active_vpp]


def describe(before, after):
    a, b = set(before), set(after)
    name = lambda s: '+'.join(ch for ch in PZTS if ch in s)
    if not a and b:
        return f'{name(b)} on'
    if a and not b:
        return f'{name(a)} off (all idle)'
    if a and b and a != b:
        return f'{name(a)} → {name(b)}'
    return 'no change in driving'


# ---------------------------------------------------------------------------
# Task 1: commands and actions
# ---------------------------------------------------------------------------
def build_actions(rec, P):
    checks = {c.get('step_at_s'): c for c in (rec.link.get('checks') or [])}
    origin = float(rec.origin.get('origin_windows_s') or (rec.recv[0] if len(rec.recv) else 0.0))
    commands = []
    for n, r in enumerate(rec.raw_actions, start=1):
        kind = (r.get('kind') or '').strip()
        C, W, Q = fnum(r.get('command_windows_s')), fnum(r.get('write_returned_windows_s')), fnum(r.get('confirmation_windows_s'))
        amp = fnum(r.get('amplitude_vpp'))
        rb = r.get('readback', '')
        if kind == 'amplitude':
            rb_ok = np.isfinite(fnum(rb)) and abs(fnum(rb) - amp) <= max(1e-6, 0.01 * abs(amp))
        else:
            rb_ok = str(rb).strip() in ('0', '0.0', 'OFF', 'off', 'False', 'false')
        commands.append(dict(
            command_no=n, action_id=r.get('action_id', ''), kind=kind, channel=r.get('channel', ''),
            state=r.get('state', ''), amplitude_vpp=amp, planned_s=fnum(r.get('planned_s')),
            command_windows_s=C, write_returned_windows_s=W, confirmation_windows_s=Q,
            command_since_first_frame_s=C - rec.recv[0], readback=rb, readback_matches=bool(rb_ok), status=r.get('status', ''),
            frame_received_before=int(fnum(r.get('command_video_first_frame'), -1)),
            frame_received_after=int(fnum(r.get('command_video_last_frame'), -1)),
            receive_bracket_ms=fnum(r.get('command_video_receive_bracket_ms')),
            k_lo=rec.k_lo(C), earliest_video_s=rec.k_lo(C) / rec.fps,
            confirmation_frame=int(fnum(r.get('confirmation_video_last_frame'), -1)),
            confirmation_k_lo=rec.k_lo(Q) if np.isfinite(Q) else np.nan,
            confirmation_after_command_s=Q - C if np.isfinite(Q) else np.nan))
    # group into actions: commands with the same planned time change the state together; the repeated
    # safety stop (stop_repeat) changes nothing and is attached to the stop action
    groups, order = defaultdict(list), []
    for c in commands:
        key = 'stop' if c['kind'] in ('stop', 'stop_repeat') else c['planned_s']
        if key not in groups:
            order.append(key)
        groups[key].append(c)
    state = initial_channels(rec)
    trans = int(round(P['transition_s'] * rec.fps))
    c0 = checks.get(0) or checks.get(0.0)
    actions = [dict(action_no=0, label='start of video: all PZTs idle', planned_s=0.0, change='initial state',
                    n_commands=0, command_ids='', first_command_windows_s=np.nan, last_command_windows_s=np.nan,
                    command_spread_ms=np.nan, frame_received_before=np.nan, frame_received_after=np.nan,
                    k_lo=0.0, first_frame=0, stable_from_frame=0, switch_video_s=0.0,
                    readback_check='passed' if c0 and c0.get('passed') else ('failed' if c0 else 'not available'),
                    readback_frame=int(math.ceil(rec.k_lo(origin + c0['finished_s']))) if c0 else np.nan,
                    all_readbacks_match=True, active_before='', active_after='none',
                    state={ch: dict(v) for ch, v in state.items()})]
    for n, key in enumerate(order, start=1):
        cs = groups[key]
        main = [c for c in cs if c['kind'] != 'stop_repeat']
        before = active_set(state, P['active_vpp'])
        for c in main:
            ch = state.get(c['channel'])
            if ch is None:
                continue
            if c['kind'] == 'amplitude':
                ch['vpp'] = c['amplitude_vpp']
                ch['output'] = True
            elif c['kind'] == 'stop':
                ch['output'] = False
        after = active_set(state, P['active_vpp'])
        kind = 'stop' if key == 'stop' else 'amplitude'
        label = 'stop: all outputs off' if kind == 'stop' else describe(before, after)
        if kind == 'stop' and before:
            label = f"stop: {'+'.join(before)} off, all outputs off"
        Cs = [c['command_windows_s'] for c in main]
        klo = [c['k_lo'] for c in main]
        chk = checks.get(main[0]['planned_s']) if kind != 'stop' else None
        if kind == 'stop':
            readback = 'passed' if all(c['readback_matches'] for c in main) else 'failed'
            rb_frame = max(c['confirmation_frame'] for c in main)
        elif chk is not None:
            readback = 'passed' if chk.get('passed') else 'failed'
            rb_frame = int(math.ceil(rec.k_lo(origin + chk['finished_s'])))
        else:
            readback = 'not available'
            rb_frame = max(c['confirmation_frame'] for c in main)
        first = int(math.ceil(min(klo)))
        actions.append(dict(
            action_no=n, label=label, planned_s=main[0]['planned_s'], change=kind,
            n_commands=len(cs), command_ids=' '.join(c['action_id'] for c in cs),
            first_command_windows_s=min(Cs), last_command_windows_s=max(Cs),
            command_spread_ms=(max(Cs) - min(Cs)) * 1e3,
            frame_received_before=min(c['frame_received_before'] for c in main),
            frame_received_after=max(c['frame_received_after'] for c in main),
            k_lo=min(klo), first_frame=first, stable_from_frame=int(math.ceil(max(klo))) + trans,
            switch_video_s=min(klo) / rec.fps,
            readback_check=readback, readback_frame=rb_frame,
            all_readbacks_match=all(c['readback_matches'] for c in main),
            active_before='+'.join(before) or 'none', active_after='+'.join(after) or 'none',
            state={ch: dict(v) for ch, v in state.items()}))
        for c in cs:
            c['action_no'] = n
    return commands, actions


def action_rows(actions, n_frames, fps):
    rows = []
    for k, a in enumerate(actions):
        nxt = actions[k + 1]['first_frame'] if k + 1 < len(actions) else n_frames
        r = {x: a[x] for x in ('action_no', 'label', 'change', 'planned_s', 'switch_video_s', 'k_lo', 'first_frame',
                               'stable_from_frame', 'frame_received_before', 'frame_received_after', 'n_commands',
                               'command_ids', 'first_command_windows_s', 'last_command_windows_s',
                               'command_spread_ms', 'readback_check', 'readback_frame', 'all_readbacks_match',
                               'active_before', 'active_after')}
        r.update(next_action_frame=nxt, duration_s=(nxt - a['first_frame']) / fps)
        for ch in PZTS:
            s = a['state'][ch]
            r.update({f'{ch}_output': 'on' if s['output'] else 'off', f'{ch}_vpp': s['vpp'] if s['output'] else 0.0,
                      f'{ch}_freq_hz': s['freq_hz'], f'{ch}_phase_deg': s['phase_deg'], f'{ch}_mode': s['mode'],
                      f'{ch}_device': s['device']})
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Task 2: every frame with its PZT state
# ---------------------------------------------------------------------------
def align_frames(rec, actions, n_frames, P):
    out = []
    firsts = [a['first_frame'] for a in actions]
    for f in range(n_frames):
        k = max(0, int(np.searchsorted(firsts, f, side='right')) - 1)
        a = actions[k]
        trans = k > 0 and a['first_frame'] <= f < a['stable_from_frame']
        rb = a['readback_frame']
        r = dict(frame=f, video_s=rec.video_s[f] if f < len(rec.video_s) else f / rec.fps,
                 received_windows_s=rec.recv[f] if f < len(rec.recv) else np.nan,
                 receive_delay_excess_ms=(rec.excess[f] - rec.offset) * 1e3 if f < len(rec.excess) else np.nan,
                 action_no=a['action_no'], action=a['label'],
                 t_since_action_s=(f - a['k_lo']) / rec.fps if k > 0 else f / rec.fps,
                 time_quality='transition' if trans else 'stable', in_transition=bool(trans),
                 readback_confirmed=bool(np.isfinite(rb) and f >= rb and a['readback_check'] == 'passed'),
                 active_pzts=a['active_after'])
        for ch in PZTS:
            s = a['state'][ch]
            r[f'{ch}_vpp'] = s['vpp'] if s['output'] else 0.0
            r[f'{ch}_output'] = 'on' if s['output'] else 'off'
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Task 3: responses
# ---------------------------------------------------------------------------
class Tables:
    """Round-2 tables, indexed for fast window queries."""

    def __init__(self, an):
        rd = lambda n: read_csv(os.path.join(an, n))
        self.motion = rd('motion.csv')
        self.shape = rd('shape.csv')
        self.coll = rd('collective.csv')
        self.events = rd('field_events.csv') if os.path.exists(os.path.join(an, 'field_events.csv')) else []
        self.onsets = rd('onsets.csv') if os.path.exists(os.path.join(an, 'onsets.csv')) else []
        self.report = load_json(os.path.join(an, 'report.json'), {})
        obj = defaultdict(list)
        for m, s in zip(self.motion, self.shape):
            obj[int(m['object_id'])].append((m, s))
        self.obj = {}
        for o, rows in obj.items():
            rows.sort(key=lambda ms: int(ms[0]['frame']))
            g = lambda k, src=0, f=fnum: np.array([f(ms[src][k]) for ms in rows])
            self.obj[o] = dict(
                frame=g('frame', 0, int), segment=g('segment', 0, int), id=g('id', 0, int),
                x=g('x_px'), y=g('y_px'), speed=g('speed_px_s'), edge=g('at_image_edge', 0, lambda v: int(fnum(v, 0))),
                area=g('area_px2', 1), aspect=g('aspect_ratio', 1),
                quality=np.array([ms[1]['shape_quality'] for ms in rows]))
        self.cframe = np.array([int(r['frame']) for r in self.coll])
        self.ckeys = ['n_regions', 'total_area_px2', 'spread_px', 'cluster_area_share', 'largest_share',
                      'nn_gap_median_px', 'centroid_x_px', 'centroid_y_px', 'uncertain_area_share']
        self.c = {k: np.array([fnum(r.get(k)) for r in self.coll]) for k in self.ckeys}
        self.cquality = np.array([r.get('field_quality', '') for r in self.coll])


def windows(actions, k, n_frames, fps, P):
    a = actions[k]
    nxt = actions[k + 1]['first_frame'] if k + 1 < len(actions) else n_frames
    prev_stable = actions[k - 1]['stable_from_frame'] if k > 0 else 0
    f_sw, f_st = a['first_frame'], max(a['stable_from_frame'], a['first_frame'])
    pre = (max(prev_stable, f_sw - int(round(P['pre_s'] * fps))), f_sw - 1)
    early = (f_st, min(nxt, f_st + int(round(P['early_s'] * fps))) - 1)
    during = (f_st, nxt - 1)
    late = (max(f_st, nxt - int(round(P['late_s'] * fps))), nxt - 1)
    return dict(pre=pre, early=early, during=during, late=late, switch=f_sw, stable=f_st, next=nxt)


def in_win(frames, w):
    return (frames >= w[0]) & (frames <= w[1])


def departure(o, w, a, fps, P):
    """First frame at which a region at rest visibly leaves its rest position (same trusted segment)."""
    f_sw = w['switch']
    pre = in_win(o['frame'], (f_sw - int(round(0.5 * fps)), f_sw - 1))
    if pre.sum() < 5:
        return None
    seg = o['segment'][np.nonzero(pre)[0][-1]]
    restm = pre & (o['segment'] == seg)
    if restm.sum() < 5 or median(o['speed'][restm]) >= P['rest_speed_px_s']:
        return None
    rx, ry = np.median(o['x'][restm]), np.median(o['y'][restm])
    d_rest = np.hypot(o['x'][restm] - rx, o['y'][restm] - ry)
    jitter = 1.4826 * float(np.median(np.abs(d_rest - np.median(d_rest))))
    thr = max(P['departure_min_px'], P['departure_jitter_k'] * jitter, float(np.percentile(d_rest, 100)) * 1.5)
    look = (o['segment'] == seg) & in_win(o['frame'], (f_sw - 6, f_sw + int(round(P['departure_search_s'] * fps))))
    idx = np.nonzero(look)[0]
    d = np.hypot(o['x'][idx] - rx, o['y'][idx] - ry)
    h = P['departure_hold_frames']
    rest_speed = median(o['speed'][restm])
    need = max(P['response_min_speed_px_s'], P['response_speed_ratio'] * rest_speed)
    for j in range(len(idx)):
        if d[j] > thr and j + h < len(idx) and np.all(d[j:j + h + 1] > thr) \
                and o['frame'][idx[j + h]] - o['frame'][idx[j]] == h:
            f = int(o['frame'][idx[j]])
            e = idx[(o['frame'][idx] >= f) & (o['frame'][idx] <= f + int(round(0.3 * fps)))]
            e1 = idx[(o['frame'][idx] >= f) & (o['frame'][idx] < f + max(3, int(round(0.1 * fps))))]
            if median(o['speed'][e1]) < need:     # slow drift past the threshold: keep looking
                continue
            dx, dy = o['x'][e[-1]] - rx, o['y'][e[-1]] - ry
            return dict(frame=f, threshold_px=thr, jitter_px=jitter,
                        delay_min_s=max(0.0, (f - 1 - a['k_lo']) / fps), delay_max_s=(f - a['k_lo']) / fps,
                        direction_deg=float(np.degrees(np.arctan2(-dy, dx))), first_0_3s_px=float(np.hypot(dx, dy)))
    return None


def placebo_rate(T, actions, n_frames, fps, P):
    """How often a region at rest 'responds' at a moment when nothing was switched.

    The same detection is run at made-up switch times inside the actions (1.25 s and 2.0 s after a switch,
    away from the next one). Returns responders / regions at rest, and the counts.
    """
    at_rest = responded = 0
    for k in range(1, len(actions)):
        a = actions[k]
        nxt = actions[k + 1]['first_frame'] if k + 1 < len(actions) else n_frames
        for off in (1.25, 2.0):
            p = a['stable_from_frame'] + int(round(off * fps))
            if p + int(round((P['response_window_s'] + 0.4) * fps)) >= nxt:
                continue
            fake = dict(k_lo=float(p), first_frame=p)
            w = dict(switch=p)
            for o in T.obj.values():
                pre = in_win(o['frame'], (p - int(round(0.5 * fps)), p - 1))
                if pre.sum() < 5 or median(o['speed'][pre]) >= P['rest_speed_px_s']:
                    continue
                at_rest += 1
                dep = departure(o, w, fake, fps, P)
                if dep and dep['frame'] >= p and dep['delay_max_s'] <= P['response_window_s']:
                    responded += 1
    return (responded / at_rest if at_rest else np.nan), responded, at_rest


def object_response(T, o_id, actions, k, w, fps, P, events_by_obj):
    o = T.obj[o_id]
    a = actions[k]
    m = {n: in_win(o['frame'], w[n]) for n in ('pre', 'early', 'during', 'late')}
    nfr = {n: max(1, w[n][1] - w[n][0] + 1) for n in m}
    if not (m['pre'].any() or m['during'].any()):
        return None
    sp = {n: median(o['speed'][m[n]]) for n in m}
    sp_max = float(np.nanmax(o['speed'][m['during']])) if m['during'].any() and np.isfinite(o['speed'][m['during']]).any() else np.nan
    # displacement accumulated inside trusted segments (jumps at connect / separate cuts left out)
    dx = dy = 0.0
    span = in_win(o['frame'], (w['switch'], w['next'] - 1))
    for sg in np.unique(o['segment'][span]):
        ii = np.nonzero(span & (o['segment'] == sg))[0]
        dx += o['x'][ii[-1]] - o['x'][ii[0]]
        dy += o['y'][ii[-1]] - o['y'][ii[0]]
    okq = o['quality'] != 'poor'
    a_pre, a_late = median(o['area'][m['pre'] & okq]), median(o['area'][m['late'] & okq])
    r_pre, r_late = median(o['aspect'][m['pre'] & okq]), median(o['aspect'][m['late'] & okq])
    wall = lambda mm: float(np.mean(o['quality'][mm] == 'wall contact')) if mm.any() else np.nan
    dep = departure(o, w, a, fps, P)
    at_rest = np.isfinite(sp['pre']) and sp['pre'] < P['rest_speed_px_s']
    responded = bool(dep and dep['delay_max_s'] <= P['response_window_s'] and dep['frame'] >= w['switch'])
    if not m['pre'].any():
        motion = 'appeared during action'
    elif not m['during'].any():
        motion = 'not visible during action'
    elif at_rest and (responded or (np.isfinite(sp['early']) and sp['early'] >= P['moving_speed_px_s'])):
        motion = 'started moving'
    elif at_rest:
        motion = 'stayed at rest' if not (np.isfinite(sp['during']) and sp['during'] >= P['rest_speed_px_s']) else 'moved later'
    elif np.isfinite(sp['late']) and sp['late'] < P['rest_speed_px_s']:
        motion = 'stopped'
    else:
        motion = 'kept moving'
    deformed = None
    if np.isfinite(a_pre) and np.isfinite(a_late):
        deformed = bool(abs(a_late / a_pre - 1) >= P['area_change'] or
                        (np.isfinite(r_pre) and np.isfinite(r_late) and abs(r_late - r_pre) >= P['aspect_change']))
    ev = [e for e in events_by_obj.get(o_id, []) if w['switch'] <= e['frame'] <= w['next'] - 1]
    flags = []
    if m['pre'].any() and m['pre'].sum() < 0.8 * nfr['pre']:
        flags.append('partial before')
    if m['during'].any() and m['during'].sum() < 0.8 * nfr['during']:
        flags.append('partial during')
    if (wall(m['pre']) or 0) > 0.5 or (wall(m['late']) or 0) > 0.5:
        flags.append('wall contact')
    if o['edge'][span].any():
        flags.append('image edge')
    if any(e['type'] in ('connect', 'separate') for e in ev):
        flags.append('connect/separate')
    if any(e['type'] in ('enter view', 'leave view') for e in ev):
        flags.append('enter/leave view')
    if dep and dep['frame'] < w['switch']:
        flags.append('moved before switch')
    ids = sorted({int(i) for i in o['id'][m['pre'] | m['during']]})
    return dict(
        action_no=a['action_no'], action=a['label'], entity='object', object_id=o_id, ids=' '.join(map(str, ids)),
        coverage_before=float(m['pre'].sum() / nfr['pre']), coverage_during=float(m['during'].sum() / nfr['during']),
        speed_before_px_s=sp['pre'], speed_early_px_s=sp['early'], speed_during_px_s=sp['during'],
        speed_late_px_s=sp['late'], speed_max_px_s=sp_max, at_rest_before=bool(at_rest), motion=motion,
        responded_within_window=responded,
        first_change_frame=dep['frame'] if dep else None,
        delay_min_s=dep['delay_min_s'] if dep else None, delay_max_s=dep['delay_max_s'] if dep else None,
        first_direction_deg=dep['direction_deg'] if dep else None,
        disp_x_px=dx, disp_y_px=dy, disp_px=float(np.hypot(dx, dy)),
        disp_direction_deg=float(np.degrees(np.arctan2(-dy, dx))) if np.hypot(dx, dy) > 0 else None,
        area_before_px2=a_pre, area_after_px2=a_late,
        area_ratio=a_late / a_pre if np.isfinite(a_pre) and np.isfinite(a_late) and a_pre > 0 else None,
        aspect_before=r_pre, aspect_after=r_late, deformed=deformed,
        wall_share_before=wall(m['pre']), wall_share_after=wall(m['late']),
        connect_events=sum(e['type'] == 'connect' for e in ev), separate_events=sum(e['type'] == 'separate' for e in ev),
        flags=', '.join(flags), quality='good' if not flags else 'check')


def field_response(T, actions, k, w, fps, P, obj_rows, onset_rate):
    a = actions[k]
    cm = {n: in_win(T.cframe, w[n]) for n in ('pre', 'late', 'during')}
    r = dict(action_no=a['action_no'], action=a['label'], entity='field', object_id=None)
    for key, short in (('n_regions', 'n_regions'), ('total_area_px2', 'total_area_px2'), ('spread_px', 'spread_px'),
                       ('cluster_area_share', 'cluster_share'), ('largest_share', 'largest_share'),
                       ('nn_gap_median_px', 'nn_gap_px')):
        b, l_ = median(T.c[key][cm['pre']]), median(T.c[key][cm['late']])
        r[f'{short}_before'], r[f'{short}_after'], r[f'{short}_change'] = b, l_, l_ - b if np.isfinite(b) and np.isfinite(l_) else None
    cx = (median(T.c['centroid_x_px'][cm['late']]) - median(T.c['centroid_x_px'][cm['pre']]))
    cy = (median(T.c['centroid_y_px'][cm['late']]) - median(T.c['centroid_y_px'][cm['pre']]))
    r.update(centroid_shift_px=float(np.hypot(cx, cy)) if np.isfinite(cx) and np.isfinite(cy) else None)
    ev = [e for e in T.events if w['switch'] <= int(e['frame']) <= w['next'] - 1 and e.get('counted_in_field') in ('1', 'True')]
    for t in ('connect', 'separate', 'enter view', 'leave view'):
        r[f"{t.replace(' ', '_')}_events"] = sum(e['type'] == t for e in ev)
    resp = [o for o in obj_rows if o['action_no'] == a['action_no']]
    r['objects_tracked'] = len(resp)
    r['objects_at_rest_before'] = sum(o['at_rest_before'] for o in resp)
    r['objects_responded'] = sum(o['responded_within_window'] for o in resp)
    r['objects_responded_expected_by_chance'] = (P['_placebo_rate'] * r['objects_at_rest_before']
                                                 if np.isfinite(P.get('_placebo_rate', np.nan)) else None)
    dl = [o['delay_max_s'] for o in resp if o['responded_within_window']]
    r['first_change_delay_max_s'] = min(dl) if dl else None
    r['median_change_delay_max_s'] = float(np.median(dl)) if dl else None
    win = int(round(P['response_window_s'] * fps))
    starts = [o for o in T.onsets if w['switch'] <= int(o['frame']) < w['switch'] + win]
    r['motion_starts_in_window'] = len(starts)
    r['motion_starts_expected_by_chance'] = onset_rate * P['response_window_s']
    r['uncertain_area_share_before'] = median(T.c['uncertain_area_share'][cm['pre']])
    r['field_quality'] = 'uncertain' if (np.any(T.cquality[cm['pre'] | cm['late']] == 'uncertain')) else 'good'
    r['quality'] = r['field_quality']
    return r


def onset_background_rate(T, actions, fps, n_frames, P):
    """Round-2 motion starts per second away from every switch (how often regions start moving anyway)."""
    excl = np.zeros(n_frames, bool)
    for a in actions[1:]:
        excl[max(0, a['first_frame'] - int(0.2 * fps)):min(n_frames, a['first_frame'] + int(round((P['response_window_s'] + 0.2) * fps)))] = True
    n = sum(1 for o in T.onsets if not excl[min(n_frames - 1, int(o['frame']))])
    dur = (~excl).sum() / fps
    return n / dur if dur > 0 else np.nan


# ---------------------------------------------------------------------------
# Task 5: model data
# ---------------------------------------------------------------------------
def pzt_vec(a):
    return {ch: (a['state'][ch]['vpp'] if a['state'][ch]['output'] else 0.0) for ch in PZTS}


def model_samples(T, actions, resp, fps, P, exp):
    out = []
    for r in resp:
        k = r['action_no']
        if k == 0:
            continue
        before, after = pzt_vec(actions[k - 1]), pzt_vec(actions[k])
        s = dict(experiment=exp, action_no=k, action=r['action'], entity=r['entity'], object_id=r['object_id'],
                 switch_video_s=actions[k]['switch_video_s'])
        for ch in PZTS:
            s[f'{ch}_vpp_before'], s[f'{ch}_vpp_after'] = before[ch], after[ch]
        s['freq_hz'] = actions[k]['state']['PZT1']['freq_hz']
        if r['entity'] == 'object':
            s.update(speed_before_px_s=r['speed_before_px_s'], area_before_px2=r['area_before_px2'],
                     aspect_before=r['aspect_before'], speed_after_px_s=r['speed_late_px_s'],
                     area_after_px2=r['area_after_px2'], aspect_after=r['aspect_after'],
                     disp_x_px=r['disp_x_px'], disp_y_px=r['disp_y_px'], motion=r['motion'],
                     responded=r['responded_within_window'], delay_max_s=r['delay_max_s'], deformed=r['deformed'],
                     coverage_before=r['coverage_before'], coverage_during=r['coverage_during'],
                     trusted_only=True, flags=r['flags'])
        else:
            s.update({f'field_{x}_{y}': r.get(f'{x}_{y}') for x in ('n_regions', 'total_area_px2', 'spread_px',
                                                                    'cluster_share', 'nn_gap_px') for y in ('before', 'after')})
            s.update(objects_responded=r['objects_responded'], field_quality=r['field_quality'])
        s['time_quality'] = 'transition excluded'
        out.append(s)
    return out


def model_steps(T, aligned, fps, P, exp):
    dt = max(1, int(round(P['step_dt_s'] * fps)))
    al = {r['frame']: r for r in aligned}
    out = []
    for o_id, o in T.obj.items():
        idx = {int(f): i for i, f in enumerate(o['frame'])}
        for i, f in enumerate(o['frame']):
            f = int(f)
            if f % dt or (f + dt) not in idx:
                continue
            j = idx[f + dt]
            if o['segment'][j] != o['segment'][i]:
                continue
            a0, a1 = al.get(f), al.get(f + dt)
            if a0 is None or a1 is None:
                continue
            r = dict(experiment=exp, frame=f, t_s=f / fps, entity='object', object_id=o_id, id=int(o['id'][i]),
                     segment=int(o['segment'][i]), x_px=o['x'][i], y_px=o['y'][i], speed_px_s=o['speed'][i],
                     area_px2=o['area'][i], aspect_ratio=o['aspect'][i], shape_quality=o['quality'][i],
                     at_image_edge=int(o['edge'][i]))
            for ch in PZTS:
                r[f'{ch}_vpp'] = a0[f'{ch}_vpp']
            r.update(action_no=a0['action_no'], action_changes=a1['action_no'] != a0['action_no'],
                     time_quality='transition' if (a0['in_transition'] or a1['in_transition'] or a1['action_no'] != a0['action_no']) else 'stable',
                     next_x_px=o['x'][j], next_y_px=o['y'][j], next_speed_px_s=o['speed'][j],
                     next_area_px2=o['area'][j], next_aspect_ratio=o['aspect'][j], next_shape_quality=o['quality'][j])
            out.append(r)
    for i, f in enumerate(T.cframe):
        f = int(f)
        if f % dt or f + dt > T.cframe[-1]:
            continue
        a0, a1 = al.get(f), al.get(f + dt)
        if a0 is None or a1 is None:
            continue
        j = i + dt
        r = dict(experiment=exp, frame=f, t_s=f / fps, entity='field', object_id=None)
        for k in ('n_regions', 'total_area_px2', 'spread_px', 'cluster_area_share', 'nn_gap_median_px'):
            r[f'field_{k}'] = T.c[k][i]
            r[f'next_field_{k}'] = T.c[k][j]
        for ch in PZTS:
            r[f'{ch}_vpp'] = a0[f'{ch}_vpp']
        r.update(action_no=a0['action_no'], action_changes=a1['action_no'] != a0['action_no'],
                 time_quality='transition' if (a0['in_transition'] or a1['in_transition'] or a1['action_no'] != a0['action_no']) else 'stable',
                 field_quality=T.cquality[i])
        out.append(r)
    out.sort(key=lambda r: (r['frame'], r['entity'] != 'field', r['object_id'] or 0))
    return out


# ---------------------------------------------------------------------------
def write(path, rows, header=None):
    header = header or (list(rows[0].keys()) if rows else [])
    seen = list(header)
    for r in rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    analyze.write_csv(path, seen, rows)


def run(result_dir, pzt_dir, video=None, name=None, figures=True, verbose=True, **over):
    P = dict(DEFAULTS, **{k: v for k, v in over.items() if v is not None})
    log = print if verbose else (lambda *a, **k: None)
    an = os.path.join(result_dir, 'analysis')
    if not os.path.exists(os.path.join(an, 'report.json')):
        log('Round-2 analysis missing; running analyze.py first ...')
        analyze.run(result_dir, video=video, verbose=verbose)
    out = os.path.join(an, 'pzt_analysis')
    os.makedirs(out, exist_ok=True)
    rec = PztRecord(pzt_dir)
    T = Tables(an)
    D = analyze.Data(result_dir, dict(analyze.DEFAULTS, **(T.report.get('parameters') or {})), video=video)
    fps = D.fps
    n_frames = D.n_frames
    name = name or os.path.basename(os.path.normpath(pzt_dir))
    inputs_before = {f: sha256(os.path.join(result_dir, f)) for f in ('objects.csv', 'events.csv', 'review.json')
                     if os.path.exists(os.path.join(result_dir, f))}

    # ---- task 1 ----
    commands, actions = build_actions(rec, P)
    act_rows = action_rows(actions, n_frames, fps)
    log(f'task 1: {len(commands)} commands -> {len(actions) - 1} actions: ' + '; '.join(
        f"{a['switch_video_s']:.2f} s {a['label']}" for a in actions[1:]))
    # ---- task 2 ----
    aligned = align_frames(rec, actions, n_frames, P)
    # ---- task 3 ----
    events_by_obj = defaultdict(list)
    for e in T.events:
        for o in (e.get('object_ids') or '').split():
            events_by_obj[int(o)].append(dict(frame=int(e['frame']), type=e['type']))
    obj_rows, field_rows = [], []
    rate = onset_background_rate(T, actions, fps, n_frames, P)
    prate, p_resp, p_rest = placebo_rate(T, actions, n_frames, fps, P)
    P['_placebo_rate'] = prate
    log(f'control: at made-up switch times {p_resp} of {p_rest} regions at rest "responded" ({prate * 100:.1f} %)'
        if p_rest else 'control: no made-up switch times available')
    for k in range(1, len(actions)):
        w = windows(actions, k, n_frames, fps, P)
        actions[k]['windows'] = w
        for o_id in sorted(T.obj):
            r = object_response(T, o_id, actions, k, w, fps, P, events_by_obj)
            if r:
                obj_rows.append(r)
    for k in range(1, len(actions)):
        field_rows.append(field_response(T, actions, k, actions[k]['windows'], fps, P, obj_rows, rate))
    resp = []
    for k in range(1, len(actions)):
        resp += [r for r in field_rows if r['action_no'] == k] + [r for r in obj_rows if r['action_no'] == k]
    log(f'task 3: {len(obj_rows)} object responses, {len(field_rows)} field responses')
    # ---- task 5 ----
    samples = model_samples(T, actions, resp, fps, P, name)
    steps = model_steps(T, aligned, fps, P, name)
    log(f'task 5: {len(samples)} action samples, {len(steps)} 0.1 s steps')

    write(os.path.join(out, 'commands.csv'), commands)
    write(os.path.join(out, 'actions.csv'), act_rows)
    write(os.path.join(out, 'aligned_frames.csv'), aligned)
    write(os.path.join(out, 'responses.csv'), resp)
    write(os.path.join(out, 'model_samples.csv'), samples)
    write(os.path.join(out, 'model_steps.csv'), steps)

    precision = timing_summary(rec, actions, obj_rows, P)
    checks = verify(rec, D, commands, actions, aligned, obj_rows, T, video, name, result_dir, inputs_before, n_frames)
    fig_files, check_files = [], []
    if figures:
        import pzt_figures
        fig_files = pzt_figures.make_all(D, T, rec, actions, aligned, obj_rows, field_rows, out, P)
        if video:
            check_files = pzt_figures.action_checks(D, T, actions, obj_rows, video, out, P)
        log(f'figures: {len(fig_files)}, action check sheets: {len(check_files)}')
    summary = []
    for f in field_rows:
        k = f['action_no']
        rs = [o for o in obj_rows if o['action_no'] == k]
        summary.append(dict(action_no=k, action=f['action'], switch_video_s=round(actions[k]['switch_video_s'], 3),
                            first_frame=actions[k]['first_frame'], readback=actions[k]['readback_check'],
                            objects_tracked=f['objects_tracked'], objects_at_rest_before=f['objects_at_rest_before'],
                            objects_responded=f['objects_responded'],
                            objects_responded_expected_by_chance=(round(f['objects_responded_expected_by_chance'], 2)
                                                                  if f['objects_responded_expected_by_chance'] is not None else None),
                            first_change_delay_max_s=f['first_change_delay_max_s'],
                            moved=[o['object_id'] for o in rs if o['motion'] == 'started moving'],
                            deformed=[o['object_id'] for o in rs if o['deformed']],
                            motion_starts_in_window=f['motion_starts_in_window'],
                            motion_starts_expected_by_chance=round(f['motion_starts_expected_by_chance'], 2),
                            cluster_share_change=f['cluster_share_change'], spread_change_px=f['spread_px_change'],
                            field_quality=f['field_quality']))
    notes = quality_notes(rec, T, D, actions, obj_rows, P, name)
    report = dict(
        created=now(), experiment=name, result_dir=result_dir, pzt_dir=pzt_dir, video=video,
        experiment_id=rec.experiment_id, fps=fps, frames=n_frames,
        parameters={k: v for k, v in P.items() if not k.startswith('_')},
        control=dict(method='the same response detection at made-up switch times (1.25 s and 2.0 s after each '
                            'switch, away from the next one)', regions_at_rest=p_rest, responded=p_resp,
                     false_response_rate=prate),
        actions=[{k: v for k, v in r.items()} for r in act_rows],
        time_alignment=precision, action_summary=summary,
        checks=checks, all_checks_passed=all(c['passed'] for c in checks),
        data_quality=notes, figures=fig_files, action_check_sheets=check_files,
        files={'commands.csv': 'every channel command with software times, frames and readback',
               'actions.csv': 'one row per change of the 4-PZT state, with the full state of all four PZTs',
               'aligned_frames.csv': 'every video frame with its PZT state and time quality',
               'responses.csv': 'per action: whole field and every trusted object, before vs after',
               'model_samples.csv': 'per action and entity: state before, PZT action, state after, quality',
               'model_steps.csv': 'every 0.1 s: state, PZT state, state 0.1 s later (trusted objects and field)',
               'figures/': 'timeline, responses aligned to the actions, timing', 'checks/': 'video frames per action'})
    with open(os.path.join(out, 'report.json'), 'w', encoding='utf-8') as fh:
        json.dump(json_safe(report), fh, ensure_ascii=False, indent=1)
    for c in checks:
        log(('  ok    ' if c['passed'] else '  FAIL  ') + c['check'] + (f"  ({c['detail']})" if c.get('detail') else ''))
    for s in summary:
        log(f"  action {s['action_no']} {s['switch_video_s']:6.2f} s  {s['action']:<28} responded {s['objects_responded']}/"
            f"{s['objects_at_rest_before']} at rest (chance {s['objects_responded_expected_by_chance']}), first change <= "
            + (f"{s['first_change_delay_max_s'] * 1e3:.0f} ms" if s['first_change_delay_max_s'] is not None else '-'))
    for n in notes:
        log('  note: ' + n)
    log(f'written to {out}')
    return report


def timing_summary(rec, actions, obj_rows, P):
    ex = (rec.excess - rec.offset) * 1e3
    per = []
    for a in actions[1:]:
        d = [o for o in obj_rows if o['action_no'] == a['action_no'] and o['responded_within_window']]
        per.append(dict(action_no=a['action_no'], action=a['label'], earliest_frame=round(a['k_lo'], 2),
                        frames_received_around=[a['frame_received_before'], a['frame_received_after']],
                        commands_spread_ms=round(a['command_spread_ms'], 2),
                        first_changed_frame=min((o['first_change_frame'] for o in d), default=None),
                        readback=a['readback_check']))
    resp_delays = [o['delay_max_s'] for o in obj_rows if o['responded_within_window']]
    return dict(
        clock='commands and frame receive times share one Windows monotonic clock (QPC)',
        method='earliest possible exposure of each frame = receive time minus the smallest receive delay seen in '
               'this recording (camera frame clock is regular); command -> fractional frame k_lo',
        smallest_delay_frame=rec.min_delay_frame,
        receive_delay_excess_ms=dict(median=float(np.median(ex)), p95=float(np.percentile(ex, 95)), max=float(ex.max())),
        bracket_method_note='the original association (frame received just after the command) is up to '
                            f'{ex.max():.0f} ms too early, because frames arrive with a variable delay',
        unknown='the smallest camera / encoding / network delay, the AFG output delay and the acoustic build-up are '
                'not measured; they can only make the true switch frame later than k_lo',
        transition_s=P['transition_s'],
        observed_first_change_delay_s=dict(min=min(resp_delays) if resp_delays else None,
                                           median=float(np.median(resp_delays)) if resp_delays else None,
                                           max=max(resp_delays) if resp_delays else None),
        observed_meaning='time from k_lo to the first frame in which a region at rest had visibly moved; this is an '
                         'upper bound of (unknown delays + response), resolution 1 frame',
        per_action=per)


def verify(rec, D, commands, actions, aligned, obj_rows, T, video, name, result_dir, before, n_frames):
    res = []

    def add(check, ok, detail=''):
        res.append(dict(check=check, passed=bool(ok), detail=detail))

    ent = rec.manifest_entry(name)
    if video and ent and ent.get('video_sha256'):
        h = sha256(video)
        add('video file is the one these PZT records belong to (SHA-256 from manifest.json)', h == ent['video_sha256'],
            h[:16] + '…')
    add('one PZT time record per video frame, frame numbers 0..N-1',
        len(rec.frame_index) == n_frames and np.array_equal(rec.frame_index, np.arange(n_frames)),
        f'{len(rec.frame_index)} records, {n_frames} frames in the analysis')
    add('frame receive times increase', bool(np.all(np.diff(rec.recv) > 0)))
    add('every command has a readback that matches it', all(c['readback_matches'] for c in commands),
        f"{sum(c['readback_matches'] for c in commands)}/{len(commands)}")
    chk = rec.link.get('checks') or []
    add('AFG state checks after every step passed', bool(chk) and all(c.get('passed') for c in chk), f'{len(chk)} checks')
    lr = rec.link_report or {}
    if lr:
        add('original time-association report passed', bool(lr.get('passed')))
    bad = [c['action_id'] for c in commands if not (c['frame_received_before'] <= c['k_lo'] + 1e-9 or c['frame_received_before'] < 0)]
    add('earliest exposure frame is never before the frame received before the command', not bad, ', '.join(bad))
    firsts = [a['first_frame'] for a in actions]
    add('actions are in time order', firsts == sorted(firsts))
    st = [r['action_no'] for r in aligned]
    add('every frame has exactly one PZT state', len(aligned) == n_frames and all(np.diff(st) >= 0))
    add('responses use trusted object data only', all(o['object_id'] in T.obj for o in obj_rows))
    after = {f: sha256(os.path.join(result_dir, f)) for f in before}
    add('tracking and review files unchanged', after == before)
    return res


def quality_notes(rec, T, D, actions, obj_rows, P, name):
    notes = []
    rep = T.report or {}
    rows = rep.get('rows') or {}
    if not rows.get('trusted'):
        notes.append(f'{name}: no trusted tracks yet (not reviewed). Only the whole field is analysed; review this '
                     'experiment in review/review_tool.py, rerun analyze.py and then this script to get object responses.')
    elif rows.get('unreviewed'):
        notes.append(f"{rows['unreviewed']} region-frames are still unreviewed; they count as uncertain in the field.")
    notes.append('Experiment A / B are two recordings; the actions are named after the PZTs (e.g. "PZT1 → PZT2"), '
                 'not after an action A / B.')
    notes.append('A change right after an action shows that the two happen together in time. Whether this setting '
                 'causes it reliably needs repeated experiments; compare "motion_starts_in_window" with '
                 '"motion_starts_expected_by_chance" in responses.csv.')
    notes.append(f'Timing: frames from the earliest possible switch frame up to {P["transition_s"]:.2f} s later are '
                 'flagged "transition" and left out of the before / after statistics. The software confirmation '
                 '(readback) arrives about 1 s after a change and is not the moment the voltage changed.')
    stop = [a for a in actions if a['change'] == 'stop']
    if stop and stop[0]['active_before'] == 'none':
        notes.append('The final stop switches the outputs off while they were already at the idle level '
                     f"({actions[0]['state']['PZT1']['vpp']} Vpp); it serves as a control: little or no response is expected.")
    flagged = [o for o in obj_rows if o['flags']]
    if flagged:
        c = defaultdict(int)
        for o in flagged:
            for f in o['flags'].split(', '):
                c[f] += 1
        notes.append('Object responses with flags (check before use): ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items(), key=lambda kv: -kv[1])) + '.')
    return notes


def main():
    ap = argparse.ArgumentParser(description='Round 3: link the PZT actions to the microbubble changes')
    ap.add_argument('--result', required=True, help='result folder (with analysis/ from round 2)')
    ap.add_argument('--pzt', required=True, help='PZT record folder, e.g. data/pzt/expA')
    ap.add_argument('--video', help='original video; needed for the action check sheets and the video check')
    ap.add_argument('--name', help='experiment name (default: PZT folder name)')
    ap.add_argument('--transition', type=float, help='transition window after a switch in seconds (default 0.1)')
    ap.add_argument('--no-figures', action='store_true')
    a = ap.parse_args()
    rep = run(a.result, a.pzt, video=a.video, name=a.name, figures=not a.no_figures, transition_s=a.transition)
    sys.exit(0 if rep['all_checks_passed'] else 1)


if __name__ == '__main__':
    main()
