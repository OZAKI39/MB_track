"""
pzt_figures.py - figures for round 3 (analysis/pzt_analysis.py)

figures/timeline.png     the four PZTs over time, above the main objects' speed and the whole-field curves
figures/peri_action.png  every action in its own column, curves aligned to the switch (time 0)
figures/timing.png       per action: frames received around the command, earliest possible frame,
                         transition window and the first frames in which regions had moved
checks/action_<n>.png    video frames just before the switch, 0.5 s later and at the end of the action,
                         with the displacement of the regions that started moving
"""
import os

import cv2
import numpy as np

import figures as F
from figures import plt, SLOTS, INK, INK2, MUTED, AXIS

PZT_COLORS = {'PZT1': SLOTS[0], 'PZT2': SLOTS[1], 'PZT3': SLOTS[2], 'PZT4': SLOTS[6]}
PZTS = ['PZT1', 'PZT2', 'PZT3', 'PZT4']


def _mains(T):
    return (T.report or {}).get('main_objects', [])


def _switch_lines(axs, actions, fps, label_ax=None):
    for a in actions[1:]:
        t = a['first_frame'] / fps
        for ax in axs:
            ax.axvline(t, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        if label_ax is not None:
            label_ax.text(t, 1.02, f"{a['action_no']}", transform=label_ax.get_xaxis_transform(), ha='center',
                          va='bottom', fontsize=8, color=INK2)


def pzt_lanes(ax, aligned, actions, fps, t0=0.0, t1=None, shift=0.0):
    """Four lanes; a bar where the PZT drives (amplitude at or above the active level)."""
    f = np.array([r['frame'] for r in aligned])
    t = f / fps - shift
    for i, ch in enumerate(PZTS):
        y = 3 - i
        on = np.array([r[f'{ch}_vpp'] >= 0.1 for r in aligned])
        off = np.array([r[f'{ch}_output'] == 'off' for r in aligned])
        for mask, kw in ((on, dict(color=PZT_COLORS[ch], alpha=0.9)), (off, dict(color=AXIS, alpha=0.6, hatch='///'))):
            if not mask.any():
                continue
            edges = np.diff(np.r_[0, mask.astype(int), 0])
            for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
                ax.add_patch(plt.Rectangle((t[s], y - 0.32), t[e - 1] - t[s] + 1 / fps, 0.64, lw=0, **kw))
        ax.axhline(y, color=AXIS, lw=0.6, zorder=0)
    trans = np.array([r['in_transition'] for r in aligned])
    edges = np.diff(np.r_[0, trans.astype(int), 0])
    for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        ax.axvspan(t[s], t[e - 1] + 1 / fps, color=MUTED, alpha=0.25, lw=0)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(PZTS)
    ax.set_ylim(-0.6, 3.6)
    ax.grid(False)
    ax.set_xlim(t0, t1 if t1 is not None else t[-1])


def fig_timeline(D, T, aligned, actions, obj_rows, out, P, exp):
    fps = D.fps
    mains = _mains(T)
    cmap = F.color_map(mains)
    T_end = D.n_frames / fps
    rows = [('lanes', 1.1), ('speed', 1.3), ('share', 1.0), ('spread', 1.0), ('n', 0.9)]
    fig, axs = plt.subplots(len(rows), 1, figsize=(12, 10.5), sharex=True,
                            gridspec_kw=dict(height_ratios=[h for _, h in rows]))
    pzt_lanes(axs[0], aligned, actions, fps, 0, T_end)
    axs[0].set_title(f'{exp}: PZT drive (bar = {actions[1]["state"]["PZT1"]["freq_hz"] / 1e6:.1f} MHz, '
                     f'{max(a["state"]["PZT1"]["vpp"] for a in actions):.1f} Vpp; hatched = output off; '
                     'grey band = transition)', fontsize=10)
    for a in actions[1:]:
        axs[0].text(a['first_frame'] / fps + 0.06, 3.58, str(a['action_no']), fontsize=8, fontweight='bold',
                    color=INK2, va='top', ha='left')
    fig.text(0.01, 0.003, 'Actions: ' + '   '.join(f"{a['action_no']} {a['label']}" for a in actions[1:]),
             fontsize=8, color=INK2, ha='left', va='bottom')
    w = max(3, int(round(0.25 * fps)) | 1)
    ax = axs[1]
    for o in mains[:len(SLOTS)]:
        ob = T.obj.get(o['object_id'])
        if ob is None:
            continue
        sp = F.rolling_median(ob['speed'], w)
        ax.plot(*F.nan_breaks(ob['frame'] / fps, sp, 1.5 / fps), color=cmap[o['object_id']], lw=1.3, label=f"#{o['object_id']}")
    for r in obj_rows:
        if r['responded_within_window'] and r['object_id'] in cmap:
            ax.plot(r['first_change_frame'] / fps, 0, marker='^', ms=6, color=cmap[r['object_id']], mec='white', mew=0.5,
                    transform=ax.get_xaxis_transform(), clip_on=False)
    if not any(o['object_id'] in T.obj for o in mains):
        ax.text(0.5, 0.5, 'No trusted objects yet: review this experiment (review_tool.py), rerun analyze.py '
                'and pzt_analysis.py to see object responses.', transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color=INK2)
        ax.set_yticks([])
    ax.set_title('Speed of the main objects (0.25 s running median); ▲ = first visible change after an action')
    ax.set_ylabel('px/s')
    ax.set_ylim(bottom=0)
    if mains:
        ax.legend(ncol=min(8, len(mains)), loc='upper right', fontsize=7)
    t = T.cframe / fps
    for ax, keys, title, unit in ((axs[2], [('cluster_area_share', 'in clusters', SLOTS[0]), ('largest_share', 'in the largest region', SLOTS[1])],
                                   'Whole field: share of the visible area', 'share'),
                                  (axs[3], [('spread_px', None, SLOTS[0])], 'Whole field: spread (area-weighted)', 'px'),
                                  (axs[4], [('n_regions', None, SLOTS[0])], 'Whole field: number of regions', 'count')):
        for k, lab, c in keys:
            ax.plot(t, T.c[k], color=c, lw=0.5, alpha=0.35)
            ax.plot(t, F.rolling_median(T.c[k], w), color=c, lw=1.5, label=lab)
        ax.set_title(title)
        ax.set_ylabel(unit)
        if len(keys) > 1:
            ax.legend(loc='lower right', ncol=2, fontsize=7)
    axs[2].set_ylim(0, 1.02)
    _switch_lines(axs, actions, fps)
    axs[-1].set_xlabel('video time (s)')
    axs[-1].set_xlim(0, T_end)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return F.save(fig, os.path.join(out, 'figures', 'timeline.png'))


def fig_peri(D, T, actions, obj_rows, aligned, out, P, exp):
    fps = D.fps
    mains = _mains(T)
    cmap = F.color_map(mains)
    acts = actions[1:]
    n = len(acts)
    rows = ['lanes', 'speed', 'area', 'share', 'spread']
    fig, axs = plt.subplots(len(rows), n, figsize=(2.6 * n + 1.2, 9.5), sharex=True, sharey='row', squeeze=False,
                            gridspec_kw=dict(height_ratios=[0.55, 1.2, 1, 1, 1]))
    lo, hi = -1.0, 2.5
    for j, a in enumerate(acts):
        k0 = a['k_lo']
        sh = k0 / fps
        pzt_lanes(axs[0, j], aligned, actions, fps, lo, hi, shift=sh)
        axs[0, j].set_title(f"{a['action_no']} · {a['switch_video_s']:.2f} s\n{a['label']}", fontsize=8.5, loc='left')
        for o in mains[:len(SLOTS)]:
            ob = T.obj.get(o['object_id'])
            if ob is None:
                continue
            tt = (ob['frame'] - k0) / fps
            m = (tt >= lo) & (tt <= hi)
            if not m.any():
                continue
            c = cmap[o['object_id']]
            axs[1, j].plot(*F.nan_breaks(tt[m], ob['speed'][m], 1.5 / fps), color=c, lw=1.1)
            pre = (tt >= -1.0) & (tt < 0) & (ob['quality'] != 'poor')
            if pre.sum() >= 3:
                ref = np.median(ob['area'][pre])
                y = np.where(ob['quality'][m] != 'poor', ob['area'][m] / ref, np.nan)
                axs[2, j].plot(*F.nan_breaks(tt[m], y, 1.5 / fps), color=c, lw=1.1)
        tt = (T.cframe - k0) / fps
        m = (tt >= lo) & (tt <= hi)
        axs[3, j].plot(tt[m], T.c['cluster_area_share'][m], color=SLOTS[0], lw=1.1)
        axs[3, j].plot(tt[m], T.c['largest_share'][m], color=SLOTS[1], lw=1.1)
        axs[4, j].plot(tt[m], T.c['spread_px'][m], color=SLOTS[0], lw=1.1)
        for i in range(1, len(rows)):
            axs[i, j].axvspan(0, P['transition_s'], color=MUTED, alpha=0.25, lw=0)
            axs[i, j].axvline(0, color=INK2, lw=0.8, ls=(0, (3, 3)))
        axs[-1, j].set_xlabel('s from switch')
        axs[-1, j].set_xlim(lo, hi)
    axs[1, 0].set_ylabel('speed\n(px/s)')
    axs[2, 0].set_ylabel('area / area\nbefore')
    axs[3, 0].set_ylabel('area share\n(blue clusters,\norange largest)')
    axs[4, 0].set_ylabel('spread (px)')
    axs[1, 0].set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=cmap[o['object_id']], lw=1.5, label=f"#{o['object_id']}") for o in mains[:len(SLOTS)]]
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.suptitle(f'{exp}: every action aligned to its earliest possible switch frame (0 s); grey = transition window',
                 x=0.01, ha='left', fontsize=10, fontweight='bold')
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return F.save(fig, os.path.join(out, 'figures', 'peri_action.png'))


def fig_timing(D, T, rec, actions, obj_rows, out, P, exp):
    fps = D.fps
    mains = _mains(T)
    cmap = F.color_map(mains)
    acts = actions[1:]
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(13, 0.55 * len(acts) + 2.2), gridspec_kw=dict(width_ratios=[2.6, 1]))
    trans = P['transition_s'] * fps
    for i, a in enumerate(acts):
        y = len(acts) - i
        k0 = a['k_lo']
        ax.axhspan(y - 0.35, y + 0.35, xmin=0, xmax=1, color='none')
        ax.add_patch(plt.Rectangle((0, y - 0.3), trans, 0.6, color=MUTED, alpha=0.25, lw=0))
        ax.plot([a['frame_received_before'] - k0, a['frame_received_after'] - k0], [y, y], '|', ms=12, color=AXIS, mew=2)
        ax.plot([0], [y], marker='|', ms=16, color=INK, mew=2)
        for r in obj_rows:
            if r['action_no'] == a['action_no'] and r['first_change_frame'] is not None:
                c = cmap.get(r['object_id'], MUTED)
                ax.plot(r['first_change_frame'] - k0, y + (0.12 if r['object_id'] in cmap else -0.12), 'o', ms=5, color=c,
                        mec='white', mew=0.5, zorder=3)
        ax.text(16.3, y, f"{a['label']}  ·  readback {a['readback_check']}", va='center', fontsize=8, color=INK2)
    ax.set_yticks(range(len(acts), 0, -1))
    ax.set_yticklabels([f"{a['action_no']} · {a['switch_video_s']:.2f} s" for a in acts], fontsize=8)
    ax.set_xlim(-8, 16)
    ax.set_xlabel('frames after the earliest possible switch frame (k_lo)')
    ax.set_title('When each action can first appear in the video, and when regions first moved', fontsize=10)
    ax.grid(axis='y', visible=False)
    leg = [plt.Line2D([], [], ls='', marker='|', ms=12, mew=2, color=AXIS, label='frames received just before / after the command'),
           plt.Line2D([], [], ls='', marker='|', ms=14, mew=2, color=INK, label='earliest possible frame (k_lo)'),
           plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.25, label=f'transition window ({P["transition_s"]:.2f} s)'),
           plt.Line2D([], [], ls='', marker='o', color=MUTED, label='first visible change of a region')]
    ax.legend(handles=leg, loc='upper left', bbox_to_anchor=(0, -0.18), ncol=2, fontsize=8)
    ex = (rec.excess - rec.offset) * 1e3
    axh.hist(ex, bins=30, color=SLOTS[0], alpha=0.85)
    axh.set_title('Frame arrival delay above the smallest one', fontsize=10)
    axh.set_xlabel('ms')
    axh.set_ylabel('frames')
    fig.tight_layout()
    return F.save(fig, os.path.join(out, 'figures', 'timing.png'))


def make_all(D, T, rec, actions, aligned, obj_rows, field_rows, out, P):
    exp = os.path.basename(os.path.normpath(rec.dir))
    files = [fig_timeline(D, T, aligned, actions, obj_rows, out, P, exp),
             fig_peri(D, T, actions, obj_rows, aligned, out, P, exp),
             fig_timing(D, T, rec, actions, obj_rows, out, P, exp)]
    return [os.path.relpath(f, out) for f in files]


def action_checks(D, T, actions, obj_rows, video, out, P):
    fps = D.fps
    crop = (D.stats.get('params') or {}).get('crop')
    sets = []
    for a in actions[1:]:
        w = a['windows']
        fs = [max(0, w['switch'] - 1), min(D.n_frames - 1, w['stable'] + int(round(0.5 * fps))), max(0, w['next'] - 1)]
        sets.append((a, fs))
    frames = F.read_frames(video, [f for _, fs in sets for f in fs], crop)
    files = []
    os.makedirs(os.path.join(out, 'checks'), exist_ok=True)
    for a, fs in sets:
        movers = [r for r in obj_rows if r['action_no'] == a['action_no'] and r['motion'] == 'started moving']
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.9))
        names = ['just before the switch', '0.5 s after the transition', 'end of the action']
        start = {}
        for r in movers:
            ob = T.obj[r['object_id']]
            m = ob['frame'] < a['windows']['switch']
            if m.any():
                start[r['object_id']] = (ob['x'][np.nonzero(m)[0][-1]], ob['y'][np.nonzero(m)[0][-1]])
        for ax, f, nm in zip(axs, fs, names):
            im = frames.get(f)
            if im is not None:
                im = F.draw_regions(D, im, f, focus_ids=set(), thick=2)
                for oid, (x0, y0) in start.items():
                    ob = T.obj[oid]
                    m = ob['frame'] == f
                    if m.any() and f > fs[0]:
                        x1, y1 = ob['x'][m][0], ob['y'][m][0]
                        cv2.arrowedLine(im, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 5, tipLength=0.25)
                        cv2.arrowedLine(im, (int(x0), int(y0)), (int(x1), int(y1)), (40, 200, 255), 2, tipLength=0.25)
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f'{nm}: frame {f} · {f / fps:.2f} s', fontsize=9, loc='left')
        moved = ', '.join(f"#{r['object_id']}" for r in movers) or 'none'
        fig.suptitle(f"Action {a['action_no']} at {a['switch_video_s']:.2f} s: {a['label']}   ·   started moving: {moved}"
                     '   ·   yellow arrows: displacement since the switch; outlines: green trusted, orange uncertain, red false target',
                     x=0.01, ha='left', fontsize=9.5, fontweight='bold')
        fig.tight_layout()
        p = F.save(fig, os.path.join(out, 'checks', f"action_{a['action_no']:02d}.png"))
        files.append(os.path.relpath(p, out))
    return files
