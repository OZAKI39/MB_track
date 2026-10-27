"""
figures.py - task 5: figures and key-moment check sheets for analysis/analyze.py

All figures are PNG files in <result>/analysis/figures/ :
    trajectories.png        every trusted segment on the background image; main objects in colour
    motion_summary.png      main objects: distance from the first trusted position, and speed
    shape_summary.png       main objects: area and aspect ratio
    collective.png          whole field: number of regions, total area, spread, clustering, gaps, events
    objects/object_<id>.png one sheet per main object: path, x / y, speed, area, aspect ratio,
                            circularity, orientation; segment cuts and shape quality marked

Check sheets in <result>/analysis/checks/ show the video frames next to the curves at a few
automatically chosen moments (largest change of a main object, fastest motion, biggest change of the
whole field), so the numbers can be compared with what the video shows.
"""
import os
from collections import defaultdict

import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# reference palette (dataviz skill): categorical slots in fixed order, chart ink and surfaces
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'
OTHER = '#b9b8b0'
ROLE_BGR = {'confirmed': (60, 175, 40), 'uncertain': (0, 160, 235), 'excluded': (60, 60, 220)}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif', 'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.labelsize': 9, 'axes.labelcolor': INK2, 'axes.edgecolor': AXIS,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'text.color': INK, 'legend.frameon': False, 'legend.fontsize': 8, 'lines.linewidth': 1.6,
})


def color_map(mains):
    return {o['object_id']: SLOTS[k] for k, o in enumerate(mains[:len(SLOTS)])}


def nan_breaks(t, y, maxgap):
    """Insert NaN where consecutive samples are further apart than maxgap, so lines are not drawn across gaps."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 2:
        return t, y
    br = np.nonzero(np.diff(t) > maxgap)[0]
    return np.insert(t, br + 1, np.nan), np.insert(y, br + 1, np.nan)


def rolling_median(y, w):
    y = np.asarray(y, float)
    if w <= 1 or len(y) < 3:
        return y
    h = w // 2
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - h): i + h + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmedian(seg)
    return out


def bg_image(D):
    if D.bg is None:
        return np.full((D.H, D.W), 200, np.uint8)
    b = np.clip(D.bg, 0, 255)
    b = (b - np.percentile(b, 1)) / max(1e-6, np.percentile(b, 99.5) - np.percentile(b, 1))
    return (np.clip(b, 0, 1) * 160 + 85).astype(np.uint8)   # lighten so coloured paths stand out


def by_segment(rows):
    out = defaultdict(list)
    for r in rows:
        out[r['segment']].append(r)
    return out


def col(rows, k):
    return np.array([np.nan if r.get(k) is None else r[k] for r in rows], float)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def fig_trajectories(D, out, segs, motion, mains):
    cmap = color_map(mains)
    ms = by_segment(motion)
    h, w = D.H, D.W
    fig, ax = plt.subplots(figsize=(9, 9 * h / w + 0.6))
    ax.imshow(bg_image(D), cmap='gray', vmin=0, vmax=255, extent=(0, w, h, 0))
    ax.grid(False)
    for s in segs:
        rows = ms[s['segment']]
        x, y = col(rows, 'x_px'), col(rows, 'y_px')
        c = cmap.get(s['object_id'])
        if c is None:
            ax.plot(x, y, color='#5a5955', lw=0.8, alpha=0.8)
    for s in segs:
        c = cmap.get(s['object_id'])
        if c is None:
            continue
        rows = ms[s['segment']]
        x, y = col(rows, 'x_px'), col(rows, 'y_px')
        ax.plot(x, y, color='white', lw=3.6, solid_capstyle='round')
        ax.plot(x, y, color=c, lw=2.0, solid_capstyle='round')
    for o in mains[:len(SLOTS)]:
        ss = [s for s in segs if s['object_id'] == o['object_id']]
        s0, s1 = ss[0], ss[-1]
        ax.plot(s0['x_start_px'], s0['y_start_px'], 'o', ms=6, mfc='white', mec=cmap[o['object_id']], mew=2)
        ax.plot(s1['x_end_px'], s1['y_end_px'], 's', ms=6, mfc=cmap[o['object_id']], mec='white', mew=1.2)
        ax.annotate(f"#{o['object_id']}", (s0['x_start_px'], s0['y_start_px']), xytext=(6, -6),
                    textcoords='offset points', fontsize=9, fontweight='bold', color=INK,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [Line2D([], [], color=cmap[o['object_id']], lw=2, label=f"#{o['object_id']}")
               for o in mains[:len(SLOTS)]]
    handles += [Line2D([], [], color='#5a5955', lw=0.8, label='other trusted segments'),
                Line2D([], [], ls='', marker='o', mfc='white', mec=INK2, label='first trusted position'),
                Line2D([], [], ls='', marker='s', mfc=INK2, mec='white', label='last trusted position')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_title(f'Trusted trajectories: {len(segs)} segments, main objects in colour', loc='left')
    return save(fig, os.path.join(out, 'figures', 'trajectories.png'))


def object_series(D, motion, shape, oid):
    m = [r for r in motion if r['object_id'] == oid]
    s = [r for r in shape if r['object_id'] == oid]
    return m, s


def fig_motion_summary(D, out, motion, mains, joint=None):
    cmap = color_map(mains)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    gap = 1.5 / D.fps
    w = max(3, int(round(0.25 * D.fps)) | 1)
    for o in mains[:len(SLOTS)]:
        m = [r for r in motion if r['object_id'] == o['object_id']]
        t = col(m, 't_s')
        x, y = col(m, 'x_px'), col(m, 'y_px')
        d = np.hypot(x - x[0], y - y[0])
        c = cmap[o['object_id']]
        axs[0].plot(*nan_breaks(t, d, gap), color=c, label=f"#{o['object_id']}")
        sp = rolling_median(col(m, 'speed_px_s'), w)
        axs[1].plot(*nan_breaks(t, sp, gap), color=c)
    axs[0].set_title('Distance from the first trusted position')
    axs[0].set_ylabel('px')
    axs[1].set_title('Speed of the region (0.25 s running median)')
    axs[1].set_ylabel('px/s')
    axs[1].set_xlabel('time (s)')
    axs[0].legend(ncol=min(8, len(mains)), loc='lower left', bbox_to_anchor=(0, 1.08), handlelength=1.4)
    for a in axs:
        a.set_xlim(0, D.n_frames / D.fps)
        a.set_ylim(bottom=0)
    mark_joint(axs, joint)
    if joint:
        fig.text(0.01, -0.01, 'Yellow bands: several regions start moving within 0.1 s of each other '
                 '(candidate moments to compare with the PZT log).', color=MUTED, fontsize=8)
    fig.tight_layout()
    return save(fig, os.path.join(out, 'figures', 'motion_summary.png'))


def fig_shape_summary(D, out, shape, mains, joint=None):
    cmap = color_map(mains)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    gap = 1.5 / D.fps
    for o in mains[:len(SLOTS)]:
        s = [r for r in shape if r['object_id'] == o['object_id'] and r['shape_quality'] != 'poor']
        if not s:
            continue
        t = col(s, 't_s')
        c = cmap[o['object_id']]
        axs[0].plot(*nan_breaks(t, col(s, 'area_px2'), gap), color=c, label=f"#{o['object_id']}")
        axs[1].plot(*nan_breaks(t, col(s, 'aspect_ratio'), gap), color=c, lw=1.1)
    axs[0].set_title('Visible area (frames with poor shape quality left out)')
    axs[0].set_ylabel('px²')
    axs[1].set_title('Aspect ratio (long axis / short axis; 1 = round)')
    axs[1].set_ylabel('ratio')
    axs[1].set_xlabel('time (s)')
    axs[0].legend(ncol=min(8, len(mains)), loc='lower left', bbox_to_anchor=(0, 1.08), handlelength=1.4)
    axs[0].set_ylim(bottom=0)
    axs[1].set_ylim(bottom=1)
    for a in axs:
        a.set_xlim(0, D.n_frames / D.fps)
    mark_joint(axs, joint)
    fig.text(0.01, -0.01, 'Per-object sheets (figures/objects/) show which frames touch a wall.'
             + (' Yellow bands: several regions start moving together.' if joint else ''), color=MUTED, fontsize=8)
    fig.tight_layout()
    return save(fig, os.path.join(out, 'figures', 'shape_summary.png'))


def shade_uncertain(ax, t, unc):
    """Light band where the whole-field values are uncertain."""
    if not unc.any():
        return
    edges = np.diff(np.r_[0, unc.astype(int), 0])
    for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        ax.axvspan(t[a], t[b - 1] + (t[1] - t[0]), color='#fab219', alpha=0.13, lw=0)


def fig_collective(D, out, coll, conns, joint=None):
    t = col(coll, 't_s')
    unc = np.array([r['field_quality'] == 'uncertain' for r in coll])
    w = max(3, int(round(0.25 * D.fps)) | 1)
    panels = [
        ('Number of visible regions', 'count', [('n_regions', 'all regions (except false targets)', SLOTS[0]),
                                               ('n_confirmed', 'trusted regions only', SLOTS[1])]),
        ('Total visible area', 'px²', [('total_area_px2', None, SLOTS[0])]),
        ('Spread: area-weighted distance from the common centre', 'px', [('spread_px', None, SLOTS[0])]),
        ('Share of the visible area in clusters / in the largest region', 'share',
         [('cluster_area_share', f'in clusters (≥ {D.P["cluster_area_px"]} px²)', SLOTS[0]),
          ('largest_share', 'in the largest region', SLOTS[1])]),
        ('Median gap to the nearest other region (edge to edge)', 'px', [('nn_gap_median_px', None, SLOTS[0])]),
    ]
    fig, axs = plt.subplots(len(panels) + 1, 1, figsize=(10, 11.5), sharex=True,
                            gridspec_kw=dict(height_ratios=[1] * len(panels) + [0.7]))
    for ax, (title, unit, series) in zip(axs, panels):
        shade_uncertain(ax, t, unc)
        for k, label, c in series:
            y = col(coll, k)
            ax.plot(t, y, color=c, lw=0.6, alpha=0.35)
            ax.plot(t, rolling_median(y, w), color=c, lw=1.6, label=label)
        ax.set_title(title)
        ax.set_ylabel(unit)
        if len(series) > 1:
            ax.legend(loc='upper right', ncol=2)
    axs[3].set_ylim(0, 1.02)
    ax = axs[-1]
    rows = {'connect': (3, 'v', SLOTS[6]), 'separate': (2, '^', SLOTS[2]),
            'enter view': (1, '>', SLOTS[0]), 'leave view': (0, '<', SLOTS[1])}
    for c in conns:
        if not c['counted_in_field']:
            continue
        y, mk, colr = rows[c['type']]
        a = c.get('region_max_area_px2') or 0
        ms = 4 + 8 * min(1.0, np.sqrt(max(a, 0) / 3000.0))
        ax.plot(c['t_s'], y, mk, ms=ms, color=colr, mec='white', mew=0.5)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(['connect', 'separate', 'enter view', 'leave view'])
    ax.grid(False)
    ax.set_title('Field events (bigger marker = bigger region; enter / leave = field-of-view changes)')
    ax.set_xlabel('time (s)')
    ax.set_xlim(0, D.n_frames / D.fps)
    mark_joint(axs, joint)
    note = ('Thin line = every frame, thick line = 0.25 s running median. A jump at an "enter view" / '
            '"leave view" marker is a field-of-view effect. '
            + ('Yellow bands = several regions start moving together. ' if joint else '')
            + ('Orange band = uncertain regions hold more than '
               f'{D.P["field_uncertain_share"] * 100:.0f} % of the visible area.' if unc.any() else ''))
    fig.text(0.01, -0.005, note, color=MUTED, fontsize=8)
    fig.tight_layout()
    return save(fig, os.path.join(out, 'figures', 'collective.png'))


def fig_object(D, out, segs, motion, shape, o, color, static):
    oid = o['object_id']
    ss = [s for s in segs if s['object_id'] == oid]
    m, sh = object_series(D, motion, shape, oid)
    ms, shs = by_segment(m), by_segment(sh)
    gap = 1.5 / D.fps
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(7, 2, width_ratios=[1, 1.55], hspace=0.55, wspace=0.12)
    # ---- path on the background ----
    axp = fig.add_subplot(gs[:, 0])
    xs = np.concatenate([col(ms[s['segment']], 'x_px') for s in ss])
    ys = np.concatenate([col(ms[s['segment']], 'y_px') for s in ss])
    first = (int(ss[0]['start_frame']), int(ss[0]['track']))
    last = (int(ss[-1]['end_frame']), int(ss[-1]['track']))
    cs = [D.contour.get(first), D.contour.get(last)]
    pts = np.vstack([np.c_[xs, ys]] + [c for c in cs if c is not None])
    pad = 40
    x0, y0 = max(0, int(pts[:, 0].min()) - pad), max(0, int(pts[:, 1].min()) - pad)
    x1, y1 = min(D.W, int(pts[:, 0].max()) + pad), min(D.H, int(pts[:, 1].max()) + pad)
    side = max(x1 - x0, y1 - y0, 160)
    cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = int(max(0, cxm - side / 2)), int(min(D.W, cxm + side / 2))
    y0, y1 = int(max(0, cym - side / 2)), int(min(D.H, cym + side / 2))
    axp.imshow(bg_image(D)[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=255, extent=(x0, x1, y1, y0))
    axp.grid(False)
    for s in ss:
        r = ms[s['segment']]
        axp.plot(col(r, 'x_px'), col(r, 'y_px'), color='white', lw=3.4)
        axp.plot(col(r, 'x_px'), col(r, 'y_px'), color=color, lw=1.8)
    for c, ls, lab in ((cs[0], '--', f'outline at {first[0] / D.fps:.2f} s'),
                       (cs[1], '-', f'outline at {last[0] / D.fps:.2f} s')):
        if c is not None:
            cc = np.vstack([c, c[:1]])
            axp.plot(cc[:, 0], cc[:, 1], ls, color=INK, lw=1.1, label=lab)
    axp.plot(ss[0]['x_start_px'], ss[0]['y_start_px'], 'o', ms=7, mfc='white', mec=color, mew=2, label='first trusted position')
    axp.plot(ss[-1]['x_end_px'], ss[-1]['y_end_px'], 's', ms=7, mfc=color, mec='white', mew=1.2, label='last trusted position')
    axp.set_xlim(x0, x1)
    axp.set_ylim(y1, y0)
    axp.set_xlabel('x (px)')
    axp.set_ylabel('y (px)')
    axp.legend(loc='upper left', bbox_to_anchor=(0, -0.11), ncol=2, fontsize=8)
    ids = ', '.join(f'#{i}' for i in o['ids'])
    axp.set_title(f"Object #{oid}  (track IDs {ids})", loc='left')

    # ---- curves ----
    axs = [fig.add_subplot(gs[i, 1]) for i in range(7)]
    for a in axs[1:]:
        a.sharex(axs[0])
    T = D.n_frames / D.fps
    t_lo = max(0, min(s['t_start_s'] for s in ss) - 0.3)
    t_hi = min(T, max(s['t_end_s'] for s in ss) + 0.3)
    for s in ss:
        r, q = ms[s['segment']], shs[s['segment']]
        t = col(r, 't_s')
        axs[0].plot(*nan_breaks(t, col(r, 'x_px'), gap), color=color, label='x' if s is ss[0] else None)
        axs[0].plot(*nan_breaks(t, col(r, 'y_px'), gap), color=SLOTS[1] if color != SLOTS[1] else SLOTS[0],
                    label='y' if s is ss[0] else None)
        sp = col(r, 'speed_px_s')
        edge = np.array([z['at_image_edge'] for z in r])
        axs[1].plot(*nan_breaks(t, sp, gap), color=color, lw=0.9, alpha=0.45)
        axs[1].plot(*nan_breaks(t, np.where(edge, np.nan, sp), gap), color=color, lw=1.8)
        qual = np.array([z['shape_quality'] for z in q])
        tt = col(q, 't_s')
        for i, (k, ax) in enumerate((('area_px2', axs[2]), ('aspect_ratio', axs[3]), ('circularity', axs[4]),
                                     ('solidity', axs[5]))):
            y = col(q, k)
            ax.plot(*nan_breaks(tt, np.where(qual != 'poor', y, np.nan), gap), color=color, lw=0.9, alpha=0.45)
            ax.plot(*nan_breaks(tt, np.where(qual == 'good', y, np.nan), gap), color=color, lw=1.8)
        yo = col(q, 'orientation_deg')
        ok = np.array([z['orientation_ok'] for z in q])
        axs[6].plot(tt[ok], yo[ok], '.', ms=2.5, color=color)
    cuts = sorted({s['start_frame'] for s in ss[1:]})
    for a in axs:
        for f in cuts:
            a.axvline(f / D.fps, color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=0)
        a.set_xlim(t_lo, t_hi)
        a.tick_params(labelbottom=False)
    axs[-1].tick_params(labelbottom=True)
    axs[-1].set_xlabel('time (s)')
    titles = [('Position (centroid)', 'px'), ('Speed (over ±0.05 s; faint = cut by the image edge)', 'px/s'),
              ('Visible area   (faint = pressed against a wall or debris)', 'px²'),
              ('Aspect ratio (1 = round)', 'ratio'), ('Circularity (1 = smooth circle)', ''),
              ('Solidity (1 = no dents)', ''), ('Orientation of the long axis (only when clearly elongated)', 'deg')]
    for a, (tt_, u) in zip(axs, titles):
        a.set_title(tt_, fontsize=9)
        a.set_ylabel(u)
    axs[0].legend(loc='upper right', ncol=2)
    axs[1].set_ylim(bottom=0)
    axs[2].set_ylim(bottom=0)
    axs[3].set_ylim(bottom=1)
    axs[6].set_ylim(0, 180)
    axs[6].set_yticks([0, 90, 180])
    fig.text(0.37, 0.005, 'Dashed vertical lines: the trusted data are cut there (connect / separate event, '
             'untrusted part or another ID). Curves are not joined across a cut. Shape curves leave out frames '
             'where the region is small or cut by the image edge.', color=MUTED, fontsize=8)
    return save(fig, os.path.join(out, 'figures', 'objects', f'object_{oid}.png'))


def mark_joint(axs, joint):
    """Thin vertical bands where several regions start moving together."""
    for j in joint or []:
        for a in axs:
            a.axvspan(j['t_first_s'] - 0.02, j['t_last_s'] + 0.02, color=SLOTS[3], alpha=0.25, lw=0, zorder=0)


def make_all(D, out, segs, motion, shape, coll, conns, mains, static, joint=None):
    files = []
    if segs:
        files.append(fig_trajectories(D, out, segs, motion, mains))
    if mains:
        files.append(fig_motion_summary(D, out, motion, mains, joint))
        files.append(fig_shape_summary(D, out, shape, mains, joint))
    files.append(fig_collective(D, out, coll, conns, joint))
    cmap = color_map(mains)
    for k, o in enumerate(mains):
        files.append(fig_object(D, out, segs, motion, shape, o, cmap.get(o['object_id'], SLOTS[k % len(SLOTS)]), static))
    return [os.path.relpath(f, out) for f in files]


# ---------------------------------------------------------------------------
# Key-moment check sheets
# ---------------------------------------------------------------------------
def read_frames(video, frames, crop=None):
    cap = cv2.VideoCapture(video)
    out = {}
    for f in sorted(set(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if ok:
            if crop:
                x, y, w, h = crop
                im = im[y:y + h, x:x + w]
            out[f] = im
    cap.release()
    return out


def choose_moments(D, segs, motion, shape, coll, conns, mains, n):
    """Pick moments worth comparing with the video.

    Frames where the shape is 'poor' (small or cut by the image edge) are skipped for the object checks,
    because there the cause of a change is already known.
    """
    picks = []
    hw = max(1, int(round(0.1 * D.fps)))
    shs = by_segment(shape)
    main_ids = [o['object_id'] for o in mains]
    for o in mains[:2]:
        best = None
        for s in [s for s in segs if s['object_id'] == o['object_id']]:
            q = shs[s['segment']]
            if len(q) < 2 * hw + 1:
                continue
            a = col(q, 'area_px2')
            poor = np.array([r['shape_quality'] == 'poor' for r in q])
            d = np.abs(a[2 * hw:] - a[:-2 * hw]) / np.maximum(a[:-2 * hw], 1)
            d[poor[2 * hw:] | poor[:-2 * hw] | poor[hw:-hw]] = -1
            i = int(np.argmax(d))
            if d[i] > 0 and (best is None or d[i] > best[0]):
                best = (d[i], q[i + hw]['frame'], s['track'])
        if best:
            picks.append(dict(kind='object', what=f'largest change of the visible area within one trusted segment '
                                                  f'(over {2 * hw / D.fps:.1f} s, shape not poor)',
                              object_id=o['object_id'], id=best[2], frame=int(best[1]), half=hw))
    # largest area jump at a connect event of a main object
    best = None
    for c in conns:
        if c['type'] != 'connect' or not c['counted_in_field']:
            continue
        host = c['result_id']
        ks = D.rows_by_track.get(host)
        if ks is None or int(D.object_id[ks[0]]) not in main_ids:
            continue
        fr = D.F[ks]
        before, after = ks[fr < c['frame']], ks[fr >= c['frame']]
        if not len(before) or not len(after):
            continue
        jump = abs(D.area[after[0]] - D.area[before[-1]]) / max(1, D.area[before[-1]])
        if best is None or jump > best[0]:
            best = (jump, c, host)
    if best:
        c = best[2]
        picks.append(dict(kind='object', what=f"connect event: #{best[1]['ids'].split()[0]} connects with #{c}; "
                                              'the trusted data are cut here',
                          object_id=int(D.object_id[D.rows_by_track[c][0]]), id=c, frame=int(best[1]['frame']),
                          half=max(1, int(round(0.05 * D.fps)))))
    # fastest motion of a main object, not at the image edge
    best = None
    for r in motion:
        if (r['object_id'] in main_ids[:5] and not r['at_image_edge'] and np.isfinite(r['speed_px_s'])
                and (best is None or r['speed_px_s'] > best['speed_px_s'])):
            best = r
    if best:
        picks.append(dict(kind='object', what='fastest motion of a main object (not at the image edge)',
                          object_id=best['object_id'], id=best['id'], frame=int(best['frame']), half=hw))
    # whole field: biggest changes over 0.5 s
    hf = max(1, int(round(0.25 * D.fps)))
    for k, what in (('total_area_px2', 'largest change of the total visible area (over 0.5 s)'),
                    ('cluster_area_share', 'largest change of the share of area in clusters (over 0.5 s)'),
                    ('spread_px', 'largest change of the spread (over 0.5 s)')):
        y = rolling_median(col(coll, k), max(3, hf | 1))
        if len(y) > 2 * hf + 1:
            d = np.abs(y[2 * hf:] - y[:-2 * hf])
            d[~np.isfinite(d)] = -1
            i = int(np.argmax(d)) + hf
            near = [c for c in conns if c['type'] in ('enter view', 'leave view') and c['counted_in_field']
                    and abs(c['frame'] - i) <= 2 * hf and (c.get('region_max_area_px2') or 0) >= D.P['cluster_area_px']]
            picks.append(dict(kind='field', what=what, key=k, frame=i, half=hf,
                              field_of_view=[f"#{c['ids']} {c['type']} at {c['t_s']:.2f} s" for c in near]))
    return picks[:n]


def draw_regions(D, im, f, focus_ids=(), all_regions=True, thick=1):
    out = im.copy()
    for k in D.rows_by_frame.get(f, []):
        i = int(D.I[k])
        c = D.contour.get((f, i))
        if c is None:
            continue
        if i in focus_ids:
            cv2.polylines(out, [c.reshape(-1, 1, 2)], True, (255, 255, 255), 3)
            cv2.polylines(out, [c.reshape(-1, 1, 2)], True, (40, 200, 255), 1)
        elif all_regions:
            cv2.polylines(out, [c.reshape(-1, 1, 2)], True, ROLE_BGR[D.role[k]], thick)
    return out


def field_box(D, fa, fb, top=3, margin=110, min_side=420):
    """Crop for a whole-field check: around the regions whose area changed most between two frames."""
    def areas(f):
        out = defaultdict(float)
        for k in D.rows_by_frame.get(f, []):
            if D.role[k] != 'excluded':
                out[int(D.object_id[k])] += D.area[k]
        return out
    A, B = areas(fa), areas(fb)
    ch = sorted(set(A) | set(B), key=lambda o: -abs(A.get(o, 0) - B.get(o, 0)))[:top]
    pts = []
    for f in (fa, fb):
        for k in D.rows_by_frame.get(f, []):
            if int(D.object_id[k]) in ch:
                c = D.contour.get((f, int(D.I[k])))
                pts.append(c if c is not None else np.array([[D.cx[k], D.cy[k]]]))
    if not pts:
        return 0, 0, D.W, D.H
    pts = np.vstack(pts)
    x0, y0 = pts.min(0) - margin
    x1, y1 = pts.max(0) + margin
    side = max(x1 - x0, y1 - y0, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = int(max(0, cx - side / 2)), int(min(D.W, cx + side / 2))
    y0, y1 = int(max(0, cy - side / 2)), int(min(D.H, cy + side / 2))
    return x0, y0, x1, y1


def key_moment_checks(D, video, out, segs, motion, shape, coll, conns, mains, n=8):
    picks = choose_moments(D, segs, motion, shape, coll, conns, mains, n)
    crop = (D.stats.get('params') or {}).get('crop')
    clampf = lambda f: max(0, min(D.n_frames - 1, f))
    need = [clampf(p['frame'] + d) for p in picks for d in (-p['half'], 0, p['half'])]
    frames = read_frames(video, need, crop)
    os.makedirs(os.path.join(out, 'checks'), exist_ok=True)
    mrow = {(r['frame'], r['id']): r for r in motion}
    srow = {(r['frame'], r['id']): r for r in shape}
    crow = {r['frame']: r for r in coll}
    items = []
    for k, p in enumerate(picks, start=1):
        fs = [clampf(p['frame'] + d) for d in (-p['half'], 0, p['half'])]
        field = p['kind'] == 'field'
        fig = plt.figure(figsize=(13, 8.6 if field else 7.4))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.6 if field else 1.35, 1], hspace=0.3, wspace=0.05)
        values = []
        if not field:
            ks = D.rows_by_track[p['id']]
            j = ks[np.argmin(np.abs(D.F[ks] - p['frame']))]
            cx, cy = D.cx[j], D.cy[j]
            r = int(max(70, 1.6 * D.major[j]))
            box = (int(max(0, cx - r)), int(max(0, cy - r)), int(min(D.W, cx + r)), int(min(D.H, cy + r)))
        else:
            box = field_box(D, fs[0], fs[2])
        for c_, f in enumerate(fs):
            ax = fig.add_subplot(gs[0, c_])
            im = frames.get(f)
            if im is not None:
                im = draw_regions(D, im, f, focus_ids={p.get('id')}, thick=2 if field else 1)
                x0, y0, x1, y1 = box
                ax.imshow(cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), extent=(x0, x1, y1, y0))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if not field:
                s_, m_ = srow.get((f, p['id'])), mrow.get((f, p['id']))
                if s_ and m_:
                    txt = (f"area {s_['area_px2']:.0f} px², aspect {s_['aspect_ratio']:.2f}\n"
                           f"speed {m_['speed_px_s']:.0f} px/s, shape: {s_['shape_quality']}")
                else:
                    txt = 'not in this trusted segment'
                values.append(dict(frame=f, area_px2=s_ and s_['area_px2'], aspect_ratio=s_ and s_['aspect_ratio'],
                                   speed_px_s=m_ and m_['speed_px_s'], shape_quality=s_ and s_['shape_quality']))
            else:
                c = crow[f]
                txt = (f"{c['n_regions']} regions, clusters hold {c['cluster_area_share'] * 100:.0f} % of the area\n"
                       f"spread {c['spread_px']:.0f} px, total area {c['total_area_px2']:.0f} px²")
                values.append(dict(frame=f, n_regions=c['n_regions'], total_area_px2=c['total_area_px2'],
                                   cluster_area_share=c['cluster_area_share'], spread_px=c['spread_px']))
            ax.set_title(f'frame {f}  ·  {f / D.fps:.3f} s', fontsize=9, loc='left')
            ax.text(0.0, -0.03, txt, transform=ax.transAxes, va='top', fontsize=8, color=INK2)
        lo, hi = p['frame'] - 6 * p['half'], p['frame'] + 6 * p['half']
        if not field:
            keys = [('area_px2', 'visible area (px²)', shape), ('speed_px_s', 'speed (px/s)', motion)]
            for c_, (key, lab, rows) in enumerate(keys):
                ax = fig.add_subplot(gs[1, c_])
                rr = [r for r in rows if r['object_id'] == p['object_id'] and lo <= r['frame'] <= hi]
                for sgm, g in by_segment(rr).items():
                    ax.plot(col(g, 't_s'), col(g, key), color=SLOTS[0], marker='.', ms=3)
                for f in fs:
                    ax.axvline(f / D.fps, color=SLOTS[1], lw=0.9, ls='--')
                ax.set_title(lab, fontsize=9)
                ax.set_xlabel('time (s)')
        else:
            other = 'spread_px' if p['key'] != 'spread_px' else 'n_regions'
            for c_, key in enumerate((p['key'], other)):
                ax = fig.add_subplot(gs[1, c_])
                rr = [r for r in coll if lo <= r['frame'] <= hi]
                ax.plot(col(rr, 't_s'), col(rr, key), color=SLOTS[0])
                for f in fs:
                    ax.axvline(f / D.fps, color=SLOTS[1], lw=0.9, ls='--')
                for c in conns:
                    if c['type'] in ('enter view', 'leave view') and lo <= c['frame'] <= hi and c['counted_in_field'] \
                            and (c.get('region_max_area_px2') or 0) >= D.P['cluster_area_px']:
                        ax.axvline(c['t_s'], color=MUTED, lw=1.2, ls=':')
                ax.set_title(key.replace('_px2', ' (px²)').replace('_px', ' (px)').replace('_', ' '), fontsize=9)
                ax.set_xlabel('time (s)')
        axn = fig.add_subplot(gs[1, 2])
        axn.axis('off')
        head = f"Object #{p['object_id']} (track #{p['id']})" if not field else 'Whole field'
        axn.text(0, 0.98, f'Check {k}: {head}', fontsize=10, fontweight='bold', va='top')
        axn.text(0, 0.84, p['what'], fontsize=8.5, va='top', color=INK2, wrap=True)
        extra = ''
        if field and p.get('field_of_view'):
            extra = 'Region entering / leaving the view near this moment\n(dotted lines): ' + ', '.join(p['field_of_view'][:3])
        axn.text(0, 0.60, ('Yellow outline = the region in question; other outlines: ' if not field else
                           'Close-up of the regions that changed most. Outlines: ')
                 + 'green = trusted, orange = uncertain, red = false target.'
                 + '\nDashed lines in the curves = the three frames above.' + ('\n\n' + extra if extra else ''),
                 fontsize=8, va='top', color=MUTED)
        path = save(fig, os.path.join(out, 'checks', f'check_{k:02d}.png'))
        items.append(dict(file=os.path.relpath(path, out), what=p['what'], kind=p['kind'],
                          object_id=p.get('object_id'), id=p.get('id'), frames=fs, values=values,
                          field_of_view_events=p.get('field_of_view', [])))
    return items
