"""
compare_figures.py - figures for baseline/compare.py

figures/masks.png              the channel masks (MobileSAM as in the original procedure, ideal) and prompts
figures/detection_sweep.png    what the baseline finds for each threshold: share of the human-checked
                               regions found (by size), and extra regions per frame (walls, debris, other)
figures/detection_examples.png two frames at the original threshold and at the tuned one
figures/tracking.png           single-target tracking: per object, where the baseline tracker is in every
                               frame, next to what the review found for our method
figures/tracking_examples.png  the first frame in which the baseline left each object
"""
import os
from collections import defaultdict

import cv2
import numpy as np

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'analysis'))
import figures as F  # noqa: E402
from figures import plt, SLOTS, INK2, MUTED  # noqa: E402

STATE_COLOR = {'on target': '#1baf7a', 'other object': '#e34948', 'false target': '#e87ba4',
               'not on a region': '#eda100', 'lost': '#898781'}
AFTER = '#6250d6'
BLOB_BGR = {'matched': (122, 175, 27), 'wall': (72, 73, 227), 'false target': (164, 123, 232), 'other': (0, 161, 237)}


def save(fig, out, name):
    return F.save(fig, os.path.join(out, 'figures', name))


def fig_masks(roi, masks, mask_dir, out):
    import channel_mask as CM
    crop = cv2.imread(os.path.join(mask_dir, 'first_frame_roi.png'))
    thr = cv2.imread(os.path.join(mask_dir, 'thresholded_prompt_image.png'))
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2))
    axs[0].imshow(cv2.cvtColor(thr, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Adaptive threshold + flood fill at the prompt points (input to SAM)', fontsize=9)
    axs[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    for (name, m), c, ls in zip(masks.items(), (SLOTS[0], SLOTS[2]), ('-', '--')):
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for k, cc in enumerate(cs):
            cc = cc.reshape(-1, 2)
            axs[1].plot(np.r_[cc[:, 0], cc[0, 0]], np.r_[cc[:, 1], cc[0, 1]], ls, color=c, lw=1.4,
                        label={'mobilesam': 'MobileSAM mask', 'ideal': 'ideal mask'}.get(name, name) if k == 0 else None)
    fg, bg = np.array(CM.to_roi(CM.FOREGROUND)), np.array(CM.to_roi(CM.BACKGROUND))
    axs[1].plot(fg[:, 0], fg[:, 1], 'o', ms=5, color=SLOTS[2], mec='white', label='foreground prompts')
    axs[1].plot(bg[:, 0], bg[:, 1], 'X', ms=6, color=SLOTS[7], mec='white', label='background prompts')
    axs[1].legend(loc='upper left', fontsize=7, frameon=True)
    axs[1].set_title('Channel masks on the first frame (square ROI)', fontsize=9)
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    fig.tight_layout()
    return save(fig, out, 'masks.png')


def fig_sweep(det, tuned, out):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for mname, ls in (('mobilesam', '-'), ('ideal', '--')):
        rs = sorted([r for r in det if r['mask'] == mname], key=lambda r: r['threshold'])
        t = [r['threshold'] for r in rs]
        for k, (key, lab) in enumerate((('recall_single', 'single bubbles (<100 px²)'),
                                        ('recall_small_group', 'small groups (100-300 px²)'),
                                        ('recall_cluster', 'clusters (≥300 px²)'))):
            axs[0].plot(t, [r[key] for r in rs], ls, color=SLOTS[k], lw=1.8, label=f'{lab}' if mname == 'mobilesam' else None)
        for k, (key, lab) in enumerate((('blobs_per_frame_wall', 'on a wall or a fixed dark spot'),
                                        ('blobs_per_frame_false_target', 'on a false target (debris)'),
                                        ('blobs_per_frame_other', 'elsewhere'),
                                        ('blobs_per_frame_matched', 'on a real bubble region'))):
            axs[1].plot(t, [r[key] for r in rs], ls, color=[SLOTS[7], SLOTS[4], SLOTS[3], SLOTS[2]][k], lw=1.8,
                        label=lab if mname == 'mobilesam' else None)
    for a in axs:
        a.axvline(100, color=INK2, lw=0.9, ls=':')
        a.axvline(tuned, color=INK2, lw=0.9, ls='-.')
        a.set_xlabel('threshold (grey level, 0-255)')
    for a in axs:
        a.text(100, 0.98, ' original 100', fontsize=8, color=INK2, transform=a.get_xaxis_transform(), va='top')
        a.text(tuned, 0.98, f' tuned {tuned}', fontsize=8, color=INK2, transform=a.get_xaxis_transform(), va='top')
    axs[0].set_ylim(0, 1.05)
    axs[0].set_ylabel('share of human-checked regions found')
    axs[0].set_title('Baseline detection: which real bubble regions it finds', fontsize=10)
    axs[0].legend(loc='lower right', fontsize=8)
    axs[1].set_yscale('symlog', linthresh=1)
    axs[1].set_ylim(0, 500)
    axs[1].set_ylabel('blue regions per frame (≥5 px)')
    axs[1].set_title('Baseline detection: where its blue regions are', fontsize=10)
    axs[1].legend(loc='upper left', bbox_to_anchor=(0, 0.92), fontsize=8)
    fig.text(0.01, -0.02, 'Solid: MobileSAM channel mask (original procedure). Dashed: ideal channel mask. '
             'Our method, after review, keeps every human-checked region by construction; 7 % of its '
             'region-frames were marked as false targets in the review.', fontsize=8, color=MUTED)
    fig.tight_layout()
    return save(fig, out, 'detection_sweep.png')


def fig_examples(D, examples, roi, tuned, out):
    keys = sorted(examples)
    if not keys:
        return None
    frames = sorted({f for _, f in keys})[:2]
    thrs = sorted({t for t, _ in keys})
    fig, axs = plt.subplots(len(frames), len(thrs), figsize=(6.2 * len(thrs), 6.2 * len(frames)), squeeze=False)
    x0, y0 = roi[0], roi[1]
    for i, f in enumerate(frames):
        for j, t in enumerate(thrs):
            ax = axs[i, j]
            if (t, f) not in examples:
                ax.axis('off')
                continue
            crop, bm, cl, cat = examples[(t, f)]
            img = crop.copy()
            for comp, c in cat.items():
                img[cl == comp] = BLOB_BGR[c]
            # human-checked regions the baseline missed: yellow outline
            for k in D.rows_by_frame.get(f, []):
                if D.role[k] == 'excluded':
                    continue
                c = D.contour.get((f, int(D.I[k])))
                if c is None:
                    continue
                c = (c.reshape(-1, 2) - [x0, y0]).astype(np.int32)
                m = np.zeros(bm.shape, np.uint8)
                cv2.fillPoly(m, [c.reshape(-1, 1, 2)], 1)
                if (bm & (m > 0)).sum() < 3:
                    cv2.polylines(img, [c.reshape(-1, 1, 2)], True, (40, 220, 255), 2)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f'frame {f} · threshold {t}' + (' (original)' if t == 100 else ' (tuned)' if t == tuned else ''),
                         fontsize=9, loc='left')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    fig.text(0.01, 0.003, 'Baseline blue pixels: green = on a real bubble region, red = on a wall or a fixed dark spot, '
             'pink = on a false target, orange = elsewhere. Yellow outline = human-checked region the baseline missed.',
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return save(fig, out, 'detection_examples.png')


def fig_tracking(D, plan, trk, ours, out):
    by = defaultdict(list)
    for p in plan:
        by[p['oid']].append(p)
    oids = list(by)
    cfgs = list(dict.fromkeys(p['cfg'] for p in plan))
    T = D.n_frames / D.fps
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(oids) * (len(cfgs) + 1) + 1.6))
    y = 0
    yt, yl = [], []
    for oid in oids:
        ks = np.nonzero(D.object_id == oid)[0]
        # ours: trusted / untrusted / not detected over the object's lifetime
        f0, f1 = int(D.F[ks].min()), int(D.F[ks].max())
        st = {int(D.F[k]): D.status[k] for k in ks}
        for f in range(f0, f1 + 1):
            c = {'trusted': '#1baf7a', 'untrusted': '#e34948'}.get(st.get(f), '#d8d6cf')
            ax.add_patch(plt.Rectangle((f / D.fps, y - 0.35), 1 / D.fps, 0.7, color=c, lw=0))
        yt.append(y)
        yl.append(f'#{oid} ours (after review)')
        y -= 1
        for p in by[oid]:
            for r in p['out']:
                if r['phase'] == 'target present':
                    c = STATE_COLOR[r['state']]
                elif r['phase'] == 'after target':
                    c = AFTER if r['ok'] else '#d8d6cf'
                else:
                    c = '#d8d6cf'
                ax.add_patch(plt.Rectangle((r['frame'] / D.fps, y - 0.35), 1 / D.fps, 0.7, color=c, lw=0))
            yt.append(y)
            yl.append(f"#{oid} baseline · {p['cfg'][0]} · thr {p['cfg'][1]}")
            y -= 1
        y -= 0.4
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=8)
    ax.set_ylim(y + 0.2, 0.8)
    ax.set_xlim(0, T)
    ax.set_xlabel('time (s)')
    ax.grid(axis='y', visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in STATE_COLOR.items()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=AFTER, label='object has ended,\nbaseline still reports\na position'))
    handles += [plt.Rectangle((0, 0), 1, 1, color='#1baf7a', label='ours: trusted'),
                plt.Rectangle((0, 0), 1, 1, color='#e34948', label='ours: untrusted'),
                plt.Rectangle((0, 0), 1, 1, color='#d8d6cf', label='not detected / no reference')]
    ax.legend(handles=handles[:6] + handles[6:7], loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8,
              title='baseline is …', title_fontsize=8)
    ax.set_title('Single-target tracking of the main objects: baseline (3 settings) vs. ours after review', fontsize=10)
    fig.tight_layout()
    return save(fig, out, 'tracking.png')


def fig_track_examples(D, video, roi, plan, out):
    picks = []
    for p in plan:
        o = [r for r in p['out'] if r['phase'] == 'target present']
        r = next((r for r in o if r['state'] not in ('on target',)), None)
        if r:
            picks.append((p, r))
    if not picks:
        return None
    picks = picks[:8]
    frames = F.read_frames(video, [r['frame'] for _, r in picks])
    n = len(picks)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.5 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis('off')
    for ax, (p, r) in zip(axs.ravel(), picks):
        im = frames.get(r['frame'])
        if im is None:
            continue
        im = F.draw_regions(D, im, r['frame'], focus_ids=set(int(i) for i in D.I[np.nonzero(D.object_id == p['oid'])[0]]))
        cx, cy = (r['ref_x_px'], r['ref_y_px'])
        if r['x_px'] is not None:
            cv2.drawMarker(im, (int(r['x_px']), int(r['y_px'])), (60, 60, 230), cv2.MARKER_CROSS, 18, 3)
            cx, cy = (cx + r['x_px']) / 2, (cy + r['y_px']) / 2
        s = 110
        x0, y0 = int(max(0, cx - s)), int(max(0, cy - s))
        ax.imshow(cv2.cvtColor(im[y0:y0 + 2 * s, x0:x0 + 2 * s], cv2.COLOR_BGR2RGB))
        ax.set_title(f"#{p['oid']} · {p['cfg'][0]} thr {p['cfg'][1]}\nframe {r['frame']} ({r['t_s']:.2f} s): {r['state']}",
                     fontsize=8.5, loc='left')
    fig.text(0.01, 0.005, 'Yellow outline = the object (reference); red cross = baseline position.', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return save(fig, out, 'tracking_examples.png')


def make_all(D, video, roi, masks, det, tuned, plan, trk, ours, vc, examples, out, P, mask_dir):
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    files = [fig_masks(roi, masks, mask_dir, out), fig_sweep(det, tuned, out),
             fig_examples(D, examples, roi, tuned, out), fig_tracking(D, plan, trk, ours, out),
             fig_track_examples(D, video, roi, plan, out)]
    return [f for f in files if f]
