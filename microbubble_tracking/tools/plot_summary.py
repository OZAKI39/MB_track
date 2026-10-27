#!/usr/bin/env python3
"""
从一组结果（objects.csv + background.npy）画出主要微团的轨迹、面积和圆度随时间的变化。

用法：
    python tools/plot_summary.py results/B_offline --title "实验 B" --out B_summary.png

“主要微团”= 出现 ≥ 100 帧且面积中位数 ≥ 500 像素的轨迹，最多取出现时间最长的 6 个。
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager as fm
import numpy as np

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, INK2, GRID, SURF = '#1f1f1e', '#5f5e5a', '#e4e3df', '#fcfcfb'


def use_cjk_font():
    """尽量找一个能显示中文的字体（Linux / Windows / macOS 常见字体）。"""
    for path in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc'):
        if os.path.exists(path):
            fm.fontManager.addfont(path)
    names = {f.name for f in fm.fontManager.ttflist}
    for name in ('Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti SC',
                 'Source Han Sans SC', 'WenQuanYi Micro Hei'):
        if name in names:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False


def smooth(y, k):
    y = np.asarray(y, float)
    return np.array([np.median(y[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(y))])


def place_labels(ax, ends):
    """在线条末端直接标注编号，彼此太近时向上错开。"""
    lo, hi = ax.get_ylim()
    gap, ys = (hi - lo) * 0.06, []
    for y, t, i in sorted(ends):
        y2 = max(y, ys[-1] + gap) if ys else y
        ys.append(y2)
        ax.annotate(f'#{i}', (t, y), xytext=(t + 0.25, y2), textcoords='data', fontsize=10,
                    color=INK, va='center', annotation_clip=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result_dir')
    ap.add_argument('--title', default='主要微团的位移与形状')
    ap.add_argument('--out', default='summary.png')
    a = ap.parse_args()
    use_cjk_font()
    tr = defaultdict(list)
    with open(os.path.join(a.result_dir, 'objects.csv')) as fh:
        for r in csv.DictReader(fh):
            tr[int(r['id'])].append({k: float(v) for k, v in r.items()})
    main_ids = [i for i, v in tr.items() if len(v) >= 100 and np.median([r['area'] for r in v]) >= 500]
    main_ids = sorted(sorted(main_ids, key=lambda i: -len(tr[i]))[:6], key=lambda i: tr[i][0]['frame'])
    bg = np.load(os.path.join(a.result_dir, 'background.npy'))
    t_max = max(r['t'] for v in tr.values() for r in v)

    fig = plt.figure(figsize=(15, 6.4), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.35, wspace=0.12,
                          left=0.03, right=0.95, top=0.88, bottom=0.09)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax0.imshow(bg, cmap='gray', vmin=60, vmax=230, alpha=0.55)
    halo = [pe.Stroke(linewidth=4, foreground='white', alpha=0.85), pe.Normal()]
    ends1, ends2 = [], []
    for c, i in zip(PAL, main_ids):
        v = tr[i]
        x, y = [r['cx'] for r in v], [r['cy'] for r in v]
        t = np.array([r['t'] for r in v])
        ax0.plot(x, y, color=c, lw=2, path_effects=halo, solid_capstyle='round')
        ax0.plot(x[0], y[0], 'o', ms=8, mfc='white', mec=c, mew=2)
        ax0.plot(x[-1], y[-1], 'o', ms=8, color=c, mec='white', mew=1.5)
        ax0.annotate(f'#{i}', (x[-1], y[-1]), xytext=(7, -7), textcoords='offset points', fontsize=11, color=INK,
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
        area = smooth([r['area'] for r in v], 5)
        circ = smooth([r['circularity'] for r in v], 31)
        ax1.plot(t, area, color=c, lw=2)
        ax2.plot(t, circ, color=c, lw=2)
        ends1.append((area[-1], t[-1], i))
        ends2.append((circ[-1], t[-1], i))
    ax1.set_ylim(0, None)
    ax2.set_xlim(0, t_max + 0.5)
    place_labels(ax1, ends1)
    place_labels(ax2, ends2)
    # 只显示轨迹经过的区域
    if main_ids:
        xs = np.concatenate([[r['cx'] for r in tr[i]] for i in main_ids])
        x0, x1 = max(0, xs.min() - 150), min(bg.shape[1], xs.max() + 150)
        if x1 - x0 < 0.8 * bg.shape[0]:
            mid = (x0 + x1) / 2
            x0, x1 = max(0, mid - 0.4 * bg.shape[0]), min(bg.shape[1], mid + 0.4 * bg.shape[0])
        ax0.set_xlim(x0, x1)
    ax0.set_ylim(bg.shape[0], 0)
    ax0.set_xticks([])
    ax0.set_yticks([])
    for s in ax0.spines.values():
        s.set_visible(False)
    ax0.set_title('主要微团的轨迹（空心=起点，实心=终点）', fontsize=12, color=INK, loc='left')
    for ax, yl in ((ax1, '面积（像素）'), (ax2, '圆度 4πA/P²（滑动中位数）')):
        ax.set_facecolor(SURF)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color('#b9b8b2')
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_ylabel(yl, color=INK2, fontsize=10)
    ax2.set_xlabel('时间（s）', color=INK2, fontsize=10)
    ax1.set_title('团的面积随时间变化（台阶 = 融合或分裂）', fontsize=12, color=INK, loc='left')
    ax2.set_title('团的形状随时间变化', fontsize=12, color=INK, loc='left')
    fig.legend(handles=[plt.Line2D([], [], color=c, lw=2, label=f'#{i}') for c, i in zip(PAL, main_ids)],
               loc='upper right', ncol=max(1, len(main_ids)), frameon=False, fontsize=10,
               bbox_to_anchor=(0.95, 0.985))
    fig.suptitle(a.title, x=0.03, y=0.975, ha='left', fontsize=14, color=INK)
    fig.savefig(a.out, dpi=130, facecolor=SURF)
    print(a.out, 'main clusters:', main_ids)


if __name__ == '__main__':
    main()
