#!/usr/bin/env python3
"""
汇总并比较 results/ 下各组运行。

用法：python tools/compare_runs.py results

输出：
  results/run_summary.csv   每组运行的耗时、编号数、事件计数
  results/comparison.csv    各组相对“离线 + 参考背景”结果（视为基准）的一致性
  results/comparison.md     同上，Markdown 表格

一致性的算法：每 5 帧取一帧，把两组结果的目标按质心距离 < 8 像素配对。
  检出率 = 基准中的目标有多少在被测结果里找到；
  准确率 = 被测结果中的目标有多少在基准里找到。
小目标指面积 30–300 像素，团指面积 ≥ 300 像素。
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

RUNS = ['A_offline_noref', 'B_offline_noref', 'A_offline', 'B_offline', 'A_online', 'B_online', 'B_online_firstframe']
PAIRS = [('A_offline', 'A_online'), ('A_offline', 'A_offline_noref'),
         ('B_offline', 'B_online'), ('B_offline', 'B_offline_noref'), ('B_offline', 'B_online_firstframe')]


def load(path):
    rows = defaultdict(list)
    with open(os.path.join(path, 'objects.csv')) as fh:
        for r in csv.DictReader(fh):
            rows[int(r['frame'])].append((float(r['cx']), float(r['cy']), float(r['area'])))
    return rows


def match_rate(src, dst, frames, amin, amax, tol=8.0):
    """src 中面积在 [amin, amax) 的目标，有多少能在 dst（面积 ≥ amin/2）中找到。"""
    hits = []
    for f in frames:
        S = np.array([(x, y) for x, y, a in src.get(f, []) if amin <= a < amax])
        T = np.array([(x, y) for x, y, a in dst.get(f, []) if a >= amin / 2])
        if len(S) == 0:
            continue
        if len(T) == 0:
            hits.append(0.0)
            continue
        d = np.linalg.norm(S[:, None] - T[None], axis=2).min(1)
        hits.append(float(np.mean(d < tol)))
    return float(np.mean(hits)) if hits else float('nan')


def main(res):
    # 1. 运行汇总
    summary = []
    for name in RUNS:
        p = os.path.join(res, name, 'stats.json')
        if not os.path.exists(p):
            continue
        s = json.load(open(p, encoding='utf-8'))
        summary.append(dict(run=name, video=s['video'], mode=s['mode'], background=s['background'],
                            frames=s['frames'], ms_mean=s['ms_per_frame_mean'], ms_p95=s['ms_per_frame_p95'],
                            confirmed_ids=s['confirmed_ids'], **s['events']))
    with open(os.path.join(res, 'run_summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # 2. 一致性
    cache = {}
    get = lambda n: cache.setdefault(n, load(os.path.join(res, n)))
    comp = []
    for ref, test in PAIRS:
        if not (os.path.exists(os.path.join(res, ref)) and os.path.exists(os.path.join(res, test))):
            continue
        a, b = get(ref), get(test)
        frames = range(0, max(max(a), max(b)) + 1, 5)
        n_all = max(max(a), max(b)) + 1
        area = lambda d: np.mean([sum(x[2] for x in d.get(f, []) if x[2] >= 300) for f in range(n_all)])
        comp.append(dict(
            reference=ref, test=test,
            small_recall=round(match_rate(a, b, frames, 30, 300), 3),
            small_precision=round(match_rate(b, a, frames, 30, 300), 3),
            cluster_recall=round(match_rate(a, b, frames, 300, 1e12), 3),
            cluster_precision=round(match_rate(b, a, frames, 300, 1e12), 3),
            cluster_area_ref=round(area(a)), cluster_area_test=round(area(b))))
    with open(os.path.join(res, 'comparison.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(comp[0]))
        w.writeheader()
        w.writerows(comp)

    md = ['## 各组运行', '',
          '| 运行 | 模式 | 背景来源 | 平均 ms/帧 | P95 ms/帧 | 确认编号数 | 融合 | 分裂 |',
          '|---|---|---|---|---|---|---|---|']
    for s in summary:
        md.append(f"| {s['run']} | {s['mode']} | {s['background']} | {s['ms_mean']} | {s['ms_p95']} | "
                  f"{s['confirmed_ids']} | {s['merge']} | {s['split']} |")
    md += ['', '## 与基准（离线 + 参考背景）的一致性', '',
           '| 被测 | 基准 | 小目标检出率 | 小目标准确率 | 团检出率 | 团准确率 | 团总面积/帧（基准 → 被测） |',
           '|---|---|---|---|---|---|---|']
    for c in comp:
        md.append(f"| {c['test']} | {c['reference']} | {c['small_recall']:.1%} | {c['small_precision']:.1%} | "
                  f"{c['cluster_recall']:.1%} | {c['cluster_precision']:.1%} | "
                  f"{c['cluster_area_ref']} → {c['cluster_area_test']} |")
    open(os.path.join(res, 'comparison.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results')
