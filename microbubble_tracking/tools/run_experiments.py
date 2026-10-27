#!/usr/bin/env python3
"""
复现本项目 results/ 下的全部结果。

在项目根目录运行：
    python tools/run_experiments.py
可选：
    --raw-dir data/raw   原始视频所在目录（需包含 expA.mp4、expB.mp4）
    --out results        输出目录
    --no-render          不生成叠加视频（快很多）

共 7 组运行：
  A_offline_noref / B_offline_noref   每段视频只用自己估计背景（离线）
  A_offline / B_offline               离线，并用另一段实验的背景合并（推荐的离线结果）
  A_online / B_online                 实时模式，参考背景来自另一段实验（模拟“先有参考背景”）
  B_online_firstframe                 实时模式，不给参考背景（对照组，展示为什么需要参考背景）
之后自动运行 compare_runs.py 和 plot_summary.py。
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKER = os.path.join(ROOT, 'bubble_tracker.py')
TOOLS = os.path.join(ROOT, 'tools')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', default=os.path.join(ROOT, 'data', 'raw'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--no-render', action='store_true')
    a = ap.parse_args()
    A = os.path.join(a.raw_dir, 'expA.mp4')
    B = os.path.join(a.raw_dir, 'expB.mp4')
    R = lambda name: os.path.join(a.out, name)
    bgA = os.path.join(R('A_offline_noref'), 'background.npy')
    bgB = os.path.join(R('B_offline_noref'), 'background.npy')
    render = [] if a.no_render else ['--render']
    runs = [
        ('A_offline_noref', [A]),
        ('B_offline_noref', [B]),
        ('A_offline', [A, '--ref', bgB, '--save-contours'] + render),
        ('B_offline', [B, '--ref', bgA, '--save-contours'] + render),
        ('A_online', [A, '--bg', 'online', '--ref', bgB, '--save-contours'] + render),
        ('B_online', [B, '--bg', 'online', '--ref', bgA, '--save-contours'] + render),
        ('B_online_firstframe', [B, '--bg', 'online'] + render),
    ]
    os.makedirs(a.out, exist_ok=True)
    for name, args in runs:
        t0 = time.time()
        print(f'=== {name}', flush=True)
        subprocess.run([sys.executable, TRACKER] + args + ['--out', R(name)], check=True,
                       stdout=subprocess.DEVNULL)
        print(f'    done in {time.time() - t0:.0f} s', flush=True)
    # 本次装片的最佳参考背景（A、B 两段合并、对齐到 B）
    shutil.copy(os.path.join(R('B_offline'), 'background.npy'), R('session_background.npy'))
    subprocess.run([sys.executable, os.path.join(TOOLS, 'compare_runs.py'), a.out], check=True)
    fig = os.path.join(a.out, 'figures')
    os.makedirs(fig, exist_ok=True)
    for exp in 'AB':
        subprocess.run([sys.executable, os.path.join(TOOLS, 'plot_summary.py'), R(f'{exp}_offline'),
                        '--title', f'实验 {exp}：主要微团的位移与形状（离线结果）',
                        '--out', os.path.join(fig, f'exp{exp}_summary.png')], check=True)


if __name__ == '__main__':
    main()
