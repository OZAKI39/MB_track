"""Bounded offline replay with existing native diagnostics; no instrument access."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/lzy/projects/new_microbubble_experiment')
OUT = Path(__file__).resolve().parent
CODE = ROOT / 'customer_workspace/microbubble_tracking'
PERF = '/usr/lib/linux-tools/6.18.54.1-microsoft-standard-WSL2/bin/perf'


def main():
    records = []
    for label in ('A', 'B'):
        run = OUT / ('native_' + label)
        observer = ['/mnt/c/Windows/System32/wsl.exe', '-d', 'Ubuntu', '-u', 'root', '--',
                    sys.executable, str(CODE / 'native_stall_watch.py'), str(run),
                    str(OUT / f'watch_{label}.jsonl'), '--duration', '85', '--interval', '0.005',
                    '--perf', PERF, '--perf-targets', 'consumer', 'writer', '--frequency', '999']
        replay = [sys.executable, '-m', 'customer_workspace.microbubble_tracking.paced_replay',
                  '--video', str(CODE / f'data/raw/exp{label}.mp4'),
                  '--reference', str(CODE / 'results/B_offline/background.npy'),
                  '--pzt', str(CODE / f'data/pzt/exp{label}'), '--output-dir', str(run),
                  '--repeats', '3', '--opencv-threads', '24', '--conversion-threads', '1',
                  '--measure-stages', '--diagnose-stalls']
        item = {'label': label, 'observer': observer, 'replay': replay, 'start_unix': time.time()}
        records.append(item)
        (OUT / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        print('START', label, flush=True)
        with (OUT / f'observer_{label}.log').open('x') as native_log, (OUT / f'replay_{label}.log').open('x') as replay_log:
            watch = subprocess.Popen(observer, stdout=native_log, stderr=subprocess.STDOUT)
            try:
                result = subprocess.run(replay, cwd=ROOT, stdout=replay_log, stderr=subprocess.STDOUT, timeout=85)
                item['replay_exit'] = result.returncode
            finally:
                item['observer_exit'] = watch.wait(timeout=25)
        item['finish_unix'] = time.time()
        (OUT / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        print('DONE', label, item['replay_exit'], item['observer_exit'], flush=True)
        if item['replay_exit'] or item['observer_exit']:
            raise RuntimeError('Replay or diagnostic capture failed')


if __name__ == '__main__':
    main()
