"""Summarize recorded stalls; keep stack samples only inside measured slow stages."""
import bisect
import collections
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
if '--quiet' in sys.argv:
    OUT = OUT / 'quiet'
elif '--paused' in sys.argv:
    OUT = OUT / 'paused'
PERF = '/home/lzy/projects/new_microbubble_experiment/customer_workspace/dependencies/native_perf/installed/usr/lib/linux-tools-6.8.0-147/perf'


def inspect(label):
    run = OUT / ('native_' + label)
    stages = list(csv.DictReader((run / 'consumer_stalls.csv').open()))
    groups = collections.defaultdict(list)
    for r in stages:
        groups[int(r['frame'])].append(r)
    frames = []
    for frame, rows in groups.items():
        item = {'frame': frame, 'start': float(rows[0]['start_s']), 'end': float(rows[-1]['end_s'])}
        for name in ('wall_ms','thread_cpu_ms','process_cpu_ms','minor_faults','major_faults','voluntary_switches','involuntary_switches'):
            item[name] = sum(float(r[name]) for r in rows)
        item['stages'] = rows
        frames.append(item)
    starts = [x['start'] for x in frames]
    slow = sorted(frames, key=lambda x:x['wall_ms'], reverse=True)[:30]
    slow_ids = {x['frame'] for x in slow}
    for x in slow:
        x['samples'] = []
    by_id = {x['frame']: x for x in slow}
    meta = json.loads((run / 'consumer_stalls.meta.json').read_text())
    perf = '/usr/lib/linux-tools/6.18.54.1-microsoft-standard-WSL2/bin/perf' if label in ('A', 'B') else PERF
    command = [perf, 'script', '-i', str(OUT / f'watch_{label}.data'), '--ns', '-F', 'tid,time,event,ip,sym,dso']
    cpus = collections.defaultdict(list)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=dict(os.environ, LD_LIBRARY_PATH='/home/lzy/projects/new_microbubble_experiment/customer_workspace/dependencies/native_perf/installed/usr/lib/x86_64-linux-gnu'))
    block = []
    def record(lines):
        if not lines: return
        header = lines[0].strip().split()
        if len(header) < 3 or not header[0].isdigit() or int(header[0]) != meta['native_thread_id']:
            return
        timestamp = float(header[1].rstrip(':'))
        index = bisect.bisect_right(starts,timestamp)-1
        if index<0 or timestamp>frames[index]['end']: return
        frame = frames[index]['frame']
        if frame in slow_ids:
            by_id[frame]['samples'].append('\n'.join(lines))
    for line in process.stdout:
        if line.strip(): block.append(line.rstrip())
        else: record(block); block=[]
    record(block)
    error = process.stderr.read()
    if process.wait(): raise RuntimeError(error)
    for line in (OUT / f'watch_{label}.jsonl').open():
        row = json.loads(line)
        index = bisect.bisect_right(starts,row['time_s'])-1
        if index < 0 or row['time_s'] > frames[index]['end']: continue
        for thread in row['threads']:
            if thread['name'] == 'consumer' and not thread['stat'].startswith('unavailable'):
                cpu = int(thread['stat'].split(') ',1)[1].split()[36])
                cpus[cpu].append(index)
    cpu_result = {}
    for cpu, indices in cpus.items():
        unique = sorted(set(indices))
        durations = sorted(frames[i]['wall_ms'] for i in unique)  # frame index continuous if no gaps
        cpu_result[cpu] = {'frames':len(unique), 'median':durations[len(durations)//2], 'max':max(durations),'slow_frames':sum(x>16.667 for x in durations)}
    result = {'slow':slow, 'cpu_distribution':cpu_result, 'frames':len(frames), 'over_budget':sum(x['wall_ms']>16.667 for x in frames)}
    (OUT / f'native_{label}_inspection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(label, 'frames',len(frames),'over',result['over_budget'])
    for x in slow[:10]:
        print(x['frame'],round(x['wall_ms'],3),round(x['thread_cpu_ms'],3),x['minor_faults'],x['major_faults'],x['voluntary_switches'],x['involuntary_switches'])
    print('CPU',cpu_result)


if __name__=='__main__':
    inspect(sys.argv[1])
