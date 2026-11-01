"""Recompute this diagnostic report from preserved CSV/JSONL records; no hardware I/O."""
import bisect
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent


def table(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def stats(values):
    return dict(zip(('median', 'p99', 'max', 'min'),
                    map(float, np.percentile(values, [50, 99, 100, 0]))))


runs = []
for run in sorted(BASE.iterdir()):
    if not (run / 'writer_report.json').exists():
        continue
    writer = json.loads((run / 'writer_report.json').read_text())
    source = json.loads((run / 'source_report.json').read_text())
    with sqlite3.connect((run / 'tracking.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        saved = db.execute('SELECT COUNT(*) FROM frames').fetchone()[0]
        integrity = db.execute('PRAGMA quick_check').fetchone()[0]
    assert writer['submitted_frames'] == writer['committed_frames'] == saved
    assert writer['all_submitted_saved'] and not writer['uncommitted']
    assert integrity == 'ok'
    result = {
        'run': run.name,
        'offered': source['offered_frames'],
        'saved': saved,
        'peak_pending': writer['peak_pending_records'],
        'all_submitted_saved': True,
        'sqlite_quick_check': integrity,
        'stages_ms': {},
        'worst_native_samples': [],
    }
    worst = []
    for role, stages in (
        ('source', ('convert_bgr', 'convert_gray', 'compute_gray',
                    'release_previous_gray', 'release_bgr')),
        ('writer', ('database_statements', 'database_commit')),
    ):
        records = table(run / f'{role}_stalls.csv')
        for stage in stages:
            selected = [r for r in records if r['stage'] == stage
                        and (role != 'source' or int(r['frame']) > 0)]
            if not selected:
                continue
            result['stages_ms'][stage] = stats([float(r['wall_ms']) for r in selected])
            for row in sorted(selected, key=lambda r: float(r['wall_ms']), reverse=True)[:3]:
                worst.append({'role': role, 'measurement': row, 'samples': []})
    watch = BASE / f'{run.name}_watch.jsonl'
    if watch.exists():
        commits = [r for r in table(run / 'writer_stalls.csv')
                   if r['stage'] == 'database_commit']
        starts = [float(r['start_s']) for r in commits]
        counts = Counter()
        with watch.open() as file:
            for line in file:
                sample = json.loads(line)
                now = sample['time_s']
                end = now + sample['sample_ms'] / 1000
                index = bisect.bisect_right(starts, now) - 1
                for thread in sample['threads']:
                    # Include a sample only if the whole sampling bracket is in a commit.
                    if thread['name'] == 'writer' and index >= 0 and end <= float(commits[index]['end_s']):
                        counts[thread['wchan']] += 1
                for event in worst:
                    row = event['measurement']
                    if float(row['start_s']) <= now and end <= float(row['end_s']):
                        event['samples'].extend(
                            dict(time_s=now, **t) for t in sample['threads']
                            if t['name'] == event['role']
                        )
        result['writer_wait_samples_inside_commit'] = dict(counts)
    result['worst_native_samples'] = worst
    runs.append(result)

host = {}
for name in ('host_counters_2.jsonl', 'host_after_close_series.jsonl'):
    records = [json.loads(line) for line in (BASE / name).read_text().splitlines()]
    available = [r['memory']['AvailableMBytes'] for r in records]
    rates = []
    for old, new in zip(records, records[1:]):
        dt = (new['start_unix_ms'] + new['end_unix_ms']
              - old['start_unix_ms'] - old['end_unix_ms']) / 2000
        pages = new['memory']['PagesInputPerSec'] - old['memory']['PagesInputPerSec']
        if dt > 0 and pages >= 0:
            rates.append(pages * 4096 / 2**20 / dt)
    host[name] = {'available_mb': stats(available), 'page_in_mib_s_approx': stats(rates),
                  'samples': len(records)}

payload = {'runs': runs, 'host': host,
           'notes': ['Source frame 0 excluded; writer includes all batches.',
                     'Host raw page-in counters differenced over own UTC sampling brackets; approximate.',
                     'repeat_A_3 and repeat_A_4 overlap the user action; exclude from before/after comparisons.',
                     'Native sampling and diagnostic probes add overhead; no formal throughput claim.']}
with (BASE / 'evidence_summary.json').open('x') as file:
    json.dump(payload, file, indent=2)
    file.write('\n')
with (BASE / 'comparison.csv').open('x', newline='') as file:
    fields = ['run', 'offered', 'saved', 'peak_pending', 'stage', 'median_ms', 'p99_ms', 'max_ms']
    writer = csv.DictWriter(file, fields)
    writer.writeheader()
    for run in runs:
        for stage, values in run['stages_ms'].items():
            writer.writerow({**{key: run[key] for key in fields[:4]}, 'stage': stage,
                             **{f'{key}_ms': values[key] for key in ('median', 'p99', 'max')}})
print(json.dumps({'host': host, 'saved_frames': sum(r['saved'] for r in runs),
                  'offered_frames': sum(r['offered'] for r in runs),
                  'runs': len(runs)}, indent=2))
for run in runs:
    print(run['run'], run.get('writer_wait_samples_inside_commit'))
