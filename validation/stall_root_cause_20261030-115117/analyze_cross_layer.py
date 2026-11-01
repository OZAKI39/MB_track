"""Preserve and compare measured Linux stages with Windows ETW events.

Times interpolated between request/response samples are estimates, not hardware sync.
The independent enclosing sample bounds are retained instead of claiming exact matching.
"""
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HOST = Path('/mnt/c/Users/lzy/AppData/Local/Temp/MB_track_stall_20261030-115117/capture')


def read_csv(path):
    with path.open(errors='replace') as file:
        return list(csv.DictReader(file))


def archive_events(source, target, names):
    with source.open(errors='replace') as file, target.open('x', newline='') as out:
        writer = csv.writer(out)
        for row in csv.reader(file, skipinitialspace=True):
            if row and row[0] in names and row[1].isdigit():
                writer.writerow(row)


for name in ['trace_header_raw.txt', 'header_events.txt']:
    shutil.copy2(HOST / name, HERE / name)
for source, target in [('B_window.txt', 'B_window_faults.csv'),
                       ('B_second_window.txt', 'B_second_window_faults.csv')]:
    archive_events(HOST / source, HERE / target, {'HardFault'})
archive_events(HOST / 'disk_events.txt', HERE / 'host_flushes.csv', {'DiskFlush'})

raw = (HERE / 'trace_header_raw.txt').read_text()
origin = int(re.search(r'#\s+0\s+#\s+(0x[0-9a-f]+)', raw).group(1), 16)
bridge = read_csv(HERE / 'clock_bridge.csv')[1:]  # exclude initial PowerShell startup
wsl = np.array([(float(r['wsl_before_s']) + float(r['wsl_after_s'])) / 2 for r in bridge])
etw = np.array([(int(r['windows_qpc']) - origin) / int(r['windows_frequency']) for r in bridge])
residual = [float((np.interp(wsl[i], wsl[[i-1, i+1]], etw[[i-1, i+1]]) - etw[i])*1000)
            for i in range(1, len(bridge)-1)]
result = {
    'etl_qpc_origin': origin,
    'etl_frequency_hz': 10000000,
    'rtt_ms_median_p99_max': np.percentile([float(r['roundtrip_ms']) for r in bridge], [50, 99, 100]).tolist(),
    'interpolation_residual_abs_ms_median_p99_max': np.percentile(np.abs(residual), [50, 99, 100]).tolist(),
    'clock_limit': 'Interpolation is estimated. Enclosing samples, local residuals and +/-15 ms sensitivity are saved; neither is a certified synchronization bound.',
    'conversions': [],
}
source = read_csv(HERE / 'repro_B/source_stalls.csv')
for frame, name in [(572, 'B_window_faults.csv'), (3363, 'B_second_window_faults.csv')]:
    row = next(r for r in source if r['frame'] == str(frame) and r['stage'] == 'convert_bgr')
    start, end = map(float, (row['start_s'], row['end_s']))
    lo, hi = np.interp([start, end], wsl, etw) * 1e6
    lower = int(np.searchsorted(wsl, start)) - 1
    upper = int(np.searchsorted(wsl, end))
    with (HERE / name).open() as file:
        faults = [r for r in csv.reader(file) if 'vmmemWSL' in r[2]]
    measurements = {}
    for padding in (0, 15000):
        selected = [r for r in faults if lo-padding <= int(r[1]) and int(r[1])-int(r[7]) <= hi+padding]
        per_thread = {}
        for r in selected:
            values = per_thread.setdefault(r[3], {'count': 0, 'bytes': 0, 'elapsed_us': 0})
            values['count'] += 1
            values['bytes'] += int(r[6], 16)
            values['elapsed_us'] += int(r[7])
        measurements[f'padding_{padding}_us'] = per_thread
    result['conversions'].append({
        'linux_stage': row,
        'estimated_etw_start_us': lo, 'estimated_etw_end_us': hi,
        'enclosing_etw_bracket_us': [etw[lower]*1e6, etw[upper]*1e6],
        'nearby_interpolation_residual_ms': residual[max(0,lower-2):upper+1],
        'hardfaults_by_host_thread': measurements,
    })

with (HERE / 'host_flushes.csv').open() as file:
    flushes = list(csv.reader(file))
result['host_flush_summary'] = {}
for disk in ('0', '1'):
    rows = [r for r in flushes if r[7] == disk]
    result['host_flush_summary'][disk] = {'count': len(rows), 'max_elapsed_us': max(int(r[6]) for r in rows)}
with (HERE / 'cross_layer_summary.json').open('x') as file:
    json.dump(result, file, indent=2)
    file.write('\n')
print(json.dumps(result, indent=2))
