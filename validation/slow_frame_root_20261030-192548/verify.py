"""Check persistence and unchanged scientific results for the diagnostic replays."""
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent
BASELINE=OUT.parent/'paced_five_retests_20261030-191557'


def fingerprint(path, tables=None, limit=None):
    result={}
    with sqlite3.connect(f'file:{path / "tracking.sqlite"}?mode=ro',uri=True) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
        for table in tables or ('frames','observations','tracks','events','track_updates','event_updates','input_rejections'):
            columns=[r[1] for r in db.execute(f'PRAGMA table_info({table})') if r[1]!='processing_ms']
            order='source_frame,track_id' if table=='observations' else columns[0]
            condition=f' WHERE source_frame < {int(limit)}' if limit is not None else ''
            digest=hashlib.sha256(); count=0
            for row in db.execute(f'SELECT {",".join(columns)} FROM {table}{condition} ORDER BY {order}'):
                digest.update(repr(row).encode()); digest.update(b'\n'); count+=1
            result[table]={'rows':count,'sha256':digest.hexdigest()}
    return result


def main():
    names=['native_A','native_B','native_host24','native_host1','quiet/native_host24','quiet/native_hostB']
    if (OUT/'paused/native_hostB/run_state.json').exists():
        names+=['paused/native_host24','paused/native_hostB']
    report={'runs':[],'comparisons':{}}
    for name in names:
        path=OUT/name
        run=json.loads((path/'paced_report.json').read_text())
        writer=json.loads((path/'writer_report.json').read_text())
        assert run['status']=='completed' and run['all_offered_frames_accounted']
        assert writer['worker_finished'] and writer['all_submitted_saved'] and writer['pending_records']==0 and not writer['uncommitted']
        fp=fingerprint(path)
        assert fp['frames']['rows']==run['processed_frames']==writer['committed_frames']
        timing=list(csv.DictReader((path/'consumer_timing.csv').open()))
        gc=list(csv.DictReader((path/'consumer_stalls.gc.csv').open()))
        lifecycle=list(csv.DictReader((path/'frame_outcomes.csv').open()))
        item=dict(name=name,offered=run['offered_frames'],processed=run['processed_frames'],saved=writer['committed_frames'],
                  process_ms=run['metrics']['process_ms'],slow_frames=sum(float(r['process_ms'])>run['frame_budget_ms'] for r in timing),
                  gaps=[r for r in lifecycle if r['outcome']!='processed'],
                  gc_max_ms=max(float(r['wall_ms']) for r in gc if int(r['frame'])>=0),
                  queue_peak=writer['peak_pending_records'],rss_mib=run['memory_sampled_mb']['consumer_rss_mb'],fingerprints=fp)
        report['runs'].append(item)
    comparisons=report['comparisons']
    comparisons['single_thread_vs_24_threads_full_3960_frames']=fingerprint(OUT/'native_A')==fingerprint(OUT/'native_host1')
    comparisons['quiet_A_first_2640_frames']=fingerprint(OUT/'native_A',('frames','observations'),2640)==fingerprint(OUT/'quiet/native_host24',('frames','observations'))
    comparisons['quiet_B_vs_previous_formal_run']=fingerprint(BASELINE/'expB_1')==fingerprint(OUT/'quiet/native_hostB')
    if (OUT/'paused/native_hostB/run_state.json').exists():
        for label in ('host24','hostB'):
            comparisons['paused_vs_quiet_'+label]=fingerprint(OUT/'quiet'/('native_'+label))==fingerprint(OUT/'paused'/('native_'+label))
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))
    print('Processed/saved',sum(r['processed'] for r in report['runs']),'gaps',sum(len(r['gaps']) for r in report['runs']))
    assert all(comparisons.values()), 'Scientific results differ; inspect verification.json'


if __name__=='__main__':
    main()
