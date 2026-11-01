"""Export short ETW windows using measured Windows/WSL clock correspondence."""
import bisect
import csv
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE=Path(__file__).resolve().parent
QUIET='--quiet' in sys.argv
PAUSED='--paused' in sys.argv
PHASE='paused' if PAUSED else 'quiet' if QUIET else ''
OUT=BASE/PHASE if PHASE else BASE
HOST=Path('/mnt/c/Users/lzy/AppData/Local/Temp')/(BASE.name+('_'+PHASE if PHASE else ''))
ETL=HOST/'capture/host.etl'
XPERF='/mnt/c/Program Files (x86)/Windows Kits/10/Windows Performance Toolkit/xperf.exe'


def win(path):
    return subprocess.check_output(['wslpath','-w',str(path)],text=True).strip()


def main():
    b=ETL.open('rb').read(512)
    origin=struct.unpack_from('<Q',b,88)[0]
    freq=struct.unpack_from('<Q',b,360)[0]
    start_filetime=struct.unpack_from('<Q',b,368)[0]
    assert freq==10000000 and 10**12 < origin < 10**14
    clocks=[json.loads(line) for line in (OUT/'clock_bridge.jsonl').open()]
    clocks=[r for r in clocks if r['receive_s']-r['send_s']<.004]
    x=np.array([(r['send_s']+r['receive_s'])/2 for r in clocks])
    y=np.array([(r['qpc']-origin)/freq for r in clocks])
    assert all(r['frequency']==freq for r in clocks)
    # ETL FILETIME and first-event QPC must describe the same trace start.
    errors=[(r['unix_s']-(start_filetime/1e7-11644473600))-v for r,v in zip(clocks,y)]
    assert abs(np.median(errors))<.05, errors[:5]
    residuals=y[1:-1]-np.interp(x[1:-1],x[::2],y[::2])
    report={'origin_qpc':origin,'frequency':freq,'trace_start_filetime':start_filetime,
            'clock_unix_check_median_s':float(np.median(errors)),
            'rtt_ms':list(map(float,np.percentile([(r['receive_s']-r['send_s'])*1000 for r in clocks],[50,99,100]))),
            'windows':[]}
    selected={'host24':[2503,1972,1171,2502,1973,1320,1321], 'host1':[2277,1436,2276,2278]}
    if QUIET:
        selected={'host24':[576,367,368,369,577], 'hostB':[326,327,325,328]}
    if PAUSED:
        selected={'host24':[2488,367], 'hostB':[159,326,327]}
    for label,indices in selected.items():
        rows=list(csv.DictReader((OUT/f'native_{label}/consumer_stalls.csv').open()))
        groups={i:[r for r in rows if int(r['frame'])==i] for i in indices}
        for index,stages in groups.items():
            if not stages: continue
            start=float(stages[0]['start_s']); end=float(stages[-1]['end_s'])
            a,z=np.interp([start,end],x,y)
            name=f'etw_{label}_{index}.txt'
            command=[XPERF,'-i',win(ETL),'-o',win(HOST/name),'-a','dumper','-range',str(max(0,int((a-.025)*1e6))),str(int((z+.025)*1e6))]
            with (OUT/(name+'.log')).open('a') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=60)
            if result.returncode: raise RuntimeError(command)
            data=[]
            for r in csv.reader((HOST/name).open(errors='replace')):
                r=[v.strip() for v in r]
                if len(r)>1 and r[1].isdigit() and (a-.025)*1e6 <= int(r[1]) <= (z+.025)*1e6:
                    data.append(r)
            (OUT/(name+'.json')).write_text(json.dumps(data,separators=(',',':'))+'\n')
            j=bisect.bisect_left(x,start)
            local=clocks[max(0,j-2):j+3]
            report['windows'].append(dict(label=label,frame=index,start_s=start,end_s=end,start_etw_s=a,end_etw_s=z,stages=stages,clock_samples=local,file=name+'.json',command=command))
            print(label,index,len(data),flush=True)
    (OUT/'clock_mapping.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,args in [('trace_stats.txt',['-a','tracestats']),('trace_header.txt',['-a','dumper','-range','0','1000'])]:
        command=[XPERF,'-i',win(ETL),'-o',win(HOST/name),*args]
        with (OUT/(name+'.log')).open('a') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        (OUT/name).write_bytes((HOST/name).read_bytes())


if __name__=='__main__':
    main()
