"""Compare exact Windows CPU slices with slow and neighboring analysis frames."""
import collections
import json
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parent
if '--quiet' in sys.argv:
    OUT=OUT/'quiet'
elif '--paused' in sys.argv:
    OUT=OUT/'paused'
mapping=json.loads((OUT/'clock_mapping.json').read_text())
result=[]
for window in mapping['windows']:
    lo=window['start_etw_s']*1e6
    hi=window['end_etw_s']*1e6
    rows=json.loads((OUT/window['file']).read_text())
    running={}
    used=collections.Counter()
    per_cpu=collections.Counter()
    faults=[]
    ready={}
    waits=[]
    for row in rows:
        stamp=int(row[1])
        if row[0]=='CSwitch':
            cpu=int(row[16])
            if cpu in running:
                start,process,tid=running[cpu]
                if int(row[9])==tid:
                    overlap=max(0,min(hi,stamp)-max(lo,start))/1000
                    used[process]+=overlap
                    per_cpu[cpu]+=overlap
            running[cpu]=(stamp,row[2],int(row[3]))
            if 'vmmem' in row[8] and row[12]=='Ready':
                ready[int(row[9])]=(stamp,row[13],row[2])
            if 'vmmem' in row[2] and int(row[3]) in ready:
                start,reason,next_process=ready.pop(int(row[3]))
                if start<hi and stamp>lo:
                    waits.append(dict(tid=int(row[3]),start_us=start,end_us=stamp,wait_ms=(stamp-start)/1000,reason=reason,next_process=next_process))
        if row[0] in ('HardFault','InMemoryStoreFault') and lo<=stamp<=hi:
            faults.append(row)
    duration=(hi-lo)/1000
    main=next((x for x in (OUT/f'native_{window["label"]}_inspection.json').exists() and json.loads((OUT/f'native_{window["label"]}_inspection.json').read_text())['slow'] or [] if x['frame']==window['frame']),None)
    result.append(dict(label=window['label'],frame=window['frame'],duration_ms=duration,
                       process_cpu_ms=dict(used.most_common()),
                       code_cpu_ms=sum(v for k,v in used.items() if k.startswith('Code.exe')),
                       sampled_profile_counts=dict(collections.Counter(row[2] for row in rows if row[0]=='SampledProfile' and lo<=int(row[1])<=hi)),
                       faults=faults,vmmem_ready_waits=sorted(waits,key=lambda x:-x['wait_ms'])[:10],
                       linux_thread_cpu_ms=main['thread_cpu_ms'] if main else None))
(OUT/'host_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
for row in result:
 print(row['label'],row['frame'],'frame_ms',round(row['duration_ms'],3),'Code_CPU_ms',round(row['code_cpu_ms'],3),'top',list(row['process_cpu_ms'].items())[:4], 'fault_events',len(row['faults']))
