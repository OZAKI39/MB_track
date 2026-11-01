"""Bounded offline reproduction with existing diagnostic tools; no instrument access."""
import json
import subprocess
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CODE=ROOT/'customer_workspace/microbubble_tracking'
PYTHON=ROOT/'.venv/bin/python'
PERFROOT=ROOT/'customer_workspace/dependencies/native_perf/installed/usr'
PERF=PERFROOT/'lib/linux-tools-6.8.0-147/perf'
LIB=PERFROOT/'lib/x86_64-linux-gnu'
PREFIX=['/mnt/c/Windows/System32/wsl.exe','-d','Ubuntu','-u','root','--']
EVENTS=['block:block_rq_insert','block:block_rq_issue','block:block_rq_complete',
'ext4:ext4_sync_file_enter','ext4:ext4_sync_file_exit','jbd2:jbd2_start_commit','jbd2:jbd2_end_commit',
'migrate:mm_migrate_pages_start','migrate:mm_migrate_pages','huge_memory:mm_collapse_huge_page',
'compaction:mm_compaction_begin','compaction:mm_compaction_end',
'vmscan:mm_vmscan_direct_reclaim_begin','vmscan:mm_vmscan_direct_reclaim_end','hyperv:balloon_status']
command=PREFIX+['env',f'LD_LIBRARY_PATH={LIB}',str(PERF),'record','-a','--clockid','mono','-m','256','-o',str(HERE/'system.data')]
for event in EVENTS: command+=['-e',event]
command+=['-e','sched:sched_switch','--filter','prev_comm == "python" || next_comm == "python"',
'-e','sched:sched_wakeup','--filter','comm == "python"','--','sleep','155']
(HERE/'capture_commands.json').write_text(json.dumps({'system':command},indent=2))
log=(HERE/'system.log').open('x')
recorder=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
for name,reference in [('A','B'),('B','A')]:
 run=HERE/f'repro_{name}'
 observer_log=(HERE/f'repro_{name}_watch.log').open('x')
 observer=subprocess.Popen(PREFIX+[str(PYTHON),str(CODE/'native_stall_watch.py'),str(run),str(HERE/f'repro_{name}_watch.jsonl'),'--duration','85','--perf',str(PERF),'--perf-lib',str(LIB)],stdout=observer_log,stderr=subprocess.STDOUT)
 args=[str(PYTHON),'paced_replay.py','--video',f'data/raw/exp{name}.mp4','--reference',f'results/{reference}_offline/background.npy','--pzt',f'data/pzt/exp{name}','--output-dir',str(run),'--repeats','3','--opencv-threads','24','--conversion-threads','1','--measure-stages','--diagnose-stalls']
 with (HERE/f'repro_{name}.log').open('x') as f:
  result=subprocess.run(args,cwd=CODE,stdout=f,stderr=subprocess.STDOUT)
 print(name,'exit',result.returncode,flush=True)
 observer.wait(timeout=90)
 observer_log.close()
 print(name,'observer',observer.returncode,flush=True)
recorder.wait(timeout=170)
log.close()
print('system recorder',recorder.returncode,flush=True)
