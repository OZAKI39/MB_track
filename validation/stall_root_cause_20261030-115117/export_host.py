"""Export existing Windows ETL through installed Microsoft xperf; no recording starts."""
from pathlib import Path
import subprocess,shutil,json
here=Path(__file__).resolve().parent
host=Path('/mnt/c/Users/lzy/AppData/Local/Temp/MB_track_stall_20261030-115117/capture')
assert json.loads((host/'finished.json').read_text())['stop_exit_code']==0
xperf='/mnt/c/Program Files (x86)/Windows Kits/10/Windows Performance Toolkit/xperf.exe'
win=r'C:\Users\lzy\AppData\Local\Temp\MB_track_stall_20261030-115117\capture'
for name,action in [('trace_stats.txt',['tracestats','-detail']),('disk_detail.csv',['diskio','-detail']),('hardfaults.txt',['hardfault','-bytes']),('cpu_processes.txt',['cswitch','-process'])]:
 with (here/(name+'.log')).open('x') as log:
  rc=subprocess.run([xperf,'-i',win+r'\host.etl','-o',win+'\\'+name,'-a',*action],stdout=log,stderr=subprocess.STDOUT).returncode
 print(name,rc,flush=True)
 if rc==0:shutil.copy2(host/name,here/name)
for name in ['host.etl','recording.json','finished.json','start.log','stop.log']:
 shutil.copy2(host/name,here/name)
