"""Offline thread-count control plus bounded host capture and clock correspondence."""
import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path('/home/lzy/projects/new_microbubble_experiment')
CODE = ROOT / 'customer_workspace/microbubble_tracking'
BASE = Path(__file__).resolve().parent
QUIET = '--quiet' in sys.argv
PAUSED = '--paused' in sys.argv
PHASE = 'paused' if PAUSED else 'quiet' if QUIET else ''
OUT = BASE / PHASE if PHASE else BASE
if PHASE:
    OUT.mkdir(exist_ok=False)
HOST = Path('/mnt/c/Users/lzy/AppData/Local/Temp') / (BASE.name + ('_' + PHASE if PHASE else ''))
CAPTURE = HOST / 'capture'
PS = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
PERF = ROOT / 'customer_workspace/dependencies/native_perf/installed/usr/lib/linux-tools-6.8.0-147/perf'
LIB = ROOT / 'customer_workspace/dependencies/native_perf/installed/usr/lib/x86_64-linux-gnu'


def windows_path(path):
    return subprocess.check_output(['wslpath','-w',str(path)],text=True).strip()


def bridge(stop):
    command = "while ($null -ne [Console]::ReadLine()) { [Console]::WriteLine((@{ unix_s=([DateTime]::UtcNow.Ticks - 621355968000000000)/10000000.0; qpc=[System.Diagnostics.Stopwatch]::GetTimestamp(); frequency=[System.Diagnostics.Stopwatch]::Frequency } | ConvertTo-Json -Compress)) }"
    with (OUT / 'clock_bridge.log').open('x') as log, (OUT / 'clock_bridge.jsonl').open('x',buffering=1) as rows:
        process = subprocess.Popen([PS,'-NoProfile','-Command',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        try:
            while not stop.is_set():
                started = time.perf_counter()
                process.stdin.write('clock\n')
                process.stdin.flush()
                row = json.loads(process.stdout.readline())
                row.update(send_s=started,receive_s=time.perf_counter())
                rows.write(json.dumps(row)+'\n')
                stop.wait(0.25)
        finally:
            process.stdin.close()
            process.wait(timeout=5)


def main():
    HOST.mkdir(exist_ok=False)
    for name in ('windows_stall_capture.ps1','windows_stall_capture.wprp'):
        shutil.copyfile(CODE / name,HOST / name)
    argument = '-NoProfile -ExecutionPolicy Bypass -File "'+windows_path(HOST/'windows_stall_capture.ps1')+'" -Output "'+windows_path(CAPTURE)+'" -DurationSeconds 180'
    script = "$ErrorActionPreference='Stop'; Start-Process powershell.exe -Verb RunAs -PassThru -ArgumentList '"+argument.replace("'","''")+"' | Select-Object -ExpandProperty Id"
    encoded = base64.b64encode(script.encode('utf-16le')).decode('ascii')
    with (OUT/'elevation.log').open('x') as log:
        subprocess.run([PS,'-NoProfile','-EncodedCommand',encoded],stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
    deadline=time.monotonic()+45
    while not (CAPTURE/'recording.json').exists():
        if (CAPTURE/'error.json').exists() or time.monotonic()>deadline:
            raise RuntimeError('Host capture did not start')
        time.sleep(0.2)
    print('HOST RECORDING',flush=True)
    stop=threading.Event()
    clock=threading.Thread(target=bridge,args=(stop,),daemon=True)
    clock.start()
    records=[]
    pause_process=None
    pause_log=None
    try:
        if PAUSED:
            shutil.copyfile(BASE/'pause_renderers.ps1',HOST/'pause_renderers.ps1')
            pause_log=(OUT/'pause.log').open('x')
            pause_process=subprocess.Popen([PS,'-NoProfile','-File',windows_path(HOST/'pause_renderers.ps1'),'-Output',windows_path(HOST/'pause'),'-DurationSeconds','90'],stdout=pause_log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+20
            while not (HOST/'pause/paused.json').exists():
                if pause_process.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Renderer pause did not start; see pause.log')
                time.sleep(.1)
            print('RENDERERS PAUSED; AUTOMATIC RESTORE <= 90 S',flush=True)
        conditions = (('host24',24,'A',2),('hostB',24,'B',1)) if PHASE else (('host24',24,'A',3),('host1',1,'A',3))
        for name,threads,label,repeats in conditions:
            run=OUT/('native_'+name)
            observer=['/mnt/c/Windows/System32/wsl.exe','-d','Ubuntu','-u','root','--',sys.executable,str(CODE/'native_stall_watch.py'),str(run),str(OUT/f'watch_{name}.jsonl'),'--duration','82','--interval','0.005','--perf',str(PERF),'--perf-lib',str(LIB),'--perf-targets','consumer','writer','--frequency','999']
            replay=[sys.executable,'-m','customer_workspace.microbubble_tracking.paced_replay','--video',str(CODE/f'data/raw/exp{label}.mp4'),'--reference',str(CODE/'results/B_offline/background.npy'),'--pzt',str(CODE/f'data/pzt/exp{label}'),'--output-dir',str(run),'--repeats',str(repeats),'--opencv-threads',str(threads),'--conversion-threads','1','--measure-stages','--diagnose-stalls']
            item=dict(name=name,observer=observer,replay=replay)
            records.append(item)
            (OUT/'host_commands.json').write_text(json.dumps(records,indent=2)+'\n')
            print('START',name,flush=True)
            with (OUT/f'{name}_observer.log').open('x') as obslog,(OUT/f'{name}_replay.log').open('x') as runlog:
                watch=subprocess.Popen(observer,stdout=obslog,stderr=subprocess.STDOUT)
                try:
                    result=subprocess.run(replay,cwd=ROOT,stdout=runlog,stderr=subprocess.STDOUT,timeout=85)
                    item['replay_exit']=result.returncode
                finally:
                    item['observer_exit']=watch.wait(timeout=25)
            (OUT/'host_commands.json').write_text(json.dumps(records,indent=2)+'\n')
            print('DONE',name,item['replay_exit'],item['observer_exit'],flush=True)
            if item['replay_exit'] or item['observer_exit']:
                raise RuntimeError('Replay or native capture failed')
    finally:
        try:
            if pause_process is not None:
                if (HOST/'pause').exists():
                    (HOST/'pause/resume.request').touch()
                pause_process.wait(timeout=15)
                pause_log.close()
                finished=HOST/'pause/finished.json'
                if not finished.exists():
                    raise RuntimeError('Pause script failed before recording its state; inspect pause.log')
                state=json.loads(finished.read_text(encoding='utf-8-sig'))
                (OUT/'pause_finished.json').write_text(json.dumps(state,indent=2)+'\n')
                if any(r.get('status',0)!=0 for r in state):
                    raise RuntimeError('Renderer resume returned an error; inspect pause_finished.json')
                print('RENDERERS RESTORED',flush=True)
        finally:
            stop.set()
            clock.join(timeout=5)
            (CAPTURE/'stop.request').touch()
    deadline=time.monotonic()+45
    while not (CAPTURE/'finished.json').exists():
        if time.monotonic()>deadline: raise RuntimeError('Host trace shutdown not confirmed')
        time.sleep(.2)
    target=OUT/'host_capture'
    target.mkdir()
    for name in ('recording.json','finished.json','start.log','stop.log','host.etl'):
        shutil.copyfile(CAPTURE/name,target/name)
    print('HOST STOPPED',flush=True)


if __name__=='__main__':
    main()
