"""Two identical offline runs under one bounded Windows recording."""
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATHS = json.loads((HERE / 'paths.json').read_text())
CODE = Path(PATHS['code'])
ROOT = Path(PATHS['root'])
HOST = Path(PATHS['host'])
PS = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
label = sys.argv[1]
assert label in ('baseline', 'background_off')
group = HERE / label
group.mkdir()
host_output = HOST / label
win_script = subprocess.check_output(['wslpath', '-w', str(HOST / 'windows_stall_capture.ps1')], text=True).strip()
win_output = subprocess.check_output(['wslpath', '-w', str(host_output)], text=True).strip()
argument = f'-NoProfile -ExecutionPolicy Bypass -File "{win_script}" -Output "{win_output}" -DurationSeconds 180'
command = "$ErrorActionPreference='Stop'; $p=Start-Process powershell.exe -Verb RunAs -PassThru -ArgumentList '" + argument.replace("'", "''") + "'; $p.Id"
encoded = base64.b64encode(command.encode('utf-16le')).decode('ascii')
with (group / 'elevation.log').open('x') as log:
    subprocess.run([PS, '-NoProfile', '-EncodedCommand', encoded], stdout=log, stderr=subprocess.STDOUT, check=True)
deadline = time.monotonic() + 60
while not (host_output / 'recording.json').exists():
    if (host_output / 'error.json').exists() or time.monotonic() > deadline:
        raise RuntimeError('Windows recording failed to start')
    time.sleep(0.2)
print(label, 'Windows capture started', flush=True)
commands = []
try:
    for index in (1, 2):
        output = group / f'run_{index}'
        args = [str(ROOT / '.venv/bin/python'), str(CODE / 'memory_pressure_check.py'),
                '--video', str(CODE / 'data/raw/expB.mp4'),
                '--reference', str(CODE / 'results/A_offline/background.npy'),
                '--pzt', str(CODE / 'data/pzt/expB'), '--output-dir', str(output),
                '--label', label, '--repeats', '3']
        commands.append(args)
        (group / 'commands.json').write_text(json.dumps(commands, indent=2))
        with (group / f'run_{index}.log').open('x') as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(label, 'run', index, 'completed', flush=True)
finally:
    (host_output / 'stop.request').touch()
deadline = time.monotonic() + 90
while not (host_output / 'finished.json').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('Windows trace stop not yet confirmed')
    time.sleep(0.5)
finished = json.loads((host_output / 'finished.json').read_text())
assert finished['stop_exit_code'] == 0
print(label, 'Windows capture stopped', flush=True)
