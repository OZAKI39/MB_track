"""Measure Windows UTC inside WSL request/response bounds, without subtracting clocks."""
import csv,json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent
command="while ($null -ne [Console]::ReadLine()) { [Console]::WriteLine((@{unix_s=([DateTime]::UtcNow.Ticks - 621355968000000000)/10000000.0; qpc=[System.Diagnostics.Stopwatch]::GetTimestamp(); frequency=[System.Diagnostics.Stopwatch]::Frequency} | ConvertTo-Json -Compress)) }"
proc=subprocess.Popen(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
with (p/'clock_bridge.csv').open('x',newline='',buffering=1) as file:
 writer=csv.writer(file)
 writer.writerow(['wsl_before_s','wsl_after_s','linux_unix_s','windows_unix_s','windows_qpc','windows_frequency','roundtrip_ms'])
 for i in range(120):
  start=time.perf_counter()
  proc.stdin.write('\n');proc.stdin.flush()
  response=json.loads(proc.stdout.readline())
  end=time.perf_counter()
  writer.writerow([start,end,time.time(),response['unix_s'],response['qpc'],response['frequency'],(end-start)*1000])
  time.sleep(1)
proc.stdin.close();proc.wait(timeout=5)
