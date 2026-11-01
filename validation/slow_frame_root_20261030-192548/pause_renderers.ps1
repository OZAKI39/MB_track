# Optional diagnostic only: requires explicit approval before execution.
# Pause exactly the validated VS Code renderer processes; never terminate them.
param(
    [Parameter(Mandatory = $true)][string]$Output,
    [ValidateRange(10, 90)][int]$DurationSeconds = 90,
    [int[]]$RendererIds = @(27244, 14412)
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw 'Output already exists.' }
$targets = @()
foreach ($rendererId in $RendererIds) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$rendererId"
    if ($null -eq $process -or $process.Name -ne 'Code.exe' -or
        $process.CommandLine -notmatch '--type=renderer(?:\s|$)') {
        throw "PID $rendererId is not a validated VS Code renderer."
    }
    $targets += $process
}
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class RendererPause {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
    [DllImport("kernel32.dll")]
    public static extern bool CloseHandle(IntPtr process);
    [DllImport("ntdll.dll")]
    public static extern int NtSuspendProcess(IntPtr process);
    [DllImport("ntdll.dll")]
    public static extern int NtResumeProcess(IntPtr process);
}
'@
New-Item -ItemType Directory -Path $Output | Out-Null
$paused = @()
$records = @()
try {
    foreach ($target in $targets) {
        $handle = [RendererPause]::OpenProcess(0x0800, $false, $target.ProcessId)
        if ($handle -eq [IntPtr]::Zero) { throw "Cannot open PID $($target.ProcessId)." }
        $status = [RendererPause]::NtSuspendProcess($handle)
        if ($status -ne 0) {
            [RendererPause]::CloseHandle($handle) | Out-Null
            throw "Suspend failed for PID $($target.ProcessId): $status"
        }
        $paused += @{id=$target.ProcessId; handle=$handle}
        $records += @{id=$target.ProcessId; action='paused'; unix_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()}
    }
    $records | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output 'paused.json')
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $DurationSeconds) {
        if (Test-Path -LiteralPath (Join-Path $Output 'resume.request')) { break }
        Start-Sleep -Milliseconds 200
    }
} finally {
    foreach ($item in $paused) {
        $status = [RendererPause]::NtResumeProcess($item.handle)
        $records += @{id=$item.id; action='resumed'; status=$status; unix_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()}
        [RendererPause]::CloseHandle($item.handle) | Out-Null
    }
    $records | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output 'finished.json')
}
