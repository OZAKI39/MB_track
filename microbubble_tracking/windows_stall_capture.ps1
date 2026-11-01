# Temporary diagnostic recording only. Does not change drivers or device settings.
param(
    [Parameter(Mandatory = $true)][string]$Output,
    [ValidateRange(5, 300)][int]$DurationSeconds = 180,
    [ValidateSet('Stalls', 'Gpu', 'GpuLight', 'Acpi')][string]$ProfilePreset = 'Stalls'
)
$ErrorActionPreference = 'Stop'
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object System.Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Windows kernel tracing requires administrator consent.'
}
if (Test-Path -LiteralPath $Output) { throw 'Output directory already exists.' }
New-Item -ItemType Directory -Path $Output | Out-Null
$instance = 'MBTrack_' + [Guid]::NewGuid().ToString('N')
$profile = Join-Path $PSScriptRoot 'windows_stall_capture.wprp'
$wpr = Join-Path $env:SystemRoot 'System32\wpr.exe'
$started = $false
$encoding = New-Object System.Text.UTF8Encoding($false)
function Write-State($Name, $Data) {
    $Data.unix_ms = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    [System.IO.File]::WriteAllText((Join-Path $Output $Name),
        ($Data | ConvertTo-Json -Compress), $encoding)
}
try {
    # Gpu is an explicit diagnostic option; existing storage/memory callers keep their profile.
    $startArguments = @('-start', ($profile + '!MBStalls.Light'))
    if ($ProfilePreset -eq 'Gpu') {
        $startArguments = @('-start', 'GeneralProfile', '-start', 'GPU')
    } elseif ($ProfilePreset -eq 'GpuLight') {
        $gpuProfile = Join-Path $PSScriptRoot 'windows_gpu_light.wprp'
        $startArguments = @('-start', ($gpuProfile + '!MBGpu.Light'))
    } elseif ($ProfilePreset -eq 'Acpi') {
        $acpiProfile = Join-Path $PSScriptRoot 'windows_acpi_trace.wprp'
        $startArguments = @('-start', ($acpiProfile + '!MBAcpi.Light'))
    }
    & $wpr @startArguments -filemode -recordtempto $Output -instancename $instance 2>&1 |
        Out-File (Join-Path $Output 'start.log')
    if ($LASTEXITCODE -ne 0) { throw "WPR start failed: $LASTEXITCODE" }
    $started = $true
    Write-State 'recording.json' @{instance = $instance; duration_limit_s = $DurationSeconds; profile = $ProfilePreset}
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $DurationSeconds) {
        if (Test-Path -LiteralPath (Join-Path $Output 'stop.request')) { break }
        Start-Sleep -Milliseconds 200
    }
} catch {
    Write-State 'error.json' @{error = $_.Exception.Message}
} finally {
    if ($started) {
        & $wpr -stop (Join-Path $Output 'host.etl') -instancename $instance 2>&1 |
            Out-File (Join-Path $Output 'stop.log')
        Write-State 'finished.json' @{instance = $instance; stop_exit_code = $LASTEXITCODE}
    }
}
