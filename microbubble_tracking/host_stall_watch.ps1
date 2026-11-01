# Read-only Windows counters. Raw cumulative values allow rates to be recomputed.
param(
    [Parameter(Mandatory = $true)][string]$Output,
    [ValidateRange(1, 3600)][int]$DurationSeconds = 100,
    [string]$StopFile = ''
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw 'Output already exists' }
$encoding = New-Object System.Text.UTF8Encoding($false)
$writer = New-Object System.IO.StreamWriter($Output, $false, $encoding)
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$totalKiB = [long](Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize
try {
    while ($timer.Elapsed.TotalSeconds -lt $DurationSeconds) {
        if ($StopFile -and (Test-Path -LiteralPath $StopFile)) { break }
        $start = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $memory = Get-CimInstance Win32_PerfRawData_PerfOS_Memory |
            Select-Object AvailableMBytes, CommittedBytes, CommitLimit,
                PagesInputPerSec, PageReadsPerSec, PagesOutputPerSec,
                PageWritesPerSec, Timestamp_Sys100NS
        $disk = @(Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk |
            Select-Object Name, DiskReadBytesPerSec, DiskWriteBytesPerSec,
                DiskReadsPerSec, DiskWritesPerSec, CurrentDiskQueueLength,
                AvgDisksecPerRead, AvgDisksecPerRead_Base,
                AvgDisksecPerWrite, AvgDisksecPerWrite_Base,
                Frequency_PerfTime, Timestamp_Sys100NS)
        $vm = @(Get-CimInstance Win32_PerfRawData_PerfProc_Process |
            Where-Object { $_.Name -like 'vmmem*' } |
            Select-Object Name, IDProcess, WorkingSet, WorkingSetPrivate,
                PageFileBytes, PageFaultsPerSec, PercentProcessorTime,
                Timestamp_Sys100NS)
        $row = [ordered]@{
            start_unix_ms = $start
            end_unix_ms = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
            total_physical_bytes = $totalKiB * 1024
            memory = $memory
            disks = $disk
            wsl_process = $vm
            # Working set is resident RAM; PageFileBytes is committed allocation,
            # not a measurement of how many bytes were read from the page file.
        }
        $writer.WriteLine(($row | ConvertTo-Json -Depth 5 -Compress))
        $writer.Flush()
        Start-Sleep -Milliseconds 800
    }
} finally {
    $writer.Dispose()
}
