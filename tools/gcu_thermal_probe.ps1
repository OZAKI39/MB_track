# Read only the five EC bytes used by this machine's existing firmware/control
# centre for temperature and fan RPM. No arbitrary address or write operation.
param(
    [Parameter(Mandatory = $true)][string]$Output,
    [ValidateRange(1, 30)][int]$Samples = 10
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$computer = Get-CimInstance Win32_ComputerSystem
if ($computer.Manufacturer -ne 'MECHREVO' -or $computer.Model -ne 'JiguangX Series GM6PR0C') {
    throw 'This EC sensor mapping has only been checked on MECHREVO GM6PR0C.'
}
$gcuProcesses = @(Get-CimInstance Win32_Process -Filter "Name='GCUService.exe'")
if ($gcuProcesses.Count -ne 1 -or -not $gcuProcesses[0].ExecutablePath) {
    throw 'Cannot verify the installed controller. Run this read-only probe as administrator.'
}
$gcuHash = (Get-FileHash -LiteralPath $gcuProcesses[0].ExecutablePath -Algorithm SHA256).Hash
if ($gcuHash -ne '785150abc1b7344434459e74ed17ebaf487e58d24fe3bcb6d356cf8a29465d11') {
    throw 'Controller version changed; recheck the sensor mapping before use.'
}
if (Test-Path -LiteralPath $Output) { throw 'Output already exists.' }
New-Item -ItemType Directory -Path $Output | Out-Null
$encoding = New-Object System.Text.UTF8Encoding($false)
$instance = New-Object System.Management.ManagementObject(
    "root\WMI", "AcpiTest_MULong.InstanceName='ACPI\PNP0C14\1_1'", $null)
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class GcuProbeGpu {
    [DllImport("nvml.dll")] public static extern int nvmlInit_v2();
    [DllImport("nvml.dll")] public static extern int nvmlDeviceGetHandleByIndex_v2(uint index, out IntPtr device);
    [DllImport("nvml.dll")] public static extern int nvmlDeviceGetTemperature(IntPtr device, uint sensor, out uint value);
    [DllImport("nvml.dll")] public static extern int nvmlShutdown();
}
'@
$gpuHandle = [IntPtr]::Zero
$gpuStarted = $false

function Read-SensorByte([int]$Address) {
    if ($Address -notin @(0x43e, 0x464, 0x465, 0x46b, 0x46c)) {
        throw 'Address is not an approved temperature/fan read.'
    }
    # Matches WMIEC.WMIReadECRAM, not the 0x0500 CPU/GPU settings command.
    $inputData = $instance.GetMethodParameters('GetSetULong')
    $inputData['Data'] = [UInt64]1099511627776 + [UInt64]$Address
    $reply = $instance.InvokeMethod('GetSetULong', $inputData, $null)
    $raw = [UInt64]$reply['Return']
    # RKBC returns CMDL/CMDH together. The vendor byte reader uses only the low
    # byte (conv.u1); retain the full response so the conversion is auditable.
    $script:rawValues[('0x{0:x}' -f $Address)] = $raw
    if ($raw -gt 65535 -or $raw -eq 0xfefe) {
        throw "Invalid EC read at $Address : $raw"
    }
    return [int]($raw -band 255)
}

try {
    if ([GcuProbeGpu]::nvmlInit_v2() -ne 0) { throw 'NVML initialization failed.' }
    $gpuStarted = $true
    if ([GcuProbeGpu]::nvmlDeviceGetHandleByIndex_v2(0, [ref]$gpuHandle) -ne 0) {
        throw 'Cannot identify the GPU temperature sensor.'
    }
    for ($index = 0; $index -lt $Samples; $index++) {
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $script:rawValues = @{}
        $row = [ordered]@{utc=[DateTime]::UtcNow.ToString('o'); index=$index}
        try {
            $row.cpu_c = (Read-SensorByte 0x43e) -band 127
            $cpuHigh = Read-SensorByte 0x464
            $cpuLow = Read-SensorByte 0x465
            $gpuHigh = Read-SensorByte 0x46c
            $gpuLow = Read-SensorByte 0x46b
            $row.cpu_fan_rpm = ($cpuHigh -shl 8) -bor $cpuLow
            $row.gpu_fan_rpm = ($gpuHigh -shl 8) -bor $gpuLow
            $gpuC = [uint32]0
            if ([GcuProbeGpu]::nvmlDeviceGetTemperature($gpuHandle, 0, [ref]$gpuC) -ne 0) {
                throw 'GPU temperature read failed.'
            }
            $row.gpu_c = $gpuC
            $row.ok = $true
        } catch {
            $row.ok = $false
            $row.error = $_.Exception.Message
        }
        $row.read_ms = $timer.Elapsed.TotalMilliseconds
        $row.ec_raw = $script:rawValues
        [IO.File]::AppendAllText((Join-Path $Output 'samples.jsonl'),
            ($row | ConvertTo-Json -Compress) + "`n", $encoding)
        if (-not $row.ok) { throw $row.error }
        Start-Sleep -Milliseconds ([Math]::Max(0, 1000 - [int]$timer.Elapsed.TotalMilliseconds))
    }
    [IO.File]::WriteAllText((Join-Path $Output 'finished.json'), '{"completed":true}', $encoding)
} catch {
    [IO.File]::WriteAllText((Join-Path $Output 'failed.json'),
        (@{error=$_.Exception.Message} | ConvertTo-Json), $encoding)
    throw
} finally {
    $instance.Dispose()
    if ($gpuStarted) { [GcuProbeGpu]::nvmlShutdown() | Out-Null }
}
