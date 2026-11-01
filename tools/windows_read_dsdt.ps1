# Read a firmware table through the documented Windows API. Never execute or change ACPI methods.
param([Parameter(Mandatory = $true)][string]$Output)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw 'Output already exists.' }
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class MbFirmwareRead {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint GetSystemFirmwareTable(uint provider, uint table, IntPtr data, uint size);
}
'@
$size = [MbFirmwareRead]::GetSystemFirmwareTable(0x41435049, 0x54445344, [IntPtr]::Zero, 0)
if ($size -eq 0) { throw ('DSDT size query failed: ' + [Runtime.InteropServices.Marshal]::GetLastWin32Error()) }
$buffer = [Runtime.InteropServices.Marshal]::AllocHGlobal([int]$size)
try {
    $read = [MbFirmwareRead]::GetSystemFirmwareTable(0x41435049, 0x54445344, $buffer, $size)
    if ($read -ne $size) { throw 'DSDT table size changed or read failed.' }
    $bytes = New-Object byte[] $size
    [Runtime.InteropServices.Marshal]::Copy($buffer, $bytes, 0, [int]$size)
    [IO.File]::WriteAllBytes($Output, $bytes)
    Write-Output ('Read-only DSDT bytes: ' + $size)
} finally {
    [Runtime.InteropServices.Marshal]::FreeHGlobal($buffer)
}
