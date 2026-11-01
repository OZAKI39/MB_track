# Read-only sensor/service inspection. Never stops services or invokes firmware methods.
param([Parameter(Mandatory = $true)][string]$Output)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if (Test-Path -LiteralPath $Output) { throw 'Output already exists.' }
New-Item -ItemType Directory -Path $Output | Out-Null
$encoding = New-Object System.Text.UTF8Encoding($false)
$result = [ordered]@{utc = [DateTime]::UtcNow.ToString('o'); queries = @()}
function Read-Query($Label, [scriptblock]$Query) {
    try {
        $values = @(& $Query)
        $script:result.queries += @{label=$Label; ok=$true; values=$values}
    } catch {
        $script:result.queries += @{label=$Label; ok=$false; error=$_.Exception.Message}
    }
}
Read-Query 'thermal_zones' {
    Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature |
        Select-Object InstanceName,CurrentTemperature,CriticalTripPoint,ActiveTripPoint
}
Read-Query 'thermal_performance' {
    Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation |
        Select-Object Name,Temperature,HighPrecisionTemperature,PercentPassiveLimit
}
Read-Query 'fan_status' {
    Get-CimInstance Win32_Fan | Select-Object Name,Status,DesiredSpeed,VariableSpeed
}
Read-Query 'service' {
    Get-CimInstance Win32_Service -Filter "Name='GCUBridge'" |
        Select-Object Name,State,ProcessId,StartMode,PathName
}
Read-Query 'processes' {
    Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^GCU(Service|Bridge)\.exe$'} |
        Select-Object Name,ProcessId,ParentProcessId,CreationDate,ExecutablePath
}
Read-Query 'openhardwaremonitor' {
    Get-CimInstance -Namespace root/OpenHardwareMonitor -ClassName Sensor |
        Where-Object {$_.SensorType -in @('Temperature','Fan')} |
        Select-Object Name,SensorType,Value,Identifier
}
Read-Query 'librehardwaremonitor' {
    Get-CimInstance -Namespace root/LibreHardwareMonitor -ClassName Sensor |
        Where-Object {$_.SensorType -in @('Temperature','Fan')} |
        Select-Object Name,SensorType,Value,Identifier
}
Read-Query 'firmware_interface_schema_only' {
    Get-CimClass -Namespace root/wmi -ClassName AcpiTest_MULong |
        Select-Object -ExpandProperty CimClassMethods |
        Select-Object Name,ReturnType,Parameters,Qualifiers
}
[IO.File]::WriteAllText((Join-Path $Output 'snapshot.json'),
    ($result | ConvertTo-Json -Depth 8), $encoding)
[IO.File]::WriteAllText((Join-Path $Output 'finished.json'), '{"completed":true}', $encoding)
