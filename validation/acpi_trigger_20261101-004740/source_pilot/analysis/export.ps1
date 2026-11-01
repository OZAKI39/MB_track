$ErrorActionPreference='Stop'
[Reflection.Assembly]::LoadFrom('C:\Users\lzy\AppData\Local\Temp\acpi_trigger_20261101-004740\source_pilot\traceevent_analysis\Microsoft.Diagnostics.Tracing.TraceEvent.dll') | Out-Null
Add-Type -Path 'C:\Users\lzy\AppData\Local\Temp\acpi_trigger_20261101-004740\source_pilot\traceevent_analysis\gpu_host_events.cs' -ReferencedAssemblies 'C:\Users\lzy\AppData\Local\Temp\acpi_trigger_20261101-004740\source_pilot\traceevent_analysis\Microsoft.Diagnostics.Tracing.TraceEvent.dll'
[GpuHostEvents]::ReadOrigins('C:\Users\lzy\AppData\Local\Temp\acpi_trigger_20261101-004740\source_pilot\capture\host.etl','C:\Users\lzy\AppData\Local\Temp\acpi_trigger_20261101-004740\source_pilot\analysis_selected_events.tsv')
