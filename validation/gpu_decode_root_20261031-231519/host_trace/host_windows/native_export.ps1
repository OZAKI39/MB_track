$ErrorActionPreference='Stop'
$filter=@{Path='C:\Users\lzy\AppData\Local\Temp\gpu_decode_root_20261031-231519\host_trace\capture\host.etl'; StartTime=[DateTime]::FromFileTimeUtc(134379590747539232); EndTime=[DateTime]::FromFileTimeUtc(134379590748051856)}
$writer=New-Object System.IO.StreamWriter('C:\Users\lzy\AppData\Local\Temp\gpu_decode_root_20261031-231519\host_trace\window_A_805_native.jsonl',$false,(New-Object System.Text.UTF8Encoding($false)))
try {Get-WinEvent -FilterHashtable $filter -Oldest | ForEach-Object { $writer.WriteLine((@{time=$_.TimeCreated.ToUniversalTime().ToString('o'); id=$_.Id; provider=$_.ProviderName; pid=$_.ProcessId; tid=$_.ThreadId; xml=$_.ToXml()} | ConvertTo-Json -Compress)) }} finally {$writer.Dispose()}
