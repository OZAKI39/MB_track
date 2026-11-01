
$ErrorActionPreference='Stop'
$start=[DateTimeOffset]::Parse('2026-10-30T11:32:31.2109435Z').LocalDateTime
$end=[DateTimeOffset]::Parse('2026-10-30T11:35:10.7564397Z').LocalDateTime
$result=[ordered]@{start_utc='2026-10-30T11:32:31.2109435Z';end_utc='2026-10-30T11:35:10.7564397Z';queries=@()}
foreach($spec in @(@{LogName='Application';Id=@(1000,1002)},@{LogName='System';Id=@(2004)})) {
 $query=@{LogName=$spec.LogName;Id=$spec.Id;StartTime=$start;EndTime=$end}
 $events=@(); $failure=$null
 try {$events=@(Get-WinEvent -FilterHashtable $query -ErrorAction Stop | ForEach-Object {
   @{id=$_.Id;provider=$_.ProviderName;time_utc=$_.TimeCreated.ToUniversalTime().ToString('o');message=$_.Message}
 })} catch {if($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*') {$failure=$_.Exception.Message}}
 $result.queries+=@{log=$spec.LogName;events=$events;error=$failure}
}
$result | ConvertTo-Json -Depth 7 -Compress
