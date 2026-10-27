param(
    [string]$Run = "run2"
)

# Blocks until the given training run finishes, then reports how it ended.
# Used so the agent is notified the moment the last epoch completes.
$ErrorActionPreference = "Stop"
$runDir = Join-Path $PSScriptRoot "..\ml\artifacts\$Run"
$metrics = Join-Path $runDir "metrics.json"

while ($true) {
    $pids = @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -match "train\.py" } |
        Select-Object -ExpandProperty ProcessId)
    if ($pids.Count -eq 0) {
        if (Test-Path $metrics) {
            Write-Output "DONE: training exited and $metrics exists"
        } else {
            Write-Output "FAILED: training exited but $metrics is missing"
        }
        break
    }
    Start-Sleep -Seconds 60
}
