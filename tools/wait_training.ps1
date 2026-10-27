# Blocks until the 84-class training run finishes, then reports how it ended.
# Used so the agent can be notified the moment epoch 12 completes.
$ErrorActionPreference = "Stop"
$runDir = Join-Path $PSScriptRoot "..\ml\artifacts\run1"
$metrics = Join-Path $runDir "metrics.json"

function Get-TrainPids {
    Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -match "train\.py" } |
        Select-Object -ExpandProperty ProcessId
}

while ($true) {
    $pids = @(Get-TrainPids)
    if ($pids.Count -eq 0) {
        if (Test-Path $metrics) {
            Write-Output "DONE: training exited and metrics.json exists"
        } else {
            Write-Output "FAILED: training exited but metrics.json is missing"
        }
        break
    }
    Start-Sleep -Seconds 60
}
