# Scheduler for Tier-1 round-2 ablations (Plan A).
# - Waits until 22:30 local time, then launches the 7-run ablation suite in WSL (full speed, 3-4 parallel phases).
# - At 23:00 locks GPU graphics clock to 2000 MHz (user-approved) to reduce noise/heat overnight.
# - Restores clocks when the suite finishes (or on failure).
# Log: E:\FINO\logs\r2_scheduler.log

$ErrorActionPreference = 'Continue'
$start = Get-Date '2026-10-26 01:00:00'
$lockAt = Get-Date '2026-10-26 01:30:00'
$log = 'E:\FINO\logs\r2_scheduler.log'
New-Item -ItemType Directory -Force -Path 'E:\FINO\logs' | Out-Null

function Log([string]$m) {
    $line = "$(Get-Date -Format o) $m"
    Add-Content -Path $log -Value $line
    Write-Output $line
}

Log "scheduler started; start=$start lock=$lockAt"
$ws = 'C:\Windows\System32\nvidia-smi.exe'
$wsl = 'C:\Windows\System32\wsl.exe'
$args = @('-d', 'FINO-Ubuntu', '-u', 'root', '--', '/bin/bash', '-lc', 'bash /mnt/e/FINO/wsl_setup/abl_round2.sh')

try {
    while ((Get-Date) -lt $start) {
        $remain = [int]($start - (Get-Date)).TotalSeconds
        Log "waiting ${remain}s"
        Start-Sleep -Seconds ([Math]::Min(300, [Math]::Max(30, $remain)))
    }
    Log 'launching ablation orchestrator'
    $p = Start-Process -FilePath $wsl -ArgumentList $args -NoNewWindow -PassThru
    Log "orchestrator pid=$($p.Id)"

    if ((Get-Date) -lt $lockAt) {
        $wait = [int]($lockAt - (Get-Date)).TotalSeconds
        Log "waiting ${wait}s before clock lock"
        Start-Sleep -Seconds ([Math]::Max(0, $wait))
    }
    Log 'locking GPU clocks to 2000 MHz'
    & $ws -lgc 2000 2>&1 | ForEach-Object { Log "nvidia-smi: $_" }
    $locked = $true

    $p.WaitForExit()
    Log "orchestrator exited rc=$($p.ExitCode)"
}
catch {
    Log "ERROR: $($_.Exception.Message)"
}
finally {
    Log 'restoring GPU clocks'
    & $ws -rgc 2>&1 | ForEach-Object { Log "nvidia-smi: $_" }
    Log 'DONE'
}
