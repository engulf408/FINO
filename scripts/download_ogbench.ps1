# download_ogbench.ps1
# Re-entrant parallel HTTP Range downloader for OGBench datasets.
# Windows PowerShell 5.1 compatible. HTTP work is done by curl.exe (bundled with Windows 10+).
#
# Usage (execution policy is restricted, so invoke through -Command, not by running the .ps1 directly):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& 'E:\FINO\scripts\download_ogbench.ps1' -Concurrency 24 -ChunkMB 4 -IncludeOptional"
#
# Design notes:
#   * Start-Process -PassThru returns Process objects whose .ExitCode is unavailable ($null) in this
#     environment, so an attempt is judged successful ONLY by exact on-disk byte counts.
#   * Byte-level resume inside each chunk: curl writes the remaining byte range to a .tmp file, which is
#     appended to chunk_XXXXX.bin when the attempt ends, so a dropped connection never discards bytes.
#   * A chunk is done only when its file size equals the exact expected size.
#   * Chunks are merged into <dest>\<name>.part, then validated:
#       (a) byte count == server Content-Length
#       (b) CRC of every zip member read fully (catches corrupt merged data)
#       (c) numpy np.load(...).files non-empty
#     Only after all checks pass is .part renamed to the final .npz.
#   * Existing complete .npz files are skipped after verification.
#   * Retries use exponential backoff (max 8 attempts per chunk).
#   * Progress is written to E:\FINO\logs\download.log every 30 seconds.

param(
    [int]$Concurrency = 24,
    [int]$ChunkMB = 4,
    [int]$MaxRetries = 8,
    [string]$DestDir = 'C:\Users\33709\.ogbench\data',
    [string]$CacheDir = 'E:\FINO\.dl_cache',
    [string]$LogFile = 'E:\FINO\logs\download.log',
    [string]$PythonExe = 'C:\Users\33709\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe',
    [string[]]$Only = @(),
    [switch]$IncludeOptional,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressIntervalSec = 30

# ---------------------------------------------------------------- helpers
function Format-Bytes {
    param([double]$b)
    if ($b -ge 1GB) { return ('{0:N2} GB' -f ($b / 1GB)) }
    if ($b -ge 1MB) { return ('{0:N2} MB' -f ($b / 1MB)) }
    if ($b -ge 1KB) { return ('{0:N1} KB' -f ($b / 1KB)) }
    return ('{0} B' -f [long]$b)
}

function Format-Duration {
    param([double]$s)
    return ([TimeSpan]::FromSeconds($s)).ToString('hh\:mm\:ss')
}

function Write-Log {
    param([string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    $logDir = Split-Path -Parent $LogFile
    if ($logDir -and -not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    try { Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8 } catch { }
    Write-Host $line
}

function Append-File {
    param([string]$Source, [string]$Dest)
    $buf = New-Object byte[] (1048576)
    $in = [System.IO.File]::OpenRead($Source)
    try {
        $out = [System.IO.File]::Open($Dest, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        try {
            while (($r = $in.Read($buf, 0, $buf.Length)) -gt 0) { $out.Write($buf, 0, $r) }
        } finally { $out.Dispose() }
    } finally { $in.Dispose() }
}

function Get-RemoteLength {
    param([string]$Url)
    $lastErr = $null
    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        try {
            $out = & curl.exe -sS -I --connect-timeout 30 --max-time 60 $Url 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw ('curl HEAD exit={0} output={1}' -f $LASTEXITCODE, (($out | Out-String).Trim()))
            }
            $text = ($out | Out-String)
            if ($text -match '(?im)^content-length:\s*(\d+)') {
                return [long]$Matches[1]
            }
            throw 'no content-length header in HEAD response'
        } catch {
            $lastErr = $_.Exception.Message
            if ($attempt -eq $MaxRetries) { break }
            $delay = [int]([math]::Min(60, [math]::Pow(2, $attempt))) + (Get-Random -Minimum 0 -Maximum 3)
            Write-Log ('RETRY_HEAD url={0} attempt={1}/{2} delay={3}s err={4}' -f $Url, $attempt, $MaxRetries, $delay, $lastErr)
            Start-Sleep -Seconds $delay
        }
    }
    throw ('could not determine content-length for {0}: {1}' -f $Url, $lastErr)
}

function Test-Npz {
    param([string]$Path, [string]$Python)
    $zipEntries = 0
    $zipBytes = [long]0
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction Stop
        $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
        try {
            $zipEntries = $zip.Entries.Count
            if ($zipEntries -lt 1) { throw 'zip archive has zero entries' }
            # Reading every member to EOF makes .NET verify each entry CRC32.
            $buf = New-Object byte[] (1048576)
            foreach ($entry in $zip.Entries) {
                $stream = $entry.Open()
                try {
                    while (($r = $stream.Read($buf, 0, $buf.Length)) -gt 0) { $zipBytes += $r }
                } finally { $stream.Dispose() }
            }
        } finally { $zip.Dispose() }
    } catch {
        return [pscustomobject]@{ Ok = $false; Detail = ('zip/CRC check failed: {0}' -f $_.Exception.Message) }
    }
    if ($Python -and (Test-Path -LiteralPath $Python)) {
        # Helper .py file instead of -c: Windows PowerShell 5.1 strips embedded double quotes
        # when passing arguments to native executables.
        $verifyPy = Join-Path $CacheDir '_verify_npz.py'
        if (-not (Test-Path -LiteralPath $verifyPy)) {
            $pyLines = @(
                'import sys',
                'import numpy as np',
                'f = np.load(sys.argv[1], allow_pickle=False)',
                'files = list(f.files)',
                'assert len(files) > 0, "npz has no arrays"',
                'print("numpy_files=" + str(len(files)))'
            )
            Set-Content -LiteralPath $verifyPy -Value $pyLines -Encoding ASCII
        }
        $out = & $Python $verifyPy $Path 2>&1
        if ($LASTEXITCODE -ne 0) {
            return [pscustomobject]@{ Ok = $false; Detail = ('numpy load failed: {0}' -f (($out | Out-String).Trim())) }
        }
        return [pscustomobject]@{ Ok = $true; Detail = ('zip_entries={0}; zip_payload={1}; {2}' -f $zipEntries, $zipBytes, (($out | Out-String).Trim())) }
    }
    return [pscustomobject]@{ Ok = $true; Detail = ('zip_entries={0}; zip_payload={1}; numpy=not-run' -f $zipEntries, $zipBytes) }
}

function Get-ChunkPlan {
    param([long]$Length, [int]$ChunkMB)
    $chunkSize = [long]$ChunkMB * 1MB
    if ($Length -le 8MB) { $chunkSize = [long](512KB) }   # tiny files: allow more parallelism
    $count = [int][math]::Ceiling($Length / [double]$chunkSize)
    return [pscustomobject]@{ ChunkSize = $chunkSize; Count = $count }
}

function Remove-StalePart {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue }
}

# ---------------------------------------------------------------- downloader
function Download-File {
    param([string]$Name, [string]$Url)

    $finalPath = Join-Path $DestDir $Name
    $partPath = $finalPath + '.part'
    $fileStart = Get-Date

    Write-Log ('START file={0} url={1}' -f $Name, $Url)
    $remoteLen = Get-RemoteLength -Url $Url
    Write-Log ('REMOTE file={0} content_length={1} ({2})' -f $Name, $remoteLen, (Format-Bytes $remoteLen))

    # --- skip if a complete final file already exists
    if ((Test-Path -LiteralPath $finalPath) -and (-not $Force)) {
        $existingLen = (Get-Item -LiteralPath $finalPath).Length
        if ($existingLen -eq $remoteLen) {
            $v = Test-Npz -Path $finalPath -Python $PythonExe
            if ($v.Ok) {
                $elapsed = ((Get-Date) - $fileStart).TotalSeconds
                Write-Log ('SKIP file={0} bytes={1} validation="{2}"' -f $Name, $remoteLen, $v.Detail)
                return [pscustomobject]@{ Name = $Name; Status = 'SKIP'; Size = $remoteLen; Elapsed = $elapsed; Validation = $v.Detail }
            }
            Write-Log ('EXISTING file={0} size matches but validation failed ({1}); removing' -f $Name, $v.Detail)
        } else {
            Write-Log ('EXISTING file={0} size={1} != remote={2}; removing' -f $Name, $existingLen, $remoteLen)
        }
        Remove-Item -LiteralPath $finalPath -Force -ErrorAction SilentlyContinue
    }

    $cacheFileDir = Join-Path $CacheDir $Name
    $statePath = Join-Path $cacheFileDir 'state.json'
    $spec = Get-ChunkPlan -Length $remoteLen -ChunkMB $ChunkMB

    # --- re-entrant chunk cache state
    $stateOk = $false
    if (Test-Path -LiteralPath $statePath) {
        try {
            $st = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
            if (([long]$st.Length -eq $remoteLen) -and ([long]$st.ChunkSize -eq $spec.ChunkSize)) { $stateOk = $true }
        } catch { }
    }
    if (-not $stateOk) {
        if (Test-Path -LiteralPath $cacheFileDir) { Remove-Item -LiteralPath $cacheFileDir -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $cacheFileDir | Out-Null
        [pscustomobject]@{
            Url = $Url; Length = $remoteLen; ChunkSize = $spec.ChunkSize
            ChunkCount = $spec.Count; Created = (Get-Date).ToString('o')
        } | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    }

    # --- chunk table (resume: keep exact chunks, recover .tmp leftovers from a killed run)
    $chunks = New-Object System.Collections.Generic.List[object]
    for ($i = 0; $i -lt $spec.Count; $i++) {
        $start = [long]$i * $spec.ChunkSize
        $end = [long][math]::Min($start + $spec.ChunkSize - 1, $remoteLen - 1)
        $expected = [long]($end - $start + 1)
        $chunkFile = Join-Path $cacheFileDir ('chunk_{0:D5}.bin' -f $i)
        $tempFile = Join-Path $cacheFileDir ('chunk_{0:D5}.tmp' -f $i)
        $errFile = Join-Path $cacheFileDir ('chunk_{0:D5}.err' -f $i)

        if (Test-Path -LiteralPath $tempFile) {
            $tmpLen = (Get-Item -LiteralPath $tempFile).Length
            $curLen = [long]0
            if (Test-Path -LiteralPath $chunkFile) { $curLen = (Get-Item -LiteralPath $chunkFile).Length }
            if (($tmpLen -gt 0) -and ($curLen -le $expected) -and (($curLen + $tmpLen) -le $expected)) {
                if ($curLen -gt 0) { Append-File -Source $tempFile -Dest $chunkFile }
                else { Move-Item -LiteralPath $tempFile -Destination $chunkFile -Force }
            }
            if (Test-Path -LiteralPath $tempFile) { Remove-Item -LiteralPath $tempFile -Force -ErrorAction SilentlyContinue }
        }
        if ((Test-Path -LiteralPath $chunkFile) -and ((Get-Item -LiteralPath $chunkFile).Length -gt $expected)) {
            Remove-Item -LiteralPath $chunkFile -Force -ErrorAction SilentlyContinue
        }

        $chunkState = 'pending'
        if ((Test-Path -LiteralPath $chunkFile) -and ((Get-Item -LiteralPath $chunkFile).Length -eq $expected)) {
            $chunkState = 'done'
        }
        $chunks.Add([pscustomobject]@{
            Index = $i; Start = $start; End = $end; Expected = $expected
            ChunkFile = $chunkFile; TempFile = $tempFile; ErrFile = $errFile
            State = $chunkState; Attempts = 0; NextAttempt = (Get-Date); Proc = $null
        })
    }
    $resumed = @($chunks | Where-Object { $_.State -eq 'done' }).Count
    Write-Log ('PLAN file={0} length={1} chunks={2} chunk_size={3} concurrency={4} already_done={5}' -f $Name, $remoteLen, $spec.Count, (Format-Bytes $spec.ChunkSize), $Concurrency, $resumed)

    # --- parallel download loop
    $active = @{}
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $lastProgress = Get-Date
    $lastBytes = [long]0
    $lastTime = Get-Date

    while (@($chunks | Where-Object { $_.State -ne 'done' }).Count -gt 0) {
        $now = Get-Date
        # --- start as many chunks as concurrency allows (each from its current byte offset)
        foreach ($c in $chunks) {
            if ($active.Count -ge $Concurrency) { break }
            if ($c.State -ne 'pending') { continue }
            if ($now -lt $c.NextAttempt) { continue }

            $curLen = [long]0
            if (Test-Path -LiteralPath $c.ChunkFile) { $curLen = (Get-Item -LiteralPath $c.ChunkFile).Length }
            if ($curLen -eq $c.Expected) { $c.State = 'done'; continue }
            if ($curLen -gt $c.Expected) {
                Remove-Item -LiteralPath $c.ChunkFile -Force -ErrorAction SilentlyContinue
                $curLen = 0
            }
            $reqStart = $c.Start + $curLen
            if ($reqStart -gt $c.End) {
                Remove-Item -LiteralPath $c.ChunkFile -Force -ErrorAction SilentlyContinue
                $curLen = 0
                $reqStart = $c.Start
            }
            if (Test-Path -LiteralPath $c.TempFile) { Remove-Item -LiteralPath $c.TempFile -Force -ErrorAction SilentlyContinue }

            $range = '{0}-{1}' -f $reqStart, $c.End
            $curlArgs = @(
                '-sS', '--fail', '--location',
                '--connect-timeout', '30',
                '--max-time', '1800',
                '--speed-limit', '200', '--speed-time', '120',
                '-r', $range,
                '-o', $c.TempFile,
                $Url
            )
            try {
                $c.Proc = Start-Process -FilePath 'curl.exe' -ArgumentList $curlArgs -PassThru -RedirectStandardError $c.ErrFile
                $c.State = 'running'
                $active[$c.Index] = $c
            } catch {
                $c.Attempts++
                if ($c.Attempts -ge $MaxRetries) {
                    throw ('chunk {0} of {1}: could not start curl after {2} attempts: {3}' -f $c.Index, $Name, $c.Attempts, $_.Exception.Message)
                }
                $delay = [int]([math]::Min(60, [math]::Pow(2, $c.Attempts))) + (Get-Random -Minimum 0 -Maximum 3)
                $c.NextAttempt = (Get-Date).AddSeconds($delay)
                Write-Log ('RETRY file={0} chunk={1} attempt={2}/{3} start-failed delay={4}s err={5}' -f $Name, $c.Index, $c.Attempts, $MaxRetries, $delay, $_.Exception.Message)
            }
        }

        # --- collect finished attempts: append whatever bytes arrived, then judge by exact size
        $finished = @()
        foreach ($idx in @($active.Keys)) {
            $c = $active[$idx]
            try {
                if ($c.Proc.HasExited) { $finished += $c }
            } catch {
                $finished += $c
            }
        }
        foreach ($c in $finished) {
            [void]$active.Remove($c.Index)
            try { $c.Proc.Dispose() } catch { }

            $curLen = [long]0
            if (Test-Path -LiteralPath $c.ChunkFile) { $curLen = (Get-Item -LiteralPath $c.ChunkFile).Length }
            if (Test-Path -LiteralPath $c.TempFile) {
                $tmpLen = (Get-Item -LiteralPath $c.TempFile).Length
                if (($tmpLen -gt 0) -and ($curLen -le $c.Expected) -and (($curLen + $tmpLen) -le $c.Expected)) {
                    if ($curLen -gt 0) { Append-File -Source $c.TempFile -Dest $c.ChunkFile }
                    else { Move-Item -LiteralPath $c.TempFile -Destination $c.ChunkFile -Force }
                }
                if (Test-Path -LiteralPath $c.TempFile) { Remove-Item -LiteralPath $c.TempFile -Force -ErrorAction SilentlyContinue }
                $curLen = [long]0
                if (Test-Path -LiteralPath $c.ChunkFile) { $curLen = (Get-Item -LiteralPath $c.ChunkFile).Length }
            }

            if ($curLen -eq $c.Expected) { $c.State = 'done'; continue }
            if ($curLen -gt $c.Expected) {
                Remove-Item -LiteralPath $c.ChunkFile -Force -ErrorAction SilentlyContinue
                $curLen = 0
            }

            $errText = ''
            if (Test-Path -LiteralPath $c.ErrFile) {
                $errText = (Get-Content -LiteralPath $c.ErrFile -Raw -ErrorAction SilentlyContinue)
                if ($null -eq $errText) { $errText = '' }
                $errText = ($errText | Out-String).Trim()
                if ($errText.Length -gt 200) { $errText = $errText.Substring(0, 200) }
            }
            $c.Attempts++
            if ($c.Attempts -ge $MaxRetries) {
                throw ('chunk {0} of {1} failed after {2} attempts (have={3} expected={4} err={5})' -f $c.Index, $Name, $c.Attempts, $curLen, $c.Expected, $errText)
            }
            $delay = [int]([math]::Min(60, [math]::Pow(2, $c.Attempts))) + (Get-Random -Minimum 0 -Maximum 3)
            $c.NextAttempt = (Get-Date).AddSeconds($delay)
            $c.State = 'pending'
            Write-Log ('RETRY file={0} chunk={1} attempt={2}/{3} have={4}/{5} delay={6}s err={7}' -f $Name, $c.Index, $c.Attempts, $MaxRetries, (Format-Bytes $curLen), (Format-Bytes $c.Expected), $delay, $errText)
        }

        # --- progress log every N seconds
        $now = Get-Date
        if (($now - $lastProgress).TotalSeconds -ge $ProgressIntervalSec) {
            $downloaded = [long]0
            $doneChunks = 0
            foreach ($c in $chunks) {
                if (Test-Path -LiteralPath $c.ChunkFile) { $downloaded += (Get-Item -LiteralPath $c.ChunkFile).Length }
                if (Test-Path -LiteralPath $c.TempFile) { $downloaded += (Get-Item -LiteralPath $c.TempFile).Length }
                if ($c.State -eq 'done') { $doneChunks++ }
            }
            $dt = ($now - $lastTime).TotalSeconds
            $inst = 0.0
            if ($dt -gt 0) { $inst = ($downloaded - $lastBytes) / $dt }
            $pct = 0.0
            if ($remoteLen -gt 0) { $pct = 100.0 * $downloaded / $remoteLen }
            Write-Log ('PROGRESS file={0} downloaded={1}/{2} ({3:N1}%) speed={4}/s avg={5}/s chunks={6}/{7} elapsed={8}' -f $Name, (Format-Bytes $downloaded), (Format-Bytes $remoteLen), $pct, (Format-Bytes $inst), (Format-Bytes ($downloaded / [math]::Max(1.0, $sw.Elapsed.TotalSeconds))), $doneChunks, $spec.Count, (Format-Duration $sw.Elapsed.TotalSeconds))
            $lastProgress = $now
            $lastBytes = $downloaded
            $lastTime = $now
        }
        Start-Sleep -Milliseconds 1000
    }
    Write-Log ('DOWNLOADED file={0} bytes={1} elapsed={2}' -f $Name, $remoteLen, (Format-Duration $sw.Elapsed.TotalSeconds))

    # --- merge chunks -> .part
    Remove-StalePart -Path $partPath
    Write-Log ('MERGE file={0} -> {1}' -f $Name, $partPath)
    $written = [long]0
    $buf = New-Object byte[] (1048576)
    $out = [System.IO.File]::Open($partPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        foreach ($c in ($chunks | Sort-Object Index)) {
            $in = [System.IO.File]::OpenRead($c.ChunkFile)
            try {
                while (($r = $in.Read($buf, 0, $buf.Length)) -gt 0) {
                    $out.Write($buf, 0, $r)
                    $written += $r
                }
            } finally { $in.Dispose() }
        }
    } finally { $out.Dispose() }

    if ($written -ne $remoteLen) {
        Remove-StalePart -Path $partPath
        throw ('merge byte count mismatch for {0}: written={1} expected={2}' -f $Name, $written, $remoteLen)
    }
    $partLen = (Get-Item -LiteralPath $partPath).Length
    if ($partLen -ne $remoteLen) {
        Remove-StalePart -Path $partPath
        throw ('merged file size mismatch for {0}: got={1} expected={2}' -f $Name, $partLen, $remoteLen)
    }

    # --- validate npz (zip CRC over every member + numpy load)
    $v = Test-Npz -Path $partPath -Python $PythonExe
    if (-not $v.Ok) {
        Remove-StalePart -Path $partPath
        if (Test-Path -LiteralPath $cacheFileDir) { Remove-Item -LiteralPath $cacheFileDir -Recurse -Force -ErrorAction SilentlyContinue }
        throw ('npz validation failed for {0}: {1} (chunk cache cleared, will re-download)' -f $Name, $v.Detail)
    }

    # --- atomic-ish rename
    Move-Item -LiteralPath $partPath -Destination $finalPath -Force
    $elapsed = ((Get-Date) - $fileStart).TotalSeconds
    $avg = $remoteLen / [math]::Max(1.0, $elapsed)
    if (Test-Path -LiteralPath $cacheFileDir) { Remove-Item -LiteralPath $cacheFileDir -Recurse -Force -ErrorAction SilentlyContinue }
    Write-Log ('DONE file={0} size={1} validation="{2}" elapsed={3} avg_speed={4}/s' -f $Name, $remoteLen, $v.Detail, (Format-Duration $elapsed), (Format-Bytes $avg))
    return [pscustomobject]@{ Name = $Name; Status = 'DONE'; Size = $remoteLen; Elapsed = $elapsed; Validation = $v.Detail }
}

# ---------------------------------------------------------------- main
$scriptStart = Get-Date
New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
$logDir = Split-Path -Parent $LogFile
if ($logDir -and -not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }

Write-Log '=== download_ogbench.ps1 START ==='
Write-Log ('CONFIG dest={0} cache={1} concurrency={2} chunk_mb={3} max_retries={4} python={5}' -f $DestDir, $CacheDir, $Concurrency, $ChunkMB, $MaxRetries, $PythonExe)

$queue = @(
    @{ Name = 'pointmaze-medium-navigate-v0.npz';     Url = 'https://rail.eecs.berkeley.edu/datasets/ogbench/pointmaze-medium-navigate-v0.npz' },
    @{ Name = 'pointmaze-medium-navigate-v0-val.npz'; Url = 'https://rail.eecs.berkeley.edu/datasets/ogbench/pointmaze-medium-navigate-v0-val.npz' },
    @{ Name = 'antmaze-large-navigate-v0.npz';        Url = 'https://rail.eecs.berkeley.edu/datasets/ogbench/antmaze-large-navigate-v0.npz' },
    @{ Name = 'antmaze-large-navigate-v0-val.npz';    Url = 'https://rail.eecs.berkeley.edu/datasets/ogbench/antmaze-large-navigate-v0-val.npz' }
)
if ($IncludeOptional) {
    $queue += @(
        @{ Name = 'cube-double-play-v0.npz';     Url = 'https://rail.eecs.berkeley.edu/datasets/ogbench/cube-double-play-v0.npz' },
        @{ Name = 'cube-double-play-v0-val.npz'; Url = 'https://rail.eecs.berkeley.edu/datasets/ogbench/cube-double-play-v0-val.npz' }
    )
}
if ($Only.Count -gt 0) {
    $queue = @($queue | Where-Object { $Only -contains $_.Name })
}
if ($queue.Count -eq 0) { throw 'empty download queue' }

$results = @()
foreach ($item in $queue) {
    $done = $null
    for ($fa = 1; $fa -le 2; $fa++) {
        try {
            $done = Download-File -Name $item.Name -Url $item.Url
            break
        } catch {
            Write-Log ('FAILED file={0} attempt={1}/2 err={2}' -f $item.Name, $fa, (($_.Exception.Message | Out-String).Trim()))
            if ($fa -ge 2) { throw }
            Start-Sleep -Seconds 10
        }
    }
    $results += $done
}

$totalElapsed = ((Get-Date) - $scriptStart).TotalSeconds
$totalBytes = [long]0
foreach ($r in $results) { $totalBytes += [long]$r.Size }
$nDone = @($results | Where-Object { $_.Status -eq 'DONE' }).Count
$nSkip = @($results | Where-Object { $_.Status -eq 'SKIP' }).Count
Write-Log ('SUMMARY files={0} downloaded={1} skipped={2} total_bytes={3} total_elapsed={4}' -f $results.Count, $nDone, $nSkip, $totalBytes, (Format-Duration $totalElapsed))
foreach ($r in $results) {
    Write-Log ('RESULT file={0} status={1} bytes={2} elapsed={3}s validation="{4}"' -f $r.Name, $r.Status, $r.Size, [math]::Round($r.Elapsed, 1), $r.Validation)
}
Write-Log '### DOWNLOAD SCRIPT COMPLETE ###'
