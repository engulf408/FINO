# FINO 复现：重启后的 WSL2 收尾脚本（以管理员身份运行一次即可）
# 做三件事：确认/安装 WSL2 内核 -> 默认 WSL2 -> 导入 Ubuntu 到 E:\FINO\wsl\FINO-Ubuntu
$ErrorActionPreference = 'Continue'
$log = 'E:\FINO\wsl_setup\install_after_reboot.log'
New-Item -ItemType Directory -Force -Path 'E:\FINO\wsl_setup' | Out-Null
function L([string]$m) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Write-Host $line
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
L ("START admin={0}" -f $isAdmin)

L 'STEP 1: check wsl --version'
& wsl.exe --version 2>&1 | ForEach-Object { L ("  " + $_) }
$wslOK = ($LASTEXITCODE -eq 0)
L ("wsl_version_ok={0}" -f $wslOK)

if (-not $wslOK) {
    L 'STEP 1b: install WSL kernel MSI (wsl_update_x64.msi)'
    $msi = 'E:\FINO\wsl_setup\wsl_update_x64.msi'
    if (Test-Path -LiteralPath $msi) {
        $p = Start-Process msiexec.exe -ArgumentList "/i `"$msi`" /qn /norestart" -Wait -PassThru
        L ("msi exit={0}" -f $p.ExitCode)
    } else {
        L "WARN: $msi not found"
    }
    L 'STEP 1c: try wsl --update --web-download'
    & wsl.exe --update --web-download 2>&1 | ForEach-Object { L ("  " + $_) }
    L ("update exit={0}" -f $LASTEXITCODE)
}

L 'STEP 2: wsl --set-default-version 2'
& wsl.exe --set-default-version 2 2>&1 | ForEach-Object { L ("  " + $_) }
L ("exit={0}" -f $LASTEXITCODE)

L 'STEP 3: import Ubuntu rootfs'
$distroDir = 'E:\FINO\wsl\FINO-Ubuntu'
$rootfs = 'E:\FINO\wsl_setup\ubuntu-base-24.04.3.tar.gz'
$already = $false
& wsl.exe -l -q 2>&1 | ForEach-Object {
    if ($_ -match 'FINO-Ubuntu') { $already = $true }
}
if (-not (Test-Path -LiteralPath $rootfs)) {
    L "ERROR: rootfs not found: $rootfs"
} elseif ($already) {
    L 'distro FINO-Ubuntu already exists, skip import'
} else {
    New-Item -ItemType Directory -Force -Path $distroDir | Out-Null
    & wsl.exe --import FINO-Ubuntu $distroDir $rootfs --version 2 2>&1 | ForEach-Object { L ("  " + $_) }
    L ("import exit={0}" -f $LASTEXITCODE)
}

L 'STEP 4: verify distro'
& wsl.exe -l -v 2>&1 | ForEach-Object { L ("  " + $_) }
& wsl.exe -d FINO-Ubuntu -u root -- /bin/bash -lc "cat /etc/os-release | head -2; uname -r" 2>&1 | ForEach-Object { L ("  " + $_) }

L 'DONE. You can close this window; Linux-side setup continues from DeepSeek Harness.'
