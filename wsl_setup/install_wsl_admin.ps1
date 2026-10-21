# FINO 复现：WSL2 安装脚本（必须以管理员身份运行）
# 由 install_wsl_admin.cmd 调用；日志写入 E:\FINO\wsl_setup\install.log
$ErrorActionPreference = 'Continue'
$log = 'E:\FINO\wsl_setup\install.log'
New-Item -ItemType Directory -Force -Path 'E:\FINO\wsl_setup' | Out-Null
function L([string]$m) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Write-Host $line
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
L ("START admin={0} user={1}" -f $isAdmin, [Security.Principal.WindowsIdentity]::GetCurrent().Name)
if (-not $isAdmin) {
    L 'ERROR: not elevated. Right-click install_wsl_admin.cmd -> Run as administrator.'
    exit 1
}

L 'STEP 1/5: enable feature Microsoft-Windows-Subsystem-Linux'
& dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
L ("STEP 1 exit={0}" -f $LASTEXITCODE)

L 'STEP 2/5: enable feature VirtualMachinePlatform'
& dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
L ("STEP 2 exit={0}" -f $LASTEXITCODE)

L 'STEP 3/5: wsl --update (downloads WSL2 kernel; may take a few minutes)'
& wsl.exe --update --web-download 2>&1 | ForEach-Object { L ("  " + $_) }
L ("STEP 3 exit={0}" -f $LASTEXITCODE)

L 'STEP 4/5: wsl --set-default-version 2'
& wsl.exe --set-default-version 2 2>&1 | ForEach-Object { L ("  " + $_) }
L ("STEP 4 exit={0}" -f $LASTEXITCODE)

L 'STEP 5/5: import Ubuntu rootfs -> E:\FINO\wsl\FINO-Ubuntu (may fail before reboot; retried after reboot)'
$distroDir = 'E:\FINO\wsl\FINO-Ubuntu'
$rootfs = 'E:\FINO\wsl_setup\ubuntu-base-24.04.3.tar.gz'
if (-not (Test-Path -LiteralPath $rootfs)) {
    L "ERROR: rootfs not found: $rootfs"
} elseif (Test-Path -LiteralPath (Join-Path $distroDir 'ext4.vhdx')) {
    L 'distro already imported, skip'
} else {
    New-Item -ItemType Directory -Force -Path $distroDir | Out-Null
    & wsl.exe --import FINO-Ubuntu $distroDir $rootfs --version 2 2>&1 | ForEach-Object { L ("  " + $_) }
    L ("STEP 5 exit={0}" -f $LASTEXITCODE)
}

L 'ALL STEPS DONE. Please REBOOT Windows, then reopen DeepSeek Harness and say "已重启".'
