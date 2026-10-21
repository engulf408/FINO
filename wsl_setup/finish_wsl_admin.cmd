@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo  FINO reproduction - WSL2 after-reboot setup
echo  Log: E:\FINO\wsl_setup\install_after_reboot.log
echo ============================================================
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0finish_wsl_admin.ps1"
echo.
echo ============================================================
echo  Done. Close this window and continue in DeepSeek Harness.
echo ============================================================
pause
