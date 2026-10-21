@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo  FINO reproduction - WSL2 setup (requires administrator)
echo  Log: E:\FINO\wsl_setup\install.log
echo ============================================================
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_wsl_admin.ps1"
echo.
echo ============================================================
echo  If you see "ALL STEPS DONE", please REBOOT Windows now.
echo ============================================================
pause
