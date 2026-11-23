@echo off
rem SPDX-License-Identifier: AGPL-3.0-only
setlocal DisableDelayedExpansion
rem Double-click to set up and open CanopySplit Studio.
rem The execution policy flag applies only to this PowerShell process.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -Sta -ExecutionPolicy Bypass -File "%~dp0pipeline\start_windows.ps1" %*
if errorlevel 1 (
  echo.
  echo CanopySplit Studio could not start. You can retry by opening this file again.
  pause
  exit /b 1
)
