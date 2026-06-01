@echo off
REM Wrapper so download_from_runpod.ps1 works from cmd.exe, double-click, or anywhere else.
REM All arguments are forwarded to the PowerShell script.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_from_runpod.ps1" %*
