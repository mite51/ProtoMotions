@echo off
REM Wrapper so upload_to_runpod.ps1 works from cmd.exe, double-click, or anywhere else.
REM All arguments are forwarded to the PowerShell script.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0upload_to_runpod.ps1" %*
