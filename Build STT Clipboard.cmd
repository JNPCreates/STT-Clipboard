@echo off
setlocal
cd /d "%~dp0"
if defined STT_PYTHON goto run
set "STT_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%STT_PYTHON%" goto run
set "STT_PYTHON=%~dp0..\..\..\.venv\Scripts\python.exe"
if exist "%STT_PYTHON%" goto run
set "STT_PYTHON=python"
:run
"%STT_PYTHON%" "%~dp0build_exe.py" %*
pause
