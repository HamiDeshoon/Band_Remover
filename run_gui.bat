@echo off
title DeStripe - Band Artifact Remover (Python 3.8 / PyQt5)
cd /d "%~dp0"

if exist "D:\Program Files\Python\VENV\8\Scripts\python.exe" (
    "D:\Program Files\Python\VENV\8\Scripts\python.exe" main.py %*
) else (
    py -3.8 main.py %* || python main.py %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
