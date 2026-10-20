@echo off
title DeStripe - Band Artifact Remover
cd /d "%~dp0"
py -3 gui/destripe_app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
