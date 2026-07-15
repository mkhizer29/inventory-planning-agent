@echo off
setlocal EnableExtensions
REM ─────────────────────────────────────────────────────────────
REM  Inventory Planning Agent — ETL launcher (portable)
REM  Usage:
REM    run_etl.bat                                   (source from .env)
REM    run_etl.bat --source staging
REM    run_etl.bat --source local_backup --sales-since 2024-01-01
REM  Runnable by double-click or from cmd/PowerShell.
REM ─────────────────────────────────────────────────────────────

REM Repo root = the parent of this script's folder (script lives in inventory_etl\)
pushd "%~dp0.." >nul
set "REPO_ROOT=%CD%"

REM Prefer the repo virtual environment; otherwise fall back to Python on PATH.
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set "PY=%REPO_ROOT%\.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

REM Confirm the interpreter actually runs.
"%PY%" -c "import sys" 1>nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Could not locate a working Python interpreter.
    echo   Option A: create the repo virtual environment ^-^-^>  py -m venv .venv
    echo   Option B: install Python and make sure 'python' is on your PATH.
    popd >nul
    endlocal & exit /b 9009
)

REM Safety net: make 'etl' importable even without 'pip install -e .'.
set "PYTHONPATH=%REPO_ROOT%\inventory_etl;%PYTHONPATH%"

"%PY%" -m etl.run_etl %*
set "RC=%ERRORLEVEL%"
popd >nul

echo.
echo ============================================================
echo  ETL finished (exit code %RC%).
echo  Outputs: inventory_etl\output\
echo    - inventory.db              (SQLite warehouse)
echo    - csv\*.csv                 (one CSV per table)
echo    - data_quality_report.md    (report card)
echo ============================================================

REM Pause only when launched by double-click (so shells/CI don't hang).
echo %CMDCMDLINE% | findstr /L /I "%~f0" >nul 2>&1
if %ERRORLEVEL%==0 pause

endlocal & exit /b %RC%
