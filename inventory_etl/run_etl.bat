@echo off
REM ─────────────────────────────────────────────────────────────
REM  Inventory Planning Agent — ETL launcher (double-click to run)
REM  Usage:
REM    run_etl.bat                 -> uses ETL_SOURCE from .env (staging)
REM    run_etl.bat --source local_backup
REM    run_etl.bat --source staging --sales-since 2024-01-01
REM ─────────────────────────────────────────────────────────────
cd /d "%~dp0"
"C:\Users\Bilal\Python312emb\python.exe" -m etl.run_etl %*
echo.
echo ============================================================
echo  ETL finished. Outputs are in the "output" folder:
echo    - output\inventory.db          (database)
echo    - output\csv\*.csv             (spreadsheets)
echo    - output\data_quality_report.md (report card)
echo ============================================================
pause
