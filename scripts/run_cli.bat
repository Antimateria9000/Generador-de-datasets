@echo off
setlocal
cd /d "%~dp0\.."
".venv\Scripts\python.exe" export_ohlcv_csv.py %*
