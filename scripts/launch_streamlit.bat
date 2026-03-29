@echo off
setlocal
cd /d "%~dp0\.."
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
