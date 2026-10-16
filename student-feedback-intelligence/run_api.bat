@echo off
cd /d "%~dp0\.."
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
pause
