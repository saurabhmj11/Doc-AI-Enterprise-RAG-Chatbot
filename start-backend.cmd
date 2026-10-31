@echo off
REM Start the Doc-AI API (FastAPI) on http://localhost:8000
cd /d "%~dp0backend"
.venv\Scripts\python.exe -m uvicorn main:app --port 8000
