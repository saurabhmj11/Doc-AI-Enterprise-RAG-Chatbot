@echo off
REM Start the Doc-AI web UI (Vite) on http://localhost:5173 - proxies /api to :8000
cd /d "%~dp0frontend"
npx vite --port 5173 --strictPort
