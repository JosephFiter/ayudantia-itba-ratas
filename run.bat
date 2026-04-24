@echo off
cd /d "%~dp0"
.venv\Scripts\streamlit.exe run app/main.py --server.maxUploadSize 2000
