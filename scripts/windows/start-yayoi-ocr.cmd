@echo off
rem yayoi-ocr launcher (trial). Keep this file ASCII-only (cmd.exe code page).
setlocal
set "HERE=%~dp0"
set "YAYOI_MODELS_DIR=%HERE%models"
if not exist "%HERE%venv\Scripts\python.exe" (
  echo Run setup.ps1 first. venv not found: %HERE%venv
  pause
  exit /b 1
)
rem Console is minimized (closing it quits the app; error messages stay visible there).
start "yayoi-ocr" /min "%HERE%venv\Scripts\python.exe" -m yayoi_ocr.app
endlocal
