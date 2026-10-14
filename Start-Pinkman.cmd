@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Pinkman needs the project virtual environment. Follow README.md Setup first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pinkman run --ollama --ui %*
set "PINKMAN_EXIT=%ERRORLEVEL%"
if not "%PINKMAN_EXIT%"=="0" pause
exit /b %PINKMAN_EXIT%
