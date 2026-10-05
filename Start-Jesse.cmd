@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Jesse needs the project virtual environment. Follow README.md Setup first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m jesse run --ollama --ui %*
set "JESSE_EXIT=%ERRORLEVEL%"
if not "%JESSE_EXIT%"=="0" pause
exit /b %JESSE_EXIT%
