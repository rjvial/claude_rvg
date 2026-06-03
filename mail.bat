@echo off
REM Start the always-live mail app server, then open it in the browser.
REM Leave this window open — it IS the running app. Close it to stop.
set "VENV=%USERPROFILE%\.venvs\claude_rvg\Scripts\python.exe"
if exist "%VENV%" (
  "%VENV%" "%~dp0scripts\serve_app.py" %*
) else (
  python "%~dp0scripts\serve_app.py" %*
)
if errorlevel 1 pause
