@echo off
REM ===================================================================
REM  Mail Graph — one-click installer.
REM  Double-click this file ONCE. It generates the app icon, creates
REM  "Mail Graph" shortcuts (Desktop + Start menu), and opens the app.
REM  Afterwards, just use the "Mail Graph" icon — it starts the server
REM  (and Neo4j) on its own and opens in a standalone window.
REM ===================================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_app.ps1"
if errorlevel 1 (
  echo.
  echo Install failed - see the message above.
  pause
)
