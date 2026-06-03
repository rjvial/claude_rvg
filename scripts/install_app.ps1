# Install "Mail Graph" as a self-sufficient desktop app, then launch it.
#
# What it does (idempotent — safe to re-run):
#   1. Generates the app icon (data/mailgraph.ico).
#   2. Creates "Mail Graph" shortcuts on the Desktop and Start menu that launch
#      the server with a hidden console (via mailgraph_launch.vbs) and that
#      starts Neo4j + the server if needed, then opens the standalone window.
#   3. Launches the app right now, so install + open happen in one action.
#
# Run it via "Install Mail Graph.bat" (double-click) or directly:
#   powershell -ExecutionPolicy Bypass -File scripts\install_app.ps1

$ErrorActionPreference = "Stop"

$Root     = Split-Path -Parent $PSScriptRoot         # repo root (scripts\..)
$Scripts  = Join-Path $Root "scripts"
$Launcher = Join-Path $Scripts "mailgraph_launch.vbs"
$Ico      = Join-Path $Root "data\mailgraph.ico"
$WScript  = Join-Path $env:SystemRoot "System32\wscript.exe"

# Prefer the project venv interpreter; fall back to PATH.
$VenvDir = Join-Path $env:USERPROFILE ".venvs\claude_rvg\Scripts"
$Py  = Join-Path $VenvDir "python.exe"
if (-not (Test-Path $Py))  { $Py  = "python" }

Write-Host "Generating app icon..."
& $Py (Join-Path $Scripts "make_icon.py")

# Copy the icon to a LOCAL path. The repo lives on Google Drive (I:), which is
# often not mounted yet at login when Windows draws the desktop — an IconLocation
# on I: then fails to load and the shortcut shows the blank-page fallback. A copy
# under %LOCALAPPDATA% is always readable, so the icon renders reliably.
$LocalIcoDir = Join-Path $env:LOCALAPPDATA "MailGraph"
$LocalIco    = Join-Path $LocalIcoDir "mailgraph.ico"
if (Test-Path $Ico) {
    New-Item -ItemType Directory -Force -Path $LocalIcoDir | Out-Null
    Copy-Item -Path $Ico -Destination $LocalIco -Force
    $IconFor = $LocalIco
} else {
    $IconFor = $Ico
}

# Shortcuts launch via wscript -> the hidden-console VBS launcher (NOT pythonw),
# so the Ask feature's child CLIs (claude, uvx) inherit a console and never pop
# a terminal window. The server shuts itself down when the window is closed, so
# there's no separate stop step.
function New-Shortcut($LinkPath) {
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($LinkPath)
    $sc.TargetPath       = $WScript
    $sc.Arguments        = '"' + $Launcher + '"'
    $sc.WorkingDirectory = $Root
    $sc.Description       = "Mail Graph"
    if (Test-Path $IconFor) { $sc.IconLocation = "$IconFor,0" }
    $sc.Save()
    Write-Host "  $LinkPath"
}

Write-Host "Creating shortcuts..."
$Desktop = [Environment]::GetFolderPath("Desktop")
$Menu    = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
New-Shortcut (Join-Path $Desktop "Mail Graph.lnk")
New-Shortcut (Join-Path $Menu    "Mail Graph.lnk")

Write-Host "Launching Mail Graph..."
Start-Process -FilePath $WScript -ArgumentList ('"' + $Launcher + '"') -WorkingDirectory $Root

Write-Host ""
Write-Host "Done. Use the 'Mail Graph' icon to start (Desktop or Start menu) -"
Write-Host "it starts everything and opens in its own window. Closing the"
Write-Host "window shuts the server down automatically."
