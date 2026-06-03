' Mail Graph launcher.
'
' Starts serve_app with a HIDDEN console (python.exe, window style 0) rather
' than pythonw.exe (which has no console at all). The difference matters: the
' graph-RAG "Ask" shells out to claude.exe, which in turn spawns uvx for the
' Neo4j MCP server. With a no-console parent (pythonw) those children pop their
' own terminal windows and a console CLI can hang with no console to attach to.
' Giving the server a hidden console lets the whole child tree inherit it — no
' popup terminals, and the CLIs run normally. The window is never shown.
Option Explicit
Dim sh, fso, here, root, py, script, cmd
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)          ' ...\scripts
root = fso.GetParentFolderName(here)                            ' repo root
py = sh.ExpandEnvironmentStrings("%USERPROFILE%\.venvs\claude_rvg\Scripts\python.exe")
If Not fso.FileExists(py) Then py = "python"
script = fso.BuildPath(here, "serve_app.py")
sh.CurrentDirectory = root
cmd = """" & py & """ """ & script & """"
sh.Run cmd, 0, False    ' 0 = hidden window, False = fire-and-forget
