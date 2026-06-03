# Setup Guide

End-to-end checklist to get from a fresh machine to a working Gmail → Neo4j
knowledge graph. Each step is a copy-pasteable command or a click-by-click in
the Google Cloud Console.

> Working directory throughout: `I:\Mi unidad\claude_rvg`

---

## 1. Install prerequisites

### Neo4j (native Windows — no Docker, no WSL)

Neo4j runs as a **native Windows install** under `C:\neo4j`. There is no Docker
and no WSL in the loop. The app reaches it over Bolt at `bolt://localhost:7687`;
if a `neo4j` Windows service is registered it uses that, otherwise it launches
the bundled `neo4j console` itself and stops it on exit.

Everything lives self-contained under `C:\neo4j` (off the I: Drive — Drive's
File Stream cannot host a database). Paths are overridable via the
`CLAUDE_RVG_NEO4J_HOME` / `CLAUDE_RVG_NEO4J_JAVA_HOME` / `CLAUDE_RVG_NEO4J_SERVICE`
env vars; the defaults match the layout below.

**a) Java 21 (bundled, Neo4j-only).** Neo4j 5.x supports only Java 17/21 — a
newer system JDK (e.g. 23) is unsupported. Use a portable Temurin JDK 21 pointed
at by Neo4j alone, leaving the system Java untouched:

```powershell
$ProgressPreference='SilentlyContinue'
New-Item -ItemType Directory -Force C:\neo4j\dl | Out-Null
Invoke-WebRequest "https://api.adoptium.net/v3/binary/latest/21/ga/windows/x64/jdk/hotspot/normal/eclipse?project=jdk" -OutFile C:\neo4j\dl\temurin-jdk21.zip
Expand-Archive C:\neo4j\dl\temurin-jdk21.zip -DestinationPath C:\neo4j -Force   # -> C:\neo4j\jdk-21.x.y+z
```

**b) Neo4j 5.26 Community.** Match the 5.26.x line (it carries the vector index
and `standard-folding` full-text analyzer the graph-RAG relies on):

```powershell
Invoke-WebRequest "https://dist.neo4j.org/neo4j-community-5.26.26-windows.zip" -OutFile C:\neo4j\dl\neo4j.zip
Expand-Archive C:\neo4j\dl\neo4j.zip -DestinationPath C:\neo4j -Force          # -> C:\neo4j\neo4j-community-5.26.26
```

**c) APOC plugin** (needed for `apoc.nodes.link` during load):

```powershell
Invoke-WebRequest "https://github.com/neo4j/apoc/releases/download/5.26.26/apoc-5.26.26-core.jar" -OutFile "C:\neo4j\neo4j-community-5.26.26\plugins\apoc-5.26.26-core.jar"
```

**d) Configure** — append to `conf\neo4j.conf` and create `conf\apoc.conf`:

```powershell
$N='C:\neo4j\neo4j-community-5.26.26'
Add-Content "$N\conf\neo4j.conf" @'

# === claude_rvg native install ===
server.memory.heap.initial_size=512m
server.memory.heap.max_size=2G
server.memory.pagecache.size=1G
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*
server.jvm.additional=--add-modules jdk.incubator.vector
'@
"apoc.export.file.enabled=true`napoc.import.file.enabled=true" | Out-File "$N\conf\apoc.conf" -Encoding ascii
```

**e) Set the password** (do this before the first start; must match `.env` —
see step 3):

```powershell
$env:JAVA_HOME='C:\neo4j\jdk-21.0.11+10'   # adjust to the extracted JDK folder
& 'C:\neo4j\neo4j-community-5.26.26\bin\neo4j-admin.bat' dbms set-initial-password '<your password>'
```

**f) (Recommended) Register a Windows service** so Neo4j is always warm and the
app pays no DB cold start. Needs an **Administrator** PowerShell:

```powershell
$env:JAVA_HOME='C:\neo4j\jdk-21.0.11+10'
& 'C:\neo4j\neo4j-community-5.26.26\bin\neo4j.bat' windows-service install
Set-Service -Name neo4j -StartupType Automatic
Start-Service neo4j
```

> Without the service the app still works — it launches `neo4j console` on boot
> and stops it on exit (≈25 s cold start instead of always-warm).

### Python 3.11 (specifically)
Use **3.11**, not 3.12+. Talon transitively requires `cchardet`, which has no
prebuilt Windows wheel for Python 3.10+ and would force a C compile against
MSVC Build Tools. We work around it with `faust-cchardet` (a drop-in fork
with wheels), which is published for 3.11 but not consistently for 3.12.

Verify:

```powershell
py -3.11 --version    # should print Python 3.11.x
```

If you don't have 3.11, install from
<https://www.python.org/downloads/release/python-3119/>.

### Python virtualenv + dependencies

> **Put the venv off Google Drive.** A `.venv` inside this repo lives on
> Drive's File Stream — every one of the thousands of file writes pip does
> goes through Drive's sync layer, making installs and script startup
> ~5–10× slower. Put it on the local SSD.

```powershell
$VENV = "$env:USERPROFILE\.venvs\claude_rvg"
py -3.11 -m venv $VENV
& "$VENV\Scripts\python.exe" -m pip install --upgrade pip

# Two-step install — see requirements.txt header for the reason.
& "$VENV\Scripts\python.exe" -m pip install -r requirements.txt
& "$VENV\Scripts\python.exe" -m pip install --no-deps talon
```

Activate it for the rest of this session:

```powershell
& "$env:USERPROFILE\.venvs\claude_rvg\Scripts\Activate.ps1"
```

Smoke-test:

```powershell
python -c "from talon import quotations; import neo4j; print('ok')"
```

---

## 2. Google Cloud / Gmail API — three accounts

You'll create **one** OAuth Desktop client and authorize **three** Google
accounts against it. This is the only manual setup step.

1. Go to <https://console.cloud.google.com/>.
2. Create a new project (e.g. `claude-rvg-gmail-kg`) or pick an existing
   sandbox one.
3. **APIs & Services → Library** → search "Gmail API" → **Enable**.
4. **APIs & Services → OAuth consent screen**:
   - User type: **External**.
   - App name: `claude-rvg`; user support email + developer email: your
     primary address.
   - Scopes: **Add or Remove Scopes** → filter `gmail.readonly` → check
     `https://www.googleapis.com/auth/gmail.readonly` → **Update**.
   - **Test users:** add **all three** Google accounts you want to graph:
     - `you@gmail.com`
     - `you@work.example.com`
     - `you@org.example.com`
   - Save through to the end.
5. **APIs & Services → Credentials → + Create credentials → OAuth client ID**:
   - Application type: **Desktop app**.
   - Name: `claude-rvg-desktop`.
   - **Create** → **Download JSON**.
6. Save the downloaded file as `data\credentials.json` (rename if needed —
   the file Google downloads has a long generated name).

Then run the OAuth handshake **once per account**. Each opens a browser
window; sign in with the Google account whose label you're using:

```powershell
python scripts\pull_gmail.py --account gmail     --auth
python scripts\pull_gmail.py --account work --auth
python scripts\pull_gmail.py --account org      --auth
```

Each writes `data\token_<label>.json`. Expected output per run:
`OK. Token saved to ...data\token_<label>.json`.

> The label is your choice — it just determines which `token_<label>.json`,
> `pulled_msg_ids_<label>.txt`, and `sync_state_<label>.json` get used.
> Pick something short and memorable per mailbox.

---

## 3. Start Neo4j

> **The store lives under `C:\neo4j\...\data`** (native, on the local SSD) — not
> on Google Drive. Drive's File Stream is a virtual filesystem with a tiny local
> cache and cannot host a database.

Create `.env` at the project root with the three Neo4j variables. The password
**must equal** the one you set in step 1e (`set-initial-password`):

```
NEO4J_PASSWORD=<the password from step 1e>
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
```

Save as UTF-8 *without* a BOM.

**Normally you don't start Neo4j by hand** — launching the app (`mail.bat` /
`serve_app.py`) makes it reachable automatically: it starts the `neo4j` Windows
service if one is registered (step 1f), otherwise it launches the bundled
`neo4j console` and stops that on exit. To start it manually:

```powershell
# If you registered the service (step 1f):
Start-Service neo4j
# Otherwise, run the bundled server in a console (Ctrl+C to stop):
$env:JAVA_HOME='C:\neo4j\jdk-21.0.11+10'
& 'C:\neo4j\neo4j-community-5.26.26\bin\neo4j.bat' console
```

Wait ~15 seconds for Neo4j to come online, then open <http://localhost:7474> in
a browser. Log in with `neo4j` / your password.

Apply schema constraints (the script reads `NEO4J_PASSWORD` from the
environment, so load it from `.env` first):

```powershell
$env:NEO4J_PASSWORD = (Get-Content .env | Where-Object { $_ -match '^NEO4J_PASSWORD=' }) -replace '^NEO4J_PASSWORD=',''
python scripts\load_neo4j.py --setup
```

---

## 4. Sample run (1 month) — verifies the pipeline end-to-end

**Step 1 — pull the last 30 days from each account.** Adjust the date:

```powershell
$since = (Get-Date).AddMonths(-1).ToString('yyyy-MM-dd')
python scripts\pull_gmail.py --account gmail     --since $since
python scripts\pull_gmail.py --account work --since $since
python scripts\pull_gmail.py --account org      --since $since
```

**Step 2 — strip quotes and signatures:**

```powershell
python scripts\clean_bodies.py
```

**Step 3 — build the concept vocabulary** (deterministic, zero tokens):

```powershell
python scripts\build_concepts.py
```

This scans `data\emails_clean.jsonl` and produces `data\concepts.json`,
the controlled keyword vocabulary used by `load_neo4j.py` to link every
message to the concepts it mentions.

**Step 4 — load to Neo4j:**

```powershell
python scripts\load_neo4j.py
```

Open <http://localhost:7474> and try a sanity query:

```cypher
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC;
```

---

## 5. Full backfill (12 months)

Once the sample looks good:

```powershell
$since = (Get-Date).AddMonths(-12).ToString('yyyy-MM-dd')
python scripts\pull_gmail.py --account gmail     --since $since
python scripts\pull_gmail.py --account work --since $since
python scripts\pull_gmail.py --account org      --since $since
python scripts\clean_bodies.py
python scripts\build_concepts.py
python scripts\load_neo4j.py
```

Or just `python scripts\run_pipeline.py --months 12` — the orchestrator
does all of this in order, plus calendar pulls, matter clustering, and a
sanity summary at the end. Fully resumable.

---

## 6. Incremental sync (daily)

Run one command per account; each does its own pull + clean + load on
whatever's new since the last run:

```powershell
python scripts\sync_incremental.py --account gmail
python scripts\sync_incremental.py --account work
python scripts\sync_incremental.py --account org
```

The pull uses Gmail's history API for cheap deltas. If the saved
`history_id` ages past Gmail's ~7-day retention (e.g. the routine stopped
for a week), it falls back to a date-based pull automatically.

To automate, use the `/schedule` skill in Claude Code to set up daily
routines — one per account, or one combined routine that chains all
three.

---

## Defaults you may want to override

| Setting           | Where                         | Default                                                   |
| ----------------- | ----------------------------- | --------------------------------------------------------- |
| Gmail query       | `pull_gmail.py` `--query` arg | `after:<--since> -category:promotions -social -updates -forums` |
| Concept min count | `build_concepts.py --min-msgs` | 10 (lower = more concepts)                               |
| Neo4j memory      | `C:\neo4j\neo4j-community-5.26.26\conf\neo4j.conf` | heap 2G, page cache 1G                |
| Neo4j install dir | `CLAUDE_RVG_NEO4J_HOME` env (default `C:\neo4j\neo4j-community-5.26.26`) | bundled JDK via `CLAUDE_RVG_NEO4J_JAVA_HOME` |
| Disclaimer regexes| `clean_bodies.py`             | Generic English + Spanish "confidential" patterns         |

---

## Troubleshooting

**`pull_gmail.py` fails with `invalid_grant`**
The cached `data\token_<label>.json` is stale or revoked. Delete that one
file and re-run `pull_gmail.py --account <label> --auth`.

**`pull_gmail.py --auth` opens browser but says "Access blocked: app has not
completed verification"**
Your Google account isn't on the Test Users list of the consent screen.
Go back to OAuth consent screen → Test users → add the missing email.

**`pull_gmail.py --auth` hangs on the localhost callback**
The flow uses `run_local_server(port=0)` which picks a free port and
listens for the OAuth callback. If your firewall blocks loopback, the
browser succeeds but Python never sees the callback. Allow Python through
Windows Defender Firewall, or run from a different shell.

**`load_neo4j.py` errors with `Neo.ClientError.Procedure.ProcedureNotFound: apoc.merge.node`**
APOC didn't load. Confirm `apoc-5.26.x-core.jar` is in
`C:\neo4j\neo4j-community-5.26.26\plugins\`, that `conf\neo4j.conf` has
`dbms.security.procedures.unrestricted=apoc.*`, then restart Neo4j. Check
`C:\neo4j\neo4j-community-5.26.26\logs\neo4j.log` for plugin-load lines.

**Neo4j won't start; `logs\neo4j.log` mentions an unsupported Java version**
Neo4j 5.x supports only Java 17/21. The service/console must use the bundled
JDK 21 — set `JAVA_HOME=C:\neo4j\jdk-21.0.11+10` before `neo4j console`, and
register the Windows service (step 1f) with that `JAVA_HOME` set so it bakes the
right JVM. A newer system JDK (e.g. 23) on PATH is *not* used by the app's
launch path, but a hand-run `neo4j console` without `JAVA_HOME` would pick it up.

**Port 7687 already in use / app says "Neo4j unavailable"**
Another Neo4j (or a stale `neo4j console` the app launched and didn't get to
stop) holds the port. Find and stop it:
`Get-NetTCPConnection -LocalPort 7687 -State Listen | %{ Get-Process -Id $_.OwningProcess }`,
then `Stop-Process` the `java.exe` under `C:\neo4j\...`. Neo4j is crash-safe.

**Auth fails right after first start with `unauthorized` then `too many times in a row`**
A few early auth misses (before the store finishes initialising) can trip
Neo4j's in-memory per-user rate-limiter, which then rejects even correct
passwords until it clears. Restart Neo4j (`Restart-Service neo4j`, or stop/start
the console) and let it settle. The app's boot poller retries until auth
succeeds.

**Wrong password / `.env` mismatch**
The app reads `NEO4J_PASSWORD` from the project-root `.env`; it must equal what
`neo4j-admin dbms set-initial-password` set (step 1e). To reset it, stop Neo4j
and re-run `set-initial-password`, or set a fresh one and update `.env`.

**Talon `extract_signature` raises**
Check `data\emails.jsonl` for that message — sometimes very short bodies break
the heuristic. The script already wraps this in a try/except and falls back to
the unstripped text, so this should only show as a warning.
