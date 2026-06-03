# claude\_rvg

Gmail → Neo4j knowledge graph. Pulls mail from three accounts (you@gmail.com, you@work.example.com, you@org.example.com) and loads it into a single Neo4j graph tagged by `account\\\_owner`. Backfill once, daily incremental sync.

**Deterministic only — no LLM entity extraction.** Orgs come from email-domain (Layer 2, via `data/orgs\\\_seed.json`). Keywords come from the curated `data/concepts.json` vocabulary (Layer 3b). LLM extraction was removed 2026-05-25; see `scripts/wipe\\\_llm\\\_extraction.cypher` for the one-time Neo4j migration.

See `SETUP.md` for one-time setup (GCP OAuth, native Neo4j, venv). Neo4j runs as a **native Windows install** (no Docker, no WSL) under `C:\neo4j`; the app reaches it over Bolt at `bolt://localhost:7687` and, if no `neo4j` Windows service is registered, launches the bundled `neo4j console` itself.

## "run plan.md" protocol

When the user says **"run plan.md"** (or "run pipeline"), Claude MUST run this protocol BEFORE invoking `run\\\_pipeline.py`. Never assume defaults, never erase without an explicit ask.

### Step 0 — Detect preloaded state, ask keep/erase

Probe for preloaded data:

* **Files:** check whether any of `data/emails.jsonl`, `emails\\\_clean.jsonl`, `pulled\\\_msg\\\_ids\\\_<label>.txt`, `cleaned\\\_msg\\\_ids.txt`, `sync\\\_state\\\_<label>.json` exist.
* **Graph:** `MATCH (n) RETURN count(n) AS n` via the Neo4j MCP read tool.

Report what's there (sizes / line counts / node-count-by-label). Then ask **two independent questions** — they may be answered differently:

* **"Existing pulled data files detected (X bytes, Y lines). Keep or erase?"** → erase translates to `--reset-data`.
* **"Existing Neo4j graph has N nodes. Keep or erase?"** → erase translates to `--reset-graph`.

If both probes come back empty, skip Step 0 entirely.

### Step 1 — Backfill window

Ask: **"How many months of emails do you want to retrieve?"** — number or `all`.

### Step 2 — Invoke the orchestrator

```powershell
python scripts\\\\run\\\_pipeline.py --months <N> \\\[--reset-data] \\\[--reset-graph]
python scripts\\\\run\\\_pipeline.py --all-time   \\\[--reset-data] \\\[--reset-graph]
```

Orchestrator runs: pre-flight (venv, OAuth, Neo4j reachable over Bolt) → preloaded-state probe → optional reset → cleanup → pull → repair → clean → build-concepts → load → embed → cluster-matters → sanity → outputs-summary. All idempotent; interrupt and re-run freely (the embed step is a resumable checkpoint — only Messages still missing an embedding get encoded, so a re-run resumes where the previous one stopped). **Pulls stay sequential** — `pull\\\_gmail.py` has no file lock; parallel writes corrupt `emails.jsonl`.

## Entry points

* `scripts/run\\\_pipeline.py` — orchestrates pull → repair → clean → build-concepts → load → embed → cluster
* `scripts/sync\\\_incremental.py` — daily delta sync (per account)
* `scripts/serve\\\_app.py` — local app server (also via `mail.bat`); caches the rendered page at startup, so restart it after editing `graph\\\_app.py`
* `scripts/pull\\\_gmail.py` / `pull\\\_calendar.py` — per-account pullers; **run sequentially**, never in parallel
* `scripts/style\\\_profiles.py` — autonomous per-recipient writing-style learner for Liam Compose. At serve\\\_app boot a background pass mines the user's own sent mail to their top-50 recipients (addresses of one person merged by normalised name) and distils, via one `claude -p` (Haiku) call each, a "style card" of the traits common to the majority of those emails (greeting, sign-off, language, formality, length, recurring phrases). Stored in `data/style\\\_profiles.json`; injected into `_do\\\_compose\\\_draft`'s first turn so a draft to that person reads as if the user wrote it. Incremental (only new/stale/grown recipients re-distil). Manual control: `GET /api/style`, `POST /api/style/rebuild {force?}`

## Neo4j schema

**Nodes** (key in parens):

* `Person` (`email`) — `name`
* `Org` (`canonical\\\_name`) — `domain?`, `aliases\\\[]` (email-domain derived only)
* `Thread` (compound: `gmail\\\_thread\\\_id` + `account\\\_owner`)
* `Message` (compound: `gmail\\\_message\\\_id` + `account\\\_owner`) — `sent\\\_at`, `body\\\_clean`, `embedding`
* `Attachment` (compound: `gmail\\\_message\\\_id` + `account\\\_owner` + `part\\\_id`)
* `Concept` (`key`) — controlled vocabulary from `build\\\_concepts.py`
* `Event` — calendar events, deduped on `(ical\\\_uid, start)`
* `Matter` (`canonical\\\_key`) — multi-thread unification node from `cluster\\\_matters.py`

**Relationships:**

* `Person -\\\[:SENT]-> Message`, `Message -\\\[:RECEIVED\\\_BY {kind}]-> Person`
* `Message -\\\[:IN\\\_THREAD {seq}]-> Thread`, `Thread -\\\[:PART\\\_OF]-> Matter`
* `Message -\\\[:HAS\\\_ATTACHMENT]-> Attachment`
* `Message -\\\[:REPLY\\\_TO]-> Message`, `Message -\\\[:NEXT\\\_IN\\\_THREAD]-> Message`, `Thread -\\\[:STARTS\\\_WITH]-> Message`
* `Person -\\\[:WORKS\\\_AT]-> Org` (from email domain via `orgs\\\_seed.json`)
* `Message -\\\[:MENTIONS {count, in\\\_subject}]-> Concept`, `Thread -\\\[:MENTIONS]-> Concept`
* `Event -\\\[:ORGANIZED]-> Person`, `Event -\\\[:INVITED]-> Person`

Vector index on `Message.embedding` (1024-dim, cosine) for graph-RAG. Full-text index over subject + body\_clean.

## Data layout

All runtime data lives under `data/` (gitignored). OAuth lives in `data/credentials.json` + `data/token\\\_<label>.json`. Never copy these outside `data/`. Liam's learned state also lives here: `data/compose\\\_memory.json` + `data/ask\\\_memory.json` (long-term style/fact memory, per scope) and `data/style\\\_profiles.json` (per-recipient writing-style cards from `style\\\_profiles.py`). All are derived/rebuildable — safe to delete. `data/recipient\\\_aliases.json` is the one HAND-EDITED file here: groups of email addresses that are the same person but whose names don't auto-merge (edit it, then `POST /api/style/rebuild`).

