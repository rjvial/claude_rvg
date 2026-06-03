# claude_rvg

Gmail → Neo4j knowledge graph. Pulls mail from multiple Gmail accounts and
loads it into a single Neo4j graph tagged by `account_owner`, then serves a
local app for graph-RAG Q&A and assisted compose. Backfill once, daily
incremental sync.

**Deterministic entity layer — no LLM entity extraction.** Organizations are
derived from email domains; keywords come from a curated concept vocabulary.

## What it does

- Multi-account Gmail + Calendar backfill and daily incremental sync
- Cleans message bodies and reconstructs the true RFC reply tree per thread
- Loads People / Orgs / Threads / Messages / Attachments / Events into Neo4j
- Local embeddings (`intfloat/multilingual-e5-large`) + a vector index for
  graph-RAG retrieval over whole conversations
- Multi-thread **Matter** unification across related threads
- Local web app to ask questions over your mail and draft replies in your own
  writing style

## Stack

Python · Neo4j 5.x (native install) · sentence-transformers · Gmail/Calendar
APIs · Claude Code (`claude -p`) for the answer and compose steps.

## Setup

See [SETUP.md](SETUP.md) for one-time setup (GCP OAuth, Neo4j, venv).
Project conventions and the data model live in [CLAUDE.md](CLAUDE.md).

## Entry points

- `scripts/run_pipeline.py` — orchestrates pull → clean → load → embed → cluster
- `scripts/serve_app.py` — local app server
- `scripts/sync_incremental.py` — daily per-account delta sync

## Privacy

All runtime data — mail, OAuth tokens, `.env`, `.mcp.json` — lives under
`data/` (and a few root config files) and is **gitignored**. Example fixtures
and account labels in the source are generic placeholders, not real data.
