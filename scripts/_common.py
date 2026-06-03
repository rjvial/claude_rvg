"""Shared helpers for the claude_rvg scripts.

Centralises boilerplate that used to be duplicated across the codebase:
venv bootstrap, .env loading, Neo4j driver factory, UTF-8 stream setup,
and the subprocess runner.

All entry scripts in scripts/ may `from _common import ...` directly —
Python puts the script's directory on sys.path[0] at startup, so the
import resolves whether the script is run by the user or as a subprocess
launched from run_pipeline.py / serve_app.py.
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCRIPTS = ROOT / "scripts"

# Account labels are no longer hardcoded — they live in data/accounts.json so
# the app can add/remove mailboxes at runtime. The three originals are the
# fallback when the file is absent (fresh checkout / first run). A label is
# used directly in per-account filenames (token_<label>.json, …) and as
# Message.account_owner in the graph, so it MUST be filesystem- and
# Cypher-safe: validate every externally-supplied label with
# valid_account_label() before it ever reaches a path or query.
ACCOUNTS_FILE = DATA_DIR / "accounts.json"
DEFAULT_ACCOUNTS = ["gmail", "work", "org"]
_LABEL_RE = re.compile(r"^[a-z0-9_-]{1,32}$")

# Gmail auto-categories we don't want in the graph: marketing, social-network
# notifications, bulk "updates", and group/forum mail. Single source of truth,
# used in three places that must stay in sync:
#   - pull_gmail.DEFAULT_QUERY_SUFFIX derives its -category: clauses from this
#     (excludes them from the backfill at the Gmail API level);
#   - sync_incremental drops messages carrying these labels in its fetch loop
#     (the history API ignores the search query, so the backfill filter alone
#     wouldn't catch incrementally-synced promo mail);
#   - load_neo4j skips them at load time — the authoritative gate, so the graph
#     stays category-free regardless of what's sitting in the jsonl files.
# Spam is always exempt (it's pulled deliberately for the Spam page and may also
# carry a CATEGORY_ label): callers gate on "SPAM not in labels" before this.
EXCLUDED_CATEGORIES = ["promotions", "social", "updates", "forums"]
# The matching Gmail label_ids (CATEGORY_<UPPER>).
EXCLUDED_CATEGORY_LABELS = frozenset(
    f"CATEGORY_{c.upper()}" for c in EXCLUDED_CATEGORIES
)


def is_excluded_category(label_ids) -> bool:
    """True if a message should be dropped for being promo/social/updates/forums.
    Spam is exempt — it's kept for the Spam page even when also categorized."""
    labels = label_ids or []
    if "SPAM" in labels:
        return False
    return bool(EXCLUDED_CATEGORY_LABELS.intersection(labels))


def valid_account_label(label: str) -> bool:
    """A label is safe iff it is 1–32 chars of [a-z0-9_-]. This is the single
    gate that keeps a user-supplied label out of path traversal
    (token_<label>.json) and Cypher injection (account_owner)."""
    return bool(isinstance(label, str) and _LABEL_RE.match(label))


def load_accounts() -> list[str]:
    """Current account labels, read fresh from data/accounts.json (so a long-
    lived process like serve_app sees runtime add/remove). Falls back to
    DEFAULT_ACCOUNTS when the file is missing or malformed; never writes (the
    file is created by save_accounts on the first add/remove). Invalid labels
    in the file are dropped defensively."""
    try:
        raw = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return list(DEFAULT_ACCOUNTS)
    if not isinstance(raw, list):
        return list(DEFAULT_ACCOUNTS)
    seen: set[str] = set()
    out: list[str] = []
    for label in raw:
        if valid_account_label(label) and label not in seen:
            seen.add(label)
            out.append(label)
    return out or list(DEFAULT_ACCOUNTS)


def save_accounts(labels: list[str]) -> None:
    """Persist the account-label list to data/accounts.json. Skips anything
    that fails validation so a bad entry can't be written.

    Written atomically (temp file + os.replace) so a crash mid-write can't
    leave a truncated/corrupt accounts.json — which load_accounts() would
    silently treat as 'absent' and fall back to the three defaults, dropping
    runtime-added accounts and resurrecting removed ones. The persisted list
    is what keeps accounts open across app restarts, so it must never be
    half-written."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    clean: list[str] = []
    for label in labels:
        if valid_account_label(label) and label not in seen:
            seen.add(label)
            clean.append(label)
    tmp = ACCOUNTS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, ACCOUNTS_FILE)


# Snapshot for short-lived scripts that import the name once at startup
# (run_pipeline, embed_messages, …). Long-lived processes must call
# load_accounts() instead so they pick up runtime changes.
ACCOUNTS = load_accounts()


# --- native Neo4j (no Docker / no WSL) --------------------------------------
# Neo4j 5.x runs as a native install — a Windows service when one is registered,
# otherwise an app-launched `neo4j console` child process. Either way it's
# reached over Bolt at NEO4J_URI (localhost:7687); there is no Docker and no WSL
# in the loop. The install layout defaults to C:\neo4j and is overridable via
# the CLAUDE_RVG_NEO4J_* env vars. Neo4j 5.x supports only Java 17/21, so a
# bundled JDK is pointed at explicitly (the system Java may be a newer,
# unsupported release).
NEO4J_HOME = Path(os.environ.get(
    "CLAUDE_RVG_NEO4J_HOME", r"C:\neo4j\neo4j-community-5.26.26"))
NEO4J_JAVA_HOME = Path(os.environ.get(
    "CLAUDE_RVG_NEO4J_JAVA_HOME", r"C:\neo4j\jdk-21.0.11+10"))
NEO4J_SERVICE = os.environ.get("CLAUDE_RVG_NEO4J_SERVICE", "neo4j")


def force_utf8() -> None:
    """UTF-8 everywhere — env (for child processes) + this process' streams."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from data/.env (preferred) or project root .env
    into os.environ. setdefault — won't clobber a shell-set value."""
    for cand in (DATA_DIR / ".env", ROOT / ".env"):
        if not cand.exists():
            continue
        for raw in cand.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(
                key.strip(),
                val.strip().strip('"').strip("'"),
            )
        return


def bootstrap_venv() -> None:
    """Re-exec under the SETUP.md venv if pipeline deps are missing. Call
    at the top of every entry script (before importing google-api libs) so a
    wrong-interpreter invocation fails fast instead of 30s in."""
    try:
        import google.auth  # noqa: F401
        return
    except ImportError:
        pass
    venv_python = Path.home() / ".venvs" / "claude_rvg" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not venv_python.exists():
        sys.exit(
            f"\n[bootstrap] Pipeline deps missing under {sys.executable}\n"
            f"            and no venv at {venv_python}.\n"
            f"            Create one per SETUP.md, then re-run."
        )
    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            sys.exit(
                f"\n[bootstrap] Running venv but `google.auth` still missing.\n"
                f"            Run: {venv_python} -m pip install -r "
                f"requirements.txt"
            )
    except OSError:
        pass
    print(f"[bootstrap] re-exec under venv python: {venv_python}")
    sys.exit(subprocess.run([str(venv_python), *sys.argv]).returncode)


def neo4j_driver():
    """Return a Neo4j driver configured from .env + environment. Caller
    is responsible for .close()."""
    from neo4j import GraphDatabase
    load_dotenv()
    return GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"),
              os.environ.get("NEO4J_PASSWORD", "change-me-please")),
    )


def run_step(label: str, cmd: list[str], cwd: Path | None = ROOT) -> None:
    """Run a subprocess and exit on nonzero. Prints label + elapsed."""
    print(f"\n[{label}] $ {' '.join(map(str, cmd))}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd)
    dt = time.time() - t0
    if r.returncode != 0:
        sys.exit(f"\n[{label}] FAILED (exit {r.returncode}) after {dt:.1f}s")
    print(f"[{label}] OK ({dt:.1f}s)")


# --- shared rendering helpers used by graph_app.py + graph_rag.py
# Colours for per-sender lane assignment in the in-browser railroad.
PALETTE = ["#4f9dde", "#56c596", "#e0a458", "#c98bdb", "#e07a7a",
           "#7ec8c3", "#b5b95a", "#d98cae", "#6f86d6", "#9ec46b",
           "#d77fb3", "#67b7dc"]


def esc(value: object) -> str:
    """HTML-escape, treating None as empty."""
    return html.escape("" if value is None else str(value))


def _account_from_url(stored_url: str | None) -> str:
    """The owning account email embedded in a stored gmail_url — either the
    older /u/<email>/ path form or the newer ?authuser=<email> form."""
    m = re.search(r"(?:/u/|authuser=)([^/#&?]+)", stored_url or "")
    return m.group(1) if m else ""


def gmail_search_url(stored_url: str | None,
                     rfc822_message_id: str | None) -> str:
    """Gmail 'rfc822msgid:' search link — lands on a 1-result search; click
    that result to open the message expanded in its thread.

    A 1-click 'open this message expanded' URL is not achievable: Gmail
    expands a specific message only when the URL carries that message's
    internal web id (the FMfcgz… form), which the Gmail API never exposes.
    '#all/<apiId>' and '#search/<q>/<apiId>' open the thread without
    expanding; '#search/<q>/<threadId>' expands the newest message, not the
    searched one. This link is for reaching native Gmail (attachments, reply)."""
    acct = _account_from_url(stored_url)
    if not acct or not rfc822_message_id:
        return ""
    return (f"https://mail.google.com/mail/?authuser={acct}"
            f"#search/rfc822msgid%3A{quote(rfc822_message_id, safe='')}")
