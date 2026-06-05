"""Always-live mail app — a tiny local server that keeps itself current.

Unlike the static data/mail_app.html (a frozen snapshot), this serves the
app from an in-memory cache that a background thread keeps fresh:

  - On start: load the graph from Neo4j, serve immediately.
  - Background: every --interval seconds, pull new mail (incremental Gmail
    sync, per account). If anything arrived, the new records are inline-
    cleaned, written to Neo4j with Layer 1 + REPLY_TO cypher only, the
    body store is updated, and the page cache is rebuilt. Embeddings run
    on a background thread so the reload doesn't wait for them.
  - The page polls /api/version; when the cache changes it shows a
    "click to reload" banner. The ↻ Sync button forces a sync now.

The sync is always incremental — the iterate-every-message layers
(NEXT_IN_THREAD backbone, domain orgs, calendar events) backfill on the
next run of scripts/run_pipeline.py, not on every sync. A new reply threads
correctly immediately because REPLY_TO's cypher is per-row.

Routes:  GET /              the app
         GET /api/version   {version, syncing, messages}
         GET /api/accounts  {accounts: [{label, email, authed}], known_emails}
         GET /api/auth/status ?account → {status: idle|running|ok|error:…}
         POST /api/auth     {account} → trigger the Google OAuth consent flow
                            server-side (opens the browser). {started, error?}
         POST /api/accounts/add    {label} → validate + OAuth a NEW mailbox;
                            persisted to data/accounts.json on success.
         POST /api/accounts/remove {label, purge?} → detach an account (delete
                            token/state files, drop from the list); purge=true
                            also deletes its mail from the graph + data files.
         GET /api/quote     ?mid&acct → original message for reply/forward
                            prefill: {from_email, from_name, sent_at, subject,
                            to, cc, body}
         GET /api/bodysearch ?q → {q, hits: [[mid, acct], …]} message keys
                            whose FULL clean body contains q (substring); backs
                            the body column box / body: terms searching the
                            whole message, not just the snippet in the payload.
         POST /api/sync     trigger an immediate sync
         POST /api/compose  {account, mode: send|draft, to, cc, bcc, subject,
                            body, in_reply_to?, references?, thread_id?,
                            attachments?} → {ok, id?, thread_id?, error?}
         POST /api/trash    {messages: [{mid, acct}, …]} → moves each
                            message to Gmail Trash, purges it from local
                            data files + Neo4j, rebuilds the cache.
                            Returns {ok, trashed, failed: [{mid,acct,error}]}.
         POST /api/seen     {messages: [{mid, acct}, …]} → clears the UNREAD
                            label on each message in Gmail + Neo4j (mark read).
                            Returns {ok, marked, failed: [{mid,acct,error}]}.
         POST /api/notspam  {messages: [{mid, acct}, …]} → removes the SPAM
                            label and adds INBOX on each message in Gmail +
                            Neo4j (move out of spam). Returns
                            {ok, unspammed, failed: [{mid,acct,error}]}.
         POST /api/markspam {messages: [{mid, acct}, …]} → adds the SPAM label
                            and removes INBOX on each message in Gmail + Neo4j
                            (move to spam). Returns
                            {ok, spammed, failed: [{mid,acct,error}]}.
         POST /api/ask      {q, session_id?} → text/event-stream — graph-RAG
                            chat. SSE events: phase, sources, thinking, tool,
                            done {answer, session_id}, error. The client
                            renders thinking/tool events live in the
                            "Thinking…" bubble and swaps in the answer on done.
         POST /api/ask/feedback {rating: "up"|"down", question, answer, note?,
                            session_id?} → logs the rating to
                            data/ask_feedback.jsonl; a down-vote also distils a
                            durable correction into Ask's long-term memory in
                            the background. Returns {ok, learning}.

Usage:
    python scripts/serve_app.py
    python scripts/serve_app.py --port 8765 --interval 600 --no-open
"""
from __future__ import annotations

import argparse
import gzip
import http.server
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from _common import (
    DATA_DIR,
    NEO4J_HOME,
    NEO4J_JAVA_HOME,
    NEO4J_SERVICE,
    ROOT,
    SCRIPTS,
    bootstrap_venv,
    force_utf8,
    load_accounts,
    save_accounts,
    valid_account_label,
)

force_utf8()
bootstrap_venv()

sys.path.insert(0, str(SCRIPTS))

# Windows: spawn child CLIs (claude, docker) WITHOUT flashing a console window.
# This matters when the server runs console-less via pythonw (the installed
# "Mail Graph" app): otherwise each `claude auth status` / `claude -p` / docker
# call pops a black console window. 0 on non-Windows (flag doesn't exist).
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# --- shared state -----------------------------------------------------------
# `html` is the rendered shell with the JSON payload inlined as a
# <script type="application/json"> tag; `html_gz` is its precomputed gzipped
# form so /          ships ~4–5× fewer bytes on the wire without re-zipping
# per request. `payload_json`/`payload_gz` mirror just the JSON payload so
# /api/payload can hand the client a fresh data set without forcing a full
# page reload after a sync. Raw bodies live in data/bodies.sqlite.
_state = {"html": "<h1>Loading…</h1>", "html_gz": b"",
          "payload_json": "", "payload_gz": b"",
          "version": 0, "syncing": False, "messages": 0}
_lock = threading.Lock()                 # guards _state
# Notified whenever `version` or `syncing` changes. Wraps _lock so the SSE
# /api/events handler can do `with _state_cv: cv.wait(...)` to block cheaply
# until the next state transition instead of polling.
_state_cv = threading.Condition(_lock)
_sync_lock = threading.Lock()            # ensures only one sync at a time
_ask_lock = threading.Lock()             # one /api/ask (claude -p) at a time
# Client-controlled "hold" timestamp: while now() < _hold_until, the
# background sync_loop skips its tick. The composer's selection-mode UI
# heartbeats /api/sync/hold so trashing a selection isn't racing against a
# fresh pull that would shift MSGS indices under the user.
_hold_lock = threading.Lock()
_hold_until: float = 0.0
# Serialize compose calls so two concurrent submits can't race on the same
# Gmail token refresh. They're rare (manual user clicks) so a global lock
# is fine.
_compose_lock = threading.Lock()
# Serialize OAuth sign-in flows triggered from the app: run_auth_flow binds a
# transient localhost redirect port and opens a browser, so only one can run at
# a time. _auth_status tracks each label's current/last flow result so the
# client can poll /api/auth/status for completion.
_auth_lock = threading.Lock()
_auth_status: dict[str, str] = {}
_auth_status_lock = threading.Lock()


def _safe_err(context: str, exc: BaseException) -> str:
    """Log the full exception server-side and return a sanitized, low-detail
    token for the client — the exception class name only, never str(exc),
    which can carry filesystem paths, tokens or internal state. Defense in
    depth: the Host/Origin guard already blocks cross-origin reads, but we
    don't hand raw exception text to the browser regardless. The local user
    still sees the full detail in the serve_app console."""
    print(f"[serve] {context}: {type(exc).__name__}: {exc}")
    return type(exc).__name__


# --- app settings (data/settings.json) --------------------------------------
# A tiny key/value store for user-tunable app settings exposed in the Settings
# panel. Currently just the LLM model used by /api/ask. Kept here (not in
# _common) because it's server-only and tied to the claude -p invocation below.
SETTINGS_FILE = DATA_DIR / "settings.json"
# UI model key → the value passed to `claude -p --model`. "default" (or any
# unknown key) means: pass no --model, i.e. use Claude Code's own model.
LLM_MODELS = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


def _load_settings() -> dict:
    try:
        d = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_settings(d: dict) -> None:
    import os
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, SETTINGS_FILE)


def _get_llm_model_key() -> str:
    """The chosen UI model key, or 'default' when unset/invalid."""
    key = _load_settings().get("llm_model") or "default"
    return key if key in LLM_MODELS else "default"


def _llm_model_id() -> str | None:
    """The `claude -p --model` value for the chosen key, or None for default."""
    return LLM_MODELS.get(_get_llm_model_key())


def _get_auto_learn(scope: str = "ask") -> bool:
    """Whether the given assistant scope auto-learns long-term memory after each
    turn (default on). Ask and Compose have independent toggles and stores."""
    key = "compose_auto_learn" if scope == "compose" else "ask_auto_learn"
    return bool(_load_settings().get(key, True))


# Cheap model used for the background memory-extraction pass.
_LEARN_MODEL = "claude-haiku-4-5-20251001"


def _learn_from_turn(question: str, answer: str) -> None:
    """Background pass (Haiku, no MCP): pull any NEW durable preference/fact
    from this Q&A and append it to long-term memory. Best-effort and silent."""
    claude = shutil.which("claude")
    if not claude:
        return
    try:
        import ask_memory
        existing = ask_memory.list_memories("ask")
    except Exception:
        return
    existing_txt = "\n".join(f"- ({m['kind']}) {m['text']}"
                             for m in existing) or "(none yet)"
    system = (
        "You maintain a long-term memory for a user of a personal email "
        "assistant. This memory captures HOW the user wants questions answered "
        "— the CRAFT of a good response — NOT facts about their mailbox.\n"
        "From the user's QUESTION (and the assistant's ANSWER for context), "
        "extract any NEW, durable preference about the SHAPE of a good answer:\n"
        "- structure / format (how to organize: ordering, sections, tables vs "
        "bullets vs prose, what to lead with);\n"
        "- logic / reasoning (how to reason to the answer: what to check, how to "
        "handle ambiguity, how to weigh or cite evidence, when to say 'unknown');\n"
        "- breadth / scope (how broad or narrow to go: exhaustive vs focused, "
        "whether to include context/caveats/alternatives, level of detail);\n"
        "- tone, language, length, and citing conventions.\n"
        "STRICT: do NOT learn any FACT about the emails, mailbox, people, "
        "organizations, projects, dates, or events — those are NEVER memory. "
        "Capture only GENERAL, durable response-craft lessons, never the one-off "
        "content of this specific answer. If the lesson is implicit in how a good "
        "answer to THIS question should have been structured/reasoned/scoped, "
        "phrase it as a general rule. Do NOT repeat anything already in EXISTING "
        "MEMORY. If nothing qualifies, output []. Output ONLY a JSON array of "
        "{\"text\":\"…\",\"kind\":\"style\"} (max 3 items, each one short "
        "imperative sentence). Every item MUST be kind 'style'.")
    user = (f"EXISTING MEMORY:\n{existing_txt}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:\n{(answer or '')[:4000]}\n\nReturn the JSON array.")
    cmd = [claude, "-p", "--model", _LEARN_MODEL,
           "--strict-mcp-config",                 # skip the neo4j MCP (not used)
           "--append-system-prompt", system]
    if claude.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c", *cmd]
    try:
        out = subprocess.run(cmd, cwd=ROOT, input=user, capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=90, creationflags=_NO_WINDOW)
    except Exception as e:
        print("[serve] memory-learn call failed:", type(e).__name__)
        return
    for item in _parse_json_array(out.stdout or ""):
        if isinstance(item, dict) and (item.get("text") or "").strip():
            try:
                # Ask memory is response-craft only — force 'style', never 'fact'.
                rec = ask_memory.add(str(item["text"]).strip(),
                                     "style", source="auto", scope="ask")
                if rec:
                    print(f"[serve] ask learned ({rec['kind']}): {rec['text']}")
            except Exception:
                pass


def _learn_from_compose(instruction: str, draft: str) -> None:
    """Background pass (Haiku, no MCP): pull any NEW durable email-writing
    preference/fact from this compose turn and append it to Compose's OWN
    long-term memory (scope 'compose', kept separate from Ask). Best-effort and
    silent — never blocks or affects the draft the user just received."""
    claude = shutil.which("claude")
    if not claude:
        return
    try:
        import ask_memory
        existing = ask_memory.list_memories("compose")
    except Exception:
        return
    existing_txt = "\n".join(f"- ({m['kind']}) {m['text']}"
                             for m in existing) or "(none yet)"
    system = (
        "You maintain a long-term memory for the EMAIL-COMPOSING assistant of a "
        "user (separate from their question-answering assistant). From the "
        "user's INSTRUCTION (and the DRAFT it produced, for context), extract "
        "any NEW, durable item worth remembering when WRITING future emails for "
        "this user:\n"
        "- kind 'style': a lasting writing preference (language, formality / "
        "tone, greeting or sign-off, length, signature, things to always or "
        "never do).\n"
        "- kind 'fact': a lasting fact about the user or a recurring "
        "correspondent (role, relationship, how they should be addressed) that "
        "would help future drafts.\n"
        "Rules: only GENERAL, durable items — never the one-off content of THIS "
        "email, and never this message's specific recipient/subject unless it "
        "is clearly a recurring correspondent. Do NOT repeat anything already "
        "in EXISTING MEMORY. If nothing qualifies, output []. Output ONLY a JSON "
        "array of {\"text\":\"…\",\"kind\":\"style\"|\"fact\"} (max 3 items, each "
        "one short sentence).")
    user = (f"EXISTING MEMORY:\n{existing_txt}\n\n"
            f"INSTRUCTION:\n{instruction}\n\n"
            f"DRAFT:\n{(draft or '')[:4000]}\n\nReturn the JSON array.")
    cmd = [claude, "-p", "--model", _LEARN_MODEL,
           "--strict-mcp-config",                 # skip the neo4j MCP (not used)
           "--append-system-prompt", system]
    if claude.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c", *cmd]
    try:
        out = subprocess.run(cmd, cwd=ROOT, input=user, capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=90, creationflags=_NO_WINDOW)
    except Exception as e:
        print("[serve] compose memory-learn call failed:", type(e).__name__)
        return
    for item in _parse_json_array(out.stdout or ""):
        if isinstance(item, dict) and (item.get("text") or "").strip():
            try:
                rec = ask_memory.add(str(item["text"]).strip(),
                                     item.get("kind", "fact"), source="auto",
                                     scope="compose")
                if rec:
                    print(f"[serve] compose learned ({rec['kind']}): "
                          f"{rec['text']}")
            except Exception:
                pass


def _parse_json_array(text: str) -> list:
    """Parse a JSON array from model output that may be wrapped in ``` fences
    or surrounded by prose. Returns [] on failure."""
    s = (text or "").strip()
    if "```" in s:                                # strip ```json … ``` fences
        import re
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
    i, j = s.find("["), s.rfind("]")
    if i == -1 or j == -1 or j < i:
        return []
    try:
        data = json.loads(s[i:j + 1])
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


# --- Answer feedback (👍 / 👎) ----------------------------------------------
# Each rating the user gives an Ask answer is appended to this append-only
# JSONL log (a labeled corpus of Q&A pairs, handy for the eval harness). A
# down-vote ALSO closes the improvement loop: a background pass distils a
# durable correction from the (question, answer, note) and stores it in Ask's
# long-term memory, which is injected into every future Ask prompt — so the
# next question already benefits.
ASK_FEEDBACK_FILE = DATA_DIR / "ask_feedback.jsonl"
_feedback_lock = threading.Lock()


def _record_ask_feedback(question: str, answer: str, rating: str,
                         note: str, session_id: str) -> dict:
    """Append one rating to data/ask_feedback.jsonl and return the record."""
    rec = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rating": "up" if rating == "up" else "down",
        "question": (question or "").strip(),
        "answer": (answer or "").strip()[:8000],
        "note": (note or "").strip()[:1000],
        "session_id": (session_id or "").strip(),
    }
    with _feedback_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with ASK_FEEDBACK_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _learn_from_feedback(question: str, answer: str, note: str,
                         rating: str = "down") -> None:
    """Background pass (Haiku, no MCP): turn a thumbs rating into a single
    durable, GENERAL response-craft lesson in Ask's long-term memory.

    * THUMBS-DOWN → a CORRECTION so future answers avoid the same problem.
    * THUMBS-UP   → a REINFORCEMENT: name the response-craft choice that made
      this answer good, phrased as a rule to KEEP doing.

    Either way the lesson is about the SHAPE of a good answer (structure, logic,
    breadth, tone) — never a fact about the mailbox. The user's NOTE is the
    primary signal; a blank note still runs (the model infers the likely
    lesson). Best-effort and silent."""
    claude = shutil.which("claude")
    if not claude:
        return
    try:
        import ask_memory
        existing = ask_memory.list_memories("ask")
    except Exception:
        return
    existing_txt = "\n".join(f"- ({m['kind']}) {m['text']}"
                             for m in existing) or "(none yet)"
    up = rating == "up"
    craft = (
        "- structure / format (how to organize the answer);\n"
        "- logic / reasoning (how to reason, handle ambiguity, weigh/cite "
        "evidence, when to say 'unknown');\n"
        "- breadth / scope (how broad or narrow, level of detail, what context "
        "or caveats to include);\n"
        "- tone, language, length, or citing conventions.\n")
    if up:
        verdict = "THUMBS-UP — the answer was good."
        task = (
            "Name the SINGLE response-craft choice that made this answer good, "
            "phrased as a durable, GENERAL rule to KEEP doing on FUTURE answers "
            "— never praise specific to this one answer's content. The lesson "
            "must be about one of:\n" + craft +
            "Use the user's NOTE as the primary signal of what they liked; if it "
            "is blank, infer the most likely craft strength from the question "
            "and answer. Be selective: capture only a DISTINCTIVE, non-obvious "
            "strength worth always repeating — if the answer was merely fine "
            "with nothing notable to reinforce, output [].")
        ans_tag = "ANSWER (thumbs-up)"
        note_tag = "USER NOTE ON WHAT THEY LIKED"
        emoji = "\U0001F44D"
    else:
        verdict = "THUMBS-DOWN."
        task = (
            "Distil a SINGLE durable, GENERAL correction about the SHAPE of a "
            "good answer that, if remembered, would make FUTURE answers better "
            "— never a fix specific to this one answer's content. The lesson "
            "must be about one of:\n" + craft +
            "Use the user's NOTE as the primary signal of what was wrong; if it "
            "is blank, infer the most likely general craft lesson from the "
            "question and answer.")
        ans_tag = "ANSWER (thumbs-down)"
        note_tag = "USER NOTE ON WHAT WAS WRONG"
        emoji = "\U0001F44E"
    system = (
        "You maintain a long-term memory for a user's personal email "
        "assistant. This memory captures HOW the user wants questions answered "
        "— the CRAFT of a good response — NOT facts about their mailbox. The "
        f"user just gave the assistant's ANSWER a {verdict} " + task +
        "\nSTRICT: do NOT learn any FACT about the emails, mailbox, people, "
        "organizations, projects, or events — those are NEVER memory; the "
        "lesson must be a general response-craft rule, not knowledge the "
        "assistant 'should have known'. Do NOT repeat or near-duplicate "
        "anything already in EXISTING MEMORY. If no general, durable craft "
        "lesson can be drawn, output []. Output ONLY a JSON array with AT MOST "
        "ONE item: {\"text\":\"…\",\"kind\":\"style\"} — one short imperative "
        "sentence. The item MUST be kind 'style'.")
    user = (f"EXISTING MEMORY:\n{existing_txt}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"{ans_tag}:\n{(answer or '')[:4000]}\n\n"
            f"{note_tag}:\n{note.strip() or '(none given)'}"
            f"\n\nReturn the JSON array.")
    cmd = [claude, "-p", "--model", _LEARN_MODEL,
           "--strict-mcp-config",                 # skip the neo4j MCP (not used)
           "--append-system-prompt", system]
    if claude.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c", *cmd]
    try:
        out = subprocess.run(cmd, cwd=ROOT, input=user, capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=90, creationflags=_NO_WINDOW)
    except Exception as e:
        print("[serve] feedback-learn call failed:", type(e).__name__)
        return
    for item in _parse_json_array(out.stdout or "")[:1]:
        if isinstance(item, dict) and (item.get("text") or "").strip():
            try:
                # Ask memory is response-craft only — force 'style', never 'fact'.
                rec = ask_memory.add(str(item["text"]).strip(),
                                     "style", source="auto", scope="ask")
                if rec:
                    print(f"[serve] ask learned from {emoji} ({rec['kind']}): "
                          f"{rec['text']}")
            except Exception:
                pass


def _do_ask_feedback(payload: dict) -> dict:
    """Handle POST /api/ask/feedback: log the rating, and on EITHER vote kick
    off the background learning pass (a down-vote distils a correction, an
    up-vote reinforces what made the answer good). Gated by the same auto-learn
    toggle as the post-answer pass, so turning learning off disables it too."""
    rating = (payload.get("rating") or "").strip().lower()
    if rating not in ("up", "down"):
        return {"ok": False, "error": "rating must be 'up' or 'down'"}
    question = payload.get("question") or ""
    answer = payload.get("answer") or ""
    note = payload.get("note") or ""
    session_id = payload.get("session_id") or ""
    try:
        _record_ask_feedback(question, answer, rating, note, session_id)
    except Exception as e:
        return {"ok": False, "error": _safe_err("ask/feedback", e)}
    learning = False
    if _get_auto_learn("ask"):
        threading.Thread(target=_learn_from_feedback,
                         args=(question, answer, note, rating),
                         daemon=True).start()
        learning = True
    return {"ok": True, "learning": learning}


# --- Claude Code subscription auth (claude auth …) --------------------------
# /api/ask shells out to `claude -p`, which authenticates with the local Claude
# Code subscription (no API key). The Settings panel surfaces that login so the
# user can sign in / out without touching a terminal — mirroring the Gmail
# OAuth flow. `claude auth login` opens a browser and runs its own localhost
# callback; we spawn it, scrape any printed URL (in case auto-open fails) and
# let the client poll `claude auth status` until login settles.
_claude_login_lock = threading.Lock()
_claude_login_proc = None                # the running `claude auth login` Popen
_claude_login_state = {"running": False, "url": "", "error": ""}


def _claude_cmd(*args: str) -> list[str]:
    claude = shutil.which("claude")
    if not claude:
        return []
    cmd = [claude, *args]
    if claude.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c", *cmd]
    return cmd


def _claude_status() -> dict:
    """Parse `claude auth status --json`. Always returns a dict with at least
    `installed` and `loggedIn`; never raises."""
    cmd = _claude_cmd("auth", "status", "--json")
    if not cmd:
        return {"installed": False, "loggedIn": False}
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=30,
                             creationflags=_NO_WINDOW)
        data = json.loads((out.stdout or "").strip() or "{}")
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    data["installed"] = True
    data.setdefault("loggedIn", False)
    return data


def _claude_login_start() -> dict:
    """Spawn `claude auth login` (browser OAuth) in the background. Returns
    immediately; the client polls /api/claude/status for completion."""
    global _claude_login_proc
    cmd = _claude_cmd("auth", "login", "--claudeai")
    if not cmd:
        return {"started": False, "error": "the `claude` CLI is not on PATH"}
    with _claude_login_lock:
        if _claude_login_proc and _claude_login_proc.poll() is None:
            return {"started": True}            # one already running
        _claude_login_state.update(running=True, url="", error="")
        try:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=_NO_WINDOW)
        except Exception as e:
            _claude_login_state.update(running=False)
            return {"started": False, "error": _safe_err("claude login", e)}
        _claude_login_proc = proc

    def _reader():
        import re
        url_re = re.compile(r"https://\S+")
        try:
            for line in proc.stdout:
                m = url_re.search(line)
                if m:
                    with _claude_login_lock:
                        _claude_login_state["url"] = m.group(0).rstrip(".,)")
        except Exception:
            pass
        finally:
            proc.wait()
            with _claude_login_lock:
                _claude_login_state["running"] = False
    threading.Thread(target=_reader, daemon=True).start()
    return {"started": True}


def _claude_logout() -> dict:
    cmd = _claude_cmd("auth", "logout")
    if not cmd:
        return {"ok": False, "error": "the `claude` CLI is not on PATH"}
    try:
        subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=30,
                       creationflags=_NO_WINDOW)
    except Exception as e:
        return {"ok": False, "error": _safe_err("claude logout", e)}
    return {"ok": True}


# /api/ask is graph-RAG. scripts/graph_rag.py first runs semantic retrieval
# (embed the question → vector-search the message_embedding index → expand
# each hit through the graph) and assembles a context bundle of the most
# relevant mail. That bundle is handed to headless Claude Code (`claude -p`),
# which writes the final answer. claude -p authenticates with the Claude Code
# subscription — no API key — and loads the project's .mcp.json (the neo4j
# server) since it runs with cwd=ROOT, so it can still run read-only Cypher
# for structural follow-ups. Only the read-only neo4j tools are allow-listed,
# so it can never modify the graph.
ASK_SYSTEM = (
    "You are Liam, the user's personal mail assistant. (The name is 'mail' "
    "reversed.) If the user asks who you are or what your name is, say you're "
    "Liam; otherwise just help — don't announce your name in every answer. "
    "You answer questions about a Gmail knowledge graph stored in Neo4j. "
    "The graph is a single MERGED index of three mailboxes — you@gmail.com, "
    "you@work.example.com and you@org.example.com (Message.account_owner is 'gmail', "
    "'work' or 'org'). All three are fully indexed and searchable from "
    "here; never tell the user a mailbox is unavailable — every one of them "
    "is in this graph."
    "\n\nTREATMENT TIERS (m.bucket). Every Message has a `bucket`: 'primary' is "
    "real correspondence; the lite tiers are bulk/auto-categorized mail — "
    "'promotions', 'social', 'updates', 'forums' and 'spam'. The RETRIEVED "
    "CONTEXT bundle is built ONLY from 'primary' mail (lite is never embedded), "
    "so lite never appears there. By default IGNORE lite mail and, in any "
    "aggregation over the graph, restrict to primary with "
    "`WHERE coalesce(m.bucket,'primary') = 'primary'`. ONLY when the user "
    "explicitly asks about promotions / newsletters / a brand's offers / spam "
    "etc. should you go after lite mail — it IS in the full-text index and "
    "queryable by bucket (e.g. CALL db.index.fulltext.queryNodes('message_text', "
    "'\\\"codigo de descuento\\\"') then filter the hits, or MATCH (m:Message) "
    "WHERE m.bucket = 'promotions' …)."
    "\n\nThe user's prompt contains a RETRIEVED CONTEXT section: message "
    "bodies found by semantic search, each tagged with a number [n] and "
    "grouped by conversation; messages marked ★ matched the question "
    "directly. In large conversations some bodies are abbreviated to fit a "
    "budget. Cite specific messages inline with bare bracket markers, e.g. "
    "[3] — do NOT write URLs; the app turns each [n] into a clickable link, "
    "so a bare [3] is the correct and complete citation."
    "\n\nYou have read-only Neo4j tools (read_neo4j_cypher, get_neo4j_schema) "
    "covering all three mailboxes. USE THEM ACTIVELY when more detail would "
    "help — don't settle for the bundle alone. The patterns that pay off "
    "most:"
    "\n  (a) Fetch a cited message in full when its body was abbreviated or "
    "you need an exact quote:"
    "\n      MATCH (m:Message {gmail_message_id:'…', account_owner:'…'})"
    "\n      RETURN m.subject, m.sent_at, m.body_clean"
    "\n  (b) Keyword-search the whole corpus for exact terms semantic search "
    "may have missed (a name, an identifier, a phrase). The full-text index "
    "supports Lucene syntax (OR, AND, \"quoted phrases\") and is "
    "ACCENT-FOLDING — searching the plain ASCII spelling matches the accented "
    "form and vice-versa (queryNodes(..., 'Munoz') finds 'Muñoz', "
    "'Munoz' finds 'Muñoz'), so prefer it for any name with a tilde/accent. "
    "(A raw CONTAINS on body text is NOT accent-folded — 'munoz' will miss "
    "'Muñoz' — so for CONTAINS search by an unaccented STEM, e.g. 'muno'.)"
    "\n      CALL db.index.fulltext.queryNodes('message_text', 'Rivera') "
    "YIELD node, score"
    "\n      RETURN node.gmail_message_id, node.account_owner,"
    "\n             node.subject, node.sent_at"
    "\n      ORDER BY score DESC LIMIT 10"
    "\n  (c) Then fetch the bodies of the hits you care about (pattern a) "
    "before drawing conclusions."
    "\nNode labels: Message, Person, Org, Thread, Matter, Event. "
    "Never modify the graph."
    "\n\nBREADTH, TIMELINE & ENUMERATION QUESTIONS. When the question asks for "
    "a span ('history', 'evolution', 'a lo largo de los años', 'desde el "
    "principio'), for the WHOLE ARC of a person or matter ('qué pasó con X', "
    "'what happened with X', 'desde que apareció hasta que se fue', 'cómo "
    "entró y cómo salió', 'la historia de X', 'mi relación con X'), for "
    "EVERYTHING ('all', 'every', 'todas las veces', 'lista todos', 'enumera'), "
    "or for a count, the RETRIEVED CONTEXT bundle is only a sample and is NOT "
    "sufficient — you MUST aggregate over the whole graph with Cypher before "
    "answering:"
    "\n  1. Resolve the anchor entity deterministically — never by guessing on "
    "body text. For an organisation, resolve it through its Org node and use "
    "the domain:"
    "\n      MATCH (o:Org) WHERE toLower(o.canonical_name) CONTAINS 'acme' "
    "RETURN o.canonical_name, o.domain, o.aliases"
    "\n     then match its people by address — both directions — e.g. "
    "(p:Person)-[:SENT]->(m) and (m)-[:RECEIVED_BY]->(p) WHERE p.email ENDS "
    "WITH '@'+domain. For a PERSON, first find their address(es) (one human "
    "may have several); a first name alone is ambiguous, so confirm identity "
    "by email / RUT before attributing anything to them, and keep look-alikes "
    "apart (e.g. a parent vs a child with similar names)."
    "\n  1b. WITH the person vs ABOUT the person — RUN BOTH. The arc of a "
    "person is NOT just the mail they sent or received. Two distinct searches, "
    "and you must run the second:"
    "\n      • WITH — mail they are on: (p:Person {email:'…'})-[:SENT|RECEIVED_BY]-(m). "
    "\n      • ABOUT — mail that DISCUSSES them but on which they are NOT a "
    "participant: an intro/referral before they appear, a decision or letter "
    "about ending the engagement, a hand-off after they leave. These have NO "
    "graph edge to the person, so the WITH query can never surface them — find "
    "them with the accent-folding full-text index on the name/surname "
    "(queryNodes('message_text', 'Munoz')), then EXCLUDE the ones already "
    "in the WITH set. The single most important message about someone — e.g. "
    "the family deciding to let them go — is often one they were never copied "
    "on. NEVER conclude a person 'only appeared in <year>', 'left without a "
    "trace', or 'there is no farewell/decision' from the WITH set alone: the "
    "first and last chapters of the arc usually live in the ABOUT set. Run the "
    "ABOUT search before you state any span, first-appearance, or absence."
    "\n  2. Aggregate first, then narrate. Run a per-year (or per-month) "
    "histogram across the FULL range and walk EVERY active period in your "
    "answer — do not skip or silently collapse years:"
    "\n      …WHERE <anchor> WITH substring(m.sent_at,0,4) AS yr, "
    "count(DISTINCT m) AS n RETURN yr, n ORDER BY yr"
    "\n  3. For 'all the times / todas las veces / lista todos' questions, "
    "enumerate each occurrence as its OWN dated, cited item. Do NOT merge "
    "distinct events into one summary; if two events really are the same "
    "ongoing matter, say so explicitly AND still list each as its own dated "
    "entry — never reduce them to 'a single process' and stop there. Then run "
    "a completeness check before you finish ('is there any instance in the "
    "corpus I have not listed?'). Search VERB and NOUN forms of the key term "
    "(e.g. citación / citaron / citado / 'te citaron'), because the decisive "
    "fact often lives in a body under an unrelated subject."
    "\n\nSCHEMA NOTE: this graph is deterministic. There are NO Concept or "
    "Topic nodes and NO (:Message)-[:MENTIONS]->… edges (LLM extraction and "
    "the Concept layer were removed). Org membership comes only from the "
    "sender's email domain via (:Person)-[:WORKS_AT]->(:Org). Do not query "
    "removed structure — call get_neo4j_schema if unsure."
    "\n\nIf after iterating you still cannot find what was asked, say so "
    "plainly — never invent a limitation."
    "\n\nThis is an ongoing conversation; follow-ups build on earlier "
    "answers. Each turn carries its own freshly RETRIEVED CONTEXT, so [n] "
    "markers are local to that turn. Be substantive — depth matters — and "
    "end every answer with a one-line offer to go deeper on a specific "
    "angle (e.g. \"¿Quieres que profundice en X, Y o Z?\" or the English "
    "equivalent, matching the language of the question)."
)


def _retrieve_context(question: str) -> tuple[str, list]:
    """Graph-RAG retrieval: semantic vector search + graph expansion, rendered
    as a context bundle. Returns (context_text, sources) — see
    graph_rag.build_context. Raises with a setup hint if the vector index is
    missing (load_neo4j.py --setup + embed_messages.py not run yet)."""
    import graph_app
    import graph_rag
    drv = graph_app.driver()
    try:
        with drv.session() as s:
            seeds = graph_rag.retrieve(s, question)
    except Exception as e:
        if "message_embedding" in str(e) or "queryNodes" in str(e):
            raise RuntimeError(
                "the message_embedding vector index is missing — run "
                "`python scripts/load_neo4j.py --setup` then "
                "`python scripts/embed_messages.py` to enable graph-RAG")
        raise
    finally:
        drv.close()
    return graph_rag.build_context(seeds)


def _stream_ask(write_event, question: str, session_id: str | None) -> None:
    """Graph-RAG chat turn, streamed as a sequence of events the client
    renders live in the "Thinking…" bubble. `write_event(payload)` is called
    per event; it returns False when the client disconnects, at which point
    we abort and terminate the subprocess.

    Event types emitted:
      phase    — {phase: "retrieving" | "thinking"}
      sources  — {sources: [...]} (the same list build_context produces)
      thinking — {text: "..."} (one assistant text message)
      tool     — {name, detail} (one tool_use the model issued)
      done     — {answer, session_id} (final answer + resumable id)
      error    — {message}

    Uses `claude -p --output-format stream-json --verbose`, which emits one
    JSON event per line (system init, assistant messages with text/tool_use
    blocks, user messages with tool_results, and a final result event)."""
    if not _ask_lock.acquire(blocking=False):
        write_event({"type": "error",
                     "message": "busy — a question is already being answered"})
        return
    proc = None
    try:
        write_event({"type": "phase", "phase": "retrieving"})
        try:
            context, sources = _retrieve_context(question)
        except Exception as e:
            # RuntimeError here carries a curated, safe setup hint (e.g. the
            # vector index is missing); anything else is sanitized.
            msg = (str(e) if isinstance(e, RuntimeError)
                   else _safe_err("ask: retrieve", e))
            write_event({"type": "error", "message": msg})
            return
        if not write_event({"type": "sources", "sources": sources}):
            return

        # Long-term memory: learned style preferences (all) + the facts most
        # relevant to this question, injected so Ask adapts across sessions.
        try:
            import ask_memory
            mem_block = ask_memory.format_block(question)
        except Exception as e:
            print("[serve] ask_memory recall failed:", e)
            mem_block = ""

        prompt = ((mem_block + "\n") if mem_block else "") + (
                  f"QUESTION:\n{question}\n\n"
                  f"RETRIEVED CONTEXT (semantic search over the mailbox for "
                  f"this question — cite with the [n] markers you use):"
                  f"\n\n{context}\n")

        claude = shutil.which("claude")
        if not claude:
            write_event({"type": "error",
                         "message": "the `claude` CLI is not on PATH"})
            return

        def build_cmd(resume: str | None) -> list[str]:
            cmd = [claude, "-p",
                   "--output-format", "stream-json", "--verbose",
                   "--allowedTools",
                   "mcp__neo4j__read_neo4j_cypher,mcp__neo4j__get_neo4j_schema"]
            if resume:
                cmd += ["--resume", resume]
            else:
                cmd += ["--append-system-prompt", ASK_SYSTEM]
                model_id = _llm_model_id()      # None → Claude Code default
                if model_id:
                    cmd += ["--model", model_id]
            if claude.lower().endswith((".cmd", ".bat")):
                cmd = ["cmd", "/c", *cmd]
            return cmd

        def run_once(resume: str | None) -> tuple[str, str, bool]:
            nonlocal proc
            proc = subprocess.Popen(build_cmd(resume), cwd=ROOT,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True, encoding="utf-8",
                                    errors="replace", bufsize=1,
                                    creationflags=_NO_WINDOW)
            # Feed the (large) prompt from a thread so a full pipe buffer
            # can't deadlock the streaming-stdout read loop.
            def _feed():
                try:
                    proc.stdin.write(prompt)
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            threading.Thread(target=_feed, daemon=True).start()

            final_answer = ""
            final_sid = resume or ""
            errored = False
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = ev.get("type")
                if etype == "assistant":
                    for block in (ev.get("message", {}).get("content")
                                  or []):
                        btype = block.get("type")
                        if btype == "text":
                            text = (block.get("text") or "").strip()
                            if text:
                                if not write_event({"type": "thinking",
                                                    "text": text}):
                                    proc.terminate()
                                    return "", "", True
                        elif btype == "tool_use":
                            name = block.get("name") or "tool"
                            tin = block.get("input") or {}
                            if "cypher" in name.lower():
                                detail = tin.get("query") or ""
                                label = "Querying the graph"
                            elif "schema" in name.lower():
                                detail = ""
                                label = "Reading the graph schema"
                            else:
                                detail = ""
                                label = name
                            if not write_event({"type": "tool",
                                                "name": label,
                                                "detail": detail}):
                                proc.terminate()
                                return "", "", True
                elif etype == "result":
                    final_answer = (ev.get("result") or "").strip()
                    final_sid = ev.get("session_id") or final_sid
                    if ev.get("is_error"):
                        errored = True
            # Drain any stderr for diagnostics on a non-zero exit.
            proc.wait()
            if proc.returncode != 0 and not final_answer:
                errored = True
            return final_answer, final_sid, errored

        write_event({"type": "phase", "phase": "thinking"})
        answer, sid, errored = run_once(session_id)
        if errored and session_id:
            # Stored session may have expired/been pruned — retry fresh.
            write_event({"type": "phase", "phase": "retrying"})
            answer, sid, errored = run_once(None)

        if errored or not answer:
            write_event({"type": "error",
                         "message": answer
                         or "claude returned an error"})
            return
        write_event({"type": "done", "answer": answer, "session_id": sid})
        # Auto-learn: in the background, extract any durable preference/fact
        # from this turn and save it to long-term memory (never blocks/affects
        # the answer the user just got).
        if _get_auto_learn("ask"):
            threading.Thread(target=_learn_from_turn,
                             args=(question, answer), daemon=True).start()
    finally:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        _ask_lock.release()


# Reference to the live HTTP server, set in main(). The /api/shutdown endpoint
# (the app's power button) uses it to stop the server gracefully so the page can
# show a visible "stopping → stopped" state before the window closes.
_srv = None


def rebuild() -> None:
    """Re-query Neo4j, refresh the cached HTML + its gzipped form, and bring
    the body store up to date with emails.jsonl."""
    import body_store
    added = body_store.refresh()
    if added:
        print(f"[serve] body store +{added:,} rows "
              f"(total {body_store.count():,})")
    import graph_app
    drv = graph_app.driver()
    try:
        # lean=True drops body_clean from every message — the panel falls
        # back to the snippet on open and /api/body upgrades to the real
        # HTML on click. Cuts the rendered page from ~100 MB to ~5 MB.
        # build_payload takes the driver and parallelises its 4 Cypher
        # queries internally.
        payload = graph_app.build_payload(drv, lean=True)
    finally:
        drv.close()
    payload_json = json.dumps(payload, ensure_ascii=False)
    # Fingerprint the render template so a code/template change forces a re-render
    # even when the data is unchanged (otherwise the no-op fast path below would
    # keep serving a page rendered by old code after an upgrade + restart).
    import hashlib
    page_hash = hashlib.md5(graph_app.PAGE.encode("utf-8")).hexdigest()
    # No-op fast path: if the rebuilt data AND the template are byte-identical to
    # what's already cached, don't bump the version. This happens when a periodic
    # sync pulls no new mail. Skipping the bump avoids a spurious "new data,
    # reload" banner.
    with _lock:
        cur_v = _state["version"]
        unchanged = (cur_v > 0 and payload_json == _state["payload_json"]
                     and _state.get("page_hash") == page_hash)
    if unchanged:
        print(f"[serve] rebuild: no change (version {cur_v})")
        return
    # Bake the next version into the page up front (peek current + 1) so the
    # client's baseline matches the data it loads. The background rebuild may bump
    # the version before the client's SSE stream connects; baking keeps the reload
    # banner correct in that race.
    v = cur_v + 1
    html = graph_app.render_page(payload, version=v)
    html_bytes = html.encode("utf-8")
    payload_bytes = payload_json.encode("utf-8")
    # gzip level 1 — over loopback the bandwidth difference between level 1
    # and 6 doesn't matter, but level 1 is ~5× faster CPU. Rebuild used to
    # spend ~1s on gzip alone; this brings it down to ~200ms.
    html_gz = gzip.compress(html_bytes, compresslevel=1)
    payload_gz = gzip.compress(payload_bytes, compresslevel=1)
    with _state_cv:
        _state["html"] = html
        _state["html_gz"] = html_gz
        _state["payload_json"] = payload_json
        _state["payload_gz"] = payload_gz
        _state["messages"] = len(payload["msgs"])
        _state["version"] = v
        _state["page_hash"] = page_hash
        _state_cv.notify_all()
    print(f"[serve] cache rebuilt — {len(payload['msgs']):,} messages "
          f"(version {v}, {len(html_bytes)/1_048_576:.1f} MB → "
          f"{len(html_gz)/1_048_576:.1f} MB gzipped)")


def _run_module_main(mod_name: str, argv_args: tuple[str, ...] = ()) -> None:
    """Invoke another script's main() in-process — saves ~3-5s of subprocess
    startup per call (sentence-transformers, talon, neo4j driver are slow to
    import). sys.argv is patched so the target's argparse sees only its own
    args, not serve_app's."""
    import importlib
    mod = importlib.import_module(mod_name)
    old_argv = sys.argv
    sys.argv = [mod_name + ".py", *argv_args]
    try:
        mod.main()
    finally:
        sys.argv = old_argv


def _pull_one(account: str) -> tuple[list[dict], int]:
    """Pull a single account. Returns (new_records, label_changes) straight
    from fetch_new_messages: the records it appended this call, plus the count
    of existing messages whose read/unread/spam label was revised against
    Gmail."""
    import sync_incremental
    try:
        records, changed = sync_incremental.fetch_new_messages(
            SimpleNamespace(account=account, force_date=False))
    except SystemExit as e:
        print(f"[serve] {account}: skipped — {e}")
        return [], 0
    except Exception as e:
        print(f"[serve] {account}: pull failed — {type(e).__name__}: {e}")
        return [], 0
    return records, changed


def _clean_records_inline(records: list[dict]) -> None:
    """Clean just the new records (skips clean_bodies.main()'s 40k-line
    rescan). Delegates to the shared clean_bodies.clean_records."""
    import clean_bodies
    clean_bodies.clean_records(records)


def _fast_load_records(records: list[dict]) -> None:
    """Targeted Neo4j load — Layer 1 + REPLY_TO for just the new records,
    reusing serve_app's graph_app driver. Delegates to the shared
    load_neo4j.fast_load_records (see it for what's skipped)."""
    if not records:
        return
    import load_neo4j
    import graph_app
    drv = graph_app.driver()
    try:
        load_neo4j.fast_load_records(records, driver=drv)
    finally:
        drv.close()


def reconcile_graph() -> int:
    """Self-heal the graph against emails.jsonl — the safety net that makes mail
    loss impossible. A sync writes each message to emails.jsonl (a durable,
    append-only log) and advances its Gmail cursor BEFORE the Neo4j load commits;
    so a crash/kill/load-error in that window leaves the message in the log but
    absent from the graph, and a normal re-sync skips it (cursor already past it,
    id already in the pulled-tracker). This reconciliation closes that gap: it
    compares emails.jsonl directly to the graph (NOT the trackers) and loads
    anything missing, idempotently. Run on every boot, so any message stranded by
    a previous session is recovered before the user looks. Returns the count
    loaded.

    The graph should already contain every non-draft line of emails.jsonl, so in
    the healthy case this finds nothing and is a cheap no-op (one query + one
    sequential read)."""
    import graph_app
    import clean_bodies
    import load_neo4j
    import body_store
    ej = DATA_DIR / "emails.jsonl"
    if not ej.exists():
        return 0
    drv = graph_app.driver()
    try:
        # Compound key (account_owner, gmail_message_id) — same as the node key.
        present: set[tuple] = set()
        with drv.session() as session:
            for r in session.run("MATCH (m:Message) RETURN m.account_owner AS a, "
                                  "m.gmail_message_id AS g"):
                present.add((r["a"], r["g"]))
        missing: list[dict] = []
        seen: set[tuple] = set()
        with ej.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if "DRAFT" in (rec.get("label_ids") or []):
                    continue
                # Raw log uses "message_id"; the node key is "gmail_message_id"
                # (same value). Dedup within the log too (autosave copies etc.).
                key = (rec.get("account_owner"), rec.get("message_id"))
                if not key[1] or key in present or key in seen:
                    continue
                seen.add(key)
                missing.append(rec)
        if not missing:
            return 0
        print(f"[serve] reconcile: {len(missing)} message(s) in emails.jsonl "
              f"missing from the graph — loading (self-heal)", flush=True)
        clean_bodies.clean_records(missing)
        load_neo4j.fast_load_records(missing, driver=drv)
        body_store.refresh()
        return len(missing)
    finally:
        drv.close()


def do_sync(*, only_account: str | None = None,
            defer_embed: bool = True) -> None:
    """Incremental Gmail pull → inline clean → targeted Neo4j load → rebuild
    page cache. Always uses the fast incremental path: only the records new
    in THIS sync are processed, and only Layer 1 + REPLY_TO are written for
    them. The iterate-every-message layers (NEXT_IN_THREAD backbone, domain
    orgs, calendar events) backfill on the next run of
    scripts/run_pipeline.py — not needed for routine sync.

    Knobs:
      only_account=<label> — pull just that mailbox (composer after send).
                             None pulls all configured accounts.
      defer_embed=True     — embeddings run on a background daemon thread so
                             the page reload doesn't wait for them. graph-RAG
                             catches up shortly after. Pass False only if a
                             caller specifically needs /api/ask current
                             before returning.

    Skipped silently if another sync is in progress."""
    if not _sync_lock.acquire(blocking=False):
        print("[serve] sync already in progress — skipped")
        return
    try:
        with _state_cv:
            _state["syncing"] = True
            _state_cv.notify_all()

        valid_accts = load_accounts()
        if only_account and only_account not in valid_accts:
            print(f"[serve] unknown account '{only_account}' — pulling all")
            only_account = None
        accounts = [only_account] if only_account else list(valid_accts)

        all_new: list[dict] = []
        label_changes = 0
        for acct in accounts:
            recs, changed = _pull_one(acct)
            all_new.extend(recs)
            label_changes += changed
        if not all_new:
            # No new mail, but read/unread flips still need the cache rebuilt
            # so the bold/unread rendering catches up with Gmail.
            if label_changes:
                print(f"[serve] sync — {label_changes} read/unread change(s), "
                      f"no new mail; rebuilding")
                rebuild()
            else:
                print("[serve] sync — no new mail")
            return

        print(f"[serve] {len(all_new)} new message(s) — cleaning + loading…")
        _clean_records_inline(all_new)
        _fast_load_records(all_new)
        # Keep body_store current so /api/body works for the new messages.
        import body_store
        body_store.refresh()
        rebuild()

        if defer_embed:
            threading.Thread(
                target=lambda: _run_module_main("embed_messages"),
                daemon=True).start()
        else:
            _run_module_main("embed_messages")
    finally:
        with _state_cv:
            _state["syncing"] = False
            _state_cv.notify_all()
        _sync_lock.release()


def sync_loop(interval: int) -> None:
    time.sleep(6)                        # let the first page load settle
    while True:
        try:
            with _hold_lock:
                held = time.time() < _hold_until
            if held:
                print("[serve] auto-sync paused — client holds the lock "
                      "(selection in progress)")
            else:
                do_sync()
        except Exception as e:
            print(f"[serve] sync loop error: {type(e).__name__}: {e}")
        time.sleep(interval)


def warm_embedder() -> None:
    """Pre-load the graph-RAG embedding model so the first /api/ask doesn't
    pay the load cost. Runs in a background thread; on failure (e.g.
    sentence-transformers not installed) the first ask just loads it then,
    or surfaces the missing dependency."""
    try:
        import graph_rag
        graph_rag.get_model()
        print("[serve] embedding model ready — /api/ask graph-RAG is live")
    except Exception as e:
        print(f"[serve] embedding model not pre-loaded "
              f"({type(e).__name__}: {e}) — /api/ask will load it on "
              f"first use")


def build_style_profiles() -> None:
    """Autonomous background pass: learn / refresh the per-recipient writing-
    style cards Liam Compose injects so drafts sound like the user. Incremental
    (see style_profiles.build) — a restart with no new mail makes no claude
    calls. Best-effort and silent; never blocks boot. Sleeps first so it
    doesn't compete with the first page load and initial sync."""
    time.sleep(25)
    try:
        import style_profiles
        my = [e for e in _account_emails().values() if e]
        if not my:
            print("[serve] style profiles: no account emails yet — skipping")
            return
        style_profiles.build(my)
    except Exception as e:
        print(f"[serve] style profiles build failed: "
              f"{type(e).__name__}: {e}")


# --- compose: account list, quote fetch, send/draft -----------------------

def _account_emails() -> dict[str, str]:
    """Return {label: email} for every configured account, reading the cached
    address from data/sync_state_<label>.json (populated by pull_gmail on
    first auth). Labels without a cached email are still returned with ''."""
    import pull_gmail
    out: dict[str, str] = {}
    for label in load_accounts():
        p = pull_gmail.sync_state_path(label)
        email = ""
        if p.exists():
            try:
                email = (json.loads(p.read_text(encoding="utf-8"))
                         .get("account_email") or "")
            except (json.JSONDecodeError, OSError):
                email = ""
        out[label] = email
    return out


def _account_status() -> list[dict]:
    """Per-account sign-in status for the accounts panel: label, the cached
    email, and whether a valid (or refreshable) token exists right now.
    load_credentials returns creds only when the token is present and either
    valid or refreshable, so it doubles as the authed check."""
    import pull_gmail
    emails = _account_emails()
    out: list[dict] = []
    for label in load_accounts():
        try:
            authed = pull_gmail.load_credentials(label) is not None
        except Exception:
            authed = False
        out.append({"label": label, "email": emails.get(label, ""),
                    "authed": authed})
    return out


def _start_auth(account: str, *, allow_new: bool) -> dict:
    """Kick off the Google OAuth consent flow for one account label from the
    app instead of the terminal. run_auth_flow opens the system browser and
    runs a transient localhost redirect server, blocking until consent
    completes — so we run it on a background thread and let the client poll
    /api/auth/status. Only one flow at a time (it binds a port + the browser).

    allow_new=False  → re-auth/sign-in of an EXISTING label (must already be
                       in the account list).
    allow_new=True   → ADD a brand-new label; it is appended to accounts.json
                       only after a successful consent, so the list never
                       contains a label with no token.

    The label is validated against valid_account_label() before it can reach
    a filename (token_<label>.json) — the security gate for user-supplied
    labels. Returns immediately with {started: bool, error?: str}."""
    import pull_gmail
    if not valid_account_label(account):
        return {"started": False,
                "error": "invalid label — use a-z, 0-9, '-' or '_' (max 32)"}
    existing = load_accounts()
    if allow_new and account in existing:
        return {"started": False,
                "error": f"account '{account}' already exists"}
    if not allow_new and account not in existing:
        return {"started": False, "error": f"unknown account '{account}'"}
    if not _auth_lock.acquire(blocking=False):
        return {"started": False,
                "error": "another sign-in is already in progress"}
    with _auth_status_lock:
        _auth_status[account] = "running"

    def _run():
        try:
            # Bound the wait so an abandoned consent (tab closed) eventually
            # frees _auth_lock instead of blocking all future sign-ins.
            pull_gmail.run_auth_flow(account, timeout_seconds=300)
            # run_auth_flow deliberately drops the cached account_email; refetch
            # it via getProfile so the panel and composer dropdown show the
            # address immediately instead of after the next sync.
            try:
                svc = pull_gmail.gmail_service(account)
                pull_gmail.get_account_email(svc, account)
            except Exception:
                pass
            # Persist a brand-new account ONLY now that consent succeeded.
            if account not in load_accounts():
                save_accounts(load_accounts() + [account])
            with _auth_status_lock:
                _auth_status[account] = "ok"
        except (Exception, SystemExit) as e:
            # SystemExit fires when credentials.json is missing (run_auth_flow
            # raises it) — surface it as an error rather than killing the thread.
            with _auth_status_lock:
                _auth_status[account] = "error:" + _safe_err(
                    f"auth flow ({account})", e)
        finally:
            _auth_lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return {"started": True}


def _remove_account(label: str, *, purge: bool) -> dict:
    """Detach an account: drop it from accounts.json and delete its per-account
    token/state/resume files so it stops syncing and can't send. With
    purge=True, also delete that mailbox's data from the graph (Message /
    Thread / Attachment with account_owner=label) and rewrite the jsonl data
    files without it — destructive and irreversible. The graph purge + the
    big file rewrite run on a background thread; the detach itself is
    synchronous so the client gets an immediate, consistent answer."""
    if not valid_account_label(label):
        return {"ok": False, "error": "invalid label"}
    if label not in load_accounts():
        return {"ok": False, "error": f"unknown account '{label}'"}

    # 1. Remove from the list first so no sync picks it up again.
    save_accounts([a for a in load_accounts() if a != label])

    # 2. Delete the small per-account files (token, sync state, resume sets).
    import pull_gmail
    removed_files = []
    for p in (pull_gmail.token_path(label),
              pull_gmail.sync_state_path(label),
              pull_gmail.pulled_ids_path(label),
              DATA_DIR / f"pulled_event_ids_{label}.txt"):
        try:
            if p.exists():
                p.unlink()
                removed_files.append(p.name)
        except OSError as e:
            print(f"[serve] remove {label}: could not delete {p.name}: {e}")

    # Stop reporting a stale auth status for a label that's gone.
    with _auth_status_lock:
        _auth_status.pop(label, None)

    if purge:
        threading.Thread(target=_purge_account, args=(label,),
                         daemon=True).start()

    return {"ok": True, "removed": label, "purge": purge,
            "files_deleted": removed_files}


def _purge_account(label: str) -> None:
    """Background: delete a removed account's mail from the graph and rewrite
    the jsonl data files without it, then rebuild the page cache. Best-effort;
    errors are logged, not surfaced (the client already got its OK)."""
    import graph_app
    try:
        drv = graph_app.driver()
        try:
            with drv.session() as s:
                # Batched detach-delete so a large mailbox doesn't blow the
                # transaction. account_owner is on Message/Thread/Attachment.
                for lbl in ("Message", "Attachment", "Thread"):
                    s.run(
                        f"MATCH (n:{lbl} {{account_owner: $label}}) "
                        f"CALL (n) {{ DETACH DELETE n }} "
                        f"IN TRANSACTIONS OF 5000 ROWS",
                        label=label).consume()
                # Drop Threads/Matters left with no messages.
                s.run("MATCH (t:Thread) WHERE NOT (t)<-[:IN_THREAD]-(:Message) "
                      "DETACH DELETE t").consume()
                s.run("MATCH (mt:Matter) WHERE NOT (mt)<-[:PART_OF]-(:Thread) "
                      "DETACH DELETE mt").consume()
        finally:
            drv.close()
    except Exception as e:
        print(f"[serve] purge {label}: graph delete failed: "
              f"{type(e).__name__}: {e}")

    # Rewrite the jsonl files dropping this account's records.
    try:
        import os
        # Serialise the file rewrites against sync's append path: a concurrent
        # sync appending to emails.jsonl during our read→os.replace would have
        # its just-added lines silently clobbered. _sync_lock is sync's own
        # gate (do_sync holds it for the whole pull), so holding it here makes
        # the two mutually exclusive. refresh()/rebuild() below only read and
        # tolerate a sync appending afterwards, so they stay outside the lock.
        with _sync_lock:
            for path in (DATA_DIR / "emails.jsonl",
                         DATA_DIR / "emails_clean.jsonl"):
                if not path.exists():
                    continue
                tmp = path.with_suffix(path.suffix + ".tmp")
                with path.open("r", encoding="utf-8") as fin, \
                        tmp.open("w", encoding="utf-8") as fout:
                    for line in fin:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            fout.write(line)
                            continue
                        if (rec.get("account_owner") or "") == label:
                            continue
                        fout.write(line)
                os.replace(tmp, path)
            # cleaned_msg_ids.txt entries are "<acct>\t<mid>".
            p = DATA_DIR / "cleaned_msg_ids.txt"
            if p.exists():
                kept = [l for l in p.read_text(encoding="utf-8").splitlines()
                        if l.strip() and not l.startswith(f"{label}\t")]
                p.write_text("\n".join(kept) + ("\n" if kept else ""),
                             encoding="utf-8")
        import body_store
        body_store.refresh()
        rebuild()
        print(f"[serve] purge {label}: graph + files purged, cache rebuilt")
    except Exception as e:
        print(f"[serve] purge {label}: file cleanup failed: "
              f"{type(e).__name__}: {e}")


def _known_emails() -> list[str]:
    """All Person.email values in the graph — used for composer autocomplete."""
    import graph_app
    drv = graph_app.driver()
    try:
        with drv.session() as s:
            rows = s.run(
                "MATCH (p:Person) WHERE p.email IS NOT NULL "
                "RETURN p.email AS e, p.name AS n").data()
    finally:
        drv.close()
    # Stable, name-first ordering when both exist; emails alone fall back to
    # alphabetical so the frontend autocomplete is predictable.
    rows.sort(key=lambda r: ((r.get("n") or "").lower(),
                             (r.get("e") or "").lower()))
    return [r["e"] for r in rows if r.get("e")]


_UNSAFE_TAGS = ("script", "style", "iframe", "object", "embed", "link",
                "meta", "form")


def _sanitize_quote_html(html: str) -> str:
    """Strip executable / network-loading constructs from email HTML before
    feeding it into the composer's contenteditable. The panel renders body
    HTML inside a sandboxed iframe, but the composer edits the markup in-
    place, so we have to scrub here. Lightweight on purpose — Gmail already
    delivered the message, this is just defense-in-depth for the user's own
    contenteditable surface."""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return html
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(list(_UNSAFE_TAGS)):
        tag.decompose()
    for tag in soup.find_all(True):
        # Drop event-handler attrs and javascript: URLs.
        for attr in list(tag.attrs):
            if attr.lower().startswith("on"):
                del tag.attrs[attr]
                continue
            val = tag.attrs[attr]
            if isinstance(val, str) and val.strip().lower().startswith(
                    "javascript:"):
                del tag.attrs[attr]
    # Unwrap <html>/<body> so the composer doesn't end up with nested
    # document scaffolding inside its editor.
    body = soup.body
    if body is not None:
        return body.decode_contents()
    return str(soup)


def _body_search(term: str) -> list[list[str]]:
    """Message keys whose FULL clean body contains `term` — case-insensitive
    substring, the same semantics as the client's other filter boxes (so it's
    accent-sensitive too: it matches what you'd see typing in the body column).

    Returns a list of [mid, acct] pairs. The body column box and body:/text:
    terms use this to search the whole message instead of the 240-char snippet
    that ships in the lean page payload. Blank / single-char terms return [] —
    the client only calls this for terms of length >= 2, and a 1-char scan
    would match almost everything for no value."""
    term = (term or "").strip().lower()
    if len(term) < 2:
        return []
    import graph_app
    drv = graph_app.driver()
    try:
        with drv.session() as s:
            rows = s.run(
                "MATCH (m:Message) "
                "WHERE m.body_clean IS NOT NULL "
                "  AND toLower(m.body_clean) CONTAINS $q "
                "RETURN m.gmail_message_id AS mid, m.account_owner AS acct",
                q=term).data()
    finally:
        drv.close()
    return [[r["mid"], r["acct"]] for r in rows]


def _fetch_quote(mid: str, acct: str) -> dict | None:
    """Return the original message fields needed to prefill a reply/forward
    composer (sender, recipients, subject, cleaned body, RFC822 ids)."""
    if not mid or not acct:
        return None
    import graph_app
    drv = graph_app.driver()
    try:
        with drv.session() as s:
            row = s.run(
                "MATCH (m:Message {gmail_message_id:$mid, account_owner:$acct}) "
                "OPTIONAL MATCH (sender:Person)-[:SENT]->(m) "
                "OPTIONAL MATCH (m)-[r:RECEIVED_BY]->(rcpt:Person) "
                "WITH m, sender, "
                "     collect(DISTINCT CASE WHEN r.kind='to' THEN rcpt.email END) AS too, "
                "     collect(DISTINCT CASE WHEN r.kind='cc' THEN rcpt.email END) AS ccc "
                "RETURN m.subject AS subject, m.sent_at AS sent_at, "
                "       m.body_clean AS body, m.rfc822_message_id AS rfc822, "
                "       m.references AS refs, m.gmail_message_id AS gmid, "
                "       m.account_owner AS acct, "
                "       sender.email AS from_email, sender.name AS from_name, "
                "       [x IN too WHERE x IS NOT NULL] AS too, "
                "       [x IN ccc WHERE x IS NOT NULL] AS ccc",
                mid=mid, acct=acct).single()
    finally:
        drv.close()
    if not row:
        return None
    # Thread id is needed by the client to keep the reply in the same Gmail
    # conversation — pull it via the IN_THREAD relationship.
    drv = graph_app.driver()
    try:
        with drv.session() as s:
            tid_row = s.run(
                "MATCH (m:Message {gmail_message_id:$mid, account_owner:$acct})"
                "-[:IN_THREAD]->(t:Thread) RETURN t.gmail_thread_id AS tid",
                mid=mid, acct=acct).single()
    finally:
        drv.close()
    # The graph stores body_clean only; the raw HTML lives in emails.jsonl
    # (indexed by body_store). Pull it so the composer can render a real
    # quoted message on reply/forward instead of stripped plain text.
    # Sanitize before returning — the composer puts this into innerHTML.
    try:
        import body_store
        body_html = _sanitize_quote_html(body_store.get(mid, acct) or "")
    except Exception:
        body_html = ""
    return {
        "from_email": row["from_email"] or "",
        "from_name":  row["from_name"] or "",
        "sent_at":    row["sent_at"] or "",
        "subject":    row["subject"] or "",
        "to":         row["too"] or [],
        "cc":         row["ccc"] or [],
        "body":       row["body"] or "",
        "body_html":  body_html,
        "rfc822":     row["rfc822"] or "",
        "references": row["refs"] or [],
        "thread_id":  (tid_row["tid"] if tid_row else "") or "",
    }


def _do_compose(payload: dict) -> dict:
    """Sign + dispatch one compose request. Returns the JSON the client
    receives — {ok, id?, thread_id?, error?}."""
    import _send_mail
    acct = (payload.get("account") or "").strip()
    mode = (payload.get("mode") or "send").strip().lower()
    if acct not in load_accounts():
        return {"ok": False,
                "error": f"unknown account '{acct}'"}
    if mode not in ("send", "draft"):
        return {"ok": False, "error": f"mode must be send or draft, got '{mode}'"}

    accts = _account_emails()
    from_email = accts.get(acct, "")
    if not from_email:
        return {"ok": False,
                "error": f"no cached email for {acct} — run "
                         f"pull_gmail.py --account {acct} --auth first"}

    to_list  = payload.get("to") or []
    cc_list  = payload.get("cc") or []
    bcc_list = payload.get("bcc") or []
    if not (to_list or cc_list or bcc_list) and mode == "send":
        return {"ok": False, "error": "at least one recipient required to send"}

    thread_id = (payload.get("thread_id") or "").strip() or None

    with _compose_lock:
        try:
            service = _send_mail.gmail_service(acct)
        except SystemExit as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False,
                    "error": f"{_safe_err('compose: gmail_service', e)} — "
                             f"reconnect '{acct}' via the ⚙ Accounts panel "
                             f"(or `pull_gmail.py --account {acct} --auth`)."}

        # Verify the token actually belongs to the account the user chose.
        # Gmail sends as the authenticated user regardless of the MIME From:
        # header, so a swapped token (this has happened — see history) would
        # silently misroute. The dropdown's cached email is what the user
        # trusted; if the token disagrees, refuse to send and refresh the
        # cache so the dropdown updates on next open.
        try:
            prof = service.users().getProfile(userId="me").execute()
            actual = (prof.get("emailAddress") or "").lower()
        except Exception as e:
            return {"ok": False,
                    "error": f"could not verify account identity: "
                             f"{_safe_err('compose: getProfile', e)}"}
        expected = from_email.lower()
        if actual != expected:
            # Update sync_state so the next /api/accounts call reflects truth.
            import pull_gmail
            sp = pull_gmail.sync_state_path(acct)
            try:
                state = (json.loads(sp.read_text(encoding="utf-8"))
                         if sp.exists() else {})
                state["account_email"] = actual
                sp.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                              encoding="utf-8")
            except (json.JSONDecodeError, OSError):
                pass
            return {"ok": False,
                    "error": f"token for '{acct}' authenticates as {actual}, "
                             f"not {expected}. Re-run "
                             f"`pull_gmail.py --account {acct} --auth` and "
                             f"pick the correct Google account."}

        raw = _send_mail.build_message(
            from_email=actual,    # use the verified address, not the cache
            to=to_list, cc=cc_list, bcc=bcc_list,
            subject=payload.get("subject") or "",
            body=payload.get("body") or "",
            is_html=bool(payload.get("is_html")),
            in_reply_to=payload.get("in_reply_to") or None,
            references=payload.get("references") or [],
            attachments=payload.get("attachments") or [],
        )

        try:
            if mode == "send":
                resp = _send_mail.send_message(service, raw, thread_id)
            else:
                resp = _send_mail.create_draft(service, raw, thread_id)
        except Exception as e:
            return {"ok": False,
                    "error": f"send failed: {_safe_err('compose: send', e)}"}

    return {
        "ok": True, "mode": mode,
        "id": resp.get("id") or "",
        "thread_id": (resp.get("threadId")
                      or (resp.get("message") or {}).get("threadId") or ""),
    }


# --- compose with Liam: draft an email body from a plain-language brief ----
# Liam (the same claude -p assistant behind /api/ask) drafts the email the user
# is about to send. No graph/MCP access here — it writes from the brief plus
# the context the composer already has (recipients, subject, and for a
# reply/forward the original message). Returns a JSON object so the client can
# slot the subject + body into the form. Serialized so two rapid clicks can't
# spawn parallel claude processes.
_draft_lock = threading.Lock()

COMPOSE_SYSTEM = (
    "You are Liam, the user's personal mail-writing assistant. Write ONE email "
    "for the user to send, following their instruction. "
    "Respond with ONLY a JSON object: {\"subject\": \"…\", \"body\": \"…\"} — "
    "no prose, no code fences, nothing else.\n"
    "- body: plain text (NO HTML, NO markdown). Separate paragraphs with a "
    "blank line. Make it ready to send — natural greeting and sign-off — unless "
    "the instruction says otherwise.\n"
    "- subject: for a NEW message, propose a short, specific subject; for a "
    "reply or forward return \"\" (the subject is already set).\n"
    "Write in the language of the instruction (or of the original message when "
    "replying). Default to a concise, professional tone unless the user asks "
    "for another. Never invent facts, names, dates, figures or commitments that "
    "weren't given — if something essential is missing, leave a clear "
    "[placeholder] for the user to fill. When replying or forwarding, write "
    "ONLY the user's new text — never restate the quoted original or add an "
    "'On … wrote:' header (the composer keeps the quote)."
)


def _parse_json_object(text: str) -> dict:
    """Parse a single JSON object from model output that may be wrapped in
    ``` fences or surrounded by prose. Returns {} on failure."""
    s = (text or "").strip()
    if "```" in s:
        import re
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        return {}
    try:
        d = json.loads(s[i:j + 1])
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


# Prepended on a FOLLOW-UP turn (a resumed drafting conversation): tells Liam
# to revise the draft already in the editor rather than write a new one.
COMPOSE_FOLLOWUP_HINT = (
    "This is a FOLLOW-UP in the same drafting conversation — the user wants you "
    "to REVISE the current draft, not start over. The CURRENT DRAFT below is "
    "the source of truth (it may include the user's own hand-edits since your "
    "last version): revise THAT to satisfy the new instruction, keeping "
    "everything that already works and changing only what the instruction "
    "asks. Same output contract: respond with ONLY the JSON object "
    "{\"subject\": \"…\", \"body\": \"…\"} and nothing else."
)


def _do_compose_draft(payload: dict) -> dict:
    """Have Liam draft — or, on a follow-up, REVISE — an email from the user's
    brief. Stateful: if the caller passes a `session_id`, the claude
    conversation is resumed so a follow-up instruction ('make it shorter',
    'warmer tone', 'add a line about the invoice') revises the current draft in
    place; the editor's current subject/body ride along each turn so Liam
    respects hand-edits. Returns {ok, subject, body, session_id} or
    {ok: False, error}."""
    instruction = (payload.get("instruction") or "").strip()
    if not instruction:
        return {"ok": False, "error": "tell Liam what to write"}

    claude = shutil.which("claude")
    if not claude:
        return {"ok": False, "error": "the `claude` CLI is not on PATH"}

    mode = (payload.get("mode") or "new").strip().lower()
    to = payload.get("to") or []
    cc = payload.get("cc") or []
    subject = (payload.get("subject") or "").strip()
    from_email = (payload.get("from_email") or "").strip()
    original = (payload.get("original") or "").strip()
    session_id = (payload.get("session_id") or "").strip() or None
    cur_subject = (payload.get("cur_subject") or "").strip()
    cur_body = (payload.get("cur_body") or "").strip()

    # Carry the user's learned style preferences (tone, language, length) so a
    # drafted email matches how they like Liam to write — from the COMPOSE
    # scope only, kept separate from the Ask assistant's memory. Only on the
    # FIRST turn; a resumed session already carries them in its history.
    mem = ""
    style = ""
    style_name = ""
    if not session_id:
        try:
            import ask_memory
            mem = ask_memory.format_block(instruction, scope="compose")
        except Exception:
            mem = ""
        # Per-recipient writing style: detect the To addressee and, if we've
        # learned how the user writes to them, inject that voice so the draft
        # reads as if they wrote it. style_name is echoed to the client so the
        # UI can confirm whose style was applied.
        try:
            import style_profiles
            prof = style_profiles.find_profile(to)
            if prof:
                style = style_profiles.profile_block(prof)
                style_name = prof.get("name", "")
        except Exception:
            style = ""

    def build_prompt(resume: bool) -> str:
        parts: list[str] = []
        if resume:
            parts.append(COMPOSE_FOLLOWUP_HINT)
            cur = ("Subject: " + (cur_subject or "(unchanged)") + "\n\n"
                   + (cur_body or "(the draft is currently empty)"))
            parts.append("CURRENT DRAFT (revise this):\n" + cur[:8000])
            parts.append("NEW INSTRUCTION:\n" + instruction)
        else:
            if mem:
                parts.append(mem)
            if style:
                parts.append(style)
            parts.append(f"You are writing as: {from_email or '(the user)'}")
            if to:
                parts.append("To: " + ", ".join(to))
            if cc:
                parts.append("Cc: " + ", ".join(cc))
            parts.append(f"Message type: {mode}")
            if subject:
                parts.append(f"Current subject: {subject}")
            if original and mode in ("reply", "reply-all", "forward"):
                parts.append("ORIGINAL MESSAGE (context only — do NOT repeat "
                             "it):\n" + original[:6000])
            parts.append("INSTRUCTION:\n" + instruction)
        parts.append("Return the JSON object now.")
        return "\n\n".join(parts)

    def build_cmd(resume_sid: str | None) -> list[str]:
        # --output-format json wraps the answer in an envelope that carries the
        # session_id, so the client can resume for the next follow-up.
        cmd = [claude, "-p", "--output-format", "json",
               "--strict-mcp-config"]                  # no MCP needed to draft
        if resume_sid:
            cmd += ["--resume", resume_sid]
        else:
            cmd += ["--append-system-prompt", COMPOSE_SYSTEM]
            model_id = _llm_model_id()                  # None → CC default
            if model_id:
                cmd += ["--model", model_id]
        if claude.lower().endswith((".cmd", ".bat")):
            cmd = ["cmd", "/c", *cmd]
        return cmd

    def run_once(resume_sid: str | None):
        """Returns (raw_text, new_session_id, error_or_None)."""
        try:
            out = subprocess.run(build_cmd(resume_sid), cwd=ROOT,
                                 input=build_prompt(bool(resume_sid)),
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace",
                                 timeout=120, creationflags=_NO_WINDOW)
        except Exception as e:
            return "", "", _safe_err("compose draft", e)
        env = _parse_json_object(out.stdout or "")
        new_sid = (env.get("session_id") or "") if isinstance(env, dict) else ""
        is_err = bool(env.get("is_error")) if isinstance(env, dict) else False
        # On a JSON envelope the model's text is the `result` string; otherwise
        # (older CLI / non-JSON) fall back to raw stdout.
        raw = (env.get("result") if isinstance(env, dict)
               and env.get("result") is not None else (out.stdout or ""))
        if (is_err or out.returncode != 0) and not (raw or "").strip():
            return "", new_sid, "claude returned an error"
        return raw, new_sid, None

    if not _draft_lock.acquire(blocking=False):
        return {"ok": False, "error": "Liam is already drafting — one moment"}
    try:
        raw, new_sid, err = run_once(session_id)
        if err and session_id:
            # The stored session may have expired/been pruned — retry fresh.
            raw, new_sid, err = run_once(None)
    finally:
        _draft_lock.release()
    if err:
        return {"ok": False, "error": err}

    obj = _parse_json_object(raw or "")
    if obj:
        out_subject = (obj.get("subject") or "").strip()
        body = (obj.get("body") or "").strip()
    else:
        # Model didn't return clean JSON — fall back to using its raw text as
        # the body so the user still gets a draft rather than an error.
        out_subject, body = "", (raw or "").strip()
    if not body:
        return {"ok": False, "error": "Liam returned nothing — try again"}
    # Auto-learn email-writing preferences into the COMPOSE memory (background,
    # never blocks or affects the draft just produced).
    if _get_auto_learn("compose"):
        threading.Thread(target=_learn_from_compose,
                         args=(instruction, body), daemon=True).start()
    return {"ok": True, "subject": out_subject, "body": body,
            "session_id": new_sid, "style_applied": style_name}


# --- trash: move messages to Gmail Trash + purge local mirror ------------

def _purge_neo4j(trashed: set[tuple[str, str]]) -> None:
    """Fast path: detach-delete the Message + its Attachment(s). This is what
    callers need before responding to the client so the graph is consistent
    with what the user just clicked. Orphan-thread cleanup is deferred to the
    background sweep below — it scans every Thread node and isn't worth
    making the client wait for."""
    if not trashed:
        return
    import graph_app
    rows = [{"mid": m, "acct": a} for (a, m) in trashed]
    drv = graph_app.driver()
    try:
        with drv.session() as session:
            session.run("""
                UNWIND $rows AS row
                MATCH (m:Message {gmail_message_id: row.mid,
                                  account_owner: row.acct})
                OPTIONAL MATCH (m)-[:HAS_ATTACHMENT]->(a:Attachment)
                DETACH DELETE a, m
            """, rows=rows).consume()
    finally:
        drv.close()


def _clear_unread_neo4j(read: set[tuple[str, str]]) -> None:
    """Drop 'UNREAD' from each message's label_ids in Neo4j so a later cache
    rebuild keeps the read state. Mirrors _purge_neo4j's driver handling."""
    if not read:
        return
    import graph_app
    rows = [{"mid": m, "acct": a} for (a, m) in read]
    drv = graph_app.driver()
    try:
        with drv.session() as session:
            session.run("""
                UNWIND $rows AS row
                MATCH (m:Message {gmail_message_id: row.mid,
                                  account_owner: row.acct})
                SET m.label_ids =
                    [x IN coalesce(m.label_ids, []) WHERE x <> 'UNREAD']
            """, rows=rows).consume()
    finally:
        drv.close()


def _clear_spam_neo4j(unspammed: set[tuple[str, str]]) -> None:
    """Drop 'SPAM' and ensure 'INBOX' on each message's label_ids in Neo4j, so
    the message reads as non-spam (graph_app derives Message.spam from the
    SPAM label) and a later cache rebuild keeps it out of the spam page.
    Mirrors _clear_unread_neo4j's driver handling."""
    if not unspammed:
        return
    import graph_app
    rows = [{"mid": m, "acct": a} for (a, m) in unspammed]
    drv = graph_app.driver()
    try:
        with drv.session() as session:
            session.run("""
                UNWIND $rows AS row
                MATCH (m:Message {gmail_message_id: row.mid,
                                  account_owner: row.acct})
                SET m.bucket = CASE
                        WHEN 'CATEGORY_PROMOTIONS' IN coalesce(m.label_ids, []) THEN 'promotions'
                        WHEN 'CATEGORY_SOCIAL'     IN coalesce(m.label_ids, []) THEN 'social'
                        WHEN 'CATEGORY_UPDATES'    IN coalesce(m.label_ids, []) THEN 'updates'
                        WHEN 'CATEGORY_FORUMS'     IN coalesce(m.label_ids, []) THEN 'forums'
                        ELSE 'primary' END,
                    m.label_ids =
                    [x IN coalesce(m.label_ids, [])
                       WHERE x <> 'SPAM' AND x <> 'INBOX'] + 'INBOX'
            """, rows=rows).consume()
    finally:
        drv.close()


def _set_spam_neo4j(spammed: set[tuple[str, str]]) -> None:
    """Add 'SPAM' and drop 'INBOX' on each message's label_ids in Neo4j, so the
    message reads as spam (graph_app derives Message.spam from the SPAM label)
    and a later cache rebuild keeps it on the spam page. The inverse of
    _clear_spam_neo4j."""
    if not spammed:
        return
    import graph_app
    rows = [{"mid": m, "acct": a} for (a, m) in spammed]
    drv = graph_app.driver()
    try:
        with drv.session() as session:
            session.run("""
                UNWIND $rows AS row
                MATCH (m:Message {gmail_message_id: row.mid,
                                  account_owner: row.acct})
                SET m.bucket = 'spam',
                    m.label_ids =
                    [x IN coalesce(m.label_ids, [])
                       WHERE x <> 'SPAM' AND x <> 'INBOX'] + 'SPAM'
            """, rows=rows).consume()
    finally:
        drv.close()


def _purge_files_and_rebuild(trashed: set[tuple[str, str]]) -> None:
    """Slow path: rewrite emails.jsonl + emails_clean.jsonl filtering the
    trashed records out, prune the tracker files, drop orphan Threads, and
    rebuild the page cache. Runs on a background thread because, on Google-
    Drive-backed storage, the emails.jsonl rewrite alone can take 60-90s
    even when nothing actually changes (the whole 750 MB file is re-walked).

    Errors are logged but never surface to the client — by the time this
    runs, the user has already received an OK response and the graph is
    correctly updated."""
    if not trashed:
        return
    try:
        import os
        # Serialise every data-file rewrite against sync's append path under
        # _sync_lock (do_sync holds it for the whole pull). Without this, a
        # sync appending to emails.jsonl during our read→os.replace window
        # (60-90s on the Drive mount) would have its just-synced lines silently
        # dropped by the replace. The Neo4j sweep + rebuild() below don't touch
        # these files, so they stay outside the lock to keep the hold short.
        with _sync_lock:
            # Files keyed by (acct, mid) compound key.
            for path in (DATA_DIR / "emails.jsonl",
                         DATA_DIR / "emails_clean.jsonl"):
                if not path.exists():
                    continue
                tmp = path.with_suffix(path.suffix + ".tmp")
                with path.open("r", encoding="utf-8") as fin, \
                        tmp.open("w", encoding="utf-8") as fout:
                    for line in fin:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            fout.write(line)
                            continue
                        key = (rec.get("account_owner") or "",
                               rec.get("message_id") or "")
                        if key in trashed:
                            continue
                        fout.write(line)
                os.replace(tmp, path)

            # cleaned_msg_ids.txt — drop matching <acct>\t<mid> entries.
            p = DATA_DIR / "cleaned_msg_ids.txt"
            if p.exists():
                drop = {f"{a}\t{m}" for (a, m) in trashed}
                kept = [l for l in p.read_text(encoding="utf-8").splitlines()
                        if l.strip() and l.strip() not in drop]
                p.write_text("\n".join(kept) + ("\n" if kept else ""),
                             encoding="utf-8")

            # pulled_msg_ids_<acct>.txt — drop the mids per account so a
            # Gmail-side restore-from-Trash lets a future sync re-fetch them.
            by_acct: dict[str, set[str]] = {}
            for a, m in trashed:
                by_acct.setdefault(a, set()).add(m)
            for acct, mids in by_acct.items():
                p = DATA_DIR / f"pulled_msg_ids_{acct}.txt"
                if not p.exists():
                    continue
                kept = [l for l in p.read_text(encoding="utf-8").splitlines()
                        if l.strip() and l.strip() not in mids]
                p.write_text("\n".join(kept) + ("\n" if kept else ""),
                             encoding="utf-8")

        # Orphan-thread sweep — Threads whose only Messages were just deleted.
        import graph_app
        drv = graph_app.driver()
        try:
            with drv.session() as session:
                session.run("""
                    MATCH (t:Thread)
                    WHERE NOT (t)<-[:IN_THREAD]-(:Message)
                    DETACH DELETE t
                """).consume()
        finally:
            drv.close()

        # Final step: rebuild the page cache so a full reload sees the
        # deletions. The client also tracks deletions locally (REMOVED set)
        # for instant visual feedback, but the cache must catch up so new
        # tabs / reloads don't show stale rows.
        rebuild()
    except Exception as e:
        print(f"[serve] background trash cleanup failed: "
              f"{type(e).__name__}: {e}")


def _do_trash(payload: dict) -> dict:
    """Move each requested message to Gmail Trash, then purge it from local
    state. Authorization is per-account via the gmail.modify scope."""
    import _send_mail
    items = payload.get("messages") or []
    if not items:
        return {"ok": False, "error": "no messages specified"}

    # Group by account so we instantiate one Gmail service per account.
    valid_accts = load_accounts()
    by_acct: dict[str, list[str]] = {}
    for m in items:
        acct = (m.get("acct") or "").strip()
        mid = (m.get("mid") or "").strip()
        if acct in valid_accts and mid:
            by_acct.setdefault(acct, []).append(mid)

    trashed: set[tuple[str, str]] = set()
    failed: list[dict] = []

    for acct, mids in by_acct.items():
        try:
            service = _send_mail.gmail_service(acct)
        except SystemExit as e:
            for mid in mids:
                failed.append({"mid": mid, "acct": acct, "error": str(e)})
            continue
        except Exception as e:
            # RefreshError fires when a stored token was granted under fewer
            # scopes than SCOPES currently lists (Google won't upgrade on
            # refresh — re-consent required). Catch any auth-time exception
            # so the request returns a useful JSON error, not HTTP 000.
            msg = (f"{_safe_err('trash: gmail_service', e)} — reconnect "
                   f"'{acct}' via the ⚙ Accounts panel (or "
                   f"`pull_gmail.py --account {acct} --auth`; needs the "
                   f"gmail.modify scope).")
            for mid in mids:
                failed.append({"mid": mid, "acct": acct, "error": msg})
            continue
        for mid in mids:
            try:
                service.users().messages().trash(
                    userId="me", id=mid).execute()
                trashed.add((acct, mid))
            except Exception as e:
                failed.append({"mid": mid, "acct": acct,
                               "error": _safe_err("trash: modify", e)})

    if trashed:
        # Fast path the client waits for: Neo4j is now consistent with what
        # the user just clicked. The slow file-rewrite + page-cache rebuild
        # runs in the background — frontend tracks the deleted ids in its
        # own REMOVED set for instant visual feedback until the cache
        # version bumps and pollVersion offers the reload banner.
        _purge_neo4j(trashed)
        threading.Thread(target=_purge_files_and_rebuild,
                         args=(trashed,), daemon=True).start()

    return {"ok": len(trashed) > 0 or not items,
            "trashed": len(trashed),
            "failed": failed}


def _do_mark_read(payload: dict) -> dict:
    """Clear the UNREAD label on each requested message in Gmail (and drop it
    from the Neo4j node), so a message opened in this app reads as 'read' in
    every mail client. Mirrors _do_trash's per-account service handling;
    authorization is the same gmail.modify scope.

    Uses Gmail's batchModify — one request clears the label on up to 1000 ids,
    instead of one round-trip per message (which made a large selection crawl).
    batchModify is all-or-nothing, so on a batch error we fall back to per-
    message modify for that chunk to salvage the valid ids."""
    import _send_mail
    items = payload.get("messages") or []
    if not items:
        return {"ok": False, "error": "no messages specified"}

    valid_accts = load_accounts()
    by_acct: dict[str, list[str]] = {}
    for m in items:
        acct = (m.get("acct") or "").strip()
        mid = (m.get("mid") or "").strip()
        if acct in valid_accts and mid:
            by_acct.setdefault(acct, []).append(mid)

    read: set[tuple[str, str]] = set()
    failed: list[dict] = []

    for acct, mids in by_acct.items():
        try:
            service = _send_mail.gmail_service(acct)
        except SystemExit as e:
            for mid in mids:
                failed.append({"mid": mid, "acct": acct, "error": str(e)})
            continue
        except Exception as e:
            msg = (f"{_safe_err('label: gmail_service', e)} — reconnect "
                   f"'{acct}' via the ⚙ Accounts panel (or "
                   f"`pull_gmail.py --account {acct} --auth`; needs the "
                   f"gmail.modify scope).")
            for mid in mids:
                failed.append({"mid": mid, "acct": acct, "error": msg})
            continue
        for start in range(0, len(mids), 1000):
            chunk = mids[start:start + 1000]
            try:
                service.users().messages().batchModify(
                    userId="me",
                    body={"ids": chunk,
                          "removeLabelIds": ["UNREAD"]}).execute()
                read.update((acct, mid) for mid in chunk)
            except Exception:
                # batchModify rejects the whole batch if any id is bad (e.g.
                # trashed elsewhere). Retry per-message so the good ids land.
                for mid in chunk:
                    try:
                        service.users().messages().modify(
                            userId="me", id=mid,
                            body={"removeLabelIds": ["UNREAD"]}).execute()
                        read.add((acct, mid))
                    except Exception as e:
                        failed.append({"mid": mid, "acct": acct,
                                       "error": _safe_err("label modify", e)})

    if read:
        _clear_unread_neo4j(read)

    return {"ok": len(read) > 0 or not items,
            "marked": len(read),
            "failed": failed}


def _do_not_spam(payload: dict) -> dict:
    """Move each requested message out of Spam: remove the SPAM label and add
    INBOX in Gmail (and update the Neo4j node), so it reads as a normal inbox
    message everywhere. Mirrors _do_mark_read's per-account, batchModify-with-
    per-message-fallback handling; authorization is the same gmail.modify
    scope."""
    import _send_mail
    items = payload.get("messages") or []
    if not items:
        return {"ok": False, "error": "no messages specified"}

    valid_accts = load_accounts()
    by_acct: dict[str, list[str]] = {}
    for m in items:
        acct = (m.get("acct") or "").strip()
        mid = (m.get("mid") or "").strip()
        if acct in valid_accts and mid:
            by_acct.setdefault(acct, []).append(mid)

    unspammed: set[tuple[str, str]] = set()
    failed: list[dict] = []
    body = {"removeLabelIds": ["SPAM"], "addLabelIds": ["INBOX"]}

    for acct, mids in by_acct.items():
        try:
            service = _send_mail.gmail_service(acct)
        except SystemExit as e:
            for mid in mids:
                failed.append({"mid": mid, "acct": acct, "error": str(e)})
            continue
        except Exception as e:
            msg = (f"{_safe_err('label: gmail_service', e)} — reconnect "
                   f"'{acct}' via the ⚙ Accounts panel (or "
                   f"`pull_gmail.py --account {acct} --auth`; needs the "
                   f"gmail.modify scope).")
            for mid in mids:
                failed.append({"mid": mid, "acct": acct, "error": msg})
            continue
        for start in range(0, len(mids), 1000):
            chunk = mids[start:start + 1000]
            try:
                service.users().messages().batchModify(
                    userId="me",
                    body={"ids": chunk, **body}).execute()
                unspammed.update((acct, mid) for mid in chunk)
            except Exception:
                # batchModify rejects the whole batch if any id is bad (e.g.
                # trashed elsewhere). Retry per-message so the good ids land.
                for mid in chunk:
                    try:
                        service.users().messages().modify(
                            userId="me", id=mid, body=body).execute()
                        unspammed.add((acct, mid))
                    except Exception as e:
                        failed.append({"mid": mid, "acct": acct,
                                       "error": _safe_err("label modify", e)})

    if unspammed:
        _clear_spam_neo4j(unspammed)

    return {"ok": len(unspammed) > 0 or not items,
            "unspammed": len(unspammed),
            "failed": failed}


def _do_mark_spam(payload: dict) -> dict:
    """Mark each requested message as spam: add the SPAM label and remove INBOX
    in Gmail (and update the Neo4j node), so it moves to the spam page. The
    inverse of _do_not_spam; same per-account, batchModify-with-per-message-
    fallback handling and gmail.modify scope."""
    import _send_mail
    items = payload.get("messages") or []
    if not items:
        return {"ok": False, "error": "no messages specified"}

    valid_accts = load_accounts()
    by_acct: dict[str, list[str]] = {}
    for m in items:
        acct = (m.get("acct") or "").strip()
        mid = (m.get("mid") or "").strip()
        if acct in valid_accts and mid:
            by_acct.setdefault(acct, []).append(mid)

    spammed: set[tuple[str, str]] = set()
    failed: list[dict] = []
    body = {"addLabelIds": ["SPAM"], "removeLabelIds": ["INBOX"]}

    for acct, mids in by_acct.items():
        try:
            service = _send_mail.gmail_service(acct)
        except SystemExit as e:
            for mid in mids:
                failed.append({"mid": mid, "acct": acct, "error": str(e)})
            continue
        except Exception as e:
            msg = (f"{_safe_err('label: gmail_service', e)} — reconnect "
                   f"'{acct}' via the ⚙ Accounts panel (or "
                   f"`pull_gmail.py --account {acct} --auth`; needs the "
                   f"gmail.modify scope).")
            for mid in mids:
                failed.append({"mid": mid, "acct": acct, "error": msg})
            continue
        for start in range(0, len(mids), 1000):
            chunk = mids[start:start + 1000]
            try:
                service.users().messages().batchModify(
                    userId="me",
                    body={"ids": chunk, **body}).execute()
                spammed.update((acct, mid) for mid in chunk)
            except Exception:
                # batchModify rejects the whole batch if any id is bad (e.g.
                # trashed elsewhere). Retry per-message so the good ids land.
                for mid in chunk:
                    try:
                        service.users().messages().modify(
                            userId="me", id=mid, body=body).execute()
                        spammed.add((acct, mid))
                    except Exception as e:
                        failed.append({"mid": mid, "acct": acct,
                                       "error": _safe_err("label modify", e)})

    if spammed:
        _set_spam_neo4j(spammed)

    return {"ok": len(spammed) > 0 or not items,
            "spammed": len(spammed),
            "failed": failed}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):           # quiet — we print our own lines
        pass

    def _allowed_hosts(self) -> set[str]:
        """Loopback Host/Origin authorities this server answers to, on its
        actual bound port."""
        port = self.server.server_address[1]
        return {f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"}

    def _guard(self) -> bool:
        """Reject DNS-rebinding and cross-site (CSRF) requests before any
        handler runs. The app has no authentication — it trusts that only the
        local user's own browser reaches it — so we enforce that:

          1. Host must be a loopback authority on our port. Under DNS
             rebinding the browser sends the attacker's domain as Host
             (e.g. evil.com:8765), so this kills rebinding-based reads of
             /api/payload, /api/accounts, etc.
          2. Any request carrying a cross-origin Origin is refused. A cross-
             site POST (even a CORS "simple request" with text/plain that
             skips preflight) always carries the attacker page's Origin, so
             this kills CSRF against /api/compose, /api/trash, /api/auth, …
             Same-origin requests send our own Origin (or none, e.g. top-level
             GET navigations and same-origin GETs) and pass.

        Returns True if allowed; on rejection it has already sent 403."""
        allowed = self._allowed_hosts()
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in allowed:
            self._send(403, "forbidden: bad Host", "text/plain")
            return False
        origin = (self.headers.get("Origin") or "").strip()
        if origin and urlsplit(origin).netloc.lower() not in allowed:
            self._send(403, "forbidden: cross-origin", "text/plain")
            return False
        return True

    def _send(self, code: int, body, ctype: str, *,
              body_gz: bytes | None = None) -> None:
        """Send a response, gzip-encoding the body when the client accepts it
        and (a) a precomputed gzip blob was supplied, or (b) the body is large
        enough to be worth compressing on the fly."""
        b = body.encode("utf-8") if isinstance(body, str) else body
        accept = (self.headers.get("Accept-Encoding") or "").lower()
        gz_out: bytes | None = None
        if "gzip" in accept:
            if body_gz is not None:
                gz_out = body_gz
            elif len(b) > 1024:
                gz_out = gzip.compress(b, compresslevel=6)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if gz_out is not None:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(gz_out)))
        else:
            self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(gz_out if gz_out is not None else b)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse_write(self, payload) -> bool:
        """Write one Server-Sent Events frame. Returns False if the client
        has disconnected (so the caller can abort/terminate work in flight)."""
        try:
            self.wfile.write(
                ("data: " + json.dumps(payload, ensure_ascii=False)
                 + "\n\n").encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def do_GET(self):
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with _lock:
                html = _state["html"]
                html_gz = _state["html_gz"]
            self._send(200, html, "text/html; charset=utf-8",
                       body_gz=html_gz or None)
        elif path == "/manifest.webmanifest":
            # PWA manifest — lets the browser "Install" the app as a standalone
            # window titled "Mail Graph" with no address bar.
            import graph_app
            self._send(200, graph_app.MANIFEST_JSON,
                       "application/manifest+json; charset=utf-8")
        elif path in ("/icon.svg", "/favicon.ico"):
            # App / tab icon (SVG works for both the <link rel=icon> and the
            # browser's implicit /favicon.ico probe).
            import graph_app
            self._send(200, graph_app.ICON_SVG, "image/svg+xml; charset=utf-8")
        elif path == "/api/boot":
            # Cold-start progress for the loading splash: {ready, phase, error}.
            self._send(200, json.dumps(_boot_snapshot()), "application/json")
        elif path == "/api/version":
            with _lock:
                v = {"version": _state["version"],
                     "syncing": _state["syncing"],
                     "messages": _state["messages"]}
            self._send(200, json.dumps(v), "application/json")
        elif path == "/api/payload":
            # Just the JSON payload — the same data inlined into / as the
            # __app-data tag, served standalone so the client can swap in
            # fresh data after a sync without forcing a full page reload.
            with _lock:
                pj = _state["payload_json"]
                pgz = _state["payload_gz"]
            self._send(200, pj or "{}", "application/json",
                       body_gz=pgz or None)
        elif path == "/api/events":
            # Server-Sent Events stream of {version, syncing, messages} —
            # one frame on connect, then one frame per change. Replaces the
            # 30s pollVersion loop with a push model: zero traffic between
            # syncs, and the syncing-indicator flips the instant the
            # background loop starts a pull. Keep-alive comment every 25s so
            # intermediate proxies don't drop the idle connection.
            self.send_response(200)
            self.send_header("Content-Type",
                             "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            with _lock:
                last_v = _state["version"]
                last_s = _state["syncing"]
                initial = {"version": last_v, "syncing": last_s,
                           "messages": _state["messages"]}
            try:
                self.wfile.write(
                    ("data: " + json.dumps(initial)
                     + "\n\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            # This stream now represents one open app window, keyed by its
            # per-window cid (?cid=…; synthesized if an older page omits it).
            # The keepalive write doubles as a liveness probe: if the window
            # closes without a /api/closing beacon, the next write fails and
            # the `finally` drops the cid — the implicit fallback that lets the
            # idle watchdog shut the server down.
            qs = parse_qs(urlsplit(self.path).query)
            cid = (qs.get("cid") or [""])[0] or _next_anon_cid()
            _client_connected(cid)
            try:
                while True:
                    with _state_cv:
                        cur_v = _state["version"]
                        cur_s = _state["syncing"]
                        if cur_v == last_v and cur_s == last_s:
                            _state_cv.wait(timeout=SSE_KEEPALIVE_SECS)
                            cur_v = _state["version"]
                            cur_s = _state["syncing"]
                        messages = _state["messages"]
                    changed = (cur_v != last_v or cur_s != last_s)
                    last_v, last_s = cur_v, cur_s
                    try:
                        if changed:
                            self.wfile.write(
                                ("data: " + json.dumps(
                                    {"version": cur_v, "syncing": cur_s,
                                     "messages": messages})
                                 + "\n\n").encode("utf-8"))
                        else:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
            finally:
                _client_disconnected(cid)
        elif path == "/api/body":
            import body_store
            qs = parse_qs(self.path.partition("?")[2])
            mid = (qs.get("mid") or [""])[0]
            acct = (qs.get("acct") or [""])[0]
            self._send(200, body_store.get(mid, acct),
                       "text/html; charset=utf-8")
        elif path == "/api/bodysearch":
            # Full-body filter: keys of messages whose complete clean body
            # contains ?q. The lean page payload only carries a 240-char
            # snippet, so the body column box / body: terms ask the server
            # (which has body_clean for every message) for the match set.
            qs = parse_qs(self.path.partition("?")[2])
            term = (qs.get("q") or [""])[0]
            try:
                hits = _body_search(term)
            except Exception as e:
                self._send(500, json.dumps(
                    {"error": _safe_err("api/bodysearch", e)}),
                    "application/json")
                return
            self._send(200, json.dumps({"q": term, "hits": hits}),
                       "application/json")
        elif path == "/api/style":
            # Status of the per-recipient writing-style profiles Liam Compose
            # learns autonomously: when last built, count, and the recipients.
            try:
                import style_profiles
                self._send(200, json.dumps(style_profiles.status()),
                           "application/json")
            except Exception as e:
                self._send(500, json.dumps(
                    {"error": _safe_err("api/style", e)}), "application/json")
        elif path == "/api/accounts":
            try:
                # known_emails is a Neo4j-backed recipient-autocomplete extra.
                # If the DB is momentarily unreachable, degrade gracefully to an
                # empty list rather than 500 — Compose only needs `accounts`
                # (labels + emails from the token files, no Neo4j), so it must
                # not break just because autocomplete can't load.
                try:
                    known = _known_emails()
                except Exception as e:
                    print(f"[serve] api/accounts: known_emails skipped "
                          f"({type(e).__name__}: {e})")
                    known = []
                data = {"accounts": _account_status(), "known_emails": known}
                self._send(200, json.dumps(data), "application/json")
            except Exception as e:
                self._send(500, json.dumps(
                    {"error": _safe_err("api/accounts", e)}),
                    "application/json")
        elif path == "/api/auth/status":
            # Poll target for the accounts panel: the current/last result of a
            # sign-in flow for ?account=<label> — "idle" | "running" | "ok" |
            # "error:<msg>".
            qs = parse_qs(self.path.partition("?")[2])
            acct = (qs.get("account") or [""])[0]
            with _auth_status_lock:
                st = _auth_status.get(acct, "idle")
            self._send(200, json.dumps({"status": st}), "application/json")
        elif path == "/api/settings":
            # Settings panel state. Reports the chosen LLM model key plus the
            # available options so the UI never hard-codes the model list.
            data = {"llm_model": _get_llm_model_key(),
                    "llm_models": list(LLM_MODELS.keys()),
                    "ask_auto_learn": _get_auto_learn("ask"),
                    "compose_auto_learn": _get_auto_learn("compose")}
            self._send(200, json.dumps(data), "application/json")
        elif path == "/api/memory":
            # Long-term memory for the Settings → Memory panel. ?scope=ask
            # (default) or ?scope=compose selects which assistant's store.
            try:
                import ask_memory
                qs = parse_qs(urlsplit(self.path).query)
                scope = (qs.get("scope") or ["ask"])[0]
                scope = scope if scope in ("ask", "compose") else "ask"
                data = {"scope": scope,
                        "memories": ask_memory.list_memories(scope),
                        "auto_learn": _get_auto_learn(scope)}
                self._send(200, json.dumps(data), "application/json")
            except Exception as e:
                self._send(500, json.dumps(
                    {"error": _safe_err("api/memory", e)}), "application/json")
        elif path == "/api/claude/status":
            # Claude Code subscription login state, for the Settings panel.
            with _claude_login_lock:
                login = dict(_claude_login_state)
            data = {**_claude_status(), "login": login}
            self._send(200, json.dumps(data), "application/json")
        elif path == "/api/quote":
            qs = parse_qs(self.path.partition("?")[2])
            mid = (qs.get("mid") or [""])[0]
            acct = (qs.get("acct") or [""])[0]
            try:
                q = _fetch_quote(mid, acct)
            except Exception as e:
                self._send(500, json.dumps(
                    {"error": _safe_err("api/quote", e)}),
                    "application/json")
                return
            if q is None:
                self._send(404, json.dumps({"error": "message not found"}),
                           "application/json")
                return
            self._send(200, json.dumps(q), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if not self._guard():
            return
        p = self.path.split("?", 1)[0]
        if p == "/api/boot/retry":
            # Re-run a failed cold-start boot (e.g. after the user started
            # Docker). No-op if already running or ready.
            _start_boot()
            self._send(200, json.dumps(_boot_snapshot()), "application/json")
        elif p == "/api/style/rebuild":
            # Force a re-distillation of the per-recipient writing-style cards
            # now (vs the autonomous, freshness-gated startup pass). Optional
            # body {force: bool} re-distils even unchanged recipients.
            n = int(self.headers.get("Content-Length", 0) or 0)
            params: dict = {}
            if n:
                try:
                    params = json.loads(self.rfile.read(n) or b"{}")
                except (json.JSONDecodeError, ValueError):
                    params = {}
            force = bool(params.get("force", False))

            def _run_style():
                try:
                    import style_profiles
                    my = [e for e in _account_emails().values() if e]
                    style_profiles.build(my, force=force)
                except Exception as e:
                    print(f"[serve] style rebuild failed: {type(e).__name__}")
            threading.Thread(target=_run_style, daemon=True).start()
            self._send(200, json.dumps({"started": True}), "application/json")
        elif p == "/api/sync":
            # Optional JSON body: {account?: str, defer_embed?: bool}.
            # account → fast targeted sync for that mailbox only.
            n = int(self.headers.get("Content-Length", 0) or 0)
            params: dict = {}
            if n:
                try:
                    params = json.loads(self.rfile.read(n) or b"{}")
                except (json.JSONDecodeError, ValueError):
                    params = {}
            only_account = (params.get("account") or "").strip() or None
            # Missing key → match the background sync_loop's default
            # (defer_embed=True) so the manual ↻ Sync button behaves the
            # same as auto-sync. Callers that need inline embedding must
            # pass defer_embed=false explicitly.
            defer_embed = bool(params.get("defer_embed", True))
            threading.Thread(
                target=lambda: do_sync(only_account=only_account,
                                       defer_embed=defer_embed),
                daemon=True).start()
            self._send(200, json.dumps({"started": True}),
                       "application/json")
        elif p == "/api/closing":
            # Beacon from a window's `pagehide` handler: this app window is
            # going away. Drop its cid immediately so the idle watchdog can
            # shut the server down without waiting to notice the dropped SSE
            # socket. Idempotent with the SSE handler's own cleanup (set
            # discard). The body is empty; the cid rides in the query string
            # so a navigator.sendBeacon with no payload works.
            qs = parse_qs(urlsplit(self.path).query)
            cid = (qs.get("cid") or [""])[0]
            if cid:
                _client_disconnected(cid)
            self._send(204, b"", "text/plain")
        elif p == "/api/shutdown":
            # The app's power button: stop the server now (gracefully) so the
            # page can show a visible "stopping → stopped" state. Reply first,
            # then shut down on a short delay so this response flushes first.
            self._send(200, json.dumps({"ok": True}), "application/json")

            def _stop():
                time.sleep(0.4)
                if _srv is not None:
                    _srv.shutdown()
            threading.Thread(target=_stop, daemon=True).start()
        elif p == "/api/sync/hold":
            # Client heartbeat: {"seconds": N} extends the auto-sync hold to
            # max(current, now+N). seconds=0 (or missing) clears the hold.
            # The TTL means a crashed client never permanently disables sync.
            n = int(self.headers.get("Content-Length", 0) or 0)
            params: dict = {}
            if n:
                try:
                    params = json.loads(self.rfile.read(n) or b"{}")
                except (json.JSONDecodeError, ValueError):
                    params = {}
            try:
                seconds = float(params.get("seconds") or 0)
            except (TypeError, ValueError):
                seconds = 0.0
            global _hold_until
            with _hold_lock:
                if seconds <= 0:
                    _hold_until = 0.0
                else:
                    _hold_until = max(_hold_until, time.time() + seconds)
                held_until = _hold_until
            self._send(200, json.dumps({"held_until": held_until}),
                       "application/json")
        elif p == "/api/auth":
            # Trigger the OAuth consent flow for {account} server-side (opens
            # the user's browser for Google sign-in). Returns immediately;
            # the client polls /api/auth/status for completion.
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400,
                           json.dumps({"started": False, "error": "bad JSON"}),
                           "application/json")
                return
            result = _start_auth((payload.get("account") or "").strip(),
                                 allow_new=False)
            self._send(200 if result.get("started") else 400,
                       json.dumps(result), "application/json")
        elif p == "/api/accounts/add":
            # Add a brand-new mailbox: validate the label, then run the OAuth
            # consent flow (label is persisted only on success). Client polls
            # /api/auth/status?account=<label> exactly like /api/auth.
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400,
                           json.dumps({"started": False, "error": "bad JSON"}),
                           "application/json")
                return
            result = _start_auth((payload.get("label") or "").strip().lower(),
                                 allow_new=True)
            self._send(200 if result.get("started") else 400,
                       json.dumps(result), "application/json")
        elif p == "/api/accounts/remove":
            # Detach an account ({label}); with {purge:true} also delete its
            # mail from the graph + data files (destructive, runs in the
            # background).
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400,
                           json.dumps({"ok": False, "error": "bad JSON"}),
                           "application/json")
                return
            result = _remove_account((payload.get("label") or "").strip().lower(),
                                     purge=bool(payload.get("purge")))
            self._send(200 if result.get("ok") else 400,
                       json.dumps(result), "application/json")
        elif p == "/api/settings":
            # Persist a settings change. Only known keys are accepted; the LLM
            # model is validated against LLM_MODELS (unknown → 'default').
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400, json.dumps({"ok": False, "error": "bad JSON"}),
                           "application/json")
                return
            s = _load_settings()
            if "llm_model" in payload:
                key = payload.get("llm_model")
                s["llm_model"] = key if key in LLM_MODELS else "default"
            if "ask_auto_learn" in payload:
                s["ask_auto_learn"] = bool(payload.get("ask_auto_learn"))
            if "compose_auto_learn" in payload:
                s["compose_auto_learn"] = bool(payload.get("compose_auto_learn"))
            _save_settings(s)
            self._send(200, json.dumps({"ok": True,
                       "llm_model": _get_llm_model_key(),
                       "ask_auto_learn": _get_auto_learn("ask"),
                       "compose_auto_learn": _get_auto_learn("compose")}),
                       "application/json")
        elif p == "/api/memory/add":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400, json.dumps({"ok": False, "error": "bad JSON"}),
                           "application/json")
                return
            try:
                import ask_memory
                scope = (payload.get("scope") or "ask")
                scope = scope if scope in ("ask", "compose") else "ask"
                rec = ask_memory.add((payload.get("text") or ""),
                                     payload.get("kind") or "fact",
                                     source="user", scope=scope)
                self._send(200, json.dumps({"ok": bool(rec), "memory": rec}),
                           "application/json")
            except Exception as e:
                self._send(500, json.dumps(
                    {"ok": False, "error": _safe_err("api/memory/add", e)}),
                    "application/json")
        elif p == "/api/memory/delete":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400, json.dumps({"ok": False, "error": "bad JSON"}),
                           "application/json")
                return
            try:
                import ask_memory
                scope = (payload.get("scope") or "ask")
                scope = scope if scope in ("ask", "compose") else "ask"
                ok = ask_memory.delete((payload.get("id") or "").strip(),
                                       scope=scope)
                self._send(200, json.dumps({"ok": ok}), "application/json")
            except Exception as e:
                self._send(500, json.dumps(
                    {"ok": False, "error": _safe_err("api/memory/delete", e)}),
                    "application/json")
        elif p == "/api/claude/login":
            # Launch the Claude Code subscription OAuth flow (opens a browser).
            result = _claude_login_start()
            self._send(200 if result.get("started") else 400,
                       json.dumps(result), "application/json")
        elif p == "/api/claude/logout":
            result = _claude_logout()
            self._send(200 if result.get("ok") else 400,
                       json.dumps(result), "application/json")
        elif p == "/api/compose":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400,
                           json.dumps({"ok": False, "error": "bad JSON"}),
                           "application/json")
                return
            result = _do_compose(payload)
            self._send(200 if result.get("ok") else 400,
                       json.dumps(result), "application/json")
        elif p == "/api/compose/draft":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400,
                           json.dumps({"ok": False, "error": "bad JSON"}),
                           "application/json")
                return
            result = _do_compose_draft(payload)
            self._send(200 if result.get("ok") else 400,
                       json.dumps(result), "application/json")
        elif p == "/api/trash":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400,
                           json.dumps({"ok": False, "error": "bad JSON"}),
                           "application/json")
                return
            result = _do_trash(payload)
            self._send(200 if result.get("ok") else 400,
                       json.dumps(result), "application/json")
        elif p == "/api/seen":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400,
                           json.dumps({"ok": False, "error": "bad JSON"}),
                           "application/json")
                return
            result = _do_mark_read(payload)
            self._send(200 if result.get("ok") else 400,
                       json.dumps(result), "application/json")
        elif p == "/api/notspam":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400,
                           json.dumps({"ok": False, "error": "bad JSON"}),
                           "application/json")
                return
            result = _do_not_spam(payload)
            self._send(200 if result.get("ok") else 400,
                       json.dumps(result), "application/json")
        elif p == "/api/markspam":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400,
                           json.dumps({"ok": False, "error": "bad JSON"}),
                           "application/json")
                return
            result = _do_mark_spam(payload)
            self._send(200 if result.get("ok") else 400,
                       json.dumps(result), "application/json")
        elif p == "/api/ask":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
                q = (data.get("q") or "").strip()
                sid = (data.get("session_id") or "").strip() or None
            except (json.JSONDecodeError, ValueError):
                q, sid = "", None
            if not q:
                self._send(400, json.dumps({"error": "empty question"}),
                           "application/json")
                return
            # Stream the answer as Server-Sent Events. The client renders
            # incremental events (phase / thinking text / tool_use) live in
            # the "Thinking…" bubble, then swaps in the final answer when the
            # `done` event arrives.
            self.send_response(200)
            self.send_header("Content-Type",
                             "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                _stream_ask(self._sse_write, q, sid)
            except Exception as e:
                self._sse_write({"type": "error",
                                 "message": _safe_err("api/ask", e)})
        elif p == "/api/ask/feedback":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._send(400, json.dumps({"ok": False, "error": "bad JSON"}),
                           "application/json")
                return
            result = _do_ask_feedback(payload)
            self._send(200 if result.get("ok") else 400,
                       json.dumps(result), "application/json")
        else:
            self._send(404, "not found", "text/plain")


# Default size of the standalone app window (W,H in DIP), matching the size
# the window settled at in normal use. Chrome applies it only the first time a
# given app window is created; afterwards it remembers the user's own resize.
_APP_WINDOW_SIZE = "1255,832"


def _open_app_window(url: str) -> None:
    """Open the app in a Chromium 'app' window — a standalone window (no address
    bar or tabs) at a fixed size (_APP_WINDOW_SIZE). Falls back to the default
    browser if no Chromium is found.

    A DEDICATED profile dir (--user-data-dir) is used so we always launch our
    OWN Chrome instance: if we reused the user's normal Chrome, then whenever
    their main Chrome was already running our launch would just hand the URL to
    that instance and IGNORE --window-size (and --app sizing). A separate
    instance honors the flags. The profile persists under %LOCALAPPDATA%\\MailGraph
    so app state (history, etc.) carries across launches."""
    import os
    rels = (r"Google\Chrome\Application\chrome.exe",
            r"Microsoft\Edge\Application\msedge.exe")
    bases = [os.environ.get("ProgramFiles", r"C:\Program Files"),
             os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
             os.environ.get("LOCALAPPDATA", "")]
    cands = [os.path.join(b, r) for b in bases if b for r in rels]
    cands += [shutil.which("chrome"), shutil.which("msedge")]
    profile = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                           "MailGraph", "chrome-profile")
    for exe in cands:
        if exe and os.path.exists(exe):
            try:
                # --app = single standalone window. --window-size restores the
                # previous fixed size. --user-data-dir forces our OWN instance
                # so the size is honored even if Chrome is already open.
                # --no-first-run/--no-default-browser-check suppress the fresh
                # profile's welcome prompts.
                subprocess.Popen([exe, f"--app={url}",
                                  f"--window-size={_APP_WINDOW_SIZE}",
                                  f"--user-data-dir={profile}",
                                  "--no-first-run",
                                  "--no-default-browser-check"])
                return
            except OSError:
                continue
    webbrowser.open(url)


_neo4j_proc = None          # Popen of an app-launched `neo4j console`, if any


def _start_native_neo4j() -> None:
    """Make the native Neo4j reachable — no Docker, no WSL. Two paths, tried in
    order:

      1. A registered Windows service (`neo4j` by default). A best-effort
         Start-Service nudges it up; if it's set to Automatic it's already
         running and this is a no-op. The service is the preferred end state:
         always warm, so the app pays no DB cold start.
      2. If there's no service (or we lack rights to start it), launch the
         bundled `bin\\neo4j console` as a child process with JAVA_HOME pinned to
         the bundled JDK. This keeps the app self-sufficient with no admin step,
         and we stop it on exit (see _shutdown_native_neo4j).

    Returns immediately; the caller polls _neo4j_reachable() for readiness."""
    global _neo4j_proc
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"Start-Service -Name '{NEO4J_SERVICE}' -ErrorAction Stop"],
                capture_output=True, text=True, timeout=60,
                creationflags=_NO_WINDOW)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
    if _neo4j_reachable():
        return                              # service (or an already-up instance)
    # Fallback: launch the bundled server ourselves.
    if _neo4j_proc is not None and _neo4j_proc.poll() is None:
        return                              # we already started one
    bat = NEO4J_HOME / "bin" / ("neo4j.bat" if sys.platform == "win32"
                                else "neo4j")
    if not bat.exists():
        return
    import os
    env = dict(os.environ)
    env["JAVA_HOME"] = str(NEO4J_JAVA_HOME)
    try:
        _neo4j_proc = subprocess.Popen(
            [str(bat), "console"], env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    except OSError:
        _neo4j_proc = None


def _shutdown_native_neo4j() -> None:
    """On app exit, stop only a Neo4j we launched ourselves (the console
    fallback). A Windows service — or an instance that was already running
    before the app started — is left alone. Best-effort and quiet.

    `neo4j console` runs java under a cmd.exe shim, so terminating the Popen
    alone would orphan the JVM (and leave port 7687 held). taskkill /T tears
    down the whole tree; Neo4j is crash-safe, so a forced kill on exit is fine."""
    global _neo4j_proc
    if _neo4j_proc is not None and _neo4j_proc.poll() is None:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(_neo4j_proc.pid),
                                "/T", "/F"], capture_output=True, text=True,
                               timeout=60, creationflags=_NO_WINDOW)
            else:
                _neo4j_proc.terminate()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
    _neo4j_proc = None


def _neo4j_reachable() -> bool:
    """True only once the default `neo4j` database is actually queryable — not
    merely that Bolt + auth answer. On a cold start (especially a freshly
    migrated store) the server accepts connections and authenticates seconds
    before the database finishes recovering; gating on verify_connectivity
    alone let boot advance to the graph load too early and fail with
    DatabaseUnavailable. A trivial query against the default db is the real
    readiness signal, and it doesn't touch auth so it can't trip the lockout."""
    import graph_app
    drv = graph_app.driver()
    try:
        with drv.session() as s:
            # Must touch the graph store, not a constant: Neo4j answers
            # `RETURN 1` while the database is still recovering, so it would
            # report ready too early. A scan forces DatabaseUnavailable until
            # the store is actually online. .consume() (not .single()) so it
            # also passes on an empty graph.
            s.run("MATCH (n) RETURN n LIMIT 1").consume()
        return True
    except Exception:
        return False
    finally:
        drv.close()


# --- boot sequence ----------------------------------------------------------
# Boot serves the loading splash immediately and runs the slow work — ensure
# Neo4j is up → load the graph → build the page cache — in a background thread.
# The splash polls /api/boot for `phase`/`ready`/`error` and
# reloads into the real app once `ready` flips true. /api/boot/retry re-runs a
# failed boot. With native Neo4j the only thing shutdown reverses is a console
# instance the app itself launched (see _shutdown_native_neo4j); a Windows
# service is left running.
_boot = {"ready": False, "running": False, "phase": "Starting…", "error": ""}
_boot_lock = threading.Lock()
_sync_interval = 600


def _set_boot(**kw) -> None:
    with _boot_lock:
        _boot.update(kw)


def _boot_snapshot() -> dict:
    with _boot_lock:
        return dict(_boot)


def _notify_dialog(msg: str) -> None:
    """Native popup so a console-less (pythonw) launch isn't silent."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, "Mail Graph", 0x10)
    except Exception:
        pass


def _boot_sequence() -> None:
    try:
        if not _neo4j_reachable():
            _set_boot(phase="Starting Neo4j…")
            _start_native_neo4j()
            ok = False
            for _ in range(40):                 # up to ~80s for a cold console
                time.sleep(2)
                if _neo4j_reachable():
                    ok = True
                    break
            if not ok:
                msg = ("Neo4j is not reachable.\nStart the Windows service "
                       f"'{NEO4J_SERVICE}', or check the native install at "
                       f"{NEO4J_HOME}, then press Retry.")
                _set_boot(phase="Neo4j unavailable", error=msg, running=False)
                _notify_dialog(msg)
                return
        _set_boot(phase="Loading graph…", error="")
        # Even once the store answers a probe, the database can briefly report
        # DatabaseUnavailable while it finishes coming online after a cold
        # start. Retry the graph load a few times before surfacing an error.
        for attempt in range(15):               # ~30s
            try:
                rebuild()                       # swaps _state["html"] → real app
                break
            except Exception as e:
                if "Unavailable" not in repr(e) or attempt == 14:
                    raise
                time.sleep(2)
        _set_boot(phase="Starting background services…")
        threading.Thread(target=warm_embedder, daemon=True).start()
        threading.Thread(target=sync_loop, args=(_sync_interval,),
                         daemon=True).start()
        threading.Thread(target=build_style_profiles, daemon=True).start()
        # Self-heal: recover any mail stranded by an interrupted previous-session
        # sync (in emails.jsonl but never loaded). Background so it never delays
        # startup; rebuilds the page if it recovered anything so the window's
        # reload banner offers the now-complete graph.
        threading.Thread(target=_reconcile_on_boot, daemon=True).start()
        _set_boot(phase="Ready", ready=True, running=False)
    except Exception as e:
        _set_boot(phase="Startup failed", error=_safe_err("boot", e),
                  running=False)


def _reconcile_on_boot() -> None:
    """Run reconcile_graph() once at startup and rebuild the page cache if it
    recovered anything. Best-effort — a failure here must never break boot."""
    try:
        n = reconcile_graph()
        if n:
            print(f"[serve] reconcile recovered {n} message(s); rebuilding",
                  flush=True)
            rebuild()
    except Exception as e:
        print(f"[serve] reconcile skipped: {type(e).__name__}: {e}", flush=True)


def _start_boot() -> None:
    """Kick off the boot sequence once (idempotent: no-op if already running or
    already ready)."""
    with _boot_lock:
        if _boot["running"] or _boot["ready"]:
            return
        _boot.update(running=True, error="", phase="Starting…")
    threading.Thread(target=_boot_sequence, daemon=True).start()


def _install_loading_page() -> None:
    """Put the loading splash in the page cache so it's served at "/" until the
    real cache build (rebuild()) replaces it."""
    import graph_app
    with _state_cv:
        _state["html"] = graph_app.LOADING_PAGE
        _state["html_gz"] = b""                 # small; _send gzips on the fly
        _state_cv.notify_all()


# --- window-presence tracking (auto-exit when the app window closes) --------
# Each open app window holds one /api/events SSE stream, identified by a random
# per-window client id (cid). We track the set of live cids; when the last one
# goes away and none reconnects within IDLE_GRACE_SECS (a reload briefly drops
# to empty then reconnects with a fresh cid, so the grace avoids killing the
# server on reload), the server shuts itself down.
#
# A window leaves the set by EITHER path, whichever happens first:
#   - explicit: the page fires navigator.sendBeacon("/api/closing?cid=…") from
#     its `pagehide` handler the instant it's closed/navigated away — immediate
#     and reliable even when the OS keeps the TCP socket warm.
#   - implicit: the SSE stream's keepalive write fails (socket finally torn
#     down) and its handler's `finally` discards the cid. This is the fallback
#     for browsers without sendBeacon or a beacon lost in flight.
# Using a SET (not a counter) makes both paths idempotent — discarding an
# already-gone cid is a no-op, so a beacon followed by the socket teardown
# (or vice-versa) can't double-count.
#
# SSE_KEEPALIVE_SECS bounds how fast the implicit fallback notices a dead
# socket. IDLE_GRACE_SECS is generous enough that a page reload reconnects
# before the watchdog fires (killing the server mid-reload would leave the
# reloaded page unable to reconnect).
SSE_KEEPALIVE_SECS = 3
IDLE_GRACE_SECS = 10
_present_lock = threading.Lock()
_present_cids: set[str] = set()   # cids of open /api/events streams
_ever_connected = False           # an app window has connected at least once
_anon_seq = 0                     # fallback cid source for clients sending none


def _next_anon_cid() -> str:
    """A unique cid for an /api/events stream that arrived without one (older
    cached page, or a non-app client). Keeps each such stream counted
    independently so presence tracking still works."""
    global _anon_seq
    with _present_lock:
        _anon_seq += 1
        return f"anon-{_anon_seq}"


def _client_connected(cid: str) -> None:
    global _ever_connected
    with _present_lock:
        _present_cids.add(cid)
        _ever_connected = True


def _client_disconnected(cid: str) -> None:
    with _present_lock:
        _present_cids.discard(cid)


def _idle_watchdog(srv) -> None:
    """Shut the server down once every app window has been closed for
    IDLE_GRACE_SECS. Armed only after a window has connected, so it never fires
    during the loading splash."""
    import time
    idle = 0.0
    while True:
        time.sleep(1)
        with _present_lock:
            present, ever = len(_present_cids), _ever_connected
        if ever and present == 0:
            idle += 1
            if idle >= IDLE_GRACE_SECS:
                print("[serve] app window closed — shutting down")
                srv.shutdown()
                return
        else:
            idle = 0.0


def _server_already_up(port: int) -> bool:
    """Is a serve_app already answering on this loopback port? A quick TCP
    connect is enough — we don't need a full HTTP round-trip, just to know the
    port is actively served (not merely bindable). Robust against SO_REUSEADDR,
    which would otherwise let a second bind succeed silently."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _force_kill_port(port: int) -> None:
    """Force-terminate the process listening on `port`. Windows-first (netstat
    + taskkill); falls back to lsof + kill elsewhere. Best-effort — never
    raises. Used only when a graceful shutdown didn't free the port."""
    try:
        if sys.platform.startswith("win"):
            out = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                                 capture_output=True, text=True,
                                 timeout=5).stdout
            pids, needle = set(), f":{port}"
            for line in out.splitlines():
                parts = line.split()
                # LISTENING rows look like: TCP  127.0.0.1:8765  0.0.0.0:0  LISTENING  <pid>
                if (len(parts) >= 5 and parts[-2] == "LISTENING"
                        and parts[1].endswith(needle)):
                    pids.add(parts[-1])
            for pid in pids:
                if pid and pid != "0":
                    # No /T: don't take down child processes (e.g. a Neo4j
                    # console) — the fresh server reuses a DB that's still up.
                    subprocess.run(["taskkill", "/PID", pid, "/F"],
                                   capture_output=True, timeout=5)
        else:
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}"],
                                 capture_output=True, text=True,
                                 timeout=5).stdout
            for pid in out.split():
                subprocess.run(["kill", "-9", pid],
                               capture_output=True, timeout=5)
    except Exception as e:
        print(f"[serve] force-kill skipped: {e}")


def _kill_existing_server(port: int) -> None:
    """Ensure no previous serve_app is holding the port before we start. A fresh
    launch (double-clicking the mail icon) must always start a NEW server so
    code/template changes take effect — never hand off to a stale one. Try a
    graceful shutdown first (the same /api/shutdown the power button uses); if
    the port doesn't free, force-kill whatever is listening on it."""
    if not _server_already_up(port):
        return
    print(f"[serve] a previous server is running on :{port} — stopping it")
    try:                                # 1) graceful: ask it to shut down
        import urllib.request
        urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{port}/api/shutdown",
                                   data=b"", method="POST"),
            timeout=2)
    except Exception:
        pass
    for _ in range(20):                 # 2) wait up to ~5s for the port to free
        if not _server_already_up(port):
            print("[serve] previous server stopped")
            return
        time.sleep(0.25)
    _force_kill_port(port)              # 3) still up — force-kill it
    for _ in range(12):                 # wait up to ~3s for the kill to land
        if not _server_already_up(port):
            print("[serve] previous server force-killed")
            return
        time.sleep(0.25)
    print(f"[serve] WARNING: port :{port} still busy after kill attempts")


def main() -> int:
    force_utf8()

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--interval", type=int, default=600,
                    help="Background sync interval in seconds (default 600).")
    ap.add_argument("--no-open", action="store_true",
                    help="Don't open a browser on start.")
    ap.add_argument("--tab", action="store_true",
                    help="Open a normal browser tab instead of a standalone "
                         "app window.")
    ap.add_argument("--no-auto-exit", action="store_true",
                    help="Keep the server running after the app window is "
                         "closed (default: shut down with the window).")
    args = ap.parse_args()

    url = f"http://localhost:{args.port}/"

    def open_ui() -> None:
        if args.no_open:
            return
        (webbrowser.open if args.tab else _open_app_window)(url)

    # First thing: stop any previous server holding the port so this launch
    # always starts a FRESH server (picking up code/template changes) instead
    # of handing off to a stale one. We can't rely on the bind failing: on
    # Windows the socket defaults to SO_REUSEADDR, so a second
    # ThreadingHTTPServer would happily bind the SAME port — a zombie duplicate
    # that gets no clients and lingers forever. So we probe + kill first, and
    # also disable address reuse so a concurrent double-launch loses the bind
    # race instead of duplicating.
    _kill_existing_server(args.port)
    http.server.ThreadingHTTPServer.allow_reuse_address = False
    try:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port),
                                              Handler)
    except OSError:
        # Port still busy despite the kill attempt (a wedged process we
        # couldn't terminate). Don't silently hand off to it — report and exit.
        print(f"Port :{args.port} is still in use and could not be freed. "
              f"Close any stray server and try again.")
        return 1
    srv.daemon_threads = True
    global _srv
    _srv = srv                    # let /api/shutdown (the power button) stop us

    # Show the loading splash and open the window immediately; the slow work
    # (start Neo4j → load the graph → build the cache) runs in a background
    # thread and the splash reports progress via /api/boot until it's ready.
    global _sync_interval
    _sync_interval = args.interval
    # Always start on the loading page: a fresh launch must never flash a stale
    # page from a previous run (it would hide just-changed code until the rebuild
    # lands). The loading splash polls /api/boot and auto-reloads to the live
    # page the moment the background boot (Neo4j + rebuild) is ready.
    _install_loading_page()
    _start_boot()
    print(f"\nServing the mail app at {url}")
    print(f"Auto-syncs new mail every {args.interval}s. Ctrl+C to stop.\n")
    # Tie the server's lifetime to the app window: when the last window closes
    # its /api/events stream drops and the watchdog shuts the server down.
    if not args.no_auto_exit:
        threading.Thread(target=_idle_watchdog, args=(srv,),
                         daemon=True).start()
    open_ui()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.server_close()
        # Stop only a Neo4j console the app itself launched; a Windows service
        # is left running so the DB stays warm for the next launch.
        _shutdown_native_neo4j()
    return 0


if __name__ == "__main__":
    sys.exit(main())
