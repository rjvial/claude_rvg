"""Persistent long-term memory for Liam, scoped per assistant feature.

Liam has two independent functionalities and each keeps its OWN memory, so a
preference taught to one never bleeds into the other:

  * scope ``"ask"``     → ``data/ask_memory.json`` — how the user likes the
    graph-RAG Ask assistant to answer: the CRAFT of a good response (structure,
    logic/reasoning, breadth/scope, depth, tone, language, citing). It does NOT
    learn facts about the mailbox — Ask stores only ``style`` memories.
  * scope ``"compose"`` → ``data/compose_memory.json`` — how the user likes
    Liam to WRITE emails (greeting/sign-off, formality, language) plus facts
    about recipients worth reusing when drafting.

The ``"ask"`` filename is unchanged, so memories collected before the split
remain Ask's and Compose simply starts empty.

Within each scope there are two kinds of memory, both global to the user:

  * ``style`` — durable presentation/writing preferences. The set is small, so
    every style memory in the scope is injected every time.
  * ``fact``  — durable facts / topics / people / projects. These accumulate
    freely, so only the few most relevant to the current question/brief are
    retrieved (cosine over the same embedding model used for message retrieval)
    and injected.

Memories are added automatically by a background extraction pass after each
answer/draft (see serve_app ``_learn_from_turn`` / ``_learn_from_compose``)
and/or by the user via the Settings → Memory panel. Records carry ``source`` =
"auto" | "user". Each memory's embedding is cached inline as ``vec``
(normalized, same 'passage:' convention as embed_messages) and used for
near-duplicate detection; facts additionally use it for relevance retrieval.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

from _common import DATA_DIR

_SCOPES = ("ask", "compose")
_LOCK = threading.RLock()
_MAX_FACTS_INJECTED = 6
_DUP_SIM = 0.92          # cosine ≥ this between two facts ⇒ duplicate, skip
# Every style memory in a scope is injected into EVERY prompt, and auto-learn
# can add up to 3 per turn — unbounded, weeks of use accumulate dozens of
# (often mutually contradictory) rules that degrade answers. Cap the set;
# when it overflows, the oldest AUTO-learned rules are dropped first (rules
# the user typed in by hand are never auto-pruned).
_MAX_STYLE = 20


# ── store ──────────────────────────────────────────────────────────────────
def _file_for(scope: str):
    """The JSON file backing a scope. Unknown scopes fall back to 'ask' so a
    bad value can never escape the data dir or silently lose writes."""
    scope = scope if scope in _SCOPES else "ask"
    return DATA_DIR / f"{scope}_memory.json"


def _load(scope: str = "ask") -> dict:
    try:
        d = json.loads(_file_for(scope).read_text(encoding="utf-8"))
        if isinstance(d, dict) and isinstance(d.get("memories"), list):
            return d
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"memories": []}


def _save(scope: str, d: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    f = _file_for(scope)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, f)


# ── embedding (reuses the warmed graph-RAG model) ──────────────────────────
def _embed_passage(text: str) -> list[float] | None:
    try:
        from graph_rag import get_model
        return get_model().encode("passage: " + text.strip(),
                                  normalize_embeddings=True).tolist()
    except Exception:
        return None


def _cos(a, b) -> float:
    # vectors are L2-normalized ⇒ cosine == dot product
    return sum(x * y for x, y in zip(a, b)) if a and b else 0.0


def _public(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "vec"}


# ── public API ─────────────────────────────────────────────────────────────
def list_memories(scope: str = "ask") -> list[dict]:
    """All memories in a scope (newest last), without the embedding vectors."""
    with _LOCK:
        return [_public(m) for m in _load(scope)["memories"]]


def add(text: str, kind: str, source: str = "user",
        scope: str = "ask") -> dict | None:
    """Add a memory to a scope. Returns the new record, or None if blank or a
    duplicate (exact text match, or a near-duplicate by cosine within the same
    kind). Near-dup detection matters for 'style' too: thumbs-up reinforcement
    keeps proposing similar response-craft rules, and every style memory is
    injected on every prompt, so paraphrases must not accumulate."""
    text = (text or "").strip()
    if not text:
        return None
    kind = kind if kind in ("style", "fact") else "fact"
    vec = _embed_passage(text)        # all kinds get a vector for near-dup checks
    with _LOCK:
        d = _load(scope)
        low = text.lower()
        for m in d["memories"]:
            if (m.get("text") or "").strip().lower() == low:
                return None
            if (m.get("kind") == kind and vec
                    and m.get("vec") and _cos(vec, m["vec"]) >= _DUP_SIM):
                return None
        entry = {"id": uuid.uuid4().hex[:12], "text": text, "kind": kind,
                 "source": source,
                 "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if vec is not None:
            entry["vec"] = vec
        d["memories"].append(entry)
        # Enforce the style cap (see _MAX_STYLE): drop the oldest auto-learned
        # style rules until the scope fits. List order is insertion order.
        n_style = sum(1 for m in d["memories"] if m.get("kind") == "style")
        excess = n_style - _MAX_STYLE
        if excess > 0:
            dropped = 0
            kept = []
            for m in d["memories"]:
                if (dropped < excess and m.get("kind") == "style"
                        and m.get("source") == "auto" and m is not entry):
                    dropped += 1
                    continue
                kept.append(m)
            if dropped:
                d["memories"] = kept
        _save(scope, d)
        return _public(entry)


def delete(mem_id: str, scope: str = "ask") -> bool:
    with _LOCK:
        d = _load(scope)
        kept = [m for m in d["memories"] if m.get("id") != mem_id]
        if len(kept) == len(d["memories"]):
            return False
        d["memories"] = kept
        _save(scope, d)
        return True


def recall(question: str, k_facts: int = _MAX_FACTS_INJECTED,
           scope: str = "ask") -> dict:
    """Return {'style': [text…], 'facts': [text…]} for prompt injection: every
    style preference in the scope plus the k facts most relevant to
    `question`."""
    with _LOCK:
        mems = _load(scope)["memories"]
    style = [m["text"] for m in mems if m.get("kind") == "style"]
    facts = [m for m in mems if m.get("kind") == "fact"]
    if not facts:
        return {"style": style, "facts": []}
    try:
        from graph_rag import embed_query
        q = embed_query(question)
        ranked = sorted(facts, key=lambda m: _cos(q, m.get("vec")),
                        reverse=True)
        chosen = [m["text"] for m in ranked[:k_facts]]
    except Exception:
        chosen = [m["text"] for m in facts[-k_facts:]]   # fallback: most recent
    return {"style": style, "facts": chosen}


def format_block(question: str, scope: str = "ask") -> str:
    """The 'USER MEMORY' block injected into the prompt (empty if none)."""
    r = recall(question, scope=scope)
    if not r["style"] and not r["facts"]:
        return ""
    lines = ["USER MEMORY — learned about this user across past sessions. "
             "HONOR the preferences every time; use the context only when "
             "it is actually relevant (never force it in)."]
    if r["style"]:
        lines.append("\nPreferences:")
        lines += [f"- {t}" for t in r["style"]]
    if r["facts"]:
        lines.append("\nRelevant context:")
        lines += [f"- {t}" for t in r["facts"]]
    return "\n".join(lines) + "\n"
