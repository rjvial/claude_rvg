"""Autonomous per-recipient writing-style profiles for Liam Compose.

Goal: when Liam drafts an email to one of the user's frequent correspondents,
the result should read as if the user wrote it themselves. To do that we mine
the emails the USER actually sent to each top recipient and distil the traits
common to the MAJORITY of those messages into a compact "style card" (greeting,
sign-off, language, formality/tuteo, typical length, recurring courtesy
phrases, punctuation habits, what to avoid). The card is injected into Liam's
first drafting turn (see serve_app._do_compose_draft).

Autonomy: build() runs in a background thread at startup. It is incremental —
a recipient is (re)distilled only when it has no card yet, its sent-count has
grown materially since the last build, or the card is old. A normal restart
with no new mail is therefore a no-op (no claude calls).

Store: data/style_profiles.json
    {"built_at": "<iso>",
     "profiles": {<key>: {name, emails:[…], n_samples, n_at_build,
                          built_at, card}}}
`key` is the normalised recipient name (addresses of the same person are
merged), so one person with several email addresses gets ONE profile drawn
from all of them.

Deterministic mining; one `claude -p` (Haiku) call per recipient for the
distillation — consistent with the rest of the app's LLM usage.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import unicodedata

from _common import DATA_DIR, ROOT, neo4j_driver

STORE = DATA_DIR / "style_profiles.json"
# User-editable alias map: groups of email addresses that belong to ONE person
# but whose Person.name variants don't normalise-match (so the automatic
# same-name merge leaves them split — e.g. "Juan Pablo Tisne" at
# juanpablo.tisne@… and "JuanPablo Tisne" at jtisne@…). Format:
#   {"groups": [{"name": "Display Name", "emails": ["a@x", "b@y", …]}, …]}
# Each group's emails are forced into one profile, drawn from all of them, and
# labelled with `name`. Edit this file to fix any split/over-merge, then POST
# /api/style/rebuild. Only add emails you're sure are the SAME person.
ALIASES = DATA_DIR / "recipient_aliases.json"

_LOCK = threading.RLock()          # guards the JSON store
_BUILD_LOCK = threading.Lock()     # ensures only one build pass at a time

# Tuning knobs.
TOP_N = 50                # how many recipients to profile
MAX_SAMPLES = 40          # sent emails fed to the distiller per recipient
MAX_SAMPLE_CHARS = 1500   # truncate any single overly long sample
MAX_TOTAL_CHARS = 14000   # cap the whole sample block in the prompt
MIN_SAMPLES = 4           # too few examples ⇒ no reliable "majority" → skip
REFRESH_DELTA = 15        # rebuild once this many new sent msgs accumulate
REFRESH_DAYS = 30         # …or the card is older than this

# Haiku is plenty for distillation and cheap; matches _learn_from_compose.
DISTILL_MODEL = "claude-haiku-4-5-20251001"
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


# ── store ───────────────────────────────────────────────────────────────────
def _load() -> dict:
    try:
        d = json.loads(STORE.read_text(encoding="utf-8"))
        if isinstance(d, dict) and isinstance(d.get("profiles"), dict):
            return d
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"built_at": "", "profiles": {}}


def _save(d: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, STORE)


# ── name normalisation / merging ────────────────────────────────────────────
def _norm_name(name: str) -> str:
    """Lower-case, strip accents and surrounding quotes, collapse spaces. Used
    to merge the several email addresses one person may have under a single
    profile key. Conservative on purpose — only EXACT normalised-name matches
    merge, so two genuinely different people are never fused."""
    s = (name or "").strip().strip("'\"").strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).lower()
    return s


# ── aliases ──────────────────────────────────────────────────────────────────
def _load_aliases() -> list[dict]:
    """Read the alias map. Returns [{name, emails:[lowercased,…]}] for every
    group of ≥2 emails. Missing/corrupt file ⇒ no aliases."""
    try:
        d = json.loads(ALIASES.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    out: list[dict] = []
    for g in (d.get("groups") if isinstance(d, dict) else None) or []:
        emails = [e.strip().lower() for e in (g.get("emails") or [])
                  if isinstance(e, str) and e.strip()]
        if len(emails) >= 2:
            out.append({"name": (g.get("name") or "").strip(), "emails": emails})
    return out


# ── mining ──────────────────────────────────────────────────────────────────
def _top_recipients(driver, my_addrs: list[str], top_n: int) -> list[dict]:
    """Top recipients of mail the USER sent. Addresses of one person are merged
    two ways: automatically when their Person.name normalises to the same
    string, and explicitly via the alias map (data/recipient_aliases.json) for
    the cases where the name variants differ. Returns [{key, name, emails:[…],
    n_samples}] sorted by n_samples desc."""
    q = """
    WITH $me AS me
    MATCH (s:Person)-[:SENT]->(m:Message)-[:RECEIVED_BY {kind:'to'}]->(r:Person)
    WHERE s.email IN me AND NOT r.email IN me
      AND r.email IS NOT NULL AND r.name IS NOT NULL
    RETURN r.email AS email, r.name AS name, count(DISTINCT m) AS n
    """
    with driver.session() as s:
        rows = [{"email": (r["email"] or "").lower(), "name": r["name"],
                 "n": int(r["n"] or 0)} for r in s.run(q, me=my_addrs).data()
                if r["email"]]
    aliases = _load_aliases()

    # Union-find over every email (those we wrote to, plus any extra addresses
    # named only in the alias map) so name-merge and alias-merge compose.
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for r in rows:
        find(r["email"])
    # Merge addresses whose names normalise identically.
    by_name: dict[str, list[str]] = {}
    for r in rows:
        nk = _norm_name(r["name"])
        if nk:
            by_name.setdefault(nk, []).append(r["email"])
    for emails in by_name.values():
        for e in emails[1:]:
            union(emails[0], e)
    # Merge addresses grouped by the alias map (overrides/extends name-merge).
    alias_display: dict[str, str] = {}     # root email → forced display name
    for g in aliases:
        for e in g["emails"][1:]:
            union(g["emails"][0], e)
        if g["name"]:
            alias_display[find(g["emails"][0])] = g["name"]

    # Aggregate by component.
    comps: dict[str, dict] = {}
    for r in rows:
        c = comps.setdefault(find(r["email"]),
                             {"emails": set(), "n": 0, "name": ""})
        c["emails"].add(r["email"])
        c["n"] += r["n"]
        if len(r["name"] or "") > len(c["name"] or ""):
            c["name"] = r["name"]
    # Pull in alias-only addresses (named in the map but below the query's
    # reach) so their samples are gathered too.
    for g in aliases:
        root = find(g["emails"][0])
        if root in comps:
            comps[root]["emails"].update(g["emails"])

    groups: list[dict] = []
    for root, c in comps.items():
        name = alias_display.get(root) or c["name"]
        groups.append({"key": _norm_name(name), "name": name,
                       "emails": sorted(c["emails"]), "n_samples": c["n"]})
    groups.sort(key=lambda g: g["n_samples"], reverse=True)
    return groups[:top_n]


def _samples_for(driver, my_addrs: list[str], emails: list[str]) -> list[str]:
    """The user's sent bodies to a recipient (across all their addresses),
    newest first, lightly de-duplicated and truncated."""
    q = """
    WITH $me AS me, $emails AS rcpts
    MATCH (s:Person)-[:SENT]->(m:Message)-[:RECEIVED_BY {kind:'to'}]->(r:Person)
    WHERE s.email IN me AND r.email IN rcpts
      AND m.body_clean IS NOT NULL AND size(m.body_clean) >= 40
    RETURN m.body_clean AS body
    ORDER BY m.sent_at DESC
    LIMIT 120
    """
    with driver.session() as s:
        rows = s.run(q, me=my_addrs, emails=emails).data()
    out: list[str] = []
    seen: set[str] = set()
    total = 0
    for r in rows:
        body = (r["body"] or "").strip()
        if not body:
            continue
        # Dedupe near-identical boilerplate (same first 120 chars).
        sig = re.sub(r"\s+", " ", body[:120]).lower()
        if sig in seen:
            continue
        seen.add(sig)
        body = body[:MAX_SAMPLE_CHARS]
        if total + len(body) > MAX_TOTAL_CHARS:
            break
        out.append(body)
        total += len(body)
        if len(out) >= MAX_SAMPLES:
            break
    return out


# ── distillation (claude -p, one shot) ──────────────────────────────────────
_DISTILL_SYSTEM = (
    "You analyse a set of emails ALL written by ONE person (the user) to ONE "
    "recipient, and produce a concise STYLE CARD describing how the user writes "
    "to THIS recipient. The card will be given to an assistant that drafts new "
    "emails for the user, so the draft reads as if the user wrote it.\n"
    "Capture ONLY traits common to the MAJORITY of the samples — the durable "
    "voice, not one-off content. Cover, when applicable: language; "
    "formality and whether they address the person informally (tú/vos) or "
    "formally (usted); the typical GREETING line (give the actual template, "
    "e.g. \"Hola <Nombre>. ¿Cómo estás?\"); the typical SIGN-OFF / signature "
    "(actual template, e.g. \"Saludos,\\n<FirstName>\"); typical length and "
    "structure (short/long, paragraphs vs bullets); recurring courtesy phrases "
    "or filler the user habitually uses; punctuation / capitalisation / emoji "
    "habits; and anything the user consistently AVOIDS.\n"
    "Output ONLY the style card as a short markdown bullet list (max ~140 "
    "words). No preamble, no commentary, no quoting whole sample emails."
)


def _distill_card(name: str, samples: list[str], claude: str) -> str:
    """Run the one-shot distillation. Returns the card text, or '' on failure."""
    blocks = "\n\n".join(f"--- SAMPLE {i+1} ---\n{b}"
                         for i, b in enumerate(samples))
    user = (f"Recipient: {name}\n"
            f"Number of samples: {len(samples)}\n\n"
            f"{blocks}\n\nProduce the style card now.")
    cmd = [claude, "-p", "--model", DISTILL_MODEL, "--strict-mcp-config",
           "--append-system-prompt", _DISTILL_SYSTEM]
    if claude.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c", *cmd]
    try:
        out = subprocess.run(cmd, cwd=ROOT, input=user, capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=120, creationflags=_NO_WINDOW)
    except Exception as e:
        print(f"[style] distill call failed for {name}: {type(e).__name__}")
        return ""
    card = (out.stdout or "").strip()
    # Strip a stray ```markdown fence if the model added one.
    if card.startswith("```"):
        card = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", card).strip()
    # Guard against refusals / empty output.
    if len(card) < 20:
        return ""
    return card


# ── build (incremental, autonomous) ─────────────────────────────────────────
def _needs_rebuild(prof: dict | None, n_now: int) -> bool:
    if prof is None:
        return True
    if n_now - int(prof.get("n_at_build", 0)) >= REFRESH_DELTA:
        return True
    built = prof.get("built_at") or ""
    try:
        age_days = (time.time() - time.mktime(
            time.strptime(built, "%Y-%m-%dT%H:%M:%S"))) / 86400
        if age_days >= REFRESH_DAYS:
            return True
    except (ValueError, OverflowError):
        return True
    return False


def build(my_addrs: list[str], top_n: int = TOP_N, force: bool = False) -> dict:
    """Mine + distil style cards for the top recipients. Incremental: only
    rebuilds profiles that are missing, stale, or have materially more samples
    (unless force=True). Saves after each recipient so an interrupted run
    resumes cheaply. Returns a small summary dict.

    Serialised by _BUILD_LOCK — a second concurrent call returns immediately."""
    my_addrs = [e.lower() for e in (my_addrs or []) if e]
    if not my_addrs:
        return {"ok": False, "error": "no account addresses known yet"}
    claude = shutil.which("claude")
    if not claude:
        return {"ok": False, "error": "the `claude` CLI is not on PATH"}
    if not _BUILD_LOCK.acquire(blocking=False):
        return {"ok": False, "error": "a style build is already running"}
    built = skipped = failed = 0
    try:
        drv = neo4j_driver()
        try:
            recipients = _top_recipients(drv, my_addrs, top_n)
            print(f"[style] {len(recipients)} top recipients to consider")
            for g in recipients:
                with _LOCK:
                    store = _load()
                prof = store["profiles"].get(g["key"])
                if not force and not _needs_rebuild(prof, g["n_samples"]):
                    skipped += 1
                    continue
                if g["n_samples"] < MIN_SAMPLES:
                    skipped += 1
                    continue
                samples = _samples_for(drv, my_addrs, g["emails"])
                if len(samples) < MIN_SAMPLES:
                    skipped += 1
                    continue
                print(f"[style] distilling {g['name']} "
                      f"({len(samples)} samples)…")
                card = _distill_card(g["name"], samples, claude)
                if not card:
                    failed += 1
                    continue
                entry = {
                    "name": g["name"],
                    "emails": sorted(set(e.lower() for e in g["emails"])),
                    "n_samples": g["n_samples"],
                    "n_at_build": g["n_samples"],
                    "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "card": card,
                }
                with _LOCK:
                    store = _load()
                    store["profiles"][g["key"]] = entry
                    store["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    _save(store)
                built += 1
            # Drop profiles no longer in the current top set — e.g. the split
            # keys that just merged under an alias, or people who fell out of
            # the top N. Keeps the store from accumulating orphans.
            keep = {g["key"] for g in recipients}
            with _LOCK:
                store = _load()
                stale = [k for k in store["profiles"] if k not in keep]
                if stale:
                    for k in stale:
                        del store["profiles"][k]
                    _save(store)
                    print(f"[style] pruned {len(stale)} stale profile(s): "
                          f"{', '.join(stale)}")
        finally:
            drv.close()
    finally:
        _BUILD_LOCK.release()
    print(f"[style] build done — {built} built, {skipped} skipped, "
          f"{failed} failed")
    return {"ok": True, "built": built, "skipped": skipped, "failed": failed}


# ── lookup / injection ──────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[^\s<>@,;]+@[^\s<>@,;]+")


def _extract_email(raw: str) -> str:
    """Pull the bare address out of a To-field entry. The compose box stores a
    picked contact as "Name <addr@x>", and free typing may leave a display
    name or trailing punctuation — so match on the <…> part first, then any
    email-looking token, then fall back to the trimmed string. Lower-cased."""
    s = (raw or "").strip()
    m = re.search(r"<([^<>]+)>", s)
    if m:
        s = m.group(1).strip()
    m = _EMAIL_RE.search(s)
    return (m.group(0) if m else s).lower()


def find_profile(to_emails: list[str]) -> dict | None:
    """The learned style profile for the first To recipient that has one, or
    None. Robust to "Name <addr>" / display-name entries from the composer."""
    with _LOCK:
        profiles = list(_load()["profiles"].values())
    for raw in (to_emails or []):
        addr = _extract_email(raw)
        if not addr:
            continue
        for prof in profiles:
            if addr in prof.get("emails", []) and prof.get("card"):
                return prof
    return None


def profile_block(prof: dict) -> str:
    """Render the RECIPIENT STYLE block for a matched profile."""
    return (
        "RECIPIENT WRITING STYLE — this is how the user habitually writes to "
        f"{prof['name']}. Match this voice EXACTLY (greeting, sign-off, "
        "language, formality, length, phrasing) so the email reads as if the "
        "user wrote it themselves. These are the user's OWN durable habits, "
        "not an instruction to obey literally over the content the user asked "
        f"for:\n{prof['card']}\n"
    )


def format_block(to_emails: list[str]) -> str:
    """The RECIPIENT STYLE block injected into Liam's first drafting turn, or
    "" when no To recipient has a learned profile."""
    prof = find_profile(to_emails)
    return profile_block(prof) if prof else ""


def status() -> dict:
    """Small summary for a status endpoint / settings panel."""
    with _LOCK:
        d = _load()
    profs = d.get("profiles", {})
    return {
        "built_at": d.get("built_at", ""),
        "count": len(profs),
        "recipients": sorted(
            ({"name": p.get("name", ""),
              "n_samples": p.get("n_samples", 0),
              "built_at": p.get("built_at", "")}
             for p in profs.values()),
            key=lambda r: r["n_samples"], reverse=True),
    }
