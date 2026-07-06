"""Graph-RAG retrieval core for the /api/ask endpoint.

The unit of retrieval is the CONVERSATION, not the message. An answer about a
matter's status or history needs its whole arc — not a handful of
keyword-matched replies — so:

  1. Hybrid — embed the question and vector-search the message_embedding
     index, AND keyword-search the message_text full-text index; when the
     question names a known Person/Org, a third leg pulls that entity's own
     mail through the identity layer (ALIAS_OF clusters, WORKS_AT). The
     candidate lists are fused by reciprocal rank (exact names/identifiers
     that don't embed near the question still surface). Candidates are
     grouped by conversation (Matter if the thread has one, else Thread),
     and the conversations are ranked by their best-matching messages.
     Anchored entities also get an ENTITY CARD — a deterministic profile
     (addresses, org, per-year counts, WITH/ABOUT split) prepended to the
     bundle so spans and counts come from the graph, not the LLM.
  2. Conversation-complete — the top few conversations are pulled WHOLE:
     every message, in chronological order, with its body, sender, Orgs and
     attachment names.

A character budget keeps the bundle bounded: a conversation that fits is
shown in full; in a larger one the messages that matched the question keep
full bodies while the rest are trimmed to a gist; an enormous one is windowed
to its most relevant messages — so the whole arc still fits the model's
context, with depth where the query actually landed.

build_context() renders the result into a numbered, citable block grouped by
conversation, plus a parallel `sources` list mapping each [n] to a working
Gmail link. The /api/ask answer step feeds the block to the LLM, which also
has the read-only Neo4j Cypher tools for structural follow-ups.

The embedding model is loaded lazily and cached process-wide (get_model());
serve_app.py warms it once at startup so /api/ask stays fast.
"""
from __future__ import annotations

import re
import sys
import threading
import unicodedata
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# The Gmail-link builder the app's message panel uses — the raw m.gmail_url
# is a form Gmail doesn't open; gmail_search_url rewrites it.
from _common import gmail_search_url  # noqa: E402
# Single source of truth for the model name / dimension (see embed_messages).
from embed_messages import MODEL_NAME, hf_offline_if_cached  # noqa: E402

VECTOR_INDEX = "message_embedding"
CANDIDATE_POOL = 150         # vector hits scanned to locate relevant conversations
FULLTEXT_POOL = 50           # keyword hits fused in alongside the vector pool
RRF_K = 60                   # reciprocal-rank-fusion damping constant
MAX_CONVERSATIONS = 10       # conversations considered for the bundle
CONV_SCORE_TOP = 3           # extra hits (beyond the best) that nudge the score
TOTAL_BUDGET = 120_000       # total body chars across the whole bundle
CONV_BUDGET = 14_000         # body chars allotted to one conversation
BODY_CHARS = 4000            # per-message body cap when not budget-constrained
MIN_BODY = 240               # per-message body floor for a large conversation

_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_model():
    """Lazily load and cache the SentenceTransformer. First call downloads
    the model (~2GB) and takes a few seconds; later calls are instant. The
    lock makes a background warm-up and a concurrent request safe — only one
    thread loads, the rest wait for it."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                hf_offline_if_cached()
                from sentence_transformers import SentenceTransformer
                _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def embed_query(text: str) -> list[float]:
    """Embed a search query. e5 wants a 'query:' prefix (documents got
    'passage:' in embed_messages.py); both must use the same model."""
    vec = get_model().encode("query: " + (text or "").strip(),
                             normalize_embeddings=True)
    return vec.tolist()


# Stage 1 — vector search. Returns a pool of candidate messages, each tagged
# with the element ids of its Thread and Matter so the Python side can group
# candidates into conversations and rank them.
CANDIDATES_CYPHER = """
CALL db.index.vector.queryNodes($index, $candidates, $qvec)
     YIELD node AS m, score
OPTIONAL MATCH (m)-[:IN_THREAD]->(t:Thread)
OPTIONAL MATCH (t)-[:PART_OF]->(mt:Matter)
RETURN m.gmail_message_id AS mid, m.account_owner AS acct, score,
       elementId(mt) AS matter_eid, elementId(t) AS thread_eid
ORDER BY score DESC
"""

# Keyword leg of the hybrid retriever: the accent-folding full-text index
# catches exact tokens (names, identifiers, invoice numbers) that don't embed
# close to the question. Filtered to primary like the vector leg (only primary
# mail is embedded, so the bundle contract stays: lite never appears).
FULLTEXT_CANDIDATES_CYPHER = """
CALL db.index.fulltext.queryNodes('message_text', $q)
     YIELD node AS m, score
WHERE coalesce(m.bucket, 'primary') = 'primary'
OPTIONAL MATCH (m)-[:IN_THREAD]->(t:Thread)
OPTIONAL MATCH (t)-[:PART_OF]->(mt:Matter)
RETURN m.gmail_message_id AS mid, m.account_owner AS acct, score,
       elementId(mt) AS matter_eid, elementId(t) AS thread_eid
ORDER BY score DESC
LIMIT $k
"""


def _lucene_escape(term: str) -> str:
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)', r"\\\1", term)


def _lucene_query(question: str, max_terms: int = 12) -> str:
    """OR-of-escaped-tokens Lucene query from the question (tokens ≥3 chars,
    deduped, capped). Lucene's idf keeps common words from dominating."""
    toks: list[str] = []
    for t in re.findall(r"\w{3,}", (question or "").lower()):
        if t not in toks:
            toks.append(t)
    return " OR ".join(_lucene_escape(t) for t in toks[:max_terms])

# ---------------------------------------------------------------------------
# Entity anchoring — a third retrieval leg plus the ENTITY CARDS block.
#
# The graph carries a deterministic identity layer (ALIAS_OF merges of one
# human's addresses, WORKS_AT from email domains) that text retrieval alone
# cannot see: mail where a person is only a header participant never mentions
# their name in the body, and one human is often several addresses. When the
# question NAMES a person or organisation we therefore
#   (a) feed that entity's mail into the candidate pool as an extra RRF leg
#       (recency-ranked — the vector/full-text legs already cover topical
#       relevance), and
#   (b) render an ENTITY CARD of deterministic facts — every address, org,
#       primary-mail count with first/last dates, per-year histogram, and the
#       WITH/ABOUT split — so the answer model narrates from grounded numbers
#       instead of being trusted to aggregate them itself over Cypher.
ENTITY_POOL = 40             # entity-leg messages fed into the RRF pool
MAX_PERSON_ANCHORS = 3       # persons profiled per question
MAX_ORG_ANCHORS = 2          # orgs profiled per question
OWNER_SHARE_CAP = 0.25       # drop a "person" on >25% of all mail (the user)
FREEMAIL_DOMAINS = {"gmail.com", "hotmail.com", "outlook.com", "yahoo.com",
                    "icloud.com", "live.com", "msn.com"}

PERSON_HITS_CYPHER = """
CALL db.index.fulltext.queryNodes('person_name', $q) YIELD node, score
RETURN node.email AS email, node.name AS name, score
ORDER BY score DESC LIMIT 12
"""

# The full alias cluster of one address: hop to the canonical Person (if this
# address is an alias), then collect the canonical plus every alias of it.
ALIAS_CLUSTER_CYPHER = """
MATCH (p:Person {email: $email})
OPTIONAL MATCH (p)-[:ALIAS_OF]->(c:Person)
WITH coalesce(c, p) AS canon
OPTIONAL MATCH (a:Person)-[:ALIAS_OF]->(canon)
RETURN canon.email AS email, canon.name AS name,
       [x IN collect(DISTINCT a) | {email: x.email, name: x.name}] AS aliases
"""

TOTAL_PRIMARY_CYPHER = """
MATCH (m:Message) WHERE coalesce(m.bucket, 'primary') = 'primary'
RETURN count(m) AS n
"""

PERSON_STATS_CYPHER = """
MATCH (p:Person) WHERE p.email IN $emails
MATCH (p)-[:SENT|RECEIVED_BY]-(m:Message)
WHERE coalesce(m.bucket, 'primary') = 'primary'
WITH DISTINCT m
RETURN count(m) AS total, min(m.sent_at) AS first, max(m.sent_at) AS last
"""

PERSON_HIST_CYPHER = """
MATCH (p:Person) WHERE p.email IN $emails
MATCH (p)-[:SENT|RECEIVED_BY]-(m:Message)
WHERE coalesce(m.bucket, 'primary') = 'primary'
WITH DISTINCT m
WITH substring(m.sent_at, 0, 4) AS yr
WHERE yr IS NOT NULL AND yr <> ''
RETURN yr, count(*) AS n ORDER BY yr
"""

PERSON_ORGS_CYPHER = """
MATCH (p:Person) WHERE p.email IN $emails
MATCH (p)-[:WORKS_AT]->(o:Org)
RETURN DISTINCT o.canonical_name AS name, o.domain AS domain
"""

# ABOUT mail: the name appears in a body/subject but the person is NOT a
# participant — intros before they appear, decisions after they leave. These
# have no graph edge to the person, so only full-text can find them.
PERSON_ABOUT_CYPHER = """
CALL db.index.fulltext.queryNodes('message_text', $tok) YIELD node AS m
WHERE coalesce(m.bucket, 'primary') = 'primary'
  AND NOT EXISTS { MATCH (m)-[:SENT|RECEIVED_BY]-(p:Person)
                   WHERE p.email IN $emails }
RETURN count(m) AS n, min(m.sent_at) AS first, max(m.sent_at) AS last
"""

PERSON_CANDIDATES_CYPHER = """
MATCH (p:Person) WHERE p.email IN $emails
MATCH (p)-[:SENT|RECEIVED_BY]-(m:Message)
WHERE coalesce(m.bucket, 'primary') = 'primary'
WITH DISTINCT m
ORDER BY m.sent_at DESC
LIMIT $k
OPTIONAL MATCH (m)-[:IN_THREAD]->(t:Thread)
OPTIONAL MATCH (t)-[:PART_OF]->(mt:Matter)
RETURN m.gmail_message_id AS mid, m.account_owner AS acct,
       elementId(mt) AS matter_eid, elementId(t) AS thread_eid
"""

ORG_LIST_CYPHER = """
MATCH (o:Org)
RETURN o.canonical_name AS name, o.domain AS domain, o.aliases AS aliases
"""

ORG_STATS_CYPHER = """
MATCH (:Org {canonical_name: $name})<-[:WORKS_AT]-(p:Person)
MATCH (p)-[:SENT|RECEIVED_BY]-(m:Message)
WHERE coalesce(m.bucket, 'primary') = 'primary'
WITH DISTINCT m
RETURN count(m) AS total, min(m.sent_at) AS first, max(m.sent_at) AS last
"""

ORG_HIST_CYPHER = """
MATCH (:Org {canonical_name: $name})<-[:WORKS_AT]-(p:Person)
MATCH (p)-[:SENT|RECEIVED_BY]-(m:Message)
WHERE coalesce(m.bucket, 'primary') = 'primary'
WITH DISTINCT m
WITH substring(m.sent_at, 0, 4) AS yr
WHERE yr IS NOT NULL AND yr <> ''
RETURN yr, count(*) AS n ORDER BY yr
"""

ORG_PEOPLE_CYPHER = """
MATCH (:Org {canonical_name: $name})<-[:WORKS_AT]-(p:Person)
MATCH (p)-[:SENT|RECEIVED_BY]-(m:Message)
WHERE coalesce(m.bucket, 'primary') = 'primary'
WITH p, count(DISTINCT m) AS n ORDER BY n DESC LIMIT 5
RETURN p.name AS name, p.email AS email, n
"""

ORG_CANDIDATES_CYPHER = """
MATCH (:Org {canonical_name: $name})<-[:WORKS_AT]-(p:Person)
MATCH (p)-[:SENT|RECEIVED_BY]-(m:Message)
WHERE coalesce(m.bucket, 'primary') = 'primary'
WITH DISTINCT m
ORDER BY m.sent_at DESC
LIMIT $k
OPTIONAL MATCH (m)-[:IN_THREAD]->(t:Thread)
OPTIONAL MATCH (t)-[:PART_OF]->(mt:Matter)
RETURN m.gmail_message_id AS mid, m.account_owner AS acct,
       elementId(mt) AS matter_eid, elementId(t) AS thread_eid
"""


def _fold(s: str) -> str:
    """Accent-strip + lowercase, mirroring the accent folding of the
    person_name / message_text indexes ('Muñoz' → 'munoz')."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(ch)).lower()


def _about_token(name: str) -> str | None:
    """The token for the ABOUT search: the most distinctive name token,
    with length as the distinctiveness proxy ('Quinteros' over 'Martin').
    Position is meaningless — names come as both 'First Last' and
    'last, first'. A lone first name would match half the mailbox, so
    require ≥2 name tokens and a ≥4-char pick."""
    toks = re.findall(r"\w{3,}", name or "")
    if len(toks) < 2:
        return None
    longest = max(toks, key=len)
    return longest if len(longest) >= 4 else None


def _person_anchors(_read, question: str) -> list[dict]:
    """Persons the question NAMES, resolved deterministically: full-text hits
    on person_name whose matched name token literally appears in the question
    (accent-folded), expanded to their whole ALIAS_OF cluster and profiled.
    The user themself is excluded via OWNER_SHARE_CAP — anchoring on someone
    who is on a quarter of the corpus retrieves nothing meaningful."""
    qtokens = {_fold(t) for t in re.findall(r"\w{3,}", question or "")}
    lucene = _lucene_query(question)
    if not lucene or not qtokens:
        return []
    try:
        hits = _read(PERSON_HITS_CYPHER, q=lucene)
    except Exception:
        return []                    # person_name index missing — no anchors
    # Keep only the hits whose name matches the MOST question tokens: a
    # "Martin Quinteros" question (2 tokens matched) must not also anchor
    # every other Martin (1), while a surname-only question ("correos de
    # Vial", max 1) still anchors every Vial.
    scored: list[tuple[int, dict]] = []
    for h in hits:
        ntoks = {_fold(t) for t in re.findall(r"\w{3,}", h.get("name") or "")}
        nmatch = len(ntoks & qtokens)
        if nmatch and h.get("email"):
            scored.append((nmatch, h))
    if not scored:
        return []                    # Lucene fuzz — no name in the question
    best_nmatch = max(n for n, _ in scored)

    total_primary = None
    anchors: list[dict] = []
    taken: set[str] = set()          # emails already claimed by a cluster
    for nmatch, h in scored:
        if len(anchors) >= MAX_PERSON_ANCHORS:
            break
        name, email = h.get("name") or "", h["email"]
        if nmatch < best_nmatch or email in taken:
            continue
        cl = _read(ALIAS_CLUSTER_CYPHER, email=email)
        if not cl:
            continue
        canon = cl[0]
        addresses = ([{"email": canon["email"], "name": canon["name"]}]
                     + [a for a in (canon["aliases"] or []) if a.get("email")])
        emails = [a["email"] for a in addresses]
        if taken.intersection(emails):
            continue                 # same human, already profiled
        taken.update(emails)
        stats = _read(PERSON_STATS_CYPHER, emails=emails)[0]
        if total_primary is None:
            total_primary = _read(TOTAL_PRIMARY_CYPHER)[0]["n"]
        if total_primary and stats["total"] > OWNER_SHARE_CAP * total_primary:
            continue                 # the mailbox owner / an on-everything hub
        about, tok = None, _about_token(canon["name"] or name)
        if tok:
            try:
                ab = _read(PERSON_ABOUT_CYPHER, tok=_lucene_escape(tok),
                           emails=emails)[0]
                about = {**ab, "token": tok}
            except Exception:
                about = None
        anchors.append({
            "kind": "person",
            "name": canon["name"] or name or canon["email"],
            "emails": emails,
            "addresses": addresses,
            "orgs": _read(PERSON_ORGS_CYPHER, emails=emails),
            "hist": _read(PERSON_HIST_CYPHER, emails=emails),
            "about": about,
            **stats,
        })
    return anchors


def _org_anchors(_read, question: str) -> list[dict]:
    """Orgs the question NAMES: an Org node whose canonical name or alias
    appears (accent-folded, word-bounded) in the question. Freemail-domain
    orgs are never anchors — 'gmail' in a question is about the mailbox, not
    an organisation."""
    qfold = _fold(question)
    if not qfold:
        return []
    matched: list[tuple[int, dict]] = []
    for o in _read(ORG_LIST_CYPHER):
        if (o.get("domain") or "").lower() in FREEMAIL_DOMAINS:
            continue
        best = 0
        for nm in [o.get("name")] + list(o.get("aliases") or []):
            f = _fold(nm or "")
            if (len(f) >= 4 and len(f) > best and
                    re.search(r"(?<!\w)" + re.escape(f) + r"(?!\w)", qfold)):
                best = len(f)
        if best:
            matched.append((best, o))
    matched.sort(key=lambda x: -x[0])          # longest match = most specific
    anchors: list[dict] = []
    for _, o in matched[:MAX_ORG_ANCHORS]:
        stats = _read(ORG_STATS_CYPHER, name=o["name"])[0]
        anchors.append({
            "kind": "org",
            "name": o["name"],
            "domain": o.get("domain"),
            "aliases": list(o.get("aliases") or []),
            "people": _read(ORG_PEOPLE_CYPHER, name=o["name"]),
            "hist": _read(ORG_HIST_CYPHER, name=o["name"]),
            **stats,
        })
    return anchors


def resolve_anchors(session, question: str) -> list[dict]:
    """Detect the Person / Org entities the question names and profile them
    from the graph. Returns anchor dicts consumed by retrieve() (the entity
    retrieval leg) and entity_cards() (the card block). Deterministic and
    fail-soft: any error just means no anchors — the vector and full-text
    legs still retrieve."""
    # First param deliberately not named `q` — see retrieve()'s _read.
    def _read(cypher, **params):
        return session.execute_read(
            lambda tx: tx.run(cypher, **params).data())

    anchors: list[dict] = []
    for part in (_person_anchors, _org_anchors):
        try:
            anchors += part(_read, question)
        except Exception:
            pass
    return anchors


def _span_line(n, first, last) -> str:
    if not n:
        return "none found"
    return f"{n} message(s), {(first or '?')[:10]} → {(last or '?')[:10]}"


def _hist_line(hist: list[dict]) -> str:
    return (" · ".join(f"{h['yr']}:{h['n']}" for h in hist)
            + "  (unlisted years: 0)")


def entity_cards(anchors: list[dict]) -> str:
    """Render resolve_anchors() output as the ENTITY CARDS block that opens
    the context bundle ('' when nothing was detected). Pure formatting —
    every number was computed deterministically in resolve_anchors(), so the
    answer model is told to treat it as ground truth."""
    if not anchors:
        return ""
    out = ["=== ENTITY CARDS — deterministic profiles of the people/orgs "
           "named in the question, computed from the graph's identity layer "
           "(ALIAS_OF, WORKS_AT) and its indexes. Counts cover PRIMARY mail "
           "only. Trust these numbers for spans, counts and first/last "
           "appearances; use Cypher to fetch the messages behind them, not "
           "to recompute them. ==="]
    for a in anchors:
        lines: list[str] = []
        if a["kind"] == "person":
            lines.append(f"• PERSON: {a['name']}")
            lines.append("  Addresses (same human, ALIAS_OF-merged): "
                         + "; ".join(
                             d["email"]
                             + (f" ({d['name']})" if d.get("name") else "")
                             for d in a["addresses"]))
            if a["orgs"]:
                lines.append("  Works at: " + ", ".join(
                    o["name"]
                    + (f" ({o['domain']})" if o.get("domain") else "")
                    for o in a["orgs"]))
            lines.append("  Mail WITH them (they are a participant): "
                         + _span_line(a["total"], a["first"], a["last"]))
            if a["hist"]:
                lines.append("  Per year: " + _hist_line(a["hist"]))
            ab = a.get("about")
            if ab is None:
                lines.append("  Mail ABOUT them: not computed (no "
                             "distinctive surname) — full-text-search the "
                             "name if the arc matters.")
            else:
                lines.append(f"  Mail ABOUT them (body mentions "
                             f"'{ab['token']}' but they are NOT a "
                             "participant): "
                             + _span_line(ab["n"], ab["first"], ab["last"]))
        else:
            head = f"• ORG: {a['name']}"
            if a.get("domain"):
                head += f" — domain {a['domain']}"
            lines.append(head)
            if a.get("aliases"):
                lines.append("  Aliases: " + ", ".join(a["aliases"]))
            if a["people"]:
                lines.append("  Most-seen people there: " + "; ".join(
                    ((f"{p['name']} <{p['email']}>" if p.get("name")
                      else p["email"]) + f" ({p['n']})")
                    for p in a["people"]))
            lines.append("  Mail with their people: "
                         + _span_line(a["total"], a["first"], a["last"]))
            if a["hist"]:
                lines.append("  Per year: " + _hist_line(a["hist"]))
        out.append("\n".join(lines))
    return "\n\n".join(out)


# Stage 2 — pull a whole conversation. Three anchors (Matter / Thread / lone
# Message) share the same expansion tail: sender + Orgs + attachments per
# message, every message of the conversation, chronological.
#
# "Orgs" per message are derived from the sender's WORKS_AT edge (Layer 2,
# deterministic email-domain mapping via orgs_seed.json) — there is no
# Message-MENTIONS-Org edge anymore (LLM entity extraction was removed
# 2026-05-25; Concept MENTIONS was removed 2026-05-26).
_EXPAND_TAIL = """
CALL (m) {
  OPTIONAL MATCH (s:Person)-[:SENT]->(m)
  RETURN s.email AS from_email, s.name AS from_name
}
CALL (m) {
  OPTIONAL MATCH (sender:Person)-[:SENT]->(m)
  OPTIONAL MATCH (sender)-[:WORKS_AT]->(o:Org)
  RETURN collect(DISTINCT o.canonical_name) AS orgs
}
CALL (m) {
  OPTIONAL MATCH (m)-[:HAS_ATTACHMENT]->(a:Attachment)
  RETURN collect(DISTINCT a.filename) AS attachments
}
RETURN m.gmail_message_id AS mid, m.account_owner AS acct,
       m.subject AS subject, m.sent_at AS sent_at, m.body_clean AS body,
       m.snippet AS snippet, m.gmail_url AS gmail_url,
       m.rfc822_message_id AS rfc822,
       from_email, from_name, orgs, attachments, conv_label
ORDER BY m.sent_at ASC, m.gmail_message_id ASC
"""

MATTER_MESSAGES_CYPHER = """
MATCH (mt:Matter) WHERE elementId(mt) = $eid
MATCH (mt)<-[:PART_OF]-(:Thread)<-[:IN_THREAD]-(m:Message)
WITH DISTINCT m, mt.canonical_subject AS conv_label
""" + _EXPAND_TAIL

THREAD_MESSAGES_CYPHER = """
MATCH (t:Thread) WHERE elementId(t) = $eid
MATCH (t)<-[:IN_THREAD]-(m:Message)
WITH m, t.subject AS conv_label
""" + _EXPAND_TAIL

MESSAGE_CYPHER = """
MATCH (m:Message {gmail_message_id: $mid, account_owner: $acct})
WITH m, m.subject AS conv_label
""" + _EXPAND_TAIL


def _conv_score(scores: list[float]) -> float:
    """A conversation ranks by its single BEST hit, plus a small bonus for
    having a few more strong hits.

    Summing the top-N (the old rule) rewarded raw size: a 285-message thread
    has far more surface area to land hits than a 3-message one, so big
    conversations floated to the top regardless of how *relevant* they were,
    and — pulled whole — ate the budget. Anchoring on the best hit makes
    relevance density the primary signal; the dampened bonus (mean of the next
    few hits, weighted 0.15) still nudges a conversation with several genuine
    matches above one with a lone fluke, without letting size dominate."""
    s = sorted(scores, reverse=True)
    best = s[0] if s else 0.0
    extra = s[1:CONV_SCORE_TOP]
    bonus = (sum(extra) / len(extra)) if extra else 0.0
    return best + 0.15 * bonus


def retrieve(session, question: str, k: int = MAX_CONVERSATIONS,
             anchors: list[dict] | None = None) -> list[dict]:
    """Graph-RAG retrieval. Fuses three candidate legs — vector, full-text,
    and (when the question names a known person/org) that entity's own mail —
    groups the hits into conversations, ranks them, and pulls the top k
    conversations WHOLE. Returns a list of conversation dicts (best first):
        {kind, label, score, messages: [<every message, chronological>]}
    Each message carries `hit` / `score` flagging whether it matched the
    query. build_context() turns this into the prompt bundle.

    `anchors` is resolve_anchors() output; pass it when the caller already
    resolved anchors (serve_app does, to also render entity cards), or leave
    None to resolve here."""
    # execute_read = managed transactions with driver-level retry on
    # transient failures, vs bare session.run which fails them through.
    # NB the first param must not be named `q` — several queries take a
    # $q Cypher parameter, and the keyword collision raises TypeError.
    def _read(cypher, **params):
        return session.execute_read(
            lambda tx: tx.run(cypher, **params).data())

    qvec = embed_query(question)
    vec_rows = _read(CANDIDATES_CYPHER, index=VECTOR_INDEX,
                     candidates=CANDIDATE_POOL, qvec=qvec)
    ft_rows: list[dict] = []
    lucene = _lucene_query(question)
    if lucene:
        try:
            ft_rows = _read(FULLTEXT_CANDIDATES_CYPHER,
                            q=lucene, k=FULLTEXT_POOL)
        except Exception:
            ft_rows = []          # index missing / bad query — vector-only

    # Reciprocal-rank fusion: cosine and Lucene scores live on different
    # scales, so fuse by RANK. A message found by both legs outranks one
    # found by either alone; a keyword-only hit (exact name/identifier the
    # embedding missed) still enters the pool.
    fused: dict[tuple, dict] = {}

    def _absorb(rows: list[dict]) -> None:
        for rank, c in enumerate(rows):
            key = (c["mid"], c["acct"])
            e = fused.get(key)
            if e is None:
                e = dict(c)
                e["score"] = 0.0
                fused[key] = e
            e["score"] += 1.0 / (RRF_K + rank + 1)

    _absorb(vec_rows)
    _absorb(ft_rows)

    # Entity leg: mail linked to the persons/orgs the question names, via the
    # graph's identity layer (ALIAS_OF / WORKS_AT). Recency-ranked — the
    # other two legs already rank by topical relevance; this one guarantees
    # the entity's own conversations enter the pool even when the name never
    # appears in a body (header-only participation embeds/searches as nothing).
    if anchors is None:
        anchors = resolve_anchors(session, question)
    for a in anchors:
        try:
            if a["kind"] == "person":
                rows = _read(PERSON_CANDIDATES_CYPHER,
                             emails=a["emails"], k=ENTITY_POOL)
            else:
                rows = _read(ORG_CANDIDATES_CYPHER,
                             name=a["name"], k=ENTITY_POOL)
        except Exception:
            rows = []
        _absorb(rows)

    candidates = sorted(fused.values(), key=lambda c: -c["score"])
    if not candidates:
        return []

    # Group candidates into conversations (Matter > Thread > lone message).
    convs: dict[str, dict] = {}
    for c in candidates:
        meid, teid = c.get("matter_eid"), c.get("thread_eid")
        if meid:
            gkey, kind = meid, "matter"
        elif teid:
            gkey, kind = teid, "thread"
        else:
            gkey, kind = c["mid"], "message"
        g = convs.setdefault(gkey, {"kind": kind, "meid": meid, "teid": teid,
                                    "mid": c["mid"], "acct": c["acct"],
                                    "scores": []})
        g["scores"].append(c["score"])

    ranked = sorted(convs.values(), key=lambda g: _conv_score(g["scores"]),
                    reverse=True)[:k]
    hit_scores = {(c["mid"], c["acct"]): c["score"] for c in candidates}

    out: list[dict] = []
    for g in ranked:
        if g["kind"] == "matter":
            msgs = _read(MATTER_MESSAGES_CYPHER, eid=g["meid"])
        elif g["kind"] == "thread":
            msgs = _read(THREAD_MESSAGES_CYPHER, eid=g["teid"])
        else:
            msgs = _read(MESSAGE_CYPHER, mid=g["mid"], acct=g["acct"])
        if not msgs:
            continue
        for m in msgs:
            key = (m["mid"], m["acct"])
            m["score"] = hit_scores.get(key)
            m["hit"] = key in hit_scores
        out.append({
            "kind": "Matter" if g["kind"] == "matter" else "Thread",
            "label": msgs[0].get("conv_label") or "(conversation)",
            "score": _conv_score(g["scores"]),
            "messages": msgs,
        })
    return out


def _fmt_person(name: str | None, email: str | None) -> str:
    name, email = (name or "").strip(), (email or "").strip()
    if name and email:
        return f"{name} <{email}>"
    return name or email or "(unknown)"


def build_context(conversations: list[dict]) -> tuple[str, list[dict]]:
    """Render retrieved conversations into (a) a numbered, citable text bundle
    and (b) a parallel `sources` list mapping each [n] to that message's
    subject / sender / date and a working Gmail link.

    Each conversation is shown whole and in chronological order. If it fits
    the budget, every message is shown at full length; if not, the messages
    that matched the question keep full bodies and the rest are trimmed to a
    gist, and an enormous conversation is windowed to its most relevant
    messages. Rendering stops once TOTAL_BUDGET is reached.

    The answer step tells the model to cite [n] markers only (never URLs) —
    the client builds the links from `sources`."""
    if not conversations:
        return "(no relevant messages found)", []

    blocks: list[str] = []
    sources: list[dict] = []
    n = 0
    total = 0

    for conv in conversations:
        all_msgs = conv["messages"]
        msgs = all_msgs
        windowed = False
        # Too large to show every message even at the floor? Window to the
        # most relevant (query hits first), then restore chronological order.
        if len(msgs) * MIN_BODY > CONV_BUDGET:
            keep = max(1, CONV_BUDGET // MIN_BODY)
            msgs = sorted(msgs, key=lambda m: (m.get("hit", False),
                                               m.get("score") or 0.0),
                          reverse=True)[:keep]
            msgs = sorted(msgs, key=lambda m: m.get("sent_at") or "")
            windowed = True

        # Body caps. If the conversation fits whole at full length, every
        # message gets it. If not, the messages that matched the question
        # (★) keep full-length bodies and the rest are trimmed to a gist —
        # depth where the query landed, breadth everywhere else.
        natural = sum(min(len((m.get("body") or "").strip()), BODY_CHARS)
                      for m in msgs)
        if natural <= CONV_BUDGET:
            hit_cap = other_cap = BODY_CHARS
            fit = "all in full"
        else:
            n_other = sum(1 for m in msgs if not m.get("hit"))
            other_cap = MIN_BODY
            n_hit = max(1, len(msgs) - n_other)
            hit_cap = max(MIN_BODY, min(
                BODY_CHARS, (CONV_BUDGET - n_other * other_cap) // n_hit))
            fit = "★ messages in full, the rest trimmed to a gist"
        shown = (f"showing the {len(msgs)} most relevant — {fit}"
                 if windowed else f"all shown — {fit}")
        blocks.append(
            f'=== {conv["kind"]}: "{conv["label"]}" — '
            f'{len(all_msgs)} message(s) in this conversation, {shown}, '
            f'oldest first. Messages marked ★ matched the question. ===')

        for m in msgs:
            n += 1
            who = _fmt_person(m.get("from_name"), m.get("from_email"))
            sources.append({
                "n": n,
                "subject": m.get("subject") or "(no subject)",
                "from": who,
                "date": (m.get("sent_at") or "")[:10],
                "url": gmail_search_url(m.get("gmail_url"), m.get("rfc822")),
            })
            cap = hit_cap if m.get("hit") else other_cap
            body = (m.get("body") or "").strip()
            if len(body) > cap:
                body = body[:cap].rstrip() + " …[truncated]"
            if not body:
                body = (m.get("snippet") or "").strip() or "(no body text)"
            total += len(body)
            meta: list[str] = []
            if m.get("orgs"):
                meta.append("Orgs: " + ", ".join(m["orgs"]))
            if m.get("attachments"):
                meta.append("Attachments: " + ", ".join(m["attachments"]))
            lines = [
                f'[{n}]{" ★" if m.get("hit") else ""} '
                f'{m.get("sent_at") or "?"} — '
                f'"{m.get("subject") or "(no subject)"}" — from {who} '
                f'· mailbox: {m.get("acct") or "?"}',
            ]
            if meta:
                lines.append("    " + "  |  ".join(meta))
            lines.append("    " + body.replace("\n", "\n    "))
            blocks.append("\n".join(lines))

        if total > TOTAL_BUDGET:
            blocks.append("[... further conversations omitted to stay within "
                          "the context budget; ask a narrower question or "
                          "use the Neo4j tools for more ...]")
            break

    return "\n\n".join(blocks), sources
