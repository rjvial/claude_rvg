"""Graph-RAG retrieval core for the /api/ask endpoint.

The unit of retrieval is the CONVERSATION, not the message. An answer about a
matter's status or history needs its whole arc — not a handful of
keyword-matched replies — so:

  1. Semantic — embed the question and vector-search the message_embedding
     index for a pool of candidate messages. Candidates are grouped by
     conversation (Matter if the thread has one, else Thread), and the
     conversations are ranked by their best-matching messages.
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

import sys
import threading
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


def retrieve(session, question: str,
             k: int = MAX_CONVERSATIONS) -> list[dict]:
    """Graph-RAG retrieval. Vector-searches a candidate pool, groups the hits
    into conversations, ranks them, and pulls the top k conversations WHOLE.
    Returns a list of conversation dicts (best first), each:
        {kind, label, score, messages: [<every message, chronological>]}
    Each message carries `hit` / `score` flagging whether it matched the
    query. build_context() turns this into the prompt bundle."""
    qvec = embed_query(question)
    candidates = [dict(r) for r in session.run(
        CANDIDATES_CYPHER, index=VECTOR_INDEX, candidates=CANDIDATE_POOL,
        qvec=qvec)]
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
            rows = session.run(MATTER_MESSAGES_CYPHER, eid=g["meid"])
        elif g["kind"] == "thread":
            rows = session.run(THREAD_MESSAGES_CYPHER, eid=g["teid"])
        else:
            rows = session.run(MESSAGE_CYPHER, mid=g["mid"], acct=g["acct"])
        msgs = [dict(r) for r in rows]
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
