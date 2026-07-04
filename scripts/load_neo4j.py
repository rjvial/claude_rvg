"""Load emails_clean.jsonl into Neo4j.

All layers are deterministic and zero-token (no LLM extraction):
  1. Messages, Threads, Persons, SENT, RECEIVED_BY, IN_THREAD — from headers.
  1b. REPLY_TO tree (true RFC conversation via In-Reply-To/References) +
     NEXT_IN_THREAD chronological backbone + Thread STARTS_WITH.
  2. Orgs + WORKS_AT — derived from email-domain via data/orgs_seed.json.
     Personal-email providers are skipped.
  4. Events + ORGANIZED/INVITED edges — from calendar_events.jsonl. Person
     nodes are shared with Layer 1, so calendar attendees and email senders
     unify on email. The same meeting seen in multiple account calendars
     collapses on (ical_uid, start).

Removed layers:
  - LLM entity extraction (former Layer 3): non-domain Orgs + MENTIONS/
    DISCUSSES from entities.jsonl. Removed 2026-05-25.
  - Topic nodes: superseded by Concepts on 2026-05-18.
  - Concepts (former Layer 3b): controlled-vocabulary MENTIONS from
    data/concepts.json. Removed 2026-05-26.

Usage:
    python scripts/load_neo4j.py --setup   # create constraints/indexes once
    python scripts/load_neo4j.py           # load all data
    python scripts/load_neo4j.py --batch 100
"""
from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from _attachments import is_real_attachment
from _common import DATA_DIR, message_bucket, neo4j_driver

EMAILS_CLEAN_JSONL = DATA_DIR / "emails_clean.jsonl"
ORGS_SEED_PATH = DATA_DIR / "orgs_seed.json"
CALENDAR_EVENTS_JSONL = DATA_DIR / "calendar_events.jsonl"
# Hand-edited alias groups (same file style_profiles.py reads): addresses of
# one person whose display names don't auto-merge.
RECIPIENT_ALIASES_PATH = DATA_DIR / "recipient_aliases.json"

# Pre-migration drops: the single-property message_id / thread_id constraints
# from earlier versions of this script must be dropped before we can apply the
# compound ones below.
DROPS = [
    "DROP CONSTRAINT message_id IF EXISTS",
    "DROP CONSTRAINT thread_id  IF EXISTS",
    # Project label was removed on 2026-05-14; Topic on 2026-05-25 (LLM
    # extraction removed); Concept on 2026-05-26 (controlled vocabulary
    # removed). Drop their schema if a prior version left it behind.
    "DROP CONSTRAINT project_name    IF EXISTS",
    "DROP CONSTRAINT topic_name      IF EXISTS",
    "DROP CONSTRAINT concept_key     IF EXISTS",
    "DROP INDEX      concept_canonical IF EXISTS",
    "DROP INDEX      concept_category  IF EXISTS",
    # Accent-folding migration (2026-05-30): the original message_text index used
    # the default `standard` analyzer, which tokenises "Muñoz" and "Munoz" as
    # DIFFERENT tokens — so a keyword search for the un-accented spelling silently
    # missed every accented occurrence (6 hits vs 28 for the same name). Drop it so
    # the CREATE below rebuilds it with `standard-folding`, which folds diacritics.
    # CREATE FULLTEXT ... IF NOT EXISTS will NOT change an existing index's
    # analyzer, so the drop is required for the migration to take effect.
    "DROP INDEX      message_text       IF EXISTS",
]

CONSTRAINTS = [
    "CREATE CONSTRAINT person_email   IF NOT EXISTS FOR (p:Person)  REQUIRE p.email IS UNIQUE",
    "CREATE CONSTRAINT org_name       IF NOT EXISTS FOR (o:Org)     REQUIRE o.canonical_name IS UNIQUE",
    # Matter is the multi-thread unification node written by cluster_matters.py.
    # Key is the normalized canonical subject (e.g. "proyecto andes").
    "CREATE CONSTRAINT matter_key     IF NOT EXISTS FOR (m:Matter)  REQUIRE m.canonical_key IS UNIQUE",
    # Gmail IDs are scoped per mailbox; the compound key (account_owner + ID) is
    # what's globally unique.
    "CREATE CONSTRAINT thread_id_acct  IF NOT EXISTS FOR (t:Thread)  REQUIRE (t.gmail_thread_id, t.account_owner) IS UNIQUE",
    "CREATE CONSTRAINT message_id_acct IF NOT EXISTS FOR (m:Message) REQUIRE (m.gmail_message_id, m.account_owner) IS UNIQUE",
    # Attachments are per-message, keyed by MIME part_id (stable per re-fetch).
    "CREATE CONSTRAINT attachment_key  IF NOT EXISTS FOR (a:Attachment) REQUIRE (a.gmail_message_id, a.account_owner, a.part_id) IS UNIQUE",
    "CREATE INDEX message_sent_at    IF NOT EXISTS FOR (m:Message) ON (m.sent_at)",
    # Native temporal twin of sent_at (same instant, ZonedDateTime). The ISO
    # string stays for the app payload and lexicographic ORDER BY; sent_dt
    # gives Cypher real date arithmetic (date.truncate, durations, ranges)
    # for the /api/ask aggregation queries.
    "CREATE INDEX message_sent_dt   IF NOT EXISTS FOR (m:Message) ON (m.sent_dt)",
    # Revision marker (ms) — backs serve_app's incremental cache rebuild
    # (MATCH ... WHERE m.rev > $since).
    "CREATE INDEX message_rev       IF NOT EXISTS FOR (m:Message) ON (m.rev)",
    # Accent-folding people lookup: resolving "munoz" → the Person node for
    # "María Muñoz" used to require searching message bodies. Same folding
    # analyzer as message_text.
    "CREATE FULLTEXT INDEX person_name IF NOT EXISTS "
    "FOR (p:Person) ON EACH [p.name] "
    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard-folding'}}",
    # Treatment tier (primary vs lite buckets). Lets embed/cluster/retrieval
    # filter to bucket='primary' cheaply, and the app browse by bucket.
    "CREATE INDEX message_bucket     IF NOT EXISTS FOR (m:Message) ON (m.bucket)",
    "CREATE INDEX thread_started_at  IF NOT EXISTS FOR (t:Thread)  ON (t.started_at)",
    "CREATE INDEX attachment_filename IF NOT EXISTS FOR (a:Attachment) ON (a.filename)",
    # RFC-822 Message-ID is globally unique by spec; an index lets us look up
    # by rfc822msgid: search URLs and dedupe across accounts when needed.
    "CREATE INDEX message_rfc822     IF NOT EXISTS FOR (m:Message) ON (m.rfc822_message_id)",
    # Reverse lookup for the reply tree: given a message-id, who replied to it.
    "CREATE INDEX message_in_reply_to IF NOT EXISTS FOR (m:Message) ON (m.in_reply_to)",
    # Vector index for semantic (graph-RAG) retrieval over Message.embedding,
    # written by scripts/embed_messages.py and queried by the /api/ask
    # retriever via db.index.vector.queryNodes. 1024 dims = the embedding model
    # intfloat/multilingual-e5-large; cosine matches e5's normalized vectors.
    # If you swap the embedding model, change the dimension here and re-embed
    # the whole corpus (embed_messages.py --reembed).
    "CREATE VECTOR INDEX message_embedding IF NOT EXISTS "
    "FOR (m:Message) ON (m.embedding) "
    "OPTIONS {indexConfig: {"
    "`vector.dimensions`: 1024, "
    "`vector.similarity_function`: 'cosine'}}",
    # Full-text index over subject + cleaned body — gives the /api/ask answer
    # step a fast keyword tool (db.index.fulltext.queryNodes) for exact terms
    # that semantic vectors miss (names, identifiers, quoted phrases). It
    # complements the vector index: vectors for meaning, full-text for tokens.
    # `standard-folding` analyzer: folds diacritics so a search for the ASCII
    # spelling ("Munoz") matches the accented form ("Muñoz") and vice-versa —
    # essential for Spanish names/terms in this corpus (see the DROP above).
    "CREATE FULLTEXT INDEX message_text IF NOT EXISTS "
    "FOR (n:Message) ON EACH [n.subject, n.body_clean] "
    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard-folding'}}",
    # Calendar events. event_key is a computed, always-non-null collapse key
    # = (ical_uid, start) so the SAME meeting across multiple account calendars
    # merges to one node while distinct occurrences of a recurring series stay
    # separate. ical_uid / start_ms are kept as queryable properties.
    "CREATE CONSTRAINT event_key      IF NOT EXISTS FOR (e:Event)  REQUIRE e.event_key IS UNIQUE",
    "CREATE INDEX event_start        IF NOT EXISTS FOR (e:Event)  ON (e.start_ms)",
    "CREATE INDEX event_ical_uid     IF NOT EXISTS FOR (e:Event)  ON (e.ical_uid)",
]


def setup_schema(driver) -> None:
    print("Setting up constraints and indexes...")
    with driver.session() as session:
        for stmt in DROPS:
            session.run(stmt)
        for stmt in CONSTRAINTS:
            session.run(stmt)
        # Idempotent data backfill: sent_dt for nodes loaded before the
        # property existed. session.run is auto-commit, which CALL … IN
        # TRANSACTIONS requires.
        session.run(
            "MATCH (m:Message) "
            "WHERE m.sent_dt IS NULL AND m.sent_at IS NOT NULL "
            "CALL (m) { SET m.sent_dt = datetime(m.sent_at) } "
            "IN TRANSACTIONS OF 10000 ROWS").consume()
    print("  Done.")


# ---------------------------------------------------------------------------
# Messages + threads + sender/recipient edges
# ---------------------------------------------------------------------------

LOAD_MESSAGES_CYPHER = """
UNWIND $rows AS row
MERGE (t:Thread {gmail_thread_id: row.thread_id, account_owner: row.account_owner})
  ON CREATE SET t.subject = row.subject, t.started_at = row.sent_at,
                t.last_msg_at = row.sent_at
  // Rows are NOT guaranteed chronological (backfill batches, re-pulls), so
  // started_at/last_msg_at only move outward — a mid-thread row must not
  // regress last_msg_at or inflate started_at. ISO-8601 UTC strings compare
  // correctly as text.
  ON MATCH  SET
    t.started_at = CASE WHEN row.sent_at IS NOT NULL
                         AND (t.started_at IS NULL OR row.sent_at < t.started_at)
                        THEN row.sent_at ELSE t.started_at END,
    t.last_msg_at = CASE WHEN row.sent_at IS NOT NULL
                          AND (t.last_msg_at IS NULL OR row.sent_at > t.last_msg_at)
                         THEN row.sent_at ELSE t.last_msg_at END
MERGE (m:Message {gmail_message_id: row.message_id, account_owner: row.account_owner})
  ON CREATE SET
    // Revision marker for the app server's incremental cache rebuild: every
    // creation and every later label/spam mutation stamps m.rev with server
    // time (ms), so serve_app can fetch "changed since my last rebuild"
    // instead of re-reading the whole graph. Legacy nodes without rev are
    // simply never re-fetched incrementally — correct, since any mutation
    // path stamps rev when it touches them.
    m.rev = timestamp(),
    m.sent_at = row.sent_at,
    // Native datetime twin of the ISO string (datetime(null) is null-safe).
    m.sent_dt = datetime(row.sent_at),
    m.subject = row.subject,
    m.snippet = row.snippet,
    m.body_clean = row.body_clean,
    m.has_attachments = row.has_attachments,
    m.label_ids = row.label_ids,
    m.bucket = row.bucket,
    m.gmail_url = row.gmail_url,
    m.rfc822_message_id = row.rfc822_message_id,
    m.in_reply_to = row.in_reply_to,
    m.references = row.references,
    // Per-message sender display name. Stored on the Message (not just the
    // sender Person) because Person is keyed by email alone, so senders that
    // share one address but vary the display name — e.g. every Docusign
    // notification comes from dse_*@docusign.net under a different human's
    // name — collapse into one Person and would otherwise all render with
    // whichever name won. m.from_name keeps each message's true sender.
    m.from_name = row.from_name
  ON MATCH SET
    // gmail_url is derived from ids; always refresh to the latest form
    // (e.g. promote thread-anchored URLs to per-message rfc822msgid: URLs
    // once the backfill populates rfc822_message_id).
    // NB: m.label_ids and m.bucket are intentionally NOT refreshed here — the
    // live read/spam state is maintained on existing nodes by sync_incremental
    // (apply_read_changes / apply_spam_changes), and a stale jsonl row must not
    // clobber it. bucket is set ON CREATE; legacy nodes are backfilled once.
    m.gmail_url = row.gmail_url,
    m.rfc822_message_id = coalesce(m.rfc822_message_id, row.rfc822_message_id),
    m.in_reply_to = coalesce(m.in_reply_to, row.in_reply_to),
    m.references = CASE WHEN m.references IS NULL OR size(m.references) = 0
                        THEN row.references ELSE m.references END,
    // Backfill legacy nodes that predate m.from_name; never null out a value
    // already present if a re-pulled row happens to lack the name.
    m.from_name = coalesce(row.from_name, m.from_name),
    // Backfill legacy nodes that predate m.sent_dt.
    m.sent_dt = coalesce(m.sent_dt, datetime(row.sent_at))
MERGE (m)-[:IN_THREAD]->(t)
WITH m, row
WHERE row.from_email IS NOT NULL
MERGE (sender:Person {email: row.from_email})
  ON CREATE SET sender.name = row.from_name
  ON MATCH  SET sender.name = coalesce(sender.name, row.from_name)
MERGE (sender)-[:SENT]->(m)
"""

LOAD_ATTACHMENTS_CYPHER = """
UNWIND $rows AS row
MATCH (m:Message {gmail_message_id: row.message_id, account_owner: row.account_owner})
MERGE (a:Attachment {
  gmail_message_id: row.message_id,
  account_owner: row.account_owner,
  part_id: row.part_id
})
  ON CREATE SET
    a.filename = row.filename,
    a.mime_type = row.mime_type,
    a.size = row.size,
    a.attachment_id = row.attachment_id
MERGE (m)-[:HAS_ATTACHMENT]->(a)
"""

LOAD_RECIPIENTS_CYPHER = """
UNWIND $rows AS row
MATCH (m:Message {gmail_message_id: row.message_id, account_owner: row.account_owner})
MERGE (p:Person {email: row.email})
  ON CREATE SET p.name = row.name
  ON MATCH  SET p.name = coalesce(p.name, row.name)
MERGE (m)-[r:RECEIVED_BY {kind: row.kind}]->(p)
"""


def msg_row(rec: dict) -> dict:
    from_ = rec.get("from") or {}
    # Recompute has_attachments from the filtered view so old jsonl rows
    # (whose top-level has_attachments was set before classification existed)
    # don't carry inline-image inflation into the graph.
    real_atts = any(is_real_attachment(a) for a in rec.get("attachments") or [])
    return {
        "account_owner": rec.get("account_owner") or "",
        "message_id": rec["message_id"],
        "thread_id": rec.get("thread_id"),
        "sent_at": rec.get("sent_at"),
        "subject": rec.get("subject"),
        "snippet": rec.get("snippet"),
        "body_clean": rec.get("body_clean") or "",
        "has_attachments": real_atts,
        "label_ids": rec.get("label_ids") or [],
        "bucket": message_bucket(rec.get("label_ids")),
        "gmail_url": rec.get("gmail_url"),
        "rfc822_message_id": rec.get("rfc822_message_id"),
        "in_reply_to": rec.get("in_reply_to"),
        "references": rec.get("references") or [],
        "from_email": (from_.get("email") or "").lower() or None,
        "from_name": from_.get("name"),
    }


def attachment_rows(rec: dict) -> list[dict]:
    out: list[dict] = []
    account_owner = rec.get("account_owner") or ""
    for a in rec.get("attachments") or []:
        filename = a.get("filename")
        if not filename:
            continue
        # Skip inline images, S/MIME blobs, Outlook artifacts. Uses the `kind`
        # field if pull_gmail.py wrote it, otherwise re-classifies from
        # filename + mime_type alone — so existing emails.jsonl re-loads
        # without needing a re-pull.
        if not is_real_attachment(a):
            continue
        out.append({
            "account_owner": account_owner,
            "message_id": rec["message_id"],
            "part_id": a.get("part_id") or "",
            "filename": filename,
            "mime_type": a.get("mime_type"),
            "size": a.get("size"),
            "attachment_id": a.get("attachment_id"),
        })
    return out


def recipient_rows(rec: dict) -> list[dict]:
    out: list[dict] = []
    account_owner = rec.get("account_owner") or ""
    for kind in ("to", "cc", "bcc"):
        for a in rec.get(kind) or []:
            email = (a.get("email") or "").lower()
            if not email:
                continue
            out.append({
                "account_owner": account_owner,
                "message_id": rec["message_id"],
                "email": email,
                "name": a.get("name"),
                "kind": kind,
            })
    return out


def load_messages(driver, batch: int) -> None:
    if not EMAILS_CLEAN_JSONL.exists():
        raise SystemExit(f"Missing {EMAILS_CLEAN_JSONL}.")
    total = sum(1 for _ in EMAILS_CLEAN_JSONL.open(encoding="utf-8"))

    msg_batch: list[dict] = []
    recv_batch: list[dict] = []
    att_batch: list[dict] = []

    with driver.session() as session, \
            EMAILS_CLEAN_JSONL.open(encoding="utf-8") as fin, \
            tqdm(total=total, desc="messages") as bar:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bar.update(1)
                continue
            # Promo/social/updates/forums mail is loaded like everything else —
            # msg_row() tags it with m.bucket (lite tier). The embed/cluster/
            # retrieval steps skip non-'primary' buckets, so the graph carries
            # the mail without giving it the full treatment. (Spam is loaded the
            # same way and tagged bucket='spam'.)
            msg_batch.append(msg_row(rec))
            recv_batch.extend(recipient_rows(rec))
            att_batch.extend(attachment_rows(rec))
            if len(msg_batch) >= batch:
                session.run(LOAD_MESSAGES_CYPHER, rows=msg_batch)
                if recv_batch:
                    session.run(LOAD_RECIPIENTS_CYPHER, rows=recv_batch)
                if att_batch:
                    session.run(LOAD_ATTACHMENTS_CYPHER, rows=att_batch)
                bar.update(len(msg_batch))
                msg_batch.clear()
                recv_batch.clear()
                att_batch.clear()
        if msg_batch:
            session.run(LOAD_MESSAGES_CYPHER, rows=msg_batch)
            if recv_batch:
                session.run(LOAD_RECIPIENTS_CYPHER, rows=recv_batch)
            if att_batch:
                session.run(LOAD_ATTACHMENTS_CYPHER, rows=att_batch)
            bar.update(len(msg_batch))


# ---------------------------------------------------------------------------
# Layer 1b: thread-native graph — REPLY_TO tree + NEXT_IN_THREAD backbone
# ---------------------------------------------------------------------------
#
# REPLY_TO follows the TRUE RFC conversation: child.in_reply_to (then the
# References chain, nearest ancestor first) resolved against parent
# rfc822_message_id — even across Gmail's ~100-msg thread split or accounts.
# The same email can exist as several Message nodes (one per mailbox copy);
# we keep the highest-priority candidate and, among copies of that candidate,
# prefer the parent in the child's own account, then its own Gmail thread.
#
# NEXT_IN_THREAD is the Gmail-thread-scoped chronological spine: a guaranteed
# traversable order even where headers are missing or point outside the
# corpus. REPLY_TO = truth (may cross threads); NEXT_IN_THREAD = backbone.

LOAD_REPLY_EDGES_CYPHER = """
UNWIND $rows AS row
MATCH (c:Message {gmail_message_id: row.message_id, account_owner: row.account_owner})
// Priority-ordered candidate parents: direct In-Reply-To first, then the
// References chain newest→oldest. Drop nulls and any self-reference.
WITH c, row,
     [x IN ([row.in_reply_to] + reverse(row.references))
        WHERE x IS NOT NULL AND x <> coalesce(c.rfc822_message_id, '')] AS cands
WHERE size(cands) > 0
CALL (c, cands) {
  UNWIND range(0, size(cands) - 1) AS i
  WITH c, i, cands[i] AS cand
  MATCH (p:Message {rfc822_message_id: cand})
  WHERE p <> c
  OPTIONAL MATCH (c)-[:IN_THREAD]->(ct:Thread)
  OPTIONAL MATCH (p)-[:IN_THREAD]->(pt:Thread)
  WITH c, i, cand, p,
       CASE WHEN p.account_owner = c.account_owner THEN 1 ELSE 0 END AS same_acct,
       CASE WHEN ct IS NOT NULL AND pt IS NOT NULL
                 AND ct.gmail_thread_id = pt.gmail_thread_id
                 AND ct.account_owner = pt.account_owner THEN 1 ELSE 0 END AS same_thr
  ORDER BY i ASC, same_acct DESC, same_thr DESC
  LIMIT 1
  RETURN p AS parent, cand AS matched
}
MERGE (c)-[r:REPLY_TO]->(parent)
  SET r.via = CASE WHEN matched = row.in_reply_to THEN 'in_reply_to'
                   ELSE 'references' END
"""

MARK_ROOTS_CYPHER = """
MATCH (m:Message)
SET m.is_thread_root = NOT (m)-[:REPLY_TO]->(:Message)
"""

CLEAR_BACKBONE_CYPHER = """
MATCH (:Message)-[r:NEXT_IN_THREAD]->() DELETE r
"""
CLEAR_STARTS_WITH_CYPHER = """
MATCH (:Thread)-[r:STARTS_WITH]->() DELETE r
"""
BUILD_BACKBONE_CYPHER = """
MATCH (t:Thread)<-[:IN_THREAD]-(m:Message)
WITH t, m ORDER BY m.sent_at ASC, m.gmail_message_id ASC
WITH t, collect(m) AS ms
WHERE size(ms) > 0
WITH t, ms, ms[0] AS first
MERGE (t)-[:STARTS_WITH]->(first)
WITH ms
WHERE size(ms) > 1
CALL apoc.nodes.link(ms, 'NEXT_IN_THREAD', {avoidDuplicates: true})
RETURN count(*) AS linked
"""


def reply_rows(rec: dict) -> dict:
    return {
        "account_owner": rec.get("account_owner") or "",
        "message_id": rec["message_id"],
        "in_reply_to": rec.get("in_reply_to"),
        "references": rec.get("references") or [],
    }


def load_reply_edges(driver, batch: int) -> None:
    """Build the REPLY_TO tree from RFC threading headers, then flag roots
    (messages with no in-corpus parent). Streams emails_clean.jsonl so it
    works standalone (e.g. with --skip-messages on an already-loaded graph)."""
    if not EMAILS_CLEAN_JSONL.exists():
        print(f"No {EMAILS_CLEAN_JSONL} — skipping reply tree.")
        return
    total = sum(1 for _ in EMAILS_CLEAN_JSONL.open(encoding="utf-8"))

    rep_batch: list[dict] = []
    with driver.session() as session, \
            EMAILS_CLEAN_JSONL.open(encoding="utf-8") as fin, \
            tqdm(total=total, desc="reply-tree") as bar:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bar.update(1)
                continue
            rep_batch.append(reply_rows(rec))
            if len(rep_batch) >= batch:
                session.run(LOAD_REPLY_EDGES_CYPHER, rows=rep_batch)
                bar.update(len(rep_batch))
                rep_batch.clear()
        if rep_batch:
            session.run(LOAD_REPLY_EDGES_CYPHER, rows=rep_batch)
            bar.update(len(rep_batch))
        print("  flagging thread roots...")
        session.run(MARK_ROOTS_CYPHER)


def fast_load_records(records: list[dict], driver=None) -> None:
    """Targeted Layer-1 + REPLY_TO load for a known set of new records, instead
    of streaming all of emails_clean.jsonl through main(). Skips the iterate-
    every-message layers (NEXT_IN_THREAD backbone, domain Orgs, calendar
    Events) — those backfill on the next full run of scripts/run_pipeline.py
    and aren't needed for routine incremental sync. REPLY_TO is included
    because its cypher resolves parents per-row via rfc822 id, so new replies
    thread correctly the moment they land.

    Pass a driver to reuse an open one; otherwise a temporary one is opened
    and closed. Shared by the incremental sync paths (sync_incremental,
    serve_app)."""
    if not records:
        return
    # Categorized mail is kept (lite tier) and tagged by msg_row() -> m.bucket,
    # same as the full load; no category drop here anymore.
    own = driver is None
    drv = driver or neo4j_driver()
    try:
        msg_rows = [msg_row(r) for r in records]
        recv_rows: list[dict] = []
        att_rows: list[dict] = []
        rep_rows: list[dict] = []
        for r in records:
            recv_rows.extend(recipient_rows(r))
            att_rows.extend(attachment_rows(r))
            rep_rows.append(reply_rows(r))
        with drv.session() as session:
            session.run(LOAD_MESSAGES_CYPHER, rows=msg_rows)
            if recv_rows:
                session.run(LOAD_RECIPIENTS_CYPHER, rows=recv_rows)
            if att_rows:
                session.run(LOAD_ATTACHMENTS_CYPHER, rows=att_rows)
            if rep_rows:
                session.run(LOAD_REPLY_EDGES_CYPHER, rows=rep_rows)
    finally:
        if own:
            drv.close()


def load_backbone(driver) -> None:
    """Rebuild the NEXT_IN_THREAD chronological spine + Thread STARTS_WITH.
    Full rebuild (clear then recreate) so it stays correct when messages are
    added to existing threads on a re-run."""
    print("Backbone: rebuilding NEXT_IN_THREAD + STARTS_WITH...")
    with driver.session() as session:
        session.run(CLEAR_BACKBONE_CYPHER)
        session.run(CLEAR_STARTS_WITH_CYPHER)
        session.run(BUILD_BACKBONE_CYPHER)
    print("  Done.")


# ---------------------------------------------------------------------------
# Layer 2: Org-from-domain + WORKS_AT (deterministic, zero tokens)
# ---------------------------------------------------------------------------

LOAD_DOMAIN_ORGS_CYPHER = """
UNWIND $rows AS row
MERGE (org:Org {canonical_name: row.canonical_name})
  ON CREATE SET org.domain = row.domain, org.aliases = row.aliases
  ON MATCH  SET org.domain = coalesce(org.domain, row.domain),
                org.aliases = coalesce(org.aliases, row.aliases)
"""

LOAD_WORKS_AT_CYPHER = """
UNWIND $rows AS row
MATCH (p:Person {email: row.email})
MATCH (org:Org {canonical_name: row.canonical_name})
MERGE (p)-[r:WORKS_AT]->(org)
  ON CREATE SET r.confidence = 1.0, r.source = 'domain'
"""


def load_domain_orgs(driver, seed: dict) -> None:
    domain_map: dict[str, dict] = seed.get("domains") or {}
    rows = [
        {
            "canonical_name": v["canonical_name"],
            "domain": d,
            "aliases": v.get("aliases") or [],
        }
        for d, v in domain_map.items()
    ]
    if not rows:
        return
    print(f"Layer 2a: seeding {len(rows)} Orgs from domain map...")
    with driver.session() as session:
        session.run(LOAD_DOMAIN_ORGS_CYPHER, rows=rows)


# Auto-derived Orgs: any non-personal sender/recipient domain seen on PRIMARY
# mail that isn't in orgs_seed.json still becomes an Org (canonical_name =
# the bare domain), so Liam's "resolve the org by domain" anchor works for
# organizations that were never hand-seeded. Restricting to primary-bucket
# participants keeps newsletter/promo sender domains from minting Org noise.
AUTO_ORGS_CYPHER = """
UNWIND $rows AS row
MERGE (org:Org {canonical_name: row.canonical_name})
  ON CREATE SET org.domain = row.domain, org.aliases = [],
                org.source = 'auto_domain'
  ON MATCH  SET org.domain = coalesce(org.domain, row.domain)
"""


def load_works_at(driver, seed: dict) -> None:
    domain_map: dict[str, dict] = seed.get("domains") or {}
    personal = set((seed.get("_meta") or {}).get("personal_email_providers") or [])
    rows: list[dict] = []
    auto_domains: set[str] = set()
    with EMAILS_CLEAN_JSONL.open(encoding="utf-8") as fh:
        seen: set[tuple[str, str]] = set()
        for line in fh:
            # Tolerate blank/corrupt lines (an interrupted sync can leave a
            # partial trailing line) — same guard reconcile_graph uses.
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            primary = message_bucket(rec.get("label_ids")) == "primary"
            for kind in ("from", "to", "cc", "bcc"):
                v = rec.get(kind)
                if isinstance(v, dict):
                    v = [v]
                for a in v or []:
                    email = (a.get("email") or "").lower()
                    if "@" not in email:
                        continue
                    domain = email.split("@", 1)[1]
                    if domain in personal:
                        continue
                    info = domain_map.get(domain)
                    if info:
                        canonical = info["canonical_name"]
                    elif primary and "." in domain:
                        canonical = domain          # auto-org, named by domain
                        auto_domains.add(domain)
                    else:
                        continue
                    key = (email, canonical)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({"email": email, "canonical_name": canonical})
    if auto_domains:
        print(f"Layer 2b: creating {len(auto_domains)} auto Orgs from "
              f"unseeded primary-mail domains...")
        auto_rows = [{"canonical_name": d, "domain": d}
                     for d in sorted(auto_domains)]
        with driver.session() as session:
            for i in range(0, len(auto_rows), 500):
                session.run(AUTO_ORGS_CYPHER, rows=auto_rows[i:i + 500])
    if not rows:
        return
    print(f"Layer 2b: {len(rows)} WORKS_AT edges from domain map...")
    with driver.session() as session:
        for i in range(0, len(rows), 500):
            session.run(LOAD_WORKS_AT_CYPHER, rows=rows[i:i + 500])


# ---------------------------------------------------------------------------
# Layer 2c: Person identity — ALIAS_OF edges (deterministic, zero tokens)
# ---------------------------------------------------------------------------
#
# One human often has several addresses (work + personal, old + new). The
# merge used to live only in style_profiles.py's Python union-find, invisible
# to Liam's Cypher. This layer materializes it in the graph:
#     (alias:Person)-[:ALIAS_OF]->(canonical:Person)
# from two deterministic signals: exact normalized-name equality (names with
# ≥2 tokens only, so two bare "Juan"s never fuse) and the hand-edited groups
# in data/recipient_aliases.json. Full rebuild each run (clear + recreate) —
# idempotent, and edits to the alias file take effect on the next load.

CLEAR_ALIAS_CYPHER = "MATCH ()-[r:ALIAS_OF]->() DELETE r"
LOAD_ALIAS_CYPHER = """
UNWIND $rows AS row
MATCH (a:Person {email: row.email})
MATCH (c:Person {email: row.canonical})
MERGE (a)-[r:ALIAS_OF]->(c)
  SET r.source = row.source
"""


def _norm_person_name(name: str) -> str:
    """Lower-case, strip accents/quotes, collapse spaces — mirrors
    style_profiles._norm_name so both mergers agree."""
    import re
    import unicodedata
    s = (name or "").strip().strip("'\"").strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).lower()


def load_person_aliases(driver) -> None:
    with driver.session() as s:
        people = s.run(
            "MATCH (p:Person) WHERE p.email IS NOT NULL "
            "RETURN p.email AS email, p.name AS name").data()
    emails = {(r["email"] or "").lower() for r in people if r["email"]}

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    # Signal 1 — exact normalized full-name match (≥2 tokens).
    by_name: dict[str, list[str]] = {}
    for r in people:
        email = (r["email"] or "").lower()
        nk = _norm_person_name(r["name"] or "")
        if email and nk and len(nk.split()) >= 2:
            by_name.setdefault(nk, []).append(email)
    for grp in by_name.values():
        for e in grp[1:]:
            union(grp[0], e)

    # Signal 2 — the hand-edited alias file (only emails that exist as
    # Person nodes; LOAD_ALIAS_CYPHER MATCHes would skip the rest anyway).
    manual: set[str] = set()
    try:
        d = json.loads(RECIPIENT_ALIASES_PATH.read_text(encoding="utf-8"))
        groups = (d.get("groups") if isinstance(d, dict) else None) or []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        groups = []
    for g in groups:
        ems = [e.strip().lower() for e in (g.get("emails") or [])
               if isinstance(e, str) and e.strip().lower() in emails]
        for e in ems[1:]:
            union(ems[0], e)
        manual.update(ems)

    # Components → (alias, canonical) rows. Canonical = lexicographically
    # smallest email (deterministic; Liam traverses ALIAS_OF undirected, so
    # the choice of canonical is presentational).
    comps: dict[str, list[str]] = {}
    for e in list(parent):
        comps.setdefault(find(e), []).append(e)
    rows = []
    for members in comps.values():
        if len(members) < 2:
            continue
        canonical = min(members)
        for e in members:
            if e != canonical:
                rows.append({"email": e, "canonical": canonical,
                             "source": "manual" if e in manual else "name"})
    print(f"Layer 2c: {len(rows)} ALIAS_OF edges "
          f"({sum(1 for r in rows if r['source'] == 'manual')} from the "
          f"alias file)...")
    with driver.session() as s:
        s.run(CLEAR_ALIAS_CYPHER)
        for i in range(0, len(rows), 1000):
            s.run(LOAD_ALIAS_CYPHER, rows=rows[i:i + 1000])


# ---------------------------------------------------------------------------
# Layer 4: Calendar events + ORGANIZED/INVITED (deterministic, zero tokens)
# ---------------------------------------------------------------------------
#
# Person nodes are MERGEd by email — the SAME node set as Layer 1 — so a
# calendar attendee who also sends mail is one Person, not two. The Event node
# collapses on `event_key` (= ical_uid + start) so a meeting present in
# several account calendars is a single node; `account_owners` is the list of
# calendars it was seen in (provenance). Cancelled occurrences are filtered
# out before load (a recurring series' removed instances come back as
# status=cancelled and would otherwise be ghost nodes).

LOAD_EVENTS_CYPHER = """
UNWIND $rows AS row
MERGE (e:Event {event_key: row.event_key})
  ON CREATE SET
    e.ical_uid = row.ical_uid,
    e.summary = row.summary,
    e.description = row.description,
    e.location = row.location,
    e.start_iso = row.start_iso,
    e.start_ms = row.start_ms,
    e.end_iso = row.end_iso,
    e.end_ms = row.end_ms,
    e.all_day = row.all_day,
    e.status = row.status,
    e.recurring_event_id = row.recurring_event_id,
    e.html_link = row.html_link,
    e.hangout_link = row.hangout_link,
    e.account_owners = [row.account_owner]
  ON MATCH SET
    e.account_owners = CASE
      WHEN row.account_owner IN e.account_owners THEN e.account_owners
      ELSE e.account_owners + row.account_owner END
WITH e, row
WHERE row.organizer_email IS NOT NULL
MERGE (org:Person {email: row.organizer_email})
  ON CREATE SET org.name = row.organizer_name
  ON MATCH  SET org.name = coalesce(org.name, row.organizer_name)
MERGE (org)-[:ORGANIZED]->(e)
"""

LOAD_EVENT_ATTENDEES_CYPHER = """
UNWIND $rows AS row
MATCH (e:Event {event_key: row.event_key})
MERGE (p:Person {email: row.email})
  ON CREATE SET p.name = row.name
  ON MATCH  SET p.name = coalesce(p.name, row.name)
MERGE (p)-[r:INVITED]->(e)
  SET r.response = row.response, r.optional = row.optional
"""


def _event_key(rec: dict) -> str:
    """Always-non-null collapse key. (ical_uid, start) unifies the same
    meeting across calendars and keeps recurring occurrences distinct. Falls
    back to per-account event_id only when ical_uid is absent, so null-uid
    rows never MERGE into one another."""
    uid = rec.get("ical_uid")
    start = rec.get("start") or {}
    when = start.get("epoch_ms")
    if when is None:
        when = start.get("iso")
    if uid:
        return f"{uid}|{when}"
    return f"_nouid|{rec.get('account_owner')}|{rec.get('event_id')}"


def event_row(rec: dict) -> dict:
    org = rec.get("organizer") or {}
    start = rec.get("start") or {}
    end = rec.get("end") or {}
    return {
        "event_key": _event_key(rec),
        "account_owner": rec.get("account_owner") or "",
        "ical_uid": rec.get("ical_uid"),
        "summary": rec.get("summary") or "",
        "description": rec.get("description") or "",
        "location": rec.get("location") or "",
        "start_iso": start.get("iso"),
        "start_ms": start.get("epoch_ms"),
        "end_iso": end.get("iso"),
        "end_ms": end.get("epoch_ms"),
        "all_day": bool(start.get("all_day")),
        "status": rec.get("status"),
        "recurring_event_id": rec.get("recurring_event_id"),
        "html_link": rec.get("html_link"),
        "hangout_link": rec.get("hangout_link"),
        "organizer_email": (org.get("email") or "").lower() or None,
        "organizer_name": org.get("name"),
    }


def attendee_rows(rec: dict) -> list[dict]:
    out: list[dict] = []
    key = _event_key(rec)
    for a in rec.get("attendees") or []:
        # Rooms / equipment carry an email but aren't people — keep them out
        # of the Person graph.
        if a.get("resource"):
            continue
        email = (a.get("email") or "").lower()
        if not email:
            continue
        out.append({
            "event_key": key,
            "email": email,
            "name": a.get("name"),
            "response": a.get("response_status"),
            "optional": bool(a.get("optional")),
        })
    return out


def load_events(driver, batch: int) -> None:
    if not CALENDAR_EVENTS_JSONL.exists():
        print(f"No {CALENDAR_EVENTS_JSONL} — skipping calendar load.")
        return
    total = sum(1 for _ in CALENDAR_EVENTS_JSONL.open(encoding="utf-8"))

    ev_batch: list[dict] = []
    att_batch: list[dict] = []
    skipped_cancelled = 0

    def flush(session):
        if ev_batch:
            session.run(LOAD_EVENTS_CYPHER, rows=ev_batch)
        if att_batch:
            session.run(LOAD_EVENT_ATTENDEES_CYPHER, rows=att_batch)
        ev_batch.clear()
        att_batch.clear()

    with driver.session() as session, \
            CALENDAR_EVENTS_JSONL.open(encoding="utf-8") as fin, \
            tqdm(total=total, desc="events") as bar:
        for line in fin:
            rec = json.loads(line)
            if (rec.get("status") or "").lower() == "cancelled":
                skipped_cancelled += 1
                bar.update(1)
                continue
            ev_batch.append(event_row(rec))
            att_batch.extend(attendee_rows(rec))
            if len(ev_batch) >= batch:
                n = len(ev_batch)
                flush(session)
                bar.update(n)
        if ev_batch:
            n = len(ev_batch)
            flush(session)
            bar.update(n)
    if skipped_cancelled:
        print(f"  (skipped {skipped_cancelled} cancelled event rows)")


def load_seed() -> dict:
    if not ORGS_SEED_PATH.exists():
        print(f"WARN: {ORGS_SEED_PATH} not found — skipping Org-from-domain layer.")
        return {}
    with ORGS_SEED_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true",
                        help="Create constraints and indexes (idempotent).")
    parser.add_argument("--skip-messages", action="store_true",
                        help="Skip Layer 1 (messages/persons/threads).")
    parser.add_argument("--skip-thread-graph", action="store_true",
                        help="Skip Layer 1b (REPLY_TO tree + NEXT_IN_THREAD "
                             "backbone). Independent of --skip-messages.")
    parser.add_argument("--skip-domain-orgs", action="store_true",
                        help="Skip Layer 2 (Org-from-domain + WORKS_AT).")
    parser.add_argument("--skip-aliases", action="store_true",
                        help="Skip Layer 2c (Person ALIAS_OF identity edges).")
    parser.add_argument("--skip-events", action="store_true",
                        help="Skip Layer 4 (calendar_events.jsonl).")
    parser.add_argument("--batch", type=int, default=200,
                        help="Batch size for UNWIND.")
    args = parser.parse_args()

    driver = neo4j_driver()
    try:
        if args.setup:
            setup_schema(driver)
            return
        if not args.skip_messages:
            load_messages(driver, args.batch)
        if not args.skip_thread_graph:
            load_reply_edges(driver, args.batch)
            load_backbone(driver)
        if not args.skip_domain_orgs:
            seed = load_seed()
            if seed:
                load_domain_orgs(driver, seed)
                load_works_at(driver, seed)
        if not args.skip_aliases:
            load_person_aliases(driver)
        if not args.skip_events:
            load_events(driver, args.batch)
    finally:
        driver.close()
    print("Load complete.")


if __name__ == "__main__":
    main()
