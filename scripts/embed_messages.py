"""Embed every Message into a vector for semantic (graph-RAG) retrieval.

For each Message we embed its subject + cleaned body + real attachment
filenames, and store the result as the `embedding` property on the node. The
vector index `message_embedding` (created by load_neo4j.py --setup) then
powers db.index.vector.queryNodes(...) for the /api/ask graph-RAG retriever.

Attachment names matter: a message often *is* "the deed scan" or "the signed
contract" with almost no body text, so the filename carries the meaning. We
pull names from the Attachment nodes, which load_neo4j.py already filtered to
real attachments (inline images / S-MIME blobs were dropped).

Idempotent and zero-token: only Messages whose `embedding` is still null are
embedded, so the first run backfills the whole corpus and every later run
(after an incremental sync) touches just the new mail. Use --reembed to
recompute everything, e.g. after changing the model.

Resumable on Ctrl-C / kill: encoding and writes are interleaved in checkpoint
chunks of --checkpoint messages (default 256). Each chunk is written to Neo4j
before the next one is encoded, so an interrupted run loses at most one
chunk's worth of work — a re-run picks up at the first message still missing
an embedding.

Model: intfloat/multilingual-e5-large (1024-dim, multilingual — the corpus is
Chilean Spanish). e5 wants an instruction prefix: "passage:" for the documents
embedded here, "query:" for the search query (see the retriever for that side).

Usage:
    python scripts/embed_messages.py
    python scripts/embed_messages.py --batch 32 --reembed
    python scripts/embed_messages.py --checkpoint 512
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from _common import force_utf8, neo4j_driver

# Must match the `vector.dimensions` of the message_embedding index in
# load_neo4j.py. Changing the model means changing both and re-embedding.
MODEL_NAME = "intfloat/multilingual-e5-large"
EMBED_DIM = 1024


def hf_offline_if_cached() -> None:
    """If the embedding model is already in the local Hugging Face cache, run
    the HF stack fully offline — no Hub request (so no 'unauthenticated
    requests to the HF Hub' warning) and a faster, network-independent load.
    On a fresh machine where the model is not cached yet this is a no-op, so
    the first ~2GB download still works. Must run before sentence_transformers
    / huggingface_hub is imported — the offline flags are read at import time.
    """
    if os.environ.get("HF_HUB_OFFLINE"):
        return
    hf_home = os.environ.get("HF_HOME")
    hub = (Path(hf_home) / "hub" if hf_home
           else Path.home() / ".cache" / "huggingface" / "hub")
    if (hub / ("models--" + MODEL_NAME.replace("/", "--"))).is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Pending = messages still missing an embedding (or all of them, with
# --reembed). Each scoped subquery aggregates independently so attachments ×
# recipients don't form a cartesian product; collect(DISTINCT …) drops nulls,
# so a list is [] when the OPTIONAL MATCH finds nothing.
#
# Sender, recipients and the send month are pulled so build_text() can fold
# them into the passage: the From/To headers are NOT in body_clean, so without
# this a question like "what did X tell me" or "emails from Y last March" can
# never match semantically — the only place a correspondent's name lives is the
# header. Embedding them makes who/when first-class search signal.
FETCH_CYPHER = """
MATCH (m:Message)
WHERE $reembed OR m.embedding IS NULL
CALL (m) {
  OPTIONAL MATCH (m)-[:HAS_ATTACHMENT]->(a:Attachment)
  RETURN collect(DISTINCT a.filename) AS attachments
}
CALL (m) {
  OPTIONAL MATCH (s:Person)-[:SENT]->(m)
  RETURN s.name AS from_name, s.email AS from_email
}
CALL (m) {
  OPTIONAL MATCH (m)-[:RECEIVED_BY]->(r:Person)
  RETURN collect(DISTINCT coalesce(r.name, r.email)) AS recipients
}
RETURN m.gmail_message_id AS mid,
       m.account_owner    AS acct,
       m.subject          AS subject,
       m.body_clean       AS body,
       m.sent_at          AS sent_at,
       from_name, from_email, attachments, recipients
"""

# db.create.setNodeVectorProperty stores the list as a true vector value the
# index can use (available since Neo4j 5.13).
WRITE_CYPHER = """
UNWIND $rows AS row
MATCH (m:Message {gmail_message_id: row.mid, account_owner: row.acct})
CALL db.create.setNodeVectorProperty(m, 'embedding', row.embedding)
"""


def build_text(rec: dict) -> str:
    """Embedding input for one message: a compact From/To/Date header, then the
    subject, cleaned body, and real attachment filenames. e5 expects a
    'passage:' prefix on documents.

    The header goes first and is kept short (sender + a few recipients + the
    YYYY-MM month) so it costs little of e5's 512-token window but makes the
    correspondents and timeframe searchable — they aren't in body_clean."""
    parts: list[str] = []

    header: list[str] = []
    frm = (rec.get("from_name") or rec.get("from_email") or "").strip()
    if frm:
        header.append(f"From: {frm}")
    rcpts = [r.strip() for r in (rec.get("recipients") or []) if r and r.strip()]
    if rcpts:
        # Cap the recipient list so a 50-person distribution doesn't crowd out
        # the body; the first few carry the signal for "emails to X" queries.
        header.append("To: " + ", ".join(rcpts[:6]))
    month = (rec.get("sent_at") or "")[:7]  # YYYY-MM
    if month:
        header.append(f"Date: {month}")
    if header:
        parts.append(" | ".join(header))

    subject = (rec.get("subject") or "").strip()
    if subject:
        parts.append(subject)
    body = (rec.get("body") or "").strip()
    if body:
        parts.append(body)
    names = [n.strip() for n in (rec.get("attachments") or []) if n and n.strip()]
    if names:
        parts.append("Attachments: " + ", ".join(names))
    return "passage: " + "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=64,
                    help="Encoder mini-batch size (default 64). This is the "
                         "model.encode() batch — not the resume granularity.")
    ap.add_argument("--checkpoint", type=int, default=256,
                    help="Write to Neo4j after every N encoded messages "
                         "(default 256). Smaller = less work lost on Ctrl-C, "
                         "more transactions; larger = the opposite.")
    ap.add_argument("--reembed", action="store_true",
                    help="Recompute embeddings for ALL messages, not just "
                         "those still missing one.")
    args = ap.parse_args()

    # Force UTF-8 stdout so progress/summary lines survive a cp1252 console
    # and the pipe used when this runs as a subprocess of sync_incremental /
    # serve_app.
    force_utf8()

    driver = neo4j_driver()
    try:
        with driver.session() as session:
            rows = [dict(r) for r in session.run(FETCH_CYPHER,
                                                  reembed=args.reembed)]
        if not rows:
            print("Nothing to embed — every Message already has an embedding.")
            return

        print(f"Loading embedding model {MODEL_NAME} "
              f"(first run downloads ~2GB)…")
        hf_offline_if_cached()
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        # get_embedding_dimension() is the current name; fall back to the
        # older get_sentence_embedding_dimension() on pre-5.x installs.
        get_dim = (getattr(model, "get_embedding_dimension", None)
                   or model.get_sentence_embedding_dimension)
        dim = get_dim()
        if dim != EMBED_DIM:
            raise SystemExit(
                f"Model dimension {dim} != expected {EMBED_DIM}. Update "
                f"EMBED_DIM here and `vector.dimensions` in load_neo4j.py.")

        total = len(rows)
        chunk = max(1, args.checkpoint)
        print(f"Embedding {total:,} message(s) "
              f"(checkpoint every {chunk} → resumable on Ctrl-C)…")

        # Each session.run is an autocommit transaction, so a chunk that
        # finishes writing is durable even if the next encode/write is
        # interrupted. The fetch query above only returns Messages whose
        # embedding is null, so a re-run picks up exactly where we stopped.
        n_written = 0
        with driver.session() as session, \
                tqdm(total=total, desc="embed+write", unit="msg") as bar:
            for i in range(0, total, chunk):
                sub = rows[i:i + chunk]
                texts = [build_text(r) for r in sub]
                # normalize_embeddings=True → unit vectors, the form e5 + a
                # cosine index expect.
                vectors = model.encode(
                    texts, batch_size=args.batch,
                    normalize_embeddings=True, show_progress_bar=False)
                write_rows = [
                    {"mid": r["mid"], "acct": r["acct"],
                     "embedding": v.tolist()}
                    for r, v in zip(sub, vectors)
                ]
                session.run(WRITE_CYPHER, rows=write_rows)
                n_written += len(write_rows)
                bar.update(len(write_rows))
        print(f"Done — {n_written:,} Message embeddings written.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
