"""Cluster Gmail threads into Matter nodes (post-load step).

Why this exists
---------------
Gmail caps a single ``threadId`` at ~100 messages — longer conversations get
split into a fresh thread with a new ID. The same business matter also
sometimes changes subject mid-stream (e.g. "Promesa CompraVenta Terreno X"
later becomes "Compraventa Terreno X" once the promise is signed). Both
cases produce multiple Thread nodes in the graph for one logical matter.

This script clusters those threads back together and writes::

    (:Matter {canonical_key, canonical_subject,
              n_threads, n_messages,
              first_msg_at, last_msg_at})

with ``(t:Thread)-[:PART_OF]->(m:Matter)`` connecting members.

Algorithm
---------
1. For each Thread, compute
     - ``tokens``        = lowercased meaningful subject tokens
                           (after stripping Re:/Fwd: prefixes + stopwords)
     - ``participants``  = set of all sender + recipient emails

2. Bucket threads by every (token_a, token_b) pair — two threads can only
   end up in the same Matter if they share at least 2 meaningful tokens,
   so this bucketing is exhaustive without going O(n²).

3. Inside each bucket, union-find any pair satisfying BOTH:
     - subject token Jaccard >= ``--jaccard`` (default 0.7)
     - participant overlap (Szymkiewicz–Simpson) >= ``--overlap`` (default 0.5)

4. Connected components with >= ``--min-cluster-size`` threads (default 2)
   become Matter nodes.

Re-runnable: deletes all existing (:Matter) nodes (which cascades the
PART_OF edges) and rebuilds. Run it after load_neo4j.py, or any time the
graph changes (e.g. after sync_incremental.py).
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from itertools import combinations

from _common import neo4j_driver


# ---------------------------------------------------------------------------
# Subject normalization
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(
    r"^\s*(re|fwd?|rv|fw|enc|encaminhada|tr|wg|aw)\s*:\s*",
    re.IGNORECASE,
)

# Tokens dropped before computing similarity. Mostly Spanish/English filler.
# Months and years are deliberately KEPT as tokens — they are the signal that
# differentiates recurring monthly reports ("Cartola Septiembre 2025" vs
# "Cartola Octubre 2025") from genuine multi-thread matters.
_STOPWORDS = {
    # articles, prepositions, conjunctions (es)
    "de", "la", "el", "en", "y", "a", "para", "del", "los", "las", "un",
    "una", "se", "su", "sus", "por", "que", "es", "lo", "como", "con",
    "al", "le", "les", "este", "esta", "estos", "estas", "ese", "esa",
    # same in english
    "of", "the", "for", "and", "to", "on", "at", "from", "with", "in",
    "an", "by", "is", "as", "this", "that",
    # reply markers that occasionally leak through
    "fwd", "rv", "fw",
}


def normalize_subject(s: str) -> str:
    s = (s or "").lower()
    # Strip reply/forward prefixes repeatedly ("Re: Fwd: Re: X" → "X")
    while True:
        s2, n = _PREFIX_RE.subn("", s, count=1)
        if n == 0:
            break
        s = s2
    s = re.sub(r"\s+", " ", s).strip()
    return s


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ſ0-9]{3,}")


def subject_tokens(s: str) -> set[str]:
    """Return the set of meaningful tokens from a subject (>=3 chars, no stopwords)."""
    norm = normalize_subject(s)
    return {t.lower() for t in _TOKEN_RE.findall(norm) if t.lower() not in _STOPWORDS}


# Subjects matching these patterns produce technically-correct clusters
# (same-subject mass mail, calendar invite series, out-of-office auto-replies),
# but they aren't business matters, so we skip them when creating Matter
# nodes. Override or extend with --exclude-subject on the CLI.
_DEFAULT_EXCLUDE_PATTERNS = [
    r"^updated invitation:",          # Google Calendar updates
    r"^invitation:",                  # Google Calendar new invites
    r"^canceled event:",              # Google Calendar cancels
    r"^fuera de (la )?oficina",       # Spanish out-of-office
    r"^out of office",                # English out-of-office
    r"^respuesta autom[áa]tica:",     # Spanish auto-reply
    r"^automatic reply:",             # English auto-reply
    r"^documento tributario electr",  # DTE invoice spam
    r"jpm daily",                     # JP Morgan daily market newsletter
    r"^buenos d[íi]as!?\s*-\s*jpm",   # same newsletter, alt subject
]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def participant_overlap(a: set, b: set) -> float:
    """Szymkiewicz–Simpson coefficient: |a ∩ b| / min(|a|, |b|).

    Tolerant of asymmetric thread sizes — a 100-msg thread with 20 participants
    and a 5-msg follow-up with 3 of those same people still scores 1.0.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        # Path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

FETCH_THREADS_CYPHER = """
MATCH (t:Thread)
OPTIONAL MATCH (m:Message)-[:IN_THREAD]->(t)
OPTIONAL MATCH (sender:Person)-[:SENT]->(m)
OPTIONAL MATCH (m)-[:RECEIVED_BY]->(rcpt:Person)
WITH t,
     count(DISTINCT m) AS n_msgs,
     min(m.sent_at)    AS first_msg_at,
     max(m.sent_at)    AS last_msg_at,
     collect(DISTINCT sender.email) + collect(DISTINCT rcpt.email) AS people
RETURN t.gmail_thread_id  AS thread_id,
       t.account_owner    AS account_owner,
       t.subject          AS subject,
       n_msgs, first_msg_at, last_msg_at,
       [e IN people WHERE e IS NOT NULL] AS participants
"""

DROP_EXISTING = "MATCH (m:Matter) DETACH DELETE m"

WRITE_MATTERS = """
UNWIND $matters AS m
MERGE (mt:Matter {canonical_key: m.canonical_key})
  SET mt.canonical_subject = m.canonical_subject,
      mt.n_threads         = m.n_threads,
      mt.n_messages        = m.n_messages,
      mt.first_msg_at      = m.first_msg_at,
      mt.last_msg_at       = m.last_msg_at
WITH mt, m
UNWIND m.thread_keys AS tk
MATCH (t:Thread {gmail_thread_id: tk.thread_id, account_owner: tk.account_owner})
MERGE (t)-[:PART_OF]->(mt)
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jaccard", type=float, default=0.7,
                   help="Subject token Jaccard threshold (default 0.7). "
                        "Lower to catch more renamed-subject cases at the "
                        "risk of false positives.")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="Participant overlap threshold, Szymkiewicz-Simpson "
                        "(default 0.5).")
    p.add_argument("--min-cluster-size", type=int, default=2,
                   help="Only emit Matter nodes for clusters of >= this many "
                        "threads (default 2). Single-thread matters add no "
                        "information; skip them.")
    p.add_argument("--min-subject-tokens", type=int, default=3,
                   help="A thread needs >= this many meaningful subject tokens "
                        "to join a Matter (default 3). Generic 2-token subjects "
                        "like 'reunión [de esta] semana' score Jaccard 1.0 "
                        "against each other and falsely merged dozens of "
                        "unrelated threads into one mega-matter; requiring a "
                        "third content token kills those merges.")
    p.add_argument("--exclude-subject", action="append", default=None,
                   help="Regex (case-insensitive) for subject prefixes to skip. "
                        "Repeat for multiple. Default skips automated mail "
                        "(calendar invites, out-of-office, DTE). Pass empty "
                        "string to disable defaults.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be clustered without writing to the graph.")
    args = p.parse_args()

    if args.exclude_subject is None:
        exclude_patterns = _DEFAULT_EXCLUDE_PATTERNS
    elif args.exclude_subject == [""]:
        exclude_patterns = []
    else:
        exclude_patterns = args.exclude_subject
    exclude_res = [re.compile(p, re.IGNORECASE) for p in exclude_patterns]

    print(f"Target: {os.environ.get('NEO4J_URI', 'bolt://localhost:7687')}")
    print(f"  jaccard >= {args.jaccard}, "
          f"participant overlap >= {args.overlap}, "
          f"min cluster size >= {args.min_cluster_size}")

    with neo4j_driver() as drv:
        with drv.session() as sess:
            rows = list(sess.run(FETCH_THREADS_CYPHER))
        print(f"Fetched {len(rows):,} Thread nodes.")

        threads = []
        for r in rows:
            subj = r["subject"] or ""
            threads.append({
                "key":            (r["thread_id"], r["account_owner"]),
                "thread_id":      r["thread_id"],
                "account_owner":  r["account_owner"],
                "subject":        subj,
                "norm_subject":   normalize_subject(subj),
                "tokens":         subject_tokens(subj),
                "participants":   set(r["participants"] or []),
                "n_msgs":         r["n_msgs"] or 0,
                "first_msg_at":   r["first_msg_at"],
                "last_msg_at":    r["last_msg_at"],
            })

        # Bucket by token-pair so the O(n²) comparison shrinks to O(within-bucket²)
        buckets: defaultdict[tuple, list] = defaultdict(list)
        for thr in threads:
            if len(thr["tokens"]) < args.min_subject_tokens:
                # Threads with too few meaningful tokens are too generic to
                # anchor a matter — e.g. "reunión [de esta] semana" reduces to
                # {reunion, semana}, which scores Jaccard 1.0 against every
                # other such thread and merges dozens of unrelated meetings.
                # They remain singletons.
                continue
            for pair in combinations(sorted(thr["tokens"]), 2):
                buckets[pair].append(thr)

        uf = UnionFind()
        seen_pairs: set = set()
        edges_added = 0

        for bucket_threads in buckets.values():
            if len(bucket_threads) < 2:
                continue
            for a, b in combinations(bucket_threads, 2):
                pair = tuple(sorted((a["key"], b["key"])))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if jaccard(a["tokens"], b["tokens"]) < args.jaccard:
                    continue
                if participant_overlap(a["participants"], b["participants"]) < args.overlap:
                    continue
                uf.union(a["key"], b["key"])
                edges_added += 1

        # Group threads by union-find root. Singletons are handled here too.
        clusters: defaultdict[tuple, list] = defaultdict(list)
        for thr in threads:
            clusters[uf.find(thr["key"])].append(thr)

        # Build the Matter payload
        matters_payload: list[dict] = []
        skipped_by_filter = 0
        for members in clusters.values():
            if len(members) < args.min_cluster_size:
                continue
            members_sorted = sorted(members, key=lambda t: -len(t["norm_subject"]))
            canonical_subject = members_sorted[0]["subject"]
            canonical_key     = members_sorted[0]["norm_subject"]
            if not canonical_key:
                continue  # all-empty subjects — can't key the matter
            if any(r.search(canonical_subject) or r.search(canonical_key)
                   for r in exclude_res):
                skipped_by_filter += 1
                continue
            firsts = [t["first_msg_at"] for t in members if t["first_msg_at"]]
            lasts  = [t["last_msg_at"]  for t in members if t["last_msg_at"]]
            matters_payload.append({
                "canonical_key":     canonical_key,
                "canonical_subject": canonical_subject,
                "n_threads":         len(members),
                "n_messages":        sum(t["n_msgs"] for t in members),
                "first_msg_at":      min(firsts) if firsts else None,
                "last_msg_at":       max(lasts)  if lasts  else None,
                "thread_keys": [
                    {"thread_id": t["thread_id"], "account_owner": t["account_owner"]}
                    for t in members
                ],
            })

        # If two clusters happen to produce the same canonical_key (shouldn't
        # under union-find but is theoretically possible), collapse them so
        # MERGE doesn't overwrite properties.
        collapsed: dict[str, dict] = {}
        for m in matters_payload:
            k = m["canonical_key"]
            if k not in collapsed:
                collapsed[k] = m
            else:
                cur = collapsed[k]
                cur["thread_keys"].extend(m["thread_keys"])
                cur["n_threads"]  += m["n_threads"]
                cur["n_messages"] += m["n_messages"]
                firsts = [d for d in (cur["first_msg_at"], m["first_msg_at"]) if d]
                lasts  = [d for d in (cur["last_msg_at"],  m["last_msg_at"])  if d]
                cur["first_msg_at"] = min(firsts) if firsts else None
                cur["last_msg_at"]  = max(lasts)  if lasts  else None
        matters_payload = list(collapsed.values())

        print(f"Candidate edges checked: {len(seen_pairs):,}  (added: {edges_added:,})")
        print(f"Connected components:    {len(clusters):,}")
        print(f"Skipped by subject filter: {skipped_by_filter:,}")
        print(f"Multi-thread matters:    {len(matters_payload):,}")

        if matters_payload:
            top = sorted(matters_payload, key=lambda m: -m["n_messages"])[:10]
            print("\nTop 10 matters by message count:")
            for m in top:
                subj = m["canonical_subject"][:70].replace("\n", " ")
                print(f"  {m['n_messages']:5d} msgs  "
                      f"{m['n_threads']:2d} threads  {subj}")

        if args.dry_run:
            print("\n(--dry-run: graph not modified)")
            return

        with drv.session() as sess:
            sess.run(DROP_EXISTING)
            for i in range(0, len(matters_payload), 200):
                sess.run(WRITE_MATTERS, matters=matters_payload[i:i + 200])
        print(f"\nGraph updated: {len(matters_payload):,} Matter node(s) "
              f"+ {sum(m['n_threads'] for m in matters_payload):,} PART_OF edges.")


if __name__ == "__main__":
    main()
