"""One-time migration to the treatment-tier ("bucket") model.

After the Phase-1 code change every NEW Message is tagged with `m.bucket`
('primary' for real correspondence, or a lite tier: promotions / social /
updates / forums / spam) at load time, and the embed/cluster/retrieval steps
skip non-'primary' buckets. But a graph built BEFORE that change has two
inconsistencies this script repairs — idempotently, so it's safe to re-run:

  1. Tag legacy nodes — every Message created before buckets existed has
     m.bucket = NULL. We derive the bucket from its Gmail label_ids using the
     same rule as _common.message_bucket() (spam first, then the categories).
     Only NULL buckets are touched, so live buckets maintained by the sync
     (apply_spam_changes) are never clobbered. Pass --retag-all to re-derive
     every Message from its labels (e.g. after fixing the rule).

  2. Strip lite embeddings — spam (and, after a re-pull, any categorized mail)
     may already carry an embedding from before lite was excluded, so it would
     still surface in semantic graph-RAG retrieval. We REMOVE m.embedding from
     every non-'primary' Message, evicting it from the vector index. Re-running
     embed_messages.py will NOT re-add them (it only embeds 'primary').

Neither step deletes a node or any real mail; it only sets a property and
removes vector embeddings that lite mail is not supposed to have. The full
pipeline never needs this — it's for an existing graph. Usage:

    python scripts/migrate_buckets.py --dry-run     # report only, change nothing
    python scripts/migrate_buckets.py               # interactive confirm, then migrate
    python scripts/migrate_buckets.py --yes         # non-interactive
    python scripts/migrate_buckets.py --retag-all   # re-derive bucket for ALL messages
"""
from __future__ import annotations

import argparse

from _common import bootstrap_venv, force_utf8, neo4j_driver

force_utf8()
bootstrap_venv()

BATCH = 10_000  # rows per inner transaction; keeps each commit bounded

# Cypher mirror of _common.message_bucket(): spam wins over a category label,
# then promotions > social > updates > forums, else primary. Kept in sync with
# that function and the identical CASE in sync_incremental.apply_spam_changes.
BUCKET_CASE = """CASE
        WHEN 'SPAM'               IN coalesce(m.label_ids, []) THEN 'spam'
        WHEN 'CATEGORY_PROMOTIONS' IN coalesce(m.label_ids, []) THEN 'promotions'
        WHEN 'CATEGORY_SOCIAL'     IN coalesce(m.label_ids, []) THEN 'social'
        WHEN 'CATEGORY_UPDATES'    IN coalesce(m.label_ids, []) THEN 'updates'
        WHEN 'CATEGORY_FORUMS'     IN coalesce(m.label_ids, []) THEN 'forums'
        ELSE 'primary' END"""

# A message is "lite" by its labels (derived bucket), independent of whether
# m.bucket has been set yet — so the strip count is honest in --dry-run, and the
# strip step works regardless of whether step 1 ran first.
LITE_PREDICATE = f"({BUCKET_CASE}) <> 'primary'"


def distribution(sess) -> list[dict]:
    """Current Message count grouped by bucket ('(null)' for untagged)."""
    return [dict(r) for r in sess.run(
        "MATCH (m:Message) "
        "RETURN coalesce(m.bucket, '(null)') AS bucket, count(*) AS n "
        "ORDER BY n DESC")]


def print_distribution(sess, title: str) -> None:
    print(f"\n{title}")
    rows = distribution(sess)
    total = sum(r["n"] for r in rows)
    for r in rows:
        print(f"    {r['bucket']:<12} {r['n']:>8,}")
    print(f"    {'TOTAL':<12} {total:>8,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change and stop. Makes no changes.")
    parser.add_argument("--retag-all", action="store_true",
                        help="Re-derive bucket for EVERY Message from its labels, "
                             "not just untagged (NULL) ones.")
    args = parser.parse_args()

    with neo4j_driver() as drv:
        with drv.session() as sess:
            print_distribution(sess, "Bucket distribution BEFORE:")

            tag_where = ("TRUE" if args.retag_all else "m.bucket IS NULL")
            to_tag = sess.run(
                f"MATCH (m:Message) WHERE {tag_where} "
                "RETURN count(m) AS n").single()["n"]
            to_strip = sess.run(
                "MATCH (m:Message) WHERE m.embedding IS NOT NULL "
                f"AND {LITE_PREDICATE} "
                "RETURN count(m) AS n").single()["n"]

            scope = "ALL messages" if args.retag_all else "untagged messages"
            print(f"\nPlanned changes:")
            print(f"  1. Tag {to_tag:,} {scope} with a derived bucket.")
            print(f"  2. Strip embeddings from {to_strip:,} lite message(s).")

            if args.dry_run:
                print("\nDry-run; nothing changed.")
                return
            if to_tag == 0 and to_strip == 0:
                print("\nNothing to do — graph already migrated.")
                return

            if not args.yes:
                print("\nThis sets m.bucket and removes lite embeddings (no nodes "
                      "or mail are deleted). Type 'yes' to proceed:")
                try:
                    resp = input("> ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    resp = ""
                if resp != "yes":
                    print("Aborted.")
                    return

            # Step 1 — tag. Batched so a large graph commits incrementally.
            sess.run(
                f"MATCH (m:Message) WHERE {tag_where} "
                f"CALL (m) {{ SET m.bucket = {BUCKET_CASE} }} "
                f"IN TRANSACTIONS OF {BATCH} ROWS").consume()
            print(f"  tagged {to_tag:,} message(s)")

            # Step 2 — evict lite from the vector index.
            sess.run(
                "MATCH (m:Message) WHERE m.embedding IS NOT NULL "
                f"AND {LITE_PREDICATE} "
                f"CALL (m) {{ REMOVE m.embedding }} "
                f"IN TRANSACTIONS OF {BATCH} ROWS").consume()
            print(f"  stripped {to_strip:,} lite embedding(s)")

            print_distribution(sess, "Bucket distribution AFTER:")

    print("\nBucket migration complete.")


if __name__ == "__main__":
    main()
