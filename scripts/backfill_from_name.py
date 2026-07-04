"""One-time backfill: populate Message.from_name on existing graph nodes.

Message nodes used to carry no sender display name — the From name lived only
on the email-keyed Person node, so senders sharing one address under many
display names (Docusign, no-reply bots, ticketing systems) all collapsed to a
single name. load_neo4j now writes m.from_name per message; this script
retro-fills the property onto already-loaded nodes by streaming the per-message
names straight out of emails_clean.jsonl.

Idempotent: re-running just re-sets the same values. Safe to interrupt.

    python scripts/backfill_from_name.py
"""
from __future__ import annotations

import json
import sys

from _common import DATA_DIR, force_utf8, neo4j_driver

force_utf8()

EMAILS_CLEAN_JSONL = DATA_DIR / "emails_clean.jsonl"

SET_FROM_NAME_CYPHER = """
UNWIND $rows AS row
MATCH (m:Message {gmail_message_id: row.mid, account_owner: row.acct})
SET m.from_name = row.from_name
"""


def main() -> None:
    if not EMAILS_CLEAN_JSONL.exists():
        raise SystemExit(f"No {EMAILS_CLEAN_JSONL}.")

    drv = neo4j_driver()
    batch: list[dict] = []
    total = 0
    set_count = 0

    def flush() -> None:
        nonlocal set_count
        if not batch:
            return
        with drv.session() as s:
            res = s.run(SET_FROM_NAME_CYPHER, rows=batch).consume()
            set_count += res.counters.properties_set
        batch.clear()

    try:
        with EMAILS_CLEAN_JSONL.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                from_ = rec.get("from") or {}
                name = from_.get("name")
                mid = rec.get("message_id")
                acct = rec.get("account_owner")
                if not mid or not acct or not name:
                    continue
                batch.append({"mid": mid, "acct": acct, "from_name": name})
                total += 1
                if len(batch) >= 5000:
                    flush()
                    print(f"  …{total:,} rows processed", flush=True)
        flush()
    finally:
        drv.close()

    print(f"Backfill done: {total:,} records with a sender name, "
          f"{set_count:,} Message.from_name properties set.")


if __name__ == "__main__":
    main()
