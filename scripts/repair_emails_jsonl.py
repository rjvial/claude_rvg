"""Drop malformed lines from data/emails.jsonl. Backs up original to
emails.jsonl.bak.YYYYMMDD-HHMMSS. Removes the dropped messages' IDs from
the per-account resume tracker so the next pull re-fetches them.

Idempotent: re-running on a clean file is a no-op.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMAILS = DATA / "emails.jsonl"


def main() -> None:
    if not EMAILS.exists():
        print(f"No file at {EMAILS}")
        return

    good: list[str] = []
    dropped: list[tuple[int, str]] = []  # (line_no, reason)

    with EMAILS.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                dropped.append((i, f"json:{e}"))
                continue
            if not (obj.get("account_owner") and obj.get("message_id")):
                dropped.append((i, "missing account_owner or message_id"))
                continue
            good.append(line)

    print(f"total lines: {len(good) + len(dropped)}")
    print(f"kept:        {len(good)}")
    print(f"dropped:     {len(dropped)}")
    for ln, reason in dropped[:20]:
        print(f"  line {ln}: {reason}")

    if not dropped:
        print("Nothing to do.")
        return

    # Backup
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = EMAILS.with_suffix(f".jsonl.bak.{ts}")
    shutil.copy2(EMAILS, bak)
    print(f"backup: {bak.name}")

    # Rewrite
    with EMAILS.open("w", encoding="utf-8") as fh:
        fh.writelines(good)
    print(f"rewrote {EMAILS.name} with {len(good)} clean lines")


if __name__ == "__main__":
    main()
