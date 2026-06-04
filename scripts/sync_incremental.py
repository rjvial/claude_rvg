"""Incremental Gmail → Neo4j sync, one account at a time.

Uses Gmail's history API (cheap delta) when possible; falls back to a
date-based pull if the stored history_id has expired (~7 days). Then runs
clean → load → embed on whatever is new.

Run once per account (you typically chain three calls for the 3 mailboxes).

Usage:
    python scripts/sync_incremental.py --account gmail
    python scripts/sync_incremental.py --account work
    python scripts/sync_incremental.py --account org --skip-embed
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from googleapiclient.errors import HttpError
from tqdm import tqdm

from _common import (
    DATA_DIR,
    SCRIPTS,
    force_utf8,
    neo4j_driver,
    run_step,
)

force_utf8()

EMAILS_JSONL = DATA_DIR / "emails.jsonl"

sys.path.insert(0, str(SCRIPTS))
from pull_gmail import (  # noqa: E402
    SPAM_QUERY,
    query_suffix,
    _exec_with_retry,
    append_pulled_id,
    fetch_message,
    get_account_email,
    gmail_service,
    list_message_ids,
    load_pulled_ids,
    save_sync_state,
    sync_state_path,
)


def load_state(label: str) -> dict:
    p = sync_state_path(label)
    if not p.exists():
        raise SystemExit(
            f"No {p}. Run a full backfill with pull_gmail.py for this "
            f"account first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def history_delta(service, start_history_id: str
                  ) -> tuple[list[str], set[str], set[str],
                             set[str], set[str]]:
    """Returns (added_ids, became_read, became_unread, became_spam,
    became_not_spam) since start_history_id.

      added_ids       — message IDs newly added (messageAdded).
      became_read     — IDs whose UNREAD label was removed (read elsewhere).
      became_unread   — IDs that gained the UNREAD label (re-flagged unread).
      became_spam     — IDs that gained the SPAM label (marked spam elsewhere).
      became_not_spam — IDs whose SPAM label was removed (un-spammed elsewhere).

    For each label we track the net state across the window (last event for a
    message wins), so a message toggled more than once lands in one bucket.
    Deriving spam from history (not a raw in:spam diff) keeps purged spam from
    being resurfaced: Gmail's ~30-day auto-delete is a messagesDeleted event,
    not a SPAM-label removal, so it never lands in became_not_spam.

    Raises HttpError 404 if start_history_id is too old (~>7 days).
    """
    added: set[str] = set()
    unread_state: dict[str, bool] = {}     # mid -> is UNREAD after last event
    spam_state: dict[str, bool] = {}       # mid -> is SPAM after last event
    page_token: str | None = None
    while True:
        req = service.users().history().list(
            userId="me",
            startHistoryId=start_history_id,
            historyTypes=["messageAdded", "labelAdded", "labelRemoved"],
            pageToken=page_token,
            maxResults=500,
        )
        resp = _exec_with_retry(req)
        for h in resp.get("history", []) or []:
            for ma in h.get("messagesAdded", []) or []:
                mid = (ma.get("message") or {}).get("id")
                if mid:
                    added.add(mid)
            for la in h.get("labelsAdded", []) or []:
                labels = la.get("labelIds") or []
                mid = (la.get("message") or {}).get("id")
                if not mid:
                    continue
                if "UNREAD" in labels:
                    unread_state[mid] = True
                if "SPAM" in labels:
                    spam_state[mid] = True
            for lr in h.get("labelsRemoved", []) or []:
                labels = lr.get("labelIds") or []
                mid = (lr.get("message") or {}).get("id")
                if not mid:
                    continue
                if "UNREAD" in labels:
                    unread_state[mid] = False
                if "SPAM" in labels:
                    spam_state[mid] = False
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    became_unread = {mid for mid, u in unread_state.items() if u}
    became_read = {mid for mid, u in unread_state.items() if not u}
    became_spam = {mid for mid, s in spam_state.items() if s}
    became_not_spam = {mid for mid, s in spam_state.items() if not s}
    return list(added), became_read, became_unread, became_spam, became_not_spam


def apply_read_changes(label: str, became_read: set[str],
                       became_unread: set[str]) -> int:
    """Push read/unread label flips onto existing Message nodes so the graph
    matches Gmail after mail was read (or re-flagged) in another client. The
    WHERE guards keep already-correct nodes from counting, so the return value
    is the number of nodes that actually changed."""
    if not became_read and not became_unread:
        return 0
    drv = neo4j_driver()
    changed = 0
    try:
        with drv.session() as s:
            if became_read:
                res = s.run("""
                    UNWIND $mids AS mid
                    MATCH (m:Message {gmail_message_id: mid,
                                      account_owner: $acct})
                    WHERE 'UNREAD' IN coalesce(m.label_ids, [])
                    SET m.label_ids = [x IN m.label_ids WHERE x <> 'UNREAD']
                """, mids=list(became_read), acct=label).consume()
                changed += res.counters.properties_set
            if became_unread:
                res = s.run("""
                    UNWIND $mids AS mid
                    MATCH (m:Message {gmail_message_id: mid,
                                      account_owner: $acct})
                    WHERE NOT 'UNREAD' IN coalesce(m.label_ids, [])
                    SET m.label_ids = coalesce(m.label_ids, []) + 'UNREAD'
                """, mids=list(became_unread), acct=label).consume()
                changed += res.counters.properties_set
    finally:
        drv.close()
    return changed


def apply_spam_changes(label: str, became_spam: set[str],
                       became_not_spam: set[str]) -> int:
    """Push SPAM label flips onto existing Message nodes so the graph's spam
    view matches Gmail after a message was marked (or un-marked) spam in
    another client. Mirrors apply_read_changes: the WHERE guards count only
    nodes that actually changed. became_not_spam also restores INBOX, matching
    the app's own Not-spam action.

    m.bucket is kept in lock-step with the SPAM flip: a now-spam message becomes
    the 'spam' lite bucket; an un-spammed one falls back to its category bucket
    (or 'primary'), re-derived from the surviving labels — same rule as
    _common.message_bucket(). Without this, an un-spammed promo would keep
    bucket='spam' and stay lite forever."""
    if not became_spam and not became_not_spam:
        return 0
    drv = neo4j_driver()
    changed = 0
    try:
        with drv.session() as s:
            if became_not_spam:
                res = s.run("""
                    UNWIND $mids AS mid
                    MATCH (m:Message {gmail_message_id: mid,
                                      account_owner: $acct})
                    WHERE 'SPAM' IN coalesce(m.label_ids, [])
                    SET m.bucket = CASE
                            WHEN 'CATEGORY_PROMOTIONS' IN m.label_ids THEN 'promotions'
                            WHEN 'CATEGORY_SOCIAL'     IN m.label_ids THEN 'social'
                            WHEN 'CATEGORY_UPDATES'    IN m.label_ids THEN 'updates'
                            WHEN 'CATEGORY_FORUMS'     IN m.label_ids THEN 'forums'
                            ELSE 'primary' END,
                        m.label_ids =
                        [x IN m.label_ids
                           WHERE x <> 'SPAM' AND x <> 'INBOX'] + 'INBOX'
                """, mids=list(became_not_spam), acct=label).consume()
                changed += res.counters.properties_set
            if became_spam:
                res = s.run("""
                    UNWIND $mids AS mid
                    MATCH (m:Message {gmail_message_id: mid,
                                      account_owner: $acct})
                    WHERE NOT 'SPAM' IN coalesce(m.label_ids, [])
                    SET m.bucket = 'spam',
                        m.label_ids =
                        [x IN coalesce(m.label_ids, [])
                           WHERE x <> 'INBOX'] + 'SPAM'
                """, mids=list(became_spam), acct=label).consume()
                changed += res.counters.properties_set
    finally:
        drv.close()
    return changed


def new_msg_ids_via_date(service, since_ts_ms: int) -> list[str]:
    """Fallback: list messages after a given internal-date ms."""
    since = datetime.fromtimestamp(since_ts_ms / 1000, tz=timezone.utc)
    day = since.strftime("%Y/%m/%d")
    # Include categorized mail in the fallback re-list (history-gap recovery):
    # going forward the sync keeps promo/social/updates/forums as lite-tier
    # nodes, so this catch-up query must not exclude them either.
    query = f"after:{day} {query_suffix(include_categories=True)}"
    return list(list_message_ids(service, query))


def fetch_new_messages(args) -> tuple[list[dict], int]:
    """Returns (new_records, changed): the brand-new message records appended
    to emails.jsonl this run (so callers can clean/load just those instead of
    rescanning the whole file), and the count of existing graph nodes whose
    read/unread/spam label was revised. Either can be non-empty on its own."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    label = args.account
    state = load_state(label)
    service = gmail_service(label)
    account_email = get_account_email(service, label)

    new_ids: list[str] = []
    became_read: set[str] = set()
    became_unread: set[str] = set()
    became_spam: set[str] = set()
    became_not_spam: set[str] = set()
    use_history = state.get("last_history_id") and not args.force_date

    if use_history:
        try:
            (new_ids, became_read, became_unread,
             became_spam, became_not_spam) = history_delta(
                service, state["last_history_id"])
            print(f"history API ({label}) → {len(new_ids)} new, "
                  f"{len(became_read)} read + {len(became_unread)} unread, "
                  f"{len(became_spam)} spam + {len(became_not_spam)} not-spam "
                  f"label change(s).")
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status == 404:
                print(f"history_id expired for {label}, falling back to date pull.")
                use_history = False
            else:
                raise

    if not use_history:
        # Date fallback can only find new messages, not label changes on old
        # ones — the history window that carried those flips has expired.
        since_ms = state.get("last_internal_date_ms")
        if not since_ms:
            raise SystemExit(
                f"No last_internal_date_ms in {sync_state_path(label)}. "
                f"Run a backfill."
            )
        new_ids = new_msg_ids_via_date(service, since_ms)
        print(f"date pull ({label}) → {len(new_ids)} message IDs.")

    # Revise read/unread on EXISTING graph nodes. Exclude ids that are new
    # this run — those get their labels from the fresh load below. This is
    # what lets a Sync reflect mail you read or re-flagged in another client.
    new_set = set(new_ids)
    changed = apply_read_changes(label, became_read - new_set,
                                 became_unread - new_set)
    if changed:
        print(f"  Revised {changed} read/unread label(s) for {label}.")

    # Reconcile SPAM on EXISTING graph nodes from the same history window —
    # mark/un-mark spam that happened in Gmail (or another client). Brand-new
    # spam (in new_set) is excluded: it gets its SPAM label from the fresh
    # load below. Derived from history, so Gmail's 30-day spam purge (a
    # messagesDeleted event) never lands here and can't resurface a message.
    spam_changed = apply_spam_changes(label, became_spam - new_set,
                                      became_not_spam - new_set)
    if spam_changed:
        print(f"  Reconciled {spam_changed} spam label(s) for {label}.")
        changed += spam_changed

    # Spam is invisible to both the history API delta and the date fallback
    # (includeSpamTrash defaults off), so list it explicitly each sync. The
    # per-id dedup below skips spam already pulled; only fresh spam is fetched.
    try:
        spam_ids = list(list_message_ids(service, SPAM_QUERY,
                                         include_spam_trash=True))
        if spam_ids:
            new_ids = list(dict.fromkeys(list(new_ids) + spam_ids))
    except HttpError as e:
        print(f"  spam list failed ({label}): {e}", file=sys.stderr)

    pulled = load_pulled_ids(label)
    to_fetch = [i for i in new_ids if i not in pulled]
    print(f"After dedup ({label}): {len(to_fetch)} truly new messages.")

    if not to_fetch:
        return [], changed

    # history_id and internal_date_ms must track the SAME (newest) message;
    # see the matching note in pull_gmail.py.
    latest_history_id: str | None = state.get("last_history_id")
    latest_internal_ms: int | None = state.get("last_internal_date_ms")

    skipped_drafts = 0
    new_records: list[dict] = []
    with EMAILS_JSONL.open("a", encoding="utf-8") as out:
        for mid in tqdm(to_fetch, desc=f"fetch[{label}]"):
            try:
                rec = fetch_message(service, mid, label, account_email)
            except HttpError as e:
                print(f"  skip {mid}: {e}", file=sys.stderr)
                continue
            rec_labels = rec.get("label_ids") or []
            if "DRAFT" in rec_labels:
                # The history API returns every messageAdded event regardless
                # of label, so drafts arrive here even though pull_gmail's
                # query excludes them. Mark as pulled, don't persist the body.
                append_pulled_id(label, mid)
                skipped_drafts += 1
                continue
            # Categorized mail (promo/social/updates/forums) is NO LONGER
            # dropped — it's kept as a lite-tier node, tagged at load time by
            # message_bucket(). So the incremental path now persists it like any
            # other message; load_neo4j sets m.bucket and the downstream
            # embed/cluster/retrieval steps skip non-'primary' buckets.
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            append_pulled_id(label, mid)
            new_records.append(rec)
            ms = rec.get("internal_date_ms")
            if ms and (latest_internal_ms is None or ms > latest_internal_ms):
                latest_internal_ms = ms
                latest_history_id = rec.get("history_id") or latest_history_id

    if skipped_drafts:
        print(f"  Skipped {skipped_drafts} draft message(s) for {label}.")
    save_sync_state(label, latest_history_id, latest_internal_ms)
    return new_records, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", type=str, required=True,
                        help="Account label (must match a token_<label>.json).")
    parser.add_argument("--force-date", action="store_true",
                        help="Skip history API, use date-based pull.")
    parser.add_argument("--skip-pull", action="store_true",
                        help="Skip the pull step (assume emails.jsonl is fresh).")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip the graph-RAG embedding step.")
    args = parser.parse_args()

    records: list[dict] = []
    if not args.skip_pull:
        records, changed = fetch_new_messages(args)
        if not records:
            # Label changes (if any) already landed in Neo4j inside
            # fetch_new_messages; clean/load/embed only matter for new bodies.
            print(f"Revised {changed} label(s); no new mail."
                  if changed else "Nothing to do.")
            return

    py = sys.executable
    # With records in hand (the normal pull path), clean + load just those —
    # skips rescanning all ~40k messages, which the full scripts cost ~3 min.
    # --skip-pull has no records, so fall back to the full rescan scripts.
    if not args.skip_clean:
        if records:
            import clean_bodies
            n = clean_bodies.clean_records(records)
            print(f"clean: appended {n} new record(s) to emails_clean.jsonl.")
        else:
            run_step("clean_bodies", [py, str(SCRIPTS / "clean_bodies.py")])
    if not args.skip_load:
        if records:
            import load_neo4j
            load_neo4j.fast_load_records(records)
            print(f"load: fast-loaded {len(records)} new message(s) "
                  f"(Layer 1 + REPLY_TO; backbone/orgs/events backfill on the "
                  f"next run_pipeline).")
        else:
            run_step("load_neo4j", [py, str(SCRIPTS / "load_neo4j.py")])
    if not args.skip_embed:
        # Embeds only the just-loaded messages (those still without an
        # embedding) — cheap on an incremental run.
        run_step("embed_messages", [py, str(SCRIPTS / "embed_messages.py")])

    print(f"\nIncremental sync complete for account={args.account}.")


if __name__ == "__main__":
    main()
