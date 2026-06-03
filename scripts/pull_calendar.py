"""Pull calendar events from a single Google account via the Calendar API.

Mirror of pull_gmail.py: multi-account aware, every record tagged with
`account_owner: <label>`, per-account resume/state files, one shared
`credentials.json`. OAuth scopes + auth helpers are imported from pull_gmail
so a single consent flow yields one token_<label>.json valid for both APIs.

Cross-account dedup is intentionally NOT done here. The same meeting appears
once per attendee calendar with a different per-calendar `event_id` but a
stable `ical_uid`. We keep all copies (tagged by account_owner) and collapse
on `ical_uid` at graph-load time, so provenance is preserved in the raw data.

Layout:
    data/credentials.json               ← shared Desktop OAuth client JSON
    data/token_<label>.json             ← per-account token (shared w/ gmail)
    data/pulled_event_ids_<label>.txt   ← per-account resume tracker
    data/sync_state_<label>.json        ← shared state file; this script adds
                                          calendar_sync_token / _full_sync_at
    data/calendar_events.jsonl          ← shared output; account_owner tagged

Usage:
    python scripts/pull_calendar.py --account gmail --auth
    python scripts/pull_calendar.py --account gmail --since 2024-05-14
    python scripts/pull_calendar.py --account org --since 2024-05-14 --limit 50
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from googleapiclient.discovery import build
from tqdm import tqdm

# Reuse pull_gmail's OAuth (shared SCOPES → shared token) and retry/state
# helpers so auth logic and the scope list stay single-sourced.
from pull_gmail import (
    DATA_DIR,
    _exec_with_retry,
    load_credentials,
    run_auth_flow,
    sync_state_path,
)

CALENDAR_EVENTS_JSONL = DATA_DIR / "calendar_events.jsonl"


def pulled_event_ids_path(label: str) -> Path:
    return DATA_DIR / f"pulled_event_ids_{label}.txt"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def calendar_service(label: str):
    creds = load_credentials(label)
    if not creds:
        creds = run_auth_flow(label)
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def primary_calendar_email(service) -> str:
    """The 'primary' calendar's id IS the account's email address. Used to set
    the `is_self` flags so the graph can tell the account owner apart from
    other attendees without a separate Gmail profile call.

    Deliberately does NOT swallow errors: this is the first Calendar API call,
    so a disabled-API / bad-scope / consent problem surfaces here with Google's
    own actionable message instead of silently degrading is_self and failing
    one call later with a confusing (unknown) account."""
    cal = _exec_with_retry(service.calendars().get(calendarId="primary"))
    return (cal.get("id") or "").lower()


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_events(service, time_min: str, time_max: str) -> tuple[list[dict], str | None]:
    """Window-list expanded event instances in [time_min, time_max].

    singleEvents=True expands recurring series into individual occurrences —
    each occurrence is a real meeting for graph purposes, and the window keeps
    the expansion bounded. Returns (events, next_sync_token); the sync token
    from the final page seeds future incremental pulls (pull side: a separate
    syncToken-only path, analogous to sync_incremental.py for mail)."""
    events: list[dict] = []
    page_token: str | None = None
    next_sync_token: str | None = None
    while True:
        req = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            maxResults=2500,
            pageToken=page_token,
        )
        resp = _exec_with_retry(req)
        events.extend(resp.get("items", []) or [])
        next_sync_token = resp.get("nextSyncToken") or next_sync_token
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return events, next_sync_token


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _person(obj: dict | None, self_email: str) -> dict | None:
    if not obj:
        return None
    email = (obj.get("email") or "").lower()
    if not email:
        return None
    return {
        "email": email,
        "name": obj.get("displayName") or None,
        "is_self": bool(obj.get("self")) or (bool(self_email) and email == self_email),
    }


def _when(obj: dict | None) -> dict:
    """Normalize a Calendar start/end. Timed events use `dateTime`; all-day
    events use `date`. We keep the raw ISO plus a UTC epoch-ms for ordering
    and an all_day flag so the loader doesn't re-parse."""
    obj = obj or {}
    dt = obj.get("dateTime")
    d = obj.get("date")
    iso = dt or d
    all_day = dt is None and d is not None
    ms: int | None = None
    if iso:
        try:
            s = iso.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(s)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            ms = int(parsed.timestamp() * 1000)
        except ValueError:
            ms = None
    return {"iso": iso, "epoch_ms": ms, "all_day": all_day,
            "time_zone": obj.get("timeZone")}


def normalize_event(ev: dict, account_owner: str, self_email: str) -> dict:
    attendees = [
        {
            **(_person(a, self_email) or {"email": None, "name": None, "is_self": False}),
            "response_status": a.get("responseStatus"),
            "optional": bool(a.get("optional")),
            "organizer": bool(a.get("organizer")),
            "resource": bool(a.get("resource")),
        }
        for a in (ev.get("attendees") or [])
        if (a.get("email") or "").strip()
    ]
    return {
        "account_owner": account_owner,
        "event_id": ev.get("id"),
        # Stable across calendars — the cross-account collapse key at load time.
        "ical_uid": ev.get("iCalUID"),
        "recurring_event_id": ev.get("recurringEventId"),
        "calendar_id": "primary",
        "status": ev.get("status"),
        "summary": ev.get("summary") or "",
        "description": ev.get("description") or "",
        "location": ev.get("location") or "",
        "start": _when(ev.get("start")),
        "end": _when(ev.get("end")),
        "organizer": _person(ev.get("organizer"), self_email),
        "creator": _person(ev.get("creator"), self_email),
        "attendees": attendees,
        "hangout_link": ev.get("hangoutLink"),
        "html_link": ev.get("htmlLink"),
        "created": ev.get("created"),
        "updated": ev.get("updated"),
    }


# ---------------------------------------------------------------------------
# Resume / state
# ---------------------------------------------------------------------------

def load_pulled_ids(label: str) -> set[str]:
    p = pulled_event_ids_path(label)
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}


def append_pulled_id(label: str, eid: str) -> None:
    with pulled_event_ids_path(label).open("a", encoding="utf-8") as fh:
        fh.write(eid + "\n")


def save_calendar_sync_state(label: str, sync_token: str | None) -> None:
    """Add calendar keys to the shared sync_state_<label>.json without
    clobbering the gmail keys already in it."""
    p = sync_state_path(label)
    cur: dict = {}
    if p.exists():
        cur = json.loads(p.read_text(encoding="utf-8"))
    if sync_token:
        cur["calendar_sync_token"] = sync_token
    cur["calendar_full_sync_at"] = datetime.now(timezone.utc).isoformat()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _to_rfc3339(day: str, end_of_day: bool = False) -> str:
    t = "23:59:59" if end_of_day else "00:00:00"
    return f"{day}T{t}Z"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", type=str, required=True,
                        help="Account label (gmail, work, org). "
                             "Determines per-account token/state/resume files.")
    parser.add_argument("--auth", action="store_true",
                        help="Run OAuth handshake for this account and exit.")
    parser.add_argument("--since", type=str, default=None,
                        help="Backfill window start, YYYY-MM-DD (timeMin). "
                             "Match the email backfill window.")
    parser.add_argument("--until", type=str, default=None,
                        help="Window end YYYY-MM-DD (timeMax). Default: now.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after writing N new events.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.auth:
        run_auth_flow(args.account)
        return

    if not args.since:
        parser.error("Provide --since YYYY-MM-DD (match the email window).")

    time_min = _to_rfc3339(args.since)
    time_max = (
        _to_rfc3339(args.until, end_of_day=True) if args.until
        else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    print(f"Window: {time_min} .. {time_max}")

    service = calendar_service(args.account)
    self_email = primary_calendar_email(service)
    print(f"Account email for {args.account}: {self_email or '(unknown)'}")

    pulled = load_pulled_ids(args.account)
    print(f"Already pulled for account={args.account}: {len(pulled)}.")

    events, sync_token = list_events(service, time_min, time_max)
    to_write = [e for e in events if e.get("id") and e["id"] not in pulled]
    print(f"Events in window: {len(events)}; new: {len(to_write)}.")
    if args.limit:
        to_write = to_write[:args.limit]
        print(f"  Capped to --limit {args.limit}: {len(to_write)}.")

    n = 0
    with CALENDAR_EVENTS_JSONL.open("a", encoding="utf-8") as out:
        for ev in tqdm(to_write, desc=f"events[{args.account}]"):
            rec = normalize_event(ev, args.account, self_email)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            append_pulled_id(args.account, ev["id"])
            n += 1

    save_calendar_sync_state(args.account, sync_token)
    print(f"Done. Wrote {n} new events for account={args.account}.", file=sys.stderr)


if __name__ == "__main__":
    main()
