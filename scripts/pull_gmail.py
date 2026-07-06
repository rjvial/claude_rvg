"""Pull mail from a single Google account via the Gmail API.

Multi-account aware: every record is tagged with `account_owner: <label>`, and
each account has its own per-label credentials/tokens/state. One shared
`credentials.json` (Desktop OAuth client) authorizes all of them; the per-
account `token_<label>.json` is created on first auth.

Layout:
    data/credentials.json            ← shared Desktop OAuth client JSON
    data/token_<label>.json          ← per-account refresh token (created)
    data/pulled_msg_ids_<label>.txt  ← per-account resume tracker
    data/sync_state_<label>.json     ← {last_history_id, last_internal_date_ms}
    data/emails.jsonl                ← shared output; each record has
                                       account_owner: <label>

Usage:
    python scripts/pull_gmail.py --account gmail --auth
    python scripts/pull_gmail.py --account gmail --since 2026-04-13
    python scripts/pull_gmail.py --account work --since 2025-05-13
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

from _attachments import classify_attachment
from _common import EXCLUDED_CATEGORIES

# Single source of truth for OAuth scopes. pull_calendar.py and the compose
# path in serve_app.py both import this same list so one consent flow yields
# one token_<label>.json valid for the Gmail read/compose APIs and Calendar.
# Editing this list invalidates every existing token (Google requires
# re-consent on any scope change) — re-run `--auth` per account after a change.
#
# The full-mailbox scope supersedes gmail.readonly + gmail.compose +
# gmail.modify, and is the ONLY scope Google accepts for the Trash view's
# explicit "Delete forever" (users.messages.delete / batchDelete). The app's
# default "remove" stays Gmail-standard (Trash, recoverable 30 days, then
# auto-purged) — only that opt-in button deletes permanently.
# Tokens consented under the old granular scopes keep working for everything
# except Delete forever (load_credentials doesn't force this list onto stored
# tokens); re-run `--auth` per account to unlock it.
SCOPES = [
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/calendar.readonly",
]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CREDENTIALS_PATH = DATA_DIR / "credentials.json"
EMAILS_JSONL = DATA_DIR / "emails.jsonl"

# Drop drafts at the source. Each compose-window autosave creates its own
# DRAFT-labeled Message with a distinct gmail_message_id, so without this filter
# the graph ends up with N draft copies alongside the eventual sent message. The
# history API in sync_incremental ignores this query, so a belt-and-suspenders
# DRAFT label check in the fetch loop catches anything that slips through.
_DRAFT_FILTER = "-in:draft"


def query_suffix(include_categories: bool = False) -> str:
    """Trailing Gmail-query filters for a date backfill. By default the
    promo/social/updates/forums auto-categories are excluded, so the
    full-history backfill stays category-free. Pass include_categories=True to
    drop the -category: clauses and fetch categorized mail too — used by the
    bounded `--include-categories` pull and by the incremental sync, which now
    keeps categorized mail as lite-tier nodes (see _common.message_bucket).
    Drafts are always excluded."""
    cats = ("" if include_categories
            else " ".join(f"-category:{c}" for c in EXCLUDED_CATEGORIES) + " ")
    return cats + _DRAFT_FILTER


# Spam is excluded from the normal pull (includeSpamTrash defaults off). Listed
# separately so the app's Spam page mirrors Gmail. Scoped to in:spam so the
# includeSpamTrash=True flag can't pull Trash. Bounded — Gmail purges spam ~30d.
SPAM_QUERY = "in:spam"


def token_path(label: str) -> Path:
    return DATA_DIR / f"token_{label}.json"


def pulled_ids_path(label: str) -> Path:
    return DATA_DIR / f"pulled_msg_ids_{label}.txt"


def sync_state_path(label: str) -> Path:
    return DATA_DIR / f"sync_state_{label}.json"


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------

def load_credentials(label: str) -> Credentials | None:
    p = token_path(label)
    if not p.exists():
        return None
    # No scopes arg: the credential keeps whatever scopes the user actually
    # consented to. Passing SCOPES would make refresh raise once this list
    # grows beyond an old token's grant (Google never upscopes on refresh),
    # bricking sync until every account re-consents. Instead, old tokens keep
    # working and only the calls needing a new scope fail (as 403s with a
    # reconnect hint) until that account re-runs --auth.
    creds = Credentials.from_authorized_user_file(str(p))
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        save_credentials(label, creds)
        return creds
    return None


def save_credentials(label: str, creds: Credentials) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    token_path(label).write_text(creds.to_json(), encoding="utf-8")


def run_auth_flow(label: str, timeout_seconds: int | None = None) -> Credentials:
    """Run the interactive OAuth consent flow for one account label.

    timeout_seconds bounds how long run_local_server waits for the browser
    redirect. The terminal path leaves it None (wait indefinitely), but the
    app path (serve_app POST /api/auth) passes a finite value so an abandoned
    consent — user closed the tab — doesn't leave the local redirect server
    (and serve_app's sign-in lock) hung forever."""
    if not CREDENTIALS_PATH.exists():
        raise SystemExit(
            f"Missing {CREDENTIALS_PATH}. Download a Desktop OAuth client "
            "JSON from Google Cloud Console and save it there."
        )
    flow = InstalledAppFlow.from_client_secrets_file(
        str(CREDENTIALS_PATH), SCOPES
    )
    print(
        f"Opening browser for OAuth (account label={label}). Sign in with "
        f"the Google account whose mail you want to graph for this label."
    )
    creds = flow.run_local_server(port=0, timeout_seconds=timeout_seconds)
    if not creds:
        # Older google-auth-oauthlib returns falsy instead of raising on
        # timeout; normalize so callers see a clear failure.
        raise RuntimeError("OAuth flow did not complete (timed out or "
                           "was cancelled before consent).")
    save_credentials(label, creds)
    # The cached account_email in sync_state must NOT outlive the token it
    # describes — a re-auth could land on a different Google account (it has
    # happened: tokens get swapped between labels) and a stale cache would
    # then misroute outgoing mail. Drop the cache; get_account_email refetches
    # it via getProfile on next use, so the cache always agrees with the token.
    p = sync_state_path(label)
    if p.exists():
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
            if state.pop("account_email", None) is not None:
                p.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        except (json.JSONDecodeError, OSError):
            pass
    print(f"OK. Token saved to {token_path(label)}.")
    return creds


def gmail_service(label: str):
    creds = load_credentials(label)
    if not creds:
        creds = run_auth_flow(label)
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def _exec_with_retry(req, max_tries: int = 5):
    delay = 1.0
    for attempt in range(max_tries):
        try:
            return req.execute()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in (429, 500, 502, 503, 504) and attempt < max_tries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def list_message_ids(service, query: str,
                     include_spam_trash: bool = False) -> Iterable[str]:
    # includeSpamTrash defaults to False on the Gmail API, which silently drops
    # every SPAM/TRASH message from the results. Pass True only with a query
    # that scopes to spam (e.g. "in:spam"), so Trash is never resurrected.
    page_token: str | None = None
    while True:
        req = service.users().messages().list(
            userId="me",
            q=query,
            pageToken=page_token,
            maxResults=500,
            includeSpamTrash=include_spam_trash,
        )
        resp = _exec_with_retry(req)
        for msg in resp.get("messages", []) or []:
            yield msg["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def _b64url_decode(s: str | None) -> str:
    if not s:
        return ""
    pad = "=" * (-len(s) % 4)
    raw = base64.urlsafe_b64decode(s + pad)
    return raw.decode("utf-8", errors="replace")


def _extract_bodies(payload: dict) -> tuple[str, str, list[dict]]:
    """Walk MIME parts. Return (plain, html, attachments)."""
    plain = ""
    html = ""
    attachments: list[dict] = []

    def walk(part: dict) -> None:
        nonlocal plain, html
        mime = part.get("mimeType", "")
        body = part.get("body", {}) or {}
        filename = part.get("filename") or ""
        if filename:
            # Classify with per-part headers so we can tell an inline HTML
            # signature image (Content-Disposition:inline / Content-ID:cid)
            # apart from a real user attachment that happens to share the
            # image001.png naming. Headers aren't stored in jsonl — only the
            # resulting `kind` — so the load path can re-classify cheaply.
            part_headers = part.get("headers") or []
            kind = classify_attachment(
                filename=filename,
                mime_type=mime or None,
                headers=part_headers,
            )
            attachments.append({
                "filename": filename,
                "mime_type": mime or None,
                "size": body.get("size"),
                "attachment_id": body.get("attachmentId"),
                "part_id": part.get("partId") or "",
                "kind": kind,
            })
        if mime == "text/plain" and not plain:
            plain = _b64url_decode(body.get("data"))
        elif mime == "text/html" and not html:
            html = _b64url_decode(body.get("data"))
        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(payload)
    return plain, html, attachments


def _parse_address_list(header_value: str | None) -> list[dict]:
    """Parse a To/Cc/Bcc header into a list of {email, name?}."""
    if not header_value:
        return []
    from email.utils import getaddresses
    parsed = getaddresses([header_value])
    out: list[dict] = []
    for name, email in parsed:
        if not email:
            continue
        out.append({"email": email.lower(), "name": name or None})
    return out


def _parse_single_address(header_value: str | None) -> dict | None:
    parsed = _parse_address_list(header_value)
    return parsed[0] if parsed else None


def _parse_rfc822_msgid(header_value: str | None) -> str | None:
    """Strip the angle brackets off an RFC-822 Message-ID header value.
    Returns None if absent or empty after trimming."""
    if not header_value:
        return None
    v = header_value.strip()
    if v.startswith("<") and v.endswith(">"):
        v = v[1:-1]
    return v or None


def _parse_references(header_value: str | None) -> list[str]:
    """Parse a References header into an ordered list of message-ids
    (oldest → newest, per RFC 5322), each with its angle brackets stripped.
    The list is the ancestor chain; load_neo4j walks it from the newest end
    to attach a reply to its nearest in-corpus ancestor when the direct
    In-Reply-To parent is missing."""
    if not header_value:
        return []
    out: list[str] = []
    for tok in header_value.replace("\n", " ").replace("\t", " ").split():
        tok = tok.strip()
        if tok.startswith("<") and tok.endswith(">"):
            tok = tok[1:-1]
        if tok:
            out.append(tok)
    return out


def gmail_message_url(
    account_email: str | None,
    gmail_message_id: str,
) -> str | None:
    """Per-message Gmail deep-link — opens the message directly in All Mail.

    `#all/<gmail_message_id>` routes the UI straight to the message in its
    thread. The account is selected with ?authuser=<email>: the /u/<N>/ path
    slot only accepts a numeric account index, and an email there yields
    Gmail's "Temporary Error (404) … account temporarily unavailable".
    """
    if not account_email:
        return None
    return (
        f"https://mail.google.com/mail/?authuser={account_email}"
        f"#all/{gmail_message_id}"
    )


def fetch_message(service, mid: str, account_owner: str,
                  account_email: str | None = None) -> dict:
    req = service.users().messages().get(
        userId="me", id=mid, format="full"
    )
    raw = _exec_with_retry(req)

    payload = raw.get("payload") or {}
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", []) or []}
    body_plain, body_html, attachments = _extract_bodies(payload)

    internal_ms = int(raw.get("internalDate") or 0)
    sent_at = (
        datetime.fromtimestamp(internal_ms / 1000, tz=timezone.utc).isoformat()
        if internal_ms else None
    )

    rfc822_id = _parse_rfc822_msgid(headers.get("message-id"))
    rec = {
        "account_owner": account_owner,
        "message_id": raw["id"],
        "thread_id": raw["threadId"],
        "history_id": raw.get("historyId"),
        "rfc822_message_id": rfc822_id,
        # RFC 5322 threading headers — the reply tree is derived from these
        # at load time (load_neo4j.load_reply_edges). in_reply_to is the
        # direct parent; references is the full ancestor chain (fallback).
        "in_reply_to": _parse_rfc822_msgid(headers.get("in-reply-to")),
        "references": _parse_references(headers.get("references")),
        "gmail_url": gmail_message_url(account_email, raw["id"]),
        "from": _parse_single_address(headers.get("from")),
        "to": _parse_address_list(headers.get("to")),
        "cc": _parse_address_list(headers.get("cc")),
        "bcc": _parse_address_list(headers.get("bcc")),
        "subject": headers.get("subject", ""),
        "snippet": raw.get("snippet", ""),
        "sent_at": sent_at,
        "internal_date_ms": internal_ms,
        "body_plain": body_plain,
        "body_html": body_html,
        "attachments": attachments,
        # Reflects REAL attachments only — inline signature images and crypto
        # blobs don't count toward "this message has attachments" semantics.
        "has_attachments": any(a.get("kind") == "attachment" for a in attachments),
        "label_ids": raw.get("labelIds", []),
    }
    return rec


# ---------------------------------------------------------------------------
# Resume / state
# ---------------------------------------------------------------------------

def load_pulled_ids(label: str) -> set[str]:
    p = pulled_ids_path(label)
    if not p.exists():
        return set()
    return {
        line.strip()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_pulled_id(label: str, mid: str) -> None:
    with pulled_ids_path(label).open("a", encoding="utf-8") as fh:
        fh.write(mid + "\n")


def save_sync_state(label: str, history_id: str | None,
                    internal_date_ms: int | None) -> None:
    cur: dict = {}
    p = sync_state_path(label)
    if p.exists():
        cur = json.loads(p.read_text(encoding="utf-8"))
    if history_id:
        cur["last_history_id"] = history_id
    if internal_date_ms:
        cur["last_internal_date_ms"] = internal_date_ms
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")


def get_account_email(service, label: str) -> str:
    """Return the account email, caching in sync_state_<label>.json."""
    p = sync_state_path(label)
    state: dict = {}
    if p.exists():
        state = json.loads(p.read_text(encoding="utf-8"))
    cached = state.get("account_email")
    if cached:
        return cached
    profile = _exec_with_retry(service.users().getProfile(userId="me"))
    email = (profile.get("emailAddress") or "").lower()
    if not email:
        return ""
    state["account_email"] = email
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return email


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", type=str, required=True,
                        help="Account label (e.g. gmail, work, org). "
                        "Determines per-account token/state/resume files.")
    parser.add_argument("--auth", action="store_true",
                        help="Run OAuth handshake for this account and exit.")
    parser.add_argument("--since", type=str, default=None,
                        help="Gmail-style after date YYYY-MM-DD.")
    parser.add_argument("--query", type=str, default=None,
                        help="Override the Gmail search query entirely.")
    parser.add_argument("--include-categories", action="store_true",
                        help="Drop the -category: exclusions so promo/social/"
                        "updates/forums mail is fetched too (lite tier). Use "
                        "with --since for the bounded categorized backfill.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after fetching N new messages.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.auth:
        run_auth_flow(args.account)
        return

    if not args.since and not args.query:
        parser.error("Provide --since YYYY-MM-DD or --query.")

    if args.query:
        query = args.query
    else:
        day = args.since.replace("-", "/")
        query = f"after:{day} {query_suffix(args.include_categories)}"
    print(f"Query: {query}")

    service = gmail_service(args.account)
    account_email = get_account_email(service, args.account)
    print(f"Account email for {args.account}: {account_email or '(unknown)'}")
    pulled = load_pulled_ids(args.account)
    print(f"Already pulled for account={args.account}: {len(pulled)}.")

    ids = list(list_message_ids(service, query))
    # A custom --query is an exact override; only the default date backfill
    # also grabs current spam (which the query above can't see).
    if not args.query:
        spam_ids = list(list_message_ids(service, SPAM_QUERY,
                                         include_spam_trash=True))
        print(f"Spam IDs: {len(spam_ids)}.")
        ids = list(dict.fromkeys(ids + spam_ids))   # dedup, keep order
    to_fetch = [i for i in ids if i not in pulled]
    print(f"Candidate IDs: {len(ids)}; new: {len(to_fetch)}.")
    if args.limit:
        to_fetch = to_fetch[:args.limit]
        print(f"  Capped to --limit {args.limit}: {len(to_fetch)}.")

    # history_id and internal_date_ms must track the SAME message (the newest
    # one), so that next sync's history-API anchor matches the date-fallback
    # anchor. Without this lockstep, history_id ended up holding the LAST
    # message processed (oldest in a reverse-chrono backfill), which broke
    # sync_incremental.
    latest_history_id: str | None = None
    latest_internal_ms: int | None = None

    skipped_drafts = 0
    with EMAILS_JSONL.open("a", encoding="utf-8") as out:
        for mid in tqdm(to_fetch, desc=f"fetch[{args.account}]"):
            try:
                rec = fetch_message(service, mid, args.account, account_email)
            except HttpError as e:
                print(f"  skip {mid}: {e}", file=sys.stderr)
                continue
            if "DRAFT" in (rec.get("label_ids") or []):
                # Mark as pulled so we don't refetch on every run, but never
                # write the draft body to emails.jsonl — the graph only stores
                # final sent messages.
                append_pulled_id(args.account, mid)
                skipped_drafts += 1
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            append_pulled_id(args.account, mid)
            ms = rec.get("internal_date_ms")
            if ms and (latest_internal_ms is None or ms > latest_internal_ms):
                latest_internal_ms = ms
                latest_history_id = rec.get("history_id") or latest_history_id
    if skipped_drafts:
        print(f"  Skipped {skipped_drafts} draft message(s).")

    save_sync_state(args.account, latest_history_id, latest_internal_ms)
    print(f"Done. Wrote {len(to_fetch)} new messages for account={args.account}.")


if __name__ == "__main__":
    main()
