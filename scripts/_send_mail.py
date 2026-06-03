"""Compose, send, and draft Gmail messages from the app.

Used by serve_app.py's /api/compose route. Loads per-account OAuth from
pull_gmail (token_<label>.json) and submits to Gmail's REST API:
  - users.messages.send  for mode="send"
  - users.drafts.create  for mode="draft"

Threading on reply is via the `threadId` parameter PLUS the In-Reply-To /
References RFC 822 headers — Gmail needs both to thread reliably across
clients.

Attachments arrive over the wire as base64 strings inside the JSON request
(simple and avoids multipart parsing in stdlib http.server). Gmail's 25 MB
per-message limit applies after base64 inflation.
"""
from __future__ import annotations

import base64
import mimetypes
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from typing import Iterable

from pull_gmail import SCOPES, gmail_service  # noqa: F401  (SCOPES re-export)


def _addr_list(items: Iterable[str | dict] | None) -> str:
    """Normalize a list of email strings or {email, name} dicts into a
    comma-joined header value. Bare strings pass through unchanged."""
    if not items:
        return ""
    parts: list[str] = []
    for x in items:
        if isinstance(x, dict):
            email = (x.get("email") or "").strip()
            name = (x.get("name") or "").strip()
            if not email:
                continue
            parts.append(formataddr((name, email)) if name else email)
        else:
            s = str(x).strip()
            if s:
                parts.append(s)
    return ", ".join(parts)


def _html_to_text_fallback(html: str) -> str:
    """Derive a plain-text alternative from an HTML body. The composer always
    sends an HTML primary part for rich replies/forwards, but Gmail still
    wants a text/plain alternative for non-HTML clients."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html or "", "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text("\n").strip()
    except Exception:
        # Minimal fallback: drop tags by regex. Good enough; bs4 is a project
        # dep so this branch is just defensive.
        import re
        return re.sub(r"<[^>]+>", "", html or "").strip()


def build_message(
    *,
    from_email: str,
    from_name: str = "",
    to: Iterable[str | dict] | None = None,
    cc: Iterable[str | dict] | None = None,
    bcc: Iterable[str | dict] | None = None,
    subject: str = "",
    body: str = "",
    is_html: bool = False,
    in_reply_to: str | None = None,
    references: Iterable[str] | None = None,
    attachments: Iterable[dict] | None = None,
) -> str:
    """Build an RFC 822 message (optionally multipart with files) and return
    it base64url-encoded for Gmail's `raw` field.

    `is_html=True` makes `body` the HTML primary part and adds a derived
    text/plain alternative so non-HTML clients still get readable content.

    `attachments` is a list of {filename, mime, data_b64}. data_b64 is the
    file's raw bytes already base64-encoded — typical from a browser
    FileReader.readAsDataURL.
    """
    msg = EmailMessage()
    msg["From"] = formataddr((from_name, from_email)) if from_name else from_email
    if to:
        msg["To"] = _addr_list(to)
    if cc:
        msg["Cc"] = _addr_list(cc)
    if bcc:
        msg["Bcc"] = _addr_list(bcc)
    msg["Subject"] = subject or ""
    msg["Date"] = formatdate(localtime=True)
    # A fresh Message-Id is required for Gmail to consider this a new message
    # in the thread; without one, replies can be silently rejected.
    msg["Message-Id"] = make_msgid()
    if in_reply_to:
        ir = in_reply_to.strip()
        if not ir.startswith("<"):
            ir = f"<{ir}>"
        msg["In-Reply-To"] = ir
    refs = list(references or [])
    if refs:
        formatted = " ".join(
            (r if r.strip().startswith("<") else f"<{r.strip()}>")
            for r in refs if r and r.strip()
        )
        if formatted:
            msg["References"] = formatted

    if is_html:
        # multipart/alternative: text fallback first, HTML second. Gmail
        # picks the HTML for clients that render it.
        msg.set_content(_html_to_text_fallback(body), charset="utf-8")
        msg.add_alternative(body or "", subtype="html", charset="utf-8")
    else:
        msg.set_content(body or "", subtype="plain", charset="utf-8")

    for a in attachments or []:
        filename = (a.get("filename") or "attachment").strip()
        mime = (a.get("mime") or "").strip() or (
            mimetypes.guess_type(filename)[0] or "application/octet-stream")
        data_b64 = a.get("data_b64") or ""
        try:
            raw = base64.b64decode(data_b64, validate=False)
        except Exception:
            continue
        maintype, _, subtype = mime.partition("/")
        if not subtype:
            maintype, subtype = "application", "octet-stream"
        msg.add_attachment(raw, maintype=maintype, subtype=subtype,
                           filename=filename)

    return base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")


def send_message(service, raw_b64: str, thread_id: str | None = None) -> dict:
    body: dict = {"raw": raw_b64}
    if thread_id:
        body["threadId"] = thread_id
    return service.users().messages().send(userId="me", body=body).execute()


def create_draft(service, raw_b64: str, thread_id: str | None = None) -> dict:
    message: dict = {"raw": raw_b64}
    if thread_id:
        message["threadId"] = thread_id
    body = {"message": message}
    return service.users().drafts().create(userId="me", body=body).execute()
