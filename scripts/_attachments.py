"""Classification helper for Gmail MIME attachments.

Gmail's MIME parser surfaces every named part as an "attachment", including
inline HTML signature logos (image001.png), S/MIME cryptographic blobs, and
Outlook wrapper artifacts (winmail.dat). Treating those as real attachments
inflates per-message counts ~2x in practice and pollutes downstream analytics.

This helper returns one of:
  - "attachment"       — real user-attached file (PDF, docx, eml, ics, ...)
  - "inline_image"     — embedded in HTML body via cid: reference
  - "smime"            — S/MIME signature/encryption blob
  - "outlook_artifact" — TNEF wrapper (winmail.dat) or Outlook-quoted body part

Use the gold-standard signal when available: Content-Disposition:inline and
Content-ID headers from the MIME part. When called from a re-load of historical
emails.jsonl (which doesn't preserve per-part headers) we fall back on filename
+ mime_type heuristics, which catch the dominant Outlook/Exchange patterns.
"""
from __future__ import annotations

import re

_INLINE_IMAGE_NAME = re.compile(
    r"^image\d+\.(?:png|jpe?g|gif|bmp|emz|wmz|webp|svg)$", re.IGNORECASE
)
_OUTLOOK_BODY_ARTIFACT = re.compile(r"^ATT\d+\.(?:htm|html|txt)$", re.IGNORECASE)
_SMIME_NAME = re.compile(r"^smime\.p7[sm]$", re.IGNORECASE)

_SMIME_MIMES = {
    "application/pkcs7-signature",
    "application/x-pkcs7-signature",
    "application/pkcs7-mime",
    "application/x-pkcs7-mime",
}


def _header(headers: list[dict] | None, name: str) -> str | None:
    """Case-insensitive lookup on Gmail's [{name,value}, ...] header list."""
    if not headers:
        return None
    target = name.lower()
    for h in headers:
        if (h.get("name") or "").lower() == target:
            return h.get("value")
    return None


def classify_attachment(
    *,
    filename: str,
    mime_type: str | None = None,
    headers: list[dict] | None = None,
) -> str:
    """Return one of 'attachment', 'inline_image', 'smime', 'outlook_artifact'.

    Order matters: smime and outlook_artifact are checked first because their
    filenames are unambiguous; inline-image detection prefers header evidence
    over filename regex so we don't misclassify a legitimately-attached
    'image001.png' if Content-ID is absent."""
    mime = (mime_type or "").lower()
    name = filename or ""

    # S/MIME signature blobs — unambiguous.
    if mime in _SMIME_MIMES or _SMIME_NAME.match(name):
        return "smime"

    # Outlook TNEF wrapper / body-quote artifacts.
    if name.lower() == "winmail.dat" or _OUTLOOK_BODY_ARTIFACT.match(name):
        return "outlook_artifact"

    # Inline images. Prefer header signals; fall back to filename pattern.
    disp = (_header(headers, "Content-Disposition") or "").lower()
    cid = _header(headers, "Content-ID")
    is_image_mime = mime.startswith("image/")
    if is_image_mime or not mime:
        if disp.startswith("inline"):
            return "inline_image"
        if cid:  # CID reference always means inline in HTML
            return "inline_image"
        if _INLINE_IMAGE_NAME.match(name):
            return "inline_image"

    return "attachment"


def is_real_attachment(att: dict) -> bool:
    """Convenience predicate for the load path. Uses an existing `kind` field
    if present (set by current pull_gmail.py), otherwise re-classifies from
    filename + mime_type so historical emails.jsonl still loads correctly."""
    kind = att.get("kind")
    if kind is None:
        kind = classify_attachment(
            filename=att.get("filename") or "",
            mime_type=att.get("mime_type"),
        )
    return kind == "attachment"
