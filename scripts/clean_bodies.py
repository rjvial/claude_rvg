"""Strip quoted history + signatures from message bodies.

Reads data/emails.jsonl, applies talon's quote remover (HTML- or plain-aware)
plus a heuristic signature stripper. Writes data/emails_clean.jsonl, one
record per line, with `body_clean` added.

Resumable: already-cleaned (account_owner, message_id) pairs in
data/cleaned_msg_ids.txt are skipped on re-run.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from talon import quotations
from talon.signature.bruteforce import extract_signature
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMAILS_JSONL = DATA_DIR / "emails.jsonl"
EMAILS_CLEAN_JSONL = DATA_DIR / "emails_clean.jsonl"
CLEANED_IDS_PATH = DATA_DIR / "cleaned_msg_ids.txt"

# Add company-specific disclaimer regexes here as you discover them.
# Some leak with the sentinel sentence chopped off by talon, so we also match the
# tail fragments left behind ("destinatario. Si por error...", "ninguna otra
# persona...").
DISCLAIMER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"This (e[- ]?mail|message) (and any attachments )?(is|are) "
        r"confidential.*?(\n\s*\n|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"AVISO[\s\S]{0,40}CONFIDENCIAL[\s\S]+?(?:\n\s*\n|\Z)",
        re.IGNORECASE,
    ),
    re.compile(
        r"Usted ha recibido información confidencial[\s\S]+?(?:\n\s*\n|\Z)",
        re.IGNORECASE,
    ),
    re.compile(
        r"The information you have received is both personal and confidential"
        r"[\s\S]+?(?:\n\s*\n|\Z)",
        re.IGNORECASE,
    ),
    re.compile(
        r"La información contenida en este (?:mensaje|e[- ]?mail|correo)"
        r"[\s\S]+?(?:\n\s*\n|\Z)",
        re.IGNORECASE,
    ),
    re.compile(
        r"The information contained on this (?:e[- ]?mail|message|email)"
        r"[\s\S]+?(?:\n\s*\n|\Z)",
        re.IGNORECASE,
    ),
    re.compile(
        r"destinatario\.\s*Si por error[\s\S]+?(?:Gracias\.?|\Z)",
        re.IGNORECASE,
    ),
    re.compile(
        r"ninguna otra persona[\s\S]+?(?:Gracias\.?|\Z)",
        re.IGNORECASE,
    ),
    re.compile(r"^_{5,}\s*$", re.MULTILINE),
]


# Talon is English-tuned and misses Spanish quote markers. This pattern cuts the
# body at the start of any quoted/forwarded chain so each message ends up as just
# its unique top-reply text. Covers Apple Mail / Outlook / Gmail in es + en.
QUOTE_CUT: re.Pattern[str] = re.compile(
    r"(?im)^\s*(?:"
    # Apple Mail-style forward / reply headers
    r"Inicio del mensaje reenviado\s*:"
    r"|Begin forwarded message\s*:"
    # Outlook-style original-message banners
    r"|-+\s*Mensaje original\s*-+"
    r"|-+\s*Original Message\s*-+"
    r"|-+\s*Forwarded message\s*-+"
    # Outlook header blocks — line is just the field name, value on next line(s)
    r"|De\s*:\s*$"
    r"|From\s*:\s*$"
    r"|Enviado(?:\s+el)?\s*:\s*$"
    r"|Sent\s*:\s*$"
    # Outlook header blocks — single-line form with the address on the same line
    r"|De\s*:\s+.{1,200}@.{1,200}$"
    r"|From\s*:\s+.{1,200}@.{1,200}$"
    # Inline reply markers (same line)
    r"|El\s+.{1,180}?\s+escribió\s*:"
    r"|On\s+.{1,180}?\s+wrote\s*:"
    r"|En\s+.{1,180}?\s+escribió\s*:"
    # Attribution lines that wrap across newlines — "On Mon, May 19, 2025 at ..."
    # / "El 19-05-2025, a la(s) 6:34 p.m., ..." — catch the lead-in at start of line.
    r"|On\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+"
    r"|On\s+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}[,\s]"
    r"|El\s+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}[,\s]"
    r"|El\s+(?:lun|mar|mié|mie|jue|vie|sáb|sab|dom)\b"
    # Bare wrapped "wrote:" / "escribió:" closer on its own line
    r"|.{0,80}>\s*wrote\s*:\s*$"
    r"|.{0,80}>\s*escribió\s*:\s*$"
    r")",
)


def strip_quote_chain(text: str) -> str:
    if not text:
        return text
    m = QUOTE_CUT.search(text)
    if m:
        text = text[: m.start()]
    return text.rstrip()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_disclaimers(text: str) -> str:
    for pat in DISCLAIMER_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


def clean_body(rec: dict) -> str:
    html = rec.get("body_html")
    plain = rec.get("body_plain")

    text = ""
    if html:
        # Talon's HTML quote extractor occasionally throws XPathEvalError
        # on certain markup. Fall back to plain → raw HTML stripped through
        # BeautifulSoup if it does.
        try:
            cleaned_html = quotations.extract_from_html(html)
            text = html_to_text(cleaned_html)
        except Exception:
            if plain:
                try:
                    text = quotations.extract_from_plain(plain)
                except Exception:
                    text = plain
            else:
                text = html_to_text(html)
    elif plain:
        try:
            text = quotations.extract_from_plain(plain)
        except Exception:
            text = plain
    else:
        return ""

    text = strip_quote_chain(text)

    try:
        body, _sig = extract_signature(text)
    except Exception:
        body = text

    body = strip_disclaimers(body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def dedup_key(rec: dict) -> str:
    return f"{rec.get('account_owner', '')}\t{rec['message_id']}"


def load_cleaned_keys() -> set[str]:
    if not CLEANED_IDS_PATH.exists():
        return set()
    return {
        line.rstrip("\n")
        for line in CLEANED_IDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_cleaned_key(key: str) -> None:
    with CLEANED_IDS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(key + "\n")


def clean_records(records: list[dict], *, drop_raw: bool = True) -> int:
    """Targeted clean for a known set of new records: add body_clean and append
    them to emails_clean.jsonl + their keys to cleaned_msg_ids.txt, skipping the
    full rescan of emails.jsonl that main() does. Mutates each record in place
    (adds body_clean/body_clean_empty, drops raw bodies). Already-cleaned keys
    are skipped, so it's safe to call with overlap. Returns the count written.

    Shared by the incremental sync paths (sync_incremental, serve_app)."""
    if not records:
        return 0
    cleaned = load_cleaned_keys()
    written = 0
    with EMAILS_CLEAN_JSONL.open("a", encoding="utf-8") as fout:
        for rec in records:
            key = dedup_key(rec)
            if key in cleaned:
                continue
            rec["body_clean"] = clean_body(rec)
            rec["body_clean_empty"] = not bool(rec["body_clean"].strip())
            if drop_raw:
                rec.pop("body_html", None)
                rec.pop("body_plain", None)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            append_cleaned_key(key)
            cleaned.add(key)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # --drop-raw is the default since 2026-05-14: body_html/body_plain are
    # preserved in emails.jsonl, and nothing downstream reads them from the
    # cleaned file. Keeping them here was ~100 MB of duplicated data per run.
    parser.add_argument("--keep-raw", action="store_true",
                        help="Retain body_html/body_plain in emails_clean.jsonl. "
                             "Default behavior drops them since they're already in "
                             "emails.jsonl and downstream scripts only read body_clean.")
    args = parser.parse_args()
    drop_raw = not args.keep_raw

    if not EMAILS_JSONL.exists():
        raise SystemExit(
            f"Missing {EMAILS_JSONL}. Run pull_gmail.py for each account first."
        )

    cleaned = load_cleaned_keys()
    print(f"Already cleaned: {len(cleaned)}. Will skip these.")

    total = sum(1 for _ in EMAILS_JSONL.open(encoding="utf-8"))
    processed = 0
    empty_after = 0

    with EMAILS_JSONL.open(encoding="utf-8") as fin, \
         EMAILS_CLEAN_JSONL.open("a", encoding="utf-8") as fout:
        skipped_malformed = 0
        for line in tqdm(fin, total=total, desc="clean"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                skipped_malformed += 1
                tqdm.write(f"WARN: skipping malformed line: {e}")
                continue
            key = dedup_key(rec)
            if key in cleaned:
                continue

            body_clean = clean_body(rec)
            rec["body_clean"] = body_clean
            rec["body_clean_empty"] = not bool(body_clean.strip())

            if drop_raw:
                rec.pop("body_html", None)
                rec.pop("body_plain", None)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            append_cleaned_key(key)
            processed += 1
            if rec["body_clean_empty"]:
                empty_after += 1

    print(f"Done. Processed {processed} messages.")
    if skipped_malformed:
        print(f"  Skipped {skipped_malformed} malformed line(s) "
              f"(run scripts/repair_emails_jsonl.py to clean those up).")
    print(f"  {empty_after} had empty body after cleaning "
          f"(will be skipped at extraction).")
    print(f"  Output: {EMAILS_CLEAN_JSONL}")


if __name__ == "__main__":
    main()
