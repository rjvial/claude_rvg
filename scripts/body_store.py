"""On-disk byte-offset index for raw message body_html in emails.jsonl.

The previous implementation held the entire 753 MB emails.jsonl in RAM. This
module replaces that with a small offset index plus on-demand seek+read, so:

  - Startup is instant — the index is ~2 MB on disk and loads in ms.
  - After each incremental sync, only newly-appended bytes are scanned.
  - Body lookups are sub-millisecond (open + seek + readline + json.loads).
  - We never write a multi-hundred-MB derivative cache to slow storage like
    Google Drive — only the offset map.

The index lives in data/bodies.sqlite as `idx(mid, acct, offset, length)`.
Safe to delete — next refresh() rebuilds it from emails.jsonl.
"""
from __future__ import annotations

import json
import sqlite3
import threading

from _common import DATA_DIR

DB_PATH = DATA_DIR / "bodies.sqlite"
JSONL_PATH = DATA_DIR / "emails.jsonl"

# Two distinct locks so the schema-init path can't deadlock against a thread
# already holding the write lock. The write lock serialises refresh() so two
# threads can't both try to extend the store at once; schema init is a one-
# shot ensure that any thread can call without holding the write lock.
_write_lock = threading.Lock()
_schema_lock = threading.Lock()
_schema_ready = False

# A long-lived file handle on emails.jsonl, opened on first body lookup and
# reused thereafter. Opening this file fresh on every /api/body request was
# the dominant cost on a Drive-mounted disk (~2 s of overhead). Guarded by a
# lock because seek+read is not atomic across threads.
_jsonl_lock = threading.Lock()
_jsonl_fh = None
_jsonl_size_at_open = 0


def _jsonl():
    """Lazily open emails.jsonl and reuse the handle. If the file has grown
    since we opened it (a sync happened), reopen so reads see the new tail."""
    global _jsonl_fh, _jsonl_size_at_open
    if not JSONL_PATH.exists():
        return None
    cur = JSONL_PATH.stat().st_size
    if _jsonl_fh is None or cur != _jsonl_size_at_open:
        if _jsonl_fh is not None:
            try:
                _jsonl_fh.close()
            except Exception:
                pass
        _jsonl_fh = JSONL_PATH.open("rb")
        _jsonl_size_at_open = cur
    return _jsonl_fh


def _connect() -> sqlite3.Connection:
    """One connection per call. The index is tiny (~2 MB) so synchronous=OFF
    is purely a perf nicety; data loss on crash is recoverable by rebuilding
    from emails.jsonl."""
    global _schema_ready
    conn = sqlite3.connect(DB_PATH, timeout=10)
    if not _schema_ready:
        with _schema_lock:
            if not _schema_ready:
                conn.executescript("""
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=OFF;
                    PRAGMA temp_store=MEMORY;
                    CREATE TABLE IF NOT EXISTS idx (
                        mid    TEXT NOT NULL,
                        acct   TEXT NOT NULL,
                        offset INTEGER NOT NULL,
                        length INTEGER NOT NULL,
                        has_inline_img INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (mid, acct)
                    );
                    CREATE TABLE IF NOT EXISTS meta (
                        key   TEXT PRIMARY KEY,
                        value TEXT
                    );
                """)
                # Migrate older DBs that predate the has_inline_img column.
                # SQLite errors if the column already exists, so probe first.
                cols = {r[1] for r in conn.execute(
                    "PRAGMA table_info(idx)").fetchall()}
                if "has_inline_img" not in cols:
                    conn.execute(
                        "ALTER TABLE idx ADD COLUMN has_inline_img "
                        "INTEGER NOT NULL DEFAULT 0")
                # Bump this whenever the inline-image detection logic
                # changes — refresh() will re-scan from byte 0 on next run
                # so existing rows get the new flag. v1 looked at body_html
                # for data:image/ only (missed Gmail's cid:-based pasted
                # screenshots); v2 inspects the attachments[] array.
                cur_ver = conn.execute(
                    "SELECT value FROM meta WHERE key='inline_img_ver'"
                ).fetchone()
                if (cur_ver[0] if cur_ver else "") != "2":
                    conn.execute(
                        "INSERT INTO meta(key,value) VALUES('last_offset','0') "
                        "ON CONFLICT(key) DO UPDATE SET value='0'")
                    conn.execute(
                        "INSERT INTO meta(key,value) VALUES('inline_img_ver','2') "
                        "ON CONFLICT(key) DO UPDATE SET value='2'")
                conn.commit()
                _schema_ready = True
    else:
        conn.execute("PRAGMA synchronous=OFF")
    return conn


def _last_offset(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key='last_offset'") \
              .fetchone()
    return int(row[0]) if row else 0


def _set_offset(conn: sqlite3.Connection, off: int) -> None:
    conn.execute(
        "INSERT INTO meta(key,value) VALUES('last_offset',?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(off),))


def refresh() -> int:
    """Bring the offset index up to date with emails.jsonl. Returns the number
    of new index entries written. Reads only the appended bytes since the last
    refresh — cheap unless a sync just added mail. Detects a shrunk/replaced
    file (e.g. --reset-data) and rebuilds from offset 0."""
    if not JSONL_PATH.exists():
        return 0
    with _write_lock:
        conn = _connect()
        try:
            file_size = JSONL_PATH.stat().st_size
            last_off = _last_offset(conn)
            if last_off > file_size:
                conn.execute("DELETE FROM idx")
                last_off = 0
            if last_off == file_size:
                return 0
            inserted = 0
            batch: list[tuple[str, str, int, int, int]] = []
            BATCH = 5000
            sql = ("INSERT INTO idx(mid,acct,offset,length,has_inline_img) "
                   "VALUES(?,?,?,?,?) "
                   "ON CONFLICT(mid,acct) DO UPDATE SET "
                   "offset=excluded.offset, length=excluded.length, "
                   "has_inline_img=excluded.has_inline_img")
            with JSONL_PATH.open("rb") as f:
                f.seek(last_off)
                pos = last_off
                while True:
                    line_start = pos
                    raw = f.readline()
                    if not raw:
                        break
                    pos = f.tell()
                    try:
                        r = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    mid = r.get("message_id")
                    acct = r.get("account_owner")
                    html = r.get("body_html")
                    if mid and acct and html:
                        # Gmail stores pasted screenshots as a separate MIME
                        # part with Content-Disposition:inline and a Content-
                        # ID — pull_gmail classifies these as "inline_image"
                        # and load_neo4j skips them, so they never reach the
                        # graph's Attachment nodes. We treat the presence of
                        # ANY inline_image part as "this message has embedded
                        # images" for the att-column / filter.
                        has_img = 0
                        for a in (r.get("attachments") or []):
                            k = a.get("kind")
                            if k is None:
                                # Older jsonl rows pre-classification. Re-
                                # classify cheaply from filename + mime_type;
                                # per-part headers aren't preserved so we
                                # accept the small false-negative rate on
                                # CID-only inline images that don't match
                                # the image001.png filename heuristic.
                                from _attachments import classify_attachment
                                k = classify_attachment(
                                    filename=a.get("filename") or "",
                                    mime_type=a.get("mime_type"))
                            if k == "inline_image":
                                has_img = 1
                                break
                        # Belt-and-braces: catch the rare client that inlines
                        # as a data: URL inside body_html (some webmail and
                        # older Outlook builds do this on paste).
                        if not has_img and "data:image/" in html.lower():
                            has_img = 1
                        batch.append((mid, acct, line_start,
                                      pos - line_start, has_img))
                        if len(batch) >= BATCH:
                            conn.executemany(sql, batch)
                            inserted += len(batch)
                            batch.clear()
                if batch:
                    conn.executemany(sql, batch)
                    inserted += len(batch)
                new_off = pos
            _set_offset(conn, new_off)
            conn.commit()
            return inserted
        finally:
            conn.close()


def get(mid: str, acct: str) -> str:
    """Look up the offset, seek into emails.jsonl, return body_html. Empty
    string when the message is unknown or has no HTML body."""
    if not mid or not acct or not DB_PATH.exists() or not JSONL_PATH.exists():
        return ""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT offset, length FROM idx WHERE mid=? AND acct=?",
            (mid, acct)).fetchone()
    finally:
        conn.close()
    if not row:
        return ""
    offset, length = row
    try:
        with _jsonl_lock:
            f = _jsonl()
            if f is None:
                return ""
            f.seek(offset)
            raw = f.read(length)
        r = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return ""
    return r.get("body_html") or ""


def count() -> int:
    if not DB_PATH.exists():
        return 0
    conn = _connect()
    try:
        row = conn.execute("SELECT COUNT(*) FROM idx").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def inline_img_keys() -> set[tuple[str, str]]:
    """Set of (mid, acct) for every message whose body_html contains an
    inline data: image — i.e. the user pasted a screenshot directly into
    Gmail compose and the image lives as a base64 src on an <img> rather
    than as a separate MIME attachment. The graph's Attachment nodes miss
    these, so the app's "has attachments" column would otherwise mark
    them No. Populated by refresh() during the same line scan that builds
    the offset index."""
    if not DB_PATH.exists():
        return set()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT mid, acct FROM idx WHERE has_inline_img=1").fetchall()
    finally:
        conn.close()
    return {(r[0], r[1]) for r in rows}
