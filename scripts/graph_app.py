"""Render the whole Gmail knowledge graph as a self-contained two-level app.

Level 1 — All mail: a searchable, date-sorted list of every message in the
graph, virtualized so thousands of rows stay smooth.

Level 2 — Conversation: click a message and its full conversation is drawn
as a git-graph-style railroad (newest at top, colored lanes tracing each
reply), scrolled to that message, with the same detail panel / hover / time-
gap dividers as graph_railroad.py. "← All mail" returns to the list.

Conversations are the canonical merge of Gmail threads: messages are grouped
into connected components under REPLY_TO + NEXT_IN_THREAD (+ shared Gmail
thread), so a conversation split across Gmail's ~100-message cap or across
accounts shows as one railroad.

The railroad layout is computed in the browser (the lane-packing algorithm
ported to JS), so any of the ~2000 conversations renders on click without
pre-baking them all into the DOM. The page is fully self-contained — no CDN,
no backend — and works offline.

Connection: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD from data/.env, then
.env, then the environment — same convention as load_neo4j.py.

Usage:
    python scripts/graph_app.py
    python scripts/graph_app.py --out data/mail_app.html --open
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from _common import (
    DATA_DIR,
    PALETTE,
    esc,
    force_utf8,
    gmail_search_url,
    message_bucket,
    neo4j_driver,
)

DEFAULT_OUT = DATA_DIR / "mail_app.html"

ALL_MESSAGES_CYPHER = """
MATCH (m:Message)
OPTIONAL MATCH (m)-[:IN_THREAD]->(t:Thread)
RETURN elementId(m) AS eid, m.subject AS subject, m.sent_at AS sent_at,
       m.snippet AS snippet, m.body_clean AS body, m.gmail_url AS gmail_url,
       m.rfc822_message_id AS rfc822, m.account_owner AS acct,
       m.gmail_message_id AS mid, t.gmail_thread_id AS tid,
       m.label_ids AS labels, m.bucket AS bucket
"""
ALL_EDGES_CYPHER = """
MATCH (a:Message)-[r:REPLY_TO|NEXT_IN_THREAD]->(b:Message)
RETURN elementId(a) AS s, elementId(b) AS t, type(r) AS rt
"""
# Participants + attachment filenames for every message, for the detail panel.
PARTICIPANTS_CYPHER = """
MATCH (m:Message)
OPTIONAL MATCH (sender:Person)-[:SENT]->(m)
OPTIONAL MATCH (m)-[r:RECEIVED_BY]->(rcpt:Person)
OPTIONAL MATCH (m)-[:HAS_ATTACHMENT]->(att:Attachment)
WITH m,
     collect(DISTINCT sender.email) AS frm,
     collect(DISTINCT CASE WHEN r.kind = 'to'  THEN rcpt.email END) AS too,
     collect(DISTINCT CASE WHEN r.kind = 'cc'  THEN rcpt.email END) AS ccc,
     collect(DISTINCT CASE WHEN r.kind = 'bcc' THEN rcpt.email END) AS bccc,
     collect(DISTINCT att.filename) AS atts
RETURN elementId(m) AS eid,
       [x IN frm  WHERE x IS NOT NULL] AS frm,
       [x IN too  WHERE x IS NOT NULL] AS too,
       [x IN ccc  WHERE x IS NOT NULL] AS ccc,
       [x IN bccc WHERE x IS NOT NULL] AS bccc,
       [x IN atts WHERE x IS NOT NULL] AS atts
"""
PEOPLE_CYPHER = """
MATCH (p:Person) WHERE p.email IS NOT NULL AND p.name IS NOT NULL
RETURN p.email AS email, p.name AS name
"""


class UnionFind:
    """Group messages into conversations (connected components)."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def fetch(driver, lean: bool = False) -> tuple[dict, list, dict, dict]:
    """Pull messages + edges + participants + people on one session, in series.

    These four queries used to run in a 4-worker ThreadPoolExecutor on the
    theory that the bottleneck was network round-trips, so overlapping them
    would collapse wall time to max(query_times). Measured, the opposite is
    true: the cost is the Python-side .data() deserialization of large result
    sets (40k+ rows), which is CPU-bound and GIL-serialized. Four threads
    decoding concurrently just contend on the GIL — parallel fetch clocked
    ~13-19s vs ~8-9s sequential. So we run them in series on a single session.

    `lean` (matching build_payload's flag) fetches only the first 400 chars of
    body_clean instead of the whole thing. In lean mode the body is used only
    to derive the 240-char snippet, so pulling the full 30 MB of body text
    across all messages — just to slice off 240 chars — was pure waste."""
    msg_q = ALL_MESSAGES_CYPHER
    if lean:
        msg_q = msg_q.replace("m.body_clean AS body",
                              "left(m.body_clean, 400) AS body")
    with driver.session() as s:
        msg_rows = s.run(msg_q).data()
        edge_rows = s.run(ALL_EDGES_CYPHER).data()
        part_rows = s.run(PARTICIPANTS_CYPHER).data()
        ppl_rows = s.run(PEOPLE_CYPHER).data()

    messages: dict[str, dict] = {}
    for r in msg_rows:
        messages[r["eid"]] = {
            "subject": r["subject"], "sent_at": r["sent_at"] or "",
            "snippet": r["snippet"], "body": r["body"],
            "gmail_url": r["gmail_url"], "rfc822": r["rfc822"],
            "acct": r["acct"], "tid": r["tid"], "mid": r["mid"],
            "unread": "UNREAD" in (r["labels"] or []),
            "spam": "SPAM" in (r["labels"] or []),
            # Treatment tier. Prefer the stored m.bucket; fall back to deriving
            # it from labels so the UI is correct even before migrate_buckets.py
            # has tagged legacy nodes (e.g. existing spam reads as bucket='spam').
            "bucket": r["bucket"] or message_bucket(r["labels"]),
            "from": [], "to": [], "cc": [], "bcc": [], "atts": [],
        }
    edges = [(r["s"], r["t"], r["rt"]) for r in edge_rows]
    for r in part_rows:
        m = messages.get(r["eid"])
        if m:
            m["from"], m["to"], m["cc"] = r["frm"], r["too"], r["ccc"]
            m["bcc"], m["atts"] = r["bccc"], r["atts"]
    people = {r["email"]: r["name"] for r in ppl_rows}
    return messages, edges, people, {}


def driver():
    """Neo4j driver from data/.env, then .env, then the environment."""
    return neo4j_driver()


def build_payload(driver, lean: bool = False) -> dict:
    """Query Neo4j and assemble the {msgs, palette, people} payload the app
    renders: conversations (connected components), reply parent, kind.

    Takes a Driver (not a Session) so fetch() can run its four constituent
    queries in parallel sessions.

    `lean=True` drops `body` from every message — the live serve_app uses this
    so the initial page payload is ~10× smaller; the panel falls back to the
    snippet on open and upgrades to the real HTML via /api/body on click. It
    also tells fetch() to pull only the body prefix needed for the snippet."""
    messages, edges, people, _ = fetch(driver, lean=lean)
    if not messages:
        return {"msgs": [], "palette": PALETTE, "people": {}}

    # Inline data: images (pasted screenshots) live in body_html but never
    # become Attachment nodes. body_store flags them during its line scan.
    # We treat them as attachments for column + filter purposes.
    try:
        import body_store
        body_store.refresh()
        inline_img_set = body_store.inline_img_keys()
    except Exception:
        inline_img_set = set()

    # --- conversations: connected components ------------------------------
    uf = UnionFind()
    for eid in messages:
        uf.add(eid)
    for a, b, _rt in edges:
        if a in messages and b in messages:
            uf.union(a, b)
    by_thread: dict[tuple, list[str]] = {}
    for eid, m in messages.items():
        by_thread.setdefault((m["tid"], m["acct"]), []).append(eid)
    for grp in by_thread.values():
        for other in grp[1:]:
            uf.union(grp[0], other)

    # --- reply parent: REPLY_TO wins, else NEXT_IN_THREAD predecessor -----
    reply_parent: dict[str, str] = {}
    next_pred: dict[str, str] = {}
    for a, b, rt in edges:
        if a not in messages or b not in messages:
            continue
        if rt == "REPLY_TO":
            reply_parent.setdefault(a, b)
        elif rt == "NEXT_IN_THREAD":
            next_pred.setdefault(b, a)
    parent_of = {eid: reply_parent.get(eid) or next_pred.get(eid)
                 for eid in messages}

    # --- compact integer indexing ----------------------------------------
    order = sorted(messages, key=lambda e: messages[e]["sent_at"])
    idx = {eid: i for i, eid in enumerate(order)}
    conv_id = {eid: idx.get(uf.find(eid), -1) for eid in messages}

    def sender_of(eid: str) -> str:
        frm = messages[eid]["from"]
        return frm[0] if frm else ""

    def kind_of(eid: str) -> str:
        p = parent_of.get(eid)
        if not p or p not in messages:
            return "root"
        cs, ps = sender_of(eid), sender_of(p)
        if cs and ps and cs == ps:
            return "follow-up"
        subj = (messages[eid]["subject"] or "").lstrip().lower()
        return "forward" if subj.startswith(("fwd:", "fw:")) else "reply"

    msgs_js: list[dict] = []
    for eid in order:
        m = messages[eid]
        snd = sender_of(eid)
        body = m["body"] or ""
        snippet = (body or m["snippet"] or "").replace("\n", " ").strip()
        row = {
            "conv": conv_id[eid],
            "par": idx[parent_of[eid]] if parent_of.get(eid) in idx else -1,
            "kind": kind_of(eid),
            "from": snd,
            "name": people.get(snd, ""),
            "subj": m["subject"] or "(no subject)",
            "snip": snippet[:240],
            "sent": m["sent_at"],
            "acct": m["acct"] or "",
            "mid": m["mid"] or "",
            "tid": m["tid"] or "",
            "unread": m["unread"],
            "spam": m["spam"],
            "bucket": m["bucket"],
            "to": m["to"], "cc": m["cc"], "bcc": m["bcc"], "atts": m["atts"],
            "inline_img": (m["mid"], m["acct"]) in inline_img_set,
            "url": gmail_search_url(m["gmail_url"], m["rfc822"]),
        }
        if not lean:
            row["body"] = body
        msgs_js.append(row)
    return {"msgs": msgs_js, "palette": PALETTE, "people": people}


def render_page(payload: dict, title: str = "Mail Graph",
                version: int = 0) -> str:
    r"""Embed a payload (from build_payload) into the app HTML.

    The payload is injected inside a <script type="application/json"> tag and
    parsed with JSON.parse — much faster than parsing as a JS literal for
    multi-MB payloads. We escape `</` → `<\/` so a stray `</script>` inside a
    subject/body can't break out of the data tag (still valid JSON).

    `version` is baked into the page so the client knows which cache version the
    data it loaded corresponds to. serve_app serves a disk snapshot at boot for
    instant startup; if the background rebuild bumps the version before the
    client's SSE stream connects, the baked baseline still differs from the
    server's, so the reload banner fires. Static exports pass 0 (live mode off)."""
    n_conv = len({m["conv"] for m in payload["msgs"]})
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return (PAGE
            .replace("__TITLE__", esc(title))
            .replace("__COUNT__", f"{len(payload['msgs']):,}")
            .replace("__CONVS__", f"{n_conv:,}")
            .replace("__DATA_VERSION__", str(int(version)))
            .replace("__DATA__", data))


def main() -> int:
    force_utf8()

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--title", default="claude_rvg — mail")
    ap.add_argument("--open", action="store_true",
                    help="Open the result in the default browser.")
    args = ap.parse_args()

    drv = driver()
    try:
        payload = build_payload(drv)
    finally:
        drv.close()
    if not payload["msgs"]:
        sys.exit("No messages in the graph.")

    html = render_page(payload, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    n_conv = len({m["conv"] for m in payload["msgs"]})
    mb = args.out.stat().st_size / 1_048_576
    print(f"Wrote {args.out}  ({len(payload['msgs']):,} messages, "
          f"{n_conv:,} conversations, {mb:.1f} MB)")
    print("Self-contained — no internet needed to view.")
    if args.open:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


# PWA manifest + icon — let the user "Install" the app so it opens in its own
# standalone window (showing "Mail Graph", no address bar) instead of a browser
# tab pinned to http://localhost:8765/. serve_app.py serves these at
# /manifest.webmanifest and /icon.svg. The SVG is a maskable, full-bleed clay
# square with a paper-white envelope glyph kept inside the central safe zone.
APP_NAME = "Mail Graph"
ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
    '<rect width="512" height="512" fill="#CC785C"/>'
    '<g fill="none" stroke="#FAF9F5" stroke-width="26" '
    'stroke-linejoin="round" stroke-linecap="round">'
    '<rect x="116" y="160" width="280" height="192" rx="22"/>'
    '<path d="M130 182 L256 280 L382 182"/>'
    '</g></svg>'
)
MANIFEST_JSON = json.dumps({
    "name": APP_NAME,
    "short_name": APP_NAME,
    "start_url": "/",
    "scope": "/",
    "display": "standalone",
    "background_color": "#FAF9F5",
    "theme_color": "#CC785C",
    "icons": [
        {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml",
         "purpose": "any maskable"},
    ],
}, ensure_ascii=False)


# Loading splash — served immediately at "/" on a cold start (before the graph
# is loaded), so the app window can open right away instead of waiting on Neo4j
# + the cache build. It polls /api/boot for the current phase and reloads into
# the real app once the server reports ready; on failure it shows the error and
# a Retry button (POST /api/boot/retry).
LOADING_PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Mail Graph</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/svg+xml" href="/icon.svg">
<style>
  :root{--bg:#FAF9F5;--ink:#141413;--ink-3:#8F8B80;--accent:#CC785C;
        --line:#E3DFD6}
  *{box-sizing:border-box}
  html,body{height:100%;margin:0}
  body{background:var(--bg);color:var(--ink);display:flex;
    align-items:center;justify-content:center;
    font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
  .card{width:min(420px,90vw);text-align:center;padding:34px 30px}
  .logo{width:76px;height:76px;margin:0 auto 22px;display:block;
    border-radius:18px;box-shadow:0 10px 26px rgba(204,120,92,.28)}
  h1{font:600 20px/1.3 Georgia,'Times New Roman',serif;margin:0}
  .phase{color:var(--ink-3);min-height:20px;margin-top:16px;font-size:13px}
  .bar{margin:22px auto 0;width:100%;height:4px;border-radius:3px;
    background:var(--line);overflow:hidden}
  .bar>i{display:block;height:100%;width:35%;border-radius:3px;
    background:var(--accent);animation:slide 1.1s ease-in-out infinite}
  @keyframes slide{0%{margin-left:-35%}100%{margin-left:100%}}
  .err{display:none;margin-top:18px;color:#A84A43;font-size:13px;
    white-space:pre-line}
  .retry{display:none;margin-top:16px;background:var(--accent);color:#fff;
    border:0;border-radius:8px;padding:9px 18px;font-size:13px;cursor:pointer}
  .retry:hover{background:#B05E40}
</style></head>
<body>
  <div class="card">
    <img class="logo" src="/icon.svg" alt="">
    <h1>Mail Graph</h1>
    <div class="phase" id="phase">Starting…</div>
    <div class="bar" id="bar"><i></i></div>
    <div class="err" id="err"></div>
    <button class="retry" id="retry">Retry</button>
  </div>
<script>
  var $ = function(id){ return document.getElementById(id); };
  async function tick(){
    var d;
    try{ d = await (await fetch("/api/boot", {cache:"no-store"})).json(); }
    catch(e){ $("phase").textContent = "Waiting for server…"; return; }
    if(d.ready){ location.reload(); return; }
    $("phase").textContent = d.phase || "Loading…";
    if(d.error){
      $("bar").style.display = "none";
      $("err").style.display = "block"; $("err").textContent = d.error;
      $("retry").style.display = "inline-block";
    }else{
      $("bar").style.display = "block";
      $("err").style.display = "none";
      $("retry").style.display = "none";
    }
  }
  $("retry").addEventListener("click", async function(){
    $("retry").style.display = "none"; $("err").style.display = "none";
    $("bar").style.display = "block"; $("phase").textContent = "Retrying…";
    try{ await fetch("/api/boot/retry", {method:"POST"}); }catch(e){}
  });
  tick(); setInterval(tick, 700);
</script>
</body></html>"""


PAGE = r"""<!DOCTYPE html>
<html lang="en" translate="no"><head><meta charset="utf-8"><title>__TITLE__</title>
<meta name="google" content="notranslate">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#CC785C">
<meta name="apple-mobile-web-app-title" content="Mail Graph">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" type="image/svg+xml" href="/icon.svg">
<link rel="apple-touch-icon" href="/icon.svg">
<style>
  :root{
    --bg:#FAF9F5;          /* app background — warm paper */
    --surface:#F0EEE6;     /* header, filter bar, panel — deeper cream */
    --raised:#FFFEFB;      /* inputs */
    --ink:#141413;         /* primary text */
    --ink-2:#5F5B53;       /* secondary text */
    --ink-3:#8F8B80;       /* faint text */
    --line:#E6E3D8;        /* hairline dividers */
    --line-2:#D6D2C4;      /* input / button borders */
    --accent:#CC785C;      /* Anthropic clay-coral */
    --accent-deep:#B05E40; /* coral — hover / links */
    --tint:#EFE0D8;        /* soft coral wash — selection */
    --tint-soft:#F7EDE6;   /* paler coral — same-conversation siblings */
    --tint-2:#E4D1B0;      /* warm kraft-brown fill — spotlight match */
    --hover:#EEEBE1;       /* neutral hover surface */
    --serif:Georgia,"Times New Roman",serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
    font:13px ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
  #hdr{position:sticky;top:0;z-index:6;padding:9px 16px;
    background:rgba(250,249,245,.97);border-bottom:1px solid var(--line);
    display:flex;align-items:center;gap:10px}
  #hdr h1{margin:0;font:500 15px/1.3 var(--serif);color:var(--ink);
    white-space:nowrap}
  #hdr .sub{color:var(--ink-3);font-size:11.5px;white-space:nowrap}
  #search{flex:1;min-width:140px;background:var(--raised);color:var(--ink);
    border:1px solid var(--line-2);border-radius:6px;padding:5px 9px;
    font-size:12px}
  #search:focus{outline:none;border-color:var(--accent)}
  #hdr button{background:var(--surface);color:var(--ink-2);
    border:1px solid var(--line-2);border-radius:6px;padding:5px 10px;
    font-size:11.5px;cursor:pointer}
  #hdr button:hover{background:var(--hover)}
  #back{display:none}
  #banner{display:none;position:fixed;top:9px;left:50%;
    transform:translateX(-50%);z-index:20;background:#E7EFE5;color:#3F6B4A;
    border:1px solid #C4D6BF;border-radius:8px;padding:6px 14px;
    font-size:12px;cursor:pointer;box-shadow:0 6px 18px rgba(60,50,38,.16)}
  #banner:hover{background:#DCE9D9}
  .acctdot{display:inline-block;width:8px;height:8px;border-radius:50%;
    margin-right:5px;vertical-align:middle}
  /* ---- list view: shared column widths so the filter bar lines up
         exactly over the rows ---- */
  .c-sel{width:28px;flex:none;display:flex;align-items:center;
    justify-content:center}
  .c-acct{width:84px;flex:none}
  .c-date{width:120px;flex:none}
  .c-from{width:190px;flex:none}
  .c-to{width:190px;flex:none}
  .c-subj{flex:2 1 0;min-width:0}
  .c-snip{flex:3 1 0;min-width:0}
  .c-pos{width:66px;flex:none;text-align:center}
  /* brief bottom-center confirmation toast (e.g. "Copied: …") */
  #toast{position:fixed;left:50%;bottom:26px;
    transform:translateX(-50%) translateY(8px);
    background:var(--ink);color:var(--bg);font-size:12px;
    padding:7px 13px;border-radius:7px;max-width:80vw;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
    box-shadow:0 8px 24px rgba(40,32,24,.32);
    opacity:0;pointer-events:none;z-index:60;
    transition:opacity .15s ease, transform .15s ease}
  #toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
  .c-att{width:88px;flex:none;text-align:center;color:var(--ink-2);
    font-variant-numeric:tabular-nums}
  /* Attachment column header — a <select> styled to match the text
     filter inputs in the other column heads. */
  #cols .c-att select{width:100%;min-width:0;background:var(--raised);
    color:var(--ink);border:1px solid var(--line-2);border-radius:5px;
    padding:3px 7px;font-size:11px;font-family:inherit;cursor:pointer;
    -webkit-appearance:menulist;appearance:menulist}
  #cols .c-att select:focus{outline:none;border-color:var(--accent)}
  #cols .c-att select.on{border-color:var(--accent);color:var(--accent-deep);
    background:var(--tint)}
  /* The selection checkboxes (master + per-row) only appear when the user
     explicitly enters select mode via the header's ☑ Select button. */
  body:not(.select-mode) .c-sel{display:none}
  .c-sel input[type=checkbox]{margin:0;cursor:pointer;
    accent-color:var(--accent)}
  /* Remove-selected header button — accent red so destructive intent
     reads at a glance. Visible only in select mode. */
  #hdr #rmbtn{background:#C25C54;color:#fff;border-color:#A84A43}
  #hdr #rmbtn:hover{background:#A84A43}
  #hdr #rmbtn:disabled{opacity:.55;cursor:default}
  /* Mark-read header button — green for a safe, non-destructive action.
     Paired with Remove, visible only in select mode. */
  #hdr #readbtn{background:#3F6B4A;color:#fff;border-color:#345A3E}
  #hdr #readbtn:hover{background:#345A3E}
  #hdr #readbtn:disabled{opacity:.55;cursor:default}
  /* Not-spam header button — blue for a safe reclassify action. Shown only on
     the spam page while a selection is armed. */
  #hdr #notspambtn{background:#3F5F8A;color:#fff;border-color:#34507A}
  #hdr #notspambtn:hover{background:#34507A}
  #hdr #notspambtn:disabled{opacity:.55;cursor:default}
  /* Mark-as-spam header button — amber for a caution action. Shown only on
     the main mail page while a selection is armed. */
  #hdr #markspambtn{background:#B0742E;color:#fff;border-color:#946326}
  #hdr #markspambtn:hover{background:#946326}
  #hdr #markspambtn:disabled{opacity:.55;cursor:default}
  /* Bucket (tier) filter — multi-select dropdown in the header bar, replacing
     the old Spam toggle. Reuses the .acctf menu styling; sized to fit the bar
     and accent-highlighted when a non-default tier set is active. */
  #hdr .bucketf{position:relative;display:inline-block;width:auto;
    vertical-align:middle}
  #hdr .bucketf .btn{width:auto;min-width:104px}
  #hdr .bucketf.active .btn{border-color:var(--accent);color:var(--accent)}
  #cols{position:absolute;top:46px;left:0;right:0;height:32px;z-index:5;
    display:flex;align-items:center;gap:9px;padding:0 16px;
    background:var(--surface);border-bottom:1px solid var(--line)}
  /* Filter pane is collapsed by default; the #colstoggle wedge flips it.
     When hidden, the list rises to fill the freed row. */
  body.cols-hidden #cols{display:none}
  body.cols-hidden #list{top:46px}
  /* Far-left header stack: the select-all checkbox (shown only in select
     mode) sits ABOVE the filter-pane chevron, which is pushed to the bottom.
     Fixed height ≤ the other buttons so the 46px header doesn't grow — the
     list/thread top offsets assume 46px. */
  /* Far-left header stack. #leftctl stretches to the header's content height
     and centers the select-all checkbox, so it lines up on the same row as
     the Select/Cancel button. The chevron is absolutely pinned just below the
     checkbox (in the header's bottom padding) so it never grows the fixed
     46px header that the list/thread top offsets depend on. */
  #leftctl{position:relative;align-self:stretch;width:16px;
    display:flex;align-items:center;justify-content:center}
  #hdr #f-sel-all{margin:0;width:12px;height:12px;cursor:pointer;
    accent-color:var(--accent)}
  body:not(.select-mode) #hdr #f-sel-all{display:none}
  /* Filter-pane toggle: a borderless chevron under the checkbox. Higher
     specificity than `#hdr button` so it stays flat. */
  #hdr #colstoggle{position:absolute;left:50%;top:100%;
    transform:translate(-50%,-4px);padding:0;border:0;background:transparent;
    color:var(--ink-3);font-size:12px;line-height:1;cursor:pointer;
    transition:color .15s ease, transform .2s ease}
  #hdr #colstoggle:hover{color:var(--accent)}
  /* rotate the chevron down when the pane is open (keep the centering shift) */
  body:not(.cols-hidden) #hdr #colstoggle{transform:translate(-50%,-4px) rotate(90deg)}
  #cols input{width:100%;min-width:0;background:var(--raised);color:var(--ink);
    border:1px solid var(--line-2);border-radius:5px;padding:3px 7px;
    font-size:11px}
  #cols input:focus{outline:none;border-color:var(--accent)}
  /* account filter — multi-select dropdown */
  .acctf{position:relative;width:100%}
  .acctf .btn{width:100%;display:flex;align-items:center;gap:4px;
    background:var(--raised);color:var(--ink);border:1px solid var(--line-2);
    border-radius:5px;padding:3px 6px;font-size:11px;cursor:pointer;
    white-space:nowrap;overflow:hidden}
  .acctf .lbl{overflow:hidden;text-overflow:ellipsis}
  .acctf .car{margin-left:auto;color:var(--ink-3);font-size:8px}
  .acctf.open .btn{border-color:var(--accent)}
  .acctf .menu{display:none;position:absolute;top:calc(100% + 3px);left:0;
    z-index:9;min-width:152px;background:var(--surface);
    border:1px solid var(--line-2);border-radius:7px;padding:4px;
    box-shadow:0 8px 22px rgba(60,50,38,.18)}
  .acctf.open .menu{display:block}
  .acctf .opt{display:flex;align-items:center;gap:7px;padding:4px 7px;
    border-radius:5px;cursor:pointer;font-size:11.5px;white-space:nowrap}
  .acctf .opt:hover{background:var(--hover)}
  /* the checkbox + dot must keep their natural size. The #cols prefix is
     required: #cols input{width:100%} has an ID and would otherwise win
     on specificity and balloon the checkbox. */
  #cols .acctf .opt input{flex:none;width:auto;min-width:0;margin:0;
    padding:0;border:0;background:none;cursor:pointer;
    accent-color:var(--accent)}
  .acctf .opt i{flex:none}
  .acctf .opt.all{border-bottom:1px solid var(--line);border-radius:0;
    margin:0 0 3px;padding-bottom:6px}
  #list{position:absolute;top:78px;left:0;right:0;bottom:0;overflow:auto}
  #spacer{position:relative}
  .lrow{position:absolute;left:0;right:0;height:28px;display:flex;
    align-items:center;gap:9px;padding:0 16px;cursor:pointer;
    border-bottom:1px solid var(--line);white-space:nowrap}
  /* Same-conversation siblings of the cursor row — paler than the cursor's
     own wash. Listed before :hover so pointing at a sibling still shows the
     neutral hover, and before .cursor so the cursor row keeps its own style. */
  .lrow.convlit{background:var(--tint-soft)}
  .lrow:hover{background:var(--hover)}
  .lrow.cursor{background:var(--tint);box-shadow:inset 3px 0 0 var(--accent)}
  /* Unread = still carries Gmail's UNREAD label — bold like any mail client. */
  .lrow.unread .lfrom,
  .lrow.unread .lsubj{font-weight:700}
  .lrow.unread .lfrom{color:var(--ink)}
  .lrow>span{overflow:hidden;text-overflow:ellipsis}
  .lacct{color:var(--ink-3);font-size:11px}
  .ldate{color:var(--ink-3);font-variant-numeric:tabular-nums}
  .lfrom{color:var(--ink-2)}
  .lsubj{color:var(--ink)}
  .lsnip{color:var(--ink-3)}
  .lpos{color:var(--ink-3);font-size:10.5px;font-variant-numeric:tabular-nums}
  mark{background:#F1D9A3;color:var(--ink);border-radius:2px}
  .lrow.cursor mark{background:#F7E7BE}
  /* ---- thread (railroad) view ---- */
  #thread{position:absolute;top:46px;left:0;right:0;bottom:0;overflow:auto;
    display:none}
  #legend{padding:7px 16px;display:flex;flex-wrap:wrap;gap:3px 12px;
    font-size:11px;border-bottom:1px solid var(--line)}
  #legend i{display:inline-block;width:9px;height:9px;border-radius:50%;
    margin-right:5px;vertical-align:middle}
  /* legend entries + row sender names spotlight that person's messages */
  #legend .legp{cursor:pointer;border-radius:4px;padding:1px 6px;
    user-select:none}
  #legend .legp:hover{background:var(--hover)}
  #legend .legp.lega{background:var(--tint);
    box-shadow:inset 0 0 0 1px var(--accent)}
  .row{display:flex;align-items:stretch;border-bottom:1px solid var(--line);
    cursor:pointer}
  .row:hover{background:var(--hover)}
  .row.litrow{background:var(--tint-2)}
  .row.litby{background:var(--tint-2)}
  .row.selected{background:var(--tint);box-shadow:inset 3px 0 0 var(--accent)}
  .row.dim{opacity:.3}
  .row.dim.selected{opacity:1}
  .rail{flex:none;display:block}
  .ln.litln{stroke-width:4.5}
  .divider{display:flex;align-items:center;height:22px;color:var(--ink-3);
    font-size:10.5px}
  .divider .gap{padding-left:12px;white-space:nowrap}
  .msg{flex:1;min-width:0;display:flex;align-items:center;gap:8px;
    height:30px;padding:0 14px;white-space:nowrap}
  .num{flex:none;font-weight:700;font-variant-numeric:tabular-nums}
  .txt{color:var(--ink);overflow:hidden;text-overflow:ellipsis;flex:0 1 auto}
  .who{flex:none;margin-left:auto;color:var(--ink-2);cursor:pointer}
  .who:hover{text-decoration:underline}
  .date{flex:none;color:var(--ink-3);font-variant-numeric:tabular-nums}
  .rel{flex:none;font-size:10.5px;font-weight:600}
  /* ---- detail panel: an embedded pane docked below the header ---- */
  #panel{display:none;position:fixed;top:46px;right:0;bottom:0;width:380px;
    z-index:7;background:var(--surface);border-left:1px solid var(--line-2);
    padding:16px 18px;overflow:auto}
  /* When open, the list / filter bar / thread reflow into the space left
     of the panel instead of the panel floating over them. */
  body.panel #panel{display:block}
  body.panel #cols,
  body.panel #list,
  body.panel #thread{right:380px}
  #panel .x{float:right;cursor:pointer;color:var(--ink-3);font-size:18px}
  #panel .x:hover{color:var(--accent-deep)}
  /* popout button — sits to the LEFT of ✕ because floats stack in source
     order, so its DOM position must be BEFORE the close button. */
  #panel .popout{float:right;cursor:pointer;color:var(--ink-3);font-size:15px;
    margin-right:10px;line-height:18px}
  #panel .popout:hover{color:var(--accent-deep)}
  #panel .pnum{display:inline-block;font-size:10px;font-weight:700;
    padding:2px 8px;border-radius:10px;background:var(--raised);
    border:1px solid var(--line-2);color:var(--ink-2)}
  #panel h2{font:500 16px/1.35 var(--serif);color:var(--ink);margin:9px 0 4px}
  #panel .prel{color:var(--ink-3);font-style:italic;margin-bottom:10px}
  #panel .pr{display:grid;grid-template-columns:58px 1fr;gap:9px;margin:4px 0}
  #panel .pk{color:var(--ink-3);font-size:10.5px;text-transform:uppercase;
    text-align:right}
  #panel .pv{color:var(--ink);word-break:break-word}
  #panel .pbody{margin-top:11px;padding-top:10px;
    border-top:1px solid var(--line);color:var(--ink-2);font-size:12px;
    line-height:1.5;white-space:pre-wrap;word-break:break-word;
    max-height:46vh;overflow:auto}
  #panel .pbody.ishtml{max-height:none;overflow:visible;padding-top:11px;
    white-space:normal}
  #panel .pbodyhtml{width:100%;height:62vh;border:1px solid var(--line);
    border-radius:6px;background:#fff}
  #panel a.glink{display:inline-block;margin-top:12px;
    color:var(--accent-deep);font-size:12px;text-decoration:none;
    border:1px solid var(--line-2);border-radius:6px;padding:5px 10px}
  #panel a.glink:hover{border-color:var(--accent);background:var(--tint)}
  #empty{padding:40px;text-align:center;color:var(--ink-3)}
  /* Ask the graph — accent button + modal */
  #hdr #askbtn{background:var(--accent);color:#fff;
    border-color:var(--accent-deep)}
  #hdr #askbtn:hover{background:var(--accent-deep)}
  /* Non-modal floating window: the container spans the viewport but is
     click-through (pointer-events:none) so the app behind stays usable while
     Ask is open; only the card itself catches events. */
  #ask{display:none;position:fixed;inset:0;z-index:30;pointer-events:none}
  .askcard{position:absolute;top:54px;left:50%;transform:translateX(-50%);
    width:min(620px,92vw);height:min(70vh,560px);
    min-width:340px;min-height:240px;max-width:96vw;max-height:94vh;
    pointer-events:auto;resize:both;
    display:flex;flex-direction:column;background:var(--bg);
    border:1px solid var(--line-2);border-radius:12px;
    box-shadow:0 18px 50px rgba(40,32,24,.28);overflow:hidden}
  .askhd{display:flex;align-items:center;padding:13px 16px;
    font:500 14px/1.3 var(--serif);color:var(--ink);
    border-bottom:1px solid var(--line)}
  .askhd .asknew{background:var(--raised);
    border:1px solid var(--line-2);color:var(--ink-3);cursor:pointer;
    font-size:11px;padding:4px 9px;border-radius:6px}
  .askhd .asknew:hover{border-color:var(--accent);color:var(--accent-deep)}
  .askhd .x{cursor:pointer;color:var(--ink-3);font-size:16px}
  .askhd .x:hover{color:var(--accent-deep)}
  /* chat transcript */
  #asklog{flex:1;min-height:80px;overflow-y:auto;padding:14px 16px;
    display:flex;flex-direction:column;gap:9px}
  .askmsg{max-width:85%;padding:9px 12px;border-radius:10px;
    font-size:12.5px;line-height:1.55;white-space:pre-wrap;
    word-break:break-word}
  .askmsg.user{align-self:flex-end;background:var(--accent);color:#fff;
    border-bottom-right-radius:3px}
  .askmsg.bot{align-self:flex-start;background:var(--surface);
    border:1px solid var(--line);color:var(--ink);
    border-bottom-left-radius:3px}
  .askmsg.bot a{color:var(--accent-deep);word-break:break-all}
  .askmsg.intro{align-self:stretch;max-width:none;background:transparent;
    color:var(--ink-3);font-style:italic;padding:2px 0}
  .askmsg.thinking{color:var(--ink-3)}
  /* animated "model is working" indicator — three bouncing dots */
  .askdots{display:inline-flex;gap:3px;margin-left:5px;
    vertical-align:middle}
  .askdots span{width:5px;height:5px;border-radius:50%;
    background:var(--ink-3);
    animation:askbounce 1.2s infinite ease-in-out both}
  .askdots span:nth-child(2){animation-delay:.15s}
  .askdots span:nth-child(3){animation-delay:.30s}
  @keyframes askbounce{
    0%,80%,100%{transform:scale(.5);opacity:.35}
    40%{transform:scale(1);opacity:1}}
  /* live "thinking" trace — the model's reasoning + tool calls stream in
     here while the answer is being prepared */
  .askmsg.thinking .askthlbl{font-size:12px;color:var(--ink-3);
    display:inline-flex;align-items:center}
  .askmsg.thinking .askthtrace{display:flex;flex-direction:column;gap:4px;
    margin-top:8px;padding-top:7px;border-top:1px solid var(--line);
    font-size:11px;color:var(--ink-3);line-height:1.45;
    max-height:240px;overflow-y:auto}
  .askmsg.thinking .askstep{padding:1px 0;
    animation:askfade .25s ease-out both}
  @keyframes askfade{from{opacity:0;transform:translateY(-2px)}
    to{opacity:1;transform:none}}
  .askmsg.thinking .askphase{font-style:italic}
  .askmsg.thinking .asktoolname{color:var(--accent-deep);font-weight:600}
  .askmsg.thinking .asktooldetail{margin:2px 0 0 14px}
  .askmsg.thinking code{font-family:ui-monospace,Consolas,monospace;
    background:var(--raised);padding:1px 4px;border-radius:3px;
    font-size:10px;white-space:pre-wrap;word-break:break-all;
    color:var(--ink)}
  .askmsg.err{border-color:#c8857a;color:#a3503f}
  /* inline [n] citations + the Sources list under an answer */
  .askmsg .cite{color:var(--accent-deep);font-weight:600;font-size:11px;
    text-decoration:none;padding:0 1px}
  .askmsg .cite:hover{text-decoration:underline}
  .askmsg .srcs{margin-top:9px;padding-top:8px;
    border-top:1px solid var(--line)}
  .askmsg .srch{font-size:10px;font-weight:600;letter-spacing:.05em;
    text-transform:uppercase;color:var(--ink-3)}
  .askmsg .src{margin-top:4px;font-size:11px;line-height:1.45}
  .askmsg .src a{color:var(--accent-deep);text-decoration:none}
  .askmsg .src a:hover{text-decoration:underline}
  /* thumbs up/down rating bar under an answer (feeds Liam's learning) */
  .askmsg .askfb{margin-top:10px;display:flex;align-items:center;gap:6px;
    flex-wrap:wrap}
  .askfb .askfbbtn{background:var(--raised);border:1px solid var(--line-2);
    border-radius:6px;padding:2px 8px;font-size:13px;line-height:1.4;
    cursor:pointer;color:var(--ink-3)}
  .askfb .askfbbtn:hover{border-color:var(--accent);color:var(--ink)}
  .askfb .askfbbtn.on{border-color:var(--accent);background:var(--hover);
    color:var(--ink)}
  .askfb .askfbmsg{font-size:11px;color:var(--ink-3)}
  .askfb .askfbnote{flex:1;min-width:140px;padding:5px 8px;color:var(--ink);
    background:var(--raised);border:1px solid var(--line-2);border-radius:6px;
    font:12px ui-sans-serif,system-ui,sans-serif}
  .askfb .askfbnote:focus{outline:none;border-color:var(--accent)}
  .askfb .askfbsend{background:var(--accent);color:#fff;border:0;
    border-radius:6px;padding:5px 11px;font-size:12px;cursor:pointer}
  .askfb .askfbsend:hover{background:var(--accent-deep)}
  #ask .row2{display:flex;align-items:flex-end;gap:9px;padding:11px 16px;
    border-top:1px solid var(--line)}
  #askq{flex:1;padding:9px 11px;color:var(--ink);
    background:var(--raised);border:1px solid var(--line-2);
    border-radius:8px;resize:none;height:40px;max-height:120px;
    font:13px ui-sans-serif,system-ui,sans-serif}
  #askq:focus{outline:none;border-color:var(--accent)}
  #asksend{background:var(--accent);color:#fff;border:0;border-radius:7px;
    padding:9px 17px;font-size:12px;cursor:pointer;flex-shrink:0}
  #asksend:hover{background:var(--accent-deep)}
  #asksend:disabled{opacity:.55;cursor:default}
  /* ---- accounts / sign-in panel --------------------------------------- */
  #hdr #acctsbtn{background:var(--surface);color:var(--ink);
    border:1px solid var(--line-2)}
  #hdr #acctsbtn:hover{background:var(--hover);border-color:var(--accent)}
  #accts{display:none;position:fixed;inset:0;z-index:31;
    background:rgba(20,18,16,.38);align-items:flex-start;
    justify-content:center}
  /* ---- settings sections (Email Accounts / LLM Model / Claude OAuth) --- */
  .setbody{flex:1;min-height:0;overflow-y:auto}
  .setsec{padding:13px 16px;border-top:1px solid var(--line)}
  .setsec:first-child{border-top:0}
  .setsechd{font-size:11px;font-weight:700;letter-spacing:.05em;
    text-transform:uppercase;color:var(--ink-3);margin-bottom:10px}
  .setrow{display:flex;align-items:center;gap:10px}
  .setmsg{font-size:11px;color:var(--ink-3)}
  .sethint{font-size:11px;color:var(--ink-3);margin-top:9px;line-height:1.5}
  #llmmodel{background:var(--raised);color:var(--ink);
    border:1px solid var(--line-2);border-radius:7px;padding:7px 10px;
    font-size:12px;cursor:pointer}
  #llmmodel:focus{outline:none;border-color:var(--accent)}
  #claudeauth{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  #claudeauth .cainfo{flex:1;min-width:0}
  #claudeauth .caemail{font-weight:600;font-size:13px;color:var(--ink)}
  #claudeauth .cameta{font-size:11.5px;color:var(--ink-3)}
  #claudeauth .castatus{font-size:11px;font-weight:600;flex-shrink:0}
  #claudeauth .castatus.ok{color:#3f7a4f}
  #claudeauth .castatus.no{color:#a3503f}
  #claudeauth .caacts{display:flex;gap:6px;flex-shrink:0}
  #claudeauth .caacts button{border-radius:7px;padding:7px 11px;font-size:12px;
    cursor:pointer;border:1px solid var(--line-2);background:var(--surface);
    color:var(--ink-2);flex-shrink:0}
  #claudeauth .caacts button:disabled{opacity:.55;cursor:default}
  #claudeauth .calogin{background:var(--accent);color:#fff;border:0}
  #claudeauth .calogin:hover{background:var(--accent-deep)}
  #claudeauth .calogout:hover{background:var(--hover);border-color:var(--accent)}
  /* ---- memory panel ---- */
  /* Ask | Compose scope switch — each functionality has its own memory store */
  .memscope{display:flex;gap:6px;margin-bottom:11px}
  .memscope .memtab{background:var(--surface);color:var(--ink-2);
    border:1px solid var(--line-2);border-radius:7px;padding:5px 14px;
    font-size:12px;cursor:pointer}
  .memscope .memtab:hover{background:var(--hover)}
  .memscope .memtab.on{background:var(--accent);color:#fff;
    border-color:var(--accent-deep)}
  .memlearn{display:flex;align-items:center;gap:8px;font-size:12.5px;
    color:var(--ink-2);cursor:pointer;margin-bottom:11px}
  .memlearn input{width:14px;height:14px;cursor:pointer;flex-shrink:0}
  #memlist{display:flex;flex-direction:column;gap:7px;max-height:240px;
    overflow-y:auto}
  .memrow{display:flex;align-items:flex-start;gap:9px;padding:8px 10px;
    background:var(--surface);border:1px solid var(--line);border-radius:8px}
  .memrow .mtag{flex-shrink:0;font-size:9.5px;font-weight:700;
    text-transform:uppercase;letter-spacing:.04em;padding:2px 6px;
    border-radius:5px;margin-top:1px}
  .memrow .mtag.style{background:#e7efe7;color:#3f7a4f}
  .memrow .mtag.fact{background:#ece6f3;color:#6b5b8f}
  .memrow .mtext{flex:1;min-width:0;font-size:12.5px;color:var(--ink);
    line-height:1.4;word-break:break-word}
  .memrow .msrc{font-size:10px;color:var(--ink-3)}
  .memrow .mdel{flex-shrink:0;cursor:pointer;color:var(--ink-3);font-size:14px;
    line-height:1;background:none;border:0;padding:2px 4px}
  .memrow .mdel:hover{color:var(--accent-deep)}
  .memempty{color:var(--ink-3);font-size:12px;padding:4px 0}
  .memadd{display:flex;gap:7px;align-items:center;margin-top:9px}
  #memkind{background:var(--raised);color:var(--ink);border:1px solid var(--line-2);
    border-radius:7px;padding:7px 8px;font-size:12px;cursor:pointer}
  #memtext{flex:1;min-width:80px;background:var(--raised);color:var(--ink);
    border:1px solid var(--line-2);border-radius:7px;padding:7px 10px;
    font-size:12px}
  #memtext:focus{outline:none;border-color:var(--accent)}
  #memaddbtn{background:var(--accent);color:#fff;border:0;border-radius:7px;
    padding:7px 12px;font-size:12px;cursor:pointer;flex-shrink:0}
  #memaddbtn:hover{background:var(--accent-deep)}
  #acctlist{display:flex;flex-direction:column;gap:10px;overflow-y:auto}
  .acctrow{display:flex;align-items:center;gap:12px;padding:11px 13px;
    background:var(--surface);border:1px solid var(--line);border-radius:9px}
  .acctrow .ainfo{flex:1;min-width:0}
  .acctrow .alabel{font-weight:600;font-size:13px;color:var(--ink)}
  .acctrow .aemail{font-size:11.5px;color:var(--ink-3);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .acctrow .astatus{font-size:11px;font-weight:600;flex-shrink:0;
    text-align:right;max-width:200px}
  .acctrow .astatus.ok{color:#3f7a4f}
  .acctrow .astatus.no{color:#a3503f}
  .acctrow .aacts{display:flex;gap:6px;flex-shrink:0}
  .acctrow .aacts button{border-radius:7px;padding:7px 11px;font-size:12px;
    cursor:pointer;border:1px solid var(--line-2);background:var(--surface);
    color:var(--ink-2);flex-shrink:0}
  .acctrow .aacts button:disabled{opacity:.55;cursor:default}
  .acctrow .asignin{background:var(--accent);color:#fff;border:0}
  .acctrow .asignin:hover{background:var(--accent-deep)}
  .acctrow .aremove:hover{background:var(--hover);border-color:var(--accent)}
  .acctadd{display:flex;gap:8px;align-items:center;flex-wrap:wrap;
    margin-top:4px;padding:11px 13px;border:1px dashed var(--line-2);
    border-radius:9px}
  #acctnewlabel{flex:1;min-width:160px;background:var(--raised);
    color:var(--ink);border:1px solid var(--line-2);border-radius:7px;
    padding:7px 10px;font-size:12px}
  #acctnewlabel:focus{outline:none;border-color:var(--accent)}
  #acctaddbtn{background:var(--accent);color:#fff;border:0;border-radius:7px;
    padding:7px 13px;font-size:12px;cursor:pointer;flex-shrink:0}
  #acctaddbtn:hover{background:var(--accent-deep)}
  #acctaddbtn:disabled{opacity:.55;cursor:default}
  .aaddmsg{flex-basis:100%;font-size:11px;color:var(--ink-3)}
  /* ---- composer modal: New / Reply / Reply All / Forward -------------- */
  #hdr #composebtn{background:var(--surface);color:var(--ink);
    border:1px solid var(--line-2)}
  #hdr #composebtn:hover{background:var(--hover);border-color:var(--accent)}
  #panel .pacts{margin:10px 0 4px;display:flex;gap:6px;flex-wrap:wrap}
  #panel .pacts button{background:var(--raised);color:var(--ink-2);
    border:1px solid var(--line-2);border-radius:6px;padding:4px 10px;
    font-size:11.5px;cursor:pointer}
  #panel .pacts button:hover{border-color:var(--accent);color:var(--ink)}
  #compose{display:none;position:fixed;inset:0;z-index:30;pointer-events:none}
  .ccard{position:absolute;top:46px;left:50%;transform:translateX(-50%);
    width:min(720px,94vw);height:min(82vh,640px);
    min-width:380px;min-height:300px;max-width:96vw;max-height:94vh;
    pointer-events:auto;resize:both;
    display:flex;flex-direction:column;background:var(--bg);
    border:1px solid var(--line-2);border-radius:12px;
    box-shadow:0 18px 50px rgba(40,32,24,.28);overflow:hidden}
  .chd{display:flex;align-items:center;padding:11px 16px;
    font:500 14px/1.3 var(--serif);color:var(--ink);
    border-bottom:1px solid var(--line)}
  .chd .x{cursor:pointer;color:var(--ink-3);font-size:16px}
  .chd .x:hover{color:var(--accent-deep)}
  /* ---- floating-window chrome shared by compose + Ask Liam ------------- */
  .askhd, .chd{cursor:move;user-select:none}
  .winctl{margin-left:auto;display:flex;align-items:center;gap:8px}
  .winbtn{background:var(--raised);border:1px solid var(--line-2);
    color:var(--ink-3);cursor:pointer;width:24px;height:22px;border-radius:5px;
    font-size:13px;line-height:1;padding:0;display:inline-flex;
    align-items:center;justify-content:center}
  .winbtn:hover{border-color:var(--accent);color:var(--accent-deep)}
  .askcard.winmax, .ccard.winmax{top:46px!important;left:8px!important;
    transform:none!important;width:calc(100vw - 16px)!important;
    height:calc(100vh - 54px)!important;resize:none;
    max-width:none;max-height:none}
  .askcard.winmin, .ccard.winmin{height:auto!important;resize:none}
  .askcard.winmin > :not(.askhd), .ccard.winmin > :not(.chd){display:none}
  #cform{flex:1;display:flex;flex-direction:column;padding:11px 16px;gap:8px;
    overflow-y:auto;min-height:0}
  #cform .crow{display:grid;grid-template-columns:64px 1fr;gap:9px;
    align-items:center}
  #cform .crow.tall{align-items:stretch}
  #cform .clab{color:var(--ink-3);font-size:11px;text-transform:uppercase;
    text-align:right;padding-top:4px}
  #cform select, #cform input[type=text], #cform input[type=email],
  #cform #cbody{
    background:var(--raised);color:var(--ink);border:1px solid var(--line-2);
    border-radius:6px;padding:6px 9px;font-size:12.5px;width:100%;
    font-family:inherit}
  #cform select:focus, #cform input:focus, #cform #cbody:focus{
    outline:none;border-color:var(--accent)}
  /* Rich-text composer body. A contenteditable div lets reply/forward
     render the original HTML message (links, lists, formatting) instead
     of stripped plain text. Quoted blocks land in a styled blockquote. */
  #cform .cbodywrap{display:flex;flex-direction:column;min-width:0}
  #cfmt{display:flex;gap:4px;margin-bottom:6px}
  #cfmt button{background:var(--surface);border:1px solid var(--line-2);
    border-radius:5px;width:28px;height:26px;cursor:pointer;color:var(--ink);
    font-size:13px;line-height:1;padding:0}
  #cfmt button:hover{border-color:var(--accent)}
  #cfmt button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
  #cform #cbody{min-height:200px;max-height:50vh;overflow-y:auto;
    line-height:1.5;font-size:13px;white-space:normal;word-wrap:break-word}
  #cform #cbody:empty:before{content:attr(data-placeholder);
    color:var(--ink-3);pointer-events:none}
  #cform #cbody blockquote{border-left:3px solid var(--line-2);
    margin:8px 0;padding:2px 0 2px 10px;color:var(--ink-2)}
  #cform #cbody img{max-width:100%;height:auto}
  #cform #cbody a{color:var(--accent-deep)}
  #cform #cbody .gmail_quote_head{color:var(--ink-3);font-size:12px;
    margin:10px 0 4px}
  #cform .crow.ccquiet{display:none}                   /* hidden until toggled */
  #cform .ccline{display:flex;justify-content:flex-end;gap:9px;
    font-size:11px;color:var(--ink-3)}
  #cform .ccline button{background:none;border:0;color:var(--accent-deep);
    cursor:pointer;font-size:11px;padding:0}
  #cform .ccline button:hover{text-decoration:underline}
  /* attachment chips */
  #catt{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
  #catt .chip{display:inline-flex;align-items:center;gap:5px;
    background:var(--surface);border:1px solid var(--line-2);
    border-radius:4px;padding:2px 6px 2px 8px;font-size:11px;color:var(--ink-2)}
  #catt .chip .xx{cursor:pointer;color:var(--ink-3);font-size:13px;
    line-height:1}
  #catt .chip .xx:hover{color:var(--accent-deep)}
  #catt label.addfile{background:var(--raised);border:1px dashed var(--line-2);
    color:var(--ink-3);border-radius:4px;padding:2px 8px;font-size:11px;
    cursor:pointer}
  #catt label.addfile:hover{border-color:var(--accent);color:var(--accent-deep)}
  #catt input[type=file]{display:none}
  .cfoot{display:flex;align-items:center;gap:9px;padding:11px 16px;
    border-top:1px solid var(--line);background:var(--surface)}
  .cstatus{color:var(--ink-3);font-size:11.5px;flex:1;min-width:0;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .cstatus.err{color:#a3503f}
  .cstatus.ok{color:#3F6B4A}
  .cfoot .cbtn{background:var(--surface);color:var(--ink-2);
    border:1px solid var(--line-2);border-radius:7px;padding:7px 13px;
    font-size:12px;cursor:pointer}
  .cfoot .cbtn:hover{background:var(--hover)}
  .cfoot .cbtn.primary{background:var(--accent);color:#fff;
    border-color:var(--accent-deep)}
  .cfoot .cbtn.primary:hover{background:var(--accent-deep)}
  .cfoot .cbtn:disabled{opacity:.5;cursor:default}
  /* ✦ Compose with Liam — collapsible AI draft panel inside the composer.
     flex-shrink:0 so the surrounding #cform (a flex column) can't crush the
     panel when a long draft overflows — #cform scrolls instead. */
  #cassist{border:1px solid var(--line-2);border-radius:8px;
    background:var(--tint);overflow:hidden;flex-shrink:0}
  #cassist-toggle{width:100%;text-align:left;background:none;border:0;
    color:var(--accent-deep);font:500 12px/1.2 inherit;cursor:pointer;
    padding:8px 11px}
  #cassist-toggle:hover{text-decoration:underline}
  #cassist-body{display:none;flex-direction:column;gap:7px;padding:0 11px 10px}
  #cassist.open #cassist-body{display:flex}
  #cassist.open #cassist-toggle{padding-bottom:3px}
  #cassist textarea{width:100%;box-sizing:border-box;resize:vertical;
    min-height:46px;background:var(--bg);border:1px solid var(--line-2);
    border-radius:6px;padding:7px 9px;font:inherit;font-size:12.5px;
    color:var(--ink)}
  #cassist textarea:focus{outline:none;border-color:var(--accent)}
  .cassist-foot{display:flex;align-items:center;gap:9px}
  .cassist-status{flex:1;min-width:0;font-size:11px;color:var(--ink-3);
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .cassist-status.err{color:#a3503f}
  .cassist-status.ok{color:#3F6B4A}
  .cassist-note{font-size:10.5px;color:var(--ink-3);line-height:1.4}
  #cassist-draft{background:var(--accent);color:#fff;
    border:1px solid var(--accent-deep);border-radius:7px;padding:6px 13px;
    font-size:12px;cursor:pointer;white-space:nowrap}
  #cassist-draft:hover{background:var(--accent-deep)}
  #cassist-draft:disabled{opacity:.5;cursor:default}
  /* running transcript of the drafting conversation (iterative refinement) */
  #cassist-log{display:none;max-height:124px;overflow-y:auto;font-size:11.5px;
    line-height:1.5;color:var(--ink-2);background:var(--bg);
    border:1px solid var(--line);border-radius:6px;padding:6px 8px}
  #cassist-log .cmpturn{margin:2px 0}
  #cassist-log .cmpturn.err{color:#a3503f}
  #cassist-log .cmpwho{font-weight:600;color:var(--ink-3)}
  #cassist-reset{background:var(--surface);color:var(--ink);
    border:1px solid var(--line-2);border-radius:7px;padding:6px 11px;
    font-size:12px;cursor:pointer;white-space:nowrap}
  #cassist-reset:hover{border-color:var(--accent)}
  #cassist-reset[hidden]{display:none}
  /* recipient autocomplete dropdown — fixed-positioned under the active
     To/Cc/Bcc input; z-index above the modal backdrop. */
  .cdrop{position:fixed;z-index:40;background:var(--bg);
    border:1px solid var(--line-2);border-radius:6px;
    box-shadow:0 8px 22px rgba(40,32,24,.18);
    max-height:240px;overflow-y:auto;font-size:12px;min-width:240px}
  .cdrop .opt{padding:5px 10px;cursor:pointer;display:flex;
    flex-direction:column;gap:1px;line-height:1.3}
  .cdrop .opt:hover,.cdrop .opt.on{background:var(--tint)}
  .cdrop .nm{color:var(--ink)}
  .cdrop .em{color:var(--ink-3);font-size:11px}
</style></head><body class="cols-hidden">
<div id="hdr">
  <div id="leftctl">
    <input type="checkbox" id="f-sel-all" title="Select all visible rows">
    <button id="colstoggle" title="Show filters">›</button>
  </div>
  <button id="back">← All mail</button>
  <button id="selbtn"
    title="Select messages to remove or mark read">☑ Select</button>
  <button id="clear" style="display:none"
    title="Clear search & all filters (Esc)">✕ Clear</button>
  <button id="readbtn" style="display:none"
    title="Mark selected unread messages as read">✓ Mark read (0)</button>
  <button id="notspambtn" style="display:none"
    title="Move selected messages out of Spam (back to Inbox)">📥 Not spam (0)</button>
  <button id="markspambtn" style="display:none"
    title="Mark selected messages as Spam (move out of Inbox)">🚫 Mark spam (0)</button>
  <button id="rmbtn" style="display:none"
    title="Move selected messages to Gmail Trash">⌫ Remove (0)</button>
  <h1 id="title" style="display:none">All mail</h1>
  <span class="sub" id="sub">__COUNT__ messages · __CONVS__ conversations</span>
  <input id="search" type="search"
    title="Prefix a term with | to negate it, e.g. |raimundo hides rows containing raimundo"
    placeholder="Filter by text, or from: to: cc: subject: has:attachment is:unread (| to negate)">
  <button id="askbtn" title="Ask Liam about your mail">✦ Liam</button>
  <button id="composebtn" title="Compose a new message">✎ Compose</button>
  <button id="refresh" style="display:none" title="Pull new mail now">↻ Sync</button>
  <div class="acctf bucketf" id="bucketf"
    title="Filter by mail bucket (primary vs promotions/social/updates/forums/spam)"></div>
  <button id="acctsbtn" title="Settings: email accounts, LLM model, Claude sign-in">⚙ Settings</button>
</div>
<div id="banner">↻ New mail synced — click to reload</div>
<div id="accts">
  <div class="askcard">
    <div class="askhd">⚙ Settings
      <span class="x" id="acctsx">✕</span></div>
    <div class="setbody">
      <div class="setsec">
        <div class="setsechd">Email Accounts</div>
        <div id="acctlist"></div>
      </div>
      <div class="setsec">
        <div class="setsechd">LLM Model</div>
        <div class="setrow">
          <select id="llmmodel">
            <option value="default">Default (Claude Code)</option>
            <option value="opus">Opus 4.8</option>
            <option value="sonnet">Sonnet 4.6</option>
            <option value="haiku">Haiku 4.5</option>
          </select>
          <span id="llmmsg" class="setmsg"></span>
        </div>
        <div class="sethint">Model Liam uses. "Default" uses Claude
          Code's own model.</div>
      </div>
      <div class="setsec">
        <div class="setsechd">Claude OAuth</div>
        <div id="claudeauth"></div>
        <div class="sethint">Liam runs through your Claude Code subscription.
          Sign in here if Liam reports it isn't authenticated.</div>
      </div>
      <div class="setsec">
        <div class="setsechd">Memory</div>
        <div class="memscope" id="memscope">
          <button type="button" class="memtab on" data-scope="ask">Ask</button>
          <button type="button" class="memtab" data-scope="compose">Compose</button>
        </div>
        <label class="memlearn"><input type="checkbox" id="memlearn">
          <span id="memlearnlbl">Learn automatically</span></label>
        <div id="memlist"></div>
        <div class="memadd">
          <select id="memkind">
            <option value="style">style</option>
            <option value="fact">fact</option>
          </select>
          <input id="memtext" type="text" maxlength="240"
            placeholder="Add a memory">
          <button id="memaddbtn">➕ Add</button>
        </div>
        <div class="sethint" id="memhint"></div>
      </div>
    </div>
  </div>
</div>
<div id="ask">
  <div class="askcard">
    <div class="askhd">✦ Liam — your mail assistant
      <span class="winctl">
        <button class="asknew" id="asknew" title="Start a new conversation">+ New chat</button>
        <button class="winbtn" id="amin" title="Minimize">–</button>
        <button class="winbtn" id="amax" title="Maximize / restore">▢</button>
        <span class="x" id="askx">✕</span></span></div>
    <div id="asklog"></div>
    <div class="row2">
      <textarea id="askq" rows="1" placeholder="Ask Liam anything about your mail — follow-up questions welcome"></textarea>
      <button id="asksend">Ask</button>
    </div>
  </div>
</div>
<div id="cols">
  <div class="c-sel"></div>
  <div class="c-acct"><div class="acctf" id="acctf"></div></div>
  <div class="c-date"><input id="f-date" type="search" placeholder="date"
    title="Filter by date. Prefix with | to negate, e.g. |2025"></div>
  <div class="c-from"><input id="f-from" type="search" placeholder="sender"
    title="Filter by sender. Prefix with | to negate, e.g. |raimundo"></div>
  <div class="c-to"><input id="f-to" type="search" placeholder="recipient"
    title="Filter by recipient. Prefix with | to negate, e.g. |raimundo"></div>
  <div class="c-subj"><input id="f-subj" type="search" placeholder="subject"
    title="Filter by subject. Prefix with | to negate, e.g. |invoice"></div>
  <div class="c-snip"><input id="f-snip" type="search" placeholder="body"
    title="Filter by body. Prefix with | to negate, e.g. |unsubscribe"></div>
  <div class="c-att"><select id="f-att" title="Filter by attachment">
    <option value="">ALL</option>
    <option value="yes">YES</option>
    <option value="no">NO</option>
  </select></div>
  <div class="c-pos lpos" title="Position of this message in its conversation">Rel. Id</div>
</div>
<div id="list"><div id="spacer"></div></div>
<div id="thread"></div>
<aside id="panel"></aside>
<div id="compose">
  <div class="ccard">
    <div class="chd"><span id="ctitle">New message</span>
      <span class="winctl">
        <button class="winbtn" id="cmin" title="Minimize">–</button>
        <button class="winbtn" id="cmax" title="Maximize / restore">▢</button>
        <span class="x" id="cx">✕</span></span></div>
    <div id="cform">
      <div class="crow"><span class="clab">From</span>
        <select id="cfrom"></select></div>
      <div class="crow"><span class="clab">To</span>
        <input id="cto" type="text" placeholder="comma-separated emails"></div>
      <div class="ccline">
        <button type="button" id="ccc-toggle">+ Cc</button>
        <button type="button" id="cbcc-toggle">+ Bcc</button></div>
      <div class="crow ccquiet" id="ccc-row"><span class="clab">Cc</span>
        <input id="ccc" type="text" placeholder="comma-separated emails"></div>
      <div class="crow ccquiet" id="cbcc-row"><span class="clab">Bcc</span>
        <input id="cbcc" type="text" placeholder="comma-separated emails"></div>
      <div class="crow"><span class="clab">Subject</span>
        <input id="csubj" type="text"></div>
      <div id="cassist">
        <button type="button" id="cassist-toggle"
          title="Let Liam draft this email for you">✦ Compose with Liam</button>
        <div id="cassist-body">
          <div id="cassist-log"></div>
          <textarea id="cassist-q" rows="2" placeholder="Tell Liam what to write — e.g. &quot;polite reply declining the meeting, suggest next week instead&quot;. Ctrl+Enter to draft."></textarea>
          <div class="cassist-foot">
            <span class="cassist-status" id="cassist-status"></span>
            <button type="button" id="cassist-reset"
              title="Start a fresh draft (forget the current conversation)"
              hidden>↻ New draft</button>
            <button type="button" id="cassist-draft">✦ Draft</button>
          </div>
          <div class="cassist-note">Liam writes the draft into the editor, then
            you can refine it with follow-ups (&quot;shorter&quot;, &quot;warmer&quot;, …).
            Nothing is ever sent until <b>you</b> press <b>Send</b>.</div>
        </div>
      </div>
      <div class="crow tall"><span class="clab">Body</span>
        <div class="cbodywrap">
          <div id="cfmt">
            <button type="button" data-cmd="bold" title="Bold (Ctrl+B)"><b>B</b></button>
            <button type="button" data-cmd="italic" title="Italic (Ctrl+I)"><i>I</i></button>
          </div>
          <div id="cbody" contenteditable="true" role="textbox"
            aria-multiline="true" spellcheck="true"
            data-placeholder="Write your message…"></div></div></div>
      <div class="crow"><span class="clab">Files</span>
        <div id="catt">
          <label class="addfile">+ Attach
            <input type="file" id="cfile" multiple></label>
        </div></div>
    </div>
    <div class="cfoot">
      <span class="cstatus" id="cstatus"></span>
      <button class="cbtn" id="ccancel">Cancel</button>
      <button class="cbtn" id="cdraft" title="Save in Gmail's drafts folder">Save draft</button>
      <button class="cbtn primary" id="csend">Send</button>
    </div>
  </div>
</div>
<div id="cdrop" class="cdrop" style="display:none"></div>
<script id="__app-data" type="application/json">__DATA__</script>
<script>window.__BAKED_VERSION__ = __DATA_VERSION__;</script>
<script>
// Parse the embedded payload as JSON rather than evaluating it as a JS
// literal — ~10× faster for multi-MB payloads. The </script-in-string case
// is handled in render_page by escaping </ → <\/ before substitution.
// `let` (not const) so refreshPayload() can swap the data set in place after
// a sync without forcing a full page reload. The bootstrap is unchanged;
// _rebuildIndices() (defined below) reassigns these on refresh.
let DATA = JSON.parse(document.getElementById("__app-data").textContent);
let MSGS = DATA.msgs, PALETTE = DATA.palette;
const $ = s => document.getElementById(s);
const esc = s => String(s == null ? "" : s)
  .replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
// Prepended to message-body iframe srcdocs. The <meta> stops Chrome offering
// to translate a foreign-language email (the main page carries the same meta).
// <base target="_blank"> makes every link in the email open in a new browser
// tab instead of navigating inside the cramped sandboxed iframe (or doing
// nothing) — paired with the iframe's allow-popups* sandbox flags below.
const NOTRANSLATE =
  '<meta name="google" content="notranslate"><base target="_blank">';
// Wrap each filter-term match in <mark>. Splitting on a capturing regex
// keeps escaping per-piece, so terms never break HTML. Case-insensitive.
const reEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
function hl(text, terms){
  const t = String(text == null ? "" : text);
  const pat = (terms || []).filter(Boolean).map(reEsc).join("|");
  if(!pat) return esc(t);
  return t.split(new RegExp("(" + pat + ")", "ig"))
    .map((p, i) => i % 2 ? "<mark>" + esc(p) + "</mark>" : esc(p)).join("");
}
// Copy text to the clipboard. navigator.clipboard works on 127.0.0.1 (a secure
// context); the textarea+execCommand path is a fallback for anything else.
function copyText(text){
  if(navigator.clipboard && navigator.clipboard.writeText){
    return navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
  }
  fallbackCopy(text);
  return Promise.resolve();
}
function fallbackCopy(text){
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try{ document.execCommand("copy"); }catch(e){}
  document.body.removeChild(ta);
}
// Brief bottom-center confirmation toast (reused by any copy/confirm action).
let _toastTimer = null;
function showToast(msg){
  let t = $("toast");
  if(!t){
    t = document.createElement("div");
    t.id = "toast";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.remove("show"), 1600);
}
// Per-account dot colors. The three primary mailboxes get fixed, recognizable
// hues; any other labels fall through to the palette in encounter order.
const ACCT_COLOR = {
  org:      "#2E9D58",   // green
  gmail:     "#1E7BD8",   // blue
  work: "#FF7A00",   // bright orange
};
const ACCT_PAL = ["#3F9A74","#C0863C","#5670B4","#A368B8","#C25C54"];
MSGS.forEach(m => { if(m.acct && !(m.acct in ACCT_COLOR))
  ACCT_COLOR[m.acct] = ACCT_PAL[Object.keys(ACCT_COLOR).length % ACCT_PAL.length]; });

// Per-bucket (treatment-tier) dot colors + canonical display order. 'primary'
// is real correspondence; the rest are lite tiers. See _common.message_bucket.
const BUCKET_COLOR = {
  primary:    "#2E9D58",   // green  — full treatment
  promotions: "#E0A458",   // amber
  social:     "#6F86D6",   // indigo
  updates:    "#56C596",   // teal
  forums:     "#C98BDB",   // purple
  spam:       "#C25C54",   // red
};
const BUCKET_ORDER = ["primary","promotions","social","updates","forums","spam"];
const BUCKET_LABELS = {
  primary:"Primary", promotions:"Promotions", social:"Social",
  updates:"Updates", forums:"Forums", spam:"Spam",
};

// conversations: convId -> [message index...]
const CONV = {};
MSGS.forEach((m, i) => (CONV[m.conv] || (CONV[m.conv] = [])).push(i));
const dstr = s => (s || "").slice(0, 10);
const who  = m => m.name || m.from || "(unknown)";

// Position of each message within its conversation, oldest = 1 — matches the
// #N numbering in the thread view (openConv). gi -> 1-based index; the list's
// "n / total" column reads from this. Recomputed on live payload swaps.
function computeConvPos(){
  const pos = {};
  for(const cid in CONV){
    CONV[cid].slice()
      .sort((a, b) => (MSGS[a].sent || "").localeCompare(MSGS[b].sent || ""))
      .forEach((gi, k) => pos[gi] = k + 1);
  }
  return pos;
}
let CONVPOS = computeConvPos();

// Read/unread mirrors Gmail's own UNREAD label (m.unread), so a message read
// in any mail client shows as read here too. Opening an unread message (pick)
// marks the WHOLE Gmail conversation read — Gmail, the mobile apps and most
// clients group by thread and keep a conversation bold until EVERY message in
// it is read, so clearing only the opened message would leave the thread
// showing unread in those clients. We clear UNREAD on every still-unread
// message sharing this one's Gmail thread (tid + account) via /api/seen —
// which also drops the label from each Neo4j node — and flip the local flags
// so the list un-bolds on the next render. Live-mode only; on the static file
// the fetch no-ops.
function markRead(gi){
  const m = MSGS[gi];
  if(!m || !m.unread) return;
  // Every still-unread message in the same Gmail thread. Fall back to just
  // this message when it carries no thread id (shouldn't happen for synced
  // mail, but keeps a lone message working).
  const targets = (m.tid
    ? MSGS.filter(o => o && o.unread && o.tid === m.tid
                       && o.acct === m.acct && o.mid)
    : [m]);
  if(!targets.length) return;
  targets.forEach(o => { o.unread = false; });   // optimistic — un-bolds list
  fetch("api/seen", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({messages: targets.map(o =>
      ({mid: o.mid, acct: o.acct}))}),
  }).catch(() => {});
}

/* ===================== LIST VIEW (virtualized) ===================== */
const LROW = 28;
let listIdx = MSGS.map((m, i) => i)
  .sort((a, b) => (MSGS[b].sent || "").localeCompare(MSGS[a].sent || ""));
let view = listIdx, listCur = 0;        // listCur = highlighted row in view
// Bucket (treatment-tier) filter — a multi-select dropdown that replaces the
// old Spam toggle. Default shows ONLY 'primary'; the lite tiers (promotions/
// social/updates/forums/spam) are opt-in. 'primary' is always offered even if
// no such message is currently loaded. `let` so _rebuildIndices can refresh the
// option list after a payload swap; bucketSel (a Set, mutated in place) is
// preserved across the swap, like acctSel.
let BUCKETS = [...new Set(MSGS.map(m => m.bucket).filter(Boolean))]
  .sort((a, b) => BUCKET_ORDER.indexOf(a) - BUCKET_ORDER.indexOf(b));
if(!BUCKETS.includes("primary")) BUCKETS.unshift("primary");
const bucketSel = new Set(["primary"]);   // default: only primary visible
// "Filtered" means the selection differs from the primary-only default — used
// for the ✕ Clear button and the subtitle. (matches() applies the filter
// directly off bucketSel, so this is only about whether to *advertise* it.)
function bucketFiltered(){
  return !(bucketSel.size === 1 && bucketSel.has("primary"));
}

// --- search ----------------------------------------------------------
// Bare words AND-match across every field; from:/to:/cc:/bcc:/subject:/
// body:/account: scope a term; has:attachment filters; "..." is a phrase.
let PEOPLE = DATA.people;
const named = e => (PEOPLE[e] ? PEOPLE[e] + " " + e : e);
// Per-message lowercased haystacks: one string per searchable field. A free
// term ANDs by matching any of these fields (see FREE_FIELDS / matches()).
// We used to also materialise an `all` field that concatenated everything —
// that was a ~19MB string duplication for a 40k-message corpus. Iterating
// the per-field strings on free terms is still sub-millisecond per keystroke.
let HAY = MSGS.map(m => {
  const f = {
    from: m.from + " " + (m.name || ""),
    to:   m.to.map(named).join("  "),
    cc:   m.cc.map(named).join("  "),
    bcc:  m.bcc.map(named).join("  "),
    subj: m.subj || "",
    // In serve_app's lean payload `m.body` is absent — fall back to the
    // snippet so body:/snip-column searches still match something.
    body: m.body || m.snip || "",
    atts: m.atts.join("  "),
    acct: m.acct || "",
    date: (m.sent || "").slice(0, 10),
  };
  for(const k in f) f[k] = f[k].toLowerCase();
  return f;
});
// Order matters only for short-circuit speed: the fields most likely to
// match a typical free term come first, so we break out earlier on average.
// `acct` is deliberately NOT here: account_owner is one of a few short labels
// ("gmail", "org", "work"), so a free term that's a substring of one
// ("gmail", or even "mail") would match EVERY message in that account, swamping
// real hits. The account is still filterable via the multi-select header and an
// explicit account:/acct:/mailbox: term — both scoped to h.acct only.
const FREE_FIELDS = ["subj", "from", "body", "to",
                     "cc", "atts", "bcc", "date"];
const FIELD = {from:"from", de:"from", sender:"from", to:"to", para:"to",
  recipient:"to", cc:"cc", bcc:"bcc", subject:"subj", subj:"subj",
  title:"subj", body:"body", text:"body", account:"acct", acct:"acct",
  mailbox:"acct", file:"atts", attachment:"atts", attachments:"atts"};
let QTOK = [];
// Per-column filter bar — each box ANDs with the global search.
// colF.att is a tri-state: "" (any), "yes" (has attachment), "no" (none).
// Driven by the "ATT:" header <select> — see the f-att handler.
const colF = {date:"", from:"", to:"", subj:"", snip:"", att:""};

// Full-body search. The lean payload carries only a 240-char snippet per
// message, so `m.body` (HAY[i].body) is really the snippet. The body column
// box and body:/text: terms instead search the COMPLETE clean body via the
// server: BODY_HITS maps a bare lowercased term → Set of "mid|acct" keys that
// match the whole message. While a term's fetch is in flight (value null) or
// failed (value false) we fall back to substring-on-snippet so the list never
// blanks mid-type. Free bare global-search terms ALSO consult this set (in
// addition to their synchronous snippet/from/subject match), so a word buried
// deep in an old long body still surfaces — the scan is debounced and one
// request per settled term, not per keystroke.
const BODY_HITS = new Map();
// The distinct bare body terms currently in force (column box + body: tokens +
// free global-search terms), leading "|" negation stripped — what we ask the
// server for.
function activeBodyTerms(){
  const out = new Set();
  for(const t of QTOK){
    // body: scoped terms AND free bare terms both want the full body. A free
    // term matches the 240-char snippet (HAY.body) synchronously; we ALSO scan
    // the complete body server-side so deep-body mentions — common in long
    // older threads — aren't missed, which otherwise skews free-text results
    // toward recent mail (the term happens to land in a recent subject/snippet).
    if((t.f === "body" || t.f === null) && t.v){
      const bare = t.v[0] === "|" ? t.v.slice(1) : t.v;
      if(bare) out.add(bare);
    }
  }
  if(colF.snip){
    const bare = colF.snip[0] === "|" ? colF.snip.slice(1) : colF.snip;
    if(bare) out.add(bare);
  }
  return out;
}
// Fetch the full-body hit set for any active body term we don't have yet, then
// re-filter so the list snaps from snippet-match to whole-body-match. Debounced
// by its caller so typing a word triggers one scan on the settled term, not one
// per keystroke. ThreadingHTTPServer serves these off the version-poll path.
function ensureBodyHits(){
  for(const term of activeBodyTerms()){
    if(BODY_HITS.has(term)) continue;          // cached, in-flight, or failed
    if(term.length < 2){ BODY_HITS.set(term, false); continue; }  // snippet only
    BODY_HITS.set(term, null);                 // in-flight → snippet fallback
    fetch("api/bodysearch?q=" + encodeURIComponent(term), {cache:"no-store"})
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(d => {
        BODY_HITS.set(term, new Set((d.hits||[]).map(h => h[0] + "|" + h[1])));
        rebuildList();                         // upgrade snippet match → full body
      })
      .catch(() => { BODY_HITS.set(term, false); });  // keep snippet fallback
  }
}
// Match one body filter term against the full body via the hit set when we
// have it; otherwise (pending/failed/too-short) fall back to the snippet so the
// view stays populated. `v` may carry a leading "|" for negation.
function bodyTermMatch(m, h, v){
  const neg = v[0] === "|", bare = neg ? v.slice(1) : v;
  if(!bare) return true;                        // bare "|" is inert
  const hits = BODY_HITS.get(bare);
  if(hits instanceof Set){
    const inSet = hits.has(remKey(m));
    return neg ? !inSet : inSet;
  }
  return termMatch(h.body, v);                  // snippet fallback
}
// Debounced so a word typed into the body box triggers one server scan on the
// settled term, not one per keystroke (debounce() is hoisted, defined below).
const _ensureBody = debounce(ensureBodyHits, 250);

// Highlight terms per displayed column. Free terms (no field:) match every
// column; field-scoped terms (subject:, from:, …) and the per-column filter
// boxes match only their own column. Recomputed by rebuildList().
let HLT = {acct:[], date:[], from:[], to:[], subj:[], snip:[]};
function computeHL(){
  // Negated terms ("|raimundo") match rows that DON'T contain the word, so
  // there's nothing to highlight — drop them before building the mark list.
  const pos   = v => v && v[0] !== "|";
  const free  = QTOK.filter(t => t.f === null && pos(t.v)).map(t => t.v);
  const field = f => QTOK.filter(t => t.f === f && pos(t.v)).map(t => t.v);
  const col   = v => pos(v) ? [v] : [];
  HLT = {
    acct: field("acct"),   // free terms no longer match acct (see FREE_FIELDS)
    date: free.concat(col(colF.date)),
    from: free.concat(field("from"), col(colF.from)),
    to:   free.concat(field("to"),   col(colF.to)),
    subj: free.concat(field("subj"), col(colF.subj)),
    snip: free.concat(field("body"), col(colF.snip)),
  };
}

function anyFilter(){
  return QTOK.length > 0 || acctOn() || bucketFiltered()
    || Object.values(colF).some(v => v);
}
function parseQuery(q){
  // → tokens, all ANDed: {f,v} field term · {v} free term · {has} filter
  // \s* after the colon → "from:vial" and "from: vial" both parse the same.
  const toks = [], re = /(\w+):\s*"([^"]*)"|(\w+):\s*(\S+)|"([^"]*)"|(\S+)/g;
  let mm;
  while((mm = re.exec(q)) !== null){
    let field = mm[1] ?? mm[3];
    let val = mm[2] ?? mm[4] ?? mm[5] ?? mm[6];
    if(!val) continue;
    if(field){
      field = field.toLowerCase();
      if(field === "has"){ toks.push({has: val.toLowerCase()}); continue; }
      if(field === "is"){ toks.push({is: val.toLowerCase()}); continue; }
      if(FIELD[field]){ toks.push({f: FIELD[field], v: val.toLowerCase()});
        continue; }
      val = field + ":" + val;          // unknown field → plain term
    }
    toks.push({f: null, v: val.toLowerCase()});
  }
  return toks;
}
// A filter term prefixed with "|" negates: "|raimundo" matches rows whose
// field does NOT contain "raimundo". A bare "|" (nothing after it) is inert,
// so a half-typed negation never blanks the list. Used by every filter — the
// global search (free + field-scoped terms) and all per-column boxes.
function termMatch(hay, v){
  if(v[0] === "|"){
    const t = v.slice(1);
    return t ? !hay.includes(t) : true;
  }
  return hay.includes(v);
}
function matches(i){
  const h = HAY[i], m = MSGS[i];
  // Optimistically hide rows the user just trashed. The server's page-cache
  // rebuild is deferred and may take ~60-90s; the REMOVED set (keyed by
  // mid|acct) keeps the view correct in the meantime, even across a refresh
  // that re-fetches a still-stale payload.
  if(REMOVED.size && REMOVED.has(remKey(m))) return false;
  // Bucket (tier) filter: only the currently-selected buckets are shown. The
  // default selection is primary-only, so lite mail (promotions / social /
  // updates / forums / spam) stays hidden until opted in via the header
  // bucket dropdown. m.bucket is always one of BUCKETS, so an empty selection
  // simply shows nothing (mirrors deselecting every account).
  if(!bucketSel.has(m.bucket)) return false;
  for(const t of QTOK){
    if(t.has !== undefined){
      if((t.has.startsWith("att") || t.has.startsWith("file"))
         && !m.atts.length && !m.inline_img) return false;
      continue;
    }
    if(t.is !== undefined){
      if(t.is === "unread" && !m.unread) return false;
      if(t.is === "read"   &&  m.unread) return false;
      continue;
    }
    if(t.f){
      const ok = t.f === "body" ? bodyTermMatch(m, h, t.v)
                                 : termMatch(h[t.f], t.v);
      if(!ok) return false;
    } else if(t.v[0] === "|"){
      // Negated free term — exclude the row if ANY searchable field contains
      // it (mirror of the positive case below), incl. the full body once its
      // server-side hit set has loaded. Bare "|" is inert.
      const term = t.v.slice(1);
      if(term){
        for(let k = 0; k < FREE_FIELDS.length; k++){
          if(h[FREE_FIELDS[k]].includes(term)) return false;
        }
        const hits = BODY_HITS.get(term);
        if(hits instanceof Set && hits.has(remKey(m))) return false;
      }
    } else {
      // Free term — must appear in at least one searchable field. Iterating
      // FREE_FIELDS replaced the old h.all concat (~50% memory drop) and is
      // still effectively instant: short-circuits on first hit. FREE_FIELDS.body
      // is only the 240-char snippet in the lean payload, so we also accept a
      // hit in the full-body scan once it lands (BODY_HITS, fetched via
      // activeBodyTerms → ensureBodyHits). Until then the snippet match holds
      // and rebuildList re-runs when the set arrives, snapping in deep matches.
      let found = false;
      for(let k = 0; k < FREE_FIELDS.length; k++){
        if(h[FREE_FIELDS[k]].includes(t.v)){ found = true; break; }
      }
      if(!found){
        const hits = BODY_HITS.get(t.v);
        if(hits instanceof Set && hits.has(remKey(m))) found = true;
      }
      if(!found) return false;
    }
  }
  if(acctOn() && !acctSel.has(m.acct)) return false;
  if(colF.date && !termMatch((m.sent || "").toLowerCase(), colF.date))
    return false;
  if(colF.from && !termMatch(h.from, colF.from)) return false;
  if(colF.to   && !termMatch(h.to,   colF.to))   return false;
  if(colF.subj && !termMatch(h.subj, colF.subj)) return false;
  if(colF.snip && !bodyTermMatch(m, h, colF.snip)) return false;
  // Per-message att filter. .filter() calls (el, index, arr), so the extra
  // args matches() receives there are harmless — only `i` is read.
  if(colF.att){
    const has = m.atts.length > 0 || !!m.inline_img;
    if(colF.att === "yes" && !has) return false;
    if(colF.att === "no"  &&  has) return false;
  }
  return true;
}
function updateSub(){
  if(thrOpen) return;
  // Filters other than the bucket tier (search, account, per-column).
  const other = QTOK.length > 0 || acctOn() || Object.values(colF).some(v => v);
  if(other || bucketFiltered()){
    $("sub").textContent = view.length.toLocaleString() + " of "
      + MSGS.length.toLocaleString() + " messages";
    return;
  }
  // Default view: primary only, nothing else filtered. Headline counts what's
  // visible, and flags how much lite mail is hidden so it's discoverable.
  const hidden = MSGS.length - view.length;
  let s = view.length.toLocaleString() + " messages · "
    + Object.keys(CONV).length.toLocaleString() + " conversations";
  if(hidden > 0) s += " · " + hidden.toLocaleString() + " lite hidden";
  $("sub").textContent = s;
}
function rebuildList(){
  computeHL();
  // Kick off (debounced) any full-body lookups the active body filter needs;
  // the sync pass below filters on the snippet until the hits land, then a
  // second rebuildList (from ensureBodyHits) snaps to whole-body matching.
  _ensureBody();
  // Always run matches(): besides user-set filters it applies two always-on
  // partitions — the just-trashed REMOVED set and the bucket-tier filter
  // (matches() hides any message whose bucket isn't selected; the default
  // selection is primary-only). Gating this on anyFilter() would let an
  // unfiltered list bypass both, leaking lite mail and lingering trashed rows.
  view = listIdx.filter(matches);
  listCur = 0;
  $("spacer").style.height = (view.length * LROW) + "px";
  updateSub();
  updateClear();
  renderList();
  if(typeof syncSelectionUI === "function") syncSelectionUI();
}
// One message row (All mail mode).
function toLabel(arr){
  // Friendlier display: name when we know it, email otherwise. The haystack
  // (HAY[i].to) already includes both, so filter/highlight terms still match
  // even when only the email or only the name appears here.
  return arr.map(e => PEOPLE[e] || e).join(", ");
}
function msgRowHTML(m, cell, gi){
  // gi = global MSGS index; used as the checkbox's data key + selection-state.
  const checked = SELECTED.has(gi) ? " checked" : "";
  // Sender / recipient columns display a name; hovering reveals the actual
  // email address(es) and right-clicking copies them (see the #spacer
  // contextmenu handler). data-email carries the raw address for the copy.
  const toEmails = m.to.join(", ");
  const fromTip = m.from ? esc(m.from) + " — right-click to copy" : "";
  const toTip   = toEmails ? esc(toEmails) + " — right-click to copy" : "";
  return `<span class="c-sel"><input type="checkbox" data-i="${gi}"${checked}></span>`
    + `<span class="c-acct lacct"><i class="acctdot" style="background:`
    + `${ACCT_COLOR[m.acct]||"#8F8B80"}"></i>${cell("acct", m.acct)}</span>`
    + `<span class="c-date ldate">${cell("date", dstr(m.sent))}</span>`
    + `<span class="c-from lfrom" title="${fromTip}" `
    + `data-email="${esc(m.from)}">${cell("from", who(m))}</span>`
    + `<span class="c-to lfrom" title="${toTip}" `
    + `data-email="${esc(toEmails)}">${cell("to", toLabel(m.to))}</span>`
    + `<span class="c-subj lsubj">${cell("subj", m.subj)}</span>`
    + `<span class="c-snip lsnip">${cell("snip", m.snip)}</span>`
    + `<span class="c-att">${(m.atts.length || m.inline_img) ? "Yes" : "No"}</span>`
    + `<span class="c-pos lpos">${CONVPOS[gi]} / ${CONV[m.conv].length}</span>`;
}
// Row pool: a stable set of .lrow nodes attached to #spacer that we reuse
// across renders. The previous implementation tore down and recreated ~30
// divs on every scroll event (plus re-attached click listeners); the pool
// only updates `top`, `dataset`, and `innerHTML` on existing nodes. Click
// dispatch is delegated to #spacer just once (see below) so swapping
// innerHTML doesn't lose event handlers.
const _rowPool = [];
function _ensureRow(idx){
  while(_rowPool.length <= idx){
    const el = document.createElement("div");
    el.className = "lrow";
    _rowPool.push(el);
    $("spacer").appendChild(el);
  }
  return _rowPool[idx];
}
function renderList(){
  const box = $("list"), top = box.scrollTop, h = box.clientHeight;
  const first = Math.max(0, Math.floor(top / LROW) - 6);
  const last  = Math.min(view.length, Math.ceil((top + h) / LROW) + 6);
  const need = last - first;
  // Highlight filter matches in the rendered rows — only the ~visible
  // rows are processed, so this stays cheap on a large list.
  const doHL = anyFilter();
  const cell = (col, txt) => doHL ? hl(txt, HLT[col]) : esc(txt);
  // Conversation of the cursor row — its other messages get a paler wash so
  // the whole thread lights up as the cursor moves over any of its members.
  const curConv = (view.length && view[listCur] != null)
    ? MSGS[view[listCur]].conv : null;
  for(let j = 0; j < need; j++){
    const k = first + j;
    const vi = view[k];
    const el = _ensureRow(j);
    let cls = k === listCur ? "lrow cursor"
      : (curConv != null && MSGS[vi].conv === curConv) ? "lrow convlit"
      : "lrow";
    if(MSGS[vi].unread) cls += " unread";
    el.className = cls;
    el.style.top = (k * LROW) + "px";
    el.style.display = "";
    el.dataset.i = vi;
    el.dataset.k = k;
    el.innerHTML = msgRowHTML(MSGS[vi], cell, vi);
  }
  // Pool nodes beyond the visible window: hide rather than detach so the
  // pool keeps its allocated size and we avoid re-creating DOM on scroll.
  for(let j = need; j < _rowPool.length; j++){
    if(_rowPool[j].style.display !== "none"){
      _rowPool[j].style.display = "none";
    }
  }
}
// Single delegated click handler on #spacer — replaces N per-row listeners
// so swapping innerHTML on a pool row doesn't lose its handler. Reads the
// row's global index from data-i and its view position from data-k.
$("spacer").addEventListener("click", e => {
  const cb = e.target.closest('input[type=checkbox]');
  if(cb && cb.dataset.i !== undefined){
    e.stopPropagation();
    const i = +cb.dataset.i;
    if(cb.checked) SELECTED.add(i); else SELECTED.delete(i);
    syncSelectionUI();
    return;
  }
  const row = e.target.closest(".lrow");
  if(!row || row.style.display === "none") return;
  const vi = +row.dataset.i;
  const k = +row.dataset.k;
  listCur = k;
  openConv(vi);
});
// Right-click on a sender/recipient cell copies the email address(es) instead
// of opening the browser's context menu. Delegated on #spacer so it survives
// the row pool's innerHTML swaps.
$("spacer").addEventListener("contextmenu", e => {
  const cell = e.target.closest(".c-from[data-email], .c-to[data-email]");
  if(!cell) return;
  const email = (cell.dataset.email || "").trim();
  if(!email) return;
  e.preventDefault();
  copyText(email).then(() => showToast("Copied: " + email));
});
$("list").addEventListener("scroll", () => requestAnimationFrame(renderList));
// Trailing debounce — keeps filter input responsive on large lists by
// collapsing rapid keystrokes into one rebuild. 80ms is below the
// perceived-delay threshold and well above typing speed.
function debounce(fn, ms){
  let t = null;
  return function(...args){
    clearTimeout(t);
    t = setTimeout(() => fn.apply(this, args), ms);
  };
}
const _refilterSearch = debounce(value => {
  QTOK = parseQuery(value);
  if(thrOpen) showList();        // jump back to the list to show results
  $("list").scrollTop = 0;
  rebuildList();
}, 80);
$("search").addEventListener("input", e => {
  dropSelectionOnFilterChange();
  _refilterSearch(e.target.value.trim().toLowerCase());
});
const COLF_IDS = ["f-date", "f-from", "f-to", "f-subj", "f-snip"];
const _refilterCol = debounce(() => {
  if(thrOpen) showList();
  $("list").scrollTop = 0;
  rebuildList();
}, 80);
[["f-date","date"], ["f-from","from"], ["f-to","to"], ["f-subj","subj"],
 ["f-snip","snip"]].forEach(([id,k]) => {
  $(id).addEventListener("input", e => {
    // colF state must update immediately so other code (e.g. anyFilter())
    // sees the new value; only the rebuild is debounced.
    colF[k] = e.target.value.trim().toLowerCase();
    dropSelectionOnFilterChange();
    _refilterCol();
  });
});

// Attachment header: a 3-option <select>. The collapsed box reads
// "ATT: All/Yes/No"; the open dropdown lists bare ALL/YES/NO. A native
// <select> can't show different text collapsed vs. open, so we swap the
// option labels: reset to bare just before the list opens (mousedown), and
// re-apply the "ATT: " prefix to the selected option afterwards (change/blur).
const ATT_OPT = {"": "ALL",      "yes": "YES",      "no": "NO"};
const ATT_BOX = {"": "ATT: All", "yes": "ATT: Yes", "no": "ATT: No"};
function attBareLabels(){
  for(const o of $("f-att").options) o.textContent = ATT_OPT[o.value];
}
function applyAttHeader(){
  const el = $("f-att");
  el.value = colF.att || "";
  attBareLabels();
  el.options[el.selectedIndex].textContent = ATT_BOX[el.value];
  el.classList.toggle("on", !!colF.att);
}
$("f-att").addEventListener("mousedown", attBareLabels);
$("f-att").addEventListener("blur", applyAttHeader);
$("f-att").addEventListener("change", e => {
  colF.att = e.target.value || "";
  dropSelectionOnFilterChange();
  applyAttHeader();
  if(thrOpen) showList();
  $("list").scrollTop = 0;
  rebuildList();
});
applyAttHeader();                  // set the initial "ATT: All" collapsed label

// --- selection + remove --------------------------------------------------
// Set of global MSGS indices currently checked. Survives filter changes so
// you can select a few, refilter, and still see your earlier selection in
// the "⌫ Remove (N)" button. The actual checked state on a row's <input>
// is rebuilt from this set on each renderList pass.
const SELECTED = new Set();
// Stable keys (mid|acct) of messages the user trashed in this session.
// matches() skips them so they vanish from the list the instant the server
// replies — even though the server-side page cache won't catch up for ~60-90s
// while _purge_files_and_rebuild runs in the background. Keyed (not indexed)
// and STICKY across refreshes: refreshPayload only forgets a key once the
// freshly-fetched payload no longer contains it, so a trashed message can't
// flicker back into view if a sync/refresh lands before the purge rebuild.
const REMOVED = new Set();
const remKey = m => m ? (m.mid + "|" + m.acct) : null;
// Select mode is opt-in via the header's ☑ Select toggle. Outside select
// mode the checkbox column is hidden (CSS body:not(.select-mode) .c-sel),
// the Remove button is hidden, and SELECTED is kept empty.
let selectMode = false;

// Both spam-reclassify buttons are available in select mode (no conversation
// drilled in). With the unified bucket filter there's no single "spam view" to
// key off, so visibility is just select-mode; syncSelectionUI() then ENABLES
// each only when the selection contains relevant rows (Not-spam ⇐ some spam
// selected, Mark-spam ⇐ some non-spam selected). Self-gating so it can be
// called from anywhere the relevant state changes (select mode, thread back).
function updateSpamButtons(){
  const show = (selectMode && !thrOpen) ? "inline-block" : "none";
  $("notspambtn").style.display = show;
  $("markspambtn").style.display = show;
}

function enterSelectMode(){
  selectMode = true;
  document.body.classList.add("select-mode");
  $("selbtn").textContent = "✕ Cancel";
  $("selbtn").title = "Exit selection mode";
  $("rmbtn").style.display = "inline-block";
  $("readbtn").style.display = "inline-block";
  updateSpamButtons();
  syncSelectionUI();
}

function exitSelectMode(){
  selectMode = false;
  SELECTED.clear();
  document.body.classList.remove("select-mode");
  $("selbtn").textContent = "☑ Select";
  $("selbtn").title = "Select messages to remove or mark read";
  $("rmbtn").style.display = "none";
  $("readbtn").style.display = "none";
  $("notspambtn").style.display = "none";
  $("markspambtn").style.display = "none";
  // Re-render so any visible row checkboxes flip back to unchecked state.
  renderList();
  syncSelectionUI();
}

$("selbtn").addEventListener("click", () => {
  selectMode ? exitSelectMode() : enterSelectMode();
});

// Selection is scoped to the active filter — a destructive action must never
// touch rows hidden by the current filter. So when the user changes the
// filter/search/account, drop the current selection (you re-select within the
// new view). Programmatic rebuilds don't call this: refreshPayload restores the
// selection by key across a sync.
function dropSelectionOnFilterChange(){
  if(!SELECTED.size) return;
  SELECTED.clear();
  renderList();        // uncheck the visible row checkboxes immediately
  syncSelectionUI();   // reset Remove / Mark read counts + master checkbox
}
function syncSelectionUI(){
  // Counts reflect only selected rows that are CURRENTLY VISIBLE (in view), so
  // they stay correct as filters/search/sync change what's on screen — and
  // match exactly what Remove / Mark read act on (both scope to the view too).
  // Mark-read further narrows to the UNREAD visible-selected rows; Not-spam
  // narrows to the SPAM-labelled ones (all of them, in the spam view) and
  // Mark-spam to the non-spam ones (all of them, in the main view).
  let nSel = 0, nUnread = 0, nSpam = 0, nNotSpam = 0;
  for(const vi of view){
    if(SELECTED.has(vi)){
      nSel++;
      if(MSGS[vi] && MSGS[vi].unread) nUnread++;
      if(MSGS[vi] && MSGS[vi].spam) nSpam++;
      else if(MSGS[vi]) nNotSpam++;
    }
  }
  const btn = $("rmbtn");
  btn.textContent = `⌫ Remove (${nSel})`;
  btn.disabled = nSel === 0;
  const rbtn = $("readbtn");
  rbtn.textContent = `✓ Mark read (${nUnread})`;
  rbtn.disabled = nUnread === 0;
  const nsbtn = $("notspambtn");
  nsbtn.textContent = `📥 Not spam (${nSpam})`;
  nsbtn.disabled = nSpam === 0;
  const msbtn = $("markspambtn");
  msbtn.textContent = `🚫 Mark spam (${nNotSpam})`;
  msbtn.disabled = nNotSpam === 0;
  // Pause auto-sync while a selection is armed: an incremental pull would shift
  // MSGS indices and could act on the wrong messages. See updateSyncHold().
  updateSyncHold(selectMode && nSel > 0);
  // Master checkbox: checked when EVERY visible row is selected, indeterminate
  // when only some are, unchecked when none.
  const master = $("f-sel-all");
  if(!master) return;
  master.checked = nSel > 0 && nSel === view.length;
  master.indeterminate = nSel > 0 && nSel < view.length;
}

// Server hold heartbeat — keeps the background sync_loop paused while the
// user has the Remove button armed. The server hold has a TTL (we extend
// it by HOLD_TTL_SECONDS each beat), so a closed tab or crashed page
// auto-releases without leaving sync stuck off.
const HOLD_TTL_SECONDS = 60;
const HOLD_BEAT_MS = 25000;
let _holdTimer = null;
function _sendHold(seconds){
  try{
    fetch("api/sync/hold", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({seconds}),
    });
  }catch(e){ /* no server — nothing to hold */ }
}
function updateSyncHold(active){
  if(active && _holdTimer == null){
    _sendHold(HOLD_TTL_SECONDS);
    _holdTimer = setInterval(() => _sendHold(HOLD_TTL_SECONDS),
                             HOLD_BEAT_MS);
  }else if(!active && _holdTimer != null){
    clearInterval(_holdTimer);
    _holdTimer = null;
    _sendHold(0);
  }
}

$("f-sel-all").addEventListener("click", () => {
  // Click — not change — so we can read .checked BEFORE the browser flips
  // indeterminate→checked. Behavior: if any visible rows are unselected,
  // select them all; otherwise clear the visible selection.
  const master = $("f-sel-all");
  // After click, .checked reflects the new state.
  if(master.checked){
    for(const vi of view) SELECTED.add(vi);
  } else {
    for(const vi of view) SELECTED.delete(vi);
  }
  renderList();         // refresh per-row checkboxes
  syncSelectionUI();
});

// Kick off a full, all-accounts sync and refresh-in-place when it lands —
// the same path as the header ↻ Sync button (no syncParams = every account).
// Called after Remove / Mark read so server truth and the page cache catch up
// immediately instead of waiting for the next background sync. The selection
// has already been cleared (exitSelectMode released the sync hold), so this is
// safe to fire; we don't await it — it manages its own polling + refresh.
function triggerFullSync(){
  const rb = $("refresh");
  rb.textContent = "↻ syncing…";
  rb.disabled = true;
  triggerSyncAndReload();
}

async function removeSelected(){
  if(!SELECTED.size) return;
  // Trash only the selected rows that are currently visible — matches the
  // Remove (N) count. msgs is the wire payload (no global indices — those are
  // an app-internal concept).
  const targets = view.filter(i =>
    SELECTED.has(i) && MSGS[i] && MSGS[i].mid && MSGS[i].acct);
  if(!targets.length){ SELECTED.clear(); syncSelectionUI(); return; }
  const msgs = targets.map(i => ({mid: MSGS[i].mid, acct: MSGS[i].acct}));
  if(!confirm(`Move ${msgs.length} selected message(s) to Gmail Trash?\n\n`
    + `Only messages selected in the current filter/view are removed — `
    + `anything hidden by the filter is left untouched.\n\n`
    + `They'll be recoverable from Gmail's UI for 30 days, then auto-purged.`)){
    return;
  }
  const btn = $("rmbtn");
  btn.disabled = true;
  btn.textContent = "Removing…";
  try{
    const r = await fetch("api/trash", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({messages: msgs}),
    });
    const d = await r.json();
    if(!d.ok && !d.trashed){
      const err = (d.failed && d.failed[0] && d.failed[0].error)
        || d.error || "unknown";
      alert("Remove failed: " + err);
      btn.disabled = false; syncSelectionUI();
      return;
    }
    if(d.failed && d.failed.length){
      alert(`Removed ${d.trashed} message(s). ${d.failed.length} failed:\n`
        + d.failed.slice(0,3).map(f => `  ${f.mid}: ${f.error}`).join("\n")
        + (d.failed.length > 3 ? `\n  …and ${d.failed.length - 3} more` : ""));
    }
    // Mark trashed messages (by stable key) as removed in the local view. The
    // server's page-cache rebuild runs in the background and eventually bumps
    // the version; until then — and across any sync/refresh in between — the
    // REMOVED set keeps these rows hidden so they can't reappear.
    for(const i of targets) REMOVED.add(remKey(MSGS[i]));
    exitSelectMode();          // also clears SELECTED, hides Remove btn
    rebuildList();             // re-filter view, drop the trashed rows
    triggerFullSync();         // pull every account so server truth catches up
  }catch(e){
    alert("Network error: " + e);
    btn.disabled = false; syncSelectionUI();
  }
}
$("rmbtn").addEventListener("click", removeSelected);

// Mark the unread members of the selection as read — clears UNREAD in Gmail
// (and the Neo4j node) via /api/seen, then un-bolds them locally. Sync is
// already held while a selection is armed (see updateSyncHold), so the global
// indices in `targets` stay valid across the await.
async function markReadSelected(){
  // Only the visible, unread, selected rows — matches the Mark read (N) count.
  const targets = view.filter(i =>
    SELECTED.has(i) && MSGS[i] && MSGS[i].unread && MSGS[i].mid && MSGS[i].acct);
  if(!targets.length) return;
  const msgs = targets.map(i => ({mid: MSGS[i].mid, acct: MSGS[i].acct}));
  if(!confirm(`Mark ${msgs.length} selected message(s) as read?\n\n`
    + `Only messages selected in the current filter/view are marked — `
    + `anything hidden by the filter is left untouched.`)){
    return;
  }
  const btn = $("readbtn");
  btn.disabled = true;
  btn.textContent = "Marking…";
  try{
    const r = await fetch("api/seen", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({messages: msgs}),
    });
    const d = await r.json();
    if(!d.ok && !d.marked){
      const err = (d.failed && d.failed[0] && d.failed[0].error)
        || d.error || "unknown";
      alert("Mark read failed: " + err);
      btn.disabled = false; syncSelectionUI();
      return;
    }
    if(d.failed && d.failed.length){
      alert(`Marked ${d.marked} message(s) read. ${d.failed.length} failed:\n`
        + d.failed.slice(0,3).map(f => `  ${f.mid}: ${f.error}`).join("\n")
        + (d.failed.length > 3 ? `\n  …and ${d.failed.length - 3} more` : ""));
    }
    for(const i of targets) MSGS[i].unread = false;   // optimistic un-bold
    exitSelectMode();          // clears SELECTED, hides the select-mode buttons
    rebuildList();             // re-render (drops bold; re-applies is:unread)
    triggerFullSync();         // pull every account so server truth catches up
  }catch(e){
    alert("Network error: " + e);
    btn.disabled = false; syncSelectionUI();
  }
}
$("readbtn").addEventListener("click", markReadSelected);

// Move the selected spam messages out of Spam — removes the SPAM label in
// Gmail (and the Neo4j node) and restores INBOX via /api/notspam, then flips
// them to non-spam locally so they leave the spam page and reappear in the
// inbox. Mirrors markReadSelected; only meaningful on the spam page.
async function markNotSpamSelected(){
  // Only the visible, spam-labelled, selected rows — matches Not spam (N).
  const targets = view.filter(i =>
    SELECTED.has(i) && MSGS[i] && MSGS[i].spam && MSGS[i].mid && MSGS[i].acct);
  if(!targets.length) return;
  const msgs = targets.map(i => ({mid: MSGS[i].mid, acct: MSGS[i].acct}));
  if(!confirm(`Move ${msgs.length} selected message(s) out of Spam?\n\n`
    + `Only messages selected in the current view are moved — anything `
    + `hidden by the filter is left untouched.\n\n`
    + `They'll be unmarked as spam and returned to the Inbox.`)){
    return;
  }
  const btn = $("notspambtn");
  btn.disabled = true;
  btn.textContent = "Moving…";
  try{
    const r = await fetch("api/notspam", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({messages: msgs}),
    });
    const d = await r.json();
    if(!d.ok && !d.unspammed){
      const err = (d.failed && d.failed[0] && d.failed[0].error)
        || d.error || "unknown";
      alert("Not spam failed: " + err);
      btn.disabled = false; syncSelectionUI();
      return;
    }
    if(d.failed && d.failed.length){
      alert(`Moved ${d.unspammed} message(s) out of spam. ${d.failed.length} failed:\n`
        + d.failed.slice(0,3).map(f => `  ${f.mid}: ${f.error}`).join("\n")
        + (d.failed.length > 3 ? `\n  …and ${d.failed.length - 3} more` : ""));
    }
    // optimistic: leaves the spam bucket. We can't know the original category
    // client-side, so assume 'primary' (the common case); a later payload
    // refresh corrects it from the server-derived bucket if it was categorized.
    for(const i of targets){ MSGS[i].spam = false; MSGS[i].bucket = "primary"; }
    exitSelectMode();          // clears SELECTED, hides the select-mode buttons
    rebuildList();             // re-filter (spam bucket drops them; primary keeps)
    triggerFullSync();         // pull every account so server truth catches up
  }catch(e){
    alert("Network error: " + e);
    btn.disabled = false; syncSelectionUI();
  }
}
$("notspambtn").addEventListener("click", markNotSpamSelected);

// Mark the selected messages as spam — adds the SPAM label and removes INBOX
// in Gmail (and the Neo4j node) via /api/markspam, then flips them to spam
// locally so they leave the main page and appear on the spam page. The inverse
// of markNotSpamSelected; only meaningful on the main mail page.
async function markAsSpamSelected(){
  // Only the visible, non-spam, selected rows — matches Mark spam (N).
  const targets = view.filter(i =>
    SELECTED.has(i) && MSGS[i] && !MSGS[i].spam && MSGS[i].mid && MSGS[i].acct);
  if(!targets.length) return;
  const msgs = targets.map(i => ({mid: MSGS[i].mid, acct: MSGS[i].acct}));
  if(!confirm(`Mark ${msgs.length} selected message(s) as Spam?\n\n`
    + `Only messages selected in the current view are moved — anything `
    + `hidden by the filter is left untouched.\n\n`
    + `They'll be labelled spam and removed from the Inbox.`)){
    return;
  }
  const btn = $("markspambtn");
  btn.disabled = true;
  btn.textContent = "Moving…";
  try{
    const r = await fetch("api/markspam", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({messages: msgs}),
    });
    const d = await r.json();
    if(!d.ok && !d.spammed){
      const err = (d.failed && d.failed[0] && d.failed[0].error)
        || d.error || "unknown";
      alert("Mark spam failed: " + err);
      btn.disabled = false; syncSelectionUI();
      return;
    }
    if(d.failed && d.failed.length){
      alert(`Marked ${d.spammed} message(s) as spam. ${d.failed.length} failed:\n`
        + d.failed.slice(0,3).map(f => `  ${f.mid}: ${f.error}`).join("\n")
        + (d.failed.length > 3 ? `\n  …and ${d.failed.length - 3} more` : ""));
    }
    // optimistic: moves into the spam bucket (leaves the primary view)
    for(const i of targets){ MSGS[i].spam = true; MSGS[i].bucket = "spam"; }
    exitSelectMode();          // clears SELECTED, hides the select-mode buttons
    rebuildList();             // re-filter (primary view drops them; spam keeps)
    triggerFullSync();         // pull every account so server truth catches up
  }catch(e){
    alert("Network error: " + e);
    btn.disabled = false; syncSelectionUI();
  }
}
$("markspambtn").addEventListener("click", markAsSpamSelected);

// --- account filter: a multi-select dropdown (replaces the text box) -----
// `let` so _rebuildIndices() can update the list after a payload refresh
// (e.g. a brand-new mailbox appearing). acctSel is preserved across refresh
// — the user's filter survives a sync.
let ACCTS = [...new Set(MSGS.map(m => m.acct).filter(Boolean))].sort();
const acctSel = new Set(ACCTS);              // all selected = no filtering
function acctOn(){ return acctSel.size !== ACCTS.length; }
function acctLabel(){
  const n = acctSel.size;
  if(n === ACCTS.length) return "All";
  if(n === 0) return "None";
  if(n === 1) return [...acctSel][0];
  return n + "/" + ACCTS.length;
}
function syncAcct(){
  const w = $("acctf");
  w.querySelector(".lbl").textContent = acctLabel();
  $("acct-all").checked = acctSel.size === ACCTS.length;
  w.querySelectorAll(".opt input[data-a]").forEach(cb =>
    cb.checked = acctSel.has(cb.dataset.a));
}
function buildAcctFilter(){
  const w = $("acctf");
  const opts = ACCTS.map(a =>
    `<label class="opt"><input type="checkbox" data-a="${esc(a)}">`
    + `<i class="acctdot" style="background:${ACCT_COLOR[a]||"#8F8B80"}"></i>`
    + `<span>${esc(a)}</span></label>`).join("");
  w.innerHTML = '<div class="btn"><span class="lbl"></span>'
    + '<span class="car">▾</span></div><div class="menu">'
    + '<label class="opt all"><input type="checkbox" id="acct-all">'
    + '<i class="acctdot" style="visibility:hidden"></i>'   // align label
    + '<span>All accounts</span></label>' + opts + '</div>';
  w.querySelector(".btn").addEventListener("click", e => {
    e.stopPropagation();
    w.classList.toggle("open");
  });
  w.querySelectorAll(".opt input").forEach(cb => {
    cb.addEventListener("change", () => {
      if(cb.id === "acct-all"){
        acctSel.clear();
        if(cb.checked) ACCTS.forEach(a => acctSel.add(a));
      } else {
        cb.checked ? acctSel.add(cb.dataset.a) : acctSel.delete(cb.dataset.a);
      }
      dropSelectionOnFilterChange();
      syncAcct();
      if(thrOpen) showList();
      $("list").scrollTop = 0;
      rebuildList();
    });
  });
  syncAcct();
}
// close the dropdown when clicking anywhere outside it
document.addEventListener("click", e => {
  const w = $("acctf");
  if(w && !w.contains(e.target)) w.classList.remove("open");
});

// --- clear: wipe the global search and all column filters in one
// action. The button shows only while something is active;
// Escape triggers it too (see the document keydown handler). --------------
function updateClear(){
  $("clear").style.display = anyFilter() ? "inline-block" : "none";
}
function clearAll(){
  $("search").value = "";
  QTOK = [];
  COLF_IDS.forEach(id => { $(id).value = ""; });
  for(const k in colF) colF[k] = "";
  applyAttHeader();              // re-sync the "ATT:" header label/highlight
  acctSel.clear();
  ACCTS.forEach(a => acctSel.add(a));        // all accounts back on
  syncAcct();
  $("acctf").classList.remove("open");
  bucketSel.clear();
  bucketSel.add("primary");                  // back to primary-only default
  syncBucket();
  $("bucketf").classList.remove("open");
  dropSelectionOnFilterChange();
  if(thrOpen) showList();
  $("list").scrollTop = 0;
  rebuildList();                 // recomputes the view and hides the button
  $("search").focus();
}
$("clear").addEventListener("click", clearAll);

/* ===================== THREAD VIEW (railroad) ====================== */
const ROW = 30, LANE = 22, PAD = 16, DOT = 5, GAP_DAYS = 2;
const REL = {reply:"↩ replies to", forward:"⏩ forwards",
  "follow-up":"⤷ follows up", root:"● thread start"};
const RELC = {reply:"#3F7CAE", forward:"#BD5C4E",
  "follow-up":"#8A877C", root:"#B07D2B"};
let thrOpen = false, curRows = [], curNum = {}, sel = null;
const colx = c => PAD + c * LANE;

function assignLanes(rows, parentRow, color, dash){
  // git-style lane packing; rows newest→oldest. Lanes hold the row a
  // descending line points at, plus the row that owns it.
  const lanes = [], draw = []; let maxc = 0;
  rows.forEach((m, r) => {
    const kids = [];
    lanes.forEach((ln, i) => { if(ln && ln.t === r) kids.push(i); });
    let cm;
    if(kids.length) cm = Math.min(...kids);
    else { cm = lanes.indexOf(null); if(cm < 0){ cm = lanes.length; lanes.push(null); } }
    const segs = [];
    lanes.forEach((ln, i) => { if(!ln) return;
      maxc = Math.max(maxc, i);
      segs.push(ln.t === r ? ["in", i, cm, ln.c, ln.o, ln.d]
                           : ["thru", i, i, ln.c, ln.o, ln.d]); });
    const pr = parentRow[r];
    if(pr != null) segs.push(["out", cm, cm, color[r], r, dash[r]]);
    draw.push({col: cm, color: color[r], root: pr == null, segs});
    maxc = Math.max(maxc, cm);
    kids.forEach(i => { if(i !== cm) lanes[i] = null; });
    lanes[cm] = pr != null ? {t: pr, c: color[r], o: r, d: dash[r]} : null;
  });
  return {draw, ncols: maxc + 1};
}
function seg(x1,y1,x2,y2,c,o,d,curve){
  const da = d ? ' stroke-dasharray="2.6 3.2"' : '';
  if(curve){ const m=(y1+y2)/2;
    return `<path class="ln" data-o="${o}" d="M ${x1},${y1} C ${x1},${m} `
      + `${x2},${m} ${x2},${y2}" fill="none" stroke="${c}" `
      + `stroke-width="2.2"${da}/>`; }
  return `<line class="ln" data-o="${o}" x1="${x1}" y1="${y1}" x2="${x2}" `
    + `y2="${y2}" stroke="${c}" stroke-width="2.2"${da}/>`;
}
function railSvg(d, w){
  const mid = ROW/2; let s = `<svg class="rail" width="${w}" height="${ROW}">`;
  d.segs.forEach(([k,a,b,c,o,dash]) => {
    const x1=colx(a), x2=colx(b);
    if(k==="thru")      s += seg(x1,0,x1,ROW,c,o,dash);
    else if(k==="out")  s += seg(x1,mid,x1,ROW,c,o,dash);
    else if(a===b)      s += seg(x1,0,x1,mid,c,o,dash);
    else                s += seg(x1,0,x2,mid,c,o,dash,true);
  });
  const nx = colx(d.col);
  if(d.root) s += `<circle cx="${nx}" cy="${mid}" r="${DOT+2.5}" fill="none" `
    + `stroke="#141413" stroke-width="1.6"/>`;
  s += `<circle cx="${nx}" cy="${mid}" r="${DOT}" fill="${d.color}" `
    + `stroke="#FAF9F5" stroke-width="2"/></svg>`;
  return s;
}
function gapText(ms){
  const d = ms/864e5;
  if(d < 14) return Math.round(d)+" day"+(Math.round(d)===1?"":"s");
  if(d < 70) return Math.round(d/7)+" weeks";
  return Math.round(d/30)+" months";
}
function openConv(focus){
  litSender = null;                       // fresh conversation, no spotlight
  const ids = CONV[MSGS[focus].conv].slice()
    .sort((a,b) => (MSGS[a].sent||"").localeCompare(MSGS[b].sent||""));
  curNum = {}; ids.forEach((id,k) => curNum[id] = k+1);
  const sc = {};
  ids.forEach(id => { const f = MSGS[id].from;
    if(f && !(f in sc)) sc[f] = PALETTE[Object.keys(sc).length % PALETTE.length]; });
  const rows = ids.slice().reverse();                 // newest at top
  curRows = rows;
  const pos = {}; rows.forEach((id,r) => pos[id] = r);
  const parentRow = rows.map(id => {
    const p = MSGS[id].par; return (p >= 0 && p in pos) ? pos[p] : null; });
  const color = rows.map(id => sc[MSGS[id].from] || "#928E83");
  const dash  = rows.map(id => MSGS[id].kind === "follow-up");
  const {draw, ncols} = assignLanes(rows, parentRow, color, dash);
  const w = 2*PAD + Math.max(ncols-1,0)*LANE;

  let html = '<div id="legend">' + Object.entries(sc).map(([s,c]) =>
    `<span class="legp" data-s="${esc(s)}"><i style="background:${c}"></i>`
    + `${esc(s)}</span>`).join("") + '</div>';
  rows.forEach((id,r) => {
    const m = MSGS[id], p = parentRow[r];
    let rel = REL[m.kind];
    if(m.kind !== "root" && p != null) rel += " #"+curNum[rows[p]];
    html += `<div class="row" data-r="${r}" data-i="${id}">${railSvg(draw[r],w)}`
      + `<div class="msg"><span class="num" style="color:${draw[r].color}">`
      + `#${curNum[id]}</span><span class="txt">${esc(m.snip||m.subj)}</span>`
      + `<span class="who">${esc(who(m))}</span>`
      + `<span class="date">${esc(dstr(m.sent))}</span>`
      + `<span class="rel" style="color:${RELC[m.kind]}">${esc(rel)}</span>`
      + `</div></div>`;
    if(r < rows.length-1){
      const gap = Date.parse(m.sent) - Date.parse(MSGS[rows[r+1]].sent);
      if(gap > GAP_DAYS*864e5){
        html += `<div class="divider">${dividerSvg(draw,r,w)}`
          + `<span class="gap">⌄ ${esc(gapText(gap))} earlier</span></div>`;
      }
    }
  });
  $("thread").innerHTML = html;
  wireThread();
  $("title").textContent = MSGS[ids[0]].subj;
  $("title").style.display = "";
  $("sub").textContent = ids.length + " messages";
  $("list").style.display = "none";
  $("cols").style.display = "none";
  $("thread").style.display = "block";
  $("back").textContent = "← All mail";
  $("back").style.display = "inline-block";
  // Selection is a list-only concept — hide its controls on the conversation
  // page. showList restores them (and re-shows the action buttons if a
  // selection was still in progress).
  $("selbtn").style.display = "none";
  $("rmbtn").style.display = "none";
  $("readbtn").style.display = "none";
  $("notspambtn").style.display = "none";
  $("markspambtn").style.display = "none";
  $("leftctl").style.display = "none";
  $("bucketf").style.display = "none";
  thrOpen = true;
  const fr = pos[focus];
  const fe = $("thread").querySelector(`.row[data-r="${fr}"]`);
  if(fe){ fe.scrollIntoView({block:"center"}); }
  pick(focus);
}
function dividerSvg(draw, r, w){
  // vertical lane lines passing through the gap, from row r's bottom state
  let s = `<svg class="rail" width="${w}" height="22">`;
  draw[r].segs.forEach(([k,a,b,c,o,d]) => {
    if(k === "thru" || k === "out"){
      const da = d ? ' stroke-dasharray="2.6 3.2"' : '';
      s += `<line class="ln" data-o="${o}" x1="${colx(a)}" y1="0" `
        + `x2="${colx(a)}" y2="22" stroke="${c}" stroke-width="2.2"${da}/>`;
    }
  });
  return s + "</svg>";
}
function wireThread(){
  $("thread").querySelectorAll(".row").forEach(el => {
    const i = +el.dataset.i, r = +el.dataset.r;
    el.addEventListener("click", () => pick(i));
    const whoEl = el.querySelector(".who");
    if(whoEl) whoEl.addEventListener("click", e => {
      e.stopPropagation();               // spotlight the sender, don't open
      toggleSender(MSGS[i].from);
    });
    el.addEventListener("mouseenter", () => {
      const ids = curRows, kids = [r];
      ids.forEach((id,rr) => { if(MSGS[id].par >= 0 &&
        curRows.indexOf(MSGS[id].par) === r) kids.push(rr); });
      $("thread").querySelectorAll(".ln").forEach(ln => {
        if(kids.includes(+ln.dataset.o)) ln.classList.add("litln"); });
    });
    el.addEventListener("mouseleave", () =>
      $("thread").querySelectorAll(".litln").forEach(
        ln => ln.classList.remove("litln")));
  });
  $("thread").querySelectorAll("#legend .legp").forEach(sp =>
    sp.addEventListener("click", () => toggleSender(sp.dataset.s)));
}

// Spotlight one person: clicking their legend entry or a row's sender name
// brightens the messages they sent and dims the rest. Clicking the same
// person again clears it.
let litSender = null;
function applySenderLit(){
  const on = litSender !== null;
  $("thread").querySelectorAll(".row").forEach(el => {
    const mine = on && MSGS[+el.dataset.i].from === litSender;
    el.classList.toggle("litby", mine);
    el.classList.toggle("dim", on && !mine);
  });
  $("thread").querySelectorAll("#legend .legp").forEach(sp =>
    sp.classList.toggle("lega", on && sp.dataset.s === litSender));
}
function toggleSender(sender){
  if(!sender) return;
  litSender = litSender === sender ? null : sender;
  applySenderLit();
}

/* ===================== DETAIL PANEL ================================ */
// Returns the metadata block (pnum + subject + prel + per-field rows) for a
// message. Shared by pick() (embedded panel) and popoutPanel() (new window)
// so the two views stay structurally identical.
function panelInnerHTML(m, num, parNum){
  const row = (k,v) => `<div class="pr"><span class="pk">${k}</span>`
    + `<span class="pv">${v}</span></div>`;
  let h = `<span class="pnum">#${num || ""}</span>`
    + `<h2>${esc(m.subj)}</h2>`;
  let rel = REL[m.kind];
  if(m.kind !== "root" && parNum) rel += " #" + parNum;
  h += `<div class="prel">${esc(rel)}</div>`;
  if(m.sent) h += row("sent", esc(dstr(m.sent)));
  h += row("from", esc(who(m) + (m.name ? " <"+m.from+">" : "")));
  if(m.to.length)  h += row("to",  esc(m.to.join(", ")));
  if(m.cc.length)  h += row("cc",  esc(m.cc.join(", ")));
  if(m.bcc.length) h += row("bcc", esc(m.bcc.join(", ")));
  if(m.atts.length)h += row("files", esc(m.atts.join(", ")));
  return h;
}

function pick(i){
  sel = i;
  markRead(i);
  $("thread").querySelectorAll(".row").forEach(el =>
    el.classList.toggle("selected", +el.dataset.i === i));
  const m = MSGS[i];
  const num = curNum[i] || "";
  const parNum = (m.kind !== "root" && m.par >= 0) ? (curNum[m.par] || "") : "";
  // Header buttons. Popout goes FIRST in DOM order so it floats to the left
  // of ✕ (CSS floats stack in source order rightmost-first).
  let h = '<span class="x" onclick="closePanel()" title="Close">✕</span>'
    + '<span class="popout" onclick="popoutPanel(sel)" '
    + 'title="Open this message in a new window">⧉</span>'
    + panelInnerHTML(m, num, parNum);
  // Reply / Reply All / Forward — wired up after innerHTML below. We mark
  // them with data-act so a single click handler can dispatch by mode.
  h += `<div class="pacts">`
    + `<button data-act="reply" title="Reply to the sender">↩ Reply</button>`
    + `<button data-act="reply-all" title="Reply to everyone">↩ Reply all</button>`
    + `<button data-act="forward" title="Forward this message">⏩ Forward</button>`
    + `</div>`;
  // Lean payload (serve_app) ships no body — show the snippet immediately,
  // then loadBody() upgrades to the real HTML via /api/body.
  h += `<div id="pbody" class="pbody">${esc(m.body || m.snip || "(no message body)")}`
    + `</div>`;
  if(m.url)  h += `<a class="glink" href="${esc(m.url)}" target="_blank" `
    + `rel="noopener" title="Opens a Gmail search holding only this `
    + `message — click the result to open it expanded">Open in Gmail ↗</a>`;
  $("panel").innerHTML = h;
  document.body.classList.add("panel");
  $("panel").querySelectorAll(".pacts button").forEach(b =>
    b.addEventListener("click", () => openCompose(b.dataset.act, i)));
  loadBody(i);
}

// Open the message in an independent browser window. The embedded panel
// is closed at the same time — popout vs. embedded is a MOVE, not a copy:
// the panel's ↙ Re-embed and Reply/Reply All/Forward buttons close the
// popout and hand control back to the main window. One popout at a time.
//
// The popout's own DOM is self-contained (inline CSS, metadata baked in at
// write time, body upgraded via /api/body), but its buttons call back into
// the opener via window.opener.<fn>(…) — so the main window owns all state
// and the popout is just a view + dispatcher.
let popoutInfo = null;        // {win: Window, idx: number} | null

function _popoutAlive(){
  if(popoutInfo && popoutInfo.win.closed){ popoutInfo = null; }
  return popoutInfo != null;
}

function popoutPanel(i){
  if(i == null || !MSGS[i]) return;
  // Same message already popped out → just focus its window.
  if(_popoutAlive() && popoutInfo.idx === i){ popoutInfo.win.focus(); return; }
  // Single-popout model: close any prior popout before opening a new one.
  if(_popoutAlive()){ popoutInfo.win.close(); popoutInfo = null; }

  const w = window.open("", "_blank", "width=720,height=900");
  if(!w){
    alert("Pop-out was blocked. Allow pop-ups for this site, then try again.");
    return;
  }
  popoutInfo = {win: w, idx: i};
  closePanel();                  // the message moved — panel goes away

  const m = MSGS[i];
  const num = curNum[i] || "";
  const parNum = (m.kind !== "root" && m.par >= 0) ? (curNum[m.par] || "") : "";

  // Inline subset of the main panel's CSS — colour tokens are duplicated
  // here because the popout has no access to the parent's :root variables
  // once written.
  const styles = `
    *{box-sizing:border-box}
    body{margin:0;padding:16px 18px;color:#141413;background:#F0EEE6;
      font:13px ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
      overflow:auto}
    .reembed{float:right;cursor:pointer;color:#5F5B53;font-size:11.5px;
      background:#FFFEFB;border:1px solid #D6D2C4;border-radius:6px;padding:3px 10px}
    .reembed:hover{border-color:#CC785C;color:#B05E40}
    .pnum{display:inline-block;font-size:10px;font-weight:700;padding:2px 8px;
      border-radius:10px;background:#FFFEFB;border:1px solid #D6D2C4;color:#5F5B53}
    h2{font:500 16px/1.35 Georgia,"Times New Roman",serif;color:#141413;margin:9px 0 4px}
    .prel{color:#8F8B80;font-style:italic;margin-bottom:10px}
    .pr{display:grid;grid-template-columns:58px 1fr;gap:9px;margin:4px 0}
    .pk{color:#8F8B80;font-size:10.5px;text-transform:uppercase;text-align:right}
    .pv{color:#141413;word-break:break-word}
    .pacts{margin:10px 0 4px;display:flex;gap:6px;flex-wrap:wrap}
    .pacts button{background:#FFFEFB;color:#5F5B53;border:1px solid #D6D2C4;
      border-radius:6px;padding:4px 10px;font-size:11.5px;cursor:pointer;
      font-family:inherit}
    .pacts button:hover{border-color:#CC785C;color:#141413}
    .pbody{margin-top:11px;padding-top:10px;border-top:1px solid #E6E3D8;
      color:#5F5B53;font-size:12px;line-height:1.5;white-space:pre-wrap;
      word-break:break-word}
    .pbody.ishtml{white-space:normal;padding-top:11px}
    .pbodyhtml{width:100%;height:70vh;border:1px solid #E6E3D8;border-radius:6px;
      background:#fff}
    a.glink{display:inline-block;margin-top:12px;color:#B05E40;font-size:12px;
      text-decoration:none;border:1px solid #D6D2C4;border-radius:6px;padding:5px 10px}
    a.glink:hover{border-color:#CC785C;background:#EFE0D8}`;

  // Inline onclicks call back into the main window. Guard against the
  // opener being closed/navigated-away — clicking just becomes a no-op.
  const op = "window.opener && !window.opener.closed";
  const reembedBtn = `<span class="reembed" `
    + `title="Move this message back to the right pane" `
    + `onclick="if(${op}) window.opener.reembedFromPopout()">↙ Re-embed</span>`;
  const actions = `<div class="pacts">`
    + `<button onclick="if(${op}) window.opener.composeFromPopout('reply')" `
    + `title="Reply to the sender">↩ Reply</button>`
    + `<button onclick="if(${op}) window.opener.composeFromPopout('reply-all')" `
    + `title="Reply to everyone">↩ Reply all</button>`
    + `<button onclick="if(${op}) window.opener.composeFromPopout('forward')" `
    + `title="Forward this message">⏩ Forward</button>`
    + `</div>`;
  const glink = m.url
    ? `<a class="glink" href="${esc(m.url)}" target="_blank" rel="noopener">`
      + `Open in Gmail ↗</a>`
    : "";

  const doc = `<!doctype html><html lang="en"><head><meta charset="utf-8">`
    + `<title>${esc(m.subj || "(no subject)")}</title><style>${styles}</style>`
    + `</head><body>`
    + reembedBtn
    + panelInnerHTML(m, num, parNum)
    + actions
    + `<div id="pbody" class="pbody">`
    + `${esc(m.body || m.snip || "(no message body)")}</div>`
    + glink
    + `</body></html>`;
  w.document.open();
  w.document.write(doc);
  w.document.close();

  // Upgrade the placeholder text body to the real HTML message via the
  // same /api/body endpoint the embedded panel uses.
  if(!m.mid) return;
  fetch("api/body?mid=" + encodeURIComponent(m.mid)
    + "&acct=" + encodeURIComponent(m.acct), {cache: "no-store"})
    .then(r => r.ok ? r.text() : "")
    .then(html => {
      if(!html.trim() || !w || w.closed) return;
      const box = w.document.getElementById("pbody");
      if(!box) return;
      const ifr = w.document.createElement("iframe");
      ifr.className = "pbodyhtml";
      // allow-popups (+ escape) lets email links open in a real new tab; still
      // no allow-scripts / allow-same-origin, so the XSS sandbox holds.
      ifr.setAttribute("sandbox", "allow-popups allow-popups-to-escape-sandbox");
      ifr.setAttribute("translate", "no");
      ifr.srcdoc = NOTRANSLATE + html;
      box.classList.add("ishtml");
      box.replaceChildren(ifr);
    })
    .catch(() => { /* no server / popup closed */ });
}

// Called by the popout's ↙ Re-embed button via window.opener. Closes the
// popout and re-opens the embedded panel for the same message.
function reembedFromPopout(){
  if(!_popoutAlive()) return;
  const i = popoutInfo.idx;
  popoutInfo.win.close();
  popoutInfo = null;
  if(i != null && MSGS[i]) pick(i);
}

// Called by the popout's Reply / Reply all / Forward buttons. Re-embeds
// the message and opens the composer in the main window — composing only
// lives in one place, the main app's modal.
function composeFromPopout(mode){
  if(!_popoutAlive()) return;
  const i = popoutInfo.idx;
  popoutInfo.win.close();
  popoutInfo = null;
  if(i != null && MSGS[i]){
    pick(i);
    openCompose(mode, i);
  }
}
function closePanel(){ document.body.classList.remove("panel"); }
async function loadBody(i){
  // Upgrade the plain-text body to the real HTML message — served lazily by
  // serve_app.py (90 MB of HTML can't be embedded). Opened as a static file
  // there's no server, so the plain text already shown simply stays.
  const box = document.getElementById("pbody"), m = MSGS[i];
  if(!box || !m.mid) return;
  try{
    const r = await fetch("api/body?mid=" + encodeURIComponent(m.mid)
      + "&acct=" + encodeURIComponent(m.acct), {cache: "no-store"});
    if(!r.ok || sel !== i) return;            // no server, or moved on
    const html = await r.text();
    if(!html.trim() || sel !== i) return;
    const ifr = document.createElement("iframe");
    ifr.className = "pbodyhtml";
    // allow-popups (+ escape) lets links open in a new tab; still no
    // allow-scripts / allow-same-origin, so the XSS sandbox holds.
    ifr.setAttribute("sandbox", "allow-popups allow-popups-to-escape-sandbox");
    ifr.setAttribute("translate", "no");
    ifr.srcdoc = NOTRANSLATE + html;          // suppress Chrome's translate bar
    box.classList.add("ishtml");
    box.replaceChildren(ifr);
  }catch(e){ /* no server — keep the plain text */ }
}

/* ===================== NAVIGATION ================================== */
function showList(){
  thrOpen = false; sel = null;
  $("thread").style.display = "none";
  $("list").style.display = "block";
  $("back").style.display = "none";
  $("title").style.display = "none";
  // Restore the Select button; re-show the action buttons only if a
  // selection was still active when we drilled into a conversation.
  $("selbtn").style.display = "";
  $("bucketf").style.display = "";
  if(selectMode){
    $("rmbtn").style.display = "inline-block";
    $("readbtn").style.display = "inline-block";
  }
  updateSpamButtons();           // view-specific; self-gate on selectMode
  applyColsVisibility();         // honour the filter-pane toggle + show wedge
  updateSub();
  closePanel();
  renderList();
}
$("back").addEventListener("click", showList);

// Filter pane (#cols) is collapsed by default; the #colstoggle wedge flips it.
// showThread/showList set #cols' inline display, so we drive it from one place
// that also accounts for thread view (where the pane is always hidden).
function applyColsVisibility(){
  const show = !thrOpen && !document.body.classList.contains("cols-hidden");
  $("cols").style.display = show ? "flex" : "none";
  // The left control stack (select-all + chevron) is a list-only affordance.
  $("leftctl").style.display = thrOpen ? "none" : "flex";
}
function toggleCols(){
  const hidden = document.body.classList.toggle("cols-hidden");
  $("colstoggle").title = hidden ? "Show filters" : "Hide filters";
  applyColsVisibility();
}
$("colstoggle").addEventListener("click", toggleCols);

// --- bucket (tier) filter: a multi-select dropdown in the header bar that
// replaces the old Spam toggle. Mirrors buildAcctFilter(): 'primary' shows by
// default, lite tiers are opt-in. Changing it is a filter change, so it drops
// any in-progress selection (a destructive action must never touch hidden rows).
function bucketDispLabel(){
  const n = bucketSel.size;
  if(n === 0) return "None";
  if(n === BUCKETS.length) return "All mail";
  if(n === 1) return BUCKET_LABELS[[...bucketSel][0]] || [...bucketSel][0];
  return n + " tiers";
}
function syncBucket(){
  const w = $("bucketf");
  if(!w) return;
  w.querySelector(".lbl").textContent = bucketDispLabel();
  w.classList.toggle("active", bucketFiltered());
  const all = $("bucket-all");
  if(all) all.checked = bucketSel.size === BUCKETS.length;
  w.querySelectorAll(".opt input[data-b]").forEach(cb =>
    cb.checked = bucketSel.has(cb.dataset.b));
}
function buildBucketFilter(){
  const w = $("bucketf");
  if(!w) return;
  const opts = BUCKETS.map(b =>
    `<label class="opt"><input type="checkbox" data-b="${esc(b)}">`
    + `<i class="acctdot" style="background:${BUCKET_COLOR[b]||"#8F8B80"}"></i>`
    + `<span>${esc(BUCKET_LABELS[b] || b)}</span></label>`).join("");
  w.innerHTML = '<div class="btn"><span class="lbl"></span>'
    + '<span class="car">▾</span></div><div class="menu">'
    + '<label class="opt all"><input type="checkbox" id="bucket-all">'
    + '<i class="acctdot" style="visibility:hidden"></i>'   // align label
    + '<span>All mail</span></label>' + opts + '</div>';
  w.querySelector(".btn").addEventListener("click", e => {
    e.stopPropagation();
    w.classList.toggle("open");
  });
  w.querySelectorAll(".opt input").forEach(cb => {
    cb.addEventListener("change", () => {
      if(cb.id === "bucket-all"){
        bucketSel.clear();
        if(cb.checked) BUCKETS.forEach(b => bucketSel.add(b));
      } else {
        cb.checked ? bucketSel.add(cb.dataset.b) : bucketSel.delete(cb.dataset.b);
      }
      dropSelectionOnFilterChange();
      syncBucket();
      if(thrOpen) showList();
      $("list").scrollTop = 0;
      rebuildList();
    });
  });
  syncBucket();
}
// close the bucket dropdown when clicking anywhere outside it
document.addEventListener("click", e => {
  const w = $("bucketf");
  if(w && !w.contains(e.target)) w.classList.remove("open");
});

// --- Ask the graph: a chat with the knowledge graph, answered by
// serve_app.py (graph-RAG + headless Claude Code). It keeps a conversation,
// so follow-up questions build on earlier answers. The session id and
// transcript live in sessionStorage: that survives the Sync button's
// page reload (an in-progress chat isn't lost), but is empty when the app
// is opened fresh — so a new app start always shows an empty chatbot.
// Needs the server running. ----------------------------------------------
let askSid = null;
const ASK_ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;"};
function askScroll(){ const l = $("asklog"); l.scrollTop = l.scrollHeight; }
function askEscText(s){ return (s || "").replace(/[&<>]/g, c => ASK_ESC[c]); }
function askEscAttr(s){
  return (s || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;");
}
function renderAnswer(text, sources, q){
  // The model cites bare [n] markers (it is told not to emit URLs). Turn
  // each [n] into a link from `sources`, then list the cited messages
  // underneath. A stray URL is still linkified, defensively.
  const byN = {};
  (sources || []).forEach(s => { byN[s.n] = s; });
  let html = askEscText(text || "(no answer)").replace(
    /(https?:\/\/[^\s<]+)/g,
    '<a href="$1" target="_blank" rel="noopener">$1</a>');
  const cited = [];
  html = html.replace(/\[(\d+)\]/g, (full, d) => {
    const s = byN[+d];
    if(!s) return full;
    if(cited.indexOf(+d) < 0) cited.push(+d);
    if(!s.url) return full;
    return '<a href="' + askEscAttr(s.url) + '" target="_blank" '
      + 'rel="noopener" class="cite">[' + d + ']</a>';
  });
  if(cited.length){
    cited.sort((a, b) => a - b);
    html += '<div class="srcs"><div class="srch">Sources</div>';
    cited.forEach(d => {
      const s = byN[d];
      const lbl = "[" + d + "] " + askEscText(s.subject || "(no subject)")
        + " — " + askEscText(s.from || "")
        + (s.date ? " · " + askEscText(s.date) : "");
      html += '<div class="src">' + (s.url
        ? '<a href="' + askEscAttr(s.url) + '" target="_blank" '
          + 'rel="noopener">' + lbl + "</a>"
        : lbl) + "</div>";
    });
    html += "</div>";
  }
  // Action bar: Copy the answer, plus 👍 / 👎 that feed Liam's learning. The
  // question + raw answer ride in data-q / data-a so copying and voting work
  // even after a sessionStorage restore.
  html += '<div class="askfb" data-voted="" data-q="' + askEscAttr(q || "")
    + '" data-a="' + askEscAttr(text || "") + '">'
    + '<button class="askfbbtn askfbcopy" title="Copy answer">⧉ Copy</button>'
    + '<button class="askfbbtn askfbup" title="Good answer">👍</button>'
    + '<button class="askfbbtn askfbdown" title="Could be better">'
    + '👎</button>'
    + '<span class="askfbmsg"></span></div>';
  return html;
}
function askBubble(cls, text){
  const d = document.createElement("div");
  d.className = "askmsg " + cls;
  d.textContent = text;
  $("asklog").appendChild(d);
  askScroll();
  return d;
}
function saveChat(){
  try{ sessionStorage.setItem("askChat", JSON.stringify(
    {sid: askSid, html: $("asklog").innerHTML})); }catch(e){}
}
function newChat(){
  askSid = null;
  try{ sessionStorage.removeItem("askChat"); }catch(e){}
  $("asklog").innerHTML = "";
  askBubble("intro", "Hi, I'm Liam. Ask anything about your mail — I keep "
    + "the conversation, so follow-up questions work.");
}
function openAsk(){
  document.querySelector(".askcard").classList.remove("winmin");  // never reopen collapsed
  $("ask").style.display = "flex"; $("askq").focus();
}
function closeAsk(){ $("ask").style.display = "none"; }

// --- floating-window chrome: drag + minimize + maximize (compose & Ask) ----
// Both panels are now non-modal windows. The header is a drag handle (clicks on
// its buttons/inputs are ignored); resize is the native CSS grip in the bottom-
// right corner; min/max toggle classes the CSS handles. Geometry persists across
// open/close (except winmin, which open clears) so a window reopens where left.
function makeWindow(card, header){
  let drag = false, sx = 0, sy = 0, ox = 0, oy = 0;
  header.addEventListener("mousedown", e => {
    if(e.target.closest("button, input, select, textarea, .x")) return;
    if(card.classList.contains("winmax")) return;     // no drag while maximized
    const r = card.getBoundingClientRect();
    card.style.transform = "none";                    // drop the centering shift
    card.style.left = r.left + "px";
    card.style.top  = r.top  + "px";
    drag = true; sx = e.clientX; sy = e.clientY; ox = r.left; oy = r.top;
    document.body.style.userSelect = "none";
    e.preventDefault();
  });
  window.addEventListener("mousemove", e => {
    if(!drag) return;
    const w = card.offsetWidth;
    let nx = ox + (e.clientX - sx), ny = oy + (e.clientY - sy);
    nx = Math.max(60 - w, Math.min(nx, window.innerWidth - 60));  // keep a sliver
    ny = Math.max(0, Math.min(ny, window.innerHeight - 36));
    card.style.left = nx + "px"; card.style.top = ny + "px";
  });
  window.addEventListener("mouseup", () => {
    if(drag){ drag = false; document.body.style.userSelect = ""; }
  });
  const min = () => { card.classList.remove("winmax"); card.classList.toggle("winmin"); };
  const max = () => { card.classList.remove("winmin"); card.classList.toggle("winmax"); };
  header.addEventListener("dblclick", e => {
    if(e.target.closest("button, input, select, textarea, .x")) return;
    max();
  });
  return { min, max };
}
const _askWin = makeWindow(document.querySelector(".askcard"),
                           document.querySelector("#ask .askhd"));
const _cmpWin = makeWindow(document.querySelector(".ccard"),
                           document.querySelector("#compose .chd"));
$("amin").addEventListener("click", _askWin.min);
$("amax").addEventListener("click", _askWin.max);
$("cmin").addEventListener("click", _cmpWin.min);
$("cmax").addEventListener("click", _cmpWin.max);
async function runAsk(){
  const q = $("askq").value.trim();
  if(!q) return;
  pushHist(q);
  $("askq").value = "";
  askBubble("user", q);
  // Bubble is in "thinking" state: a label with bouncing dots, plus a live
  // trace below it where streamed reasoning + tool calls appear.
  const thinking = askBubble("bot thinking", "");
  thinking.innerHTML = '<div class="askthlbl">Thinking'
    + '<span class="askdots"><span></span><span></span><span></span>'
    + '</span></div><div class="askthtrace"></div>';
  const trace = thinking.querySelector(".askthtrace");
  $("asksend").disabled = true;

  let pendingSources = null, pendingAnswer = "", pendingSid = askSid;
  let gotError = false;

  function addStep(html, cls){
    const d = document.createElement("div");
    d.className = "askstep" + (cls ? " " + cls : "");
    d.innerHTML = html;
    trace.appendChild(d);
    askScroll();
  }
  const PHASE_LABEL = {
    retrieving: "Retrieving relevant mail…",
    thinking: "Reasoning over the context…",
    retrying: "Previous session expired — starting fresh…",
  };
  function dispatch(ev){
    if(ev.type === "phase"){
      const label = PHASE_LABEL[ev.phase];
      if(label) addStep(askEscText(label), "askphase");
    }else if(ev.type === "sources"){
      pendingSources = ev.sources;
    }else if(ev.type === "thinking"){
      addStep(askEscText(ev.text));
    }else if(ev.type === "tool"){
      let html = '<span class="asktoolname">⚙ '
        + askEscText(ev.name || "tool") + '</span>';
      if(ev.detail){
        html += '<div class="asktooldetail"><code>'
          + askEscText(ev.detail) + '</code></div>';
      }
      addStep(html, "asktool");
    }else if(ev.type === "done"){
      pendingAnswer = ev.answer || "";
      pendingSid = ev.session_id || pendingSid;
    }else if(ev.type === "error"){
      thinking.className = "askmsg bot err";
      thinking.textContent = "Error: " + (ev.message || "unknown");
      gotError = true;
    }
  }

  try{
    const r = await fetch("api/ask", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({q, session_id: askSid})});
    if(!r.ok || !r.body){
      thinking.className = "askmsg bot err";
      thinking.textContent = "Error: HTTP " + r.status;
      return;
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    // Parse server-sent events: each event is "data: <json>\n\n".
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      buf += decoder.decode(value, {stream: true});
      let i;
      while((i = buf.indexOf("\n\n")) >= 0){
        const raw = buf.slice(0, i).trim();
        buf = buf.slice(i + 2);
        if(!raw.startsWith("data:")) continue;
        try{ dispatch(JSON.parse(raw.slice(5).trim())); }
        catch(e){ /* ignore malformed frames */ }
      }
    }
    if(!gotError){
      askSid = pendingSid || askSid;
      thinking.className = "askmsg bot";
      thinking.innerHTML = renderAnswer(
        pendingAnswer || "(no answer)", pendingSources, q);
    }
  }catch(e){
    thinking.className = "askmsg bot err";
    thinking.textContent = "Liam needs serve_app.py running — a statically "
      + "opened file can't reach the server. (" + e + ")";
  }finally{
    $("asksend").disabled = false;
    $("askq").focus();
    askScroll();
    saveChat();
  }
}
// restore a prior conversation (survives the sync-triggered reload)
(function(){
  let saved = null;
  try{ saved = JSON.parse(sessionStorage.getItem("askChat") || "null"); }
  catch(e){}
  if(saved && saved.html){
    $("asklog").innerHTML = saved.html;
    askSid = saved.sid || null;
  }else{
    newChat();
  }
})();

// Answer rating (👍 / 👎). Delegated on the log container so it also covers
// answers restored from sessionStorage. A vote posts to api/ask/feedback,
// which logs it and distils a durable response-craft lesson into Liam's
// long-term memory: a 👎 a correction (with an optional one-line note on what
// was off), a 👍 a reinforcement of what made the answer good.
async function sendFeedback(bar, rating, note){
  try{
    await fetch("api/ask/feedback", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({rating: rating, note: note || "",
        question: bar.dataset.q || "", answer: bar.dataset.a || "",
        session_id: askSid || ""})});
  }catch(e){ /* feedback is best-effort — never block the chat on it */ }
}
function askFbDone(bar, msg){
  bar.dataset.voted = "1";
  bar.querySelectorAll(".askfbnote,.askfbsend").forEach(e => e.remove());
  const m = bar.querySelector(".askfbmsg");
  if(m) m.textContent = msg;
  saveChat();                       // persist the voted state across reloads
}
$("asklog").addEventListener("click", ev => {
  const copy = ev.target.closest(".askfbcopy");
  const up = ev.target.closest(".askfbup");
  const down = ev.target.closest(".askfbdown");
  const send = ev.target.closest(".askfbsend");
  if(!copy && !up && !down && !send) return;
  const bar = ev.target.closest(".askfb");
  if(!bar) return;
  // Copy works any time, independent of the up/down vote state.
  if(copy){
    copyText(bar.dataset.a || "").then(() => showToast("Answer copied"));
    return;
  }
  if(bar.dataset.voted) return;
  // 👍 and 👎 both reveal an optional one-line note, then Send distils it into
  // Liam's memory (a 👍 reinforces what was good, a 👎 corrects what was off).
  if(up || down){
    (up || down).classList.add("on");
    bar.dataset.pending = up ? "up" : "down";
    if(!bar.querySelector(".askfbnote")){    // reveal the note box once
      const inp = document.createElement("input");
      inp.className = "askfbnote";
      inp.type = "text";
      inp.placeholder = up ? "What did you like? (optional)"
                           : "What was off? (optional)";
      const btn = document.createElement("button");
      btn.className = "askfbsend";
      btn.textContent = "Send";
      bar.appendChild(inp);
      bar.appendChild(btn);
      inp.focus();
      inp.addEventListener("keydown", e => {
        if(e.key === "Enter"){ e.preventDefault(); btn.click(); }
      });
    }
  }else if(send){
    const inp = bar.querySelector(".askfbnote");
    const note = inp ? inp.value.trim() : "";
    const rating = bar.dataset.pending === "up" ? "up" : "down";
    sendFeedback(bar, rating, note);
    askFbDone(bar, note ? "Thanks — Liam will keep that in mind"
                        : "Thanks for the feedback");
  }
});

// Persistent PROMPT history (the questions you ask, never the answers).
// Lives in localStorage so it survives session-to-session — unlike the chat
// transcript above, which is per-tab sessionStorage by design. ↑/↓ recall past
// prompts shell-style.
const ASK_HIST_KEY = "askPromptHistory";
const ASK_HIST_MAX = 200;
let askHist = [];
try{ const h = JSON.parse(localStorage.getItem(ASK_HIST_KEY) || "[]");
     if(Array.isArray(h)) askHist = h.filter(x => typeof x === "string"); }
catch(e){}
let histIdx = askHist.length;   // == length means "editing a fresh draft"
let histDraft = "";             // unsent text stashed while browsing history

function pushHist(q){
  if(!q) return;
  // Most-recent-wins: drop an earlier identical copy so re-asking floats it to
  // the end and ↑ never shows the same prompt twice in a row.
  askHist = askHist.filter(h => h !== q);
  askHist.push(q);
  if(askHist.length > ASK_HIST_MAX) askHist = askHist.slice(-ASK_HIST_MAX);
  try{ localStorage.setItem(ASK_HIST_KEY, JSON.stringify(askHist)); }catch(e){}
  histIdx = askHist.length;
  histDraft = "";
}
function histNav(dir){          // dir: -1 = older (↑), +1 = newer (↓)
  if(!askHist.length) return false;
  const start = histIdx;
  if(histIdx === askHist.length && dir < 0) histDraft = $("askq").value;
  let i = histIdx + dir;
  if(i < 0) i = 0;
  if(i > askHist.length) i = askHist.length;
  if(i === start) return false;   // no movement — let the arrow act normally
  histIdx = i;
  const q = $("askq");
  q.value = (histIdx === askHist.length) ? histDraft : askHist[histIdx];
  const pos = q.value.length;    // caret to end of the recalled prompt
  q.selectionStart = q.selectionEnd = pos;
  return true;
}

$("askbtn").addEventListener("click", openAsk);
$("askx").addEventListener("click", closeAsk);
$("asknew").addEventListener("click", () => { newChat(); $("askq").focus(); });
$("asksend").addEventListener("click", runAsk);
$("askq").addEventListener("keydown", e => {
  // Enter sends; Shift+Enter inserts a newline. stopPropagation so the
  // keypress never reaches the document handler (which opens a row).
  if(e.key === "Enter" && !e.shiftKey){
    e.preventDefault(); e.stopPropagation(); runAsk(); return;
  }
  // ↑/↓ recall past prompts, but only when the caret sits on the first/last
  // line — otherwise the arrows move within a multi-line draft as usual.
  if(e.key === "ArrowUp" || e.key === "ArrowDown"){
    const v = e.target.value, p = e.target.selectionStart;
    const onFirstLine = v.lastIndexOf("\n", p - 1) === -1;
    const onLastLine = v.indexOf("\n", p) === -1;
    if(e.key === "ArrowUp" && onFirstLine && histNav(-1)){
      e.preventDefault(); e.stopPropagation();
    }else if(e.key === "ArrowDown" && onLastLine && histNav(1)){
      e.preventDefault(); e.stopPropagation();
    }
  }
});
$("ask").addEventListener("click", e => {
  if(e.target === $("ask")) closeAsk();   // backdrop click closes
});

/* ===================== ACCOUNTS / SIGN-IN ====================== */
// Needs serve_app.py running. The "Sign in / Reconnect" button POSTs to
// /api/auth, which runs the Google OAuth consent flow server-side — the
// server opens your browser for Google sign-in, so you never touch the
// terminal. The Google account you pick on the consent screen is stored under
// the chosen label's token_<label>.json. We then poll /api/auth/status until
// the flow finishes and re-render the panel.
function openAccts(){
  $("accts").style.display = "flex";
  renderAccts();
  loadLlmModel();
  renderClaudeAuth();
  loadMemory();
}
function closeAccts(){ $("accts").style.display = "none"; stopClaudePoll(); }

async function renderAccts(){
  const box = $("acctlist");
  box.innerHTML = '<div style="color:var(--ink-3);font-size:12px;'
    + 'padding:6px 0">Loading…</div>';
  let accounts = [];
  try{
    const r = await fetch("api/accounts", {cache: "no-store"});
    accounts = (await r.json()).accounts || [];
  }catch(e){
    box.innerHTML = '<div class="astatus no" style="padding:6px 0">'
      + 'Could not load (is serve_app running?)</div>';
    return;
  }
  box.innerHTML = "";
  accounts.forEach(a => {
    const authed = !!a.authed;
    const row = document.createElement("div");
    row.className = "acctrow";
    row.innerHTML =
      '<div class="ainfo"><div class="alabel">' + esc(a.label) + '</div>'
      + '<div class="aemail">' + (a.email ? esc(a.email)
          : (authed ? "(connected)" : "no account linked")) + '</div></div>'
      + '<span class="astatus ' + (authed ? "ok" : "no") + '">'
      + (authed ? "✓ connected" : "⚠ not connected") + '</span>'
      + '<div class="aacts">'
      + '<button class="asignin">'
      + (authed ? "Reconnect" : "Sign in") + '</button>'
      + '<button class="aremove" title="Stops syncing and deletes the token; '
      + 'you choose whether to also delete imported mail">Remove</button>'
      + '</div>';
    row.querySelector(".asignin").addEventListener(
      "click", ev => startAuth(a.label, ev.target, row));
    row.querySelector(".aremove").addEventListener(
      "click", () => removeAccount(a.label));
    box.appendChild(row);
  });

  // ── Add-account row ─────────────────────────────────────────────────
  const add = document.createElement("div");
  add.className = "acctadd";
  add.innerHTML =
    '<input id="acctnewlabel" maxlength="32" autocomplete="off" '
    + 'placeholder="new label (a-z, 0-9, - or _)">'
    + '<button id="acctaddbtn">➕ Add account</button>'
    + '<div id="acctaddmsg" class="aaddmsg"></div>';
  box.appendChild(add);
  $("acctaddbtn").addEventListener("click", addAccount);
  $("acctnewlabel").addEventListener("keydown", e => {
    if(e.key === "Enter"){ e.preventDefault(); e.stopPropagation(); addAccount(); }
  });
}

// Poll /api/auth/status for a label until the server-side OAuth flow settles.
// Calls onResult with "ok" | "error:<class>" | "timeout".
function pollAuth(label, onResult){
  let tries = 0;
  const poll = setInterval(async () => {
    tries++;
    let status = "running";
    try{
      const r = await fetch("api/auth/status?account="
        + encodeURIComponent(label), {cache: "no-store"});
      status = (await r.json()).status || "running";
    }catch(e){}
    if(status === "ok"){ clearInterval(poll); onResult("ok"); }
    else if(status.indexOf("error") === 0){ clearInterval(poll); onResult(status); }
    else if(tries > 150){ clearInterval(poll); onResult("timeout"); }  // ~5min
  }, 2000);
}

async function startAuth(label, btn, row){
  btn.disabled = true;
  const st = row.querySelector(".astatus");
  st.className = "astatus"; st.textContent = "Opening browser…";
  let d;
  try{
    const r = await fetch("api/auth", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({account: label})});
    d = await r.json();
  }catch(e){
    st.className = "astatus no"; st.textContent = "network error";
    btn.disabled = false; return;
  }
  if(!d.started){
    st.className = "astatus no";
    st.textContent = d.error || "could not start";
    btn.disabled = false; return;
  }
  st.textContent = "Waiting for Google consent…";
  pollAuth(label, status => {
    if(status === "ok"){
      ACCOUNTS_LIST = null;        // force the composer dropdown to re-fetch
      renderAccts();
    }else{
      st.className = "astatus no";
      st.textContent = status === "timeout"
        ? "timed out" : status.replace(/^error:/, "");
      btn.disabled = false;
    }
  });
}

async function addAccount(){
  const inp = $("acctnewlabel"), msg = $("acctaddmsg"), btn = $("acctaddbtn");
  const label = (inp.value || "").trim().toLowerCase();
  if(!/^[a-z0-9_-]{1,32}$/.test(label)){
    msg.textContent = "Invalid label: use a-z, 0-9, - or _ (max 32).";
    return;
  }
  btn.disabled = true; inp.disabled = true;
  msg.textContent = "Opening browser for Google consent…";
  let d;
  try{
    const r = await fetch("api/accounts/add", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({label})});
    d = await r.json();
  }catch(e){
    msg.textContent = "network error"; btn.disabled = false; inp.disabled = false;
    return;
  }
  if(!d.started){
    msg.textContent = d.error || "could not start";
    btn.disabled = false; inp.disabled = false; return;
  }
  msg.textContent = "Waiting for Google consent… (choose the account)";
  pollAuth(label, status => {
    if(status === "ok"){
      ACCOUNTS_LIST = null;
      renderAccts();            // re-renders with the new account + fresh input
    }else{
      msg.textContent = status === "timeout"
        ? "timed out" : status.replace(/^error:/, "");
      btn.disabled = false; inp.disabled = false;
    }
  });
}

async function removeAccount(label){
  if(!confirm("Remove '" + label + "'\n\nStops syncing and deletes its token. "
      + "The mail already imported is kept in the graph.\n\nContinue?")) return;
  // Single follow-up: decide whether the imported data should also be deleted.
  const purge = confirm("Also delete all imported mail for '" + label + "'?\n\n"
    + "OK = permanently delete its mail from the graph and data files "
    + "(IRREVERSIBLE).\nCancel = keep the already-imported mail.");
  try{
    const r = await fetch("api/accounts/remove", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({label, purge})});
    const d = await r.json();
    if(!d.ok){ alert("Could not remove: " + (d.error || "error")); return; }
  }catch(e){ alert("network error"); return; }
  ACCOUNTS_LIST = null;
  renderAccts();
}

/* ----- LLM model (Settings → LLM Model) ----- */
// Reads the persisted model from /api/settings and writes changes back. The
// chosen model is passed to `claude -p --model` by the /api/ask backend.
async function loadLlmModel(){
  const sel = $("llmmodel"), msg = $("llmmsg");
  if(!sel) return;
  msg.textContent = "";
  try{
    const r = await fetch("api/settings", {cache: "no-store"});
    const d = await r.json();
    sel.value = d.llm_model || "default";
  }catch(e){ /* leave the current selection */ }
}

async function saveLlmModel(){
  const sel = $("llmmodel"), msg = $("llmmsg");
  msg.textContent = "Saving…";
  try{
    const r = await fetch("api/settings", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({llm_model: sel.value})});
    const d = await r.json();
    if(!d.ok){ msg.textContent = d.error || "could not save"; return; }
    sel.value = d.llm_model || "default";
    msg.textContent = "Saved ✓";
  }catch(e){ msg.textContent = "network error"; }
}

/* ----- Memory (Settings → Memory) ----- */
// Liam keeps SEPARATE long-term memory per functionality: "ask" (how to answer
// questions) and "compose" (how to write emails). The Ask|Compose tabs switch
// which store this panel shows; every request carries the active scope so the
// two never mix. The toggle controls whether THAT functionality auto-learns.
let memScope = "ask";
const MEM_COPY = {
  ask: {
    learn: "Learn my preferences & context from my questions",
    placeholder: 'Add a memory (e.g. "answer in Spanish")',
    empty: "Nothing remembered yet — ask a few questions, or add one below.",
    hint: 'What Liam remembers for <b>Ask</b> across sessions — <b>style</b> '
      + 'preferences (always applied) and <b>facts</b> (recalled when '
      + 'relevant). Edited here or learned from your questions.',
  },
  compose: {
    learn: "Learn my email-writing style from my drafts",
    placeholder: 'Add a memory (e.g. "sign off as Rodrigo")',
    empty: "Nothing remembered yet — draft a few emails, or add one below.",
    hint: 'What Liam remembers for <b>Compose</b> across sessions — <b>style</b> '
      + 'preferences (always applied) and <b>facts</b> about recipients '
      + '(recalled when relevant). Edited here or learned when you draft.',
  },
};

function applyMemCopy(){
  const c = MEM_COPY[memScope];
  $("memlearnlbl").textContent = c.learn;
  $("memtext").placeholder = c.placeholder;
  $("memhint").innerHTML = c.hint;
  document.querySelectorAll("#memscope .memtab").forEach(b =>
    b.classList.toggle("on", b.dataset.scope === memScope));
}

async function loadMemory(){
  const list = $("memlist"), learn = $("memlearn");
  if(!list) return;
  applyMemCopy();
  list.innerHTML = '<div class="memempty">Loading…</div>';
  let d;
  try{ d = await (await fetch("api/memory?scope=" + memScope,
    {cache:"no-store"})).json(); }
  catch(e){ list.innerHTML = '<div class="memempty">Could not load</div>'; return; }
  learn.checked = !!d.auto_learn;
  renderMemory(d.memories || []);
}

function renderMemory(mems){
  const list = $("memlist");
  list.innerHTML = "";
  if(!mems.length){
    list.innerHTML = '<div class="memempty">' + MEM_COPY[memScope].empty
      + '</div>';
    return;
  }
  // style first, then facts; newest last within each group
  mems.slice().sort((a,b) => (a.kind===b.kind) ? 0 : (a.kind==="style"?-1:1))
    .forEach(m => {
      const row = document.createElement("div");
      row.className = "memrow";
      row.innerHTML =
        '<span class="mtag ' + (m.kind==="style"?"style":"fact") + '">'
        + (m.kind==="style"?"style":"fact") + '</span>'
        + '<div class="mtext">' + esc(m.text)
        + ' <span class="msrc">· ' + (m.source==="auto"?"learned":"added")
        + '</span></div>'
        + '<button class="mdel" title="Forget this">✕</button>';
      row.querySelector(".mdel").addEventListener("click",
        () => deleteMemory(m.id, row));
      list.appendChild(row);
    });
}

async function deleteMemory(id, row){
  try{
    const r = await fetch("api/memory/delete", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({id, scope: memScope})});
    if((await r.json()).ok) row.remove();
  }catch(e){}
}

async function addMemory(){
  const inp = $("memtext"), kind = $("memkind").value;
  const text = (inp.value || "").trim();
  if(!text) return;
  try{
    const r = await fetch("api/memory/add", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({text, kind, scope: memScope})});
    await r.json();
    inp.value = ""; loadMemory();   // refresh (covers add + duplicate/blank)
  }catch(e){}
}

async function saveAutoLearn(){
  const key = (memScope === "compose") ? "compose_auto_learn" : "ask_auto_learn";
  const body = {}; body[key] = $("memlearn").checked;
  try{
    await fetch("api/settings", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(body)});
  }catch(e){}
}

document.querySelectorAll("#memscope .memtab").forEach(b =>
  b.addEventListener("click", () => {
    if(b.dataset.scope === memScope) return;
    memScope = b.dataset.scope;
    loadMemory();
  }));
$("memaddbtn").addEventListener("click", addMemory);
$("memtext").addEventListener("keydown", e => {
  if(e.key === "Enter"){ e.preventDefault(); e.stopPropagation(); addMemory(); }
});
$("memlearn").addEventListener("change", saveAutoLearn);

/* ----- Claude OAuth (Settings → Claude OAuth) ----- */
// Surfaces the Claude Code subscription login that `claude -p` (Liam) uses.
// "Sign in" launches `claude auth login` server-side (opens a browser); we
// then poll /api/claude/status until the login settles.
let CLAUDE_POLL = null;

function stopClaudePoll(){
  if(CLAUDE_POLL){ clearInterval(CLAUDE_POLL); CLAUDE_POLL = null; }
}

async function renderClaudeAuth(){
  const box = $("claudeauth");
  if(!box) return;
  box.innerHTML = '<div class="cameta">Loading…</div>';
  let d;
  try{
    const r = await fetch("api/claude/status", {cache: "no-store"});
    d = await r.json();
  }catch(e){
    box.innerHTML = '<div class="cameta">Could not load '
      + '(is serve_app running?)</div>';
    return;
  }
  if(!d.installed){
    box.innerHTML = '<div class="cainfo"><div class="caemail">Not available'
      + '</div><div class="cameta">the <code>claude</code> CLI is not on '
      + 'PATH</div></div>';
    return;
  }
  const authed = !!d.loggedIn;
  const sub = d.subscriptionType ? (" · " + esc(d.subscriptionType)) : "";
  const meta = authed
    ? (esc(d.authMethod || "claude.ai") + sub)
    : "sign in to use Liam";
  const loginRunning = d.login && d.login.running;
  const loginUrl = d.login && d.login.url;
  box.innerHTML =
    '<div class="cainfo"><div class="caemail">'
    + (authed ? esc(d.email || "(signed in)") : "Not signed in") + '</div>'
    + '<div class="cameta">' + meta + '</div></div>'
    + '<span class="castatus ' + (authed ? "ok" : "no") + '">'
    + (authed ? "✓ signed in" : "⚠ not signed in") + '</span>'
    + '<div class="caacts">'
    + '<button class="calogin">'
    + (loginRunning ? "Signing in…" : (authed ? "Reconnect" : "Sign in"))
    + '</button>'
    + (authed ? '<button class="calogout">Log out</button>' : '')
    + '</div>';
  if(loginRunning && loginUrl){
    const hint = document.createElement("div");
    hint.className = "cameta";
    hint.style.flexBasis = "100%";
    hint.innerHTML = 'If the browser didn\'t open, visit: '
      + '<a href="' + esc(loginUrl) + '" target="_blank" rel="noopener">'
      + esc(loginUrl) + '</a>';
    box.appendChild(hint);
  }
  const lin = box.querySelector(".calogin");
  if(lin){
    lin.disabled = loginRunning;
    lin.addEventListener("click", claudeLogin);
  }
  const lout = box.querySelector(".calogout");
  if(lout) lout.addEventListener("click", claudeLogout);
}

async function claudeLogin(){
  const box = $("claudeauth");
  const btn = box.querySelector(".calogin");
  if(btn){ btn.disabled = true; btn.textContent = "Opening browser…"; }
  let d;
  try{
    const r = await fetch("api/claude/login", {method: "POST"});
    d = await r.json();
  }catch(e){
    if(btn){ btn.disabled = false; btn.textContent = "Sign in"; }
    return;
  }
  if(!d.started){
    if(btn){ btn.disabled = false; btn.textContent = "Sign in"; }
    alert("Could not start Claude sign-in: " + (d.error || "error"));
    return;
  }
  // Poll status until the login completes (or the flow stops running).
  stopClaudePoll();
  let tries = 0;
  CLAUDE_POLL = setInterval(async () => {
    tries++;
    let s = {};
    try{
      const r = await fetch("api/claude/status", {cache: "no-store"});
      s = await r.json();
    }catch(e){}
    const running = s.login && s.login.running;
    if(s.loggedIn || (!running && tries > 1) || tries > 150){
      stopClaudePoll();
      renderClaudeAuth();
    }
  }, 2000);
  renderClaudeAuth();   // immediately reflect the "Signing in…" / URL state
}

async function claudeLogout(){
  if(!confirm("Log out of Claude?\n\nLiam will stop working until you sign "
      + "in again.\n\nContinue?")) return;
  try{
    const r = await fetch("api/claude/logout", {method: "POST"});
    const d = await r.json();
    if(!d.ok){ alert("Could not log out: " + (d.error || "error")); return; }
  }catch(e){ alert("network error"); return; }
  renderClaudeAuth();
}

$("acctsbtn").addEventListener("click", openAccts);
$("acctsx").addEventListener("click", closeAccts);
$("llmmodel").addEventListener("change", saveLlmModel);
$("accts").addEventListener("click", e => {
  if(e.target === $("accts")){ closeAccts(); }   // backdrop click closes
});

/* ===================== COMPOSE / REPLY / FORWARD ==================== */
// Needs serve_app.py running — a static file load can't reach /api/compose.
// Modes: "new" | "reply" | "reply-all" | "forward". For reply/forward, the
// original message index `srcIdx` is set so we can submit threading headers.
let ACCOUNTS_LIST = null;          // [{label, email}] from /api/accounts
let cMode = "new", cSrc = null;    // source message index for reply/forward
let cAttach = [];                  // [{filename, mime, data_b64, size}]
let cmpSid = null;                 // claude session for the drafting chat —
                                   // null = next ✦ Draft starts a fresh one
let cBaseline = null;              // form snapshot at open (after prefill);
                                   // closeCompose compares against it to
                                   // decide whether to prompt before discard

async function fetchAccounts(){
  // Once-per-session fetch. If it fails (static file) the composer stays
  // hidden via the catch in openCompose().
  if(ACCOUNTS_LIST) return ACCOUNTS_LIST;
  const r = await fetch("api/accounts", {cache: "no-store"});
  if(!r.ok) throw new Error("HTTP " + r.status);
  const d = await r.json();
  ACCOUNTS_LIST = d.accounts || [];
  return ACCOUNTS_LIST;
}

// Preferred default sender for a brand-new message (and the fallback identity
// when a reply's receiving account can't be matched).
const DEFAULT_FROM_EMAIL = "you@org.example.com";
function preferredDefaultAccount(){
  return (ACCOUNTS_LIST.find(a => (a.email || "").toLowerCase() === DEFAULT_FROM_EMAIL)
          || ACCOUNTS_LIST.find(a => a.label === "org")
          || ACCOUNTS_LIST[0] || {});
}
function defaultAccountFor(srcIdx){
  // New message → default From to you@org.example.com. For a reply/forward, default to
  // the account that received the original (m.acct) — its address is in the
  // To/Cc list and is the natural identity to respond as — falling back to the
  // preferred default if that account isn't available.
  if(srcIdx == null) return preferredDefaultAccount().label;
  const want = MSGS[srcIdx].acct;
  return (ACCOUNTS_LIST.find(a => a.label === want)
          || preferredDefaultAccount()).label;
}

function fillAccountSelect(selectedLabel){
  const sel = $("cfrom");
  sel.innerHTML = ACCOUNTS_LIST.map(a =>
    `<option value="${esc(a.label)}">${esc(a.label)}`
    + (a.email ? ` — ${esc(a.email)}` : " (not authenticated)")
    + `</option>`).join("");
  if(selectedLabel) sel.value = selectedLabel;
}

function setCcVisible(show, which){
  const id = which === "cc" ? "ccc-row" : "cbcc-row";
  $(id).classList.toggle("ccquiet", !show);
  if(show) $(which === "cc" ? "ccc" : "cbcc").focus();
}

function showStatus(text, cls){
  const s = $("cstatus");
  s.textContent = text || "";
  s.className = "cstatus" + (cls ? " " + cls : "");
}

function renderAttachChips(){
  const box = $("catt");
  box.querySelectorAll(".chip").forEach(e => e.remove());
  // Insert chips before the "+ Attach" label.
  const addLbl = box.querySelector("label.addfile");
  cAttach.forEach((a, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    const kb = Math.max(1, Math.round(a.size / 1024));
    chip.innerHTML = `${esc(a.filename)} <span class="ink-3">(${kb} KB)</span>`
      + ` <span class="xx" data-i="${i}" title="Remove">×</span>`;
    box.insertBefore(chip, addLbl);
  });
  box.querySelectorAll(".chip .xx").forEach(x =>
    x.addEventListener("click", e => {
      cAttach.splice(+e.target.dataset.i, 1);
      renderAttachChips();
    }));
}

function readFileAsB64(file){
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      // dataURL = "data:<mime>;base64,<payload>"
      const s = r.result || "";
      const i = s.indexOf(",");
      resolve(i >= 0 ? s.slice(i + 1) : s);
    };
    r.onerror = () => reject(r.error || new Error("read failed"));
    r.readAsDataURL(file);
  });
}

async function addFiles(fileList){
  // Gmail's per-message ceiling is 25 MB (after base64 inflation ~ 33 MB raw).
  // We block at 24 MB on the raw side to leave room for headers + body.
  const MAX = 24 * 1024 * 1024;
  let already = cAttach.reduce((n, a) => n + a.size, 0);
  for(const f of fileList){
    if(already + f.size > MAX){
      showStatus(`"${f.name}" skipped — total attachments would exceed 24 MB`,
                 "err");
      continue;
    }
    try{
      const data_b64 = await readFileAsB64(f);
      cAttach.push({filename: f.name,
                    mime: f.type || "application/octet-stream",
                    data_b64, size: f.size});
      already += f.size;
    }catch(e){
      showStatus(`failed to read "${f.name}": ${e}`, "err");
    }
  }
  renderAttachChips();
}

function parseAddrs(s){
  // Split on commas/semicolons only — splitting on whitespace would shred
  // "Alice <alice@example.com>" style entries. Dedupe to be forgiving.
  return [...new Set((s || "").split(/[,;]+/)
    .map(x => x.trim()).filter(Boolean))];
}

async function openCompose(mode, srcIdx){
  cMode = mode || "new";
  cSrc = (srcIdx == null) ? null : srcIdx;
  cAttach = [];

  try{ await fetchAccounts(); }
  catch(e){
    alert("Compose needs serve_app.py running — a statically opened file "
      + "can't reach the server. (" + e + ")");
    return;
  }

  fillAccountSelect(defaultAccountFor(cSrc));
  $("ctitle").textContent = ({"new":"New message","reply":"Reply",
    "reply-all":"Reply all","forward":"Forward"})[cMode] || "New message";

  // Reset form, then prefill for reply/forward.
  // Re-enable Send/Save: a successful send leaves them disabled on purpose and
  // closes the modal, and the post-send sync refreshes the page in place (no
  // full reload), so without this the NEXT compose would open with Send dead.
  $("csend").disabled = false; $("cdraft").disabled = false;
  $("cto").value = ""; $("ccc").value = ""; $("cbcc").value = "";
  $("csubj").value = ""; $("cbody").innerHTML = "";
  setCcVisible(false, "cc"); setCcVisible(false, "bcc");
  showStatus("");
  $("cassist-q").value = ""; resetCmpChat(); setAssistOpen(false);
  renderAttachChips();

  if(cMode !== "new" && cSrc != null){
    await prefillFrom(cSrc, cMode);
  }

  // Snapshot the post-prefill state so closeCompose can detect later edits.
  // Reply/forward bodies start with quoted text — that's the baseline, not
  // "dirty", so an immediate cancel after opening doesn't prompt.
  cBaseline = formSnapshot();

  document.querySelector(".ccard").classList.remove("winmin");  // never reopen collapsed
  $("compose").style.display = "flex";
  // Cursor in body for reply/forward (so user can start typing above the
  // quote); To for a new message.
  (cMode === "new" ? $("cto") : $("cbody")).focus();
}

function formSnapshot(){
  return JSON.stringify({
    to: $("cto").value, cc: $("ccc").value, bcc: $("cbcc").value,
    subj: $("csubj").value, body: $("cbody").innerHTML, n: cAttach.length,
  });
}

function isComposeDirty(){
  return cBaseline !== null && formSnapshot() !== cBaseline;
}

// `force=true` skips the discard prompt — used after a successful send/draft
// where the user already committed and the close is just cleanup.
function closeCompose(force){
  if(!force && isComposeDirty()
      && !confirm("Discard this message?")) return;
  $("compose").style.display = "none";
  hideDrop();
  cBaseline = null;
}

async function prefillFrom(srcIdx, mode){
  const src = MSGS[srcIdx];
  // Subject: avoid double Re:/Fwd: prefixes.
  const subj = src.subj || "";
  const lower = subj.toLowerCase();
  const stripped = subj.replace(/^\s*(re|fwd|fw)\s*:\s*/i, "");
  if(mode === "forward"){
    $("csubj").value = (lower.startsWith("fwd:") || lower.startsWith("fw:"))
      ? subj : "Fwd: " + stripped;
  }else{
    $("csubj").value = lower.startsWith("re:") ? subj : "Re: " + stripped;
  }

  // Recipients
  const fromAcct = $("cfrom").value;
  const selfEmail = (ACCOUNTS_LIST.find(a => a.label === fromAcct) || {}).email
    || "";
  const notSelf = arr => arr.filter(e => e && e.toLowerCase() !== selfEmail.toLowerCase());
  if(mode === "reply"){
    $("cto").value = src.from || "";
  }else if(mode === "reply-all"){
    // Reply All: To = original sender; Cc = original To+Cc (minus self & sender)
    $("cto").value = src.from || "";
    const cc = notSelf([...(src.to || []), ...(src.cc || [])]
      .filter(e => e && e.toLowerCase() !== (src.from || "").toLowerCase()));
    if(cc.length){
      $("ccc").value = cc.join(", ");
      setCcVisible(true, "cc");
    }
  }
  // Forward leaves To/Cc blank for the user to fill.

  // Pull the original body + headers from the server, then format the quote.
  showStatus("Loading original…");
  let q = null;
  try{
    const r = await fetch("api/quote?mid=" + encodeURIComponent(src.mid)
      + "&acct=" + encodeURIComponent(src.acct), {cache: "no-store"});
    if(r.ok) q = await r.json();
  }catch(e){ /* fall through to snippet fallback */ }
  showStatus("");

  const fallbackBody = src.body || src.snip || "";
  const bodyText = (q && q.body) || fallbackBody;
  // body_html is added by /api/quote (sourced from body_store). When absent
  // (older messages with no HTML part) we wrap the cleaned text in <pre> so
  // the editor still renders something sane.
  const bodyHtml = (q && q.body_html) || ("<pre style=\"white-space:pre-wrap;"
    + "font-family:inherit;margin:0\">" + esc(bodyText) + "</pre>");
  const fromEmail = (q && q.from_email) || src.from || "";
  const fromName  = (q && q.from_name)  || src.name  || "";
  const sentAt    = (q && q.sent_at)    || src.sent  || "";

  if(mode === "forward"){
    $("cbody").innerHTML = formatForwardQuoteHtml(fromName, fromEmail, sentAt,
      (q && q.to) || src.to, (q && q.cc) || src.cc,
      (q && q.subject) || src.subj, bodyHtml);
  }else{
    $("cbody").innerHTML = formatReplyQuoteHtml(fromName, fromEmail, sentAt,
      bodyHtml);
  }

  // Stash threading info for the submit step.
  $("cbody").dataset.inReplyTo = (q && q.rfc822) || "";
  $("cbody").dataset.references = JSON.stringify(
    (q && q.references) ? [...q.references, q.rfc822 || ""].filter(Boolean) : []);
  $("cbody").dataset.threadId = (q && q.thread_id) || "";
  if(mode === "forward"){
    // Forwards start a new conversation — don't carry threading.
    $("cbody").dataset.inReplyTo = "";
    $("cbody").dataset.references = "[]";
    $("cbody").dataset.threadId = "";
  }
}

// HTML quote builders. The composer is contenteditable, so reply/forward
// drop in real markup — the original message renders with its formatting,
// the user's reply lands in the blank lines on top.
function formatReplyQuoteHtml(fromName, fromEmail, sentAt, bodyHtml){
  const when = (sentAt || "").replace("T", " ").slice(0, 16);
  const who = fromName ? `${esc(fromName)} &lt;${esc(fromEmail)}&gt;`
                       : (esc(fromEmail) || "(unknown)");
  const head = when ? `On ${esc(when)}, ${who} wrote:`
                    : `${who} wrote:`;
  return `<div><br></div><div><br></div>`
    + `<div class="gmail_quote_head">${head}</div>`
    + `<blockquote>${bodyHtml || ""}</blockquote>`;
}
function formatForwardQuoteHtml(fromName, fromEmail, sentAt, to, cc, subject,
                                bodyHtml){
  const when = (sentAt || "").replace("T", " ").slice(0, 16);
  const who = fromName ? `${esc(fromName)} &lt;${esc(fromEmail)}&gt;`
                       : (esc(fromEmail) || "(unknown)");
  let h = `<div><br></div><div><br></div>`
    + `<div class="gmail_quote_head">---------- Forwarded message ---------</div>`
    + `<div>From: ${who}</div>`;
  if(when)               h += `<div>Date: ${esc(when)}</div>`;
  h += `<div>Subject: ${esc(subject || "")}</div>`;
  if(to && to.length)    h += `<div>To: ${esc(to.join(", "))}</div>`;
  if(cc && cc.length)    h += `<div>Cc: ${esc(cc.join(", "))}</div>`;
  h += `<div><br></div>${bodyHtml || ""}`;
  return h;
}

async function submitCompose(mode, ev){
  // SEND SAFETY (defense in depth): actually sending requires a TRUSTED user
  // gesture. The Send button passes its real click event (ev.isTrusted===true);
  // any attempt to call this with mode "send" and no genuine user event — a
  // script, a synthetic click, future code — is refused here, not just at the
  // button. "draft" is never delivered (it only saves to Gmail Drafts), so it
  // isn't gated.
  if(mode === "send" && !(ev && ev.isTrusted)){
    showStatus("For your safety, email is sent only when you click Send.",
               "err");
    return;
  }
  const to_  = parseAddrs($("cto").value);
  const cc_  = parseAddrs($("ccc").value);
  const bcc_ = parseAddrs($("cbcc").value);
  if(mode === "send" && !to_.length && !cc_.length && !bcc_.length){
    showStatus("Add at least one recipient before sending.", "err");
    return;
  }
  const payload = {
    account: $("cfrom").value,
    mode,
    to: to_, cc: cc_, bcc: bcc_,
    subject: $("csubj").value,
    body: $("cbody").innerHTML,
    is_html: true,
    in_reply_to: $("cbody").dataset.inReplyTo || "",
    references: JSON.parse($("cbody").dataset.references || "[]"),
    thread_id: $("cbody").dataset.threadId || "",
    attachments: cAttach.map(a => ({filename: a.filename, mime: a.mime,
                                    data_b64: a.data_b64})),
  };

  const sendBtn = $("csend"), draftBtn = $("cdraft");
  sendBtn.disabled = true; draftBtn.disabled = true;
  showStatus(mode === "send" ? "Sending…" : "Saving draft…");
  try{
    const r = await fetch("api/compose", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if(!d.ok){
      showStatus("Error: " + (d.error || "unknown"), "err");
      sendBtn.disabled = false; draftBtn.disabled = false;
      return;
    }
    if(mode === "send"){
      // Show ✓ briefly, close the composer, THEN trigger the pull + reload
      // in the background. The page reload happens when the new message
      // lands in Neo4j; until then the user is back on the list with the
      // modal out of the way. Buttons stay disabled so a stray re-click
      // during the close beat can't double-fire.
      showStatus("✓ Sent.", "ok");
      await new Promise(r => setTimeout(r, 700));
      closeCompose(true);          // force=true: just sent, skip discard prompt
      // Fire-and-forget. triggerSyncAndReload polls /api/version and reloads
      // the page when sync completes. No onStatus — the composer is gone.
      triggerSyncAndReload(null,
        {account: $("cfrom").value, defer_embed: true});
      return;
    }
    // Draft mode: drafts are filtered out of the graph by design, so a sync
    // would be a no-op. Just confirm and close — force=true skips the
    // discard prompt since the user just committed the message to Gmail.
    showStatus("✓ Draft saved to Gmail.", "ok");
    setTimeout(() => closeCompose(true), 900);
    sendBtn.disabled = false; draftBtn.disabled = false;
  }catch(e){
    showStatus("Network error: " + e, "err");
    sendBtn.disabled = false; draftBtn.disabled = false;
  }
}

/* --- ✦ Compose with Liam: draft the email from a plain-language brief ---
   The assist panel sends the brief plus the composer's current context
   (recipients, subject, and for a reply/forward the quoted original text) to
   /api/compose/draft. Liam returns a subject + plain-text body; we slot the
   body into the editor — replacing it for a new message, or placing it ABOVE
   the preserved quote for a reply/forward — leaving the user to edit and send
   exactly as with a hand-written draft. */
function setAssistOpen(open){
  $("cassist").classList.toggle("open", !!open);
  if(open) $("cassist-q").focus();
}
function setAssistStatus(text, cls, dots){
  const s = $("cassist-status");
  s.className = "cassist-status" + (cls ? " " + cls : "");
  // dots=true appends the same bouncing-dots indicator Liam Ask uses while it
  // is working. Text is escaped since this path uses innerHTML.
  if(dots){
    s.innerHTML = esc(text || "")
      + '<span class="askdots"><span></span><span></span><span></span></span>';
  }else{
    s.textContent = text || "";   // textContent also clears any prior dots
  }
}
// Plain text (blank-line-separated paragraphs) → contenteditable-friendly
// divs, matching how the editor stores hand-typed content.
function textToHtml(text){
  const paras = (text || "").replace(/\r\n/g, "\n").split(/\n{2,}/);
  return paras.map(p => "<div>" + (esc(p).replace(/\n/g, "<br>") || "<br>")
    + "</div>").join("<div><br></div>");
}
// Liam's draft lives inside a marked wrapper (data-liam-draft) so a follow-up
// revision REPLACES it in place rather than stacking a second copy above the
// quote. cmpDraftEl() returns that wrapper (null if the user deleted it).
function cmpDraftEl(){ return $("cbody").querySelector("[data-liam-draft]"); }
function applyDraft(subject, body){
  // Subject only for a brand-new message whose subject is still blank — never
  // clobber a Re:/Fwd: line or something the user already typed.
  if(subject && cMode === "new" && !$("csubj").value.trim()){
    $("csubj").value = subject;
  }
  const html = textToHtml(body);
  const existing = cmpDraftEl();
  if(existing){
    existing.innerHTML = html;        // revision: swap the draft text in place
    return;
  }
  // First draft: wrap it in the marker and place it ABOVE whatever was already
  // in the editor (a quote for reply/forward, usually empty for new). Threading
  // data lives in #cbody.dataset (attributes), preserved across innerHTML edits.
  const rest = $("cbody").innerHTML;
  $("cbody").innerHTML = '<div data-liam-draft="1">' + html + "</div>"
    + "<div><br></div>" + rest;
}
// Append one line to the drafting transcript (you / Liam / error).
function cmpLogAdd(who, text, cls){
  const log = $("cassist-log");
  const d = document.createElement("div");
  d.className = "cmpturn" + (cls ? " " + cls : "");
  d.innerHTML = '<span class="cmpwho">' + esc(who) + "</span> " + esc(text);
  log.appendChild(d);
  log.style.display = "block";
  log.scrollTop = log.scrollHeight;
}
// Forget the conversation and start a fresh draft. Leaves the editor's current
// text untouched — the next ✦ Draft just replaces the draft region anew.
function resetCmpChat(){
  cmpSid = null;
  $("cassist-log").innerHTML = "";
  $("cassist-log").style.display = "none";
  $("cassist-reset").hidden = true;
  $("cassist-draft").textContent = "✦ Draft";
  $("cassist-q").placeholder = "Tell Liam what to write — e.g. \"polite reply "
    + "declining the meeting, suggest next week instead\". Ctrl+Enter to draft.";
  setAssistStatus("");
}
async function draftWithLiam(){
  const instruction = $("cassist-q").value.trim();
  if(!instruction){ $("cassist-q").focus(); return; }
  pushCmpHist(instruction);
  const followup = !!cmpSid;          // a resumed conversation = a revision
  const fromAcct = $("cfrom").value;
  const fromEmail = (ACCOUNTS_LIST.find(a => a.label === fromAcct) || {}).email
    || "";
  // First turn of a reply/forward: hand Liam the quoted original as context
  // (it's told not to repeat it). On a follow-up the session already has it.
  const original = (!followup && cMode !== "new")
    ? ($("cbody").innerText || "").trim() : "";
  // On a follow-up, send the CURRENT draft text (marker contents, so the quote
  // is excluded) so Liam revises what's actually in the editor, hand-edits and
  // all.
  const draftEl = cmpDraftEl();
  const curBody = followup && draftEl ? (draftEl.innerText || "").trim() : "";
  cmpLogAdd("You", instruction);
  $("cassist-q").value = "";
  const btn = $("cassist-draft");
  btn.disabled = true;
  setAssistStatus(followup ? "✦ Liam is revising" : "✦ Liam is drafting",
                  "", true);
  try{
    const r = await fetch("api/compose/draft", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        instruction, mode: cMode, session_id: cmpSid || "",
        to: parseAddrs($("cto").value), cc: parseAddrs($("ccc").value),
        subject: $("csubj").value, from_email: fromEmail, original,
        cur_subject: $("csubj").value, cur_body: curBody,
      }),
    });
    const d = await r.json();
    if(!d.ok){
      setAssistStatus("Error: " + (d.error || "unknown"), "err");
      cmpLogAdd("Liam", d.error || "something went wrong", "err");
      return;
    }
    applyDraft(d.subject, d.body);
    if(d.session_id) cmpSid = d.session_id;
    // style_applied is set (first turn only) when the To recipient has a
    // learned writing-style profile — surface whose voice Liam matched.
    const styled = d.style_applied ? ` in ${d.style_applied}'s style` : "";
    cmpLogAdd("Liam", followup ? "✓ revised the draft"
                               : "✓ drafted into the editor" + styled);
    setAssistStatus(followup ? "✓ Revised — edit freely before sending."
                             : "✓ Drafted" + styled
                               + " — refine it or edit before sending.",
                    "ok");
    // After the first draft the box becomes a refine-loop.
    $("cassist-reset").hidden = false;
    $("cassist-draft").textContent = "✦ Revise";
    $("cassist-q").placeholder = "Refine it — e.g. \"make it shorter\", "
      + "\"warmer tone\", \"add a line about the invoice\". Ctrl+Enter.";
    // Keep the Compose-with-Liam panel in view so the prompt stays reachable
    // for the next follow-up — the draft lands in the body below.
    $("cassist").scrollIntoView({block: "nearest"});
  }catch(e){
    setAssistStatus("Network error: " + e, "err");
  }finally{
    btn.disabled = false;
    // preventScroll: refocus for the next follow-up WITHOUT yanking the form
    // down to the textarea (which used to scroll the panel out of view).
    $("cassist-q").focus({preventScroll: true});
  }
}

/* --- recipient autocomplete --------------------------------------------
   Built once from every email seen as a sender/recipient anywhere in MSGS,
   joined with the name in DATA.people. Filtering is case-insensitive
   substring over "name email"; prefix matches rank first. Picking inserts
   "Name <email>" into the last comma-separated segment of the field. */
let CONTACTS = null;
let curRecipInput = null;        // the recipient input the dropdown belongs to
let dropMatches = [];
let dropIdx = 0;

function buildContacts(){
  if(CONTACTS) return CONTACTS;
  const seen = new Map();
  const add = e => {
    if(!e) return;
    const lo = String(e).toLowerCase();
    if(!seen.has(lo)){
      seen.set(lo, {email: e, name: PEOPLE[e] || PEOPLE[lo] || ""});
    } else if(!seen.get(lo).name){
      seen.get(lo).name = PEOPLE[e] || PEOPLE[lo] || "";
    }
  };
  for(const m of MSGS){
    add(m.from);
    for(const a of m.to)  add(a);
    for(const a of m.cc)  add(a);
    for(const a of m.bcc) add(a);
  }
  CONTACTS = [...seen.values()];
  CONTACTS.forEach(c => {
    c.label = c.name ? `${c.name} <${c.email}>` : c.email;
    c.hay = ((c.name || "") + " " + c.email).toLowerCase();
  });
  CONTACTS.sort((a, b) => a.label.localeCompare(b.label));
  return CONTACTS;
}

function lastSegment(value){
  // Returns {prefix, segment}. prefix includes the separator; segment is
  // the unfinished token at the end (leading whitespace trimmed for matching).
  let i = -1;
  for(let k = value.length - 1; k >= 0; k--){
    const ch = value[k];
    if(ch === "," || ch === ";"){ i = k; break; }
  }
  if(i < 0) return {prefix: "", segment: value.trimStart()};
  return {prefix: value.slice(0, i + 1),
          segment: value.slice(i + 1).trimStart()};
}

function filterContacts(q){
  q = q.toLowerCase().trim();
  if(!q) return [];
  const out = [];
  for(const c of buildContacts()){
    const i = c.hay.indexOf(q);
    if(i >= 0) out.push({c, i});
  }
  out.sort((a, b) => {
    // Prefix matches (start of name or email) come first.
    const ap = a.i === 0 || a.c.email.toLowerCase().startsWith(q);
    const bp = b.i === 0 || b.c.email.toLowerCase().startsWith(q);
    if(ap !== bp) return ap ? -1 : 1;
    return a.c.label.localeCompare(b.c.label);
  });
  return out.slice(0, 8).map(x => x.c);
}

function showDrop(input){
  curRecipInput = input;
  const {segment} = lastSegment(input.value);
  dropMatches = filterContacts(segment);
  const drop = $("cdrop");
  if(!dropMatches.length){ drop.style.display = "none"; return; }
  dropIdx = 0;
  const q = segment.toLowerCase().trim();
  drop.innerHTML = dropMatches.map((c, i) =>
    `<div class="opt${i === 0 ? " on" : ""}" data-i="${i}">`
    + `<span class="nm">${hl(c.name || c.email, [q])}</span>`
    + (c.name ? `<span class="em">${hl(c.email, [q])}</span>` : "")
    + `</div>`).join("");
  const r = input.getBoundingClientRect();
  drop.style.left = r.left + "px";
  drop.style.top  = (r.bottom + 2) + "px";
  drop.style.width = r.width + "px";
  drop.style.display = "block";
  // mousedown (not click) so the pick fires before the input's blur event,
  // which would otherwise hide the dropdown before the click registers.
  drop.querySelectorAll(".opt").forEach(el =>
    el.addEventListener("mousedown", e => {
      e.preventDefault();
      pickDrop(+el.dataset.i);
    }));
}

function hideDrop(){
  $("cdrop").style.display = "none";
  dropMatches = [];
}

function pickDrop(i){
  if(!curRecipInput || !dropMatches[i]) return;
  const c = dropMatches[i];
  const {prefix} = lastSegment(curRecipInput.value);
  const head = prefix ? prefix.replace(/\s*$/, "") + " " : "";
  curRecipInput.value = head + c.label + ", ";
  curRecipInput.focus();
  curRecipInput.setSelectionRange(curRecipInput.value.length,
                                   curRecipInput.value.length);
  hideDrop();
}

function dropKey(e){
  if($("cdrop").style.display === "none" || !dropMatches.length) return;
  if(e.key === "ArrowDown"){
    dropIdx = (dropIdx + 1) % dropMatches.length;
  } else if(e.key === "ArrowUp"){
    dropIdx = (dropIdx - 1 + dropMatches.length) % dropMatches.length;
  } else if(e.key === "Enter" || e.key === "Tab"){
    e.preventDefault();
    pickDrop(dropIdx);
    return;
  } else if(e.key === "Escape"){
    e.preventDefault();
    hideDrop();
    return;
  } else {
    return;
  }
  e.preventDefault();
  $("cdrop").querySelectorAll(".opt").forEach((el, i) =>
    el.classList.toggle("on", i === dropIdx));
}

function wireRecipientInput(input){
  input.addEventListener("input", () => showDrop(input));
  // setTimeout so a click on a suggestion fires before we hide. mousedown on
  // .opt already preventDefault()s, but the delay also covers the case where
  // the user tabs away to the next recipient field.
  input.addEventListener("blur",  () => setTimeout(() => {
    if(document.activeElement !== input) hideDrop();
  }, 100));
  input.addEventListener("keydown", dropKey);
}
["cto", "ccc", "cbcc"].forEach(id => wireRecipientInput($(id)));

$("composebtn").addEventListener("click", () => openCompose("new", null));
// Wrap so the click event isn't passed as `force` — would skip the prompt.
$("cx").addEventListener("click", () => closeCompose());
$("ccancel").addEventListener("click", () => closeCompose());
// SEND SAFETY — an email is dispatched ONLY from here, and ONLY on a genuine
// user activation. e.isTrusted is true only for a real mouse/keyboard press on
// the button; it is false for any script-synthesized click (el.click(),
// dispatchEvent). So no code path — Liam, a future bug, injected message
// content — can send without the user physically pressing Send.
// submitCompose("send") is intentionally called from nowhere else.
$("csend").addEventListener("click", e => submitCompose("send", e));
$("cdraft").addEventListener("click", () => submitCompose("draft"));
// Persistent PROMPT history for ✦ Compose with Liam — the briefs you give it,
// kept SEPARATE from Ask's history (own localStorage key). Survives reloads;
// ↑/↓ recall past briefs shell-style, exactly like the Ask box.
const CMP_HIST_KEY = "composePromptHistory";
const CMP_HIST_MAX = 200;
let cmpHist = [];
try{ const h = JSON.parse(localStorage.getItem(CMP_HIST_KEY) || "[]");
     if(Array.isArray(h)) cmpHist = h.filter(x => typeof x === "string"); }
catch(e){}
let cmpHistIdx = cmpHist.length;   // == length means "editing a fresh draft"
let cmpHistDraft = "";             // unsent text stashed while browsing history

function pushCmpHist(q){
  if(!q) return;
  // Most-recent-wins: drop an earlier identical copy so re-using a brief floats
  // it to the end and ↑ never shows the same one twice in a row.
  cmpHist = cmpHist.filter(h => h !== q);
  cmpHist.push(q);
  if(cmpHist.length > CMP_HIST_MAX) cmpHist = cmpHist.slice(-CMP_HIST_MAX);
  try{ localStorage.setItem(CMP_HIST_KEY, JSON.stringify(cmpHist)); }catch(e){}
  cmpHistIdx = cmpHist.length;
  cmpHistDraft = "";
}
function cmpHistNav(dir){          // dir: -1 = older (↑), +1 = newer (↓)
  if(!cmpHist.length) return false;
  const start = cmpHistIdx;
  if(cmpHistIdx === cmpHist.length && dir < 0)
    cmpHistDraft = $("cassist-q").value;
  let i = cmpHistIdx + dir;
  if(i < 0) i = 0;
  if(i > cmpHist.length) i = cmpHist.length;
  if(i === start) return false;   // no movement — let the arrow act normally
  cmpHistIdx = i;
  const q = $("cassist-q");
  q.value = (cmpHistIdx === cmpHist.length) ? cmpHistDraft : cmpHist[cmpHistIdx];
  const pos = q.value.length;     // caret to end of the recalled brief
  q.selectionStart = q.selectionEnd = pos;
  return true;
}

$("cassist-toggle").addEventListener("click", () =>
  setAssistOpen(!$("cassist").classList.contains("open")));
$("cassist-draft").addEventListener("click", draftWithLiam);
$("cassist-reset").addEventListener("click", () => {
  resetCmpChat(); $("cassist-q").focus();
});
$("cassist-q").addEventListener("keydown", e => {
  if((e.ctrlKey || e.metaKey) && e.key === "Enter"){
    e.preventDefault(); draftWithLiam(); return;
  }
  // ↑/↓ recall past briefs, but only when the caret sits on the first/last
  // line — otherwise the arrows move within a multi-line brief as usual.
  if(e.key === "ArrowUp" || e.key === "ArrowDown"){
    const v = e.target.value, p = e.target.selectionStart;
    const onFirstLine = v.lastIndexOf("\n", p - 1) === -1;
    const onLastLine = v.indexOf("\n", p) === -1;
    if(e.key === "ArrowUp" && onFirstLine && cmpHistNav(-1)){
      e.preventDefault();
    }else if(e.key === "ArrowDown" && onLastLine && cmpHistNav(1)){
      e.preventDefault();
    }
  }
});
// --- composer rich-text: Bold / Italic toolbar + Ctrl/Cmd+B / +I -----------
// #cbody is contenteditable and the body is sent as HTML, so execCommand's
// <b>/<i> tags carry straight through to the email.
function updateFmtState(){
  if($("compose").style.display !== "flex") return;
  ["bold", "italic"].forEach(cmd => {
    const btn = $("cfmt").querySelector('[data-cmd="' + cmd + '"]');
    if(btn) btn.classList.toggle("on", document.queryCommandState(cmd));
  });
}
$("cfmt").addEventListener("mousedown", e => {
  // mousedown + preventDefault so the caret/selection stays in #cbody
  // (a click would blur it first, collapsing the selection).
  const b = e.target.closest("button[data-cmd]");
  if(!b) return;
  e.preventDefault();
  document.execCommand(b.dataset.cmd, false, null);
  $("cbody").focus();
  updateFmtState();
});
$("cbody").addEventListener("keydown", e => {
  if((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey){
    const k = e.key.toLowerCase();
    if(k === "b" || k === "i"){
      e.preventDefault();
      document.execCommand(k === "b" ? "bold" : "italic", false, null);
      updateFmtState();
    }
  }
});
$("cbody").addEventListener("keyup", updateFmtState);
$("cbody").addEventListener("mouseup", updateFmtState);
$("ccc-toggle").addEventListener("click", () => setCcVisible(true, "cc"));
$("cbcc-toggle").addEventListener("click", () => setCcVisible(true, "bcc"));
$("cfile").addEventListener("change", e => {
  addFiles(Array.from(e.target.files || []));
  e.target.value = "";    // allow re-selecting the same file later
});

// Clipboard images (Ctrl+V after a screenshot) → attachments. Otherwise the
// browser would inline the image as a base64 data: URL inside the
// contenteditable, which bloats the HTML body and never reaches the
// recipient as a real attachment.
$("cbody").addEventListener("paste", e => {
  const items = (e.clipboardData && e.clipboardData.items) || [];
  const imgs = [];
  for(const it of items){
    if(it.kind === "file" && it.type && it.type.startsWith("image/")){
      const f = it.getAsFile();
      if(f) imgs.push(f);
    }
  }
  if(!imgs.length) return;            // no image — let default paste run
  e.preventDefault();
  // Clipboard images come named "image.png" in most browsers. Rename to
  // pasted-<YYYYMMDDhhmmss>-<i>.<ext> so the chip list reads cleanly when
  // several screenshots are pasted in a row.
  const stamp = new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14);
  const named = imgs.map((f, i) => {
    const ext = (f.type.split("/")[1] || "png").split(";")[0];
    return new File([f], `pasted-${stamp}-${i + 1}.${ext}`, {type: f.type});
  });
  addFiles(named);
});
$("compose").addEventListener("click", e => {
  if(e.target === $("compose")) closeCompose();   // backdrop click closes
});

// Keyboard: ↑/↓ move the highlighted message; the view scrolls to keep it
// in sight. In the list, Enter opens the highlighted conversation; in a
// thread, moving the highlight updates the detail panel.
function ensureVisible(k){
  const box = $("list"), y = k * LROW;
  if(y < box.scrollTop) box.scrollTop = y;
  else if(y + LROW > box.scrollTop + box.clientHeight)
    box.scrollTop = y + LROW - box.clientHeight;
}
function moveCursor(d){
  if(!view.length) return;
  listCur = Math.max(0, Math.min(view.length - 1, listCur + d));
  ensureVisible(listCur);
  renderList();
}
function moveSel(d){
  if(!curRows.length) return;
  let pos = curRows.indexOf(sel);
  pos = Math.max(0, Math.min(curRows.length - 1, (pos < 0 ? 0 : pos) + d));
  pick(curRows[pos]);
  const el = $("thread").querySelector('.row[data-i="' + curRows[pos] + '"]');
  if(el) el.scrollIntoView({block: "nearest"});
}
document.addEventListener("keydown", e => {
  if(e.key === "Escape"){
    // Cascade: close the Compose modal, Ask modal, account menu, panel,
    // thread, then filters.
    if($("compose").style.display === "flex"){ closeCompose(); return; }
    if($("ask").style.display === "flex"){ closeAsk(); return; }
    if($("accts").style.display === "flex"){ closeAccts(); return; }
    const af = $("acctf");
    if(af && af.classList.contains("open")){
      af.classList.remove("open"); return;
    }
    if(document.body.classList.contains("panel")) closePanel();
    else if(thrOpen) showList();
    else if(anyFilter()) clearAll();
    return;
  }
  // typing in a field (search box, chat input, or the contenteditable
  // compose body #cbody) — don't hijack keys
  if(e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA"
     || e.target.isContentEditable) return;
  if(e.key === "Enter"){
    if(!thrOpen && view.length){ e.preventDefault(); openConv(view[listCur]); }
    return;
  }
  let d;
  if(e.key === "ArrowDown")      d = 1;
  else if(e.key === "ArrowUp")   d = -1;
  else if(e.key === "PageDown")  d = 12;
  else if(e.key === "PageUp")    d = -12;
  else if(e.key === "Home")      d = -1e9;
  else if(e.key === "End")       d = 1e9;
  else return;
  e.preventDefault();
  if(thrOpen) moveSel(d); else moveCursor(d);
});
buildAcctFilter();
buildBucketFilter();
rebuildList();

/* === live mode — active only when served by serve_app.py =============
   On a static file the fetches just fail and the controls stay hidden. */
// Seed _ver from the version baked into the page (the cache version this data
// belongs to) so a background rebuild that lands before our SSE stream connects
// still trips the reload banner. 0 = static export → live mode stays off.
let _ver = (typeof window.__BAKED_VERSION__ === "number"
            && window.__BAKED_VERSION__ > 0) ? window.__BAKED_VERSION__ : null;

// Re-derive all MSGS-dependent lookups from the current DATA. Called only by
// refreshPayload — the initial bootstrap runs the same logic inline at the
// top of the script. Mirrors that init exactly, plus preserves user filter
// state (acctSel) across the swap. Don't call this without first assigning
// the new DATA value.
function _rebuildIndices(){
  MSGS = DATA.msgs;
  PALETTE = DATA.palette;
  PEOPLE = DATA.people;
  MSGS.forEach(m => { if(m.acct && !(m.acct in ACCT_COLOR))
    ACCT_COLOR[m.acct] = ACCT_PAL[Object.keys(ACCT_COLOR).length % ACCT_PAL.length]; });

  // CONV is a const object — mutate in place so existing closures keep the
  // same reference.
  for(const k in CONV) delete CONV[k];
  MSGS.forEach((m, i) => (CONV[m.conv] || (CONV[m.conv] = [])).push(i));
  CONVPOS = computeConvPos();

  HAY = MSGS.map(m => {
    const f = {
      from: m.from + " " + (m.name || ""),
      to:   m.to.map(named).join("  "),
      cc:   m.cc.map(named).join("  "),
      bcc:  m.bcc.map(named).join("  "),
      subj: m.subj || "",
      body: m.body || m.snip || "",
      atts: m.atts.join("  "),
      acct: m.acct || "",
      date: (m.sent || "").slice(0, 10),
    };
    for(const k in f) f[k] = f[k].toLowerCase();
    return f;
  });

  listIdx = MSGS.map((m, i) => i)
    .sort((a, b) => (MSGS[b].sent || "").localeCompare(MSGS[a].sent || ""));

  // ACCTS may have grown (brand-new mailbox) or shrunk (all mail trashed
  // from one account). Auto-add new accounts to the selection so they
  // don't sit invisible behind a stale "all selected" filter; drop accounts
  // that no longer have any messages.
  const prev = new Set(ACCTS);
  ACCTS = [...new Set(MSGS.map(m => m.acct).filter(Boolean))].sort();
  for(const a of [...acctSel]) if(!ACCTS.includes(a)) acctSel.delete(a);
  for(const a of ACCTS) if(!prev.has(a)) acctSel.add(a);

  // Rebuild the account-filter dropdown so any new accounts are clickable.
  buildAcctFilter();

  // Buckets may have grown (lite mail newly loaded). Unlike accounts, do NOT
  // auto-select new buckets — lite stays opt-in. 'primary' is always offered;
  // bucketSel (preserved across the swap) just drops any vanished bucket.
  BUCKETS = [...new Set(MSGS.map(m => m.bucket).filter(Boolean))]
    .sort((a, b) => BUCKET_ORDER.indexOf(a) - BUCKET_ORDER.indexOf(b));
  if(!BUCKETS.includes("primary")) BUCKETS.unshift("primary");
  for(const b of [...bucketSel]) if(!BUCKETS.includes(b)) bucketSel.delete(b);
  if(bucketSel.size === 0) bucketSel.add("primary");
  buildBucketFilter();
}

// Swap in a fresh payload without a full page reload. Preserves scroll
// position, current filter, selection, cursor row, and the open detail
// panel / thread view (matched by mid+acct since global indices shift).
async function refreshPayload(){
  let d;
  try{
    const r = await fetch("api/payload", {cache: "no-store"});
    if(!r.ok) return false;
    d = await r.json();
  }catch(e){ return false; }
  if(!d || !Array.isArray(d.msgs)) return false;

  // --- snapshot state we want to preserve, keyed by stable mid+acct ----
  const prevScroll = $("list").scrollTop;
  const key = m => m ? (m.mid + "|" + m.acct) : null;
  const prevCursorKey = (view && view[listCur] != null)
    ? key(MSGS[view[listCur]]) : null;
  const prevSelKey = (sel != null) ? key(MSGS[sel]) : null;
  const prevThr = thrOpen;
  const selKeys = new Set();
  for(const i of SELECTED){ const k = key(MSGS[i]); if(k) selKeys.add(k); }

  // --- swap data + reset transient sets that pointed at old indices ----
  DATA = d;
  SELECTED.clear();      // re-populated below from selKeys
  // Don't blindly forget trashed messages: the server's purge-rebuild may not
  // have landed yet, so this fresh payload can still contain a row the user
  // trashed. Drop a REMOVED key only once it's actually gone from the payload;
  // keep the rest hidden so trashed mail never flickers back into view.
  if(REMOVED.size){
    const live = new Set(d.msgs.map(m => m.mid + "|" + m.acct));
    for(const k of [...REMOVED]) if(!live.has(k)) REMOVED.delete(k);
  }
  _rebuildIndices();

  // --- restore selection by key (indices have shifted) -----------------
  if(selKeys.size){
    for(let i = 0; i < MSGS.length; i++){
      if(selKeys.has(key(MSGS[i]))) SELECTED.add(i);
    }
  }

  // --- re-run the filter against the new data + restore cursor/scroll --
  rebuildList();
  if(prevCursorKey){
    const nk = view.findIndex(i => key(MSGS[i]) === prevCursorKey);
    if(nk >= 0) listCur = nk;
  }
  $("list").scrollTop = prevScroll;
  renderList();

  // --- re-open the panel / thread on the same message if it survived ---
  if(prevSelKey){
    const ns = MSGS.findIndex(m => key(m) === prevSelKey);
    if(ns >= 0){
      if(prevThr) openConv(ns);
      else pick(ns);
    } else if(prevThr){
      // Original message gone (trashed elsewhere) — fall back to the list.
      showList();
    }
  }

  // Accept the new version + hide the "click to reload" banner.
  _ver = null;
  $("banner").style.display = "none";
  return true;
}

function handleVersionUpdate(v){
  const rb = $("refresh");
  rb.style.display = "inline-block";
  rb.textContent = v.syncing ? "↻ syncing…" : "↻ Sync";
  rb.disabled = !!v.syncing;
  if(_ver === null) _ver = v.version;
  else if(v.version !== _ver) $("banner").style.display = "block";
}

async function pollVersion(){
  try{
    const r = await fetch("api/version", {cache: "no-store"});
    if(!r.ok) return;
    handleVersionUpdate(await r.json());
  }catch(e){ /* opened as a static file — no server */ }
}

// Streaming version updates via Server-Sent Events: zero traffic between
// syncs, instant syncing-indicator flips. The 60s safety poll below covers
// transient SSE drops; we don't tear down SSE on a single error because
// the browser auto-reconnects.
let _evSrc = null;
// A random id for THIS window, sent with the SSE stream so the server can
// track this window's presence and drop it the instant the window closes
// (see signalClosing). New per page load, so a reload reconnects as a fresh
// window rather than reusing a cid the server may have just dropped.
const WINDOW_CID = "w" + Math.random().toString(36).slice(2)
                   + Date.now().toString(36);
function startEventStream(){
  if(typeof EventSource === "undefined") return;
  try{
    _evSrc = new EventSource("api/events?cid=" + encodeURIComponent(WINDOW_CID));
    _evSrc.onmessage = ev => {
      try{ handleVersionUpdate(JSON.parse(ev.data)); }catch(_){}
    };
    // onerror fires on transient drops too — the browser will retry the
    // connection on its own. We only really care about the permanent-failure
    // case (e.g. server stopped), and the 60s pollVersion safety net catches
    // that without us having to manage reconnect ourselves.
  }catch(e){ /* no SSE — pollVersion still runs */ }
}

// The server ties its own lifetime to the open app window(s): when the last
// one goes away it shuts down (so closing the app also stops the background
// server / its console). Relying on the SSE socket tearing down is unreliable
// — a Chromium "--app" window can keep the TCP connection warm after close, so
// the server wouldn't notice for a long time. So on pagehide we tell it
// explicitly with a beacon (which survives unload) and close the stream. A
// reload also fires this, but the server's grace window covers the brief gap
// until the reloaded page reconnects with a new cid.
function signalClosing(){
  try{ if(_evSrc){ _evSrc.close(); _evSrc = null; } }catch(_){}
  try{
    if(navigator.sendBeacon){
      navigator.sendBeacon("api/closing?cid=" + encodeURIComponent(WINDOW_CID));
    }
  }catch(_){}
}
// pagehide is the reliable "this page is going away" event (fires on close,
// navigation and bfcache eviction, unlike the flaky unload).
window.addEventListener("pagehide", signalClosing);

/* Trigger an immediate sync and refresh-in-place when it lands. Used by
   both the header ↻ Sync button and the composer's auto-sync-after-send.
   refreshPayload swaps the data without a page reload, preserving scroll,
   filters, selection, and the open detail panel. Falls back to a hard
   reload only if the refresh path itself fails.

   syncParams, if given, becomes the JSON body of /api/sync — the composer
   passes {account, defer_embed: true} so only the sender's mailbox gets
   pulled; the manual sync button passes nothing and pulls all accounts.

   onStatus(text) receives progress strings so the caller can show them. */
async function triggerSyncAndReload(onStatus, syncParams){
  if(onStatus) onStatus("Syncing…");
  try{
    await fetch("api/sync", {
      method: "POST",
      headers: syncParams ? {"Content-Type": "application/json"} : {},
      body: syncParams ? JSON.stringify(syncParams) : undefined,
    });
  }catch(e){}
  let polls = 0, sawSyncing = false;
  // 600ms poll = fast UX without spamming the server. 300 ticks = ~3 min
  // ceiling — generous in case a pull stalls on network.
  while(polls < 300){
    await new Promise(r => setTimeout(r, 600));
    polls++;
    let v;
    try{
      const r = await fetch("api/version", {cache: "no-store"});
      if(!r.ok) continue;
      v = await r.json();
    }catch(e){ continue; }
    if(v.syncing){
      sawSyncing = true;
      if(onStatus) onStatus("Syncing… (pulling new mail)");
      continue;
    }
    if(sawSyncing || polls >= 2){
      if(onStatus) onStatus("Refreshing…");
      const ok = await refreshPayload();
      if(!ok) location.reload();
      return;
    }
  }
  // Hard timeout — fall back to a reload.
  location.reload();
}
$("refresh").addEventListener("click", async () => {
  const rb = $("refresh");
  rb.textContent = "↻ syncing…";
  rb.disabled = true;
  await triggerSyncAndReload();
});
$("banner").addEventListener("click", async () => {
  $("banner").style.display = "none";
  const ok = await refreshPayload();
  if(!ok) location.reload();
});
pollVersion();
startEventStream();
// 60s safety poll covers SSE drops + the static-file fallback case. Down
// from the previous 30s now that SSE handles the responsive path.
setInterval(pollVersion, 60000);
</script></body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
