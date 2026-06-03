"""Run the Gmail → Neo4j KG pipeline end to end.

This is what Claude invokes when the user says "run plan.md", AFTER running
the interactive probe protocol documented in CLAUDE.md (preloaded-state
probe, keep/erase per scope, then ask backfill window).

Steps (each underlying script is idempotent, so the orchestrator is too):
  0.  Pre-flight — verify venv, OAuth tokens, Neo4j (native, over Bolt).
  0a. Probe      — report preloaded state (data files, Neo4j node counts).
  0b. Reset      — if --reset-data or --reset-graph, wipe before pull.
  0c. Cleanup    — prune non-canonical files from data/ (old .bak.* beyond
                   BACKUP_KEEP_DAYS, ad-hoc *.log tees).
  1. Pull        — pull_gmail.py per account, SEQUENTIALLY (no parallel; no
                   file lock means parallel writes corrupt emails.jsonl).
  1b. Calendar   — pull_calendar.py per account, SEQUENTIALLY (same no-lock
                   constraint on the shared calendar_events.jsonl).
  2. Repair      — repair_emails_jsonl.py drops malformed lines if any.
  3. Clean       — clean_bodies.py (talon).
  4. Load        — load_neo4j.py.
  5. Embed       — embed_messages.py refreshes Message.embedding for any
                   message still missing one (zero-token on a clean run;
                   only the newly loaded messages get embedded). Acts as a
                   resumable checkpoint — interrupt and re-run freely.
  6. Cluster     — cluster_matters.py groups Gmail-split threads into
                   Matter nodes (Gmail caps a thread at ~100 messages, and
                   long-running matters sometimes change subject).
  7. Sanity      — short cypher summary against Neo4j.
  8. Outputs     — size + mtime of every canonical file (consistency check).

Usage:
    python scripts/run_pipeline.py --months 24 --reset-data --reset-graph
    python scripts/run_pipeline.py --since 2024-05-14
    python scripts/run_pipeline.py --all-time --skip-sanity
    python scripts/run_pipeline.py --months 24 --reembed
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time

from _common import (
    ACCOUNTS,
    DATA_DIR as DATA,
    SCRIPTS,
    bootstrap_venv,
    force_utf8,
    load_dotenv,
    neo4j_driver,
    run_step,
)

# Force UTF-8 before the bootstrap re-exec so the venv child inherits the
# encoding env at startup; otherwise stdout is initialized as cp1252 on
# Windows and any non-ASCII output (banners, accents, arrows) blows up.
force_utf8()
bootstrap_venv()

# After the bootstrap, sys.executable points at the venv python (either because
# we re-exec'd above, or because we were already invoked correctly). Children
# inherit it via PYTHON.
PYTHON = sys.executable

load_dotenv()

# Canonical files that scripts in this pipeline create or maintain. Anything
# else in data/ matching the cleanup patterns is a leftover and gets pruned at
# the start of each run (with a backup-age guard).
CANONICAL_FILES = {
    ".gitkeep",
    "credentials.json",
    "orgs_seed.json",
    "emails.jsonl",
    "emails_clean.jsonl",
    "calendar_events.jsonl",
    "cleaned_msg_ids.txt",
} | {f"token_{a}.json" for a in ACCOUNTS} \
  | {f"sync_state_{a}.json" for a in ACCOUNTS} \
  | {f"pulled_msg_ids_{a}.txt" for a in ACCOUNTS} \
  | {f"pulled_event_ids_{a}.txt" for a in ACCOUNTS}

# How long to keep .bak.* and ad-hoc *.log files before pruning. The most
# recent backup / log is cheap insurance; older ones are clutter. Age-gating
# also protects an in-flight tee log (age 0 → kept).
BACKUP_KEEP_DAYS = 7

# Files erased by --reset-data. Configuration files (credentials, tokens,
# orgs_seed) are explicitly preserved — they're not pipeline state.
RESET_DATA_FILES = {
    "emails.jsonl",
    "emails_clean.jsonl",
    "calendar_events.jsonl",
    "cleaned_msg_ids.txt",
} | {f"pulled_msg_ids_{a}.txt" for a in ACCOUNTS} \
  | {f"pulled_event_ids_{a}.txt" for a in ACCOUNTS} \
  | {f"sync_state_{a}.json" for a in ACCOUNTS}

# Files preserved by --reset-data — never deleted by the reset.
RESET_DATA_PRESERVE = {
    ".gitkeep",
    "credentials.json",
    "orgs_seed.json",
    # Curated reference dictionary (long-form → abbreviation); treated as
    # data, not pipeline state.
    "abbreviations.json",
} | {f"token_{a}.json" for a in ACCOUNTS}


def banner(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def run(label: str, cmd: list[str]) -> None:
    """Project-local subprocess runner — delegates to _common.run_step but
    keeps the orchestrator's call-site spelling stable."""
    run_step(label, cmd, cwd=None)


def prune_data_dir() -> None:
    """Remove non-canonical .bak.* / *.log files older than BACKUP_KEEP_DAYS.
    Recent ones are preserved as insurance, which also protects an in-flight
    tee log (age 0 → kept). Anything else non-canonical is left alone."""
    now = time.time()
    removed: list[str] = []
    kept: list[str] = []

    for p in DATA.iterdir():
        if not p.is_file() or p.name in CANONICAL_FILES:
            continue
        if ".bak" not in p.name and not p.name.endswith(".log"):
            continue
        age_days = (now - p.stat().st_mtime) / 86400
        if age_days <= BACKUP_KEEP_DAYS:
            kept.append(f"{p.name} ({age_days:.1f}d)")
            continue
        p.unlink()
        removed.append(f"{p.name} ({age_days:.1f}d)")

    if removed:
        print(f"  pruned: {len(removed)} file(s)")
        for name in removed:
            print(f"    - {name}")
    else:
        print("  pruned: nothing")
    if kept:
        print(f"  kept (within {BACKUP_KEEP_DAYS}-day window): {', '.join(kept)}")


def detect_preloaded_state() -> dict:
    """Probe data/ and Neo4j for any preloaded state. Returns a dict the
    orchestrator prints during pre-flight so the user (via Claude) can decide
    whether to pass --reset-data / --reset-graph on the next invocation."""
    state = {"files": {}, "graph": {}}

    for name in sorted(RESET_DATA_FILES):
        p = DATA / name
        if p.exists():
            sz = p.stat().st_size
            if name.endswith(".jsonl"):
                lines = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
                state["files"][name] = {"size": sz, "lines": lines}
            else:
                state["files"][name] = {"size": sz}

    try:
        with neo4j_driver() as drv, drv.session() as sess:
            rec = sess.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n"
            )
            for row in rec:
                if row["label"]:
                    state["graph"][row["label"]] = row["n"]
    except Exception as e:
        state["graph"]["_error"] = f"{type(e).__name__}: {e}"

    return state


def print_preloaded_state(state: dict) -> None:
    if state["files"]:
        print(f"  data files present: {len(state['files'])}")
        for name, info in state["files"].items():
            extra = f" ({info['lines']} lines)" if "lines" in info else ""
            print(f"    - {name}: {info['size']:,} bytes{extra}")
    else:
        print("  data files: none")

    if "_error" in state["graph"]:
        print(f"  Neo4j: {state['graph']['_error']}")
    elif state["graph"]:
        total = sum(state["graph"].values())
        print(f"  Neo4j: {total:,} nodes across {len(state['graph'])} label(s)")
        for label, n in sorted(state["graph"].items(), key=lambda kv: -kv[1]):
            print(f"    - {label}: {n:,}")
    else:
        print("  Neo4j: empty graph")


def reset_data_files() -> None:
    """Delete pipeline state files. Preserves credentials/tokens/orgs_seed.
    Also clears any *.bak.* backups since they too are stale after a reset."""
    removed: list[str] = []
    for p in DATA.iterdir():
        if not p.is_file():
            continue
        if p.name in RESET_DATA_PRESERVE:
            continue
        if p.name in RESET_DATA_FILES or ".bak" in p.name:
            p.unlink()
            removed.append(p.name)
    if removed:
        print(f"  reset-data: deleted {len(removed)} file(s)")
        for name in sorted(removed):
            print(f"    - {name}")
    else:
        print("  reset-data: nothing to delete")


def reset_neo4j_graph() -> None:
    """Delegate to scripts/reset_neo4j.py so both the standalone tool and the
    orchestrator use the same wipe logic (nodes + relationships + indexes +
    constraints, then schema re-apply)."""
    run("reset-graph", [PYTHON, str(SCRIPTS / "reset_neo4j.py"), "--yes"])


def outputs_summary() -> None:
    """Print mtime + size for each canonical output file so the user can verify
    that everything reflects the current run."""
    rows: list[tuple[str, str, str]] = []
    for name in sorted(CANONICAL_FILES):
        p = DATA / name
        if not p.exists():
            rows.append((name, "—", "missing"))
            continue
        sz = p.stat().st_size
        mt = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        if sz >= 1_000_000:
            sz_s = f"{sz / 1_000_000:.1f}M"
        elif sz >= 1000:
            sz_s = f"{sz / 1000:.1f}K"
        else:
            sz_s = f"{sz}B"
        rows.append((name, mt, sz_s))
    w = max(len(r[0]) for r in rows)
    for name, mt, sz in rows:
        print(f"  {name.ljust(w)}  {mt}  {sz}")


def preflight() -> None:
    banner("Pre-flight")

    print(f"  python: {PYTHON}")

    creds = DATA / "credentials.json"
    if not creds.exists():
        sys.exit(f"  Missing {creds} — complete the OAuth setup in SETUP.md.")
    print("  credentials.json: OK")

    for acc in ACCOUNTS:
        tok = DATA / f"token_{acc}.json"
        if not tok.exists():
            sys.exit(f"  Missing {tok} — run `pull_gmail.py --account {acc} --auth`.")
    print(f"  tokens: {', '.join(ACCOUNTS)} all present")

    # Native Neo4j readiness: a trivial Bolt query against the default db. No
    # Docker, no WSL — the store is a Windows service (or app-launched console).
    try:
        drv = neo4j_driver()
        try:
            with drv.session() as s:
                s.run("RETURN 1").consume()
        finally:
            drv.close()
    except Exception as e:
        sys.exit(f"  Neo4j not reachable over Bolt: {e}. Start the 'neo4j' "
                 "Windows service, or launch the app (serve_app) which starts "
                 "the native server automatically.")
    print("  neo4j: reachable over Bolt")


def sanity_summary() -> None:
    try:
        import neo4j  # noqa: F401
    except ImportError:
        print("(neo4j driver not installed; skipping sanity summary)")
        return

    queries = [
        ("node counts by label",
         "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n "
         "ORDER BY n DESC"),
        ("top 10 senders",
         "MATCH (p:Person)-[:SENT]->(m:Message) "
         "RETURN p.email AS email, count(m) AS sent "
         "ORDER BY sent DESC LIMIT 10"),
        ("top 10 orgs",
         "MATCH (o:Org)<-[:MENTIONS]-(m:Message) "
         "RETURN o.canonical_name AS org, count(DISTINCT m) AS msgs "
         "ORDER BY msgs DESC LIMIT 10"),
        ("top 10 matters (multi-thread)",
         "MATCH (mt:Matter) "
         "RETURN mt.canonical_subject AS matter, mt.n_threads AS threads, "
         "       mt.n_messages AS msgs "
         "ORDER BY msgs DESC LIMIT 10"),
        ("top 10 attachment filenames",
         "MATCH (a:Attachment) "
         "RETURN a.filename AS filename, count(*) AS n "
         "ORDER BY n DESC LIMIT 10"),
        ("calendar events by source calendar",
         "MATCH (e:Event) UNWIND e.account_owners AS acct "
         "RETURN acct, count(*) AS events ORDER BY events DESC"),
        ("top 10 people by meetings (organized or invited)",
         "MATCH (p:Person)-[:ORGANIZED|INVITED]->(e:Event) "
         "RETURN p.email AS email, count(DISTINCT e) AS meetings "
         "ORDER BY meetings DESC LIMIT 10"),
        ("thread graph: edge + root counts",
         "MATCH (m:Message) "
         "OPTIONAL MATCH (:Message)-[rt:REPLY_TO]->(:Message) "
         "OPTIONAL MATCH (:Message)-[nx:NEXT_IN_THREAD]->(:Message) "
         "RETURN count(DISTINCT m) AS messages, "
         "       count(DISTINCT rt) AS reply_to, "
         "       count(DISTINCT nx) AS next_in_thread, "
         "       sum(CASE WHEN m.is_thread_root THEN 1 ELSE 0 END) AS roots"),
        ("thread graph: deepest reply chain (capped 50)",
         "MATCH path=(:Message)-[:REPLY_TO*1..50]->(:Message) "
         "RETURN max(length(path)) AS deepest_chain"),
        ("thread graph: replies crossing Gmail thread/account (true-RFC proof)",
         "MATCH (c:Message)-[:REPLY_TO]->(p:Message) "
         "MATCH (c)-[:IN_THREAD]->(ct:Thread) "
         "MATCH (p)-[:IN_THREAD]->(pt:Thread) "
         "WHERE ct.gmail_thread_id <> pt.gmail_thread_id "
         "   OR ct.account_owner <> pt.account_owner "
         "RETURN count(*) AS cross_thread_replies"),
        ("thread graph: dangling parents (reply to msg outside corpus)",
         "MATCH (m:Message) "
         "WHERE m.in_reply_to IS NOT NULL AND NOT (m)-[:REPLY_TO]->(:Message) "
         "RETURN count(m) AS dangling_parent"),
    ]

    with neo4j_driver() as drv, drv.session() as sess:
        for title, q in queries:
            print(f"\n  -- {title} --")
            rows = list(sess.run(q))
            if not rows:
                print("    (no results)")
                continue
            for row in rows:
                print("    " + "  ".join(f"{k}={v}" for k, v in row.data().items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--months", type=int,
                   help="Backfill window in months (today minus N months).")
    g.add_argument("--since",
                   help="Explicit ISO date YYYY-MM-DD for --since.")
    g.add_argument("--all-time", action="store_true",
                   help="Omit --since (pull all mail).")
    p.add_argument("--reset-data", action="store_true",
                   help="Delete pipeline state files (emails*.jsonl, trackers, "
                        "sync_state) before running. Preserves credentials, "
                        "tokens, and orgs_seed.json. Use when starting fresh.")
    p.add_argument("--reset-graph", action="store_true",
                   help="Wipe the Neo4j graph (DETACH DELETE all nodes) and re-apply "
                        "the schema before running. Useful for a clean re-load.")
    p.add_argument("--skip-sanity", action="store_true",
                   help="Don't print the post-load sanity summary.")
    p.add_argument("--skip-embed", action="store_true",
                   help="Skip the embedding step (Message.embedding stays "
                        "stale / null for any new messages).")
    p.add_argument("--reembed", action="store_true",
                   help="Recompute embeddings for ALL Messages, not just "
                        "those missing one. Use after changing the embed model.")
    args = p.parse_args()

    if args.months:
        since = (datetime.date.today() -
                 datetime.timedelta(days=args.months * 30)).isoformat()
    elif args.since:
        since = args.since
    else:
        since = None

    banner(f"Pipeline run — since={since or '(all time)'}")
    preflight()

    banner("Preloaded state probe (informational)")
    state = detect_preloaded_state()
    print_preloaded_state(state)

    if args.reset_data:
        banner("Reset: deleting pipeline state files")
        reset_data_files()
    if args.reset_graph:
        banner("Reset: wiping Neo4j graph")
        reset_neo4j_graph()

    banner("Cleanup: prune stale backups + ad-hoc logs from data/")
    prune_data_dir()

    banner("Step 1/6: Pull (sequential)")
    for acc in ACCOUNTS:
        cmd = [PYTHON, str(SCRIPTS / "pull_gmail.py"), "--account", acc]
        if since:
            cmd += ["--since", since]
        run(f"pull/{acc}", cmd)

    # pull_calendar.py requires an explicit --since (the Calendar API window).
    # For --all-time (since is None) the mail pull omits --since to fetch all;
    # calendars have no equivalent "all" switch, so floor at a date safely
    # older than any real calendar data.
    cal_since = since or "2000-01-01"
    banner(f"Step 1b/6: Pull calendars (sequential, since={cal_since})")
    for acc in ACCOUNTS:
        run(f"cal/{acc}",
            [PYTHON, str(SCRIPTS / "pull_calendar.py"),
             "--account", acc, "--since", cal_since])

    banner("Step 2/6: Repair emails.jsonl")
    run("repair", [PYTHON, str(SCRIPTS / "repair_emails_jsonl.py")])

    banner("Step 3/6: Clean bodies")
    run("clean", [PYTHON, str(SCRIPTS / "clean_bodies.py")])

    banner("Step 4/6: Load Neo4j")
    run("load", [PYTHON, str(SCRIPTS / "load_neo4j.py")])

    if not args.skip_embed:
        # Idempotent checkpoint: embed_messages.py only touches Messages whose
        # embedding is null (or every Message with --reembed). Safe to interrupt
        # and re-run — the next invocation picks up where this one left off.
        label = "embed (reembed all)" if args.reembed else "embed (missing only)"
        banner(f"Step 5/6: {label}")
        embed_cmd = [PYTHON, str(SCRIPTS / "embed_messages.py")]
        if args.reembed:
            embed_cmd.append("--reembed")
        run("embed", embed_cmd)
    else:
        banner("Step 5/6: Embed messages — SKIPPED (--skip-embed)")

    banner("Step 6/6: Cluster matters")
    run("cluster", [PYTHON, str(SCRIPTS / "cluster_matters.py")])

    if not args.skip_sanity:
        banner("Sanity summary")
        sanity_summary()

    banner("Canonical outputs (size + mtime — should all reflect this run)")
    outputs_summary()

    banner("Pipeline complete.")


if __name__ == "__main__":
    main()
