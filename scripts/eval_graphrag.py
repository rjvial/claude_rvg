#!/usr/bin/env python
"""Read-only Graph-RAG eval harness for breadth / timeline questions.

The /api/ask Graph-RAG is good at DEPTH questions ("status of matter Z") but
weak at BREADTH / TIMELINE questions that span the corpus's full 2012-2026
range ("year-by-year history of my dealings with Acme", "list every
counterparty at Globex"). This harness measures that weakness with numbers,
so retrieval / prompt changes can be compared against a baseline instead of
shipped blind.

It self-grounds: for each case the GOLD answer (the years the anchor entity was
actually active, and the set of counterparties) is computed directly from the
graph with Cypher, so no hand-labelling is needed. Two levels are scored:

  Level A — RETRIEVAL DIAGNOSTIC (deterministic, fast). Runs the real
    graph_rag.retrieve() and asks: of the entity's GOLD active years, how many
    does the static context bundle actually surface? Isolates the bundle's
    recall ceiling from the LLM.

  Level B — END-TO-END (slow; one `claude -p` per case). Runs the real ask
    pipeline — same context bundle, same read-only Neo4j tools, the LIVE
    serve_app.ASK_SYSTEM prompt — and scores the FINAL ANSWER's temporal span
    coverage and (for enumeration cases) counterparty recall. This is the metric
    the user actually experiences; a prompt edit in serve_app is picked up
    automatically because ASK_SYSTEM is imported, not copied.

Primary metric: TEMPORAL SPAN COVERAGE — distinct gold years referenced /
gold years, and the same restricted to the EARLY half of the span (where the
vector-only retriever is expected to miss). Maps directly to "can it analyse
all of them across the decades".

Nothing here writes to the graph or to production data. Usage:

    python scripts/eval_graphrag.py                 # all cases, both levels
    python scripts/eval_graphrag.py --no-e2e        # Level A only (fast)
    python scripts/eval_graphrag.py --cases acme-timeline,person-timeline
    python scripts/eval_graphrag.py --out C:/Temp/eval.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import ROOT, bootstrap_venv, force_utf8  # noqa: E402

force_utf8()
bootstrap_venv()

import graph_app    # noqa: E402  driver()
import graph_rag    # noqa: E402  retrieve(), build_context()

# The LIVE ask-step system prompt + tool allow-list. Imported (not copied) so a
# Stage-1 edit to serve_app.ASK_SYSTEM is reflected here without touching this
# file. serve_app's module-level bootstrap_venv()/force_utf8() are idempotent.
import serve_app    # noqa: E402

ALLOWED_TOOLS = "mcp__neo4j__read_neo4j_cypher,mcp__neo4j__get_neo4j_schema"
CLAUDE_TIMEOUT = 600          # seconds per end-to-end case
YEAR_RE = re.compile(r"\b(20[0-3]\d)\b")     # 2000-2039 — corpus is 2012-2026
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
MAJOR_MIN_MSGS = 5            # a "major" counterparty: >= this many messages


# --- eval cases -------------------------------------------------------------
# Anchored on real, long-history entities found in the graph (2012-2026).
# kind drives both the GOLD query and the scoring. Questions are in Spanish to
# match the corpus.
CASES = [
    {
        "id": "acme-timeline", "kind": "timeline_org",
        "anchor": ("org_domain", "acme.example.com"),
        "question": ("Hazme un resumen año por año de toda mi relación con "
                     "Acme, desde el primer correo hasta hoy."),
    },
    {
        "id": "acme-people", "kind": "enum_people_org",
        "anchor": ("org_domain", "acme.example.com"),
        "question": ("Lista todas las personas de Acme con las que he "
                     "intercambiado correos a lo largo de los años."),
    },
    {
        "id": "person-timeline", "kind": "timeline_person",
        "anchor": ("person_email", "jane@acme.example.com"),
        "question": ("Traza la historia de mis correos con Jane Doe "
                     "(jane@acme.example.com) desde el inicio hasta ahora."),
    },
]

# The cases above are GENERIC TEMPLATES. Real eval cases are anchored to private
# corpus entities (people, orgs, message-ids) and are kept out of version
# control: drop a data/eval_cases_local.py that defines its own CASES list and it
# overrides the templates here (see data/, which is gitignored).
try:
    _local = ROOT / "data"
    if str(_local) not in sys.path:
        sys.path.insert(0, str(_local))
    from eval_cases_local import CASES  # type: ignore  # noqa: F811
except ImportError:
    pass


# --- gold computation (authoritative, from the graph) -----------------------
def gold_for(session, case: dict) -> dict:
    """Compute the ground-truth answer envelope from the graph: per-year
    message volume for the anchor entity (always), plus the counterparty set
    for enumeration cases. A message counts for an org if ANY participant
    (sender or recipient) has an address at that domain; for a person if they
    are the sender or a recipient."""
    atype, aval = case["anchor"]
    if atype == "org_domain":
        rows = session.run(
            """
            MATCH (p:Person)-[:SENT|RECEIVED_BY]-(m:Message)
            WHERE toLower(p.email) ENDS WITH '@' + $val
            WITH substring(m.sent_at, 0, 4) AS yr, count(DISTINCT m) AS c
            WHERE yr <> ''
            RETURN yr, c ORDER BY yr
            """, val=aval)
    elif atype == "person_emails":   # several addresses for one person
        rows = session.run(
            """
            MATCH (p:Person)-[:SENT|RECEIVED_BY]-(m:Message)
            WHERE toLower(p.email) IN $vals
            WITH substring(m.sent_at, 0, 4) AS yr, count(DISTINCT m) AS c
            WHERE yr <> ''
            RETURN yr, c ORDER BY yr
            """, vals=[e.lower() for e in aval])
    else:  # person_email
        rows = session.run(
            """
            MATCH (p:Person)-[:SENT|RECEIVED_BY]-(m:Message)
            WHERE toLower(p.email) = toLower($val)
            WITH substring(m.sent_at, 0, 4) AS yr, count(DISTINCT m) AS c
            WHERE yr <> ''
            RETURN yr, c ORDER BY yr
            """, val=aval)
    year_vol = {r["yr"]: r["c"] for r in rows}

    people = []
    if case["kind"] == "enum_people_org":
        prows = session.run(
            """
            MATCH (p:Person)-[:SENT|RECEIVED_BY]-(m:Message)
            WHERE toLower(p.email) ENDS WITH '@' + $val
            RETURN toLower(p.email) AS email, p.name AS name,
                   count(DISTINCT m) AS msgs
            ORDER BY msgs DESC
            """, val=aval)
        people = [dict(r) for r in prows]

    return {"year_vol": year_vol, "people": people}


# --- scoring helpers --------------------------------------------------------
def _span_metrics(gold_year_vol: dict, present_years: set) -> dict:
    """Coverage of the gold active years by a set of present years, overall and
    restricted to the EARLY half of the gold span (the part a recency / nearest
    -neighbour bias drops). Volume-weighted coverage weights each year by its
    message count, so missing a busy early year hurts more than a quiet one."""
    gold_years = sorted(gold_year_vol)
    if not gold_years:
        return {"n_gold_years": 0}
    lo, hi = int(gold_years[0]), int(gold_years[-1])
    mid = lo + (hi - lo) / 2.0
    early = [y for y in gold_years if int(y) <= mid]
    late = [y for y in gold_years if int(y) > mid]
    covered = present_years & set(gold_years)
    total_vol = sum(gold_year_vol.values()) or 1
    cov_vol = sum(gold_year_vol[y] for y in covered)
    return {
        "gold_span": f"{lo}-{hi}",
        "n_gold_years": len(gold_years),
        "coverage": round(len(covered) / len(gold_years), 3),
        "coverage_vol": round(cov_vol / total_vol, 3),
        "early_coverage": (round(len([y for y in early if y in covered])
                                 / len(early), 3) if early else None),
        "late_coverage": (round(len([y for y in late if y in covered])
                                / len(late), 3) if late else None),
        "years_present": sorted(covered),
        "years_missing": [y for y in gold_years if y not in covered],
    }


def level_a(session, case: dict, gold: dict) -> dict:
    """RETRIEVAL DIAGNOSTIC: run the real retriever and measure how much of the
    anchor's gold timeline the static bundle surfaces."""
    atype, aval = case["anchor"]
    seeds = graph_rag.retrieve(session, case["question"])
    n_convs = len(seeds)
    n_msgs = sum(len(c["messages"]) for c in seeds)

    email_set = ({e.lower() for e in aval} if atype == "person_emails"
                 else {aval.lower()} if atype == "person_email" else set())
    bundle_years: set = set()          # every year present in the bundle
    anchor_years: set = set()          # years of messages that touch the anchor
    anchor_emails: set = set()         # anchor-domain senders seen (org cases)
    bundle_ids: set = set()            # gmail_message_ids present in the bundle
    for conv in seeds:
        for m in conv["messages"]:
            yr = (m.get("sent_at") or "")[:4]
            if yr:
                bundle_years.add(yr)
            if m.get("mid"):
                bundle_ids.add(m["mid"])
            fe = (m.get("from_email") or "").lower()
            on_anchor = (fe.endswith("@" + aval) if atype == "org_domain"
                         else fe in email_set)
            if on_anchor:
                if yr:
                    anchor_years.add(yr)
                if fe:
                    anchor_emails.add(fe)

    out = {
        "n_convs": n_convs, "n_msgs_in_bundle": n_msgs,
        # The bundle as a whole (what the LLM literally sees)...
        "bundle": _span_metrics(gold["year_vol"], bundle_years),
        # ...and only the messages that actually touch the anchor entity.
        "anchor_only": _span_metrics(gold["year_vol"], anchor_years),
    }
    if case["kind"] == "enum_people_org":
        major = [p for p in gold["people"] if p["msgs"] >= MAJOR_MIN_MSGS]
        found = {e for e in anchor_emails}
        out["people"] = {
            "gold_total": len(gold["people"]),
            "gold_major": len(major),
            "found_in_bundle": len(found & {p["email"] for p in major}),
            "recall_major": (round(len(found & {p["email"] for p in major})
                                   / len(major), 3) if major else None),
        }
    if case["kind"] == "enum_instances":
        gids = set(case.get("gold_msg_ids", []))
        found = gids & bundle_ids
        out["gold_msgs"] = {
            "total": len(gids), "in_bundle": len(found),
            "recall": round(len(found) / len(gids), 3) if gids else None,
            "missing": sorted(gids - found),
        }
    return out


# --- end-to-end (Level B) ---------------------------------------------------
def _run_claude(prompt: str) -> dict:
    """Run the real ask step: `claude -p` with the live ASK_SYSTEM and the
    read-only Neo4j tools, fed the same prompt serve_app builds. Returns the
    final answer plus a trace of the Cypher the model chose to run (so we can
    see whether it aggregated)."""
    claude = shutil.which("claude")
    if not claude:
        return {"error": "the `claude` CLI is not on PATH"}
    cmd = [claude, "-p", "--output-format", "stream-json", "--verbose",
           "--allowedTools", ALLOWED_TOOLS,
           "--append-system-prompt", serve_app.ASK_SYSTEM]
    if claude.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c", *cmd]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, input=prompt, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=CLAUDE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {CLAUDE_TIMEOUT}s"}

    answer = ""
    queries: list[str] = []
    n_tool = 0
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in (ev.get("message", {}).get("content") or []):
                if block.get("type") == "tool_use":
                    n_tool += 1
                    q = (block.get("input") or {}).get("query")
                    if q:
                        queries.append(q)
        elif ev.get("type") == "result":
            answer = (ev.get("result") or "").strip()
    if not answer and proc.returncode != 0:
        return {"error": f"claude exit {proc.returncode}: "
                         f"{(proc.stderr or '')[:300]}"}
    return {"answer": answer, "n_tool_calls": n_tool, "queries": queries}


def level_b(case: dict, gold: dict, seeds_context: str) -> dict:
    """END-TO-END: build the serve_app prompt, run it through claude, and score
    the final answer's temporal span coverage (and counterparty recall)."""
    prompt = (f"QUESTION:\n{case['question']}\n\n"
              f"RETRIEVED CONTEXT (semantic search over the mailbox for this "
              f"question — cite with the [n] markers you use):"
              f"\n\n{seeds_context}\n")
    res = _run_claude(prompt)
    if "error" in res:
        return res
    answer = res["answer"]
    ans_years = {y for y in YEAR_RE.findall(answer) if y in gold["year_vol"]}
    out = {
        "n_tool_calls": res["n_tool_calls"],
        "n_queries": len(res["queries"]),
        "answer_len": len(answer),
        "span": _span_metrics(gold["year_vol"], ans_years),
        "queries": res["queries"],
        "answer": answer,
    }
    if case["kind"] == "enum_people_org":
        major = [p for p in gold["people"] if p["msgs"] >= MAJOR_MIN_MSGS]
        low = answer.lower()
        ans_emails = {e.lower() for e in EMAIL_RE.findall(answer)}
        hit = 0
        for p in major:
            name = (p.get("name") or "").strip()
            # Match if the address appears, or a distinctive name token does.
            tokens = [t for t in re.split(r"[\s,<>]+", name.lower())
                      if len(t) >= 4]
            if p["email"] in ans_emails or any(t in low for t in tokens):
                hit += 1
        out["people"] = {
            "gold_major": len(major),
            "named_in_answer": hit,
            "recall_major": (round(hit / len(major), 3) if major else None),
        }
    if case["kind"] == "enum_instances":
        out["instances"] = _score_instances(case, answer)
    return out


def _score_instances(case: dict, answer: str) -> dict:
    """Score an ENUMERATION answer: did it surface each distinct gold instance,
    link the required entity (must-mention), keep the distractor citations
    separate, and avoid the 'una sola…' undercount that buries instances?
    Token presence is a proxy — instance_recall + undercount_flag together
    capture the real failure (instance retrieved but framed as a single one)."""
    low = answer.lower()
    covered = [{"key": gi["key"],
                "covered": any(t in low for t in gi["any"]),
                "year_present": gi["year"] in answer}
               for gi in case["gold_instances"]]
    n = len(covered) or 1
    return {
        "instance_recall": round(sum(c["covered"] for c in covered) / n, 3),
        "instances": covered,
        "mention_linked": all(t in low for t in case.get("must_mention", [])),
        "distractors_separated": [
            {"key": d["key"], "present": any(t in low for t in d["any"])}
            for d in case.get("distractors", [])],
        "undercount_flag": bool(case.get("undercount_re")
                                and re.search(case["undercount_re"], low)),
    }


# --- driver -----------------------------------------------------------------
def probe(question: str) -> None:
    """Ad-hoc, UNSCORED end-to-end run of one free-form question through the
    real pipeline — retrieve → build_context → live ASK_SYSTEM via claude -p.
    Prints the bundle's year span, the Cypher the agent chose to run, and the
    final answer. For spot-checking question shapes the scored cases don't
    cover (multi-entity, concept search, relationship inference)."""
    drv = graph_app.driver()
    try:
        with drv.session() as s:
            seeds = graph_rag.retrieve(s, question)
            ctx, _ = graph_rag.build_context(seeds)
    finally:
        drv.close()
    years = sorted({(m.get("sent_at") or "")[:4]
                    for c in seeds for m in c["messages"] if m.get("sent_at")})
    n_msgs = sum(len(c["messages"]) for c in seeds)
    print(f"\n[probe] question: {question}", flush=True)
    print(f"[probe] bundle: {len(seeds)} convs / {n_msgs} msgs | "
          f"years present: {years[0] if years else '—'}"
          f"…{years[-1] if years else '—'} ({len(years)} distinct)", flush=True)
    print("[probe] running claude -p (live ASK_SYSTEM + read-only Cypher)…",
          flush=True)
    prompt = (f"QUESTION:\n{question}\n\n"
              f"RETRIEVED CONTEXT (semantic search over the mailbox for this "
              f"question — cite with the [n] markers you use):\n\n{ctx}\n")
    res = _run_claude(prompt)
    if "error" in res:
        print(f"[probe] ERROR: {res['error']}", flush=True)
        return
    print(f"\n[probe] agent ran {res['n_tool_calls']} tool call(s), "
          f"{len(res['queries'])} with a Cypher query:", flush=True)
    for q in res["queries"]:
        print("   - " + " ".join(q.split())[:300], flush=True)
    print("\n" + "=" * 78 + "\nANSWER\n" + "=" * 78, flush=True)
    print(res["answer"], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only Graph-RAG eval harness.")
    ap.add_argument("--cases", help="comma-separated case ids (default: all)")
    ap.add_argument("--no-e2e", action="store_true",
                    help="Level A retrieval diagnostic only (skip claude -p)")
    ap.add_argument("--ask", help="ad-hoc: run one free-form question "
                    "end-to-end (unscored) and print the answer + queries")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run Level B N times per case and aggregate "
                    "(Level A is deterministic, run once). Use >=3 to measure "
                    "the agent's run-to-run variance, not a single sample.")
    ap.add_argument("--out", help="write full JSON results to this path")
    args = ap.parse_args()

    if args.ask:
        probe(args.ask)
        return

    cases = CASES
    if args.cases:
        want = {c.strip() for c in args.cases.split(",")}
        cases = [c for c in CASES if c["id"] in want]
        if not cases:
            sys.exit(f"no matching cases in {want}")

    print(f"[eval] {len(cases)} case(s), "
          f"{'Level A only' if args.no_e2e else 'Level A + B'}", flush=True)
    print("[eval] loading retriever (embedding model warms on first "
          "question)…", flush=True)

    drv = graph_app.driver()
    results = []
    try:
        for i, case in enumerate(cases, 1):
            print(f"\n[eval] ({i}/{len(cases)}) {case['id']} — {case['kind']}",
                  flush=True)
            with drv.session() as s:
                gold = gold_for(s, case)
                a = level_a(s, case, gold)
                # build_context off the same retrieval the diagnostic used.
                ctx, _ = graph_rag.build_context(
                    graph_rag.retrieve(s, case["question"]))
            row = {"id": case["id"], "kind": case["kind"],
                   "anchor": list(case["anchor"]),
                   "question": case["question"],
                   "gold": {"span": a["bundle"].get("gold_span"),
                            "n_years": a["bundle"].get("n_gold_years"),
                            "year_vol": gold["year_vol"],
                            "n_people": len(gold["people"])},
                   "level_a": a}
            ga = a["anchor_only"]
            print(f"        gold span {ga.get('gold_span')} "
                  f"({ga.get('n_gold_years')} active years)", flush=True)
            print(f"        Level A  bundle: {a['n_convs']} convs / "
                  f"{a['n_msgs_in_bundle']} msgs | "
                  f"anchor-year coverage {ga.get('coverage')} "
                  f"(early {ga.get('early_coverage')}, "
                  f"late {ga.get('late_coverage')})", flush=True)
            if "people" in a:
                pp = a["people"]
                print(f"        Level A  people recall(major≥{MAJOR_MIN_MSGS}) "
                      f"{pp.get('recall_major')} "
                      f"({pp.get('found_in_bundle')}/{pp.get('gold_major')})",
                      flush=True)
            if "gold_msgs" in a:
                gm = a["gold_msgs"]
                print(f"        Level A  gold-msg recall {gm.get('recall')} "
                      f"({gm.get('in_bundle')}/{gm.get('total')} key messages "
                      f"in bundle)", flush=True)

            if not args.no_e2e:
                runs = []
                for run_i in range(1, args.repeat + 1):
                    tag = (f" (run {run_i}/{args.repeat})"
                           if args.repeat > 1 else "")
                    print(f"        Level B  running claude -p{tag} …",
                          flush=True)
                    b = level_b(case, gold, ctx)
                    runs.append(b)
                    if "error" in b:
                        print(f"        Level B  ERROR: {b['error']}",
                              flush=True)
                        continue
                    sp = b["span"]
                    extra = ""
                    if "people" in b:
                        extra = (f" | people recall "
                                 f"{b['people'].get('recall_major')} "
                                 f"({b['people'].get('named_in_answer')}/"
                                 f"{b['people'].get('gold_major')})")
                    if "instances" in b:
                        ib = b["instances"]
                        extra = (f" | instance_recall {ib['instance_recall']}"
                                 f" mention {ib['mention_linked']}"
                                 f" undercount {ib['undercount_flag']}")
                    print(f"          span cov {sp.get('coverage')} "
                          f"(early {sp.get('early_coverage')}), "
                          f"{b['n_queries']} qrys{extra}", flush=True)
                row["level_b_runs"] = runs
                row["level_b"] = runs[0]
                if args.repeat > 1:
                    agg = _aggregate(case, runs)
                    row["level_b_agg"] = agg
                    print(f"        Level B  AGG/{len(runs)}: {agg['summary']}",
                          flush=True)
            results.append(row)
    finally:
        drv.close()

    # Compact summary table.
    print("\n" + "=" * 78)
    print("SUMMARY  (coverage = gold active-years referenced; early = first "
          "half of span)")
    print("=" * 78)
    hdr = f"{'case':<24}{'span':<11}{'A.cov':>7}{'A.early':>9}"
    if not args.no_e2e:
        hdr += f"{'B.cov':>7}{'B.early':>9}{'B.qrys':>8}"
    print(hdr)
    for r in results:
        ga = r["level_a"]["anchor_only"]
        line = (f"{r['id']:<24}{str(ga.get('gold_span')):<11}"
                f"{_f(ga.get('coverage')):>7}{_f(ga.get('early_coverage')):>9}")
        if not args.no_e2e and "level_b" in r and "span" in r["level_b"]:
            sp = r["level_b"]["span"]
            line += (f"{_f(sp.get('coverage')):>7}"
                     f"{_f(sp.get('early_coverage')):>9}"
                     f"{r['level_b'].get('n_queries', 0):>8}")
        print(line)
    print("=" * 78, flush=True)

    # Aggregate table — the honest view when measuring a stochastic agent.
    if not args.no_e2e and args.repeat > 1:
        print(f"\nAGGREGATE over {args.repeat} Level-B runs per case "
              "(key metric = the one that matters for that case's kind)")
        print("=" * 78)
        print(f"{'case':<24}{'key metric':<18}{'mean':>6}{'min':>6}"
              f"{'max':>6}   notes")
        for r in results:
            agg = r.get("level_b_agg")
            if not agg:
                continue
            note = ""
            if r["kind"] == "enum_instances":
                note = (f"undercount=False {agg.get('undercount_false_count')}"
                        f"/{agg.get('undercount_total')}, "
                        f"mention {agg.get('mention_true_count')}"
                        f"/{agg.get('undercount_total')}")
            print(f"{r['id']:<24}{agg['key_metric_label']:<18}"
                  f"{_f(agg['mean']):>6}{_f(agg['min']):>6}{_f(agg['max']):>6}"
                  f"   {agg['values']} {note}")
        print("=" * 78, flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False,
                                             indent=2), encoding="utf-8")
        print(f"[eval] wrote {args.out}", flush=True)


def _metric_label(case: dict) -> str:
    return {"enum_people_org": "people_recall",
            "enum_instances": "instance_recall"}.get(case["kind"],
                                                     "span_coverage")


def _key_metric(case: dict, b: dict):
    """The single number that matters for this case's kind — span coverage for
    timelines, people recall for org enumeration, instance recall for the
    multi-instance case. None on an errored run."""
    if "error" in b:
        return None
    if case["kind"] == "enum_people_org":
        return b.get("people", {}).get("recall_major")
    if case["kind"] == "enum_instances":
        return b.get("instances", {}).get("instance_recall")
    return b.get("span", {}).get("coverage")


def _aggregate(case: dict, runs: list[dict]) -> dict:
    """Mean / min / max of the key metric across repeated runs, plus the
    boolean tallies that matter for enum_instances (how often the undercount
    failure was avoided). Distributions, not single points — the right way to
    judge a stochastic agent."""
    vals = [v for v in (_key_metric(case, b) for b in runs) if v is not None]
    n = len(vals) or 1
    agg = {
        "key_metric_label": _metric_label(case),
        "values": vals,
        "mean": round(sum(vals) / n, 3) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }
    if case["kind"] == "enum_instances":
        insts = [b["instances"] for b in runs if "instances" in b]
        agg["undercount_total"] = len(insts)
        agg["undercount_false_count"] = sum(1 for i in insts
                                            if i.get("undercount_flag") is False)
        agg["mention_true_count"] = sum(1 for i in insts
                                          if i.get("mention_linked"))
        agg["summary"] = (
            f"instance_recall {agg['values']} mean {agg['mean']} | "
            f"undercount=False {agg['undercount_false_count']}/"
            f"{agg['undercount_total']} | mention=True "
            f"{agg['mention_true_count']}/{agg['undercount_total']}")
    else:
        agg["summary"] = (f"{agg['key_metric_label']} {agg['values']} "
                          f"mean {agg['mean']} (min {agg['min']}, "
                          f"max {agg['max']})")
    return agg


def _f(x) -> str:
    return "—" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    main()
