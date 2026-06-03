"""Erase the Neo4j graph completely: nodes, relationships, indexes, constraints.

This is the destructive sibling of `load_neo4j.py --setup`. It is the canonical
wipe used by `run_pipeline.py --reset-graph`; standalone invocation lets you
nuke the graph without running the full pipeline (e.g. before a schema migration
or a fresh re-load from a canonicalized emails_clean.jsonl).

Behavior:
  1. Inventory — count nodes, relationships, list constraints + indexes.
  2. Confirm  — interactive prompt unless --yes.
  3. Drop constraints (which cascade-drops their backing indexes).
  4. Drop any remaining (non-constraint-owned) indexes.
  5. DETACH DELETE all nodes in batches of 5000.
  6. Re-apply canonical schema via `load_neo4j.py --setup` (unless --no-reapply).

Usage:
    python scripts/reset_neo4j.py                   # interactive, then full wipe + re-apply
    python scripts/reset_neo4j.py --yes             # non-interactive
    python scripts/reset_neo4j.py --dry-run         # report only, change nothing
    python scripts/reset_neo4j.py --no-reapply      # wipe and leave schema empty
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from _common import SCRIPTS, neo4j_driver

BATCH = 5000  # node-delete batch size; keeps each transaction below page-cache limits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive confirmation prompt. Required "
                             "for non-interactive use (e.g. from run_pipeline.py).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be dropped and stop. Makes no changes.")
    parser.add_argument("--no-reapply", action="store_true",
                        help="Don't run load_neo4j.py --setup after the wipe. "
                             "Leaves the database empty and schema-less.")
    args = parser.parse_args()

    print(f"Target: {os.environ.get('NEO4J_URI', 'bolt://localhost:7687')}")

    with neo4j_driver() as drv:
        with drv.session() as sess:
            node_count = sess.run("MATCH (n) RETURN count(n) AS n").single()["n"]
            rel_count = sess.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]
            constraints = [
                dict(r) for r in sess.run(
                    "SHOW CONSTRAINTS YIELD name, labelsOrTypes, type"
                )
            ]
            indexes_all = [
                dict(r) for r in sess.run(
                    "SHOW INDEXES YIELD name, labelsOrTypes, type, owningConstraint"
                )
            ]
            # Backing indexes for constraints drop along with the constraint —
            # filter them out so we don't try to drop them twice.
            free_indexes = [
                ix for ix in indexes_all
                if not ix.get("owningConstraint")
            ]

        print(f"  nodes:         {node_count:,}")
        print(f"  relationships: {rel_count:,}")
        print(f"  constraints:   {len(constraints)}")
        for c in constraints:
            labels = c.get("labelsOrTypes") or []
            print(f"    - {c['name']} on {labels} ({c['type']})")
        print(f"  indexes (standalone): {len(free_indexes)}")
        for ix in free_indexes:
            labels = ix.get("labelsOrTypes") or []
            print(f"    - {ix['name']} on {labels} ({ix['type']})")

        if args.dry_run:
            print("\nDry-run; nothing dropped.")
            return

        if not args.yes:
            print(
                "\nThis will PERMANENTLY ERASE every node, relationship, "
                "constraint, and index above."
            )
            print("Type 'yes' to confirm (any other input aborts):")
            try:
                resp = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                resp = ""
            if resp != "yes":
                print("Aborted.")
                return

        with drv.session() as sess:
            for c in constraints:
                # Backticks in case a name ever contains characters needing quoting.
                sess.run(f"DROP CONSTRAINT `{c['name']}` IF EXISTS")
            print(f"  dropped {len(constraints)} constraint(s)")

            for ix in free_indexes:
                sess.run(f"DROP INDEX `{ix['name']}` IF EXISTS")
            print(f"  dropped {len(free_indexes)} standalone index(es)")

            deleted = 0
            while True:
                res = sess.run(
                    "MATCH (n) WITH n LIMIT $batch "
                    "DETACH DELETE n RETURN count(*) AS n",
                    batch=BATCH,
                ).single()
                n = res["n"] if res else 0
                if n == 0:
                    break
                deleted += n
            print(f"  deleted {deleted:,} node(s) (relationships cascaded)")

    if args.no_reapply:
        print("Skipping schema re-apply (--no-reapply).")
    else:
        print("Re-applying canonical schema via load_neo4j.py --setup...")
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "load_neo4j.py"), "--setup"]
        )
        if r.returncode != 0:
            sys.exit(f"load_neo4j.py --setup failed (exit {r.returncode})")

    print("Graph reset complete.")


if __name__ == "__main__":
    main()
