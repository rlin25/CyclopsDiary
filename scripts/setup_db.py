#!/usr/bin/env python
"""Create CyclopsDiary's collections and its one vector index. Safe to re-run.

This script only ever creates. It does not drop, rename, or empty anything, and it refuses
to touch a database outside {cyclops, cyclops_loadtest} — the Atlas cluster is shared with
unrelated databases from other work.

    python scripts/setup_db.py                 # against MONGODB_DB (default: cyclops)
    python scripts/setup_db.py --dry-run       # report what it would do, write nothing
    python scripts/setup_db.py --db cyclops_loadtest
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cyclops import config                      # noqa: E402
from cyclops.memory import db as dbmod          # noqa: E402

ENV_KEYS = ("MONGODB_URI", "MONGODB_DB", "OPENROUTER_API_KEY", "AGENT_MODEL")

BLOCKED_BY = {
    "OPENROUTER_API_KEY": "agent loop / ask.py / eval.score",
    "AGENT_MODEL": "agent loop / ask.py / eval.score",
}


def report_env() -> None:
    """Names and present/missing only — never a value."""
    print("environment:")
    for key, present in config.env_status(ENV_KEYS).items():
        mark = "set" if present else "MISSING"
        note = "" if present else f"  -> blocks {BLOCKED_BY.get(key, 'nothing in M1')}"
        print(f"  {key:<24} {mark}{note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=None, help="database name (default: MONGODB_DB)")
    parser.add_argument("--dry-run", action="store_true", help="report only; create nothing")
    parser.add_argument("--no-wait", action="store_true", help="do not wait for the vector index build")
    parser.add_argument("--timeout", type=float, default=300.0, help="vector index wait, seconds")
    args = parser.parse_args()

    report_env()
    print()

    try:
        uri = config.mongodb_uri()
    except config.MissingCredential as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 2

    name = args.db or config.mongodb_db_name()
    if name not in config.ALLOWED_DATABASES:
        print(
            f"stopped: refusing database {name!r}. This cluster hosts unrelated projects; "
            f"allowed names are {sorted(config.ALLOWED_DATABASES)}.",
            file=sys.stderr,
        )
        return 2

    client = dbmod.get_client(uri)
    try:
        server = client.server_info()["version"]
    except Exception as exc:  # pragma: no cover - connectivity is environmental
        print(f"stopped: cannot reach Atlas ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    print(f"connected to MongoDB {server}; target database: {name}")

    database = dbmod.get_db(name, client=client)
    present = sorted(database.list_collection_names())
    print(f"existing collections in {name}: {present or '(none - new database)'}")

    if args.dry_run:
        print("\n-- dry run, nothing written --")
        for collection in dbmod.ALL_COLLECTIONS:
            state = "present" if collection in present else "would create"
            kind = " (time series)" if collection == dbmod.DIARY else ""
            print(f"  {collection:<14} {state}{kind}")
        index = dbmod.vector_index_status(database)
        if index is None:
            print(f"  {dbmod.VECTOR_INDEX_NAME:<14} would create (or Atlas Search unreadable here)")
        else:
            print(f"  {dbmod.VECTOR_INDEX_NAME:<14} present, queryable={index.get('queryable')}")
        print("\nre-run without --dry-run to apply.")
        return 0

    print("\ncollections:")
    for collection, state in dbmod.ensure_collections(database).items():
        print(f"  {collection:<14} {state}")

    print("indexes:")
    for collection, state in dbmod.ensure_indexes(database).items():
        print(f"  {collection:<14} {state}")

    print("vector index:")
    try:
        state = dbmod.ensure_vector_index(database, wait=not args.no_wait, timeout_s=args.timeout)
    except RuntimeError as exc:
        print(f"  {dbmod.VECTOR_INDEX_NAME:<14} FAILED - {exc}")
        return 1
    print(f"  {dbmod.VECTOR_INDEX_NAME:<14} {state}")

    print(f"\nsetup complete for {name}. Nothing was dropped or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
