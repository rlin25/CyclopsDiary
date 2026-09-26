"""Collections, indexes, and the client that reaches them.

Two hard constraints live here:

* **Nothing in this codebase drops anything.** The Atlas cluster is shared with unrelated
  databases (`cyclopsdiary`, `lastseen_smoke`) belonging to other work. Every helper is
  create-if-absent, and `get_db()` refuses any database name outside `ALLOWED_DATABASES`.
* **`diary` is a time series collection.** That is what makes it append-only in practice,
  and it is also why it cannot hold fingerprints or carry a search index — hence the split
  into five collections described in IMPLEMENTATION_STRATEGY.md.
"""

from __future__ import annotations

import time

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from cyclops import config

DIARY = "diary"
OBJECTS = "objects"
FINGERPRINTS = "fingerprints"
PLACES = "places"
RUNS = "runs"

ALL_COLLECTIONS = (DIARY, OBJECTS, FINGERPRINTS, PLACES, RUNS)

VECTOR_INDEX_NAME = "fingerprints_vec"

#: The only vector index in the system. The sandbox tier caps total search indexes (3), so
#: object-crop fingerprints get the one; places are compared in plain code instead.
VECTOR_INDEX_DEFINITION = {
    "fields": [
        {"type": "vector", "path": "vec", "numDimensions": config.EMBED_DIM, "similarity": "cosine"},
        {"type": "filter", "path": "place_id"},
        {"type": "filter", "path": "last_seen_at"},
    ]
}


class DatabaseNotAllowed(RuntimeError):
    """Refused a database outside ALLOWED_DATABASES. The cluster hosts other projects."""


def get_client(uri: str | None = None, **kwargs) -> MongoClient:
    kwargs.setdefault("serverSelectionTimeoutMS", 15000)
    kwargs.setdefault("appname", "CyclopsDiary")
    return MongoClient(uri or config.mongodb_uri(), **kwargs)


def get_db(name: str | None = None, client: MongoClient | None = None) -> Database:
    """Open one of our own databases. Refuses anything else, by name, before connecting."""
    name = name or config.mongodb_db_name()
    if name not in config.ALLOWED_DATABASES:
        raise DatabaseNotAllowed(
            f"refusing to open database {name!r}: this cluster is shared with unrelated "
            f"projects. Allowed: {sorted(config.ALLOWED_DATABASES)}."
        )
    return (client or get_client())[name]


# --------------------------------------------------------------------------- collections

def ensure_collections(db: Database) -> dict[str, str]:
    """Create any missing collection. Never drops, never alters an existing one.

    Returns {collection: "created" | "present"} so the caller can report honestly.
    """
    existing = set(db.list_collection_names())
    report: dict[str, str] = {}

    if DIARY in existing:
        report[DIARY] = "present"
    else:
        db.create_collection(
            DIARY,
            timeseries={"timeField": "t", "metaField": "meta", "granularity": "seconds"},
        )
        report[DIARY] = "created"

    for name in (OBJECTS, FINGERPRINTS, PLACES, RUNS):
        if name in existing:
            report[name] = "present"
        else:
            db.create_collection(name)
            report[name] = "created"
    return report


def ensure_indexes(db: Database) -> dict[str, str]:
    """Ordinary indexes. `create_index` is already idempotent for an identical spec."""
    report: dict[str, str] = {}

    db[OBJECTS].create_index([("label", ASCENDING)], name="label")
    db[OBJECTS].create_index([("place_id", ASCENDING), ("status", ASCENDING)], name="place_status")
    report[OBJECTS] = "label, place_status"

    db[FINGERPRINTS].create_index([("object_id", ASCENDING)], name="object_id")
    report[FINGERPRINTS] = "object_id"

    db[PLACES].create_index([("place_id", ASCENDING)], name="place_id", unique=True)
    report[PLACES] = "place_id (unique)"

    db[RUNS].create_index([("created_at", DESCENDING)], name="created_at")
    report[RUNS] = "created_at"

    # Time series secondary indexes. metaField/timeField are always indexable; an index on
    # a measurement field (object_id) needs a recent server, so it is attempted and the
    # failure is reported rather than raised — timeline() still works without it, just with
    # a collection scan.
    db[DIARY].create_index([("meta.session", ASCENDING), ("t", ASCENDING)], name="session_t")
    diary_indexes = ["session_t"]
    try:
        db[DIARY].create_index([("object_id", ASCENDING), ("t", ASCENDING)], name="object_t")
        diary_indexes.append("object_t")
    except OperationFailure as exc:
        diary_indexes.append(f"object_t unavailable ({exc.code}) - timeline() will scan")
    report[DIARY] = ", ".join(diary_indexes)
    return report


# -------------------------------------------------------------------------- vector index

def vector_index_status(db: Database) -> dict | None:
    """The existing vector index document, or None. Returns None (not an error) when the
    deployment has no Atlas Search at all, so callers can report that plainly."""
    try:
        for index in db[FINGERPRINTS].list_search_indexes():
            if index.get("name") == VECTOR_INDEX_NAME:
                return dict(index)
    except OperationFailure:
        return None
    return None


def ensure_vector_index(db: Database, wait: bool = True, timeout_s: float = 300.0) -> str:
    """Create the fingerprints vector index if absent, then optionally wait for it.

    The build is asynchronous. Without the wait, the first `$vectorSearch` after setup
    fails with an error that reads like a bug in the query rather than an index that is
    still building.
    """
    existing = vector_index_status(db)
    if existing is None:
        try:
            db[FINGERPRINTS].create_search_index(
                SearchIndexModel(
                    definition=VECTOR_INDEX_DEFINITION,
                    name=VECTOR_INDEX_NAME,
                    type="vectorSearch",
                )
            )
        except OperationFailure as exc:
            raise RuntimeError(
                f"could not create the vector index: {exc}. Atlas Search must be available "
                "on this cluster tier — confirm the hackathon sandbox supports vector search."
            ) from exc
        state = "created"
    else:
        state = "present"

    if not wait:
        return state

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        index = vector_index_status(db)
        if index is None:
            return f"{state} (status unreadable on this deployment)"
        if index.get("queryable"):
            return f"{state}, queryable"
        if index.get("status") == "FAILED":
            raise RuntimeError(f"vector index build FAILED: {index}")
        time.sleep(3.0)
    return f"{state}, still building after {timeout_s:.0f}s"
