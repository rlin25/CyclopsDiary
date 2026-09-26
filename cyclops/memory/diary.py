"""The diary: the permanent record of what was SEEN.

Core Principle 1 — append-only. This module exposes no way to update or delete an entry, and
that absence is the point: the diary is the thing every belief must trace back to, so an
entry that could be rewritten would make every answer unverifiable after the fact.

Core Principle 2 — diary before belief. `append()` returns only after the write succeeded.
Callers update a belief on that return value, never before it and never without it.

Shape is frozen in docs/INTERFACES.md §2.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Iterable

log = logging.getLogger(__name__)

EVENTS = frozenset({"appeared", "left_view", "picked_up", "placed", "missing", "matched"})
ORIGINS = frozenset({"observed", "inferred", "stated"})

#: Events that say where something is. A `matched` event is inferred, not observed, but it
#: still carries a place — that is what makes a belief traceable to it.
PLACE_EVENTS = frozenset({"appeared", "placed", "matched"})


class DiaryError(ValueError):
    """A malformed entry. Raised before the write, so a bad entry never lands."""


class Diary:
    """Append-only writer/reader over the `diary` time series collection."""

    def __init__(self, collection):
        self._collection = collection

    # ------------------------------------------------------------------------- writing

    def append(
        self,
        *,
        t: dt.datetime,
        session: str,
        glasses: str,
        object_id: str,
        event: str,
        origin: str,
        confidence: float,
        clip_path: str,
        offset_s: float,
        place_id: str | None = None,
        carried_by: str | None = None,
        track_id: int | None = None,
        evidence: dict | None = None,
    ) -> dict:
        """Write one entry and return it exactly as stored.

        Validation happens here rather than at the call sites so that every path into the
        diary is held to the same shape — including the ones written later under time
        pressure.
        """
        if event not in EVENTS:
            raise DiaryError(f"unknown event {event!r}; expected one of {sorted(EVENTS)}")
        if origin not in ORIGINS:
            raise DiaryError(f"unknown origin {origin!r}; expected one of {sorted(ORIGINS)}")
        if t.tzinfo is None:
            raise DiaryError("t must be timezone-aware; a naive timestamp has no place on a shared timeline")
        if not 0.0 <= float(confidence) <= 1.0:
            raise DiaryError(f"confidence {confidence!r} is outside 0-1")
        if not object_id:
            raise DiaryError("object_id is required — an entry nothing can be traced to is not evidence")
        if not clip_path:
            raise DiaryError("clip_path is required — an entry with no clip cannot be shown as proof")
        if event == "matched":
            if origin != "inferred":
                raise DiaryError("a matched event is always origin='inferred'")
            if not evidence:
                raise DiaryError(
                    "a matched event must carry evidence (query_clip, matched_clip). Core "
                    "Principle 3: an inferred belief has to point at what produced it."
                )
        elif evidence is not None:
            raise DiaryError(f"evidence is only carried by matched events, not {event!r}")

        entry: dict[str, Any] = {
            "t": t.astimezone(dt.timezone.utc),
            "meta": {"session": session, "glasses": glasses},
            "object_id": object_id,
            "event": event,
            "place_id": place_id,
            "carried_by": carried_by,
            "origin": origin,
            "confidence": float(confidence),
            "clip": {"clip_path": clip_path, "offset_s": float(offset_s)},
        }
        if track_id is not None:
            entry["track_id"] = int(track_id)
        if evidence is not None:
            entry["evidence"] = evidence

        self._collection.insert_one(entry)
        entry.pop("_id", None)  # time series inserts add one; callers never need it
        log.debug("diary += %s %s %s", object_id, event, place_id)
        return entry

    # ------------------------------------------------------------------------- reading

    def entries_for(self, object_id: str, limit: int | None = None) -> list[dict]:
        cursor = self._collection.find({"object_id": object_id}).sort("t", 1)
        if limit:
            cursor = cursor.limit(limit)
        return [_clean(doc) for doc in cursor]

    def last_event(self, object_id: str) -> dict | None:
        docs = list(self._collection.find({"object_id": object_id}).sort("t", -1).limit(1))
        return _clean(docs[0]) if docs else None

    def between(
        self,
        start: dt.datetime,
        end: dt.datetime,
        object_id: str | None = None,
        session: str | None = None,
    ) -> list[dict]:
        query: dict[str, Any] = {"t": {"$gte": start, "$lte": end}}
        if object_id:
            query["object_id"] = object_id
        if session:
            query["meta.session"] = session
        return [_clean(doc) for doc in self._collection.find(query).sort("t", 1)]

    def last_confirmed_sighting(self, object_id: str) -> dict | None:
        """The most recent entry a human would call a sighting.

        Core Principle 4 leans on this: a newer observed sighting always beats an inferred
        location, and answering from one is what "degrade honestly" falls back to.
        """
        docs = list(
            self._collection.find({
                "object_id": object_id,
                "origin": "observed",
                "event": {"$in": ["appeared", "placed", "picked_up"]},
            }).sort("t", -1).limit(1)
        )
        return _clean(docs[0]) if docs else None

    def events_of_type(self, event: str, session: str | None = None) -> list[dict]:
        query: dict[str, Any] = {"event": event}
        if session:
            query["meta.session"] = session
        return [_clean(doc) for doc in self._collection.find(query).sort("t", 1)]

    def count(self) -> int:
        return self._collection.count_documents({})


def _clean(doc: dict) -> dict:
    out = dict(doc)
    out.pop("_id", None)
    return out


def to_json_safe(entries: Iterable[dict]) -> list[dict]:
    """Diary entries with datetimes as ISO strings — for the §4 cache, the agent, and the API."""
    safe = []
    for entry in entries:
        item = dict(entry)
        if isinstance(item.get("t"), dt.datetime):
            item["t"] = item["t"].astimezone(dt.timezone.utc).isoformat()
        safe.append(item)
    return safe
