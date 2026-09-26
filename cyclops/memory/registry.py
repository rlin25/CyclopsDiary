"""The contact list: one document per discovered object, plus its fingerprints.

`objects` is the revisable half of the record/belief split — unlike `diary`, these documents
are meant to be updated, because a belief changes as evidence arrives. What must never happen
is a belief changing *without* a diary entry behind it; that ordering is enforced in
`beliefs.py`, which is the only thing that should be calling `set_belief()`.

`fingerprints` is separate because the diary is a time series collection and cannot hold
vectors, and because it carries the system's only vector index.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any, Sequence

import numpy as np

from cyclops import config
from cyclops.memory import db as dbmod
from cyclops.perception import embed

log = logging.getLogger(__name__)

STATUSES = frozenset({"confirmed", "believed", "carried", "faded", "ambiguous"})

#: Statuses that assert a location. `faded` and `ambiguous` do not — they are the honest
#: "not sure", and `what_is_at()` must not list them as if they were sightings.
LOCATED_STATUSES = frozenset({"confirmed", "believed", "carried"})


class Registry:
    def __init__(self, database):
        self._db = database
        self._objects = database[dbmod.OBJECTS]
        self._fingerprints = database[dbmod.FINGERPRINTS]

    # ------------------------------------------------------------------------- objects

    def next_object_id(self) -> str:
        """`obj_N`, continuing from the highest existing N.

        Derived from what is stored rather than from a counter held in memory, so a second
        processing run does not restart numbering and collide with existing objects.
        """
        highest = 0
        for doc in self._objects.find({}):
            match = re.fullmatch(r"obj_(\d+)", str(doc.get("_id", "")))
            if match:
                highest = max(highest, int(match.group(1)))
        return f"obj_{highest + 1}"

    def create_object(
        self,
        *,
        label: str,
        status: str,
        place_id: str | None,
        carried_by: str | None,
        confidence: float,
        origin: str,
        at: dt.datetime,
        seen_by: str,
        clip_path: str,
        offset_s: float,
        object_id: str | None = None,
    ) -> dict:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; expected one of {sorted(STATUSES)}")
        object_id = object_id or self.next_object_id()
        doc = {
            "_id": object_id,
            "label": label,
            "status": status,
            "place_id": place_id,
            "carried_by": carried_by,
            "last_confirmed_at": at,
            "last_confirmed_by": seen_by,
            "belief_since": at,
            "confidence": float(confidence),
            "origin": origin,
            "last_clip": {"clip_path": clip_path, "offset_s": float(offset_s)},
            "merges": [],
        }
        self._objects.insert_one(doc)
        log.info("new object %s (%s) at %s", object_id, label, place_id)
        return doc

    def get(self, object_id: str) -> dict | None:
        return self._objects.find_one({"_id": object_id})

    def all_objects(self) -> list[dict]:
        return list(self._objects.find({}))

    def find_by_label(self, query: str) -> list[dict]:
        """Objects whose label matches, for the agent's `find_object` tool.

        Substring, case-insensitive: a person asking for "keys" should find "car keys", and
        the agent is read-only, so a generous match costs nothing but a longer list.
        """
        if not query or not query.strip():
            return []
        pattern = re.escape(query.strip())
        return list(self._objects.find({"label": {"$regex": pattern}}))

    def at_place(self, place_id: str) -> list[dict]:
        return list(self._objects.find({
            "place_id": place_id,
            "status": {"$in": sorted(LOCATED_STATUSES)},
        }))

    def set_belief(self, object_id: str, changes: dict[str, Any]) -> dict:
        """Update one object's belief, then denormalize onto its fingerprints.

        `place_id` and `last_seen_at` are copied onto every fingerprint because the vector
        index filters on them — a stale copy would make zone-1 search look in the wrong place.
        """
        unknown = set(changes) - {
            "status", "place_id", "carried_by", "last_confirmed_at", "last_confirmed_by",
            "belief_since", "confidence", "origin", "last_clip", "label",
        }
        if unknown:
            raise ValueError(f"refusing to set unknown belief fields: {sorted(unknown)}")
        if "status" in changes and changes["status"] not in STATUSES:
            raise ValueError(f"unknown status {changes['status']!r}")

        self._objects.update_one({"_id": object_id}, {"$set": changes})
        if "place_id" in changes or "last_confirmed_at" in changes:
            self._denormalize(object_id, changes)
        return self.get(object_id)

    def _denormalize(self, object_id: str, changes: dict) -> None:
        fields = {}
        if "place_id" in changes:
            fields["place_id"] = changes["place_id"]
        if "last_confirmed_at" in changes:
            fields["last_seen_at"] = changes["last_confirmed_at"]
        if not fields:
            return
        for doc in self._fingerprints.find({"object_id": object_id}):
            self._fingerprints.update_one({"_id": doc["_id"]}, {"$set": fields})

    def record_merge(self, object_id: str, note: dict) -> None:
        """Append to `merges`. Never rewrites history — the trail of what was joined into an
        object is as much evidence as the diary is."""
        self._objects.update_one({"_id": object_id}, {"$push": {"merges": note}})

    # -------------------------------------------------------------------- fingerprints

    def fingerprints_for(self, object_id: str) -> list[dict]:
        return list(self._fingerprints.find({"object_id": object_id}))

    def vectors_for(self, object_id: str) -> list[np.ndarray]:
        return [np.asarray(doc["vec"], dtype=np.float32) for doc in self.fingerprints_for(object_id)]

    def add_fingerprint(
        self,
        object_id: str,
        vec: Sequence[float],
        *,
        place_id: str | None,
        last_seen_at: dt.datetime,
        glasses: str,
    ) -> bool:
        """Store a fingerprint if it earns a slot. Returns whether it was stored.

        Refused in three cases, each for a different reason:
        - degenerate vector: nothing to compare against, and Atlas rejects it as a query
        - too similar to an existing one: the 12 slots are for varied angles
        - already at the cap: a bounded set keeps the vector index small on a sandbox tier
        """
        if embed.is_degenerate(vec):
            log.debug("fingerprint refused for %s: degenerate vector", object_id)
            return False
        existing = self.vectors_for(object_id)
        if len(existing) >= config.MAX_FINGERPRINTS_PER_OBJECT:
            return False
        if existing and not embed.differs_enough(vec, existing):
            return False
        self._fingerprints.insert_one({
            "object_id": object_id,
            "vec": [float(x) for x in embed.l2_normalize(vec)],
            "place_id": place_id,
            "last_seen_at": last_seen_at,
            "glasses": glasses,
        })
        return True

    def fingerprint_count(self, object_id: str) -> int:
        return self._fingerprints.count_documents({"object_id": object_id})
