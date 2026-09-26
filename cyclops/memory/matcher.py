"""Three-zone identity matching.

    zone 1  filtered $vectorSearch within the current place  -> join if cosine >= T_PLACE
    zone 2  unfiltered $vectorSearch across everything       -> join if cosine >= T_HIGH
    zone 3  otherwise                                        -> NEW OBJECT

Zone 3 is Core Principle 5: never merge when unsure. A duplicate object is fixable later; a
wrong merge quietly corrupts every answer after it and is not. Claude-as-referee for the
unclear zone was already tried and dropped (see MASTERPLAN's superseded decisions) — do not
reintroduce it. M6 resolves look-alikes by history instead.

Thresholds are cosine similarity. Atlas normalizes its `vectorSearchScore` for cosine indexes,
so the raw score is NOT comparable to T_PLACE/T_HIGH: this module recomputes exact cosine from
the returned vectors instead of trusting a normalization formula it does not control.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
from typing import Callable, Sequence

import numpy as np

from cyclops import config
from cyclops.memory import db as dbmod
from cyclops.perception import embed

log = logging.getLogger(__name__)

ZONE_PLACE = "place"
ZONE_GLOBAL = "global"
ZONE_NEW = "new"


@dataclasses.dataclass(frozen=True)
class MatchResult:
    """`object_id is None` means "create a new object" — the caller must not reinterpret it."""

    object_id: str | None
    zone: str
    similarity: float
    reason: str
    degenerate: bool = False

    @property
    def is_new(self) -> bool:
        return self.object_id is None


class Matcher:
    def __init__(self, database, vector_search: Callable[..., list[dict]] | None = None):
        self._db = database
        self._search = vector_search or self._atlas_search
        self.stats: collections.Counter[str] = collections.Counter()
        self._degenerate_logged = 0

    # ------------------------------------------------------------------ vector search

    def _atlas_search(self, vec: Sequence[float], place_id: str | None, limit: int) -> list[dict]:
        stage: dict = {
            "index": dbmod.VECTOR_INDEX_NAME,
            "path": "vec",
            "queryVector": [float(x) for x in vec],
            "numCandidates": max(100, limit * 20),
            "limit": limit,
        }
        if place_id is not None:
            stage["filter"] = {"place_id": {"$eq": place_id}}
        pipeline = [
            {"$vectorSearch": stage},
            # `vec` is projected so cosine is recomputed locally; `score` is kept only for
            # logging, since its normalization is Atlas's business, not ours.
            {"$project": {"_id": 0, "object_id": 1, "vec": 1,
                          "score": {"$meta": "vectorSearchScore"}}},
        ]
        return list(self._db[dbmod.FINGERPRINTS].aggregate(pipeline))

    def _best(self, vec, place_id: str | None, limit: int = 10) -> tuple[str | None, float]:
        """Best (object_id, exact cosine) from one search, or (None, 0.0)."""
        best_id, best_sim = None, 0.0
        for hit in self._search(vec, place_id, limit) or []:
            candidate = hit.get("object_id")
            if not candidate:
                continue
            stored = hit.get("vec")
            sim = embed.cosine(vec, stored) if stored is not None else float(hit.get("score", 0.0))
            if sim > best_sim:
                best_id, best_sim = candidate, sim
        return best_id, best_sim

    # -------------------------------------------------------------------------- matching

    def match(self, vec: Sequence[float] | np.ndarray, place_id: str | None) -> MatchResult:
        """Decide which object a track's fingerprint belongs to."""
        if embed.is_degenerate(vec):
            return self._degenerate()

        self.stats["searched"] += 1

        if place_id is not None:
            candidate, sim = self._best(vec, place_id)
            if candidate and sim >= config.T_PLACE:
                self.stats["zone_place"] += 1
                return MatchResult(candidate, ZONE_PLACE, sim,
                                   f"cosine {sim:.3f} >= T_PLACE {config.T_PLACE} at {place_id}")

        candidate, sim = self._best(vec, None)
        if candidate and sim >= config.T_HIGH:
            self.stats["zone_global"] += 1
            return MatchResult(candidate, ZONE_GLOBAL, sim,
                               f"cosine {sim:.3f} >= T_HIGH {config.T_HIGH} (unfiltered)")

        self.stats["new"] += 1
        return MatchResult(None, ZONE_NEW, sim,
                           f"best cosine {sim:.3f} cleared neither T_PLACE {config.T_PLACE} "
                           f"nor T_HIGH {config.T_HIGH} — new object, per never-merge-when-unsure")

    def _degenerate(self) -> MatchResult:
        """A zero-norm embedding: the encoder extracted nothing.

        Treated as the strongest possible "unsure" — skip the search entirely (Atlas refuses a
        zero query vector) and create a new object. Never a hard failure: this happens a handful
        of times across hours of real footage (motion blur, an occluding hand, a bad crop at a
        clip boundary), and per Core Principle 6 nothing freezes the pipeline over one degraded
        input.

        But it is never silent either. If this fires constantly, something upstream is broken —
        the persistence filter, or the crop size — and the symptom would otherwise only surface
        much later as an inflated object count in the contact list.
        """
        self.stats["degenerate"] += 1
        # A degenerate crop still produces a new object, so it counts toward `new` too —
        # otherwise the summary understates exactly the contact-list inflation it exists to warn
        # about.
        self.stats["new"] += 1
        count = self.stats["degenerate"]
        if self._degenerate_logged < 5 or count % 25 == 0:
            self._degenerate_logged += 1
            log.warning(
                "degenerate (zero-norm) crop embedding — skipping vector search, creating a new "
                "object (occurrence %d). If this number climbs, check the crop size and the "
                "persistence filter.", count,
            )
        return MatchResult(None, ZONE_NEW, 0.0,
                           "degenerate embedding: nothing to compare, new object", degenerate=True)

    # ------------------------------------------------------------------------- reporting

    def summary(self) -> str:
        """One line for the end of a processing run — visible, not buried in debug logs."""
        s = self.stats
        parts = [
            f"matched-in-place {s['zone_place']}",
            f"matched-globally {s['zone_global']}",
            f"new objects {s['new']} ({s['new'] - s['degenerate']} unclear, "
            f"{s['degenerate']} degenerate)",
        ]
        line = "identity matching: " + ", ".join(parts)
        attempts = s["searched"] + s["degenerate"]
        if s["degenerate"] and s["degenerate"] > max(3, 0.1 * attempts):
            line += (f"  <-- WARNING: {s['degenerate']} of {attempts} crops embedded to a zero "
                     f"vector; check crop extraction before trusting the object count")
        return line
