"""The belief state machine, and the only place a belief is allowed to change.

Ordering is the whole point of this module. Core Principle 2: the diary entry is written
first, and the belief is updated only after that write returns. If the diary write fails, the
belief does not move — so a belief can never exist that nothing in the record explains.

Core Principle 4: a newer observed sighting always beats an inferred location. That is not
just applied when sightings arrive; `record_match()` also refuses to move a belief backwards
onto an inference older than the last sighting.

Confidence fade changes wording, never stored status, until it crosses FADE_FLOOR — so fade is
computed at read time, not written by a timer. A belief only ever *changes* on evidence.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import math
from typing import Any

from cyclops import config
from cyclops.memory.diary import Diary
from cyclops.memory.registry import Registry

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- pure fade math

def fade_confidence(base_confidence: float, elapsed_s: float, half_life_s: float | None = None) -> float:
    """conf = base * exp(-ln2 * Δt / FADE_HALF_LIFE_S).

    Scaled from the belief's own confidence rather than always from 1.0: an inferred belief
    starts at the retrieval score, and fading it as though it had been a certain sighting would
    overstate it for the whole first half-life.
    """
    half_life = half_life_s if half_life_s is not None else config.FADE_HALF_LIFE_S
    if elapsed_s <= 0 or half_life <= 0:
        return float(base_confidence)
    return float(base_confidence) * math.exp(-math.log(2.0) * float(elapsed_s) / float(half_life))


def is_faded(confidence: float) -> bool:
    return float(confidence) < config.FADE_FLOOR


@dataclasses.dataclass(frozen=True)
class Belief:
    """What a read of the contact list says right now, with fade applied."""

    object_id: str
    label: str
    status: str
    stored_status: str
    place_id: str | None
    carried_by: str | None
    confidence: float
    origin: str
    last_confirmed_at: dt.datetime | None
    last_confirmed_by: str | None
    clip_path: str | None
    offset_s: float | None

    @property
    def faded(self) -> bool:
        return self.status == "faded"


class BeliefService:
    """Writes diary entries and moves beliefs, in that order, always."""

    def __init__(self, diary: Diary, registry: Registry):
        self._diary = diary
        self._registry = registry

    # ------------------------------------------------------------------- read (with fade)

    def current(self, object_id: str, now: dt.datetime | None = None) -> Belief | None:
        doc = self._registry.get(object_id)
        if doc is None:
            return None
        now = now or dt.datetime.now(dt.timezone.utc)
        stored_status = doc.get("status", "believed")
        confidence = float(doc.get("confidence", 0.0))
        status = stored_status

        # A confirmed sighting or a carried object is not subject to fade: it is either being
        # held right now, or it was last seen in place and nothing has contradicted that.
        if stored_status == "believed":
            since = doc.get("belief_since") or doc.get("last_confirmed_at")
            if since is not None:
                elapsed = (now - _aware(since)).total_seconds()
                confidence = fade_confidence(confidence, elapsed)
                if is_faded(confidence):
                    status = "faded"

        clip = doc.get("last_clip") or {}
        return Belief(
            object_id=object_id,
            label=doc.get("label", ""),
            status=status,
            stored_status=stored_status,
            place_id=doc.get("place_id"),
            carried_by=doc.get("carried_by"),
            confidence=round(confidence, 4),
            origin=doc.get("origin", "observed"),
            last_confirmed_at=_aware(doc.get("last_confirmed_at")) if doc.get("last_confirmed_at") else None,
            last_confirmed_by=doc.get("last_confirmed_by"),
            clip_path=clip.get("clip_path"),
            offset_s=clip.get("offset_s"),
        )

    # ------------------------------------------------------------------------ transitions

    def seen_at_place(
        self, *, object_id: str, place_id: str, t: dt.datetime, session: str, glasses: str,
        clip_path: str, offset_s: float, track_id: int | None = None, first_sighting: bool = False,
    ) -> dict:
        """Seen at a place -> confirmed, confidence 1.0."""
        entry = self._diary.append(
            t=t, session=session, glasses=glasses, object_id=object_id,
            event="appeared" if first_sighting else "placed",
            origin="observed", confidence=1.0, place_id=place_id,
            clip_path=clip_path, offset_s=offset_s, track_id=track_id,
        )
        self._registry.set_belief(object_id, {
            "status": "confirmed", "place_id": place_id, "carried_by": None,
            "confidence": 1.0, "origin": "observed",
            "last_confirmed_at": t, "last_confirmed_by": glasses, "belief_since": t,
            "last_clip": {"clip_path": clip_path, "offset_s": float(offset_s)},
        })
        return entry

    def picked_up(
        self, *, object_id: str, carried_by: str, t: dt.datetime, session: str, glasses: str,
        clip_path: str, offset_s: float, track_id: int | None = None,
    ) -> dict:
        """Picked up -> carried by the wearer, until seen placed."""
        entry = self._diary.append(
            t=t, session=session, glasses=glasses, object_id=object_id, event="picked_up",
            origin="observed", confidence=1.0, carried_by=carried_by,
            clip_path=clip_path, offset_s=offset_s, track_id=track_id,
        )
        self._registry.set_belief(object_id, {
            "status": "carried", "carried_by": carried_by, "place_id": None,
            "confidence": 1.0, "origin": "observed",
            "last_confirmed_at": t, "last_confirmed_by": glasses, "belief_since": t,
            "last_clip": {"clip_path": clip_path, "offset_s": float(offset_s)},
        })
        return entry

    def left_view(
        self, *, object_id: str, place_id: str | None, t: dt.datetime, session: str, glasses: str,
        clip_path: str, offset_s: float, track_id: int | None = None,
    ) -> dict:
        """Left view -> believed at its last place, from which confidence starts fading.

        A carried object that leaves view stays `carried`: the last thing actually observed is
        that someone had it, and "believed on the counter" would be a claim nothing saw.
        """
        entry = self._diary.append(
            t=t, session=session, glasses=glasses, object_id=object_id, event="left_view",
            origin="observed", confidence=1.0, place_id=place_id,
            clip_path=clip_path, offset_s=offset_s, track_id=track_id,
        )
        current = self._registry.get(object_id) or {}
        if current.get("status") == "carried":
            return entry
        self._registry.set_belief(object_id, {
            "status": "believed", "place_id": place_id or current.get("place_id"),
            "confidence": 1.0, "origin": "observed", "belief_since": t,
            "last_clip": {"clip_path": clip_path, "offset_s": float(offset_s)},
        })
        return entry

    def went_missing(
        self, *, object_id: str, place_id: str, t: dt.datetime, session: str, glasses: str,
        clip_path: str, offset_s: float,
    ) -> dict:
        """A labelled place was in view long enough and the object was not there.

        Records the absence and leaves the belief alone. Absence is not a location: what
        happens next is a retrieval attempt, and if that finds nothing the honest answer is
        still the last confirmed sighting.
        """
        return self._diary.append(
            t=t, session=session, glasses=glasses, object_id=object_id, event="missing",
            origin="observed", confidence=1.0, place_id=place_id,
            clip_path=clip_path, offset_s=offset_s,
        )

    def record_match(
        self, *, object_id: str, place_id: str, similarity: float, t: dt.datetime, session: str,
        glasses: str, query_clip: dict, matched_clip: dict,
    ) -> tuple[dict, bool]:
        """Write a `matched` event and, if it is still the newest evidence, believe it.

        Returns (diary entry, whether the belief moved). The diary entry is always written —
        the retrieval happened, and the record says so — but Core Principle 4 means an
        inference never overrides a sighting that is newer than it.
        """
        entry = self._diary.append(
            t=t, session=session, glasses=glasses, object_id=object_id, event="matched",
            origin="inferred", confidence=float(similarity), place_id=place_id,
            clip_path=matched_clip.get("clip_path", ""), offset_s=matched_clip.get("offset_s", 0.0),
            evidence={"query_clip": query_clip, "matched_clip": matched_clip},
        )

        newest_sighting = self._diary.last_confirmed_sighting(object_id)
        if newest_sighting is not None and _aware(newest_sighting["t"]) > _aware(t):
            log.info(
                "match for %s not applied: an observed sighting at %s is newer than the "
                "inference at %s (sightings beat inferences)",
                object_id, _aware(newest_sighting["t"]).isoformat(), _aware(t).isoformat(),
            )
            return entry, False

        self._registry.set_belief(object_id, {
            "status": "believed", "place_id": place_id, "carried_by": None,
            "confidence": float(similarity), "origin": "inferred", "belief_since": t,
            "last_clip": {"clip_path": matched_clip.get("clip_path", ""),
                          "offset_s": float(matched_clip.get("offset_s", 0.0))},
        })
        return entry, True

    def stated_by_human(
        self, *, object_id: str, place_id: str, t: dt.datetime, session: str, glasses: str,
        clip_path: str, offset_s: float,
    ) -> dict:
        """A human label. Traceable like anything else (Core Principle 3), origin='stated'."""
        entry = self._diary.append(
            t=t, session=session, glasses=glasses, object_id=object_id, event="placed",
            origin="stated", confidence=1.0, place_id=place_id,
            clip_path=clip_path, offset_s=offset_s,
        )
        self._registry.set_belief(object_id, {
            "status": "confirmed", "place_id": place_id, "carried_by": None,
            "confidence": 1.0, "origin": "stated",
            "last_confirmed_at": t, "last_confirmed_by": glasses, "belief_since": t,
            "last_clip": {"clip_path": clip_path, "offset_s": float(offset_s)},
        })
        return entry


def _aware(value: Any) -> dt.datetime:
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)
