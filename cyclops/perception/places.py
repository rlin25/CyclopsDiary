"""Whole-scene place recognition.

A place is a backdrop that a human labelled by hand before recording, enrolled as a set of
whole-frame fingerprints from both phones. Recognition is max cosine against those frames.

This replaced furniture-box overlap (see MASTERPLAN's superseded decisions): overlap depended
on the furniture also being detected correctly in every frame. Do not reintroduce it.

Deliberately plain code with no index: the `places` collection is small enough to hold in
memory, and the sandbox's search-index budget is spent on object fingerprints.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Iterable, Sequence

import numpy as np

from cyclops import config
from cyclops.perception import embed

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PlaceGuess:
    """`place_id` is None when nothing cleared PLACE_MIN_SIM. `sim` is still the best score
    seen, so near-misses are visible when tuning rather than collapsed to zero."""

    place_id: str | None
    sim: float
    runner_up: str | None = None
    runner_up_sim: float = 0.0

    @property
    def recognized(self) -> bool:
        return self.place_id is not None


@dataclasses.dataclass(frozen=True)
class PlaceRun:
    """A stretch of consecutive samples that agreed on one place."""

    place_id: str | None
    from_s: float
    to_s: float
    sim_mean: float

    def as_dict(self) -> dict:
        return {"from_s": self.from_s, "to_s": self.to_s,
                "place_id": self.place_id, "sim_mean": round(self.sim_mean, 4)}


class PlaceIndex:
    """Enrolled places held in memory, matched by max cosine over their frames.

    Max, not mean: the same backdrop filmed from two phones at two angles is genuinely two
    clusters, and averaging them produces a vector that matches neither well.
    """

    def __init__(self, vectors_by_place: dict[str, list[np.ndarray]]):
        self._by_place = {
            place_id: [embed.l2_normalize(v) for v in vecs
                       if not embed.is_degenerate(v)]
            for place_id, vecs in vectors_by_place.items()
        }
        empty = [p for p, v in self._by_place.items() if not v]
        if empty:
            log.warning("places with no usable fingerprints, they can never be recognized: %s",
                        sorted(empty))
        self._by_place = {p: v for p, v in self._by_place.items() if v}
        self._stacked = {p: np.stack(v) for p, v in self._by_place.items()}

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_documents(cls, documents: Iterable[dict]) -> "PlaceIndex":
        """Build from `places` collection documents: {"place_id", "vecs": [[...], ...]}."""
        by_place: dict[str, list[np.ndarray]] = {}
        for doc in documents:
            place_id = doc.get("place_id") or doc.get("_id")
            if not place_id:
                continue
            vecs = doc.get("vecs") or ([doc["vec"]] if doc.get("vec") is not None else [])
            by_place.setdefault(str(place_id), []).extend(np.asarray(v, dtype=np.float32) for v in vecs)
        return cls(by_place)

    @classmethod
    def from_db(cls, db) -> "PlaceIndex":
        from cyclops.memory import db as dbmod
        return cls.from_documents(db[dbmod.PLACES].find({}))

    # -------------------------------------------------------------------- recognition

    @property
    def place_ids(self) -> list[str]:
        return sorted(self._stacked)

    def __len__(self) -> int:
        return len(self._stacked)

    def recognize(self, frame_vec: Sequence[float] | np.ndarray) -> PlaceGuess:
        """Best labelled place for one whole-frame fingerprint.

        A degenerate frame returns "no place" rather than an arbitrary one.
        """
        if embed.is_degenerate(frame_vec) or not self._stacked:
            return PlaceGuess(None, 0.0)
        query = embed.l2_normalize(frame_vec)
        scored = sorted(
            ((float(np.max(frames @ query)), place_id) for place_id, frames in self._stacked.items()),
            reverse=True,
        )
        best_sim, best_place = scored[0]
        runner_up, runner_up_sim = (scored[1][1], scored[1][0]) if len(scored) > 1 else (None, 0.0)
        if best_sim < config.PLACE_MIN_SIM:
            return PlaceGuess(None, best_sim, runner_up, runner_up_sim)
        return PlaceGuess(best_place, best_sim, runner_up, runner_up_sim)


# ------------------------------------------------------------------------ run collapsing

def collapse_runs(samples: Sequence[tuple[float, PlaceGuess]], sample_period_s: float | None = None) -> list[PlaceRun]:
    """Turn per-frame guesses into the `places` runs the §4 cache stores.

    Runs of "no recognized place" are dropped: the cache states where something was seen, and
    an explicit run of nothing would read as evidence rather than the absence of it.
    """
    period = sample_period_s if sample_period_s is not None else (1.0 / config.SAMPLE_FPS)
    runs: list[PlaceRun] = []
    current_place: str | None = None
    start = 0.0
    last = 0.0
    sims: list[float] = []

    def flush() -> None:
        if current_place is not None and sims:
            runs.append(PlaceRun(current_place, round(start, 3), round(last + period, 3),
                                 float(np.mean(sims))))

    for offset_s, guess in samples:
        if guess.place_id != current_place:
            flush()
            current_place, start, sims = guess.place_id, offset_s, []
        if guess.place_id is not None:
            sims.append(guess.sim)
        last = offset_s
    flush()
    return runs


def visible_for(runs: Sequence[PlaceRun], place_id: str, min_seconds: float) -> bool:
    """Whether a place was continuously recognized for at least `min_seconds`.

    This is the gate on the `missing` event: a place has to be genuinely in view before the
    absence of an object there means anything.
    """
    return any(run.place_id == place_id and (run.to_s - run.from_s) >= min_seconds for run in runs)
