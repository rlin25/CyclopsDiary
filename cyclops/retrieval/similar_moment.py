"""`find_similar_moment()` — seam #1 from docs/INTERFACES.md.

**This is the M1 stub.** It returns `[]`. Track B replaces the body in M2 with the real ElideDB
query; the signature here is already the real one, so that replacement is a drop-in and nothing
on this side has to change.

It is a retrieval function, not a reasoning one. It has no concept of objects, places, or
labels — it ranks clips by motion similarity. Resolving a result to a place is the caller's job,
via our own place recognizer, and deciding what counts as good enough is the caller's job too.
"""

from __future__ import annotations

import logging

from cyclops import config
from cyclops.breaker import CircuitBreaker, ServiceUnavailable

log = logging.getLogger(__name__)

#: One breaker for the retrieval service, shared by the automatic missing-event handler and the
#: agent's `find_similar_past_moment` tool — three failures from either source is still a
#: service that is down.
breaker = CircuitBreaker("elidedb")


def find_similar_moment(
    clip_path: str,
    t_start_s: float,
    t_end_s: float,
    top_k: int = 5,
) -> list[dict]:
    """
    Query ElideDB with a clip window (the last-confirmed-visible-to-gone span for an
    object) and return the top_k most similar moments found anywhere in the ingested
    footage (both phones), ranked by motion similarity.

    Returns: [{"clip_path": str, "offset_s": float, "similarity": float}, ...]
    Empty list if nothing is found — never raises for a "no good match" case. Raises only on a
    hard failure (store unreachable, etc.), which the caller catches per the "degrade honestly"
    rule in docs/IMPLEMENTATION_STRATEGY.md.

    Constraints (from ElideDB itself — do not violate):
    - t_end_s - t_start_s must be >= 4.0 seconds. Callers pad with `pad_window()` rather than
      calling with a shorter window.
    - Precision degrades past ~top 20 results; top_k stays small.
    - Ingest must already have happened for the session's footage.

    M1 STUB: returns []. The `missing` handler is wired to this exact signature so M2 is a
    drop-in replacement.
    """
    window = float(t_end_s) - float(t_start_s)
    if window < config.ELIDE_MIN_WINDOW_S:
        # Loud, because silently querying with a short window would produce quietly worse
        # results in M2 rather than an error anyone would notice.
        raise ValueError(
            f"window {window:.2f}s is shorter than ElideDB's minimum "
            f"{config.ELIDE_MIN_WINDOW_S}s — pad it with pad_window() before calling"
        )
    if top_k > 20:
        log.warning("top_k=%d is past the point where precision degrades; capping at 20", top_k)
        top_k = 20

    log.info("find_similar_moment stub: %s [%.2f, %.2f] top_k=%d -> [] (real in M2)",
             clip_path, t_start_s, t_end_s, top_k)
    return []


def pad_window(t_start_s: float, t_end_s: float, min_length_s: float | None = None) -> tuple[float, float]:
    """Widen a window to ElideDB's minimum length, centred on the original span.

    Clamped at zero so padding never produces a negative start; the extra time is taken from
    the end in that case.
    """
    minimum = min_length_s if min_length_s is not None else config.ELIDE_MIN_WINDOW_S
    start, end = float(min(t_start_s, t_end_s)), float(max(t_start_s, t_end_s))
    shortfall = minimum - (end - start)
    if shortfall <= 0:
        return start, end
    start -= shortfall / 2.0
    end += shortfall / 2.0
    if start < 0.0:
        end += -start
        start = 0.0
    return round(start, 3), round(end, 3)


def best_match(results: list[dict], min_similarity: float | None = None) -> dict | None:
    """Top result that clears the threshold, or None.

    The threshold lives here, on the caller's side, because seam #1 returns raw scores so that
    near-misses stay inspectable during tuning.
    """
    floor = min_similarity if min_similarity is not None else config.ELIDE_MIN_SIM
    clearing = [r for r in results if float(r.get("similarity", 0.0)) >= floor]
    if not clearing:
        if results:
            best = max(float(r.get("similarity", 0.0)) for r in results)
            log.info("no retrieval result cleared ELIDE_MIN_SIM %.2f (best was %.3f)", floor, best)
        return None
    return max(clearing, key=lambda r: float(r["similarity"]))


def query(clip_path: str, t_start_s: float, t_end_s: float, top_k: int | None = None) -> list[dict]:
    """Breaker-wrapped call for the missing-event handler and the agent tool.

    Returns [] both when nothing matched and when the service could not be reached — the
    caller distinguishes them by asking `breaker.is_open`, and says plainly that a search
    could not run rather than implying nothing was found.
    """
    start, end = pad_window(t_start_s, t_end_s)
    try:
        return breaker.call(find_similar_moment, clip_path, start, end,
                            top_k or config.ELIDE_TOP_K)
    except ServiceUnavailable as exc:
        log.warning("retrieval skipped: %s", exc)
        return []
