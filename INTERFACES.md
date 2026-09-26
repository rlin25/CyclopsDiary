# CyclopsDiary — Interface Contracts

Four frozen seams. Everything else is internal to whichever module owns it and can change
freely without coordination. **These four should not change without both people agreeing**,
because something on the other side of each one is being built in parallel right now.

---

## 1. `find_similar_moment()` — Track B owns this, Track A calls it

Wraps ElideDB. **This is a retrieval function, not a reasoning or prediction function.**
It has no concept of objects, places, or labels — it only ranks clips by motion
similarity. Do not add a `rationale`, `predicted_place`, or confidence-with-explanation
field; those imply reasoning that does not happen anywhere in this path.

```python
def find_similar_moment(
    clip_s3_key: str,
    t_start_s: float,
    t_end_s: float,
    top_k: int = 5,
) -> list[dict]:
    """
    Query ElideDB with a clip window (the last-confirmed-visible-to-gone span for an
    object) and return the top_k most similar moments found anywhere in the ingested
    footage (both phones), ranked by motion similarity.

    Returns: [{"clip_s3_key": str, "offset_s": float, "similarity": float}, ...]
    Empty list if nothing clears ELIDE_MIN_SIM or on any failure — never raises for a
    "no good match" case. Raises only on a hard failure (store unreachable, etc.), which
    the caller catches per the "degrade honestly" rule in IMPLEMENTATION_STRATEGY.md.

    Constraints (from ElideDB itself — do not violate):
    - t_end_s - t_start_s must be >= 4.0 seconds (ElideDB's minimum clip length). If the
      caller's window is shorter, pad it rather than calling with < 4s.
    - Precision degrades past ~top 20 results; top_k should stay small (5 is the default
      for a reason — do not raise it to "get more options").
    - Ingest (adding footage to the ElideDB store) is a separate, one-time step per
      session's footage. It must have already happened before this function is called.
      Track B owns the ingest script; Track A does not need to know its internals, only
      that ingest has run before querying.

    The caller (Track A, in the `missing` event handler) is responsible for:
    - Resolving offset_s to a place, via the place recognizer — this function does NOT
      return a place.
    - Deciding what counts as "good enough" (ELIDE_MIN_SIM, from config) — this function
      returns raw similarity scores, unfiltered by any threshold, so the caller can log
      and inspect near-misses too.
    """
```

Track B builds and tests this against raw clips in isolation before it is ever wired into
the `missing`-event handler. It should be demoable standing alone: "here's a clip of keys
being picked up, here are the 5 most similar moments in the store."

---

## 2. Diary document shape — Track A owns this, everyone reads it

See `IMPLEMENTATION_STRATEGY.md` for the full field list and the belief state machine that
produces these. Restated here because M4/M5/eval code and the agent both depend on this
shape being stable:

```json
{
  "t": "<ISODate, required>",
  "meta": {"session": "<str>", "glasses": "A|B"},
  "object_id": "<str, required>",
  "event": "appeared|left_view|picked_up|placed|missing|matched",
  "place_id": "<str or null>",
  "carried_by": "<A|B or null>",
  "origin": "observed|inferred|stated",
  "confidence": "<float 0-1>",
  "clip": {"s3_key": "<str>", "offset_s": "<float>"},
  "track_id": "<int, optional>",
  "evidence": "<object, present only on 'matched' events — see IMPLEMENTATION_STRATEGY.md>"
}
```
No field is ever renamed without updating this file and telling the other track.

---

## 3. Agent tool signatures — Track A owns this, the agent calls these, humans see the output

Five read-only tools. The agent can ONLY report what these return — see Core Principle 7
in `MASTERPLAN.md`. Empty/null result → the agent says "I don't know," never guesses.

```python
def find_object(query: str) -> list[dict]:
    """Objects whose label/name matches. Returns [{"object_id", "label"}, ...]."""

def get_belief(object_id: str) -> dict:
    """
    Current belief + evidence for one object.
    Returns: {"status", "place_id", "carried_by", "last_confirmed_at",
              "last_confirmed_by", "confidence", "origin", "clip_url" (fresh presigned,
              generated at call time), "offset_s", "evidence": {diary entries + the
              MongoDB query used, for the evidence view}}
    """

def what_is_at(place_id: str) -> list[dict]:
    """Objects currently believed to be at a place. Returns [{"object_id", "label",
    "status"}, ...]."""

def timeline(start: str, end: str, object_id: str | None = None) -> list[dict]:
    """MongoDB aggregation over `diary` between two ISO timestamps, optionally filtered
    to one object. Returns raw diary entries in time order."""

def find_similar_past_moment(clip_s3_key: str, t_start_s: float, t_end_s: float) -> list[dict]:
    """
    Agent-facing wrapper around find_similar_moment() (seam #1), for direct questions
    like "has this happened before?" Same return shape. This is the ONLY place a human
    can trigger an ElideDB query outside the automatic `missing`-event handler.
    """
```

Answer wording (status × origin → phrasing) is specified in `subplans/M1_lean_core.md`,
not here — that's presentation logic, not an interface.

---

## 4. Composite/eval truth format — Track A owns this, M4/M5 depend on it

Frozen now even though M4/M5 aren't built until hour 2.5–3, because M1's output format
(the per-scene cache) has to match this from the start, or the eval code built later
can't consume it and there's no time left to reconcile the two.

**Hand-written scene label** (`data/labels/<session>/<scene>.yaml`):
```yaml
scene: s1_handoff
glasses: {A: clips/session1/A/IMG_0012.MOV, B: clips/session1/B/IMG_0450.MOV}
objects:
  keys: [{t: 0.0, place: kitchen_counter}, {t: 42.0, carried_by: B}, {t: 55.0, place: desk_drawer}]
  mug:  [{t: 0.0, place: hall_shelf}]
```
`t` = seconds from scene start, on the corrected shared timeline (post sync-shot offset).
Each object's list is a timeline of state changes — `place` or `carried_by`, never both
in one entry. Written by hand, once, right after filming each scene.

**Per-scene processing cache** (`data/cache/<session>/<scene>.jsonl`), written by M1's
`process_session.py` as a required side effect, not an afterthought:
one JSON object per line, each line a surviving track's full record for that scene —
detections, fingerprint, recognized places over time, and every diary event it produced.
This is what M4's composite builder reads to construct composites and their derived truth
without ever re-running the vision models.

If either shape needs a field added mid-build, add it — do not remove or rename an
existing field, since the other track may already be depending on it.
