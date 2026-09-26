# CyclopsDiary — Implementation Strategy

Cross-cutting rules that apply regardless of which milestone or module you're building.
Read this in full if you are working on Track A (M1, M3–M6). If you are working on Track
B (M2 / ElideDB only), you need only the "Time", "Stack" and "MongoDB" sections plus
`INTERFACES.md` — see `STARTUP.md` for the branch.

## Stack (fixed — do not substitute without asking)
- Python 3.11, FastAPI, a single static HTML page (no frontend framework, no Streamlit)
- Detection + tracking: Ultralytics **YOLOE prompt-free** (`yoloe-11s-seg-pf.pt`),
  `model.track(..., tracker="botsort.yaml")`
- Identity fingerprints: **DINOv2 small** via HF transformers (`facebook/dinov2-small`,
  CLS token, 384 dims, L2-normalized) — used for BOTH object crops and whole-frame place
  recognition
- Memory: **MongoDB Atlas** (hackathon sandbox cluster) via `pymongo`
- Video storage: **local disk** under `data/clips/`, served by FastAPI's static file
  mount at `/clips/`. `clip_url` is built by joining that mount's base URL with the
  diary entry's `clip_path` — no external service, no credentials, no expiry to manage
- Agent: tool-using LLM via OpenRouter (OpenAI-compatible SDK), model from env
  `AGENT_MODEL`.
- Motion retrieval: **ElideDB**, wrapping Cosmos 3's encoders — retrieval only, see
  `INTERFACES.md`
- Everything vision-related runs on the laptop. Process at **5 fps** (sample frames).

## Environment (`.env` — never commit)
```
MONGODB_URI=            # Atlas sandbox cluster
MONGODB_DB=cyclops      # load test uses cyclops_loadtest, a separate database
OPENROUTER_API_KEY=     AGENT_MODEL=
ELIDEDB_STORE_PATH=     ELIDE_MIN_SIM=0.35
```

## Time
Absolute time of a frame = clip `creation_time` (read via ffprobe) + that phone's session
offset (from the sync shot, in `data/sessions.yaml`) + seconds into the clip.
**Never** use processing time or upload time — clips are transferred by cable/AirDrop
after recording, not streamed live.

## MongoDB — collection design
| Collection | Type | Purpose |
|---|---|---|
| `diary` | **time series** (`timeField: t`, `metaField: meta`) | Append-only record. No updates, no transactions, no search index — time-series collections don't support them. |
| `objects` | regular | Contact list: one document per discovered object + its current belief |
| `fingerprints` | regular + **vector index** | Object-crop fingerprints, max 12/object, diverse angles. The only vector index — sandbox may be M0 (max 3 search/vector indexes total). |
| `places` | regular | Labeled places + whole-frame fingerprints, compared in plain code, **no index** |
| `runs` | regular | Tuning, held-out test, and load-test results |

Why split like this: the diary can't hold fingerprints (time-series limitation), and the
split mirrors the record/belief distinction directly — `diary` is the permanent record,
`objects` is the revisable belief.

`diary` document:
```json
{"t": ISODate, "meta": {"session": "s1", "glasses": "B"},
 "object_id": "obj_7", "event": "placed", "place_id": "desk_drawer", "carried_by": null,
 "origin": "observed", "confidence": 0.93,
 "clip": {"clip_path": "clips/s1/B/IMG_0450.MOV", "offset_s": 55.2}, "track_id": 14}
```
`event` ∈ `appeared | left_view | picked_up | placed | missing | matched`
`origin` ∈ `observed | inferred | stated`

A `matched` event (from ElideDB retrieval — see `INTERFACES.md`) looks like:
```json
{"t": ISODate, "meta": {"session": "s1", "glasses": "B"},
 "object_id": "obj_7", "event": "matched", "place_id": "desk_drawer",
 "origin": "inferred", "confidence": 0.41,
 "evidence": {"query_clip": {"clip_path": "...", "start_s": 40.0, "end_s": 46.0},
              "matched_clip": {"clip_path": "...", "offset_s": 12.5}},
 "clip": {"clip_path": "...", "offset_s": 12.5}}
```
`confidence` here is the raw similarity score from ElideDB — never a made-up number.
`place_id` is resolved by OUR place recognizer on the matched clip's frame, not by
ElideDB, which has no concept of places or labels.

`objects` document:
```json
{"_id": "obj_7", "label": "keys", "status": "confirmed|believed|carried|faded|ambiguous",
 "place_id": "kitchen_counter", "carried_by": null,
 "last_confirmed_at": ISODate, "last_confirmed_by": "A", "belief_since": ISODate,
 "confidence": 0.9, "origin": "observed",
 "last_clip": {"clip_path": "...", "offset_s": 12.0}, "merges": []}
```

`fingerprints` document: `{"object_id", "vec": [384 floats], "place_id", "last_seen_at",
"glasses"}` — denormalize `place_id` / `last_seen_at` on every belief change, since the
vector index filters on them.

Vector index (create idempotently in `scripts/setup_db.py`):
```json
{"fields": [
  {"type": "vector", "path": "vec", "numDimensions": 384, "similarity": "cosine"},
  {"type": "filter", "path": "place_id"},
  {"type": "filter", "path": "last_seen_at"}]}
```

## Matching: three zones (identity)
1. Filtered `$vectorSearch` (`place_id == current place`) → join if `sim ≥ T_PLACE`
2. Unfiltered `$vectorSearch` → join if `sim ≥ T_HIGH`
3. Otherwise → **new object.** (Unclear zone stays "new object" until M6.)
On join: add the new fingerprint if it differs enough from existing ones (cap 12).
This is a "never merge when unsure" rule — see Core Principle 5 in `MASTERPLAN.md`.

## Belief state machine
- Seen at a place → `confirmed`, confidence 1.0.
- Leaves view → `believed` at last place; confidence fades:
  `conf = exp(-ln2 · Δt / FADE_HALF_LIFE_S)`; below `FADE_FLOOR` → `faded`
  ("not sure, last seen …").
- Picked up → `carried` by wearer until seen placed.
- **Missing:** a labeled place is recognized in view for ≥ 3s AND a believed object at
  that place isn't detected → `missing` diary event. If the object's last event was
  `picked_up`, call `find_similar_moment()` (see `INTERFACES.md`) with the
  last-confirmed-visible-to-gone clip window. Take the top result ≥ `ELIDE_MIN_SIM`;
  resolve its place via our recognizer; write a `matched` event. Belief becomes
  `believed` at that place, `origin: inferred`.
- A belief only ever changes on evidence (a sighting, a match, or a human label) — never
  on a timer alone. Confidence fade changes wording, never status, until it crosses
  `FADE_FLOOR`.

## Degrade honestly (Core Principle 6, operationalized)
- Diary write always happens first. If it fails, no belief update.
- If ElideDB is slow/unavailable: skip the match, answer from the last confirmed
  sighting, and say plainly that a search couldn't run.
- After 3 consecutive failures of an external service, stop calling it for the next
  60 seconds rather than blocking every request on it.
- Never let a missing/uncertain result become a confident wrong answer — default to
  "I'm not sure."

## Evaluation scoring (used by M4 and M5)
At each probe: right place/carrier = **+1**, honest "not sure" = **0**, confident wrong =
**−1**. Report counts ("10 of 12"), never percentages — sample sizes are too small for
percentages to mean anything.
