# M1 — Lean Core

Track A. This is the one-shot target: everything below, and nothing else, until it passes
every check. The predictor/retrieval call is a stub (returns `[]`) until M2 lands — build
the `missing`-event handler to call `find_similar_moment()` per its real interface in
`../docs/INTERFACES.md`, just point it at a stub implementation for now so M1 doesn't block on M2.

## Repo layout
```
cyclops/
  config.py                 # env + all tunables (thresholds, fade half-life, etc.)
  timeutil.py                # clip start time (ffprobe creation_time) + session offset
  perception/detect.py       # YOLOE track -> per-frame detections; persistence filter
  perception/embed.py        # DINOv2 crop + whole-frame fingerprints
  perception/places.py       # place recognition vs labeled backdrops
  memory/db.py                # collections, indexes
  memory/diary.py             # append-only writes
  memory/registry.py          # objects (contact list) + fingerprints
  memory/matcher.py           # three-zone matching via $vectorSearch
  memory/beliefs.py           # belief state machine + confidence fade
  retrieval/similar_moment.py # stub matching docs/INTERFACES.md §1 until M2 lands
  agent/tools.py  agent/agent.py
  app/server.py  app/static/index.html
eval/score.py                # M4/M5 live in eval/ too, but score.py is needed by M1's own checks
scripts/
  setup_db.py  enroll_places.py  process_session.py  ask.py
data/
  sessions.yaml  places/  clips/<session>/<glasses>/  labels/<session>/  cache/
tests/
```

## Inputs (prepared by humans before running — not your job to generate)
- `data/sessions.yaml` — per session: clips per phone (`A`, `B`), wearer names, phone
  clock offset in seconds (from the sync shot)
- `data/places/<place_id>/*.mp4` — short backdrop clips from BOTH phones; `place_id` is
  the human-assigned label (e.g. `kitchen_counter`)
- `data/labels/<session>/<scene>.yaml` — hand-written ground truth, format frozen in
  `../docs/INTERFACES.md` §4

## Pipeline (per session)
0. **Vocabulary check (on the M5):** run YOLOE prompt-free on one clip and record the
   actual class names it emits. Confirm whether "hand", "keys", and "mug" (or their real
   equivalents) exist in the output. Set the ignore list and the hand class name in
   `config.py` from the real output, not from assumption.
1. **Places:** `enroll_places.py` fingerprints whole frames (1 fps) from each backdrop
   clip → `places` collection.
2. **Detect:** YOLOE prompt-free + BoT-SORT tracking at 5 fps sampled frames. Ignore
   structural/background classes (config list: wall, floor, table, shelf, cabinet,
   person, …) — these are not tracked objects.
3. **Persistence filter (kills empty/phantom boxes):** keep a track only if seen in
   ≥ 5 sampled frames with conf ≥ 0.35. Anything filtered out never reaches MongoDB.
4. **Place per frame:** cosine-nearest labeled place on the whole-frame fingerprint;
   accept if `sim ≥ PLACE_MIN_SIM`, else `null`.
5. **Carried detection:** object box overlaps a `hand` box in the wearer's own view for
   ≥ 1s, OR the track persists across a change of recognized place → `carried_by =
   <glasses wearer>`.
6. **Match each surviving track (three-zone — full spec in
   `../docs/IMPLEMENTATION_STRATEGY.md`):** filtered `$vectorSearch` on current place →
   unfiltered `$vectorSearch` → else new object. On join, add the fingerprint if
   sufficiently different (cap 12 per object).
7. **Consolidate into diary events**, written only on state change: `appeared`,
   `left_view`, `picked_up`, `placed`, `missing`. (`matched` comes from the M2 hook, see
   below.)
8. **Beliefs**, updated after each diary write — full state machine in
   `../docs/IMPLEMENTATION_STRATEGY.md`. The `missing` handler calls
   `retrieval.similar_moment.find_similar_moment()` (stubbed in M1, real in M2) per the
   signature in `../docs/INTERFACES.md` §1. Do not build a different signature "for now" — build
   the real one against a stub body, so M2 is a drop-in.
9. **Cache** every surviving track's full record (detections, fingerprint, recognized
   places over time, diary events produced) to
   `data/cache/<session>/<scene>.jsonl` — format frozen in `../docs/INTERFACES.md` §4. This is
   required output, not optional logging — M4/M5 depend on it existing and matching the
   frozen shape.

## Agent (read-only, 5 tools per `../docs/INTERFACES.md` §3)
Implement `find_object`, `get_belief`, `what_is_at`, `timeline`, and
`find_similar_past_moment` (the last one calling the same stub/real
`find_similar_moment()` as step 8). Wire these as tools for the OpenRouter-hosted agent
(OpenAI tool-calling format).

Answer wording by status × origin:
| Status | Wording |
|---|---|
| confirmed | "Your keys are on the kitchen counter (seen by A at 9:02)." |
| believed | "Last seen on the kitchen counter at 9:02; probably still there." |
| carried | "Last seen being carried by B at 9:15." |
| believed, origin=inferred (from a `matched` event) | "Last seen carried by B; a similar past moment points to the desk drawer (not confirmed)." |
| faded / none | "I haven't seen your keys since 9:02." / "I haven't seen your wallet." |

## App
`POST /ask {question, asker}` → `{answer, status, origin, clip_url, offset_s, evidence}`.
One static page: question box → answer card (status badge: SEEN vs BELIEVED) → video
jumps to `offset_s` → collapsible **Evidence** panel showing the Atlas query/queries and
diary entries behind the answer. Open the page on the question, never on bounding boxes
or a table — those read as "image analyzer" / "dashboard," both on the hackathon's banned
list.

## Done when ALL of these pass
- **(either machine)** `python scripts/setup_db.py` creates all collections + the one
  vector index (idempotent — safe to re-run).
- **(M5)** `python scripts/enroll_places.py` fills `places`.
- **(M5)** `python scripts/process_session.py s1` fills `diary`, `objects`,
  `fingerprints`, and writes `data/cache/s1/*.jsonl` matching the frozen format.
- **(MSI, against real cache)** `python scripts/ask.py "where are my keys?" --asker A`
  returns the correct place from the s1 handoff scene's hand-written label.
- **(MSI, against real cache)** `python -m eval.score s1` prints the +1/0/−1 totals per
  `../docs/IMPLEMENTATION_STRATEGY.md`'s scoring rule.
- **(MSI; may use `tests/fixtures/`)** `pytest` passes, including Hypothesis property
  tests for Core Principles 1–5 (append-only, diary-before-belief, traceability,
  sightings-beat-inferences, never-merge-when-unsure).
- **(MSI)** `uvicorn cyclops.app.server:app` serves the page; the answer card plays the
  clip at the right second.

Stop here. Report results. Do not start M2/M3/etc. even if they seem like a natural
continuation.
