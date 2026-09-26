# CyclopsDiary — Startup

Read this first, in every session. It tells you what else to read, which depends on which
track you're building.

## Which track is this session?
Ask the human if it isn't obvious from what they're asking you to build. Don't guess.

**Track A** (lean core, evidence view, tuning, identity-by-history —
`M1`, `M3`, `M4`, `M6`):
1. Read `MASTERPLAN.md` in full.
2. Read `IMPLEMENTATION_STRATEGY.md` in full.
3. Read `INTERFACES.md` in full — you are the primary owner of seams 2, 3, and 4, and a
   consumer of seam 1.
4. Read the specific subplan you're building (`subplans/M1_lean_core.md`, etc.).

**Track B** (Cosmos 3 / ElideDB retrieval, load test — `M2`, `M5`):
1. Read `MASTERPLAN.md`, but you can skim past the build-order table for Track A rows.
2. Read `INTERFACES.md` §1 (`find_similar_moment()`); if building `M5`, also read §4
   (the cache format M5 consumes).
3. Read `IMPLEMENTATION_STRATEGY.md`'s "Time" and "MongoDB" sections only — you don't need
   the full belief state machine to build a retrieval function, only enough to know what
   your caller expects.
4. Read `subplans/M2_cosmos_elidedb.md` in full; if building `M5`, also read
   `subplans/M5_loadtest.md` in full.

Do not read the other track's full detail unless something in your own subplan explicitly
points you there. This isn't secrecy — it's to keep each session's context focused on what
it's actually building, given how little time there is.

## Working agreement, both tracks
- **Plan before building.** State your plan for the milestone, including anything in
  these documents you think is wrong, ambiguous, or missing, before writing code. Wait
  for approval on anything that isn't clearly specified.
- **Stop at milestone boundaries.** Don't drift into the next milestone's work even if it
  seems like a natural continuation. Each milestone has its own checks — pass them, then
  stop and report.
- **Don't silently work around missing data or credentials.** If `.env` is incomplete, or
  expected input files aren't where the docs say they should be, stop and say so. Don't
  fabricate placeholder data and continue as if it were real.
  **Exception:** small synthetic fixtures are allowed ONLY under `tests/fixtures/`,
  clearly named as fixtures, for unit tests of the memory/agent layer while real cache
  files are pending. Fixtures must never be written to `data/cache/` and must never be
  used by `scripts/ask.py`, `eval/score.py`, `eval/tune.py`, or `eval/loadtest.py` — those
  only ever run against real cache output.
- **Commit after each module**, not just at the end of a milestone. The build order in
  `MASTERPLAN.md` exists partly so that whatever exists at a checkpoint is usable —
  frequent commits are what make that true in practice.
- **Don't relitigate a superseded decision.** `MASTERPLAN.md` has a table of things
  already tried and dropped. If your own reasoning leads back to one of them, say so
  explicitly and ask, rather than quietly re-implementing it.
- **The core principles in `MASTERPLAN.md` are non-negotiable**, not defaults to optimize
  away under time pressure. If a shortcut would violate one of them (e.g. editing a diary
  entry instead of appending a new one), don't take it — flag the time pressure instead
  and ask what to cut elsewhere.

## Environment and data
- Credentials live in `.env` at the repo root (see `IMPLEMENTATION_STRATEGY.md` for the
  required keys). Never commit it.
- Input data conventions (`data/sessions.yaml`, `data/places/`, `data/clips/`,
  `data/labels/`) are specified in the M1 subplan and in `INTERFACES.md` §4.
- If a check in your subplan fails because expected data isn't present yet (e.g. Session 2
  hasn't been filmed), stop and tell the human rather than blocking on it silently or
  fabricating substitute data.

## Machines
Track A's laptop (the MSI) cannot run the vision models. Track B's laptop (the M5) runs
all heavy compute. Track A still writes all Track A code — only where certain scripts
*execute* changes.

- **The M5 runs:** `scripts/enroll_places.py`, `scripts/process_session.py`, and the
  ElideDB ingest script. Nothing else heavy runs anywhere.
- **The MSI runs everything downstream of Atlas and `data/cache/`:** memory, agent, app,
  evidence view, tuning, load test, and all `pytest` tests.
- **Handoff:** Track A pushes code to GitHub; Track B pulls and runs the scripts on the
  M5. Those scripts write to Atlas directly and write `data/cache/<session>/*.jsonl`,
  which Track B commits to the repo. Video clips live under `data/clips/` on local disk
  and are never committed (the repo is public; `data/clips/` is in `.gitignore`).
- **Clips reach both machines by one copy step, before anything else runs.** Straight off
  the phones by cable or AirDrop, to the MSI *and* the M5, into the same
  `data/clips/<session>/<glasses>/` layout on each. This is a copy, not a sync system —
  nothing runs afterwards to reconcile the two machines, and nothing checks that they
  agree.
  - **Filenames are fixed at transfer time and never renamed on either machine.** That is
    what makes `clip_path` valid on both: the paths match by construction, not because
    anything verifies them. Renaming a clip on one machine silently breaks either
    processing on the M5 or playback on the MSI, with no error pointing at the cause.
  - Both machines need the full set: the M5 reads clips to process them, and the MSI
    serves them to the browser from its `/clips/` mount, so a clip present on only one
    machine produces either a missing diary entry or a dead video player.
- If a check requires running YOLOE, DINOv2, or ElideDB and this session is on the MSI,
  **stop and report that the check must run on the M5.** Do not attempt to run the
  models locally, and do not substitute results.
- Run perception on the M5 **in sequence, not in parallel** with ElideDB ingest.
  `process_session.py` takes priority — M1 is the critical path.

## When you're done with a milestone
Report which checks passed, which didn't, and — if anything is genuinely ambiguous in the
docs and you made a judgment call to keep moving — say what you assumed and why, so it can
be corrected quickly if wrong.
