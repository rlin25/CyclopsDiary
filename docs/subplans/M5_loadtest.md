# M5 — Load Test

Track B. Depends on M1's cache output. Runs on either machine — it uses no vision models,
only cached detections replayed through the memory layer. Can be built in parallel with
M4 (Track A), or sequentially — they don't depend on each other.

## What this proves, and what it does not
This proves the **long-horizon** claim: memory size and query speed stay flat as footage
volume grows to a scale your real ~15 minutes of recordings can't demonstrate on their
own. It does **not** prove accuracy at that scale — that's what M4's held-out test is for,
on real, unshifted footage. Never blur these two claims together, on the chart or in
speech. Label the chart plainly: **"Load test: replayed detections from real footage."**

## Why replay saved detections, not raw video
Re-running YOLOE/DINOv2 on ~8 hours of equivalent footage would take hours on a laptop.
Instead: the vision models already ran once (in M1) and their output is cached per
`../INTERFACES.md` §4. Replaying that cached output through the memory system (diary write →
belief update → matching) takes minutes, because the expensive part (vision) is skipped
entirely. **Do not re-invoke YOLOE, DINOv2, or ElideDB during the load test** — anything
that calls an external service or a GPU model here is a bug.

## Build `eval/loadtest.py`
1. **Separate database.** Point at `cyclops_loadtest` (a distinct Mongo database from the
   real `cyclops` one used by the live demo) so replayed data never mixes with what the
   agent answers from live. Reuse the real clip paths on disk — don't re-upload or duplicate video files.
2. **Shuffle scenes, not frames or individual clips** — same pairing rule as M4's
   composites (a scene's A/B perspectives move together). Mix scenes from whatever
   sessions are cached; unlike M4, mixing sessions here is fine since this isn't a
   held-out accuracy test.
3. **Shift timestamps forward** on each loop so the simulated timeline keeps advancing
   past real recorded time — label these as simulated/artificial on the chart, never
   implied to be real.
4. **Skip Cosmos/ElideDB calls.** The `missing`-event handler's retrieval call should be
   disabled or stubbed for the load test — it's testing memory scale, not retrieval, and
   real API/store calls at this volume would be slow and pointless. Note this omission on
   the chart or in the write-up.
5. **Checkpoint every simulated 10 minutes**, recording four numbers to the `runs`
   collection:
   - frames represented so far (cumulative)
   - diary entries so far (cumulative)
   - objects in the contact list (should level off once every distinct object has been
     seen at least once — a flattening curve is expected and good, not a bug)
   - latency of the `get_belief` MongoDB query only (not the agent's full response time,
     which includes an LLM call and would muddy a chart that's supposed to be about the
     database)
6. Run until ~8 hours of equivalent footage across both phones combined
   (~860,000 frames/phone at 30fps × 2 phones ≈ 1.7M frames total — use this as the
   target, not a hard requirement if time runs short; a smaller but still large multiple
   of real footage still makes the point).
7. **Chart:** three lines against simulated time — frames (climbing steeply), diary
   entries (rising in small steps only when something actually happens), query latency
   (flat). Save as a PNG, labeled "Load test: replayed detections from real footage."

## Honest limits to state if asked
The replay reuses the same small set of real objects and places — it proves the memory
handles *volume and time*, not that it handles *many distinct objects*. Say this plainly
if asked; understanding the limits of your own evidence is itself a point in your favor
in Q&A.

## Done when
- `eval/loadtest.py` runs against `cyclops_loadtest`, never touches the live `cyclops`
  database, and never calls YOLOE/DINOv2/ElideDB.
- The four metrics are saved to `runs` at each simulated checkpoint.
- A labeled PNG chart exists showing the three-line pattern described above.
- You can state the ~8h-equivalent frame count and the honest limitation in one sentence
  each.

Stop here. Report results.
