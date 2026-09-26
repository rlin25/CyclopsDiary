# M2 — Cosmos 3 / ElideDB Motion Retrieval

Track B. This is a single, isolated, swappable module. Build and prove it standing alone,
against real clips, before it is ever wired into Track A's `missing`-event handler.

**Read `INTERFACES.md` §1 before anything else in this file.** That is the frozen
contract. This subplan is about how you get there, not what the function returns — the
signature is not up for revision without telling Track A.

## What this is, and what it categorically is not
ElideDB (github.com/exploring-curiosity/ElideDB), used here as the retrieval layer behind
"Cosmos 3" in the pitch, is a **motion-similarity retrieval engine**. Two frozen,
off-the-shelf vision encoders turn a clip into a trace; an elastic time-alignment step
ranks stored recordings by how similar their motion is to a query clip. That's the entire
mechanism.

It is **not**:
- a language model (there is no generative model anywhere in its path)
- a reasoning or prediction engine (it does not infer "where the keys went")
- label-aware (it has no concept of objects, places, or names — only motion patterns)

Every result it returns is: a stored clip, an offset into it, and a similarity score.
Nothing else. Do not build anything on top of it that implies reasoning, and do not let
the demo narration imply reasoning either (see `MASTERPLAN.md`'s Q&A section for the
correct framing).

## Setup
0. Confirm ElideDB runs on Apple Silicon (the M5) before anything else — check whether
   its encoders support MPS/CPU or require CUDA. If it cannot run on the M5, report
   immediately; this decides the 2:00 checkpoint early (M2 out).
1. Ingest the real footage into an ElideDB store — one pass, resumable, roughly 14
   minutes of one GPU per hour of video. For a handful of minutes of real recordings this
   is trivial, but it must run **once, early**, not at query time. Both phones' footage
   for a session go into the same store, so retrieval can surface a match from the *other*
   camera.
2. Confirm the CLI/API round-trip works on your machine before writing any wrapper code:
   `elidedb add <store> <path-to-clips>`, then `elidedb query <store> <clip>` against a
   known pair of similar clips you expect to match (e.g. two different phones' footage of
   the same handoff moment, if you have it, or two separate instances of keys being set
   down).

## Build `find_similar_moment()`
Signature, constraints, and return shape are frozen in `INTERFACES.md` §1. Two things
worth restating because they're easy to get wrong under time pressure:

- **4-second minimum clip length.** If Track A calls you with a shorter window, that's
  their bug to fix (they own padding it), not yours to silently accept a shorter clip —
  fail clearly instead, so the mismatch surfaces immediately rather than as a confusing
  runtime issue later.
- **You do not resolve places.** Return `offset_s` and `similarity` and stop. Track A's
  place recognizer figures out what's at that offset. Do not be tempted to add a "best
  guess place" field — it would imply a capability that doesn't exist and would violate
  the interface contract.

## Test standing alone, before integration
Prove this works as its own demoable thing, independent of the rest of the pipeline:
"here's a clip of the keys being picked up on the counter; here are the top 5 most
similar moments found across both phones' footage, ranked, with timestamps."

If a genuine hit — the other camera's footage of the same event, or footage of the keys
being set down elsewhere — shows up in the top few results with a similarity score that
looks meaningfully higher than the rest, this module is working. If results look close to
random or the true match doesn't appear near the top, that's the signal to bring to the
2:00 checkpoint, not something to keep tuning past it.

## Wire into the `missing`-event handler
Once standing-alone testing looks good, swap Track A's stub in
`cyclops/retrieval/similar_moment.py` for your real implementation. This should be a
drop-in — if it isn't, the signature drifted from `INTERFACES.md` and needs to be fixed
back to match, not worked around on either side.

## The 2:00 checkpoint — this is a real go/no-go, not a soft deadline
By ~2:00, decide honestly:
- **In:** results are good enough that a `matched` event in the live demo would show a
  correct or at least plausible location. Proceed to full integration; the demo's
  headline scene depends on it.
- **Out:** results aren't reliable enough yet. Tell Track A immediately. The `missing`
  handler keeps calling the stub (returns `[]]`) for the *live* demo, so beliefs simply
  go to `believed`/`faded` without a match — which is still a correct, honest answer, just
  a less impressive one. Keep working on retrieval in the background for the **offline**
  eval/demo-prep path (M4/M5 composites, or a pre-recorded fallback clip of the
  headline scene using retrieval that worked once even if not reliably), where a bad
  result can't hurt a live answer.

Do not let this checkpoint slip informally. A late "actually it's working now" at 3:15 is
much less valuable than a clean decision at 2:00 that let Track A build the rest of the
day around a known state.

## Q&A framing to have ready (also in `MASTERPLAN.md`)
"Cosmos 3, via ElideDB, isn't reasoning or predicting here — it's retrieval by motion
similarity. It finds the other camera's footage of the same event; our own place
recognition reads the location off the winning clip." Be ready to show the evidence
(query clip, matched clip, similarity score) live if asked — there is no black box to
defend, which is the point.
