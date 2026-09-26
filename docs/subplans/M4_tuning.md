# M4 — Held-Out Tuning

Track A. Depends on M1's cache output (`data/cache/<session>/<scene>.jsonl`, format
frozen in `../INTERFACES.md` §4) and on Session 2 having been filmed and hand-labeled.

## What "tuning" means here — say this correctly in the demo
**No model weights change, anywhere.** The only thing being tuned is one number:
`FADE_HALF_LIFE_S`, the belief confidence decay rate. Say "the system tunes one of its
own memory settings from its mistakes, tested on a session it never saw" — not "we
trained the model." The wrong phrasing invites questions about training data and compute
that don't apply and will sound thin if asked.

## Why held-out, not tune-and-score-on-the-same-data
Tuning `FADE_HALF_LIFE_S` and then scoring on the exact same data it was tuned against is
studying the exam questions. It would look like learning but prove nothing. Session 1
(training) and Session 2 (test, filmed with different starting spots, routes, and handoff
destination — see `../MASTERPLAN.md`'s build order) make this a fair test: Session 2 is
genuinely new footage, not a re-enactment.

## Composites — volume without fabricating new visual evidence
Composites give you many orderings from a small number of real scenes, which is real
variety for **sequence-level** settings (like fade half-life) but not for
**appearance-level** ones (like the matching thresholds, `T_PLACE` / `T_HIGH`) — those
only improve with genuinely new footage. So: composite freely within a session, but
**tune only `FADE_HALF_LIFE_S`**. Leave the matching thresholds fixed at their M1
defaults.

Build `eval/composites.py`:
- Input: cached scene records from `data/cache/s1/*.jsonl` (training) or
  `data/cache/s2/*.jsonl` (test) — never mix sessions in one composite.
- Shuffle **whole scenes**, not individual clips within a scene — a scene's A and B
  perspectives must stay paired and time-shifted together, or handoffs become nonsense
  (Person A's "put down" landing hours from Person B's "picked up" in an unrelated loop).
- Gaps between chained scenes need an explicit rule per gap:
  - **Nothing moved:** only chain scenes where an object's end-state in one matches its
    start-state in the next. Truth: it stayed there for the whole gap.
  - **Something moved unseen:** truth records the object as moved immediately after it
    was last seen — the honest answer is "don't know its new spot," not a guess.
- Composite truth is derived automatically from the underlying scene labels — never
  hand-labeled directly.

## Tuning
`eval/tune.py`: grid-search `FADE_HALF_LIFE_S` over a small set of candidate values (e.g.
10 min, 30 min, 2h, 6h). Score each against Session 1 composites using the scoring rule
in `../IMPLEMENTATION_STRATEGY.md` (+1 / 0 / −1). Keep the best-scoring value.

## Testing
Score **Session 2** (never touched during tuning) twice: once with the default
`FADE_HALF_LIFE_S`, once with the tuned value. Save both runs to the `runs` collection
with their parameters and scores.

State openly, if asked: the tuned value depends on how often the composite builder
decided things "moved unseen" — this is an assumption about how often things get moved
out of sight in a household, not a universal constant.

## Done when
- `eval/composites.py` produces composites from S1 with derived truth, and separately
  from S2, without ever mixing sessions.
- `eval/tune.py` selects a `FADE_HALF_LIFE_S` from S1 composites only.
- Session 2, scored with default vs. tuned value, shows both results saved in `runs`,
  with a clear improvement to show (or an honest "no meaningful difference" if that's
  what happened — do not manufacture a result).
- You can state the assumed unseen-move rate in one sentence if asked.

Stop here. Report results.
