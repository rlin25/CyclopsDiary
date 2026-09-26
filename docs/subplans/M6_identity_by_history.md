# M6 — Identity by History

Track A (or whoever's free after M4/M5). **Only build this after M1–M5 all pass.** This is
the last thing built before the 3:30 freeze, and it only goes into the *live* demo if it
passes rehearsal by then — otherwise it stays as recorded-footage evidence and a prepared
Q&A answer, never shown live unrehearsed. See `../MASTERPLAN.md`'s superseded-decisions
table for why a live mismatch here is treated as worse than not showing it at all.

## The problem
Two visually identical objects (e.g. two matching mugs) cannot be told apart by
appearance alone — no model can do that, by definition, if they're truly identical. The
lean core's three-zone matcher (M1) handles this safely already: an unclear match becomes
a *new* object rather than a guessed merge. That's correct and safe, but it means two
identical mugs currently just become two permanently separate objects with no way to tell
which is "yours."

## The fix: history, not appearance
When two candidates are equally plausible by fingerprint similarity, use what appearance
can't provide: which one's last known place, time, and carrier makes *this* sighting most
plausible. A mug that appears on the desk ten seconds after someone was seen carrying a
mug from the counter is, with very high probability, the counter mug — not because it
looks different, but because of what happened around it.

## Build
This slots into the **unclear zone** of the three-zone matcher (step 6 in
`../IMPLEMENTATION_STRATEGY.md`), which currently just creates a new object. Replace that
default only for this specific sub-case:

1. When a sighting's fingerprint similarity is ambiguous between two or more existing
   objects (not clearly a join, not clearly new — genuinely tied or very close):
2. For each candidate, compute a plausibility score from: recency of last sighting,
   distance/relationship between the candidate's last place and the current sighting's
   place, and whether the candidate was recently `carried_by` someone who could plausibly
   have brought it here.
3. If one candidate's plausibility clearly dominates → join to that one, and record which
   evidence decided it (for the evidence view — see M3).
4. If it's still tied after history is considered → do **not** guess. Create the object
   with `status: "ambiguous"`, listing both candidate identities, and let the belief
   resolve itself when new evidence (a clearer sighting, a carry event) arrives. This is
   the same "never merge when unsure" principle from Core Principle 5 — just applied one
   level deeper, to identity itself rather than only to detection.

## Test before it goes anywhere near the live demo
Use the two-mug scene from your recordings specifically. Run it enough times to know
whether it resolves correctly, consistently — not once. "It worked once" is not
"it passed rehearsal."

## Done when
- The unclear-zone history tie-breaker is implemented and covered by a test using the
  recorded two-mug scene.
- An `ambiguous` status is handled gracefully everywhere else it could surface (agent
  wording, evidence view, beliefs) — it should never crash or produce a confident wrong
  answer, only an honest "I can't tell which one yet."
- A clear pass/fail rehearsal verdict exists by 3:30: either it's going in the live demo,
  or it isn't, with no ambiguity about which.

If it doesn't pass rehearsal: leave it in the codebase (it's still valid work, shown in
recorded footage and test scores) but do not include it in the live run-through. Use the
prepared Q&A answer from `../MASTERPLAN.md` instead.
