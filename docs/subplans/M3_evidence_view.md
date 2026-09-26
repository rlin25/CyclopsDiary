# M3 — Evidence View

Track A. Depends on M1 being done. Small, but earns real points: it's the difference
between "trust us" and showing the actual MongoDB query and diary entries behind an
answer, and it's the primary defense against the "is this just an image analyzer / a
dashboard" objection — detections and raw data appear here, behind a click, not on the
main screen.

## What to build
Expand the `evidence` field already returned by `get_belief()` (per `../INTERFACES.md` §3)
into a real UI panel on the answer card:

- The exact `$vectorSearch` (or aggregation) pipeline that produced the match, rendered
  as readable JSON or a short plain-language paraphrase of it — not hidden, not
  summarized away.
- The diary entries that the belief traces back to, in time order, each showing its
  `event`, `origin`, `confidence`, and a link to jump the video to that moment.
- For a `matched` event specifically (from M2's retrieval): show the query clip, the
  matched clip, the similarity score, and which place our own recognizer read off the
  matched clip — making visible that retrieval found the clip and our own system
  independently read the location, with no reasoning step in between.

## Placement
Collapsible, closed by default. The answer card itself stays simple (the point, the
status badge, the clip) — the evidence panel is for when someone asks "how do you know
that" or "show me it's not just guessing."

## Done when
- Every answer type (confirmed, believed, carried, matched, faded, none) renders a
  sensible evidence panel — not just the happy path.
- A `matched` event's panel clearly separates "what ElideDB returned" (clip, offset,
  similarity) from "what our own system concluded" (the place) — this distinction is the
  whole point, per M2's framing.
- Panel is legible on the projector/screen you'll actually demo on — check font size and
  contrast, not just that it renders.

Stop here. Report results.
