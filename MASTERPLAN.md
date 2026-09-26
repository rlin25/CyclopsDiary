# CyclopsDiary — Masterplan

MongoDB Harness Engineering Hackathon · Statement Two (Long Horizon Engineering)

## What this is
Two phones stand in for two pairs of Meta glasses. They feed ONE shared memory that keeps
what was SEEN (record) separate from what is BELIEVED (belief). An agent answers
"where are my keys?" from that memory, with the video clip as proof — including things
only the OTHER person's glasses saw.

Pitch framing: shared memory for many agents watching the same world. Warehouses are the
headline example; the keys are the demo.

## The one scene everything is built around
1. Person A leaves keys on the kitchen counter and walks off.
2. Person B picks them up (caught on camera) and carries them toward the desk drawer.
   The put-down itself is **missed** by both cameras.
3. Person A asks the agent, "Where are my keys?"
4. The agent answers from **Person B's glasses**: last seen carried by B; a matched clip
   (via Cosmos 3 / ElideDB motion retrieval) points to the desk drawer as the most similar
   prior event, shown as a **belief**, not a fact — with the evidence (query clip, matched
   clip, similarity score) visible.
5. Person A's glasses later see the open drawer with the keys inside; the belief flips to
   **confirmed**.

Everything in the build order below exists to make this one sequence work, provably, live.

## Core principles (non-negotiable — do not "fix" these mid-build)
1. **Diary is append-only.** Every sighting/event is written once and never edited.
2. **Diary before belief.** Write the diary entry first. If that write fails, the belief
   does not change.
3. **Every belief is traceable** to a diary entry (observed), a retrieval match
   (inferred), or a human label (stated).
4. **Sightings beat inferences.** A newer observed sighting always overrides an inferred
   location.
5. **Never merge when unsure.** An uncertain object match creates a NEW object. Duplicates
   are fixable later; wrong merges quietly corrupt every answer after them and are not.
6. **Degrade honestly.** If a service is slow/down, answer from the last confirmed
   sighting and say plainly what's missing. Never freeze the demo waiting on a timeout.
7. **The agent is read-only.** It can only report what its tools return. Empty result →
   "I don't know." It never writes to the diary or the contact list.

## Build order and checkpoints
Counting from session start. Compress proportionally if starting later — never move the
freeze or the recording block.

| By | Track A (you) | Track B (partner) |
|---|---|---|
| Now | Pre-flight: `.env`, Session 1 recording (sync shot, backdrops, scenes, handoff, look-alike mugs), hand-written answer key | Confirm Atlas Sandbox tier. ElideDB ingest on Session 1 footage. Build `find_similar_moment()` against `INTERFACES.md` |
| ~1:30 | M1 (lean core) passes end to end. **Screen-record immediately.** | `find_similar_moment()` returning sane ranked results on real clips |
| ~1:30–2:00 | M3 (evidence view) | Wire retrieval into the `missing`-event handler (M2) |
| **~2:00** | **Checkpoint: M2 in or out. No extensions.** | Same |
| ~2:15 | Film Session 2 (test session — change starting spots, routes, handoff destination) | Save cached detections for the load test |
| ~2:30–3:30 | M4 (held-out tuning) | M5 (load test) |
| **3:30** | **Feature freeze.** M6 (identity by history) only if it already passes rehearsal. | Same |
| 3:30–4:30 | Record demo + backup; rehearse the 3-minute story twice | Same |
| ~4:45 | Submit: public repo, 1-minute video, all members added | Same |

## Cut — do not build
Live streaming · Streamlit or a dashboard as the main feature · Hindsight · voice output ·
the to-do reminder harness · Online Archive (pitch only, not demoed) · auth · Voyage
(unless everything else is done early) · a Claude "referee" for uncertain matches ·
training or fine-tuning any model · Cosmos/ElideDB doing reasoning, description, or
prediction — it is retrieval only · any dependency not already named, without asking.

## Superseded decisions
Kept so nobody re-proposes these mid-build. If a conversation drifts back toward one of
these, the answer is "already tried, see below," not a re-litigation.

| Was | Why it was dropped | Now |
|---|---|---|
| Cosmos 3 describes what happened at a disappearance (running live, generating text) | Decoration — a nicer sentence, not a capability only a world model has. Slow, and NVIDIA's own docs warn it can hallucinate entities. | Cosmos 3, via ElideDB, does motion-similarity **retrieval** only. No text generation, no reasoning, no confidence rationale. |
| Cosmos 3 / ElideDB predicts an object's destination with a rationale | Misread of the tool — ElideDB has no language model and no labels anywhere in its path; it ranks clips by motion similarity, nothing more. | `find_similar_moment()` returns ranked `(clip, offset, similarity)` only. Place is resolved by our own place recognizer on the winning clip, never by the retrieval tool. |
| Unclear object matches go to Claude as a referee, comparing crops | Adds a live LLM call on the identity-matching critical path, for a case with no guaranteed right answer. | Unclear → new object, by default. Identity-by-history (M6) resolves look-alikes using place/time/carrier evidence instead, gated on rehearsal. |
| Furniture-box overlap decides an object's location | Fragile — depends on furniture also being detected correctly every frame. | Whole-scene place recognition (DINOv2 fingerprint of the backdrop) against places labeled by hand before recording. |
| Video streamed live and cut into one-minute S3 pieces | Only makes sense for a live feed; the actual pipeline processes uploaded clips. | Uploaded clips stored as-is in S3; diary entries point to a clip + offset. |
| Tune and score the self-tuning loop on the same answer key | Overfitting — tuning on the exam you then grade yourself on. | Tune (fade half-life only) on Session 1 composites; test on Session 2, never used for tuning. |
| Long-horizon claim rests on the self-tuning loop alone | 15 minutes of real footage proves nothing about scale. | A clearly labeled load test (replayed cached detections, ~8h equivalent) proves size/speed; real Session 1/2 timestamps (hours apart) prove correctness over time. |
| Look-alike objects committed to the live demo unconditionally | A live mismatch would damage the whole demo's credibility, not just that answer. | Live only if it passes rehearsal before the 3:30 freeze. Otherwise: prepared Q&A answer, not shown live. |

## Win conditions (do not lose points on these)
- Built inside the provided MongoDB Atlas Sandbox — confirm this **today**, not later.
- Public GitHub repo, with commits spread through the day (judges check for this).
- One-minute submission video, separate from the live demo.
- All team members added to the submission.
- Someone at MongoDB.local NYC on September 30th from 10 AM if you make finalists.

## Q&A answers to have ready
- **"Did you train a model?"** No. The system tunes one of its own memory settings (fade
  half-life) from its mistakes, tested on a session it never saw during tuning.
- **"What about identical items?"** Identity by history: when appearance can't decide,
  last known place/time/carrier does. When even that can't decide, the belief says so
  openly rather than guessing.
- **"Why a world model in a memory system?"** Cosmos 3, via ElideDB, retrieves by motion
  similarity — a different axis from our appearance-based vector search. It finds the
  other camera's footage of the same event; our own place recognition reads the location
  off the winning clip.
- **"Is the load test real?"** It replays real detections from real footage, clearly
  labeled as a load test. It proves memory size and query speed stay flat as footage
  grows — it does not claim to prove accuracy at that scale.
