#!/usr/bin/env python
"""Process a session's footage into the diary, the contact list, and the §4 cache.

RUNS ON THE M5 ONLY — it loads YOLOE and DINOv2.

Pipeline (M1 subplan steps 2-9; step 0 is check_vocab.py, step 1 is enroll_places.py):

    detect + track -> persistence filter -> place per frame -> carried detection
    -> three-zone identity match -> diary events on state change -> beliefs
    -> data/cache/<session>/<scene>.jsonl

Both phones are processed on ONE timeline, ordered by each clip's hand-entered `start_time`,
not phone by phone. The demo is A asking about what B saw, so a `missing` event in A's view has
to be able to see a belief that B's footage created minutes earlier.

    python scripts/process_session.py s1
    python scripts/process_session.py s1 --scene s1_handoff
    python scripts/process_session.py s1 --dry-run
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                        # noqa: E402

from cyclops import config, timeutil                      # noqa: E402
from cyclops.memory import beliefs as beliefsmod          # noqa: E402
from cyclops.memory import db as dbmod                    # noqa: E402
from cyclops.memory import diary as diarymod              # noqa: E402
from cyclops.memory import matcher as matchermod          # noqa: E402
from cyclops.memory import registry as registrymod        # noqa: E402
from cyclops.perception import detect, embed, places       # noqa: E402
from cyclops.retrieval import similar_moment              # noqa: E402

log = logging.getLogger("process_session")

#: Crops embedded per track. A track can run for hundreds of frames; 12 spread across it is
#: enough for a representative fingerprint and matches the per-object fingerprint cap.
MAX_CROPS_PER_TRACK = 12

CACHE_SCHEMA = "cyclops.cache.v1"


class Pipeline:
    def __init__(self, database, place_index: places.PlaceIndex, dry_run: bool = False):
        self.db = database
        self.place_index = place_index
        self.dry_run = dry_run
        self.diary = diarymod.Diary(database[dbmod.DIARY])
        self.registry = registrymod.Registry(database)
        self.beliefs = beliefsmod.BeliefService(self.diary, self.registry)
        self.matcher = matchermod.Matcher(database)
        self.detector = detect.Detector()
        self.embedder = embed.Embedder()
        self.counts: collections.Counter[str] = collections.Counter()

    # ------------------------------------------------------------------------ one clip

    def process_clip(self, clip: timeutil.Clip, session: dict) -> list[dict]:
        """Returns one cache record per surviving track."""
        wearer = timeutil.wearer(session, clip.glasses)
        log.info("=== %s / %s (glasses %s, worn by %s) start %s",
                 clip.scene, clip.clip_path, clip.glasses, wearer, clip.start_time.isoformat())

        frames: list[detect.FrameDetections] = []
        place_samples: list[tuple[float, places.PlaceGuess]] = []
        crops: dict[int, list] = collections.defaultdict(list)
        crop_offsets: dict[int, list[float]] = collections.defaultdict(list)

        for frame in self.detector.track_clip(clip.fs_path):
            # Whole-frame place recognition, from the same decoded frame as the crops.
            if frame.image is not None:
                guess = self.place_index.recognize(self.embedder.embed_one(frame.image))
            else:
                guess = places.PlaceGuess(None, 0.0)
            place_samples.append((frame.offset_s, guess))

            for det in frame.objects:
                if len(crops[det.track_id]) < MAX_CROPS_PER_TRACK:
                    crop = frame.crop(det.bbox)
                    if crop is not None:
                        crops[det.track_id].append(crop)
                        crop_offsets[det.track_id].append(det.offset_s)
                    else:
                        self.counts["empty_crops"] += 1

            frame.image = None  # decoded frames are not accumulated
            frames.append(frame)

        if not frames:
            log.warning("%s decoded no frames", clip.clip_path)
            return []

        sample_period = frames[1].offset_s - frames[0].offset_s if len(frames) > 1 else 1.0 / config.SAMPLE_FPS
        place_by_offset = {offset: guess for offset, guess in place_samples}
        place_runs = places.collapse_runs(place_samples, sample_period)
        log.info("%d frames sampled; places seen: %s", len(frames),
                 [r.place_id for r in place_runs] or "none recognized")

        tracks = detect.group_tracks(frames)
        kept, dropped = detect.persistence_filter(tracks)
        self.counts["tracks_seen"] += len(tracks)
        self.counts["tracks_dropped"] += len(dropped)
        for track_id, why in dropped.items():
            log.debug("dropped track %s: %s", track_id, why)

        records = []
        for track_id, track in sorted(kept.items(), key=lambda kv: kv[1].first_seen_s):
            records.append(self._process_track(
                track, clip, session, wearer, frames, place_by_offset, sample_period,
            ))

        # Which objects this clip actually saw. Taken from the records rather than from the
        # tracks, because the object_id is only known after matching.
        seen_here = {record["object_id"] for record in records}
        self._detect_missing(clip, session, place_runs, seen_here)
        return records

    # ----------------------------------------------------------------------- one track

    def _process_track(self, track, clip, session, wearer, frames, place_by_offset,
                       sample_period) -> dict:
        crop_images = self._crops_for(track, frames)
        vectors = self.embedder.embed(crop_images) if crop_images else np.zeros((0, config.EMBED_DIM), np.float32)
        fingerprint = embed.average_fingerprint(list(vectors))
        degenerate = embed.is_degenerate(fingerprint)

        track_places = [place_by_offset.get(d.offset_s, places.PlaceGuess(None, 0.0)) for d in track.detections]
        recognized = [g.place_id for g in track_places if g.place_id]
        place_at_start = recognized[0] if recognized else None
        place_at_end = recognized[-1] if recognized else None
        changed_place = bool(recognized) and place_at_start != place_at_end

        overlap_s = detect.hand_overlap_seconds(track, frames, sample_period)
        carried = overlap_s >= config.CARRY_MIN_OVERLAP_S or changed_place
        carried_by = clip.glasses if carried else None

        match = self.matcher.match(fingerprint, place_at_start)
        t_first = clip.time_at(track.first_seen_s)
        t_last = clip.time_at(track.last_seen_s)

        if match.is_new:
            object_id = self.registry.next_object_id()
            self.registry.create_object(
                label=track.class_name, status="confirmed", place_id=place_at_start,
                carried_by=None, confidence=1.0, origin="observed", at=t_first,
                seen_by=clip.glasses, clip_path=clip.clip_path, offset_s=track.first_seen_s,
                object_id=object_id,
            )
            self.counts["objects_created"] += 1
        else:
            object_id = match.object_id
            self.registry.record_merge(object_id, {
                "at": t_first, "zone": match.zone, "similarity": round(match.similarity, 4),
                "clip_path": clip.clip_path, "track_id": track.track_id,
            })
            self.counts["tracks_joined"] += 1
        log.info("track %s (%s) -> %s [%s, %s]", track.track_id, track.class_name, object_id,
                 match.zone, match.reason)

        events: list[dict] = []
        if place_at_start:
            events.append(self.beliefs.seen_at_place(
                object_id=object_id, place_id=place_at_start, t=t_first, session=clip.session,
                glasses=clip.glasses, clip_path=clip.clip_path, offset_s=track.first_seen_s,
                track_id=track.track_id, first_sighting=match.is_new,
            ))
        if carried:
            events.append(self.beliefs.picked_up(
                object_id=object_id, carried_by=carried_by, t=t_first, session=clip.session,
                glasses=clip.glasses, clip_path=clip.clip_path, offset_s=track.first_seen_s,
                track_id=track.track_id,
            ))
        if changed_place and place_at_end:
            events.append(self.beliefs.seen_at_place(
                object_id=object_id, place_id=place_at_end, t=t_last, session=clip.session,
                glasses=clip.glasses, clip_path=clip.clip_path, offset_s=track.last_seen_s,
                track_id=track.track_id,
            ))
        events.append(self.beliefs.left_view(
            object_id=object_id, place_id=place_at_end, t=t_last, session=clip.session,
            glasses=clip.glasses, clip_path=clip.clip_path, offset_s=track.last_seen_s,
            track_id=track.track_id,
        ))

        if not degenerate:
            self.registry.add_fingerprint(
                object_id, fingerprint, place_id=place_at_end or place_at_start,
                last_seen_at=t_last, glasses=clip.glasses,
            )

        return {
            "schema": CACHE_SCHEMA,
            "session": clip.session, "scene": clip.scene, "glasses": clip.glasses,
            "clip_path": clip.clip_path,
            "track_id": track.track_id,
            "object_id": object_id,
            "class_name": track.class_name,
            "first_seen_s": track.first_seen_s, "last_seen_s": track.last_seen_s,
            "carried_by": carried_by,
            "detections": [
                {
                    "offset_s": d.offset_s,
                    "t": clip.time_at(d.offset_s).isoformat(),
                    "conf": round(d.conf, 4),
                    "bbox": [round(float(x), 2) for x in d.bbox],
                    "place_id": place_by_offset.get(d.offset_s, places.PlaceGuess(None, 0.0)).place_id,
                    "place_sim": round(place_by_offset.get(d.offset_s, places.PlaceGuess(None, 0.0)).sim, 4),
                    "hand_overlap": any(detect.boxes_overlap(d.bbox, h)
                                        for f in frames if f.offset_s == d.offset_s for h in f.hands),
                }
                for d in track.detections
            ],
            "places": [r.as_dict() for r in places.collapse_runs(
                [(d.offset_s, place_by_offset.get(d.offset_s, places.PlaceGuess(None, 0.0)))
                 for d in track.detections], sample_period)],
            "fingerprint": {
                "vec": [round(float(x), 6) for x in fingerprint],
                "n_crops": len(crop_images),
                "degenerate": bool(degenerate),
            },
            "match": {"zone": match.zone, "similarity": round(match.similarity, 4), "reason": match.reason},
            "hand_overlap_s": round(overlap_s, 3),
            "diary_events": diarymod.to_json_safe(events),
        }

    @staticmethod
    def _crops_for(track, frames) -> list:
        """Crops already gathered during the decode pass, re-read here from the track's boxes.

        Kept as a separate step so a track that was filtered out never costs an embedding.
        """
        by_offset = {f.offset_s: f for f in frames}
        out = []
        step = max(1, len(track.detections) // MAX_CROPS_PER_TRACK)
        for det in track.detections[::step][:MAX_CROPS_PER_TRACK]:
            frame = by_offset.get(det.offset_s)
            crop = frame.crop(det.bbox) if frame is not None else None
            if crop is not None:
                out.append(crop)
        return out

    # -------------------------------------------------------------------- missing events

    def _detect_missing(self, clip, session, place_runs, seen_here: set[str]) -> None:
        """A labelled place in view for >= MISSING_PLACE_VISIBLE_S with a believed object absent.

        `seen_here` is the set of objects this clip matched. An object visible right now is not
        missing, however long its place was in view.
        """
        already_reported: set[str] = set()
        for run in place_runs:
            if run.place_id is None or (run.to_s - run.from_s) < config.MISSING_PLACE_VISIBLE_S:
                continue
            for obj in self.registry.at_place(run.place_id):
                object_id = obj["_id"]
                if object_id in already_reported or object_id in seen_here:
                    continue
                already_reported.add(object_id)
                t_missing = clip.time_at(run.to_s)
                log.info("%s believed at %s but not seen while that place was in view for %.1fs",
                         object_id, run.place_id, run.to_s - run.from_s)
                self.beliefs.went_missing(
                    object_id=object_id, place_id=run.place_id, t=t_missing,
                    session=clip.session, glasses=clip.glasses,
                    clip_path=clip.clip_path, offset_s=run.to_s,
                )
                self.counts["missing_events"] += 1
                self._try_retrieval(object_id, t_missing, clip)

    def _try_retrieval(self, object_id: str, t_missing: dt.datetime, clip) -> None:
        """If the object was last seen being picked up, ask retrieval where that motion went.

        Degrades honestly: a stub or unreachable store means no `matched` event and the belief
        stays on the last confirmed sighting. It never becomes a guess.
        """
        last = self.diary.last_event(object_id)
        if not last or last.get("event") != "picked_up":
            log.info("%s: last event is %s, not picked_up — no retrieval query",
                     object_id, last.get("event") if last else "none")
            return

        query_clip_path = last["clip"]["clip_path"]
        gone_at = float(last["clip"]["offset_s"])
        sighting = self.diary.last_confirmed_sighting(object_id)
        visible_at = float(sighting["clip"]["offset_s"]) if sighting else max(0.0, gone_at - 4.0)
        start, end = similar_moment.pad_window(min(visible_at, gone_at), max(visible_at, gone_at))

        results = similar_moment.query(query_clip_path, start, end)
        self.counts["retrieval_queries"] += 1
        if similar_moment.breaker.is_open:
            log.warning("%s: retrieval unavailable, answering from the last confirmed sighting", object_id)
            return
        best = similar_moment.best_match(results)
        if best is None:
            log.info("%s: no retrieval result cleared ELIDE_MIN_SIM — belief unchanged", object_id)
            return

        matched_place = self._place_at(best["clip_path"], float(best["offset_s"]))
        if matched_place is None:
            log.warning("%s: retrieval matched %s@%.1fs but our place recognizer did not "
                        "recognize that frame — no inference written",
                        object_id, best["clip_path"], best["offset_s"])
            return
        _, moved = self.beliefs.record_match(
            object_id=object_id, place_id=matched_place, similarity=float(best["similarity"]),
            t=t_missing, session=clip.session, glasses=clip.glasses,
            query_clip={"clip_path": query_clip_path, "start_s": start, "end_s": end},
            matched_clip={"clip_path": best["clip_path"], "offset_s": float(best["offset_s"])},
        )
        self.counts["matched_events"] += 1
        log.info("%s: inferred at %s (similarity %.3f, belief moved: %s)",
                 object_id, matched_place, float(best["similarity"]), moved)

    def _place_at(self, clip_path: str, offset_s: float) -> str | None:
        """Our own place recognizer on one frame of a clip. ElideDB never returns a place."""
        try:
            import cv2
        except ImportError:
            return None
        capture = cv2.VideoCapture(str(config.clip_fs_path(clip_path)))
        capture.set(cv2.CAP_PROP_POS_MSEC, offset_s * 1000.0)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            log.warning("could not read %s at %.2fs", clip_path, offset_s)
            return None
        return self.place_index.recognize(self.embedder.embed_one(frame)).place_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", help="session id as it appears in data/sessions.yaml, e.g. s1")
    parser.add_argument("--scene", action="append", help="limit to these scenes (repeatable)")
    parser.add_argument("--db", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the clips and their order; load no models and write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")

    try:
        session = timeutil.get_session(args.session)
    except timeutil.TimelineError as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 2

    clips = timeutil.all_clips(session)
    if args.scene:
        wanted = set(args.scene)
        unknown = wanted - set(timeutil.scene_names(session))
        if unknown:
            print(f"stopped: no such scene(s) {sorted(unknown)}; known: {timeutil.scene_names(session)}",
                  file=sys.stderr)
            return 2
        clips = [c for c in clips if c.scene in wanted]

    print(f"session {args.session}: {len(clips)} clip(s), processed in start_time order "
          f"(both phones on one timeline)")
    for clip in clips:
        print(f"  {clip.start_time.isoformat()}  {clip.glasses}  {clip.scene:<22} {clip.clip_path}")

    missing_files = [c for c in clips if not c.fs_path.exists()]

    if args.dry_run:
        # Reported, not fatal: a dry run is for checking the plan and the clip order, which is
        # useful on a machine that does not have the footage.
        if missing_files:
            print(f"\nnote: {len(missing_files)} of {len(clips)} clips are not on this machine:")
            for clip in missing_files:
                print(f"  {clip.clip_path} -> {clip.fs_path}")
        print("\n-- dry run, no models loaded, nothing written --")
        return 0

    if missing_files:
        print("\nstopped: these clips are not on this machine:", file=sys.stderr)
        for clip in missing_files:
            print(f"  {clip.clip_path} -> {clip.fs_path}", file=sys.stderr)
        print("Clips are copied off the phones to BOTH machines before anything else runs "
              "(see STARTUP.md).", file=sys.stderr)
        return 2

    database = dbmod.get_db(args.db)
    place_index = places.PlaceIndex.from_db(database)
    if not len(place_index):
        print("stopped: the places collection is empty. Run scripts/enroll_places.py first — "
              "without labelled places nothing can be given a location.", file=sys.stderr)
        return 2
    print(f"places enrolled: {place_index.place_ids}")

    pipeline = Pipeline(database, place_index)
    by_scene: dict[str, list[dict]] = collections.defaultdict(list)
    for clip in clips:
        by_scene[clip.scene].extend(pipeline.process_clip(clip, session))

    written = []
    for scene, records in by_scene.items():
        out = config.CACHE_DIR / args.session / f"{scene}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        written.append((out, len(records)))

    print("\n--- summary ---")
    print(pipeline.matcher.summary())
    counts = pipeline.counts
    print(f"tracks: {counts['tracks_seen']} seen, {counts['tracks_dropped']} dropped by the "
          f"persistence filter, {counts['tracks_joined']} joined to existing objects, "
          f"{counts['objects_created']} new objects")
    print(f"missing events: {counts['missing_events']}, retrieval queries: "
          f"{counts['retrieval_queries']}, matched events: {counts['matched_events']}")
    if counts["empty_crops"]:
        print(f"empty/off-frame crops skipped: {counts['empty_crops']}")
    print(similar_moment.breaker.summary())
    print("diary entries now:", pipeline.diary.count())
    for path, count in written:
        print(f"cache: {path.relative_to(config.REPO_ROOT)} ({count} tracks)")
    print("\nCommit the cache files — M4 and M5 read them instead of re-running the models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
