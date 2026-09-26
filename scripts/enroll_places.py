#!/usr/bin/env python
"""Fingerprint the labelled backdrops into the `places` collection. M1 pipeline step 1.

RUNS ON THE M5 ONLY — it loads DINOv2.

Reads `data/places/<place_id>/*.{mp4,mov,MOV}`, samples whole frames at PLACE_ENROLL_FPS, and
stores every frame's fingerprint under that place. `place_id` is the human-assigned label; the
directory name IS the label, so nothing here has to guess what a backdrop is.

Frames from both phones are kept as separate vectors rather than averaged: the same backdrop
from two angles is two clusters, and a mean of them matches neither well.

    python scripts/enroll_places.py                 # every place under data/places/
    python scripts/enroll_places.py kitchen_counter desk_drawer
    python scripts/enroll_places.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cyclops import config                       # noqa: E402
from cyclops.memory import db as dbmod           # noqa: E402
from cyclops.perception import embed             # noqa: E402

log = logging.getLogger("enroll_places")

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi"}


def sample_frames(clip: pathlib.Path, fps: float) -> list:
    """Whole frames at ~fps. Needs cv2, which ships with ultralytics on the M5."""
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "opencv is not installed here, so backdrop clips cannot be read.\n"
            "This is expected on the MSI: run this on the M5 (see STARTUP.md).\n"
            "On the M5: pip install -r requirements-vision.txt"
        ) from exc

    capture = cv2.VideoCapture(str(clip))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if source_fps <= 0:
        capture.release()
        raise SystemExit(f"cannot read the frame rate of {clip}; enrollment would sample blindly")
    stride = max(1, round(source_fps / fps))
    frames, index = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(frame)
        index += 1
    capture.release()
    log.info("%s: %d frames sampled (%.1f fps source, stride %d)", clip.name, len(frames), source_fps, stride)
    return frames


def place_dirs(root: pathlib.Path, only: list[str]) -> list[pathlib.Path]:
    if not root.exists():
        raise SystemExit(
            f"{root} does not exist. Backdrop clips are filmed and placed there by hand before "
            "enrollment (see the M1 subplan) — there is nothing to enroll yet."
        )
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if only:
        wanted = set(only)
        missing = wanted - {d.name for d in dirs}
        if missing:
            raise SystemExit(f"no such place directory: {sorted(missing)} under {root}")
        dirs = [d for d in dirs if d.name in wanted]
    return dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("places", nargs="*", help="place_ids to enroll (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="report what would be enrolled; write nothing")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")

    dirs = place_dirs(config.PLACES_DIR, args.places)
    if not dirs:
        raise SystemExit(f"no place directories under {config.PLACES_DIR}")

    plan: dict[str, list[pathlib.Path]] = {}
    for directory in dirs:
        clips = sorted(c for c in directory.iterdir() if c.suffix.lower() in VIDEO_SUFFIXES)
        if not clips:
            log.warning("%s has no backdrop clips — it can never be recognized", directory.name)
            continue
        plan[directory.name] = clips

    print(f"places to enroll: {len(plan)}")
    for place_id, clips in plan.items():
        print(f"  {place_id:<22} {len(clips)} clip(s): {[c.name for c in clips]}")
    if args.dry_run:
        print("\n-- dry run, nothing written --")
        return 0
    if not plan:
        return 1

    database = dbmod.get_db(args.db)
    embedder = embed.Embedder()
    now = dt.datetime.now(dt.timezone.utc)
    total_vectors = 0

    for place_id, clips in plan.items():
        vectors, sources = [], []
        for clip in clips:
            frames = sample_frames(clip, config.PLACE_ENROLL_FPS)
            if not frames:
                log.warning("%s: no frames decoded, skipped", clip)
                continue
            embedded = embedder.embed(frames)
            usable = [v for v in embedded if not embed.is_degenerate(v)]
            dropped = len(embedded) - len(usable)
            if dropped:
                log.warning("%s: %d/%d frames embedded to a zero vector and were dropped",
                            clip.name, dropped, len(embedded))
            vectors.extend(usable)
            sources.append(str(clip.relative_to(config.DATA_DIR)))

        if not vectors:
            log.error("%s: no usable fingerprints, NOT written — it cannot be recognized", place_id)
            continue

        # Replaces this place's document only. `places` is belief-side state, not the diary, so
        # re-enrolling is meant to overwrite; no other place is touched.
        database[dbmod.PLACES].replace_one(
            {"_id": place_id},
            {
                "_id": place_id,
                "place_id": place_id,
                "vecs": [[float(x) for x in v] for v in vectors],
                "n_frames": len(vectors),
                "sources": sources,
                "enrolled_at": now,
            },
            upsert=True,
        )
        total_vectors += len(vectors)
        print(f"  {place_id:<22} {len(vectors)} fingerprints written")

    print(f"\nenrolled {len(plan)} places, {total_vectors} fingerprints total")
    print("places collection now holds:",
          sorted(d["place_id"] for d in database[dbmod.PLACES].find({})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
