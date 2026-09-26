#!/usr/bin/env python
"""M1 pipeline step 0 — record the class names YOLOE prompt-free actually emits.

RUNS ON THE M5 ONLY. It loads YOLOE, which the MSI cannot do.

The point is to replace assumption with observation. `config.ignore_classes()` and
`config.hand_class_names()` are read from the file this writes, so the ignore list and the
hand class come from real model output. This script therefore reports candidates and
writes them as EMPTY lists for a human to fill in — it never guesses which class is a hand.

    python scripts/check_vocab.py data/clips/s1/A/IMG_0012.MOV
    python scripts/check_vocab.py <clip> --conf 0.25 --max-frames 200

Then: edit data/vocab/yoloe_classes.json, set `ignore_classes` and `hand_classes` from the
printed output, set `ignore_classes_reviewed: true`, and commit it.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cyclops import config  # noqa: E402

#: Printed as candidates for the operator to judge. NOT written to the JSON, and not
#: treated as answers — YOLOE's real vocabulary may use none of these words.
HAND_HINTS = ("hand", "finger", "palm", "arm", "wrist", "glove")
STRUCTURE_HINTS = (
    "wall", "floor", "ceiling", "table", "desk", "shelf", "cabinet", "counter", "countertop",
    "door", "window", "room", "curtain", "rug", "carpet", "person", "man", "woman", "people",
    "chair", "sofa", "couch", "bed", "stairs", "light", "lamp", "picture", "mirror",
)
DEMO_HINTS = ("key", "mug", "cup", "glass", "bottle", "wallet", "phone", "remote", "bowl")


def _load_yoloe():
    """Import ultralytics late, so `--help` works anywhere and the MSI gets a sentence
    instead of a traceback."""
    try:
        from ultralytics import YOLOE  # type: ignore
        return YOLOE
    except ImportError:
        pass
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed here, so YOLOE cannot run.\n"
            "This is expected on the MSI: run this script on the M5 (see STARTUP.md).\n"
            "On the M5: pip install -r requirements-vision.txt"
        ) from exc


def _stride_for(clip: pathlib.Path, target_fps: float, override: int | None) -> tuple[int, float | None]:
    """Frames to skip so we sample at ~target_fps. Falls back to a stated default rather
    than pretending to know the clip's frame rate."""
    if override:
        return override, None
    try:
        import cv2  # ships with ultralytics
    except ImportError:
        return 6, None
    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    capture.release()
    if fps <= 0:
        return 6, None
    return max(1, round(fps / target_fps)), fps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="one clip to probe (any session/phone)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="detection confidence floor for the probe (default 0.25, "
                             "deliberately below PERSIST_MIN_CONF so near-misses are visible)")
    parser.add_argument("--max-frames", type=int, default=300, help="stop after this many sampled frames")
    parser.add_argument("--stride", type=int, default=None, help="override frame stride")
    parser.add_argument("--out", default=str(config.VOCAB_PATH), help="output JSON path")
    args = parser.parse_args()

    clip = pathlib.Path(args.clip)
    if not clip.exists():
        print(f"clip not found: {clip}", file=sys.stderr)
        return 2

    YoloClass = _load_yoloe()
    print(f"loading {config.YOLOE_WEIGHTS} ...")
    model = YoloClass(config.YOLOE_WEIGHTS)

    stride, fps = _stride_for(clip, config.SAMPLE_FPS, args.stride)
    print(f"clip fps: {fps if fps else 'unknown'} | stride: {stride} "
          f"(target {config.SAMPLE_FPS} fps) | conf floor: {args.conf}")

    vocabulary = dict(getattr(model, "names", {}) or {})
    emitted: collections.Counter[str] = collections.Counter()
    best_conf: dict[str, float] = {}
    frames = 0

    for result in model.predict(source=str(clip), stream=True, conf=args.conf,
                                vid_stride=stride, verbose=False):
        frames += 1
        names = getattr(result, "names", vocabulary) or vocabulary
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                name = names.get(int(cls_id), f"class_{int(cls_id)}")
                emitted[name] += 1
                best_conf[name] = max(best_conf.get(name, 0.0), float(conf))
        if frames >= args.max_frames:
            break

    print(f"\nsampled {frames} frames; {len(emitted)} distinct classes emitted "
          f"(model vocabulary: {len(vocabulary)} classes)\n")
    print(f"{'class name':<34}{'hits':>7}{'best conf':>11}")
    for name, count in emitted.most_common():
        print(f"{name:<34}{count:>7}{best_conf[name]:>11.3f}")

    def candidates(hints: tuple[str, ...]) -> list[str]:
        return sorted({n for n in emitted for h in hints if h in n.lower()})

    print("\n--- candidates for you to judge (NOT written to the file) ---")
    print(f"hand-like      : {candidates(HAND_HINTS) or 'NONE FOUND'}")
    print(f"structural     : {candidates(STRUCTURE_HINTS) or 'none'}")
    print(f"demo objects   : {candidates(DEMO_HINTS) or 'NONE FOUND'}")

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "clip": str(clip),
        "weights": config.YOLOE_WEIGHTS,
        "frames_sampled": frames,
        "stride": stride,
        "clip_fps": fps,
        "conf_floor": args.conf,
        "model_vocabulary_size": len(vocabulary),
        "class_names": [name for name, _ in emitted.most_common()],
        "detection_counts": dict(emitted.most_common()),
        "best_conf": {k: round(v, 4) for k, v in sorted(best_conf.items())},
        "all_model_classes": sorted(vocabulary.values()),
        # Filled in BY HAND from the output above. Left empty on purpose.
        "ignore_classes": [],
        "hand_classes": [],
        "ignore_classes_reviewed": False,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {out}")
    print("NEXT: set ignore_classes and hand_classes from the list above, set")
    print("ignore_classes_reviewed: true, and commit the file. Perception refuses to run")
    print("until then (config.require_vocab).")
    if not candidates(HAND_HINTS):
        print("\nNOTE: no hand-like class was emitted. Carried-detection depends on a hand")
        print("box, so report this back to Track A before processing a session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
