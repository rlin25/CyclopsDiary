"""YOLOE prompt-free detection + BoT-SORT tracking, and the persistence filter.

The model half runs on the M5. The geometry and the filter are pure Python and run anywhere,
which matters because the persistence filter is the thing standing between phantom boxes and
MongoDB — it deserves tests that do not need a GPU.

Class names come from `data/vocab/yoloe_classes.json`, written by the real vocabulary check
(`scripts/check_vocab.py`). Nothing here hardcodes a class name.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Iterator, Sequence

from cyclops import config

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclasses.dataclass(frozen=True)
class Detection:
    offset_s: float
    track_id: int
    class_name: str
    conf: float
    bbox: Box


@dataclasses.dataclass
class FrameDetections:
    """One sampled frame. Hands are kept apart from objects: a hand is evidence about
    carrying, never a tracked object in its own right."""

    offset_s: float
    objects: list[Detection] = dataclasses.field(default_factory=list)
    hands: list[Box] = dataclasses.field(default_factory=list)
    #: The decoded frame (HWC BGR), carried so whole-frame and crop embeddings come from the
    #: same decode pass. Consumers embed and drop it — never accumulate frames for a clip.
    image: Any | None = None

    def crop(self, bbox: Box, pad: int = 0) -> Any | None:
        """Pixels inside `bbox`, or None if the box is empty or off-frame.

        Returning None rather than a 0x0 array is deliberate: an empty crop embeds to a zero
        vector, and the matcher's degenerate path should be reserved for genuinely blurred
        content, not for arithmetic that could have been caught here.
        """
        if self.image is None:
            return None
        height, width = self.image.shape[:2]
        x1 = max(0, int(bbox[0]) - pad)
        y1 = max(0, int(bbox[1]) - pad)
        x2 = min(width, int(bbox[2]) + pad)
        y2 = min(height, int(bbox[3]) + pad)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return self.image[y1:y2, x1:x2]


@dataclasses.dataclass
class Track:
    track_id: int
    class_name: str
    detections: list[Detection] = dataclasses.field(default_factory=list)

    @property
    def first_seen_s(self) -> float:
        return min(d.offset_s for d in self.detections)

    @property
    def last_seen_s(self) -> float:
        return max(d.offset_s for d in self.detections)

    @property
    def best_conf(self) -> float:
        return max(d.conf for d in self.detections)


# --------------------------------------------------------------------------------- geometry

def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def boxes_overlap(a: Box, b: Box) -> bool:
    """Any pixel overlap at all. Deliberately not an IoU threshold: a hand is much smaller
    than most furniture and comparable to a set of keys, so IoU would be dominated by the
    size difference rather than by contact."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


# ------------------------------------------------------------------------ persistence filter

def group_tracks(frames: Sequence[FrameDetections]) -> dict[int, Track]:
    tracks: dict[int, Track] = {}
    for frame in frames:
        for det in frame.objects:
            track = tracks.get(det.track_id)
            if track is None:
                track = tracks[det.track_id] = Track(det.track_id, det.class_name)
            track.detections.append(det)
    return tracks


def persistence_filter(tracks: dict[int, Track]) -> tuple[dict[int, Track], dict[int, str]]:
    """Keep only tracks seen in >= PERSIST_MIN_FRAMES sampled frames with conf >= PERSIST_MIN_CONF.

    Anything dropped here never reaches MongoDB. The reasons are returned rather than logged
    and forgotten: if this filter is eating real objects, the counts are how you find out.
    """
    kept: dict[int, Track] = {}
    dropped: dict[int, str] = {}
    for track_id, track in tracks.items():
        confident = [d for d in track.detections if d.conf >= config.PERSIST_MIN_CONF]
        if len(confident) < config.PERSIST_MIN_FRAMES:
            dropped[track_id] = (
                f"{track.class_name}: {len(confident)} frames at conf>={config.PERSIST_MIN_CONF} "
                f"(needs {config.PERSIST_MIN_FRAMES}; {len(track.detections)} total, "
                f"best conf {track.best_conf:.2f})"
            )
            continue
        kept[track_id] = Track(track_id, track.class_name, confident)
    if dropped:
        log.info("persistence filter dropped %d/%d tracks", len(dropped), len(tracks))
    return kept, dropped


def hand_overlap_seconds(track: Track, frames: Sequence[FrameDetections], sample_period_s: float) -> float:
    """How long this track's box touched a hand box, in the wearer's own view.

    Counted per sampled frame rather than as a continuous span: at 5 fps a real handover
    flickers in and out of overlap, and requiring strict continuity would miss it.
    """
    hands_by_offset = {round(f.offset_s, 3): f.hands for f in frames if f.hands}
    if not hands_by_offset:
        return 0.0
    touching = 0
    for det in track.detections:
        for hand in hands_by_offset.get(round(det.offset_s, 3), ()):
            if boxes_overlap(det.bbox, hand):
                touching += 1
                break
    return touching * sample_period_s


# ------------------------------------------------------------------------ the model (M5 only)

class Detector:
    """YOLOE prompt-free + BoT-SORT. Loads lazily; importing this module is free on the MSI."""

    def __init__(self, weights: str | None = None):
        self.weights = weights or config.YOLOE_WEIGHTS
        self._model = None
        self.vocab = config.require_vocab()
        self.ignore = config.ignore_classes()
        self.hand_classes = set(config.hand_class_names())
        if not self.hand_classes:
            log.warning(
                "no hand class recorded in %s — carried-detection by hand overlap is disabled, "
                "and carrying will only be inferred from a change of recognized place",
                config.VOCAB_PATH,
            )

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            try:
                from ultralytics import YOLOE as Model  # type: ignore
            except ImportError:
                from ultralytics import YOLO as Model  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed here, so YOLOE cannot run.\n"
                "This is expected on the MSI: run this on the M5 (see docs/STARTUP.md).\n"
                "On the M5: pip install -r requirements-vision.txt"
            ) from exc
        log.info("loading %s", self.weights)
        self._model = Model(self.weights)
        return self._model

    def clip_fps(self, clip_fs_path) -> float | None:
        try:
            import cv2
        except ImportError:
            return None
        capture = cv2.VideoCapture(str(clip_fs_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        capture.release()
        return fps or None

    def track_clip(self, clip_fs_path, sample_fps: float | None = None) -> Iterator[FrameDetections]:
        """Yield one FrameDetections per sampled frame, at ~sample_fps.

        Ignored classes are dropped here so they never become tracks; hand boxes are pulled
        aside for carried-detection.
        """
        model = self._load()
        target = sample_fps or config.SAMPLE_FPS
        fps = self.clip_fps(clip_fs_path)
        if fps is None:
            raise RuntimeError(
                f"cannot read the frame rate of {clip_fs_path}; without it, a frame's offset "
                "into the clip is unknown and every timestamp downstream would be wrong"
            )
        stride = max(1, round(fps / target))
        period = stride / fps
        log.info("%s: %.2f fps, stride %d -> %.2f fps sampled", clip_fs_path, fps, stride, 1.0 / period)

        for index, result in enumerate(model.track(
            source=str(clip_fs_path), tracker="botsort.yaml", persist=True,
            stream=True, conf=config.PERSIST_MIN_CONF, vid_stride=stride, verbose=False,
        )):
            frame = FrameDetections(offset_s=round(index * period, 3),
                                    image=getattr(result, "orig_img", None))
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                ids = boxes.id.tolist() if getattr(boxes, "id", None) is not None else [None] * len(boxes)
                for track_id, cls_id, conf, xyxy in zip(
                    ids, boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
                ):
                    class_name = names.get(int(cls_id), f"class_{int(cls_id)}")
                    box: Box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
                    if class_name in self.hand_classes:
                        frame.hands.append(box)
                        continue
                    if class_name in self.ignore:
                        continue
                    if track_id is None:
                        # BoT-SORT gives no id until a detection is confirmed across frames;
                        # an untracked box cannot become an object, so it is skipped rather
                        # than given a synthetic id that would fragment identities.
                        continue
                    frame.objects.append(Detection(
                        offset_s=frame.offset_s, track_id=int(track_id),
                        class_name=class_name, conf=float(conf), bbox=box,
                    ))
            yield frame
