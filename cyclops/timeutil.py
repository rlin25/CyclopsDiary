"""The shared clock, and the reader for `data/sessions.yaml`.

    absolute time of a frame = that clip's hand-entered `start_time` + seconds into the clip

`start_time` is read by hand off an on-screen clock filmed in the first couple of seconds of
each clip, and typed into `sessions.yaml`. It is not the container's `creation_time`: phone
clocks were never trusted, which is what the sync shot exists to correct. The correction is
already baked in at the moment a human types the value, so there is no ffprobe call here and
no separate per-phone offset to add.

`sessions.yaml` is the single source of truth for clip paths. A label file that disagrees is
an error, not something to reconcile.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib

import yaml

from cyclops import config

GLASSES = ("A", "B")


class TimelineError(RuntimeError):
    """Something makes a frame's absolute time, or a clip's identity, unknowable.

    Always raised rather than worked around: a wrong clock or a wrong clip path corrupts
    every answer downstream, and a stopped build is cheaper than a plausible wrong demo.
    """


@dataclasses.dataclass(frozen=True)
class Clip:
    """One clip of one scene, from one pair of glasses."""

    session: str
    scene: str
    glasses: str
    clip_path: str          # relative to data/, e.g. "clips/s1/B/IMG_0450.MOV"
    start_time: dt.datetime  # timezone-aware, hand-entered

    def time_at(self, seconds_into_clip: float) -> dt.datetime:
        return absolute_time(self.start_time, seconds_into_clip)

    @property
    def fs_path(self) -> pathlib.Path:
        return config.clip_fs_path(self.clip_path)


# ------------------------------------------------------------------------- the one rule

def absolute_time(clip_start: dt.datetime, seconds_into_clip: float) -> dt.datetime:
    """clip start_time + seconds into the clip. That is the whole conversion."""
    if clip_start.tzinfo is None:
        raise TimelineError("clip start_time must be timezone-aware")
    return clip_start + dt.timedelta(seconds=float(seconds_into_clip))


def parse_start_time(value: object, where: str) -> dt.datetime:
    """Parse a hand-entered ISO-8601 stamp that MUST carry an explicit UTC offset.

    A naive stamp is refused rather than assumed to be UTC or local. These values are typed
    by hand off a clock on screen, and the one mistake that silently shifts the whole shared
    timeline by hours is omitting the offset.
    """
    if value is None:
        raise TimelineError(
            f"{where}: start_time is missing. Read it off the on-screen clock in the clip's "
            "first seconds and enter it by hand — there is no fallback."
        )
    if isinstance(value, dt.datetime):
        # PyYAML parses unquoted timestamps into datetime itself.
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise TimelineError(
                f"{where}: start_time {value!r} is not ISO 8601. Expected e.g. "
                '"2026-09-26T09:01:58-04:00".'
            ) from exc
    if parsed.tzinfo is None:
        raise TimelineError(
            f"{where}: start_time {value!r} has no UTC offset. Write it explicitly (e.g. "
            '"...T09:01:58-04:00"); it is not assumed to be UTC or local time.'
        )
    return parsed.astimezone(dt.timezone.utc)


# ------------------------------------------------------------------------ sessions.yaml

def load_sessions(path: str | pathlib.Path | None = None) -> dict[str, dict]:
    """Read `data/sessions.yaml` into {session_id: raw session dict}, validating as it goes.

    Accepts a single session document, a list of them, or a mapping keyed by session id,
    because the file is hand-written between takes and reformatting it later should not
    break processing.
    """
    path = pathlib.Path(path or config.SESSIONS_PATH)
    if not path.exists():
        raise TimelineError(
            f"{path} not found. It is written by hand before processing (see the M1 subplan)."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        # This file is hand-written between takes; a typo should say where, not raise a
        # parser traceback at whoever is mid-session.
        raise TimelineError(f"{path} is not valid YAML:\n{exc}") from exc
    if raw is None:
        raise TimelineError(f"{path} is empty")

    if isinstance(raw, dict) and "sessions" in raw:
        raw = raw["sessions"]

    sessions: dict[str, dict] = {}
    if isinstance(raw, dict) and "session" in raw:
        sessions[str(raw["session"])] = raw
    elif isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict) or "session" not in entry:
                raise TimelineError(f"{path}: every session entry needs a `session:` key")
            sessions[str(entry["session"])] = entry
    elif isinstance(raw, dict):
        for key, entry in raw.items():
            if not isinstance(entry, dict):
                raise TimelineError(f"{path}: session {key!r} is not a mapping")
            sessions[str(key)] = {**entry, "session": str(key)}
    else:
        raise TimelineError(f"{path}: unrecognised structure (expected a session or a list)")
    if not sessions:
        raise TimelineError(f"{path}: no sessions found")

    for session_id, session in sessions.items():
        _validate_session(session_id, session, path)
    return sessions


def get_session(session_id: str, path: str | pathlib.Path | None = None) -> dict:
    sessions = load_sessions(path)
    if session_id not in sessions:
        raise TimelineError(
            f"session {session_id!r} is not in {path or config.SESSIONS_PATH}; "
            f"known sessions: {sorted(sessions)}"
        )
    return sessions[session_id]


def _validate_session(session_id: str, session: dict, path: pathlib.Path) -> None:
    if "offset_s" in session:
        raise TimelineError(
            f"{path}: session {session_id!r} still has `offset_s`. It is obsolete — each "
            "clip's hand-entered start_time already includes the sync-shot correction. "
            "Remove it rather than leaving a value nothing reads."
        )

    wearers = session.get("glasses")
    if not isinstance(wearers, dict) or not wearers:
        raise TimelineError(
            f"{path}: session {session_id!r} needs `glasses:` with a wearer per phone, e.g. "
            "glasses: {A: {wearer: Richard}, B: {wearer: Sam}}"
        )
    for glasses, entry in wearers.items():
        if glasses not in GLASSES:
            raise TimelineError(f"{path}: session {session_id!r} has unknown glasses {glasses!r}")
        wearer = entry.get("wearer") if isinstance(entry, dict) else entry
        if not wearer:
            raise TimelineError(
                f"{path}: session {session_id!r} glasses {glasses!r} has no wearer name. "
                "Answers say who saw something, so the name cannot be blank."
            )

    scenes = session.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise TimelineError(f"{path}: session {session_id!r} needs a non-empty `scenes:` list")

    # One clip path must carry one start_time, wherever it appears.
    seen: dict[str, tuple[str, dt.datetime]] = {}
    for scene in scenes:
        if not isinstance(scene, dict) or not scene.get("name"):
            raise TimelineError(f"{path}: session {session_id!r} has a scene with no `name:`")
        scene_name = str(scene["name"])
        clips = scene.get("clips")
        if not isinstance(clips, dict) or not clips:
            raise TimelineError(
                f"{path}: scene {scene_name!r} needs `clips:` with at least one of A/B. "
                "A phone that did not film the scene is simply omitted."
            )
        for glasses, clip in clips.items():
            if glasses not in GLASSES:
                raise TimelineError(f"{path}: scene {scene_name!r} has unknown glasses {glasses!r}")
            where = f"{path}: session {session_id!r} scene {scene_name!r} glasses {glasses!r}"
            if not isinstance(clip, dict):
                raise TimelineError(f"{where}: expected a mapping with `path:` and `start_time:`")
            if "offset_s" in clip:
                raise TimelineError(f"{where}: `offset_s` is obsolete; remove it")
            clip_path = clip.get("path")
            if not clip_path:
                raise TimelineError(f"{where}: no `path:`")
            start = parse_start_time(clip.get("start_time"), where)
            clip_path = str(clip_path)
            if clip_path in seen:
                prior_scene, prior_start = seen[clip_path]
                if prior_start != start:
                    raise TimelineError(
                        f"{path}: clip {clip_path!r} has two different start_time values — "
                        f"scene {prior_scene!r} says {prior_start.isoformat()}, scene "
                        f"{scene_name!r} says {start.isoformat()}. One clip started at one "
                        "time; fix whichever is wrong."
                    )
            else:
                seen[clip_path] = (scene_name, start)


# ------------------------------------------------------------------------ reading scenes

def wearer(session: dict, glasses: str) -> str:
    if glasses not in GLASSES:
        raise TimelineError(f"unknown glasses {glasses!r}; expected one of {GLASSES}")
    entry = (session.get("glasses") or {}).get(glasses)
    if entry is None:
        raise TimelineError(
            f"session {session.get('session')!r} has no glasses {glasses!r}"
        )
    return entry["wearer"] if isinstance(entry, dict) else str(entry)


def scene_names(session: dict) -> list[str]:
    return [str(scene["name"]) for scene in session["scenes"]]


def scene_clips(session: dict, scene_name: str) -> dict[str, Clip]:
    """{glasses: Clip} for one scene. A phone that did not film it is simply absent."""
    session_id = str(session["session"])
    for scene in session["scenes"]:
        if str(scene["name"]) != scene_name:
            continue
        out: dict[str, Clip] = {}
        for glasses, clip in scene["clips"].items():
            where = f"session {session_id!r} scene {scene_name!r} glasses {glasses!r}"
            out[glasses] = Clip(
                session=session_id,
                scene=scene_name,
                glasses=glasses,
                clip_path=str(clip["path"]),
                start_time=parse_start_time(clip.get("start_time"), where),
            )
        return out
    raise TimelineError(
        f"scene {scene_name!r} is not in session {session_id!r}; known scenes: {scene_names(session)}"
    )


def all_clips(session: dict) -> list[Clip]:
    """Every clip of the session, in ascending start_time — the merged two-phone timeline.

    Both phones feed one memory, so processing follows this order rather than finishing one
    phone before starting the other. Cross-camera reasoning (A asking about what B saw)
    depends on events arriving in real-world order.
    """
    clips = [clip for name in scene_names(session) for clip in scene_clips(session, name).values()]
    return sorted(clips, key=lambda c: (c.start_time, c.glasses, c.clip_path))


def check_label_clips(session: dict, scene_name: str, label_glasses: dict, label_file: str) -> None:
    """Fail loudly if a label file's `glasses:` paths disagree with sessions.yaml.

    sessions.yaml is the single source of truth. A mismatch means one of the two files is
    stale, and guessing which would attach ground truth to the wrong footage.
    """
    expected = {g: c.clip_path for g, c in scene_clips(session, scene_name).items()}
    for glasses, label_path in (label_glasses or {}).items():
        if glasses not in expected:
            raise TimelineError(
                f"{label_file}: scene {scene_name!r} labels glasses {glasses!r}, but "
                f"{config.SESSIONS_PATH} has no clip for it (phones present: {sorted(expected)})"
            )
        if str(label_path) != expected[glasses]:
            raise TimelineError(
                f"{label_file}: scene {scene_name!r} glasses {glasses!r} disagrees with "
                f"{config.SESSIONS_PATH}\n"
                f"  label file says:   {label_path}\n"
                f"  sessions.yaml says: {expected[glasses]}\n"
                "sessions.yaml is the single source of truth; fix the label file."
            )
