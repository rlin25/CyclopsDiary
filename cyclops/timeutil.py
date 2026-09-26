"""The shared clock.

Absolute time of a frame = clip `creation_time` (ffprobe) + that phone's session offset
(from the sync shot) + seconds into the clip.

Processing time and upload time are never used: clips come off the phones by cable or
AirDrop after recording, so their mtime says nothing about when the scene happened. Two
phones feeding one memory only works if both are placed on this one corrected timeline.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import shutil
import subprocess

import yaml

from cyclops import config

GLASSES = ("A", "B")


class TimelineError(RuntimeError):
    """Something makes a frame's absolute time unknowable. Never guessed around."""


# --------------------------------------------------------------------------------- ffprobe

def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def clip_creation_time(clip_path: str | pathlib.Path) -> dt.datetime:
    """Wall-clock start of a clip, from its container metadata, as an aware UTC datetime.

    Raises rather than falling back to file mtime: a silently wrong clock would corrupt
    every diary timestamp downstream, and a wrong timeline is worse than a stopped build.
    """
    path = pathlib.Path(clip_path)
    if not path.exists():
        raise TimelineError(f"clip not found: {path}")
    if not ffprobe_available():
        raise TimelineError(
            "ffprobe is not installed, so clip creation_time cannot be read. Install ffmpeg "
            "on the machine that processes clips."
        )
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_entries", "format_tags=creation_time:stream_tags=creation_time", str(path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise TimelineError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    probed = json.loads(proc.stdout or "{}")
    stamps = []
    fmt_tags = (probed.get("format") or {}).get("tags") or {}
    if fmt_tags.get("creation_time"):
        stamps.append(fmt_tags["creation_time"])
    for stream in probed.get("streams") or []:
        tag = (stream.get("tags") or {}).get("creation_time")
        if tag:
            stamps.append(tag)
    if not stamps:
        raise TimelineError(
            f"{path} has no creation_time tag. Its absolute time is unknown — do not process "
            "it until the real recording time is recovered."
        )
    return parse_iso_utc(stamps[0])


def parse_iso_utc(stamp: str) -> dt.datetime:
    """Parse an ISO-8601 stamp to an aware UTC datetime. A naive stamp is read as UTC,
    which is what ffprobe emits for phone footage."""
    text = stamp.strip().replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


# -------------------------------------------------------------------------- sessions.yaml

def load_sessions(path: str | pathlib.Path | None = None) -> dict[str, dict]:
    """Read `data/sessions.yaml` into {session_id: session_dict}.

    Accepts the three shapes the file might reasonably be hand-written in — a single
    session document, a list of them, or a mapping under `sessions:` — because it is
    written by hand between takes and reformatting it later would silently break
    processing.
    """
    path = pathlib.Path(path or config.SESSIONS_PATH)
    if not path.exists():
        raise TimelineError(
            f"{path} not found. It is written by hand before processing (see the M1 subplan)."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    return sessions


def get_session(session_id: str, path: str | pathlib.Path | None = None) -> dict:
    sessions = load_sessions(path)
    if session_id not in sessions:
        raise TimelineError(
            f"session {session_id!r} is not in {path or config.SESSIONS_PATH}; "
            f"known sessions: {sorted(sessions)}"
        )
    return sessions[session_id]


def session_offset_s(session: dict, glasses: str) -> float:
    """That phone's clock offset in seconds, measured from the sync shot.

    A missing offset is an error, not a zero: assuming zero would misalign the two phones
    by an unknown amount and quietly break cross-camera reasoning, which is the whole demo.
    """
    if glasses not in GLASSES:
        raise TimelineError(f"unknown glasses {glasses!r}; expected one of {GLASSES}")
    offsets = session.get("offset_s")
    if not isinstance(offsets, dict) or glasses not in offsets:
        raise TimelineError(
            f"session {session.get('session')!r} has no offset_s for glasses {glasses!r}. "
            "Read it off the sync shot before processing — it cannot be assumed to be 0."
        )
    return float(offsets[glasses])


# --------------------------------------------------------------------------- the one rule

def absolute_time(clip_start: dt.datetime, offset_s: float, seconds_into_clip: float) -> dt.datetime:
    """clip creation_time + phone session offset + seconds into the clip."""
    if clip_start.tzinfo is None:
        raise TimelineError("clip_start must be timezone-aware")
    return clip_start + dt.timedelta(seconds=float(offset_s) + float(seconds_into_clip))


def frame_time(
    clip_path: str | pathlib.Path,
    session: dict,
    glasses: str,
    seconds_into_clip: float,
) -> dt.datetime:
    """Convenience wrapper: the absolute time of one sampled frame."""
    return absolute_time(
        clip_creation_time(clip_path),
        session_offset_s(session, glasses),
        seconds_into_clip,
    )
