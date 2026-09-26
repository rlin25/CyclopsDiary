"""Environment access and every tunable in one place.

Two rules this module exists to enforce:

1. Nothing here reads a credential at import time. Importing `config` must work on a
   machine with no `.env` at all, so that `pytest` runs offline and a missing key surfaces
   as a clear error at the point of use, naming the build step it blocks.
2. Thresholds are never written inline at a call site. M4 tunes FADE_HALF_LIFE_S, and the
   load test overrides the database, so both must be reachable without editing code.
"""

from __future__ import annotations

import json
import os
import pathlib
import threading

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CLIPS_DIR = DATA_DIR / "clips"
LABELS_DIR = DATA_DIR / "labels"
PLACES_DIR = DATA_DIR / "places"
SESSIONS_PATH = DATA_DIR / "sessions.yaml"
VOCAB_PATH = DATA_DIR / "vocab" / "yoloe_classes.json"


class MissingCredential(RuntimeError):
    """A required environment value is absent. Carries what it blocks, so the operator
    knows which build step just stopped and why."""


# --------------------------------------------------------------------------- env loading

_env_file_cache: dict[str, str] | None = None
_env_lock = threading.Lock()


def _parse_env_file(path: pathlib.Path) -> dict[str, str]:
    """Minimal `.env` reader. Deliberately not python-dotenv: one fewer dependency, and
    the format we need is `KEY=value` with `#` comments and optional surrounding quotes."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _env_file() -> dict[str, str]:
    global _env_file_cache
    if _env_file_cache is None:
        with _env_lock:
            if _env_file_cache is None:
                _env_file_cache = _parse_env_file(ENV_PATH)
    return _env_file_cache


def reload_env() -> None:
    """Drop the cached `.env`. For tests, and for the operator filling in a key mid-session."""
    global _env_file_cache
    with _env_lock:
        _env_file_cache = None


def env(name: str, default: str | None = None) -> str | None:
    """Real environment wins over `.env`, so a one-off override needs no file edit.
    An empty value is treated as absent — a declared-but-blank key is not a value."""
    value = os.environ.get(name)
    if value is None or value == "":
        value = _env_file().get(name)
    if value is None or value == "":
        return default
    return value


def required_env(name: str, blocks: str) -> str:
    value = env(name)
    if value is None:
        raise MissingCredential(
            f"{name} is not set in the environment or {ENV_PATH}. This blocks: {blocks}."
        )
    return value


def env_status(names: tuple[str, ...]) -> dict[str, bool]:
    """Presence only, never values — safe to print in a setup report."""
    return {name: env(name) is not None for name in names}


def _float_env(name: str, default: float) -> float:
    value = env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {value!r}") from exc


def _int_env(name: str, default: int) -> int:
    value = env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


# ------------------------------------------------------------------------------- MongoDB

DEFAULT_DB = "cyclops"
LOADTEST_DB = "cyclops_loadtest"

#: The Atlas cluster is shared with unrelated databases. Nothing in this codebase may
#: touch a database outside this set, and nothing may drop anything at all.
ALLOWED_DATABASES = frozenset({DEFAULT_DB, LOADTEST_DB})


def mongodb_uri() -> str:
    return required_env("MONGODB_URI", "scripts/setup_db.py and every memory-layer write")


def mongodb_db_name() -> str:
    return env("MONGODB_DB", DEFAULT_DB)


# ----------------------------------------------------------------------- Clips on disk
# Clips are served by the app's own static mount, not by any external service. There are
# no credentials and no expiry: a clip_path is valid for as long as the file is on disk.

#: URL prefix the FastAPI app mounts `data/clips/` under.
CLIPS_MOUNT = "/clips"


def clip_fs_path(clip_path: str) -> pathlib.Path:
    """Resolve a diary entry's `clip_path` (e.g. "clips/s1/B/IMG_0450.MOV") to a file.

    `clip_path` is relative to `data/`, and is refused if it escapes it — a diary entry is
    data, and data must not be able to name a file outside the clips tree.
    """
    resolved = (DATA_DIR / clip_path).resolve()
    clips_root = CLIPS_DIR.resolve()
    if resolved != clips_root and clips_root not in resolved.parents:
        raise ValueError(f"clip_path {clip_path!r} resolves outside {clips_root}")
    return resolved


def clip_url(clip_path: str) -> str:
    """The browser-facing URL for a clip. Built at call time from `clip_path`; nothing is
    stored and nothing expires."""
    return f"{CLIPS_MOUNT}/{str(clip_path).removeprefix('clips/').lstrip('/')}"


# ---------------------------------------------------------------------------- Agent / LLM

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_api_key() -> str:
    return required_env("OPENROUTER_API_KEY", "the agent loop, scripts/ask.py, eval/score.py")


def agent_model() -> str:
    """Never defaulted. A wrong-but-plausible slug fails as a confusing 404 mid-demo, so an
    unset model is an error, not a guess."""
    return required_env("AGENT_MODEL", "the agent loop, scripts/ask.py, eval/score.py")


#: Core Principle 6: a request that hangs is worse than one that fails.
AGENT_TIMEOUT_S = _float_env("AGENT_TIMEOUT_S", 30.0)
#: Hard cap on tool-call rounds, so a confused model cannot loop forever.
AGENT_MAX_TOOL_ROUNDS = _int_env("AGENT_MAX_TOOL_ROUNDS", 6)
#: "After 3 consecutive failures of an external service, stop calling it for 60 seconds."
BREAKER_MAX_FAILURES = _int_env("BREAKER_MAX_FAILURES", 3)
BREAKER_COOLDOWN_S = _float_env("BREAKER_COOLDOWN_S", 60.0)


# ------------------------------------------------------------------------------ Retrieval

#: Caller-side threshold for find_similar_moment() results (INTERFACES.md §1).
ELIDE_MIN_SIM = _float_env("ELIDE_MIN_SIM", 0.35)
#: ElideDB's own minimum clip length. Windows shorter than this are padded, never sent.
ELIDE_MIN_WINDOW_S = 4.0
ELIDE_TOP_K = _int_env("ELIDE_TOP_K", 5)


# ----------------------------------------------------------------------------- Perception

SAMPLE_FPS = _float_env("SAMPLE_FPS", 5.0)
PLACE_ENROLL_FPS = _float_env("PLACE_ENROLL_FPS", 1.0)
EMBED_DIM = 384  # DINOv2-small CLS token
YOLOE_WEIGHTS = env("YOLOE_WEIGHTS", "yoloe-11s-seg-pf.pt")
DINOV2_MODEL = env("DINOV2_MODEL", "facebook/dinov2-small")

#: Persistence filter — a track must clear both to reach MongoDB (M1 subplan step 3).
PERSIST_MIN_FRAMES = _int_env("PERSIST_MIN_FRAMES", 5)
PERSIST_MIN_CONF = _float_env("PERSIST_MIN_CONF", 0.35)

#: Object box must overlap a hand box for this long to count as carried (step 5).
CARRY_MIN_OVERLAP_S = _float_env("CARRY_MIN_OVERLAP_S", 1.0)
#: A labeled place must be recognized this long before a missing object counts as missing.
MISSING_PLACE_VISIBLE_S = _float_env("MISSING_PLACE_VISIBLE_S", 3.0)


# ------------------------------------------------------------- Thresholds (M4 tunes these)
# STARTING VALUES, not measurements. Nothing in the planning docs specifies numbers for
# these; they need a look at real Session 1 output before the demo.

#: Zone 1 — filtered $vectorSearch within the current place.
T_PLACE = _float_env("T_PLACE", 0.75)
#: Zone 2 — unfiltered $vectorSearch across all places. Higher bar: no place agreement.
T_HIGH = _float_env("T_HIGH", 0.85)
#: Whole-frame place recognition; below this the frame has no recognized place.
PLACE_MIN_SIM = _float_env("PLACE_MIN_SIM", 0.80)

#: Belief confidence fade: conf = exp(-ln2 * dt / FADE_HALF_LIFE_S).
FADE_HALF_LIFE_S = _float_env("FADE_HALF_LIFE_S", 1800.0)
#: Below this, status becomes `faded`. Above it, fade changes wording only, never status.
FADE_FLOOR = _float_env("FADE_FLOOR", 0.25)

#: Fingerprints per object, diverse angles (IMPLEMENTATION_STRATEGY.md).
MAX_FINGERPRINTS_PER_OBJECT = _int_env("MAX_FINGERPRINTS_PER_OBJECT", 12)
#: A new fingerprint is stored only if it is at least this different from every existing
#: one, so the 12 slots hold varied angles instead of 12 near-identical frames.
FINGERPRINT_MAX_SIM = _float_env("FINGERPRINT_MAX_SIM", 0.95)


# ----------------------------------------------------------------------- Scoring (M4/M5)

SCORE_RIGHT = 1
SCORE_UNSURE = 0
SCORE_CONFIDENT_WRONG = -1


# --------------------------------------------------------------------- YOLOE class vocab
# Filled from the real vocabulary check on the M5 (M1 pipeline step 0), never guessed.
# scripts/check_vocab.py writes data/vocab/yoloe_classes.json; the two lists below are
# derived from it. Until that file exists, perception refuses to run.

class VocabularyUnknown(RuntimeError):
    """The YOLOE class vocabulary has not been recorded yet. Run scripts/check_vocab.py on
    the M5 first — class names must come from real model output."""


def _load_vocab() -> dict:
    if not VOCAB_PATH.exists():
        return {}
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def yoloe_class_names() -> list[str]:
    return list(_load_vocab().get("class_names", []))


def ignore_classes() -> frozenset[str]:
    """Structural/background classes that are never tracked objects."""
    return frozenset(_load_vocab().get("ignore_classes", []))


def hand_class_names() -> tuple[str, ...]:
    """Class name(s) YOLOE actually emits for a hand, used by carried-detection."""
    return tuple(_load_vocab().get("hand_classes", []))


def require_vocab() -> dict:
    vocab = _load_vocab()
    if not vocab.get("class_names"):
        raise VocabularyUnknown(
            f"{VOCAB_PATH} is missing or has no class_names. Run `python scripts/check_vocab.py "
            "<clip>` on the M5 and commit the result before processing a session."
        )
    if not vocab.get("ignore_classes") and not vocab.get("ignore_classes_reviewed"):
        raise VocabularyUnknown(
            f"{VOCAB_PATH} has no ignore_classes and has not been marked reviewed. Set the "
            "ignore list from the real class names, then set ignore_classes_reviewed: true."
        )
    return vocab
