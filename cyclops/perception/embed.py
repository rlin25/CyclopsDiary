"""DINOv2 fingerprints, and the vector math everything else shares.

One model serves two jobs: object-crop fingerprints (identity) and whole-frame fingerprints
(place recognition). Same encoder, same 384-dim L2-normalized CLS token — only what you feed
it differs.

The model half runs on the M5 only. The math half (`l2_normalize`, `cosine`, `is_degenerate`)
is pure numpy and runs anywhere, because the MSI needs it for place lookup and for tests.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np

from cyclops import config

log = logging.getLogger(__name__)

#: Below this norm a vector carries no direction worth comparing. Atlas refuses a zero query
#: vector outright ("Cosine similarity cannot be calculated against a zero vector"), so this
#: has to be caught before any $vectorSearch, not after.
ZERO_NORM_EPS = 1e-8


# ------------------------------------------------------------------------------ pure math

def l2_normalize(vec: Sequence[float] | np.ndarray) -> np.ndarray:
    """Unit-length copy of `vec`. A zero vector is returned unchanged rather than divided by
    zero — callers test with `is_degenerate()` and decide what to do."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < ZERO_NORM_EPS:
        return arr
    return arr / norm


def is_degenerate(vec: Sequence[float] | np.ndarray) -> bool:
    """True when the encoder produced nothing usable (all-zero or non-finite).

    Happens a handful of times across hours of real footage — motion blur, a hand occluding
    the object, a bad crop at a clip boundary. It is the model saying it could not extract
    anything, which is strictly less signal than an unclear match, so Core Principle 5's
    answer applies: no merge, new object.
    """
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return True
    return float(np.linalg.norm(arr)) < ZERO_NORM_EPS


def cosine(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Cosine similarity. Returns 0.0 if either side is degenerate — "no similarity known",
    never a spurious 1.0 from two zero vectors."""
    va, vb = np.asarray(a, dtype=np.float32).reshape(-1), np.asarray(b, dtype=np.float32).reshape(-1)
    if is_degenerate(va) or is_degenerate(vb) or va.shape != vb.shape:
        return 0.0
    return float(np.dot(l2_normalize(va), l2_normalize(vb)))


def max_cosine(vec: Sequence[float] | np.ndarray, others: Iterable[Sequence[float]]) -> float:
    best = 0.0
    for other in others:
        best = max(best, cosine(vec, other))
    return best


def differs_enough(vec: Sequence[float], existing: Iterable[Sequence[float]]) -> bool:
    """Whether a new fingerprint earns one of an object's 12 slots.

    The cap exists to hold varied angles; 12 near-identical frames of one pose would make the
    object harder to re-identify, not easier.
    """
    return max_cosine(vec, existing) < config.FINGERPRINT_MAX_SIM


# ------------------------------------------------------------------- the model (M5 only)

class Embedder:
    """DINOv2-small wrapper. Loads lazily so importing this module costs nothing on the MSI."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or config.DINOV2_MODEL
        self._device = device
        self._model = None
        self._processor = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers/torch are not installed here, so DINOv2 cannot run.\n"
                "This is expected on the MSI: run this on the M5 (see docs/STARTUP.md).\n"
                "On the M5: pip install -r requirements-vision.txt"
            ) from exc
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading %s on %s", self.model_name, device)
        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(device).eval()
        self._device = device

    def embed(self, images: Sequence) -> np.ndarray:
        """L2-normalized CLS tokens, shape (len(images), EMBED_DIM).

        Accepts anything the HF image processor takes (PIL images or HWC uint8 arrays).
        """
        if not len(images):
            return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
        self._load()
        torch = self._torch
        batch = self._processor(images=list(images), return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**batch)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        if cls.shape[1] != config.EMBED_DIM:
            raise RuntimeError(
                f"{self.model_name} produced {cls.shape[1]} dims, but the vector index is built "
                f"for {config.EMBED_DIM}. Changing the model means rebuilding the index."
            )
        return np.stack([l2_normalize(row) for row in cls])

    def embed_one(self, image) -> np.ndarray:
        return self.embed([image])[0]


def average_fingerprint(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """One representative vector for a track, from its per-frame crop embeddings.

    Degenerate frames are dropped first. If every frame was degenerate the result is a zero
    vector, and `is_degenerate()` will say so — the caller must not treat that as a match.
    """
    usable = [l2_normalize(v) for v in vectors if not is_degenerate(v)]
    if not usable:
        return np.zeros(config.EMBED_DIM, dtype=np.float32)
    return l2_normalize(np.mean(np.stack(usable), axis=0))
