"""High-level client for the three operational scenarios.

1. **Batch migration** (offline, once per stored template): minutiae template -> minutiae encoder ->
   Siamese tower -> 1024-d *migration template*.  :meth:`MigrationClient.migrate_templates`
2. **First login** (once per user): probe image -> image model -> Siamese tower (image side), compared with the
   user's migration template; on acceptance, the plain image embedding is returned for re-enrolment.
   :meth:`MigrationClient.first_login`
3. **Daily login** (every verification after re-enrolment): probe image -> image model -> cosine with the
   enrolled image embedding. The Siamese tower is not used. :meth:`MigrationClient.verify`

All embeddings are L2-normalised float32 vectors of length 1024; the score is their cosine similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from . import hub
from .minutiae import Minutiae, from_dict, parse_iso19794_2
from .preprocess import ImageLike, prepare

TemplateLike = Union[Minutiae, bytes, bytearray, dict]
ThresholdLike = Union[str, float, None]


@dataclass
class VerificationResult:
    score: float
    threshold: float
    match: bool
    operating_point: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FirstLoginResult(VerificationResult):
    image_embedding: Optional[np.ndarray] = None       # store this to re-enrol the user when match is True

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "image_embedding"}
        return d


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def _as_template(t: TemplateLike) -> Minutiae:
    if isinstance(t, Minutiae):
        return t
    if isinstance(t, (bytes, bytearray, memoryview)):
        return parse_iso19794_2(bytes(t))
    if isinstance(t, dict):
        return from_dict(t)
    raise TypeError(f"unsupported template type {type(t)}: give an ISO 19794-2 record (bytes), a Minutiae or a dict")


class MigrationClient:
    """Load the released models and run migration, first login and daily verification.

    Parameters
    ----------
    model_path: folder or .zip with the released files; default = ``$FPMIG_MODELS`` or the GitHub release.
    xla: compile the image model with XLA (about 2x faster on a GPU; much slower on a CPU — leave False there).
    """

    def __init__(self, model_path: Optional[str] = None, xla: bool = False):
        self.dir = hub.model_dir(model_path)
        self.config = hub.load_config(self.dir)
        self.thresholds = self.config.get("thresholds", {})
        self.xla = bool(xla)
        self._img = self._enc = self._ft = self._gt = None
        self._f_img = self._f_mig = self._f_first = None

    @classmethod
    def from_pretrained(cls, model_path: Optional[str] = None, **kw) -> "MigrationClient":
        return cls(model_path, **kw)

    # ------------------------------------------------------------------ lazy model loading
    def _tf(self):
        import tensorflow as tf
        return tf

    def _load_image(self):
        if self._img is None:
            from .models import build_image_encoder
            c = self.config["image_model"]
            self._img = build_image_encoder(c.get("backbone", "EfficientNetV2S"), c.get("image_size", 224), self.config.get("embedding_dim", 1024))
            self._img.load_weights(str(self.dir / "image_encoder.weights.h5"))
            tf = self._tf()
            self._f_img = tf.function(lambda z: self._img(z, training=False), jit_compile=self.xla)
        return self._img

    def _load_towers(self):
        if self._ft is None:
            from .models import build_siamese_towers
            self._ft, self._gt = build_siamese_towers(self.config.get("embedding_dim", 1024))
            self._ft.load_weights(str(self.dir / "siamese_minutiae_tower.weights.h5"))
            self._gt.load_weights(str(self.dir / "siamese_image_tower.weights.h5"))
        return self._ft, self._gt

    def _load_minutiae(self):
        if self._enc is None:
            from .models import build_minutiae_encoder
            self._enc = build_minutiae_encoder(self.config["minutiae_model"]["config"], self.config.get("embedding_dim", 1024))
            self._enc.load_weights(str(self.dir / "minutiae_encoder.weights.h5"))
            ft, _ = self._load_towers()
            tf = self._tf()
            self._f_mig = tf.function(lambda z: ft(self._enc(z, training=False), training=False))
        return self._enc

    # ------------------------------------------------------------------ thresholds
    def threshold(self, scenario: str, which: ThresholdLike = None) -> tuple[float, str]:
        """``scenario`` = 'daily' or 'first_login'; ``which`` = a float, 'eer' or 'fmr_1e-2' / 'fmr_1e-3' (validation)."""
        if isinstance(which, (int, float)):
            return float(which), "given"
        key = {"daily": "daily_image_vs_image", "first_login": "first_login_cross_space"}[scenario]
        table = self.thresholds.get(key, {})
        name = which or self.config.get("default_operating_point", {}).get(scenario, "eer")
        if name not in table:
            raise KeyError(f"no '{name}' threshold for {scenario}; available: {sorted(table)}")
        return float(table[name]), name

    # ------------------------------------------------------------------ embeddings
    def embed_images(self, images: Iterable[ImageLike], batch_size: int = 32) -> np.ndarray:
        """Image embeddings (n, 1024) for daily enrolment / verification."""
        self._load_image()
        tf = self._tf()
        size = self.config["image_model"].get("image_size", 224)
        out, batch = [], []
        for im in images:
            batch.append(prepare(im, size))
            if len(batch) == batch_size:
                out.append(self._f_img(tf.constant(np.stack(batch))).numpy())
                batch = []
        if batch:
            out.append(self._f_img(tf.constant(np.stack(batch))).numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.config.get("embedding_dim", 1024)), np.float32)

    def embed_image(self, image: ImageLike) -> np.ndarray:
        return self.embed_images([image], batch_size=1)[0]

    def migrate_templates(self, templates: Sequence[TemplateLike], batch_size: int = 64) -> np.ndarray:
        """Migration templates (n, 1024) from stored minutiae templates (offline batch job)."""
        self._load_minutiae()
        tf = self._tf()
        cfg = self.config["minutiae_model"]["config"]
        out = []
        for i in range(0, len(templates), batch_size):
            sets = np.stack([_as_template(t).to_set(n_max=cfg["N"], quality=cfg["quality"]) for t in templates[i:i + batch_size]])
            out.append(self._f_mig(tf.constant(sets)).numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.config.get("embedding_dim", 1024)), np.float32)

    def migrate_template(self, template: TemplateLike) -> np.ndarray:
        return self.migrate_templates([template], batch_size=1)[0]

    def image_side_embedding(self, image_embedding: np.ndarray) -> np.ndarray:
        """Image embedding -> the shared migration space (used at first login)."""
        _, gt = self._load_towers()
        tf = self._tf()
        z = np.asarray(image_embedding, np.float32).reshape(1, -1)
        return gt(tf.constant(z), training=False).numpy()[0]

    # ------------------------------------------------------------------ scenarios
    def first_login(self, image: ImageLike, migration_template: np.ndarray, threshold: ThresholdLike = None) -> FirstLoginResult:
        """Cross-space check of a probe image against the user's migration template (once per user)."""
        e = self.embed_image(image)
        s = cosine(self.image_side_embedding(e), migration_template)
        t, name = self.threshold("first_login", threshold)
        return FirstLoginResult(score=s, threshold=t, match=s >= t, operating_point=name, image_embedding=e)

    def verify(self, probe: Union[ImageLike, np.ndarray], enrolled_embedding: np.ndarray, threshold: ThresholdLike = None) -> VerificationResult:
        """Daily 1:1 verification: image (or its embedding) against the enrolled image embedding."""
        p = np.asarray(probe)
        e = p.astype(np.float32) if (p.ndim == 1 and p.size == enrolled_embedding.size) else self.embed_image(probe)
        s = cosine(e, enrolled_embedding)
        t, name = self.threshold("daily", threshold)
        return VerificationResult(score=s, threshold=t, match=s >= t, operating_point=name)

    def verify_images(self, image_a: ImageLike, image_b: ImageLike, threshold: ThresholdLike = None) -> VerificationResult:
        a, b = self.embed_images([image_a, image_b], batch_size=2)
        return self.verify(a, b, threshold)

    def warmup(self, n: int = 3) -> None:
        """Run a few dummy calls (graph tracing / compilation) before timing or serving."""
        size = self.config["image_model"].get("image_size", 224)
        dummy = np.full((size, size), 255, np.uint8)
        for _ in range(n):
            self.embed_image(dummy)


def save_gallery(path: Union[str, Path], ids: Sequence[str], vectors: np.ndarray, kind: str, dtype: str = "float32") -> None:
    """Store embeddings with their user ids (``kind`` = 'migration' or 'image')."""
    v = np.asarray(vectors, dtype)
    if len(ids) != len(v):
        raise ValueError("ids and vectors differ in length")
    np.savez_compressed(path, ids=np.asarray(ids, dtype=object).astype(str), vectors=v, kind=np.asarray(kind))


def load_gallery(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray, str]:
    z = np.load(path, allow_pickle=False)
    return z["ids"], z["vectors"].astype(np.float32), str(z["kind"])
