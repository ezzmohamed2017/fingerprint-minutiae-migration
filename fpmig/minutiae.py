"""Minutiae templates: parsing and conversion to the fixed-size set the minutiae encoder reads.

Conventions (identical to training):

* positions are in pixels of the original image, re-centred on the image centre and rescaled to 500 ppi,
  then divided by ``COORD_HALF`` (112);
* the direction ``theta`` is in degrees, measured in image coordinates as FingerJetFX writes it after the
  ISO angle is converted with ``theta = (360 - iso_angle * 360 / 256) mod 360``;
* type 0 = ridge ending, 1 = bifurcation, 2 = other;
* quality in [0, 1]; when a template has more than 128 minutiae the 128 of highest quality are kept.

The models were trained on FingerJetFX OSE templates in the ISO/IEC 19794-2:2005 format. Templates from
another extractor parse the same way, but accuracy has only been validated for FingerJetFX.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

COORD_HALF = 112.0
N_MAX = 128


@dataclass
class Minutiae:
    """One template. ``x``/``y`` in original-image pixels (top-left origin), ``theta`` in degrees (image convention)."""

    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    type: np.ndarray
    quality: np.ndarray
    width: int
    height: int
    ppi: float = 500.0
    finger_quality: Optional[int] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x, np.float32).reshape(-1)
        self.y = np.asarray(self.y, np.float32).reshape(-1)
        self.theta = np.asarray(self.theta, np.float32).reshape(-1)
        self.type = np.clip(np.asarray(self.type, np.int64).reshape(-1), 0, 2)
        self.quality = np.asarray(self.quality, np.float32).reshape(-1)
        n = len(self.x)
        if not (len(self.y) == len(self.theta) == len(self.type) == len(self.quality) == n):
            raise ValueError("x, y, theta, type and quality must have the same length")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height of the original image are required to centre the minutiae")
        if not self.ppi or self.ppi <= 0:
            self.ppi = 500.0

    def __len__(self) -> int:
        return len(self.x)

    # ---- the representation used in training (fpp13_minutiae.load_minutiae)
    def normalised(self) -> dict:
        f = 500.0 / float(self.ppi)
        return {
            "x": ((self.x - self.width / 2) * f).astype(np.float32),
            "y": ((self.y - self.height / 2) * f).astype(np.float32),
            "theta": np.deg2rad(self.theta).astype(np.float32),
            "type": self.type,
            "quality": self.quality,
        }

    def to_set(self, n_max: int = N_MAX, quality: bool = False) -> np.ndarray:
        return to_set(self.normalised(), n_max=n_max, quality=quality)


def to_set(m: dict, n_max: int = N_MAX, quality: bool = False) -> np.ndarray:
    """(n_max, F + 1) array: x, y (/ COORD_HALF), cos, sin, type one-hot (3), [quality], mask."""
    f_ = 8 if quality else 7
    n = len(m["x"])
    keep = np.argsort(-m["quality"], kind="stable")[:n_max] if n > n_max else np.arange(n)
    k = len(keep)
    s = np.zeros((n_max, f_ + 1), np.float32)
    s[:k, 0] = m["x"][keep] / COORD_HALF
    s[:k, 1] = m["y"][keep] / COORD_HALF
    s[:k, 2] = np.cos(m["theta"][keep])
    s[:k, 3] = np.sin(m["theta"][keep])
    s[np.arange(k), 4 + m["type"][keep]] = 1.0
    if quality:
        s[:k, 7] = m["quality"][keep]
    s[:k, f_] = 1.0
    return s


_STANDARD_PPI = (250, 300, 333, 362, 400, 500, 600, 700, 800, 1000, 1200)


def _ppi_from_header(xres_per_cm: int) -> float:
    """ISO 19794-2:2005 stores resolution in pixels per centimetre (500 ppi -> 197); snap to the nominal ppi."""
    if not xres_per_cm:
        return 500.0
    ppi = xres_per_cm * 2.54
    best = min(_STANDARD_PPI, key=lambda s: abs(s - ppi))
    return float(best) if abs(best - ppi) <= 0.01 * best else float(round(ppi))


def parse_iso19794_2(record: bytes, ppi: Optional[float] = None, width: Optional[int] = None,
                     height: Optional[int] = None, view: int = 0) -> Minutiae:
    """Parse an ISO/IEC 19794-2:2005 finger minutiae record (``FMR\\0`` + version `` 20\\0``).

    The image size and resolution are read from the record header; pass ``ppi``/``width``/``height`` to
    override them (for example when the header resolution is missing or wrong).
    """
    data = bytes(record)
    if len(data) < 28 or data[:4] != b"FMR\x00":
        raise ValueError("not an ISO/IEC 19794-2 finger minutiae record (missing 'FMR\\0')")
    version = data[4:8]
    if version not in (b" 20\x00",):
        raise ValueError(f"unsupported ISO 19794-2 version {version!r}: only the 2005 format (' 20') is supported")
    hdr_w, hdr_h, xres, _yres = struct.unpack(">HHHH", data[14:22])
    n_views = data[22]
    if view >= max(1, n_views):
        raise ValueError(f"record has {n_views} finger view(s); view {view} requested")
    pos = 24
    for _ in range(view + 1):
        finger_quality, n = data[pos + 2], data[pos + 3]
        pos += 4
        start = pos
        pos += 6 * n
        ext_len = struct.unpack(">H", data[pos:pos + 2])[0] if pos + 2 <= len(data) else 0
        view_start, view_n = start, n
        pos += 2 + ext_len
    x = np.zeros(view_n, np.float32)
    y = np.zeros(view_n, np.float32)
    t = np.zeros(view_n, np.float32)
    k = np.zeros(view_n, np.int64)
    q = np.zeros(view_n, np.float32)
    p = view_start
    for i in range(view_n):
        a, b, ang, qq = struct.unpack(">HHBB", data[p:p + 6])
        p += 6
        ty = a >> 14
        k[i] = 0 if ty == 1 else (1 if ty == 2 else 2)
        x[i] = a & 0x3FFF
        y[i] = b & 0x3FFF
        t[i] = (360.0 - ang * 360.0 / 256.0) % 360.0
        q[i] = qq / 100.0
    if ppi is None:
        ppi = _ppi_from_header(xres)
    return Minutiae(x, y, t, k, q, width=int(width or hdr_w), height=int(height or hdr_h), ppi=float(ppi),
                    finger_quality=int(finger_quality), meta={"format": "ISO/IEC 19794-2:2005"})


def from_arrays(x: Sequence[float], y: Sequence[float], theta_deg: Sequence[float], type_: Sequence[int],
                quality: Sequence[float], width: int, height: int, ppi: float = 500.0) -> Minutiae:
    """Build a template from arrays that already follow the conventions in the module docstring."""
    return Minutiae(np.asarray(x), np.asarray(y), np.asarray(theta_deg), np.asarray(type_), np.asarray(quality),
                    width=int(width), height=int(height), ppi=float(ppi))


def from_dict(d: dict) -> Minutiae:
    """From a JSON-like dict: {"width", "height", "ppi", "minutiae": [{"x", "y", "theta", "type", "quality"}, ...]}."""
    ms = d.get("minutiae", [])
    tmap = {"ending": 0, "bifurcation": 1, "other": 2}
    return Minutiae(
        [m["x"] for m in ms], [m["y"] for m in ms], [m["theta"] for m in ms],
        [tmap.get(m.get("type", 2), m.get("type", 2)) if isinstance(m.get("type", 2), str) else m.get("type", 2) for m in ms],
        [m.get("quality", 1.0) for m in ms], width=int(d["width"]), height=int(d["height"]), ppi=float(d.get("ppi", 500)),
    )
