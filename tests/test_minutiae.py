import numpy as np
import pytest

from fpmig.minutiae import COORD_HALF, Minutiae, from_dict, parse_iso19794_2, to_set
from tests.conftest import make_iso


def training_to_set(m, n_max=128, quality=False):
    # reference implementation used in training
    F_ = 8 if quality else 7; n = len(m['x']); keep = np.argsort(-m['quality'], kind='stable')[:n_max] if n > n_max else np.arange(n); k = len(keep)
    s = np.zeros((n_max, F_ + 1), np.float32)
    s[:k, 0] = m['x'][keep] / COORD_HALF; s[:k, 1] = m['y'][keep] / COORD_HALF; s[:k, 2] = np.cos(m['theta'][keep]); s[:k, 3] = np.sin(m['theta'][keep])
    s[np.arange(k), 4 + m['type'][keep]] = 1.0
    if quality: s[:k, 7] = m['quality'][keep]
    s[:k, F_] = 1.0; return s


def training_parse_iso(data):
    # reference implementation used in training
    import struct
    pos = 24; fq = data[pos + 2]; n = data[pos + 3]; pos += 4
    x = np.zeros(n, np.float32); y = np.zeros(n, np.float32); t = np.zeros(n, np.float32); k = np.zeros(n, np.uint8); q = np.zeros(n, np.float32)
    for i in range(n):
        a, b, ang, qq = struct.unpack('>HHBB', data[pos:pos + 6]); pos += 6; ty = a >> 14
        k[i] = 0 if ty == 1 else (1 if ty == 2 else 2); x[i] = a & 0x3FFF; y[i] = b & 0x3FFF; t[i] = (360.0 - ang * 360.0 / 256.0) % 360.0; q[i] = qq / 100.0
    return dict(x=x, y=y, theta=t, type=k, quality=q, finger_quality=fq)


def test_parse_matches_training_parser(iso_record):
    rec, truth = iso_record
    m = parse_iso19794_2(rec)
    ref = training_parse_iso(rec)
    for key in ("x", "y", "theta", "quality"):
        np.testing.assert_allclose(getattr(m, key), ref[key], atol=1e-5)
    np.testing.assert_array_equal(m.type, ref["type"])
    np.testing.assert_array_equal(m.x, truth["x"]); np.testing.assert_array_equal(m.type, truth["type"])
    assert (m.width, m.height, m.ppi) == (400, 500, 500.0)


def test_to_set_matches_training(iso_record):
    rec, _ = iso_record
    m = parse_iso19794_2(rec)
    # training load_minutiae: centre on the image, rescale to 500 ppi, degrees -> radians
    ref = training_parse_iso(rec); f = 500.0 / 500
    tm = {"x": ((ref["x"] - 400 / 2) * f).astype(np.float32), "y": ((ref["y"] - 500 / 2) * f).astype(np.float32),
          "theta": np.deg2rad(ref["theta"]).astype(np.float32), "type": np.clip(ref["type"].astype(np.int64), 0, 2), "quality": ref["quality"]}
    np.testing.assert_allclose(m.to_set(), training_to_set(tm), atol=1e-6)


def test_more_than_128_keeps_best_quality(rng):
    n = 150
    m = Minutiae(rng.uniform(0, 300, n), rng.uniform(0, 300, n), rng.uniform(0, 360, n), rng.integers(0, 3, n), np.linspace(0, 1, n), 300, 300)
    s = m.to_set()
    assert s.shape == (128, 8) and s[:, 7].sum() == 128


def test_ppi_and_overrides():
    rec = make_iso([10, 20], [30, 40], [0, 90], [0, 1], [0.5, 0.6], width=800, height=750, ppi=1000)
    assert parse_iso19794_2(rec).ppi == 1000.0
    assert parse_iso19794_2(rec, ppi=500, width=400, height=400).width == 400


def test_rejects_non_iso():
    with pytest.raises(ValueError):
        parse_iso19794_2(b"not a template" * 4)


def test_from_dict():
    m = from_dict({"width": 300, "height": 400, "ppi": 500, "minutiae": [{"x": 10, "y": 20, "theta": 45, "type": "bifurcation", "quality": 0.8}]})
    assert len(m) == 1 and m.type[0] == 1
