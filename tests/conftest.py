import struct

import numpy as np
import pytest


def make_iso(x, y, theta_img_deg, types, quality, width=400, height=500, ppi=500, finger_quality=60):
    """Write an ISO/IEC 19794-2:2005 record the way FingerJetFX does (angle = (360 - theta) * 256 / 360)."""
    n = len(x)
    body = b""
    for i in range(n):
        ty = {0: 1, 1: 2, 2: 0}[int(types[i])]
        ang = int(round(((360.0 - theta_img_deg[i]) % 360.0) * 256.0 / 360.0)) % 256
        body += struct.pack(">HHBB", (ty << 14) | int(x[i]), int(y[i]), ang, int(round(quality[i] * 100)))
    view = bytes([0, 0, finger_quality, n]) + body + struct.pack(">H", 0)
    res = int(round(ppi / 2.54))
    total = 24 + len(view)
    header = b"FMR\x00" + b" 20\x00" + struct.pack(">I", total)[:4] + b"\x00\x00" + struct.pack(">HHHH", width, height, res, res) + bytes([1, 0])
    assert len(header) == 24
    return header + view


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def iso_record(rng):
    n = 40
    x = rng.integers(20, 380, n); y = rng.integers(20, 480, n)
    th = (rng.integers(0, 256, n) * 360.0 / 256.0)                      # angles representable in ISO
    th = (360.0 - th) % 360.0
    ty = rng.integers(0, 2, n); q = rng.integers(10, 100, n) / 100.0
    return make_iso(x, y, th, ty, q), dict(x=x, y=y, theta=th, type=ty, quality=q)
