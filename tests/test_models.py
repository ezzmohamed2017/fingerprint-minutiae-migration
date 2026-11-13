"""Model and client tests with random weights (need TensorFlow; skipped otherwise)."""
import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from fpmig import MigrationClient, cosine  # noqa: E402
from fpmig.models import build_image_encoder, build_minutiae_encoder, build_siamese_towers  # noqa: E402
from tests.conftest import make_iso  # noqa: E402

CFG = dict(quality=False, k=24, depth=3, N=128, aggregate="max", crop=True)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    img = build_image_encoder(); img.save_weights(str(d / "image_encoder.weights.h5"))
    enc = build_minutiae_encoder(CFG); enc.save_weights(str(d / "minutiae_encoder.weights.h5"))
    ft, gt = build_siamese_towers()
    ft.save_weights(str(d / "siamese_minutiae_tower.weights.h5")); gt.save_weights(str(d / "siamese_image_tower.weights.h5"))
    cfg = {"embedding_dim": 1024, "image_model": {"backbone": "EfficientNetV2S", "image_size": 224},
           "minutiae_model": {"config": CFG},
           "thresholds": {"daily_image_vs_image": {"eer": 0.3}, "first_login_cross_space": {"eer": 0.5, "fmr_1e-3": 0.6}},
           "default_operating_point": {"daily": "eer", "first_login": "fmr_1e-3"}}
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_shapes_and_norms():
    enc = build_minutiae_encoder(CFG)
    x = np.zeros((2, 128, 8), np.float32); x[:, :30, 7] = 1; x[:, :30, :2] = np.random.default_rng(0).normal(size=(2, 30, 2))
    e = enc(x, training=False).numpy()
    assert e.shape == (2, 1024) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)


def test_client_scenarios(model_dir, rng):
    c = MigrationClient(str(model_dir))
    img = rng.integers(0, 255, (400, 300)).astype(np.uint8)
    e = c.embed_image(img)
    assert e.shape == (1024,) and abs(np.linalg.norm(e) - 1) < 1e-4
    rec = make_iso(rng.integers(20, 280, 35), rng.integers(20, 380, 35), rng.uniform(0, 360, 35), rng.integers(0, 2, 35), rng.uniform(0.2, 1, 35), 300, 400)
    mig = c.migrate_templates([rec, rec])
    assert mig.shape == (2, 1024) and np.allclose(mig[0], mig[1], atol=1e-5)
    r = c.first_login(img, mig[0])
    assert r.operating_point == "fmr_1e-3" and r.threshold == 0.6 and r.image_embedding.shape == (1024,)
    v = c.verify(img, e)
    assert v.match and abs(v.score - 1) < 1e-4
    assert abs(cosine(e, e) - 1) < 1e-6


def test_weights_reload_identically(model_dir, rng):
    a = MigrationClient(str(model_dir)); b = MigrationClient(str(model_dir))
    img = rng.integers(0, 255, (300, 300)).astype(np.uint8)
    np.testing.assert_allclose(a.embed_image(img), b.embed_image(img), atol=1e-5)
