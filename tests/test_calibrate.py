import numpy as np

from fpmig.calibrate import threshold_at_eer, threshold_at_fmr


def test_fmr_threshold():
    rng = np.random.default_rng(1)
    g = rng.normal(0.7, 0.1, 5000); i = rng.normal(0.1, 0.1, 20000)
    op = threshold_at_fmr(g, i, 1e-3)
    assert op.fmr <= 1e-3 and np.mean(i >= op.threshold) == op.fmr


def test_eer_threshold_balanced():
    rng = np.random.default_rng(2)
    g = rng.normal(0.6, 0.15, 4000); i = rng.normal(0.2, 0.15, 4000)
    op = threshold_at_eer(g, i)
    assert abs(op.fmr - op.fnmr) < 0.01 and 0.35 < op.threshold < 0.45
