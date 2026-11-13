"""Decision thresholds from genuine and impostor scores.

A threshold learned on one sensor does not transfer reliably to another one: the score distribution shifts
with the sensor, so the false match rate at a fixed threshold can be far from the one measured during
development even when the ranking quality (EER, AUC) stays good. Before deployment, collect genuine and impostor pairs on the target sensor and population and set
the threshold here (target FMR is the usual operating point; EER is reported for comparison).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np


@dataclass
class OperatingPoint:
    threshold: float
    fmr: float
    fnmr: float
    rule: str

    def as_dict(self) -> dict:
        return asdict(self)


def _rates(gen: np.ndarray, imp: np.ndarray, t: float):
    return float(np.mean(imp >= t)), float(np.mean(gen < t))


def threshold_at_fmr(genuine: Sequence[float], impostor: Sequence[float], target_fmr: float) -> OperatingPoint:
    """Lowest threshold whose false match rate (score >= threshold on impostor pairs) is <= ``target_fmr``."""
    gen = np.asarray(genuine, np.float64)
    imp = np.sort(np.asarray(impostor, np.float64))
    if len(imp) == 0 or len(gen) == 0:
        raise ValueError("need genuine and impostor scores")
    if target_fmr * len(imp) < 1:
        print(f"warning: {len(imp)} impostor scores cannot resolve FMR = {target_fmr:g}; use at least {int(np.ceil(10 / target_fmr))}")
    allowed = int(np.floor(target_fmr * len(imp)))           # impostors allowed at or above the threshold
    t = float(np.nextafter(imp[-1], np.inf)) if allowed == 0 else float(np.nextafter(imp[len(imp) - allowed - 1], np.inf))
    fmr, fnmr = _rates(gen, imp, t)
    return OperatingPoint(t, fmr, fnmr, f"FMR <= {target_fmr:g}")


def threshold_at_eer(genuine: Sequence[float], impostor: Sequence[float]) -> OperatingPoint:
    """Threshold where FMR and FNMR are closest (equal error rate)."""
    gen = np.asarray(genuine, np.float64)
    imp = np.asarray(impostor, np.float64)
    cand = np.unique(np.concatenate([gen, imp]))
    gs, is_ = np.sort(gen), np.sort(imp)
    fmr = 1.0 - np.searchsorted(is_, cand, side="left") / len(is_)      # P(impostor >= t)
    fnmr = np.searchsorted(gs, cand, side="left") / len(gs)             # P(genuine < t)
    i = int(np.argmin(np.abs(fmr - fnmr)))
    return OperatingPoint(float(cand[i]), float(fmr[i]), float(fnmr[i]), "EER")


def report(genuine: Sequence[float], impostor: Sequence[float], targets=(1e-2, 1e-3, 1e-4)) -> list[dict]:
    rows = [threshold_at_eer(genuine, impostor).as_dict()]
    for f in targets:
        rows.append(threshold_at_fmr(genuine, impostor, f).as_dict())
    return rows
