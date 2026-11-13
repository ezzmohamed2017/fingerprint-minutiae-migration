"""fpmig — migrate minutiae-template fingerprint systems to deep image-embedding verification."""
__version__ = "1.0.0"

from .minutiae import Minutiae, from_arrays, from_dict, parse_iso19794_2  # noqa: E402
from .preprocess import letterbox, load_grayscale, prepare  # noqa: E402
from .calibrate import threshold_at_eer, threshold_at_fmr  # noqa: E402
from .client import (  # noqa: E402
    FirstLoginResult,
    MigrationClient,
    VerificationResult,
    cosine,
    load_gallery,
    save_gallery,
)

__all__ = [
    "MigrationClient", "VerificationResult", "FirstLoginResult", "cosine", "save_gallery", "load_gallery",
    "Minutiae", "parse_iso19794_2", "from_arrays", "from_dict", "letterbox", "load_grayscale", "prepare",
    "threshold_at_eer", "threshold_at_fmr", "__version__",
]
