"""Image preprocessing used by the image model.

The image model was trained on the original greyscale capture (native sensor resolution) placed on a
224 x 224 white canvas with its aspect ratio preserved (``letterbox``). Apply exactly the same function
to every probe and enrolment image; do not crop, enhance or binarise the image beforehand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

IMAGE_SIZE = 224
ImageLike = Union[str, Path, bytes, np.ndarray]


def _cv2():
    try:
        import cv2  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fpmig needs OpenCV: pip install opencv-python-headless") from exc
    return cv2


def load_grayscale(image: ImageLike) -> np.ndarray:
    """Return a 2-D uint8 greyscale array from a path, encoded bytes (PNG, BMP, JPEG 2000, ...) or an array."""
    cv2 = _cv2()
    if isinstance(image, (str, Path)):
        data = np.fromfile(str(image), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"could not decode image file: {image}")
    elif isinstance(image, (bytes, bytearray, memoryview)):
        img = cv2.imdecode(np.frombuffer(bytes(image), np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("could not decode the image bytes")
    else:
        img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    if img.ndim != 2:
        raise ValueError(f"expected a greyscale image, got shape {img.shape}")
    if img.dtype != np.uint8:      # same conversion as the training data preparation
        img = (img / 257).astype(np.uint8) if img.dtype == np.uint16 else img.astype(np.uint8)
    return np.ascontiguousarray(img)


def letterbox(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Aspect-preserving resize onto a white ``size`` x ``size`` canvas (identical to the training pipeline)."""
    cv2 = _cv2()
    h, w = image.shape
    s = min(size / w, size / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.full((size, size), 255, np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def prepare(image: ImageLike, size: int = IMAGE_SIZE) -> np.ndarray:
    """Decode (if needed) and letterbox; returns uint8 of shape (size, size, 1), the image model's input."""
    return letterbox(load_grayscale(image), size)[..., None]
