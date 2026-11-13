"""Locate or download the released model files.

Order of lookup for :func:`model_dir`:

1. the ``path`` argument;
2. the ``FPMIG_MODELS`` environment variable;
3. the cache ``~/.cache/fpmig/<version>`` (downloaded from the GitHub release on first use).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from . import __version__

GITHUB_OWNER = "ezzmohamed2017"
GITHUB_REPO = "fingerprint-minutiae-migration"
MODELS_VERSION = "1.0.0"
ASSET = f"fpmig-models-v{MODELS_VERSION}.zip"
RELEASE_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/v{MODELS_VERSION}/{ASSET}"

REQUIRED = (
    "config.json",
    "image_encoder.weights.h5",
    "minutiae_encoder.weights.h5",
    "siamese_minutiae_tower.weights.h5",
    "siamese_image_tower.weights.h5",
)


def _cache_root() -> Path:
    return Path(os.environ.get("FPMIG_CACHE", Path.home() / ".cache" / "fpmig"))


def _complete(d: Path) -> bool:
    return all((d / f).is_file() for f in REQUIRED)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(d: Path) -> None:
    """Check every file listed in config.json["sha256"] against its digest."""
    cfg = json.loads((d / "config.json").read_text())
    for name, digest in cfg.get("sha256", {}).items():
        p = d / name
        if p.is_file() and sha256(p) != digest:
            raise IOError(f"{p} is corrupted (sha256 mismatch); delete {d} and download again")


def download(url: str = RELEASE_URL, dest: Optional[Path] = None, quiet: bool = False) -> Path:
    dest = Path(dest or _cache_root() / MODELS_VERSION)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / ASSET
        if not quiet:
            print(f"fpmig: downloading models from {url}")
        with urllib.request.urlopen(url) as r, open(zpath, "wb") as fh:
            shutil.copyfileobj(r, fh)
        with zipfile.ZipFile(zpath) as z:
            for member in z.namelist():
                target = (dest / member).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise IOError(f"unsafe path in archive: {member}")
            z.extractall(dest)
    # the archive may hold a single top-level folder
    if not _complete(dest):
        subs = [p for p in dest.iterdir() if p.is_dir() and _complete(p)]
        if subs:
            for p in subs[0].iterdir():
                shutil.move(str(p), dest / p.name)
    if not _complete(dest):
        raise IOError(f"the downloaded archive does not contain {REQUIRED}")
    verify(dest)
    return dest


def model_dir(path: Optional[str | os.PathLike] = None, download_if_missing: bool = True) -> Path:
    for cand in (path, os.environ.get("FPMIG_MODELS")):
        if cand:
            d = Path(cand)
            if d.suffix == ".zip" and d.is_file():
                out = _cache_root() / d.stem
                if not _complete(out):
                    out.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(d) as z:
                        z.extractall(out)
                    subs = [p for p in out.iterdir() if p.is_dir() and _complete(p)]
                    if not _complete(out) and subs:
                        return subs[0]
                return out
            if not _complete(d):
                raise FileNotFoundError(f"{d} does not contain the model files {REQUIRED}")
            return d
    d = _cache_root() / MODELS_VERSION
    if _complete(d):
        return d
    if not download_if_missing:
        raise FileNotFoundError(f"models not found in {d}; call fpmig.hub.download() or set FPMIG_MODELS")
    return download(dest=d)


def load_config(d: Path) -> dict:
    return json.loads((Path(d) / "config.json").read_text())


__all__ = ["model_dir", "download", "load_config", "verify", "RELEASE_URL", "__version__"]
