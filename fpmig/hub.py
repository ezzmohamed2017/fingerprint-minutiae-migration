"""Locate or download the pretrained model files.

Order of lookup for :func:`model_dir`:

1. the ``path`` argument (a folder or a .zip);
2. the ``FPMIG_MODELS`` environment variable;
3. the ``models/`` folder of a cloned repository (when the package is used from a clone);
4. the cache ``~/.cache/fpmig/<version>``, filled from the GitHub repository on first use.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

GITHUB_OWNER = "ezzmohamed2017"
GITHUB_REPO = "fingerprint-minutiae-migration"
MODELS_VERSION = "1.0.0"
BASE_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/raw/main/models"

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
    """Check every weight file against the SHA-256 listed in config.json."""
    cfg = json.loads((Path(d) / "config.json").read_text())
    for name, digest in cfg.get("sha256", {}).items():
        p = Path(d) / name
        if not p.is_file():
            raise FileNotFoundError(p)
        if sha256(p) != digest:
            raise IOError(f"{p} is corrupted or incomplete (sha256 mismatch); delete it and try again")


def download(dest: Optional[Path] = None, base_url: str = BASE_URL, quiet: bool = False) -> Path:
    """Download the model files from the GitHub repository into ``dest`` (default: the cache)."""
    dest = Path(dest or _cache_root() / MODELS_VERSION)
    dest.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        target = dest / name
        if target.is_file() and name != "config.json":
            continue
        if not quiet:
            print(f"fpmig: downloading {name}")
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(f"{base_url}/{name}") as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh)
        tmp.replace(target)
    verify(dest)
    return dest


def _repo_models() -> Optional[Path]:
    d = Path(__file__).resolve().parent.parent / "models"
    return d if _complete(d) else None


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
                return out if _complete(out) or not subs else subs[0]
            if not _complete(d):
                raise FileNotFoundError(f"{d} does not contain the model files {REQUIRED}")
            return d
    local = _repo_models()
    if local is not None:
        return local
    d = _cache_root() / MODELS_VERSION
    if _complete(d):
        return d
    if not download_if_missing:
        raise FileNotFoundError(f"models not found in {d}; call fpmig.hub.download() or set FPMIG_MODELS")
    return download(dest=d)


def load_config(d: Path) -> dict:
    return json.loads((Path(d) / "config.json").read_text())


__all__ = ["model_dir", "download", "load_config", "verify", "BASE_URL"]
