"""Command-line interface: ``fpmig <command> ...`` (or ``python -m fpmig``).

  fpmig info                                   model files, configuration and thresholds
  fpmig migrate  --templates DIR --out migrated.npz            batch migration of ISO 19794-2 templates
  fpmig first-login --image probe.png --gallery migrated.npz --id USER [--enrol enrolled.npz]
  fpmig enrol    --images DIR --out enrolled.npz               image embeddings (file stem = user id)
  fpmig verify   --image probe.png --gallery enrolled.npz --id USER
  fpmig calibrate --genuine g.txt --impostor i.txt             thresholds for a new sensor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

IMAGE_EXT = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".jp2", ".pgm"}
TEMPLATE_EXT = {".iso", ".ist", ".fmr", ".bin", ".dat", ".min"}


def _files(path: str, exts: set) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"no files with extensions {sorted(exts)} under {p}")
    return files


def _client(args):
    from .client import MigrationClient
    return MigrationClient(args.models, xla=getattr(args, "xla", False))


def _threshold(v):
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return v


def cmd_info(args):
    from . import hub
    d = hub.model_dir(args.models)
    cfg = hub.load_config(d)
    print(json.dumps({"model_dir": str(d), **{k: v for k, v in cfg.items() if k != "sha256"}}, indent=2))


def cmd_migrate(args):
    from .client import save_gallery
    c = _client(args)
    files = _files(args.templates, TEMPLATE_EXT)
    ids, recs, failed = [], [], []
    from .minutiae import parse_iso19794_2
    for f in files:
        try:
            recs.append(parse_iso19794_2(f.read_bytes(), ppi=args.ppi))
            ids.append(f.stem)
        except Exception as exc:  # noqa: BLE001
            failed.append((str(f), str(exc)))
    vec = c.migrate_templates(recs, batch_size=args.batch_size)
    save_gallery(args.out, ids, vec, "migration", "float16" if args.float16 else "float32")
    print(f"migrated {len(ids)} templates -> {args.out}" + (f"; {len(failed)} failed (see stderr)" if failed else ""))
    for f, e in failed:
        print(f"failed: {f}: {e}", file=sys.stderr)


def cmd_first_login(args):
    from .client import load_gallery, save_gallery
    c = _client(args)
    ids, vec, kind = load_gallery(args.gallery)
    if kind != "migration":
        raise SystemExit(f"{args.gallery} holds '{kind}' vectors; first login needs a migration gallery")
    hit = np.where(ids == args.id)[0]
    if not len(hit):
        raise SystemExit(f"user {args.id} not found in {args.gallery}")
    r = c.first_login(args.image, vec[hit[0]], _threshold(args.threshold))
    print(json.dumps({"id": args.id, **r.as_dict()}, indent=2))
    if r.match and args.enrol:
        p = Path(args.enrol)
        if p.is_file():
            eids, ev, ek = load_gallery(p)
            keep = eids != args.id
            eids, ev = np.append(eids[keep], args.id), np.vstack([ev[keep], r.image_embedding[None]])
        else:
            eids, ev = np.asarray([args.id]), r.image_embedding[None]
        save_gallery(p, eids, ev, "image")
        print(f"re-enrolled {args.id} with the image embedding -> {p}")
    sys.exit(0 if r.match else 1)


def cmd_enrol(args):
    from .client import save_gallery
    c = _client(args)
    files = _files(args.images, IMAGE_EXT)
    vec = c.embed_images(files, batch_size=args.batch_size)
    save_gallery(args.out, [f.stem for f in files], vec, "image")
    print(f"enrolled {len(files)} images -> {args.out}")


def cmd_verify(args):
    from .client import load_gallery
    c = _client(args)
    ids, vec, kind = load_gallery(args.gallery)
    if kind != "image":
        raise SystemExit(f"{args.gallery} holds '{kind}' vectors; daily verification needs an image gallery")
    hit = np.where(ids == args.id)[0]
    if not len(hit):
        raise SystemExit(f"user {args.id} not found in {args.gallery}")
    r = c.verify(args.image, vec[hit[0]], _threshold(args.threshold))
    print(json.dumps({"id": args.id, **r.as_dict()}, indent=2))
    sys.exit(0 if r.match else 1)


def cmd_calibrate(args):
    from .calibrate import report
    g = np.loadtxt(args.genuine, ndmin=1)
    i = np.loadtxt(args.impostor, ndmin=1)
    print(json.dumps(report(g, i, tuple(args.fmr)), indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fpmig", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="model folder or zip (default: $FPMIG_MODELS or the GitHub release)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("info"); p.set_defaults(fn=cmd_info)
    p = sp.add_parser("migrate", help="batch-migrate ISO 19794-2 minutiae templates")
    p.add_argument("--templates", required=True); p.add_argument("--out", required=True)
    p.add_argument("--ppi", type=float, help="override the resolution in the template headers")
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--float16", action="store_true")
    p.set_defaults(fn=cmd_migrate)
    p = sp.add_parser("first-login", help="cross-space check of a probe image against a migration template")
    p.add_argument("--image", required=True); p.add_argument("--gallery", required=True); p.add_argument("--id", required=True)
    p.add_argument("--threshold", help="float, 'eer', 'fmr_1e-2' or 'fmr_1e-3' (validation operating points)")
    p.add_argument("--enrol", help="on a match, add/replace the user's image embedding in this gallery")
    p.add_argument("--xla", action="store_true"); p.set_defaults(fn=cmd_first_login)
    p = sp.add_parser("enrol", help="image embeddings for a folder of images (file stem = user id)")
    p.add_argument("--images", required=True); p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--xla", action="store_true"); p.set_defaults(fn=cmd_enrol)
    p = sp.add_parser("verify", help="daily 1:1 verification against an enrolled image embedding")
    p.add_argument("--image", required=True); p.add_argument("--gallery", required=True); p.add_argument("--id", required=True)
    p.add_argument("--threshold"); p.add_argument("--xla", action="store_true"); p.set_defaults(fn=cmd_verify)
    p = sp.add_parser("calibrate", help="thresholds from genuine / impostor score files (one score per line)")
    p.add_argument("--genuine", required=True); p.add_argument("--impostor", required=True)
    p.add_argument("--fmr", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4]); p.set_defaults(fn=cmd_calibrate)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
