"""Batch migration of a stored population (one ISO 19794-2 record per file, file stem = user id).

    python examples/migrate_population.py --templates /data/templates --out migrated.npz

Runs on a GPU if TensorFlow sees one (about 38 minutes per million templates on one T4).
"""
import argparse
import time
from pathlib import Path

from fpmig import MigrationClient, parse_iso19794_2, save_gallery

ap = argparse.ArgumentParser()
ap.add_argument("--templates", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--models")
a = ap.parse_args()

client = MigrationClient(a.models)
files = sorted(p for p in Path(a.templates).rglob("*") if p.suffix.lower() in {".iso", ".fmr", ".ist", ".bin"})
ids, vecs, t0 = [], [], time.time()
import numpy as np
for i in range(0, len(files), a.chunk):
    part = files[i:i + a.chunk]
    recs = [parse_iso19794_2(p.read_bytes()) for p in part]
    vecs.append(client.migrate_templates(recs, batch_size=64))
    ids += [p.stem for p in part]
    print(f"{len(ids)}/{len(files)} templates, {time.time() - t0:.0f} s")
save_gallery(a.out, ids, np.concatenate(vecs), "migration")
print("saved", a.out)
