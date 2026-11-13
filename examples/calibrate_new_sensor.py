"""Set the daily-login threshold on a new sensor from labelled pairs.

pairs.csv: image_a,image_b,label   (label 1 = same finger, 0 = different fingers)
"""
import argparse
import csv

import numpy as np

from fpmig import MigrationClient, cosine, threshold_at_eer, threshold_at_fmr

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--target-fmr", type=float, default=1e-3)
ap.add_argument("--models")
a = ap.parse_args()

client = MigrationClient(a.models)
rows = list(csv.DictReader(open(a.pairs)))
paths = sorted({r["image_a"] for r in rows} | {r["image_b"] for r in rows})
emb = dict(zip(paths, client.embed_images(paths)))
s = np.array([cosine(emb[r["image_a"]], emb[r["image_b"]]) for r in rows])
y = np.array([int(r["label"]) for r in rows])
print("EER operating point:", threshold_at_eer(s[y == 1], s[y == 0]).as_dict())
print(f"FMR <= {a.target_fmr:g}:", threshold_at_fmr(s[y == 1], s[y == 0], a.target_fmr).as_dict())
