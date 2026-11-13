# fpmig — fingerprint minutiae → deep embedding migration

Pretrained models and a reference client for moving a fingerprint verification system from stored **minutiae
templates** to fast **deep image-embedding** verification, without asking enrolled users to re-register first.

| Stage | When | What runs | Output |
|---|---|---|---|
| **1. Batch migration** | once per stored template, offline | minutiae template → minutiae encoder → shared tower | 1024-d *migration template* |
| **2. First login** | once per user | probe image → image model → shared tower · cosine with the migration template | accept / reject; on accept, the user's image embedding |
| **3. Daily login** | every verification afterwards | probe image → image model · cosine with the enrolled image embedding | accept / reject |

After the first login the migration template is discarded and the user is verified by the image model alone.

## Install

```bash
pip install git+https://github.com/ezzmohamed2017/fingerprint-minutiae-migration.git
```

Requires Python ≥ 3.9, TensorFlow ≥ 2.15 and OpenCV. A GPU or NPU is recommended for the image model.

## Models

The weights are published as `fpmig-models-v1.0.0.zip` in the
[releases](https://github.com/ezzmohamed2017/fingerprint-minutiae-migration/releases) and are downloaded automatically
on first use (cached in `~/.cache/fpmig`). To use a local copy, set `FPMIG_MODELS=/path/to/folder-or-zip` or pass
`model_path=` to the client.

| File | Model |
|---|---|
| `image_encoder.weights.h5` | image encoder: letterboxed greyscale image → 1024-d embedding |
| `minutiae_encoder.weights.h5` | minutiae encoder: ISO 19794-2 template → 1024-d embedding |
| `siamese_minutiae_tower.weights.h5`, `siamese_image_tower.weights.h5` | shared tower used for migration and the first login |
| `config.json` | model settings, default thresholds, SHA-256 checksums |

## Usage (Python)

```python
from fpmig import MigrationClient

client = MigrationClient.from_pretrained()              # downloads the models on first use

# 1) batch migration of the stored ISO/IEC 19794-2 templates (offline, once)
records = [open(p, "rb").read() for p in ["u001.iso", "u002.iso"]]
migration_templates = client.migrate_templates(records)             # (2, 1024) float32

# 2) first login of user u001: probe image against the migration template
r = client.first_login("u001_probe.png", migration_templates[0])
print(r.score, r.threshold, r.match)
if r.match:
    enrolled = r.image_embedding                       # re-enrol: store this, delete the migration template

# 3) daily login: image vs image
v = client.verify("u001_today.png", enrolled)
print(v.match, v.score)

# new users can be enrolled directly from an image
e = client.embed_image("new_user.png")
```

`threshold=` accepts a number, `"eer"`, `"fmr_1e-2"` or `"fmr_1e-3"`. Defaults: `"eer"` for daily login and
`"fmr_1e-3"` for the first login, where a false match would enrol the wrong person's image.

## Usage (command line)

```bash
fpmig info                                                       # model files, settings, thresholds
fpmig migrate --templates templates/ --out migrated.npz          # *.iso / *.fmr / *.ist records, file stem = user id
fpmig first-login --image probe.png --gallery migrated.npz --id u001 --enrol enrolled.npz
fpmig verify --image today.png --gallery enrolled.npz --id u001  # exit code 0 = match, 1 = no match
fpmig enrol --images captures/ --out enrolled.npz                # enrol new users directly from images
fpmig calibrate --genuine genuine.txt --impostor impostor.txt --fmr 1e-3 1e-4
```

More complete scripts are in [`examples/`](examples).

## Input requirements

* **Images:** the original greyscale capture at native resolution (PNG, BMP, TIFF, JPEG 2000, …). The client
  resizes it onto a 224 × 224 canvas itself; do not crop, enhance or binarise it first.
* **Templates:** ISO/IEC 19794-2:2005 finger minutiae records (`FMR\0`, version ` 20`). Image size and resolution are
  read from the record header (`--ppi` / `ppi=` to override). The models were trained on templates produced by
  [FingerJetFX OSE](https://github.com/FingerJetFXOSE/FingerJetFXOSE); templates from other extractors are accepted,
  but their accuracy has not been measured. Other layouts can be passed with `fpmig.from_arrays(...)` or
  `fpmig.from_dict(...)` (see `fpmig/minutiae.py` for the conventions).

## Calibrate the thresholds for your sensor

The thresholds shipped in `config.json` were set on development data and **do not transfer reliably to a different
sensor or population**. Before deployment, score genuine and impostor pairs captured on your sensor and set the
threshold at the false match rate you need:

```python
from fpmig import threshold_at_fmr
op = threshold_at_fmr(genuine_scores, impostor_scores, target_fmr=1e-3)
client.verify(probe, enrolled, threshold=op.threshold)
```

`examples/calibrate_new_sensor.py` does this from a CSV of labelled image pairs.

## Responsible use

Fingerprints are sensitive biometric data. Use these models only with the informed consent of the people enrolled
and in line with the data-protection law that applies to you. Protect stored embeddings like the templates they
replace (ISO/IEC 24745). The release contains no images, templates or embeddings of any person.

## Licence

Code: MIT ([LICENSE](LICENSE)). Model weights: CC BY-NC 4.0, non-commercial research only ([MODEL_LICENSE.md](MODEL_LICENSE.md)).
