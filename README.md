# Fingerprint minutiae → deep embedding migration

Released models, a reference client and the complete training code for the paper:

> **Bridging Minutiae and Deep Embeddings: Fast Fingerprint Verification with Cross-Representation Template Migration**
> Mohamed Ezz, 2026 (under review). See [Citation](#citation).

Most deployed fingerprint systems store **minutiae templates**, not images, so they cannot simply recompute deep
embeddings for the people they have already enrolled. This repository provides a migration path:

| Stage | When | What runs | Output |
|---|---|---|---|
| **1. Batch migration** | once per stored template, offline | minutiae template → minutiae encoder → Siamese tower | 1024-d *migration template* |
| **2. First login** | once per user | probe image → image model → Siamese tower (image side) · cosine with the migration template | accept / reject; on accept, the user's image embedding |
| **3. Daily login** | every verification afterwards | probe image → image model · cosine with the enrolled image embedding | accept / reject |

Only stage 2 crosses feature spaces. After it, the migration template is discarded and the user is verified by
the image model alone, which matches the accuracy of the open minutiae pipeline and is **6.6× faster** on an accelerator.

## Results (MOLF test subjects, 3 repetitions, mean ± std)

| System | Test EER ↓ | Test AUC ↑ | Test F1 ↑ | Cross-sensor EER ↓ | FingerDOS EER ↓ | Probe → decision |
|---|---|---|---|---|---|---|
| **Daily login: image vs image (this repo)** | 7.18 ± 0.36 | 97.31 | 92.92 | **8.78** | 2.87 | **7.43 ms** (T4 + XLA) |
| FingerJetFX + BOZORTH3 (legacy) | 6.99 ± 1.05 | 97.16 | 92.82 | 9.73 | **1.12** | 49.22 ms (CPU) |
| Minutiae encoder, minutiae vs minutiae | 8.66 ± 0.73 | 96.62 | 91.31 | 11.00 | 3.66 | — |
| **First login: image vs migration template** | 11.04 ± 0.95 | 94.90 | 88.88 | 12.10 | 11.67 | 21.9 ms (T4) |

Batch migration costs 2.3 ms per template on one T4 GPU (about **38 minutes per million templates**).
All tables, including every ablation, are in [docs/RESULTS.md](docs/RESULTS.md).

> **Thresholds do not transfer between sensors.** At the threshold frozen on MOLF validation, the image model's
> false match rate on the unseen FingerDOS sensor was 11.9 %, although its EER there was 2.87 %. Calibrate the
> threshold on your own sensor before deployment ([Calibration](#calibrate-thresholds-for-your-sensor)).

## Install

```bash
pip install git+https://github.com/ezzmohamed2017/fingerprint-minutiae-migration.git
```

Python ≥ 3.9, TensorFlow ≥ 2.15, OpenCV. A GPU or NPU is recommended for the image model: on a CPU it is slower
than the minutiae pipeline (74 ms with 8 threads).

## Models

The weights are attached to the [GitHub release](https://github.com/ezzmohamed2017/fingerprint-minutiae-migration/releases)
as `fpmig-models-v1.0.0.zip` and are downloaded automatically on first use (cached in `~/.cache/fpmig`).
To use a local copy, set `FPMIG_MODELS=/path/to/folder-or-zip` or pass `model_path=`.

| File | Model | Parameters | GFLOPs / item |
|---|---|---|---|
| `image_encoder.weights.h5` | EfficientNetV2-S + 1024-d head, shared cross-sensor CosFace proxies (FPP04L) | 21.6 M | 5.75 |
| `minutiae_encoder.weights.h5` | local-structure encoder, k = 24, 3 layers, jointly trained (FPP13) | 1.7 M | 2.20 |
| `siamese_minutiae_tower.weights.h5`, `siamese_image_tower.weights.h5` | one shared tower with a learned offset per modality (FPP13) | 4.2 M | 0.008 |
| `config.json` | architecture settings, validation thresholds, provenance, SHA-256 of every file | | |

## Quick start (Python)

```python
from fpmig import MigrationClient, parse_iso19794_2

client = MigrationClient.from_pretrained()          # downloads the release on first use

# 1) batch migration of stored ISO/IEC 19794-2 templates (offline)
records = [open(p, "rb").read() for p in ["u001.iso", "u002.iso"]]
migration_templates = client.migrate_templates(records)            # (2, 1024) float32

# 2) first login of user u001: probe image against the migration template
r = client.first_login("u001_probe.png", migration_templates[0])
print(r.score, r.threshold, r.match)
if r.match:
    enrolled = r.image_embedding                    # re-enrol: store this, delete the migration template

# 3) daily login: image vs image
v = client.verify("u001_today.png", enrolled)
print(v.match, v.score)
```

`threshold=` accepts a number, `"eer"`, `"fmr_1e-2"` or `"fmr_1e-3"` (operating points measured on the validation
subjects). The default is `"eer"` for daily login and `"fmr_1e-3"` for the first login, because a false match there
would enrol an impostor's image under a genuine identity.

## Command line

```bash
fpmig info                                                       # files, config, thresholds
fpmig migrate --templates templates/ --out migrated.npz          # *.iso / *.fmr / *.ist ISO 19794-2 records
fpmig first-login --image probe.png --gallery migrated.npz --id u001 --enrol enrolled.npz
fpmig verify --image today.png --gallery enrolled.npz --id u001  # exit code 0 = match, 1 = no match
fpmig enrol --images captures/ --out enrolled.npz                # direct enrolment of new users (file stem = id)
fpmig calibrate --genuine genuine.txt --impostor impostor.txt --fmr 1e-3 1e-4
```

## Input requirements

* **Images:** the original greyscale capture at native resolution (PNG, BMP, TIFF, JPEG 2000, …). The client
  letterboxes it to 224 × 224 on white exactly as in training; do not crop, enhance or binarise it first.
* **Templates:** ISO/IEC 19794-2:2005 records (`FMR\0` + ` 20\0`). Image size and resolution are read from the
  header (`--ppi` / `ppi=` override). The models were trained on **FingerJetFX OSE** templates; templates from other
  extractors are accepted but their accuracy has not been measured. Other layouts can be passed with
  `fpmig.from_arrays(...)` or `fpmig.from_dict(...)` following the conventions in `fpmig/minutiae.py`.

## Calibrate thresholds for your sensor

Collect genuine and impostor pairs on the target sensor (ideally ≥ 10 / target-FMR impostor pairs), score them
with `client.verify(...).score` (daily) or `client.first_login(...).score` (first login), then:

```python
from fpmig import threshold_at_fmr
op = threshold_at_fmr(genuine_scores, impostor_scores, target_fmr=1e-3)
client.verify(probe, enrolled, threshold=op.threshold)
```

## Reproduce the experiments

`training/` contains every Colab notebook of the study (FPP01 – FPP15) and the shared libraries. They expect the
MOLF and FingerDOS datasets (obtain them from their providers) in a Google Drive folder named `FingerPrint-Final`;
see [training/README.md](training/README.md). `training/notebooks/FPP16_Export_GitHub_Release.ipynb` rebuilds the
release files from the trained outputs and checks that this client reproduces the saved embeddings.

## Repository layout

```
fpmig/            client package (preprocessing, minutiae parsing, models, client, CLI, calibration)
tests/            unit tests (pytest); model tests need TensorFlow
examples/         end-to-end scripts
docs/RESULTS.md   full result tables
training/         notebooks FPP01–FPP16 and libraries used for the paper
MODEL_CARD.md     intended use, data, metrics, limitations
```

## Responsible use

Fingerprints are sensitive biometric data. Use these models only with the informed consent of the people
enrolled and in line with the data-protection law that applies to you (for example GDPR Art. 9). Store embeddings
with the same protection as the templates they replace (ISO/IEC 24745). See [MODEL_CARD.md](MODEL_CARD.md).

## Citation

```bibtex
@misc{ezz2026bridging,
  title  = {Bridging Minutiae and Deep Embeddings: Fast Fingerprint Verification
            with Cross-Representation Template Migration},
  author = {Ezz, Mohamed},
  year   = {2026},
  note   = {Manuscript under review. Code and models: https://github.com/ezzmohamed2017/fingerprint-minutiae-migration}
}
```

## Licence

Code: MIT ([LICENSE](LICENSE)). Model weights: see [MODEL_LICENSE.md](MODEL_LICENSE.md).
