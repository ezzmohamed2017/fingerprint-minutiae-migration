# Model card — fpmig v1.0.0

## Overview

Three networks that together migrate a fingerprint verification system from stored minutiae templates to deep
image embeddings:

| Model | Input | Output | Role |
|---|---|---|---|
| Image encoder | original greyscale capture, letterboxed to 224 × 224 (uint8) | L2-normalised 1024-d vector | enrolment and daily 1:1 verification (cosine) |
| Minutiae encoder | up to 128 minutiae (position, direction, type) from an ISO/IEC 19794-2 template | 1024-d vector | batch migration of stored templates |
| Siamese tower | 1024-d vector of either modality, with a learned offset per modality | L2-normalised 1024-d vector in a shared space | first-login comparison between an image and a migrated template |

* **Image encoder:** EfficientNetV2-S (ImageNet initialisation), fully fine-tuned; dense 1024 + batch normalisation +
  L2; trained with a calibrated pair loss plus shared per-finger CosFace proxies across the two MOLF sensors
  (kernels FPP04 – FPP04L; the FPP05 attention variant was rejected on validation).
* **Minutiae encoder:** permutation-invariant local-structure encoder (k = 24 nearest neighbours described in each
  minutia's own frame, 3 edge layers, max/mean pooling); Proxy-Anchor objective (FPP06 – FPP09), then trained
  jointly with the Siamese tower on augmented templates (FPP13).
* **Siamese tower:** Dense 2048 GELU → dropout 0.1 → Dense 1024 → L2, shared by both modalities (FPP12 – FPP13).

## Training and evaluation data

* **MOLF** (IIIT-Delhi), optical flat prints from two sensors: DB1 (Lumidigm) and DB2 (SecuGen). Subject-disjoint
  splits: 60 training / 15 validation / 5 calibration / 20 test subjects, three repetitions with different splits.
* **FingerDOS**, 60 subjects: external test only, never used for training, selection or thresholds.
* Minutiae: FingerJetFX OSE (pinned commit), ISO/IEC 19794-2:2005, extracted at native resolution.
* The released weights are those of **one repetition** (named in `config.json` → `provenance`); results below are
  mean ± std over the three repetitions.

## Metrics (test subjects, at the threshold frozen on validation)

| Scenario | EER | AUC | FMR | FNMR | F1 |
|---|---|---|---|---|---|
| Daily login, image vs image — MOLF test | 7.18 ± 0.36 | 97.31 | 6.45 | 7.70 | 92.92 |
| Daily login — cross-sensor DB1 × DB2 | 8.78 ± 1.29 | 96.00 | 5.76 | 12.45 | 90.88 |
| Daily login — FingerDOS (unseen sensor) | 2.87 ± 0.22 | 99.44 | **11.94** | 1.08 | 93.47 |
| First login, image vs migration template — MOLF test | 11.04 ± 0.95 | 94.90 | 11.10 | 11.14 | 88.88 |
| First login — FingerDOS | 11.67 ± 0.73 | 94.66 | 10.95 | 12.72 | 88.17 |
| Reference: FingerJetFX + BOZORTH3 — MOLF test | 6.99 ± 1.05 | 97.16 | 6.56 | 7.80 | 92.82 |
| Reference: FingerJetFX + BOZORTH3 — FingerDOS | 1.12 ± 0.01 | 99.60 | 3.77 | 0.87 | 97.68 |

Latency, one probe, warm-up excluded: image model 7.43 ms on an NVIDIA T4 with XLA (16.3 ms without XLA; 74 ms on
8 CPU threads); FingerJetFX + BOZORTH3 49.2 ms on one CPU thread; migration 2.3 ms per template (T4, batch 64).

## Intended use

* 1:1 fingerprint verification after migrating an existing minutiae-template population, in systems where the
  operator has a lawful basis and the users' consent to process their biometric data.
* Research on fixed-length fingerprint representations and on migrating legacy biometric templates.

## Out-of-scope use

* 1:N identification or forensic / latent comparison (not evaluated).
* Covert identification, surveillance, or any use without the knowledge and consent of the people concerned.
* Deployment with the validation thresholds shipped in `config.json` on a different sensor without calibration.

## Limitations

* Trained on 60 subjects from two optical flat-print sensors; performance on capacitive, contactless, rolled or
  low-quality captures is unknown.
* Thresholds do not transfer across sensors (FingerDOS FMR 11.9 % at the MOLF threshold). Calibrate on the target
  sensor with `fpmig.calibrate`.
* The first-login (cross-space) comparison is about four EER points worse than image vs image; use it once per
  user, with a low-FMR threshold, limited attempts and, where available, confirmation by the legacy matcher.
* Only FingerJetFX templates were evaluated; other extractors place minutiae and directions differently.
* Latency was measured on a T4 GPU as a proxy for a terminal accelerator; int8 quantisation was not evaluated.
* Small validation set (15 subjects): differences below about one EER point between configurations are ties.

## Privacy and security

Embeddings are biometric data. They can leak information about the finger and must be protected like the
templates they replace (ISO/IEC 24745): encryption at rest, access control, and template protection where required.
The release contains no images, templates or embeddings of any data subject.

## Contact and citation

Mohamed Ezz — see `CITATION.cff`. Issues and questions: the GitHub issue tracker of this repository.
