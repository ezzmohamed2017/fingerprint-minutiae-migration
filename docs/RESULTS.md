# Full results

These are the supplementary tables of the paper, generated from the result workbooks of the training notebooks (`training/`).

All values are mean ± standard deviation over three repetitions with disjoint subject splits (60 training, 15 validation, 5 calibration, 20 test MOLF subjects); FingerDOS is the external test. Metrics are in percent. *Val* = MOLF validation (source-balanced over DB1, DB2 and DB1×DB2); *Test* = MOLF test (source-balanced). Candidates were selected on Val EER only; test and external results are reported for every candidate. Every table is generated directly from the released result workbooks.

## Table S1. FPP04 — backbone comparison (same head, pair loss, upper layers fine-tuned).

| Candidate | Val EER | Val AUC | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 | Params (M) | T4 ms/img | CPU ms/img |
|---|---|---|---|---|---|---|---|---|---|---|
| EfficientNetV2S | 6.17 ± 1.07 | 98.02 ± 0.39 | 7.62 ± 0.11 | 92.11 ± 0.18 | 9.26 ± 0.45 | 4.16 ± 0.31 | 94.75 ± 0.44 | 21.6 | 15.5 | 74.5 |
| VGG16 | 7.43 ± 0.94 | 97.57 ± 0.61 | 8.80 ± 0.35 | 91.14 ± 0.40 | 11.23 ± 1.21 | 4.63 ± 0.12 | 92.81 ± 1.13 | 15.2 | 2.7 | 71.2 |
| ResNet50 | 7.48 ± 0.36 | 97.20 ± 0.33 | 8.89 ± 0.90 | 91.13 ± 0.88 | 11.06 ± 0.93 | 5.04 ± 0.10 | 92.53 ± 0.83 | 25.7 | 8.2 | 40.3 |
| MobileNetV3Large | 7.64 ± 1.15 | 97.33 ± 0.70 | 8.85 ± 0.25 | 90.98 ± 0.34 | 11.23 ± 0.73 | 5.04 ± 0.15 | 92.57 ± 0.41 | 4.0 | 6.2 | 23.5 |
| ViTB16 | 8.20 ± 0.76 | 96.92 ± 0.59 | 10.29 ± 0.41 | 89.46 ± 0.56 | 12.53 ± 0.78 | 5.66 ± 0.10 | 92.72 ± 0.51 | 86.6 | 9.0 | 151.7 |
| DenseNet201 | 9.42 ± 1.56 | 96.29 ± 0.92 | 10.23 ± 1.03 | 89.66 ± 0.94 | 12.62 ± 1.07 | 5.97 ± 0.77 | 91.07 ± 1.55 | 20.3 | 28.2 | 89.6 |
| ConvNeXtTiny | 10.45 ± 0.52 | 95.81 ± 0.47 | 11.77 ± 0.49 | 88.37 ± 0.64 | 14.55 ± 1.47 | 6.41 ± 0.28 | 88.99 ± 1.44 | 28.6 | 8.7 | 207.2 |
| InceptionV3 | 13.73 ± 1.78 | 93.40 ± 1.14 | 14.50 ± 1.50 | 85.45 ± 1.42 | 17.74 ± 1.03 | 9.17 ± 0.45 | 86.96 ± 1.49 | 23.9 | 11.2 | 28.6 |

*Single-image embedding time measured inside the training notebook (warm-up excluded); the final timing study is Table S14.*

## Table S2. FPP04C — full versus partial fine-tuning of the two shortlisted backbones.

| Candidate | Val EER | Val AUC | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 |
|---|---|---|---|---|---|---|---|
| EfficientNetV2S — full fine-tuning (FPP04C) | 5.85 ± 0.87 | 98.17 ± 0.35 | 7.20 ± 0.25 | 92.65 ± 0.32 | 8.96 ± 0.49 | 3.95 ± 0.27 | 94.82 ± 0.60 |
| MobileNetV3Large — full fine-tuning (FPP04C) | 7.37 ± 1.05 | 97.41 ± 0.63 | 8.71 ± 0.25 | 91.22 ± 0.18 | 10.85 ± 0.81 | 5.01 ± 0.24 | 92.51 ± 0.87 |
| EfficientNetV2S — partial fine-tuning (FPP04) | 6.17 ± 1.07 | 98.02 ± 0.39 | 7.62 ± 0.11 | 92.11 ± 0.18 | 9.26 ± 0.45 | 4.16 ± 0.31 | 94.75 ± 0.44 |
| MobileNetV3Large — partial fine-tuning (FPP04) | 7.64 ± 1.15 | 97.33 ± 0.70 | 8.85 ± 0.25 | 90.98 ± 0.34 | 11.23 ± 0.73 | 5.04 ± 0.15 | 92.57 ± 0.41 |

## Table S3. FPP04L — objective ablation on top of FPP04C (shared per-finger cross-sensor proxies).

| Candidate | Val EER | Val AUC | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 |
|---|---|---|---|---|---|---|---|
| EfficientNetV2S + pair loss + shared CosFace proxies | 5.82 ± 1.08 | 98.07 ± 0.68 | 7.18 ± 0.36 | 92.92 ± 0.33 | 8.78 ± 1.29 | 2.87 ± 0.22 | 93.47 ± 2.02 |
| EfficientNetV2S + pair loss only | 5.87 ± 0.79 | 98.14 ± 0.33 | 7.18 ± 0.13 | 92.62 ± 0.20 | 8.94 ± 0.36 | 3.93 ± 0.26 | 94.85 ± 0.37 |
| EfficientNetV2S + pair loss + shared Proxy-Anchor proxies | 6.27 ± 1.09 | 97.69 ± 0.80 | 7.64 ± 0.28 | 92.27 ± 0.32 | 9.73 ± 0.89 | 3.83 ± 0.54 | 93.85 ± 1.49 |
| MobileNetV3Large + pair loss only | 7.32 ± 0.93 | 97.42 ± 0.63 | 8.98 ± 0.18 | 91.01 ± 0.16 | 11.20 ± 0.70 | 5.11 ± 0.06 | 92.34 ± 0.50 |
| MobileNetV3Large + pair loss + shared CosFace proxies | 8.04 ± 0.82 | 97.23 ± 0.65 | 9.17 ± 0.19 | 90.71 ± 0.21 | 11.59 ± 0.65 | 4.90 ± 0.07 | 91.76 ± 0.60 |
| MobileNetV3Large + pair loss + shared Proxy-Anchor proxies | 8.53 ± 1.17 | 96.81 ± 0.80 | 10.27 ± 0.71 | 89.72 ± 0.61 | 13.03 ± 1.29 | 4.78 ± 0.55 | 92.75 ± 0.90 |

*Selected: EfficientNetV2S + pair loss + shared CosFace proxies (image-model winner used in all later kernels).*

## Table S4. FPP05 — minutiae-guided attention on the image model (rejected).

| Candidate | Val EER | Val AUC | Val DB1 EER | Val DB2 EER | Val DB1×DB2 EER | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| FPP04L winner (bce_cosface) | 5.82 ± 1.08 | 98.07 ± 0.68 | 4.93 ± 1.44 | 5.43 ± 0.33 | 7.26 ± 1.54 | 7.18 ± 0.36 | 92.92 ± 0.33 | 8.78 ± 1.29 | 2.87 ± 0.22 | 93.47 ± 2.02 |
| mg_attention | 6.05 ± 1.18 | 97.86 ± 0.81 | 4.67 ± 1.41 | 5.35 ± 0.59 | 7.44 ± 1.60 | 7.24 ± 0.86 | 92.79 ± 0.69 | 9.25 ± 1.80 | 2.86 ± 0.18 | 93.42 ± 2.01 |
| mg_attention_head | 6.06 ± 1.20 | 97.83 ± 0.79 | 4.80 ± 1.31 | 5.54 ± 0.62 | 7.47 ± 1.61 | 7.23 ± 0.90 | 92.82 ± 0.66 | 9.32 ± 1.98 | 2.88 ± 0.15 | 93.37 ± 2.03 |

## Table S5. FPP06 — minutiae encoder architectures.

| Candidate | Val EER | Val AUC | Val DB1 EER | Val DB2 EER | Val DB1×DB2 EER | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| set_transformer | 10.13 ± 0.78 | 95.95 ± 0.71 | 7.98 ± 0.39 | 11.02 ± 1.29 | 11.53 ± 1.11 | 10.97 ± 0.74 | 89.01 ± 0.82 | 12.64 ± 1.48 | 7.27 ± 0.47 | 91.81 ± 0.41 |
| local_structure | 9.99 ± 0.53 | 96.05 ± 0.47 | 8.24 ± 1.12 | 10.28 ± 0.36 | 11.75 ± 0.56 | 10.55 ± 0.30 | 89.41 ± 0.34 | 12.56 ± 0.90 | 5.78 ± 0.27 | 92.95 ± 0.22 |
| FingerJetFX + BOZORTH3 (FPP03) | 6.51 ± 1.45 | 97.51 ± 1.41 | 4.43 ± 0.76 | 5.74 ± 2.29 | 8.59 ± 1.48 | 6.99 ± 1.05 | 92.82 ± 1.52 | 9.73 ± 2.05 | 1.12 ± 0.01 | 97.68 ± 0.90 |
| Image model (FPP04L winner) | 5.82 ± 1.08 | 98.07 ± 0.68 | 4.93 ± 1.44 | 5.43 ± 0.33 | 7.26 ± 1.54 | 7.18 ± 0.36 | 92.92 ± 0.33 | 8.78 ± 1.29 | 2.87 ± 0.22 | 93.47 ± 2.02 |

## Table S6. FPP07 — neighbourhood size k and attention in the local-structure encoder.

| Candidate | Val EER | Val AUC | Val DB1 EER | Val DB2 EER | Val DB1×DB2 EER | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| k16 (FPP06) | 9.99 ± 0.53 | 96.05 ± 0.47 | 8.24 ± 1.12 | 10.28 ± 0.36 | 11.75 ± 0.56 | 10.55 ± 0.30 | 89.41 ± 0.34 | 12.56 ± 0.90 | 5.78 ± 0.27 | 92.95 ± 0.22 |
| k8 | 11.19 ± 0.69 | 95.30 ± 0.49 | 9.43 ± 0.93 | 10.74 ± 0.46 | 13.22 ± 0.45 | 11.73 ± 0.26 | 88.21 ± 0.31 | 13.95 ± 0.74 | 6.22 ± 0.18 | 92.35 ± 0.48 |
| k24 | 8.98 ± 0.59 | 96.61 ± 0.48 | 7.24 ± 0.52 | 9.13 ± 1.14 | 10.35 ± 0.78 | 10.11 ± 0.28 | 89.93 ± 0.34 | 11.95 ± 0.83 | 5.59 ± 0.27 | 93.19 ± 0.45 |
| k32 | 9.02 ± 0.84 | 96.66 ± 0.63 | 7.43 ± 0.52 | 9.28 ± 0.95 | 10.31 ± 1.07 | 9.84 ± 0.20 | 90.10 ± 0.26 | 11.87 ± 0.86 | 5.61 ± 0.26 | 93.19 ± 0.86 |
| k24_attention | 9.56 ± 0.92 | 96.33 ± 0.71 | 8.00 ± 1.57 | 9.63 ± 0.71 | 11.06 ± 1.04 | 10.34 ± 0.18 | 89.70 ± 0.19 | 12.21 ± 0.92 | 5.31 ± 0.27 | 92.96 ± 0.72 |
| FingerJetFX + BOZORTH3 (FPP03) | 6.51 ± 1.45 | 97.51 ± 1.41 | 4.43 ± 0.76 | 5.74 ± 2.29 | 8.59 ± 1.48 | 6.99 ± 1.05 | 92.82 ± 1.52 | 9.73 ± 2.05 | 1.12 ± 0.01 | 97.68 ± 0.90 |
| Image model (FPP04L winner) | 5.82 ± 1.08 | 98.07 ± 0.68 | 4.93 ± 1.44 | 5.43 ± 0.33 | 7.26 ± 1.54 | 7.18 ± 0.36 | 92.92 ± 0.33 | 8.78 ± 1.29 | 2.87 ± 0.22 | 93.47 ± 2.02 |

## Table S7. FPP08 — quality weighting, cropping augmentation and depth (greedy chain).

| Candidate | Val EER | Val AUC | Val DB1 EER | Val DB2 EER | Val DB1×DB2 EER | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| k24 (FPP07) | 8.98 ± 0.59 | 96.61 ± 0.48 | 7.24 ± 0.52 | 9.13 ± 1.14 | 10.35 ± 0.78 | 10.11 ± 0.28 | 89.93 ± 0.34 | 11.95 ± 0.83 | 5.59 ± 0.27 | 93.19 ± 0.45 |
| quality | 9.06 ± 0.48 | 96.54 ± 0.55 | 7.15 ± 0.28 | 9.22 ± 0.68 | 10.41 ± 0.80 | 10.07 ± 0.03 | 89.95 ± 0.08 | 11.80 ± 0.93 | 5.60 ± 0.29 | 93.21 ± 0.54 |
| crop | 8.80 ± 0.81 | 96.76 ± 0.54 | 7.20 ± 0.70 | 9.20 ± 0.98 | 10.32 ± 0.97 | 9.75 ± 0.18 | 90.27 ± 0.22 | 11.44 ± 0.71 | 5.11 ± 0.12 | 93.55 ± 0.55 |
| depth3 | 8.73 ± 0.81 | 96.81 ± 0.55 | 7.24 ± 0.78 | 8.98 ± 0.77 | 10.12 ± 1.06 | 9.61 ± 0.16 | 90.36 ± 0.21 | 11.24 ± 0.87 | 5.10 ± 0.19 | 93.68 ± 0.58 |
| FingerJetFX + BOZORTH3 (FPP03) | 6.51 ± 1.45 | 97.51 ± 1.41 | 4.43 ± 0.76 | 5.74 ± 2.29 | 8.59 ± 1.48 | 6.99 ± 1.05 | 92.82 ± 1.52 | 9.73 ± 2.05 | 1.12 ± 0.01 | 97.68 ± 0.90 |
| Image model (FPP04L winner) | 5.82 ± 1.08 | 98.07 ± 0.68 | 4.93 ± 1.44 | 5.43 ± 0.33 | 7.26 ± 1.54 | 7.18 ± 0.36 | 92.92 ± 0.33 | 8.78 ± 1.29 | 2.87 ± 0.22 | 93.47 ± 2.02 |

## Table S8. FPP09 — metric-learning objectives for the minutiae encoder.

| Candidate | Val EER | Val AUC | Val DB1 EER | Val DB2 EER | Val DB1×DB2 EER | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| depth3 (FPP08) | 8.73 ± 0.81 | 96.81 ± 0.55 | 7.24 ± 0.78 | 8.98 ± 0.77 | 10.12 ± 1.06 | 9.61 ± 0.16 | 90.36 ± 0.21 | 11.24 ± 0.87 | 5.10 ± 0.19 | 93.68 ± 0.58 |
| bce_cosface | 8.24 ± 1.09 | 96.95 ± 0.74 | 6.41 ± 0.81 | 7.96 ± 1.89 | 10.33 ± 1.16 | 9.12 ± 0.77 | 91.07 ± 0.85 | 11.30 ± 1.59 | 3.83 ± 0.66 | 94.10 ± 0.62 |
| bce_proxyanchor | 7.96 ± 1.27 | 96.84 ± 0.95 | 6.24 ± 0.75 | 7.20 ± 1.53 | 10.12 ± 1.72 | 8.66 ± 0.73 | 91.31 ± 0.79 | 11.00 ± 1.56 | 3.66 ± 0.59 | 94.59 ± 1.14 |
| FingerJetFX + BOZORTH3 (FPP03) | 6.51 ± 1.45 | 97.51 ± 1.41 | 4.43 ± 0.76 | 5.74 ± 2.29 | 8.59 ± 1.48 | 6.99 ± 1.05 | 92.82 ± 1.52 | 9.73 ± 2.05 | 1.12 ± 0.01 | 97.68 ± 0.90 |
| Image model (FPP04L winner) | 5.82 ± 1.08 | 98.07 ± 0.68 | 4.93 ± 1.44 | 5.43 ± 0.33 | 7.26 ± 1.54 | 7.18 ± 0.36 | 92.92 ± 0.33 | 8.78 ± 1.29 | 2.87 ± 0.22 | 93.47 ± 2.02 |

*Selected: pair loss + Proxy-Anchor (minutiae-encoder winner).*

## Table S9. FPP10 — cross-space matching with frozen encoders (minutiae template vs image).

| Candidate | Val EER | Val AUC | Val DB1 EER | Val DB2 EER | Val DB1×DB2 EER | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| frozen_as_is | 51.41 ± 0.68 | 48.53 ± 1.17 | 51.46 ± 0.95 | 52.31 ± 1.34 | 50.53 ± 1.02 | 51.05 ± 0.81 | 48.91 ± 0.81 | 50.88 ± 1.14 | 51.42 ± 1.84 | 48.42 ± 1.83 |
| ridge | 15.15 ± 1.45 | 92.14 ± 1.19 | 14.06 ± 2.31 | 15.69 ± 1.10 | 15.53 ± 1.63 | 16.50 ± 1.80 | 83.42 ± 1.72 | 17.46 ± 1.25 | 16.59 ± 1.26 | 83.26 ± 1.01 |
| procrustes | 19.48 ± 1.83 | 88.43 ± 1.70 | 18.00 ± 2.09 | 20.54 ± 2.38 | 19.67 ± 1.73 | 21.46 ± 3.96 | 78.51 ± 3.93 | 22.61 ± 4.12 | 21.49 ± 2.42 | 78.15 ± 2.59 |
| linear_map | 15.33 ± 1.24 | 91.93 ± 1.14 | 14.19 ± 1.95 | 16.26 ± 1.01 | 15.81 ± 1.24 | 16.71 ± 1.60 | 83.26 ± 1.46 | 17.61 ± 1.14 | 16.70 ± 1.35 | 83.18 ± 1.17 |
| mlp_adapter | 14.86 ± 1.18 | 92.38 ± 0.86 | 13.09 ± 0.98 | 15.98 ± 1.14 | 15.47 ± 1.46 | 16.03 ± 2.05 | 83.89 ± 2.15 | 17.19 ± 1.73 | 16.30 ± 1.05 | 83.57 ± 1.02 |
| two_tower | 11.68 ± 1.04 | 94.50 ± 0.77 | 10.52 ± 1.23 | 12.33 ± 1.00 | 12.09 ± 1.14 | 13.12 ± 2.15 | 86.72 ± 2.45 | 14.47 ± 1.61 | 13.50 ± 1.45 | 86.51 ± 1.54 |
| xgboost | 20.88 ± 4.70 | 86.95 ± 4.49 | 19.37 ± 3.77 | 21.78 ± 4.64 | 21.43 ± 5.42 | 21.62 ± 5.77 | 78.37 ± 5.84 | 22.68 ± 5.53 | 23.06 ± 4.42 | 76.78 ± 4.43 |
| Image vs image (FPP04L winner, ceiling) | 5.82 ± 1.08 | 98.07 ± 0.68 | 4.93 ± 1.44 | 5.43 ± 0.33 | 7.26 ± 1.54 | 7.18 ± 0.36 | 92.92 ± 0.33 | 8.78 ± 1.29 | 2.87 ± 0.22 | 93.47 ± 2.02 |
| Minutiae vs minutiae (FPP09 winner, old template) | 7.96 ± 1.27 | 96.84 ± 0.95 | 6.24 ± 0.75 | 7.20 ± 1.53 | 10.12 ± 1.72 | 8.66 ± 0.73 | 91.31 ± 0.79 | 11.00 ± 1.56 | 3.66 ± 0.59 | 94.59 ± 1.14 |
| FingerJetFX + BOZORTH3 (FPP03) | 6.51 ± 1.45 | 97.51 ± 1.41 | 4.43 ± 0.76 | 5.74 ± 2.29 | 8.59 ± 1.48 | 6.99 ± 1.05 | 92.82 ± 1.52 | 9.73 ± 2.05 | 1.12 ± 0.01 | 97.68 ± 0.90 |

*In Tables S9–S12, the rows for image vs image, minutiae vs minutiae and FingerJetFX + BOZORTH3 are fixed references, not candidates.*

## Table S10. FPP11 — fine-tuning the minutiae encoder into the frozen image space.

| Candidate | Val EER | Val AUC | Val DB1 EER | Val DB2 EER | Val DB1×DB2 EER | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| pair_bce_pull | 12.51 ± 1.66 | 93.77 ± 1.34 | 11.15 ± 1.67 | 13.59 ± 1.31 | 12.77 ± 2.40 | 14.14 ± 1.04 | 85.73 ± 1.14 | 15.26 ± 0.98 | 14.42 ± 0.86 | 85.40 ± 0.67 |
| contrastive | 12.18 ± 1.11 | 94.29 ± 0.96 | 10.52 ± 1.76 | 13.28 ± 1.00 | 12.70 ± 1.33 | 13.34 ± 1.87 | 86.57 ± 2.01 | 14.78 ± 2.00 | 13.02 ± 0.48 | 86.89 ± 0.32 |
| finger_centre | 12.61 ± 1.54 | 93.82 ± 1.20 | 10.85 ± 1.34 | 14.06 ± 1.12 | 12.87 ± 2.28 | 14.15 ± 1.33 | 85.72 ± 1.36 | 15.41 ± 0.92 | 14.95 ± 1.18 | 84.84 ± 0.99 |
| best_plus_mm | 11.82 ± 0.69 | 94.51 ± 0.78 | 10.11 ± 1.06 | 12.67 ± 0.35 | 12.41 ± 1.12 | 13.27 ± 1.67 | 86.70 ± 1.57 | 14.56 ± 1.71 | 12.75 ± 0.72 | 87.19 ± 0.84 |
| FPP10 two_tower (image side adapted, reference) | 11.68 ± 1.04 | 94.50 ± 0.77 | 10.52 ± 1.23 | 12.33 ± 1.00 | 12.09 ± 1.14 | 13.12 ± 2.15 | 86.72 ± 2.45 | 14.47 ± 1.61 | 13.50 ± 1.45 | 86.51 ± 1.54 |
| Image vs image (FPP04L winner, ceiling) | 5.82 ± 1.08 | 98.07 ± 0.68 | 4.93 ± 1.44 | 5.43 ± 0.33 | 7.26 ± 1.54 | 7.18 ± 0.36 | 92.92 ± 0.33 | 8.78 ± 1.29 | 2.87 ± 0.22 | 93.47 ± 2.02 |
| Minutiae vs minutiae (FPP09 winner, old template) | 7.96 ± 1.27 | 96.84 ± 0.95 | 6.24 ± 0.75 | 7.20 ± 1.53 | 10.12 ± 1.72 | 8.66 ± 0.73 | 91.31 ± 0.79 | 11.00 ± 1.56 | 3.66 ± 0.59 | 94.59 ± 1.14 |
| FingerJetFX + BOZORTH3 (FPP03) | 6.51 ± 1.45 | 97.51 ± 1.41 | 4.43 ± 0.76 | 5.74 ± 2.29 | 8.59 ± 1.48 | 6.99 ± 1.05 | 92.82 ± 1.52 | 9.73 ± 2.05 | 1.12 ± 0.01 | 97.68 ± 0.90 |

## Table S11. FPP12 — matcher heads on both sides: step A (objectives) and step B (structures).

| Candidate | Val EER | Val AUC | Val DB1 EER | Val DB2 EER | Val DB1×DB2 EER | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| tt_bce_pull | 10.62 ± 0.92 | 95.21 ± 0.60 | 9.78 ± 0.58 | 11.94 ± 0.54 | 10.62 ± 1.46 | 11.85 ± 0.92 | 88.13 ± 0.98 | 12.91 ± 0.98 | 12.87 ± 0.75 | 86.99 ± 0.67 |
| tt_cosface | 16.68 ± 0.85 | 90.86 ± 0.77 | 15.15 ± 0.59 | 17.30 ± 1.43 | 17.71 ± 1.27 | 17.33 ± 2.05 | 82.63 ± 2.07 | 18.78 ± 2.22 | 17.68 ± 1.16 | 82.30 ± 1.12 |
| tt_arcface | 14.98 ± 1.21 | 92.36 ± 0.94 | 13.81 ± 1.17 | 15.24 ± 0.84 | 15.84 ± 1.60 | 16.00 ± 2.23 | 83.90 ± 2.33 | 17.34 ± 2.64 | 16.45 ± 1.35 | 83.54 ± 1.25 |
| tt_proxyanchor | 14.39 ± 1.03 | 92.78 ± 0.84 | 12.67 ± 1.38 | 15.19 ± 1.00 | 15.49 ± 1.20 | 15.30 ± 1.98 | 84.69 ± 2.06 | 16.48 ± 2.22 | 15.11 ± 1.58 | 84.73 ± 1.78 |
| semi_siamese | 10.74 ± 0.87 | 95.02 ± 0.53 | 9.69 ± 0.45 | 11.81 ± 0.87 | 10.73 ± 1.26 | 11.92 ± 0.70 | 87.96 ± 0.87 | 13.11 ± 0.69 | 13.12 ± 0.57 | 86.77 ± 0.47 |
| siamese | 10.59 ± 0.99 | 95.14 ± 0.72 | 9.69 ± 0.59 | 11.46 ± 0.76 | 10.86 ± 1.56 | 11.87 ± 0.87 | 88.04 ± 1.05 | 13.05 ± 0.88 | 12.71 ± 0.79 | 87.13 ± 0.87 |
| FPP10 two_tower (previous winner) | 11.68 ± 1.04 | 94.50 ± 0.77 | 10.52 ± 1.23 | 12.33 ± 1.00 | 12.09 ± 1.14 | 13.12 ± 2.15 | 86.72 ± 2.45 | 14.47 ± 1.61 | 13.50 ± 1.45 | 86.51 ± 1.54 |
| FPP11 best_plus_mm (encoder only, no towers) | 11.82 ± 0.69 | 94.51 ± 0.78 | 10.11 ± 1.06 | 12.67 ± 0.35 | 12.41 ± 1.12 | 13.27 ± 1.67 | 86.70 ± 1.57 | 14.56 ± 1.71 | 12.75 ± 0.72 | 87.19 ± 0.84 |
| Image vs image (FPP04L winner, after re-enrolment) | 5.82 ± 1.08 | 98.07 ± 0.68 | 4.93 ± 1.44 | 5.43 ± 0.33 | 7.26 ± 1.54 | 7.18 ± 0.36 | 92.92 ± 0.33 | 8.78 ± 1.29 | 2.87 ± 0.22 | 93.47 ± 2.02 |
| Minutiae vs minutiae (FPP09 winner, old template) | 7.96 ± 1.27 | 96.84 ± 0.95 | 6.24 ± 0.75 | 7.20 ± 1.53 | 10.12 ± 1.72 | 8.66 ± 0.73 | 91.31 ± 0.79 | 11.00 ± 1.56 | 3.66 ± 0.59 | 94.59 ± 1.14 |
| FingerJetFX + BOZORTH3 (FPP03) | 6.51 ± 1.45 | 97.51 ± 1.41 | 4.43 ± 0.76 | 5.74 ± 2.29 | 8.59 ± 1.48 | 6.99 ± 1.05 | 92.82 ± 1.52 | 9.73 ± 2.05 | 1.12 ± 0.01 | 97.68 ± 0.90 |

*Proxy objectives reached their best validation EER in the first epoch and then degraded (Section 7.4).*

## Table S12. FPP13 — joint training of the minutiae encoder and the Siamese tower on augmented templates.

| Candidate | Val EER | Val AUC | Val DB1 EER | Val DB2 EER | Val DB1×DB2 EER | Test EER | Test F1 | DB1×DB2 test EER | FingerDOS EER | FingerDOS F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| tower_aug | 9.58 ± 0.76 | 95.92 ± 0.49 | 8.57 ± 0.50 | 10.26 ± 1.37 | 9.93 ± 1.49 | 11.04 ± 0.95 | 88.92 ± 1.21 | 12.22 ± 1.20 | 11.25 ± 0.64 | 88.70 ± 0.76 |
| joint | 9.41 ± 0.98 | 95.97 ± 0.66 | 8.43 ± 0.66 | 10.07 ± 1.01 | 9.96 ± 1.13 | 11.04 ± 0.95 | 88.88 ± 1.13 | 12.10 ± 0.88 | 11.67 ± 0.73 | 88.17 ± 0.85 |
| joint_mm | 9.56 ± 0.82 | 95.88 ± 0.63 | 8.72 ± 0.65 | 10.43 ± 1.39 | 9.97 ± 1.24 | 10.85 ± 1.10 | 89.05 ± 1.26 | 12.14 ± 0.98 | 11.70 ± 0.75 | 88.08 ± 0.78 |
| FPP12 siamese (previous winner) | 10.59 ± 0.99 | 95.14 ± 0.72 | 9.69 ± 0.59 | 11.46 ± 0.76 | 10.86 ± 1.56 | 11.87 ± 0.87 | 88.04 ± 1.05 | 13.05 ± 0.88 | 12.71 ± 0.79 | 87.13 ± 0.87 |
| Image vs image (FPP04L winner, after re-enrolment) | 5.82 ± 1.08 | 98.07 ± 0.68 | 4.93 ± 1.44 | 5.43 ± 0.33 | 7.26 ± 1.54 | 7.18 ± 0.36 | 92.92 ± 0.33 | 8.78 ± 1.29 | 2.87 ± 0.22 | 93.47 ± 2.02 |
| Minutiae vs minutiae (FPP09 winner, old template) | 7.96 ± 1.27 | 96.84 ± 0.95 | 6.24 ± 0.75 | 7.20 ± 1.53 | 10.12 ± 1.72 | 8.66 ± 0.73 | 91.31 ± 0.79 | 11.00 ± 1.56 | 3.66 ± 0.59 | 94.59 ± 1.14 |
| FingerJetFX + BOZORTH3 (FPP03) | 6.51 ± 1.45 | 97.51 ± 1.41 | 4.43 ± 0.76 | 5.74 ± 2.29 | 8.59 ± 1.48 | 6.99 ± 1.05 | 92.82 ± 1.52 | 9.73 ± 2.05 | 1.12 ± 0.01 | 97.68 ± 0.90 |

*Selected: joint (first-login matcher used in Section 6).*

## Table S13. Final systems — all metrics on every evaluation set, at the threshold frozen on MOLF validation.

| Set | System | AUC ↑ | EER ↓ | FMR ↓ | FNMR ↓ | F1-macro ↑ | Accuracy ↑ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|---|---|---|---|---|
| MOLF validation (balanced) | Ours: image vs image (FPP05 winner) | 98.07 ± 0.68 | 5.82 ± 1.08 | 5.82 ± 1.08 | 5.83 ± 1.08 | 94.18 ± 1.08 | 94.18 ± 1.08 | 94.18 ± 1.08 | 94.17 ± 1.08 |
| MOLF validation (balanced) | FingerJetFX + BOZORTH3 (FPP03) | 97.51 ± 1.41 | 6.51 ± 1.45 | 6.34 ± 1.36 | 6.68 ± 1.55 | 93.49 ± 1.45 | 93.49 ± 1.45 | 93.63 ± 1.38 | 93.32 ± 1.55 |
| MOLF validation (balanced) | Minutiae encoder (FPP09 winner) | 96.84 ± 0.95 | 7.96 ± 1.27 | 7.96 ± 1.27 | 7.95 ± 1.26 | 92.04 ± 1.27 | 92.04 ± 1.27 | 92.04 ± 1.27 | 92.05 ± 1.26 |
| MOLF validation (balanced) | First login after migration (FPP13 winner) | 95.97 ± 0.66 | 9.41 ± 0.98 | 9.41 ± 0.98 | 9.42 ± 0.98 | 90.59 ± 0.98 | 90.59 ± 0.98 | 90.59 ± 0.98 | 90.58 ± 0.98 |
| MOLF test (balanced) | Ours: image vs image (FPP05 winner) | 97.31 ± 0.38 | 7.18 ± 0.36 | 6.45 ± 1.60 | 7.70 ± 1.81 | 92.92 ± 0.33 | 92.92 ± 0.33 | 93.50 ± 1.38 | 92.30 ± 1.81 |
| MOLF test (balanced) | FingerJetFX + BOZORTH3 (FPP03) | 97.16 ± 0.63 | 6.99 ± 1.05 | 6.56 ± 4.66 | 7.80 ± 2.03 | 92.82 ± 1.52 | 92.82 ± 1.52 | 93.54 ± 4.21 | 92.20 ± 2.03 |
| MOLF test (balanced) | Minutiae encoder (FPP09 winner) | 96.62 ± 0.48 | 8.66 ± 0.73 | 8.94 ± 2.79 | 8.44 ± 1.50 | 91.31 ± 0.79 | 91.31 ± 0.79 | 91.17 ± 2.42 | 91.56 ± 1.50 |
| MOLF test (balanced) | First login after migration (FPP13 winner) | 94.90 ± 0.75 | 11.04 ± 0.95 | 11.10 ± 0.92 | 11.14 ± 2.01 | 88.88 ± 1.13 | 88.88 ± 1.13 | 88.89 ± 0.86 | 88.86 ± 2.01 |
| DB1_test | Ours: image vs image (FPP05 winner) | 97.97 ± 0.32 | 5.83 ± 0.58 | 6.89 ± 1.76 | 5.50 ± 0.82 | 93.80 ± 0.88 | 93.81 ± 0.88 | 93.23 ± 1.58 | 94.50 ± 0.82 |
| DB1_test | FingerJetFX + BOZORTH3 (FPP03) | 97.90 ± 0.52 | 5.67 ± 0.90 | 7.22 ± 6.16 | 5.47 ± 1.43 | 93.64 ± 2.58 | 93.65 ± 2.57 | 93.16 ± 5.34 | 94.53 ± 1.43 |
| DB1_test | Minutiae encoder (FPP09 winner) | 97.54 ± 0.61 | 7.36 ± 1.13 | 10.17 ± 3.00 | 5.69 ± 0.92 | 92.06 ± 1.11 | 92.07 ± 1.10 | 90.33 ± 2.55 | 94.31 ± 0.92 |
| DB1_test | First login after migration (FPP13 winner) | 95.73 ± 1.08 | 9.96 ± 1.42 | 10.39 ± 0.85 | 9.78 ± 2.69 | 89.92 ± 1.65 | 89.92 ± 1.65 | 89.67 ± 0.95 | 90.22 ± 2.69 |
| DB2_test | Ours: image vs image (FPP05 winner) | 98.03 ± 0.29 | 5.74 ± 0.28 | 6.69 ± 1.51 | 5.17 ± 1.31 | 94.07 ± 0.32 | 94.07 ± 0.32 | 93.43 ± 1.33 | 94.83 ± 1.31 |
| DB2_test | FingerJetFX + BOZORTH3 (FPP03) | 98.05 ± 0.52 | 5.26 ± 1.55 | 5.94 ± 4.06 | 5.25 ± 1.66 | 94.40 ± 1.80 | 94.40 ± 1.80 | 94.22 ± 3.81 | 94.75 ± 1.66 |
| DB2_test | Minutiae encoder (FPP09 winner) | 97.45 ± 0.10 | 7.18 ± 0.21 | 8.28 ± 2.68 | 6.58 ± 1.72 | 92.57 ± 0.48 | 92.57 ± 0.48 | 91.93 ± 2.27 | 93.42 ± 1.72 |
| DB2_test | First login after migration (FPP13 winner) | 94.82 ± 0.86 | 11.19 ± 0.98 | 11.31 ± 1.35 | 10.81 ± 2.17 | 88.94 ± 1.25 | 88.94 ± 1.25 | 88.76 ± 1.22 | 89.19 ± 2.17 |
| DB1xDB2_test | Ours: image vs image (FPP05 winner) | 96.00 ± 1.12 | 8.78 ± 1.29 | 5.76 ± 1.84 | 12.45 ± 4.39 | 90.88 ± 1.49 | 90.90 ± 1.47 | 93.89 ± 1.55 | 87.55 ± 4.39 |
| DB1xDB2_test | FingerJetFX + BOZORTH3 (FPP03) | 95.50 ± 1.91 | 9.73 ± 2.05 | 6.51 ± 4.63 | 12.68 ± 3.54 | 90.39 ± 1.76 | 90.41 ± 1.76 | 93.27 ± 4.38 | 87.32 ± 3.54 |
| DB1xDB2_test | Minutiae encoder (FPP09 winner) | 94.95 ± 1.61 | 11.00 ± 1.56 | 8.38 ± 2.75 | 13.03 ± 2.67 | 89.29 ± 1.20 | 89.30 ± 1.20 | 91.29 ± 2.48 | 86.97 ± 2.67 |
| DB1xDB2_test | First login after migration (FPP13 winner) | 94.12 ± 0.80 | 12.10 ± 0.88 | 11.61 ± 0.56 | 12.84 ± 2.33 | 87.77 ± 1.02 | 87.77 ± 1.02 | 88.24 ± 0.41 | 87.16 ± 2.33 |
| FingerDOS_test | Ours: image vs image (FPP05 winner) | 99.44 ± 0.09 | 2.87 ± 0.22 | 11.94 ± 4.14 | 1.08 ± 0.26 | 93.47 ± 2.02 | 93.49 ± 2.00 | 89.32 ± 3.25 | 98.92 ± 0.26 |
| FingerDOS_test | FingerJetFX + BOZORTH3 (FPP03) | 99.60 ± 0.00 | 1.12 ± 0.01 | 3.77 ± 1.94 | 0.87 ± 0.14 | 97.68 ± 0.90 | 97.68 ± 0.90 | 96.36 ± 1.80 | 99.13 ± 0.14 |
| FingerDOS_test | Minutiae encoder (FPP09 winner) | 99.19 ± 0.12 | 3.66 ± 0.59 | 8.86 ± 2.27 | 1.94 ± 0.22 | 94.59 ± 1.14 | 94.60 ± 1.14 | 91.74 ± 1.92 | 98.06 ± 0.22 |
| FingerDOS_test | First login after migration (FPP13 winner) | 94.66 ± 0.56 | 11.67 ± 0.73 | 10.95 ± 0.65 | 12.72 ± 1.64 | 88.17 ± 0.85 | 88.17 ± 0.84 | 88.86 ± 0.60 | 87.28 ± 1.64 |

*FMR, FNMR, F1, accuracy, precision and recall are computed at the source-balanced validation threshold; AUC and EER are threshold-free.*

## Table S14. Daily verification latency (Intel Xeon 2.0 GHz, 8 cores; NVIDIA Tesla T4).

| Method | Hardware | Template creation (ms) | 1:1 comparison (ms) | Probe to decision (ms) | Speed-up vs legacy |
|---|---|---|---|---|---|
| FingerJetFX + BOZORTH3 (old system) | CPU 1 thread | 14.26 ± 0.11 | 34.953 ± 0.072 | 49.22 ± 0.09 | 1.0 |
| Ours: letterbox + image model + cosine | Tesla T4 fp32 | 16.26 ± 0.35 | 0.0024 | 16.26 ± 0.35 | 3.03 |
| Ours: letterbox + image model + cosine | Tesla T4 fp32 + XLA | 7.43 ± 0.05 | 0.0024 | 7.43 ± 0.05 | 6.63 |
| Ours: letterbox + image model + cosine | Tesla T4 fp16 + XLA | 7.77 ± 0.21 | 0.0024 | 7.77 ± 0.21 | 6.33 |
| Ours: letterbox + image model + cosine | CPU 1 thread | 102.22 ± 1.05 | 0.0024 | 102.22 ± 1.05 | 0.48 |
| Ours: letterbox + image model + cosine | CPU 1 thread + XLA | 459.88 ± 10.68 | 0.0024 | 459.88 ± 10.68 | 0.11 |
| Ours: letterbox + image model + cosine | CPU 8 threads | 74.14 ± 1.16 | 0.0024 | 74.14 ± 1.16 | 0.66 |

*Median over repeated runs, warm-up excluded; each configuration in its own process.*

## Table S15. Batch migration cost (minutiae encoder + Siamese tower, per stored template).

| Hardware | Per template, single (ms) | Per template, batch 64 (ms) | Minutes per million (batch 64) |
|---|---|---|---|
| T4 GPU fp32 | 2.92 ± 0.08 | 2.287 ± 0.130 | 38.1 |
| T4 GPU fp16 | 2.98 ± 0.02 | 2.541 ± 0.008 | 42.3 |
| CPU 1 thread | 81.09 ± 0.16 | 96.010 ± 0.208 | 1600.2 |
| CPU 8 threads | 62.10 ± 0.40 | 57.042 ± 0.075 | 950.7 |

## Table S16. First-login latency (once per user; without XLA).

| Method | Hardware | Template creation (ms) | Comparison (ms) | Probe to decision (ms) | Speed-up vs legacy |
|---|---|---|---|---|---|
| letterbox + image model + Siamese tower + cosine | T4 GPU fp32 | 21.92 ± 0.21 | 0.0028 | 21.92 ± 0.21 | 2.6× |
| letterbox + image model + Siamese tower + cosine | T4 GPU fp16 | 23.06 ± 0.25 | 0.0028 | 23.07 ± 0.25 | 2.5× |
| letterbox + image model + Siamese tower + cosine | CPU 1 thread | 124.92 ± 0.08 | 0.0028 | 124.93 ± 0.08 | 0.5× |
| letterbox + image model + Siamese tower + cosine | CPU 8 threads | 86.91 ± 0.59 | 0.0028 | 86.91 ± 0.59 | 0.7× |

## Table S17. Model complexity.

| Model | Parameters (M) | GFLOPs per item | Size fp32 (MB) | Size fp16 (MB) |
|---|---|---|---|---|
| image model | 21.65 | 5.755 | 82.6 | 41.3 |
| Siamese tower (shared by both sides) | 4.2 | 0.008 | 16.0 | 8.0 |
| minutiae encoder | 1.72 | 2.196 | 6.6 | 3.3 |

## Table S18. Paired differences against FingerJetFX + BOZORTH3, computed repetition by repetition (ours − baseline; for EER a negative value favours ours).

| Set | Metric | Mean difference (points) | Std | Reps where ours is better |
|---|---|---|---|---|
| DB1xDB2_test | EER | -0.96 | 1.21 | 2/3 |
| DB1xDB2_test | F1-macro | 0.49 | 1.59 | 2/3 |
| DB1xDB2_test | AUC | 0.5 | 0.91 | 2/3 |
| FingerDOS_test | EER | 1.75 | 0.23 | 0/3 |
| FingerDOS_test | F1-macro | -4.21 | 1.18 | 0/3 |
| FingerDOS_test | AUC | -0.16 | 0.09 | 0/3 |
| MOLF test (balanced) | EER | 0.19 | 0.81 | 2/3 |
| MOLF test (balanced) | F1-macro | 0.11 | 1.46 | 2/3 |
| MOLF test (balanced) | AUC | 0.15 | 0.3 | 2/3 |
| MOLF validation (balanced) | EER | -0.69 | 0.44 | 3/3 |
| MOLF validation (balanced) | F1-macro | 0.69 | 0.44 | 3/3 |
| MOLF validation (balanced) | AUC | 0.56 | 0.75 | 2/3 |

## Table S19. Threshold calibration check for the three selected models

| Model | Threshold (validation) | Threshold (calibration, 5 subj.) | Test FMR, val. thr. | Test FMR, cal. thr. | Test FNMR, val. thr. | Test FNMR, cal. thr. | Test F1, val. thr. | Test F1, cal. thr. |
|---|---|---|---|---|---|---|---|---|
| Image model (EfficientNetV2S + pair loss + shared CosFace proxies) | 0.272 ± 0.058 | 0.273 ± 0.031 | 6.45 | 6.65 | 7.70 | 8.25 | 92.92 | 92.55 |
| Minutiae encoder (pair loss + Proxy-Anchor) | 0.526 ± 0.024 | 0.535 ± 0.019 | 8.94 | 8.72 | 8.44 | 9.06 | 91.31 | 91.11 |
| First-login matcher (FPP13 joint) | 0.486 ± 0.034 | 0.448 ± 0.036 | 11.10 | 12.65 | 11.14 | 9.53 | 88.88 | 88.90 |
