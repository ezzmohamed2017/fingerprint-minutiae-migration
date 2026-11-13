# Training and evaluation notebooks

Every experiment of the paper is one Google Colab notebook ("kernel"). Each kernel starts from the saved outputs of
the previous winner, selects on the source-balanced validation EER only, and reports test and FingerDOS results for
every candidate (three repetitions, subject-disjoint splits).

| Kernel | Notebook | Question |
|---|---|---|
| FPP01 | `FPP01_Evaluation_Protocol_Subject_Disjoint_Splits` | protocol: splits, pairs, shared library |
| FPP02 | `FPP02_FingerJetFX_Minutiae_and_Image_Caches` | data preparation, FingerJetFX templates, 224 × 224 image cache |
| FPP03 | `FPP03_BOZORTH3_Minutiae_Verification_Baseline` | FingerJetFX + BOZORTH3 baseline |
| FPP04 | `FPP04_Image_Siamese_Backbone_Ablation` | eight backbones |
| FPP04C | `FPP04C_Selected_Backbone_Full_FineTuning_Augmentation` | full fine-tuning |
| FPP04L | `FPP04L_Shared_Identity_Proxy_Losses` | shared cross-sensor proxies (image winner) |
| FPP05 | `FPP05_Minutiae_Guided_Early_Layer_Attention` | minutiae-guided attention (rejected) |
| FPP06 – FPP09 | minutiae encoder notebooks | architecture, neighbourhood, augmentation, objective |
| FPP10 – FPP13 | migration notebooks | mappings, fine-tuning, Siamese heads, joint training |
| FPP14 – FPP15 | timing and report | accelerator / CPU timing, final statistics |
| FPP16 | `FPP16_Export_GitHub_Release` | export the release files and verify this client against the saved embeddings |

## Setup

1. Obtain MOLF (IIIT-Delhi) and FingerDOS from their providers and place them under
   `My Drive/FingerPrint-Final/Datasets/` (see FPP02 for the expected layout).
2. Copy `lib/fpf_lib.py`, `lib/fpf_tf.py` and `lib/fpp_lib.py` to `FingerPrint-Final/output/00_code/`.
3. Run the notebooks in order on Colab (A100 for training, T4 for FPP14 – FPP16).
