## Research Implementation Notes

This repository contains a cleaned, research-aligned reference implementation
of the SS-GAN architecture described in my MSc thesis.

The thesis methodology combines:

- semi-supervised GAN classification with `c + 1` discriminator outputs;
- supervised and unsupervised discriminator losses;
- generator adversarial loss;
- discriminator-feature-guided reconstruction loss;
- Random Key encoding for mixed hyperparameters; and
- an enhanced Artificial Bee Colony (ABC) optimization strategy.

The original academic experiments used dataset-specific preprocessing and
5-fold stratified cross-validation across NSL-KDD, MAWI and CICIoT2023.
The cleaned portfolio implementation is intentionally separated from the
original experimental pipeline and should not be interpreted as a
byte-for-byte reproduction of every thesis experiment.

## Thesis-Reported Results

| Dataset | Accuracy | F1 Score | G-Mean | AUC |
|---|---:|---:|---:|---:|
| NSL-KDD | 86.658% | 87.046% | 88.064% | 0.838 |
| MAWI | 89.054% | 90.030% | 91.128% | 0.852 |
| CICIoT2023 | 87.162% | 88.810% | 89.630% | 0.843 |

These values are reported in the submitted MSc thesis and are not generated
by the simplified portfolio runner included in this repository.
