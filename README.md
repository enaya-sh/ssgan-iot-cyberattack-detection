# SSGAN for IoT Cyberattack Detection

Research-aligned portfolio implementation accompanying my MSc Data Science
and Society thesis at Tilburg University.

## Scope

This codebase separates a clean, readable portfolio implementation from the
original research artifact. It models the thesis architecture and optimization
concepts without claiming byte-for-byte reproduction of every experimental
detail.

## Components

- `src/models.py` — Generator and c+1-class discriminator
- `src/losses.py` — supervised, unsupervised and reconstruction losses
- `src/ssgan.py` — SS-GAN training step
- `src/random_key.py` — mixed-variable Random Key decoding
- `src/improved_abc.py` — research-aligned improved ABC optimizer
- `src/evaluation.py` — Accuracy, F1, Precision, Recall, G-Mean and AUC
- `src/train.py` — compact reference training entry point

## Important Reproducibility Note

The MSc thesis describes dataset-specific preprocessing for NSL-KDD, MAWI and
CICIoT2023, plus 5-fold stratified cross-validation. Those exact experimental
pipelines are not fully recoverable from the current public repository alone.
For that reason, the code here is explicitly labeled a research-aligned
reference implementation.

## Thesis-Reported Results

| Dataset | Accuracy | F1 Score | G-Mean | AUC |
|---|---:|---:|---:|---:|
| NSL-KDD | 86.658% | 87.046% | 88.064% | 0.838 |
| MAWI | 89.054% | 90.030% | 91.128% | 0.852 |
| CICIoT2023 | 87.162% | 88.810% | 89.630% | 0.843 |

These are thesis-reported 5-fold stratified cross-validation results. They are
not outputs claimed from the simplified runner in this repository.

## Run the Reference Implementation

```bash
pip install -r requirements.txt
python -m src.train --csv path/to/clean_numeric_dataset.csv
```

The generic loader expects an already cleaned numeric CSV with the label in the
last column unless `--label-column` is provided.

## Research Context

**Program:** MSc Data Science and Society  
**University:** Tilburg University  
**Research area:** Data Science × Cybersecurity × IoT Security


## Quick Technical Check

```bash
python smoke_test.py
```

The smoke test verifies module imports, one CPU SS-GAN training step,
Random Key decoding, and a small Improved ABC optimization run.

## Scope Boundary

The code is a cleaned, research-aligned portfolio implementation. It does not
claim exact reproduction of the original experiment's dataset-specific
preprocessing, asynchronous generator/discriminator training schedule, or every
implementation choice used to obtain the thesis-reported cross-validation
results.
