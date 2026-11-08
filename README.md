# 🧠 MindMove

**Decoding imagined movement (left hand / right hand / feet / tongue) from EEG with classical,
Riemannian-geometry machine learning, for brain-computer interfaces for paralysed patients.**

* 6 classical pipelines (CSP, FBCSP, MDM, tangent space, and **FBTS+LR**, ours) on BCI Competition IV 2a.
* Official cross-session protocol (train day 1 → test day 2), Cohen's κ, Wilcoxon tests.
* **Riemannian re-centering** for new patients: cross-subject accuracy rises from 41% to 53% with no labels, and calibration needs about half as many trials.
* Robustness to broken electrodes and noise, plus interpretability (ERD maps, CSP patterns, t-SNE).
* Interactive **Streamlit BCI simulator**: a virtual wheelchair driven by real recorded brain signals.

No deep learning in the method. Everything runs on a CPU.

## Results (cross-session, 9 subjects)

| Pipeline | κ (mean ± sd) | accuracy |
|---|---|---|
| **FBTS+LR (ours)** | **0.570 ± 0.194** | 67.8 % |
| TS+SVM | 0.559 ± 0.223 | 66.9 % |
| TS+LR | 0.558 ± 0.217 | 66.8 % |
| FBCSP+SVM | 0.513 ± 0.155 | 63.5 % |
| CSP+LDA | 0.490 ± 0.226 | 61.7 % |
| MDM | 0.432 ± 0.188 | 57.4 % |
| *Competition winner (2008)* | *0.57* | |

TS+LR > CSP+LDA: p = 0.008 (Wilcoxon, 8/9 subjects). FBTS+LR vs FBCSP: +0.058 κ, p = 0.15 (not significant at n = 9).
Leave-one-subject-out TS+LR: 40.9% naive → **53.4% re-centered** (9/9 subjects better).
Robustness: one broken electrode drops CSP+LDA to chance (28.8%). With spherical-spline repair, FBTS+LR keeps 64.1% with 2 broken
channels and 53.8% with 8.
Full write-up: [report/report.md](report/report.md).

## Setup (Windows, Python 3.13)

```bash
python -m venv .venv
```
```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

## Reproduce (run in order from the project root)

```bash
.venv\Scripts\python scripts/01_download_data.py
```
Finds the 18 `.mat` files, or downloads them (MOABB if the BNCI server answers, otherwise the Kaggle mirror
`reader443/bci-competition-iv-dataset-2a-in-mat-format` through the Kaggle CLI). It then caches epochs to
`data/processed/` and verifies 288 trials/session and 72/class. Manual fallback:
```bash
kaggle datasets download -d reader443/bci-competition-iv-dataset-2a-in-mat-format -p data/raw/kaggle --unzip
```

| Step | Command | Time* | Output |
|---|---|---|---|
| Exploration | `.venv\Scripts\python scripts/02_explore.py` | ~1 min | `figures/erd_*.png`, `psd_*.png`, `results/erd_check.csv` |
| Benchmark | `.venv\Scripts\python scripts/03_benchmark.py --jobs 6` | ~25 min | `results/benchmark.csv`, `benchmark_summary.csv`, `stats.csv`, `predictions_cross_session.csv` |
| Transfer | `.venv\Scripts\python scripts/04_transfer.py` | ~5 min | `results/transfer.csv`, `calibration.csv`, `tsne.csv`, `data/processed/covs_8_30.npz` |
| Robustness | `.venv\Scripts\python scripts/05_robustness.py --jobs 9` | ~60 min (use `--repeats 5` for ~15 min) | `results/robustness.csv` |
| Final models | `.venv\Scripts\python scripts/06_train_final.py` | ~2 min | `models/A0x.joblib`, `models/best.json`, CSP/band/confusion figures |

\*on a 12-thread laptop CPU. `06` reads `results/benchmark.csv` (from step 03) and the covariance cache (from step 04).

Tests:
```bash
.venv\Scripts\python -m pytest -q
```

## Demo

```bash
.venv\Scripts\python -m streamlit run app/Home.py
```
Pages: **Brain Signals** (raw EEG / PSD / ERD), **Live BCI** (press ▶ Play: day-2 trials stream, the decoder drives the wheelchair,
threshold slider for safety; choose "LOSO (unseen patient)" for a decoder that never saw that person's labels),
**Model Lab**, **Brain Maps**, **New Patient** (k-trial calibration, live), **Robustness** (break electrodes live).
The app needs steps 01, 03, 04, 05 and 06 to have run. It never downloads anything.

## Layout

```
config.py              paths, channels, classes, windows, bands, seed
src/mindmove/
  data.py              .mat / MOABB loading, filtering, epoching, cache
  features.py          FilterCrop, FilterBank, FilterBankCSP, FilterBankTangentSpace, MIBIF
  pipelines.py         the 6 sklearn pipelines (inner-CV tuning on the training session only)
  evaluate.py          cross-session, within-session CV, kappa, Wilcoxon
  transfer.py          Riemannian re-centering, LOSO, calibration curve, unseen-patient model
  robustness.py        broken electrodes + spherical-spline repair, additive noise
  viz.py               ERD/PSD/topomap/t-SNE/result figures
  stream.py            trial-by-trial BCI stream, safety threshold, wheelchair
scripts/01-06          reproducible experiment steps (07 EEGNet = optional, not implemented)
app/                   Streamlit app (Home + 6 pages)
tests/                 pytest: loader, pipelines on synthetic data, re-centering
report/                report.md, slides_outline.md, viva_qa.md
```

## Data

BCI Competition IV 2a (Tangermann et al., 2012), BNCI Horizon 2020 database, open access. 9 subjects, 22 EEG channels,
250 Hz, 2 sessions × 288 trials. Raw and processed data live in `data/` (git-ignored).
