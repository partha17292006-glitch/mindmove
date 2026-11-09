# MindMove — Decoding Imagined Movement from Brain Signals with Classical (Riemannian) ML

> Execution plan for a fresh Claude Code session. Project root: `C:\Users\hp\Desktop\ml_project` (currently empty, not a git repo).
> Owner: Partha (AI/ML student). Deadline: **2 days**. Course = **ML project (30%)** → core method must be **classical ML** (no deep learning in the main method).

---

## 1. Context — why this project

- **Real problem:** People with ALS, spinal-cord injury or locked-in syndrome cannot move, but their motor cortex still produces *movement intentions*. A Brain-Computer Interface (BCI) decodes those intentions from EEG so a patient can steer a wheelchair or cursor by *imagining* movement.
- **The ML challenge:** EEG is tiny (µV), noisy, non-stationary, and differs hugely between people and even between days → classic "train on data, give output" ML fails. Real BCIs also need long calibration sessions for every new patient (exhausting for them).
- **What we build:** a full motor-imagery decoder (left hand / right hand / feet / tongue) using **Riemannian geometry** (EEG trials as covariance matrices living on a curved manifold of SPD matrices), benchmarked against the classical **CSP / FBCSP** methods that won BCI Competition IV — plus **cross-subject transfer** (cut calibration for a new patient), **robustness** to bad electrodes, **neuro-interpretability**, and an interactive **Streamlit BCI simulator** where a virtual wheelchair is driven by real recorded brain signals.
- **Why it impresses:** research-level, out of the box, mathematically deep (manifolds, spatial filtering, information theory), purely classical ML, and every claim is backed by an experiment.

---

## 2. Dataset — where it comes from (read this first)

### Primary: BCI Competition IV, Dataset 2a (a.k.a. `BNCI2014_001`)
| Property | Value |
|---|---|
| Subjects | 9 healthy adults (A01–A09) |
| Sessions | 2 per subject, on different days: **T** (training) and **E** (evaluation) |
| Trials | 288 per session (72 per class), 6 runs × 48 trials |
| Classes | 4 motor-imagery: left hand, right hand, both feet, tongue |
| Channels | 22 EEG (10-20 montage around motor cortex) + 3 EOG (ignored) |
| Sampling rate | 250 Hz, band-pass 0.5–100 Hz, 50 Hz notch |
| Trial timeline | t=0 fixation cross, **t=2 s cue appears**, t=2–6 s subject imagines movement |
| License | Open access (BNCI Horizon 2020 database) |

**Download route A — MOABB (automatic, preferred):**
```python
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import moabb.utils
moabb.utils.set_download_dir("data/raw")          # keep data inside the project
ds = BNCI2014_001()
paradigm = MotorImagery(n_classes=4, fmin=8, fmax=30, tmin=0.5, tmax=2.5)
X, y, meta = paradigm.get_data(ds, subjects=[1])   # X: (576, 22, 501), meta has session column
```
Files come from `bnci-horizon-2020.eu/database/data-sets/001-2014/A0xT.mat / A0xE.mat` (~43 MB each, ~800 MB total).
⚠️ On 2026-11-09 that server **timed out** from this machine. If it still fails → route B.

**Download route B — Kaggle mirror (verified to exist, same .mat files incl. true labels for E sessions):**
```bash
kaggle datasets download -d reader443/bci-competition-iv-dataset-2a-in-mat-format -p data/raw/kaggle --unzip
```
Contains `BCICIV-2a-mat/A01T.mat … A09E.mat` (18 files, ~770 MB). Kaggle CLI 2.2.5 is already installed and configured.
Then either (a) copy files into MOABB's cache folder `data/raw/MNE-bnci-data/database/data-sets/001-2014/` so MOABB picks them up, or (b) use our own loader `load_bnci_mat()` (see §5, `data.py`) which parses the struct with `scipy.io.loadmat`:
`data` = cell of runs; each run has `X` (samples × 25, µV), `trial` (cue onset sample indices), `y` (labels 1–4), `fs`, `artifacts`. Use only runs with non-empty `trial` (the first 3 runs of each file are EOG calibration runs), first 22 columns = EEG.
**Verify** after download: each session → 288 trials, 72/class, fs=250.

**Do NOT use** the official bbci.de GDF files — the evaluation sessions there lack labels.

### Optional secondary (stretch only): PhysioNet EEG Motor Movement/Imagery (109 subjects)
`mne.datasets.eegbci.load_data(subject, runs=[4,8,12])` — server verified reachable (HTTP 200). Use only if time remains, for a larger cross-subject experiment.

---

## 3. Environment

- Python **3.13.1** (`python`), Windows 11, **no GPU**, 16 GB RAM. Everything here runs on CPU in minutes.
- Create a venv in the project: `python -m venv .venv` → `.venv\Scripts\python -m pip install -r requirements.txt`
- `requirements.txt` (dry-run install on Py 3.13 verified OK):
  ```
  mne==1.13.2
  moabb==1.7.2
  pyriemann==0.12
  scikit-learn>=1.7
  numpy>=2.3
  scipy>=1.16
  pandas>=2.3
  matplotlib>=3.10
  seaborn>=0.13
  plotly>=5
  streamlit>=1.60
  joblib
  pytest
  ```
- Optional (only for the DL baseline, §6.7): `torch` (already installed globally, 2.8.0) — keep it out of requirements, put in `requirements-optional.txt`.
- Global seed `42` everywhere. Never read/print `.env` or API keys (Kaggle token lives in `~/.kaggle`, never touch it).

---

## 4. Project structure

```
ml_project/
├── PLAN.md                     # this plan
├── README.md                   # how to install, run, demo; results summary
├── requirements.txt / requirements-optional.txt
├── config.py                   # paths, FS=250, CHANNELS, CLASSES, time window, filter bands, SEED
├── src/mindmove/
│   ├── __init__.py
│   ├── data.py                 # loading (MOABB + .mat fallback), epoching, caching to data/processed/*.npz
│   ├── features.py             # FilterBankCSP transformer (+ MIBIF selection), FilterBankTangentSpace
│   ├── pipelines.py            # get_pipelines() -> dict[name -> sklearn Pipeline]
│   ├── evaluate.py             # protocols: cross-session, within-session CV, LOSO; kappa/acc/confusion; Wilcoxon
│   ├── transfer.py             # Riemannian re-centering, calibration-size curve
│   ├── robustness.py           # noise injection, electrode dropout + spherical-spline interpolation
│   ├── viz.py                  # topomaps, ERD/ERS maps, tangent-space t-SNE, confusion, bar charts
│   └── stream.py               # trial-by-trial BCI stream simulator + decision smoothing
├── scripts/
│   ├── 01_download_data.py     # route A, falls back to printing route B instructions / using kaggle files
│   ├── 02_explore.py           # figures: raw EEG, PSD, ERD/ERS at C3/Cz/C4 per class
│   ├── 03_benchmark.py         # all pipelines × all subjects × protocols -> results/benchmark.csv
│   ├── 04_transfer.py          # LOSO + calibration curve -> results/transfer.csv, calibration.csv
│   ├── 05_robustness.py        # -> results/robustness.csv
│   ├── 06_train_final.py       # fit best pipeline per subject on session T -> models/subjXX.joblib
│   └── 07_eegnet_baseline.py   # OPTIONAL DL comparison only
├── app/
│   ├── Home.py                 # streamlit entry: `streamlit run app/Home.py`
│   └── pages/ 1_Brain_Signals.py, 2_Live_BCI.py, 3_Model_Lab.py, 4_Brain_Maps.py, 5_New_Patient.py, 6_Robustness.py
├── tests/ test_data.py, test_pipelines.py, test_transfer.py
├── results/  figures/  models/  data/      # generated; add .gitignore for data/ and models/
└── report/ report.md, slides_outline.md, viva_qa.md
```

---

## 5. Data pipeline (`src/mindmove/data.py`)

- `load_subject(subject: int) -> dict(session -> (X, y))` with X shape `(288, 22, n_times)` in **volts**, y ∈ {left_hand, right_hand, feet, tongue}.
  - Get **continuous raw** data, then epoch ourselves (one code path for both download routes):
    - Route A: `BNCI2014_001().get_data(subjects=[s])` → `{subject: {session: {run: mne.io.Raw}}}`; pick 22 EEG channels, concatenate the 6 runs per session, events via `mne.events_from_annotations`.
  - Fallback `load_bnci_mat(path)` → MNE `RawArray` with standard 10-20 channel names (`Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz`), µV→V, `set_montage("standard_1020")`, events from `trial`/`y`, `mne.Epochs`.
- Epoch a wide cache **−1.5 s … +4.0 s around the cue**, filtered 1–45 Hz (baseline needed for ERD maps, full imagery period for the stream). Pipelines crop and band-pass from this cache, so filtering is done per pipeline.
- **Classification window:** 0.5–2.5 s after cue (**why:** first 0.5 s contains the visual-evoked response to the cue, not imagery; 2 s gives stable covariance; standard in the literature).
- Cache each subject as `data/processed/A0x.npz` (+ `mne.Info` pickled) so the app never downloads.
- Keep artifact-marked trials (honest, like the competition), but store the artifact flag in metadata.

---

## 6. ML methods

### 6.1 Pipelines (`pipelines.py`) — all scikit-learn compatible
| Name | Steps | Why it's included |
|---|---|---|
| `CSP+LDA` | band-pass 8–30 Hz → `mne.decoding.CSP(n_components=8, log=True)` → LDA | Textbook baseline. 8–30 Hz = mu (8–13) + beta (13–30) rhythms that desynchronize during motor imagery (ERD). |
| `FBCSP+SVM` | filter bank 9 bands (4–8, 8–12 … 36–40 Hz) → CSP per band → **MIBIF** (mutual-information feature selection, `SelectKBest(mutual_info_classif, k)`) → linear SVM | **The method that won BCI Competition IV 2a (Ang et al., 2008, κ = 0.57)** — our reference to beat. |
| `MDM` | 8–30 Hz → `pyriemann.estimation.Covariances('oas')` → `pyriemann.classification.MDM(metric='riemann')` | Pure geometry: classify by Riemannian distance to each class's geometric mean. Zero hyper-parameters. |
| `TS+LR` | 8–30 Hz → Covariances('oas') → `TangentSpace(metric='riemann')` → LogisticRegression(C tuned) | Projects the curved manifold onto a flat tangent space at the mean → any Euclidean classifier works. State of the art for classical BCI. |
| `TS+SVM` | same → SVC(linear) | Classifier comparison. |
| **`FBTS+LR` (MindMove, ours)** | filter bank (≈6 bands 4–40 Hz) → Covariances('oas') per band → TangentSpace per band → concatenate → StandardScaler → LogisticRegression (elastic-net / L2, C tuned) | Combines FBCSP's multi-band idea with Riemannian geometry. Expected best. |

- **OAS shrinkage covariance** (**why:** 22×22 covariance from 500 samples is noisy; shrinkage guarantees positive-definiteness needed for Riemannian ops).
- Hyper-parameters tuned with `GridSearchCV(cv=5)` **only on the training session** (no test leakage).

### 6.2 Evaluation protocols (`evaluate.py`)
1. **Cross-session (official competition protocol, headline result):** train on session T, test on session E, per subject. Metrics: **Cohen's kappa** (official metric), accuracy, confusion matrix. Report mean ± std over 9 subjects and compare to the winner κ=0.57.
2. **Within-session 5-fold stratified CV** (sanity check, shows the session-to-session drop = non-stationarity).
3. **Cross-subject Leave-One-Subject-Out** (train on 8, test on 1) — naive vs. re-centered (§6.3).
- Statistics: paired **Wilcoxon signed-rank** across subjects for FBTS+LR vs FBCSP and TS+LR vs CSP+LDA (`scipy.stats.wilcoxon`); state n=9 caveat honestly.
- Expected ballpark (literature): CSP+LDA ~60–65 % acc, FBCSP ~67 % (κ≈0.57), MDM ~60–65 %, TS+LR ~68–72 %. Do not hard-code; report what we get.

### 6.3 Transfer learning / "New patient" (`transfer.py`)
- **Riemannian re-centering (Zanini et al., 2018):** for each subject/session, compute the Riemannian mean M of its covariances and transform `C → M^{-1/2} C M^{-1/2}` so every subject's data is centered at the identity. Use `pyriemann.transfer` (`TLCenter`, optionally `TLStretch`, `TLRotate`) or implement directly with `pyriemann.utils.mean.mean_riemann` + `invsqrtm`.
  **Why:** the main difference between people is a "shift" on the manifold; re-centering removes it without needing labels from the new person (re-centering is unsupervised → only needs unlabeled resting/trial data).
- **Calibration curve experiment:** for target subject s, use k labeled trials/class from s's session T, k ∈ {0, 2, 5, 10, 20, 36, 72}, plus all re-centered data from the other 8 subjects; test on s's session E. Compare against "subject-only" training with the same k. Plot accuracy vs k (key slide: *"with transfer, a new patient needs N× fewer calibration trials"*).

### 6.4 Robustness (`robustness.py`)
- **Electrode failure:** at test time mark k random channels bad (k = 0…8) and repair with MNE spherical-spline interpolation (`epochs.info['bads']` + `interpolate_bads()`), vs. leaving them as noise. 20 random repeats per k.
- **Noise:** add Gaussian noise at SNR ∈ {∞, 20, 10, 5, 0 dB}.
- Output curves: accuracy vs #bad electrodes, accuracy vs SNR, for CSP+LDA vs TS+LR vs FBTS+LR.

### 6.5 Interpretability (`viz.py`)
- **CSP spatial patterns** as scalp topomaps (`mne.viz.plot_topomap`) → expect left-hand imagery focused over **C4 (right hemisphere)**, right-hand over **C3**, feet over **Cz** (contralateral organization of motor cortex).
- **ERD/ERS maps:** Morlet time-frequency (`mne.time_frequency.tfr_morlet` / `Epochs.compute_tfr`) at C3, Cz, C4 per class, baseline = −1.5…−0.5 s before cue (power drop in mu/beta = ERD).
- **Manifold view:** t-SNE of tangent-space vectors colored by class; before vs after re-centering colored by subject (shows subjects collapse together).
- **Feature importance:** LR coefficients per frequency band in FBTS → which bands matter.

### 6.6 BCI stream simulator (`stream.py`)
- Synchronous (cue-based) BCI, honest to the dataset: play held-out session E trials one by one; for each trial animate EEG (C3, Cz, C4 + all-channel heat strip) from −1 s to +4 s; at decision time output class probabilities from the subject's final model.
- **Decision rule:** if max probability < threshold (default 0.45, slider) → "uncertain, no movement" (safety: a wheelchair must not move on a guess). Map: left hand → turn left, right hand → turn right, feet → move forward, tongue → stop.
- Show predicted vs true label, running accuracy, and the wheelchair path in a small grid maze (Plotly).

### 6.7 Optional DL baseline (only if time remains; clearly labeled "comparison only")
- EEGNet (~2k params) in plain PyTorch, cross-session protocol, CPU (~5–10 min/subject) or Kaggle GPU. Purpose: viva line *"classical Riemannian ML matches/approaches DL with ~100× less compute and full interpretability"*. Skip entirely if the course forbids DL.

---

## 7. Streamlit app (`app/`) — what the TA/prof can do
| Page | Interaction |
|---|---|
| **Home** | Problem story, pipeline diagram, headline results (κ vs competition winner). |
| **1 Brain Signals** | Pick subject / class / trial / channel → raw EEG, PSD, ERD map. |
| **2 Live BCI** | Pick subject (incl. "unseen patient" LOSO model), press Play → trials stream, decoder predicts, wheelchair moves; threshold slider. |
| **3 Model Lab** | Pick pipelines → per-subject bar chart, κ table, confusion matrix, Wilcoxon p-values. |
| **4 Brain Maps** | CSP topomaps per class, t-SNE manifold before/after re-centering. |
| **5 New Patient** | Slider k calibration trials → recompute TS+LR live for chosen subject (with/without transfer) + precomputed curve. |
| **6 Robustness** | Slider #broken electrodes / SNR, toggle interpolation → live accuracy on chosen subject. |
- Heavy results are precomputed (`results/*.csv`, `models/*.joblib`, `figures/*.png`); live recomputation only for single-subject TS+LR (< 1 s). Use `st.cache_data` / `st.cache_resource`.

---

## 8. Two-day schedule

**Day 1 — core ML (target: benchmark table done)**
| Time | Task |
|---|---|
| 0:00–0:45 | venv, requirements, folder skeleton, `config.py`, `.gitignore`, `git init` |
| 0:45–2:00 | `01_download_data.py` (MOABB → Kaggle fallback), `data.py`, verify 288 trials/session for all 9 subjects, cache npz |
| 2:00–3:00 | `02_explore.py`: PSD + ERD figures (sanity: mu/beta ERD visible at C3/C4) |
| 3:00–5:30 | `features.py` (FilterBankCSP+MIBIF, FilterBankTangentSpace), `pipelines.py`, `tests/` on synthetic data |
| 5:30–7:30 | `evaluate.py` + `03_benchmark.py` (cross-session + within-session), results CSV + bar chart |
| 7:30–8:30 | `06_train_final.py`, review numbers, fix issues |

**Day 2 — the "wow" parts + presentation**
| Time | Task |
|---|---|
| 0:00–2:00 | `transfer.py` + `04_transfer.py` (LOSO naive vs re-centered, calibration curve) |
| 2:00–3:00 | `robustness.py` + `05_robustness.py` |
| 3:00–3:45 | `viz.py`: topomaps, t-SNE, band importance |
| 3:45–7:00 | Streamlit app (all 7 pages), `stream.py` |
| 7:00–8:00 | `README.md`, `report/report.md` (intro, data, methods, results, discussion, limitations), `slides_outline.md`, `viva_qa.md` |
| 8:00+ | Optional EEGNet baseline / PhysioNet stretch |

Cut order if behind: EEGNet → PhysioNet → t-SNE → robustness-noise (keep electrode dropout) → TLStretch/Rotate (keep TLCenter).

---

## 9. Report & viva material (`report/`)
- **report.md**: problem, dataset, preprocessing, the math (covariance → SPD manifold → affine-invariant Riemannian distance δ(A,B)=‖log(A^{-1/2} B A^{-1/2})‖_F → tangent space), experiments, results tables/figures, limitations (offline/synchronous, healthy subjects, n=9).
- **viva_qa.md** — prepared answers: Why 8–30 Hz? What is ERD? Why covariance matrices? Why is Euclidean averaging of covariances wrong (swelling effect)? What is the tangent space? Why OAS shrinkage? What is Cohen's kappa and why not accuracy? Why cross-session evaluation? What does re-centering do and why is it unsupervised? Why does FBCSP need feature selection? Limitations & how to make it a real product (OpenBCI headset + LSL streaming).
- **slides_outline.md**: 10–12 slides story: problem → data → idea (geometry) → results vs winner → transfer → robustness → brain maps → live demo → limitations/future.

---

## 10. Verification (definition of done)
1. `pytest` passes (loader shapes, each pipeline fits/predicts on synthetic SPD data, re-centering yields mean ≈ identity).
2. `scripts/01…06` run end-to-end from a clean checkout; all 9 subjects load with 288 trials/session, 72/class.
3. `results/benchmark.csv` has every pipeline × subject × protocol; FBCSP lands near the published κ≈0.57 (sanity check that our implementation is right).
4. ERD figures show mu/beta power drop contralateral to the imagined hand; CSP topomaps over C3/C4/Cz.
5. Re-centered LOSO > naive LOSO on average; calibration curve plotted.
6. `streamlit run app/Home.py` — every page loads with no errors, Live BCI plays a full subject, sliders recompute live.
7. README lists exact commands to reproduce + how to run the demo.

## 11. Risks & fallbacks
| Risk | Fallback |
|---|---|
| BNCI server down | Kaggle mirror (route B) — already verified listed |
| MOABB API change / error on Py 3.13 | own `load_bnci_mat()` with scipy + MNE only |
| pyriemann.transfer API differs | implement re-centering manually (mean_riemann + invsqrtm, ~10 lines) |
| Low accuracy for some subjects (A02, A05, A06 are known "hard" subjects) | expected — discuss "BCI illiteracy" honestly in report |
| Time overrun | follow cut order in §8 |

---

## 12. Phase 2 — submission deliverables (approved 2026-11-09)
1. **Source code + README** — already in repo; README gains a "Paper & slides" section.
2. **Slides** — `slides/main.tex`, LaTeX Beamer (Madrid theme, pdfLaTeX, no extra packages), own `slides/figures/`.
3. **Research paper** — `paper/main.tex`, IEEE conference format (`IEEEtran`, two-column, ~5 pages) + `paper/refs.bib` + `paper/figures/`.
4. Compile both locally (MiKTeX pdflatex + bibtex), fix errors, and build Overleaf-uploadable zips (`overleaf/paper.zip`, `overleaf/slides.zip`, git-ignored).
All numbers are taken from `results/*.csv`; nothing is invented. No PPT / Word files.

## 13. Phase 3 — detailed slide deck for the TA walkthrough (2026-11-09)
User request: the 18-slide deck is too thin; they will explain the whole project to the TA from the slides.
- Rewrite `slides/main.tex` into a ~45-slide, self-explanatory deck. Every slide has the idea, the intuition and the actual numbers, with a "Takeaway" box where it helps.
- Sections: Motivation → Neuroscience (EEG, rhythms, ERD) → Dataset & preprocessing → Feature engineering (filtering, covariance, OAS, SPD/manifold, Riemannian distance/mean, tangent space) → the 6 models (CSP, FBCSP+MIBIF, MDM, TS+LR/SVM, FBTS+LR) → evaluation (protocols, κ, Wilcoxon, no-leakage tuning) → results (summary, per-subject, stats, non-stationarity, confusion) → transfer (re-centering, LOSO, calibration) → robustness → interpretability → software & demo → limitations, conclusion → backup/Q&A slides.
- Add figures already produced by the scripts (PSD, ERD time–frequency, raw trial, LOSO, band importance, noise robustness, within-session) to `slides/figures/`.
- All numbers come from `results/*.csv`. Compile with pdflatex, fix overflow, rebuild `overleaf/slides.zip`, update README slide count.
