# MindMove: Decoding Imagined Movement from EEG with Classical (Riemannian) Machine Learning

*Course ML project. All numbers below come from `results/*.csv`, produced by `scripts/01–06`
(seed 42, CPU only).*

## 1. Problem

People with ALS, high spinal-cord injury or locked-in syndrome lose control of their muscles, but their
motor cortex still generates movement *intentions*. A motor-imagery Brain-Computer Interface (BCI) decodes these
intentions from scalp EEG, so a patient can, for example, steer a wheelchair by imagining a left-hand, right-hand,
feet or tongue movement.

Three properties make this a hard ML problem:
1. **Low SNR.** The signal of interest is a few-µV modulation of ongoing rhythms.
2. **Non-stationarity.** The signal changes between days, so a model calibrated today degrades tomorrow.
3. **Between-subject variability.** Every new user normally needs a long, tiring labelled calibration session.

MindMove builds a complete classical-ML decoder around **Riemannian geometry**, benchmarks it against the
methods that won BCI Competition IV, and attacks problems 2–3 with **unsupervised Riemannian re-centering**.
It also studies **robustness** and **interpretability**, and ships an interactive **Streamlit BCI simulator**.

## 2. Data

**BCI Competition IV, dataset 2a** (BNCI2014_001): 9 healthy subjects, 22 EEG channels (10-20 system, motor
strip), 250 Hz, 4 classes (left hand, right hand, feet, tongue), and two sessions recorded on different days (T and E),
each with 288 trials (72 per class). The cue appears 2 s into each trial and imagery lasts until 6 s.

*Acquisition.* The BNCI server timed out from our machine, so we used the Kaggle mirror of the original
`.mat` files (which include the true labels of the E sessions) and parsed them with our own loader. A MOABB loader is kept
as an alternative route. All 9 × 2 sessions were verified: 288 trials, 72/class, finite values.

**Preprocessing.** We drop the 3 EOG channels, convert µV → V, linearly interpolate the rare NaN samples, filter each
continuous run 1–45 Hz (filtering continuous data avoids epoch-edge artefacts), and cut wide epochs from −1.5 to
+4.0 s around the cue. Artefact-flagged trials are kept, as in the competition. Each pipeline then band-passes the
wide epoch and crops the **classification window 0.5–2.5 s** after the cue. The first 0.5 s contain the visual
evoked response to the cue, which is visible as the red blob in the ERD maps, and 2 s give a stable covariance estimate.

**Exploration.** The grand-average ERD maps (`figures/erd_grand_average.png`, `figures/erd_topomap_mu.png`)
show the textbook pattern. Hand imagery desynchronises mu/beta over the *opposite* hemisphere: at C4, left-hand
imagery gives −0.75 dB of mu power versus −0.54 dB for right-hand imagery, and at C3, right-hand gives −0.99 dB versus −0.53 dB for left-hand.
Feet imagery gives a focal ERD at the vertex, and tongue imagery gives a mu *increase*. Per subject, using the 8–30 Hz band, this
lateralisation holds in 6/9 subjects at C4 and 7/9 at C3 (`results/erd_check.csv`), which already hints at the weak
subjects discussed in §6.

## 3. Methods

### 3.1 From trials to points on a manifold
For a band-passed trial X ∈ ℝ^{22×501} we estimate the spatial covariance with **Oracle Approximating Shrinkage**
(OAS). OAS guarantees a well-conditioned SPD matrix from only 501 correlated samples. SPD matrices form a curved
manifold. Under the **affine-invariant Riemannian metric**

  δ(A, B) = ‖ log(A^{-1/2} B A^{-1/2}) ‖_F

distances are invariant to any invertible linear mixing of the channels, and the Riemannian (Karcher) mean avoids the
"swelling" of the Euclidean mean. Two ways of classifying on the manifold are used:
* **MDM**: assign a trial to the class whose Riemannian mean is closest.
* **Tangent space (TS)**: map each matrix to the tangent space at the overall mean,
  s = upper(log(M^{-1/2} C M^{-1/2})) (253 features), where any linear classifier applies.

### 3.2 Pipelines (`src/mindmove/pipelines.py`)
| Name | Steps |
|---|---|
| CSP+LDA | 8–30 Hz → CSP (8 filters, log-variance) → shrinkage LDA |
| FBCSP+SVM | 9 bands 4–40 Hz → CSP per band (4 filters) → MIBIF (mutual-information SelectKBest) → linear SVM |
| MDM | 8–30 Hz → OAS covariance → MDM |
| TS+LR | 8–30 Hz → OAS covariance → tangent space → logistic regression |
| TS+SVM | same → linear SVM |
| **FBTS+LR (ours)** | 6 bands 4–40 Hz → OAS covariance per band → tangent space per band → concat (1518 features) → standardise → L2 logistic regression |

FBTS+LR combines FBCSP's subject-specific frequency selection with the full covariance information of the
tangent space, instead of a few CSP directions.

**Tuning without leakage.** LR/SVM C and MIBIF k are tuned by an inner stratified 5-fold GridSearchCV on the
*training session only*. The grid wraps only the classifier head: the steps before it are stateless (filters) or
unsupervised (covariance, tangent reference, scaler), so they leak no label information, and feature
extraction runs once per fit, which is about 20× faster.

### 3.3 Protocols and statistics (`src/mindmove/evaluate.py`)
1. **Cross-session** (official): train on T, test on E, per subject. This is the headline result.
2. **Within-session**: stratified 5-fold CV inside T and inside E. This is a sanity check that measures non-stationarity.
3. **Leave-one-subject-out**: train on 8 subjects (both sessions), test on the 9th subject's E session.

Metric: **Cohen's κ** (official; κ = (acc − 0.25)/0.75 for 4 balanced classes) and accuracy. Paired **Wilcoxon**
signed-rank tests across the 9 subjects.

### 3.4 Transfer: Riemannian re-centering (`src/mindmove/transfer.py`)
For each domain (subject × session) we compute its Riemannian mean M from **unlabelled** trials and map every
covariance C → M^{-1/2} C M^{-1/2}. Every domain is then centred at the identity, and affine invariance keeps
within-domain geometry intact. We compare:
* LOSO **naive** (pool raw covariances) vs **re-centered**, with TS+LR and MDM.
* A **calibration curve**: k ∈ {0, 2, 5, 10, 20, 36, 72} labelled trials/class from the new patient's day 1, with
  5 random draws, tested on their day 2. Methods: *subject only* (standard calibration), *subject only + re-centering*,
  and *transfer* (8 other subjects + the k target trials, all re-centered). In the transfer model the k·4 target trials get the same
  total sample weight as all source trials (domain balancing). Without it, about 20 target trials are drowned out by
  about 4,600 source trials.

### 3.5 Robustness (`src/mindmove/robustness.py`)
Decoders trained on day 1 are tested on a corrupted day 2.
* **Broken electrodes.** k ∈ {0, 1, 2, 3, 4, 6, 8} random channels are replaced by white noise at 5× their standard deviation, with 20 random
  sets per k. These are either left as is or repaired with MNE **spherical-spline interpolation**.
* **Noise.** Additive white Gaussian noise at SNR ∈ {∞, 20, 10, 5, 0} dB, 5 repeats.

### 3.6 Interpretability (`src/mindmove/viz.py`)
* One-vs-rest CSP **patterns** (A = Σw, Haufe et al. 2014) of the ERD direction of each class, shown as topomaps.
* The share of FBTS+LR coefficient mass per frequency band.
* t-SNE of tangent vectors before and after re-centering.

## 4. Results

### 4.1 Benchmark (`results/benchmark.csv`, `results/benchmark_summary.csv`)

**Cross-session (train day 1 → test day 2), mean ± sd over 9 subjects**

| Pipeline | κ | accuracy | mean fit time |
|---|---|---|---|
| **FBTS+LR (ours)** | **0.570 ± 0.194** | **67.8 ± 14.5 %** | 13.9 s |
| TS+SVM | 0.559 ± 0.223 | 66.9 ± 16.7 % | 2.2 s |
| TS+LR | 0.558 ± 0.217 | 66.8 ± 16.3 % | 1.8 s |
| FBCSP+SVM | 0.513 ± 0.155 | 63.5 ± 11.6 % | 22.8 s |
| CSP+LDA | 0.490 ± 0.226 | 61.7 ± 16.9 % | 1.7 s |
| MDM | 0.432 ± 0.188 | 57.4 ± 14.1 % | 1.3 s |
| *BCI-IV 2a winner (FBCSP, 2008)* | *0.57* | | |

Per-subject κ (cross-session):

| | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 |
|---|---|---|---|---|---|---|---|---|---|
| FBTS+LR | 0.72 | 0.29 | 0.75 | 0.52 | 0.39 | 0.31 | 0.73 | 0.73 | 0.70 |
| TS+LR | 0.78 | 0.30 | 0.82 | 0.45 | 0.30 | 0.32 | 0.63 | 0.69 | 0.74 |
| FBCSP+SVM (ours) | 0.62 | 0.31 | 0.63 | 0.54 | 0.40 | 0.27 | 0.74 | 0.56 | 0.54 |
| 2008 winner | 0.68 | 0.42 | 0.75 | 0.48 | 0.40 | 0.27 | 0.77 | 0.75 | 0.61 |

* **Sanity check.** Our FBCSP re-implementation reaches κ = 0.51 and tracks the published winner subject by subject
  (A05, A06 and A07 almost identical). The gap comes from A01, A03 and A09, which is plausible given we use MNE's multiclass
  CSP rather than the original one-vs-rest variant. We did not tune it against day 2.
* **Riemannian > CSP.** TS+LR beats CSP+LDA by +0.068 κ on 8/9 subjects (Wilcoxon p = 0.008).
* **Ours.** FBTS+LR has the best mean κ (0.570, equal to the 2008 winner, using a simpler linear model). Its advantage
  over FBCSP (+0.058 κ, 5/9 subjects, p = 0.15) and over TS+LR (+0.013, p = 0.73) is **not significant** with n = 9.
* **Non-stationarity.** Within-session CV on day 2 is higher than cross-session for every pipeline
  (FBTS+LR κ 0.618 vs 0.570, CSP+LDA 0.579 vs 0.490). The day-to-day shift costs 0.04–0.09 κ.
* **Errors.** FBTS+LR is best on left hand (77% recall) and worst on feet (60%). Feet ↔ tongue is the most
  confused pair, since both are represented near the midline.

### 4.2 Transfer to a new patient (`results/transfer.csv`, `results/calibration.csv`)

**Leave-one-subject-out (no labelled data from the test subject), accuracy, mean ± sd**

| | naive pooling | re-centered |
|---|---|---|
| TS+LR | 40.9 ± 12.3 % (κ 0.21) | **53.4 ± 16.7 %** (κ 0.38) |
| MDM | 34.6 ± 11.7 % (κ 0.13) | 48.8 ± 13.2 % (κ 0.32) |

Re-centering improved TS+LR for **9/9** subjects. The t-SNE (`figures/tsne_recentering.png`) shows why: before
re-centering, trials form nine separate *subject* clusters, and afterwards they merge into one cloud.

**Calibration curve, mean accuracy on day 2 (9 subjects × 5 draws)**

| k labelled trials/class | 0 | 2 | 5 | 10 | 20 | 36 | 72 |
|---|---|---|---|---|---|---|---|
| Subject only (standard) | n/a | 46.6 | 51.9 | 56.4 | 61.0 | 64.8 | 67.2 |
| Subject only + re-centering | n/a | 48.6 | 54.3 | 59.1 | 64.6 | 67.2 | **69.6** |
| Transfer + re-centering | **53.4** | **55.6** | **56.8** | 58.4 | 61.0 | 63.0 | 64.4 |

* With **zero** labelled trials, the transfer decoder (53.4%) already matches standard calibration with about 5 trials/class.
  With 5 trials/class it beats standard calibration with 10 trials/class, so **calibration needs about half as many labelled trials** in the
  low-data regime.
* Beyond about 10–20 trials/class, the patient's own data wins, and re-centering *each session* (still unsupervised)
  is the best option. Applied to the subject's own data it raises cross-session accuracy from 67.2% to 69.6% with all
  72 trials/class, because it also removes the day-to-day shift.

### 4.3 Robustness (`results/robustness.csv`)

**Broken electrodes, mean accuracy over 9 subjects × 20 random electrode sets** (`figures/robustness_electrodes.png`)

| broken channels | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| CSP+LDA, left broken | 61.7 | 28.8 | 25.5 | 25.2 | 25.0 | 25.0 | 25.0 |
| TS+LR, left broken | 66.8 | 46.5 | 35.5 | 28.9 | 27.4 | 25.6 | 25.3 |
| FBTS+LR, left broken | 67.8 | 46.8 | 34.9 | 29.5 | 27.3 | 26.0 | 25.3 |
| CSP+LDA, interpolated | 61.7 | 60.3 | 59.1 | 57.7 | 56.6 | 54.0 | 50.3 |
| TS+LR, interpolated | 66.8 | 64.4 | 61.7 | 60.0 | 57.2 | 52.7 | 46.9 |
| **FBTS+LR, interpolated** | 67.8 | **65.8** | **64.1** | **62.5** | **60.5** | **57.2** | **53.8** |

* **A single broken electrode is catastrophic if it is not handled.** CSP+LDA drops straight to chance (28.8%), because
  CSP spatial filters put large weights on individual channels. The Riemannian pipelines degrade more gracefully (to about 47%)
  but are still badly hurt.
* **Spherical-spline interpolation recovers most of the loss.** With 2 broken channels FBTS+LR keeps 64.1% (−3.7 points),
  and even with 8 of 22 channels broken it stays at 53.8%, about twice chance. FBTS+LR is the most robust pipeline at every k.
* Lesson: bad-channel detection and repair matter more than the choice of classifier.

**Additive white noise, mean accuracy** (`figures/robustness_noise.png`)

| SNR | clean | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| CSP+LDA | 61.7 | 45.0 | 26.4 | 25.3 | 25.0 |
| TS+LR | 66.8 | 62.0 | 30.4 | 25.1 | 25.0 |
| FBTS+LR | 67.8 | 58.8 | 37.0 | 28.7 | 25.5 |

Broadband white noise is surprisingly harmful even at 20 dB, because the EEG power sits mostly below 8 Hz while the noise
spreads evenly up to 125 Hz. In the 8–30 Hz band the effective SNR is therefore much lower. The noise also adds a constant to
every covariance diagonal, which *shifts* the whole test session on the manifold relative to the training data. The tangent-space
methods hold up best at 20 dB (TS+LR −4.8 points versus CSP+LDA −16.7). By 10 dB all methods are near chance.

### 4.4 Interpretability
* One-vs-rest CSP patterns, averaged over the 9 subjects (`figures/csp_patterns.png`), peak at **CP4 for left hand**,
  **CP3 for right hand** and **CPz for feet**, i.e. over the contralateral hand areas and the midline foot area of the
  sensorimotor cortex, as anatomy predicts. Tongue peaks at FC3, more lateral and anterior.
* FBTS+LR band weights (`figures/band_importance.png`) are spread fairly evenly across bands on average (15–19% each,
  8–12 Hz highest), but differ strongly between subjects. That is the argument for a filter bank over one fixed band.
* The day-1 CV best pipeline differs per subject (TS+LR for 4, FBCSP+SVM for 3, TS+SVM and FBTS+LR for one each;
  `models/best.json`), which again shows the subject variability.

## 5. Demo: Streamlit BCI simulator (`streamlit run app/Home.py`)
Seven pages: Home (story and headline κ), Brain Signals (raw EEG, PSD, ERD per subject/class/trial), **Live BCI**
(day-2 trials replayed one by one, with class probabilities, a confidence threshold for "no move when unsure", and a virtual
wheelchair in a 10×10 room; it includes an *unseen patient* decoder trained only on other subjects), Model Lab
(per-subject bars, κ table, confusion matrices, Wilcoxon), Brain Maps (CSP patterns, ERD topomaps, interactive t-SNE,
band weights), New Patient (live retraining with k calibration trials, with and without transfer) and Robustness (live
electrode breaking and repair).

## 6. Discussion and limitations
* **Subject variability dominates.** A02, A05 and A06 stay near κ 0.3 with every method ("BCI inefficiency"). The ceiling
  is the signal, not the classifier.
* **Statistics.** n = 9 gives low power. Only the TS+LR vs CSP+LDA difference is significant.
* **Offline and synchronous.** Trials are cue-locked. A real wheelchair needs asynchronous control with an idle state.
* **Healthy subjects and lab equipment** (22 wet electrodes). Patients' ERD may be weaker or altered.
* **Accuracy.** About 68% on 4 classes is not yet safe enough to drive unsupervised. The confidence threshold trades speed
  for safety, as shown on the Live BCI page.
* **Transductive re-centering.** The target mean is estimated from the unlabelled day-2 trials it is applied to. Online, it
  would be estimated from a short unlabelled warm-up and updated recursively.

## 7. Conclusion
Purely classical, CPU-only ML reproduces the BCI-IV 2a winner. A filter-bank tangent-space model matches its κ = 0.57,
and tangent-space methods significantly beat CSP. Unsupervised Riemannian re-centering is the key tool for the practical
problems. It lifts zero-calibration cross-subject accuracy from 41% to 53%, halves the labelled calibration needed in
the low-data regime, and also reduces the day-to-day drop for a single user. Robustness tests show that bad-channel
handling matters more than the choice of classifier.

## References
* Ang, K. K. et al. (2008). Filter Bank Common Spatial Pattern (FBCSP) in BCI. *IJCNN*.
* Barachant, A. et al. (2012). Multiclass BCI classification by Riemannian geometry. *IEEE TBME*.
* Zanini, P. et al. (2018). Transfer learning: a Riemannian geometry framework with applications to BCIs. *IEEE TBME*.
* Tangermann, M. et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*.
* Haufe, S. et al. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. *NeuroImage*.
* Perrin, F. et al. (1989). Spherical splines for scalp potential and current density mapping. *EEG Clin. Neurophysiol.*
* Ledoit & Wolf / Chen et al. (2010). Shrinkage algorithms for MMSE covariance estimation (OAS). *IEEE TSP*.
