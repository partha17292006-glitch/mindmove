# MindMove: viva Q&A (prepared answers)

**Why 8–30 Hz?**
During motor imagery the sensorimotor **mu (8–13 Hz)** and **beta (13–30 Hz)** rhythms over the motor cortex
lose power. Below 8 Hz there are eye artefacts and slow drifts; above 30 Hz there is mostly muscle noise.
The FBTS pipeline does not fix the band: it lets the model weight 6 bands from 4 to 40 Hz per subject.

**What is ERD/ERS?**
Event-Related **De**synchronisation and **S**ynchronisation: a percentage power decrease or increase in a frequency
band relative to a pre-cue baseline. Neurons that are busy planning a movement stop firing in sync, so
the rhythm's amplitude drops (ERD). Our grand-average maps show it contralaterally: left-hand imagery gives
stronger mu ERD at C4 than right-hand imagery (−0.75 vs −0.54 dB), and right-hand imagery gives stronger
ERD at C3 (−0.99 vs −0.53 dB). Tongue imagery gives a mu *increase* (ERS) over the hand areas.

**Why covariance matrices?**
Band power and inter-channel coupling are exactly what changes with imagery. The spatial covariance of a band-passed
trial captures both: the diagonal holds each channel's power, the off-diagonal holds how channels co-vary.
One trial becomes one 22×22 symmetric positive-definite (SPD) matrix. CSP is itself a
function of class covariances, so the covariance is the sufficient statistic of the classical methods.

**Why is Euclidean averaging of covariances wrong (swelling effect)?**
SPD matrices do not form a vector space: subtracting two of them can give a matrix with negative eigenvalues.
The arithmetic mean has a determinant *larger* than its members ("swelling"), which inflates
power. The affine-invariant Riemannian distance δ(A,B)=‖log(A^{-1/2}BA^{-1/2})‖_F respects the
curved geometry. It is invariant to any invertible linear mixing of the channels (so electrode
gain or position differences cancel out), and its mean (the Karcher/geometric mean) does not swell.

**What is the tangent space?**
A flat (Euclidean) approximation of the manifold at a reference point, usually the Riemannian mean M:
S = log(M^{-1/2} C M^{-1/2}), vectorised (upper triangle, √2 on the off-diagonals) → 253 features for 22 channels.
Distances near M are preserved, so any linear classifier (logistic regression, SVM) can then be used. MDM classifies
on the curved manifold directly, while TS+LR uses this tangent-space trick and was better in our benchmark (κ 0.56 vs 0.43).

**Why OAS shrinkage?**
A 22×22 covariance has 253 parameters, estimated from 501 correlated samples, so the estimate is noisy and can be
ill-conditioned. Oracle Approximating Shrinkage pulls it toward a scaled identity with an analytically optimal
weight. This guarantees positive-definiteness, which logm and invsqrtm require, and reduces variance.

**What is Cohen's kappa, and why not accuracy?**
κ = (p_o − p_e)/(1 − p_e): agreement corrected for chance. With 4 balanced classes p_e = 0.25, so
κ = (acc − 0.25)/0.75. κ = 0 means chance and κ = 1 means perfect. It was the official BCI-IV metric, so it makes our numbers
comparable with the competition (winner κ = 0.57), and it does not inflate on imbalanced data.

**Why cross-session evaluation?**
A real BCI is calibrated one day and used on another. Shuffled CV within one session mixes trials
recorded minutes apart and over-estimates performance. In our data, every pipeline loses accuracy
from within-session (day 2) CV to train-day-1 → test-day-2 (e.g. FBTS+LR κ 0.62 → 0.57,
CSP+LDA 0.58 → 0.49). That gap is non-stationarity, and the cross-session protocol is the honest number.

**What does re-centering do, and why is it unsupervised?**
Each person's or session's cloud of covariance matrices sits at a different place on the manifold, because of skull thickness,
electrode impedance and mental state. Re-centering computes that domain's Riemannian mean M from *unlabelled*
trials and maps C → M^{-1/2} C M^{-1/2}, so every domain's mean becomes the identity. Because of affine invariance,
distances inside a domain are unchanged and only the shift is removed. No labels are needed, so a new patient can
just wear the cap for a minute. Our result: leave-one-subject-out TS+LR improved from 40.9% to 53.4% accuracy
(9/9 subjects better), and the t-SNE shows the nine subject clusters merging into one cloud.

**Why does FBCSP need feature selection?**
9 bands × 4 CSP filters gives 36 log-variance features from only 288 trials, and most bands are irrelevant
for a given person. Mutual-information-based selection (MIBIF) keeps the k most informative features
(k tuned by inner CV on day 1 only), which reduces over-fitting and adapts the bands to each subject.

**Did you tune on the test set?**
No. All hyper-parameters (LR C, SVM C, MIBIF k) are tuned by an inner 5-fold CV on the training session.
The steps before the classifier are either stateless (filters) or unsupervised (covariance, tangent-space
reference), so they leak no labels. The "best pipeline per subject" used in the demo is picked from day-1 CV only.

**Why is your FBCSP below the published κ = 0.57?**
We got κ = 0.51. Per subject it tracks the winner closely (A05 0.40 vs 0.40, A06 0.27 vs 0.27, A07 0.74 vs 0.77).
The gap comes from A01, A03 and A09. The original used one-vs-rest CSP and a different classifier, while we use MNE's multiclass
CSP. We deliberately did not tune it against day 2 to "hit" 0.57, because that would be test-set tuning.

**Is FBTS+LR significantly better than FBCSP?**
Not at n = 9. The mean difference is +0.058 κ, it wins on 5/9 subjects, and the Wilcoxon p = 0.15. TS+LR vs CSP+LDA *is*
significant (+0.068 κ, 8/9 subjects, p = 0.008). With 9 subjects, the smallest achievable two-sided Wilcoxon p is 0.004,
so low power is expected and we say so.

**Why are A02, A05 and A06 so bad?**
"BCI inefficiency" (or "BCI illiteracy"): an estimated 15–30% of people do not produce a clear, stable ERD. All six methods
are near κ 0.3 for them, so the limit is in the signal, not the classifier.

**Why do broken electrodes hurt so much, and why does interpolation help?**
A broken channel injects huge broadband variance, which dominates the covariance matrix and moves the trial
far from where it was on the manifold, so even one bad channel pushes accuracy toward chance. Spherical-spline
interpolation rebuilds the channel from its neighbours, and accuracy largely recovers. Practical lesson: bad-channel
*detection* (impedance check, variance outliers) matters more than the choice of classifier.

**Limitations?**
Offline and synchronous (cue-based) data rather than self-paced control; 9 healthy subjects, not patients; 22 wet electrodes in a lab;
4 classes at about 68% accuracy is still too error-prone for unsupervised wheelchair driving, which is why the demo has a
confidence threshold ("no move" when unsure).

**How would this become a real product?**
A cheap dry-electrode headset (e.g. OpenBCI, 8–16 channels over C3/Cz/C4) streaming over Lab Streaming Layer (LSL) to the
same scikit-learn pipeline. Then a 1-minute unlabelled recording for re-centering plus a few labelled trials (transfer),
online re-centering to track drift during the day, an asynchronous "idle" class, and decision smoothing over several windows.

**Why no deep learning?**
The course asks for classical ML. Classical Riemannian methods are also data-efficient (288 trials), train in seconds on a
CPU, are mathematically interpretable (topomaps, band weights), and in the literature they match EEGNet-class
models on this dataset.
