# MindMove: slides outline (12 slides, ~10 min + live demo)

1. **Title.** *MindMove: decoding imagined movement from brain signals with Riemannian geometry.* One line: "steering a wheelchair by thought, with classical ML".
2. **The problem.** ALS / spinal-cord injury / locked-in patients: the brain still plans movement, the body can't execute it. Two pain points: EEG is noisy and non-stationary, and every patient needs long calibration.
3. **Data.** BCI Competition IV 2a: 9 subjects, 22 channels, 4 imagined movements, 2 days × 288 trials. Trial timeline (fixation → cue at 2 s → imagery). Figure: `figures/raw_trial_A03_left_hand.png`.
4. **What the brain does.** ERD: mu/beta power drops over the opposite hemisphere. Figures: `erd_topomap_mu.png`, `erd_grand_average.png`.
5. **The idea: geometry.** Trial → 22×22 covariance → a point on the SPD manifold. Riemannian distance, mean, tangent space; why Euclidean averaging "swells". Pipeline diagram (Home page).
6. **Six pipelines.** CSP+LDA, FBCSP+SVM (2008 winner), MDM, TS+LR, TS+SVM, FBTS+LR (ours = filter bank + tangent space). Protocol: train day 1 → test day 2, κ.
7. **Results vs the competition winner.** `benchmark_cross_session.png` plus table: FBTS+LR κ 0.57, TS+LR 0.56, FBCSP 0.51, CSP+LDA 0.49, MDM 0.43. TS+LR > CSP+LDA, Wilcoxon p = 0.008. Honest note: FBTS vs FBCSP p = 0.15 (n = 9).
8. **Non-stationarity.** Within-day CV vs cross-day drop for every pipeline. Confusion matrix: feet ↔ tongue is the hardest pair.
9. **New patient = transfer.** t-SNE before/after re-centering (`tsne_recentering.png`). LOSO 40.9% → 53.4% (9/9 subjects better). Calibration curve: zero labelled trials gives 53%; 5 trials/class with transfer ≈ 10 trials/class without.
10. **Robustness.** One broken electrode sends every decoder to chance; spherical-spline interpolation recovers most of it (`robustness_electrodes.png`). Noise curve.
11. **Brain maps / interpretability.** CSP patterns peak at CP4 (left hand), CP3 (right hand), CPz (feet), matching motor-cortex anatomy. Band importance.
12. **Live demo + limitations / future.** Streamlit Live BCI: wheelchair driven by replayed day-2 EEG, with a safety threshold. Limits: offline, cue-based, healthy subjects, n = 9. Future: OpenBCI + LSL, online re-centering, idle class.
