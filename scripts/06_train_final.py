"""Step 6 — final per-subject models + interpretability figures.

  models/A0x.joblib     {pipeline name: model fitted on session T} for all 6 pipelines
                        + "LOSO (unseen patient)": re-centered TS+LR trained on the other 8 subjects
  models/best.json      best pipeline per subject, chosen by within-session CV on session T only
                        (WHY: choosing on session E would leak the test set)
  results/band_importance.csv, results/csp_patterns.npz
  figures/csp_patterns.png, figures/band_importance.png, figures/confusion_<pipeline>.png
"""
import argparse
import json
import warnings

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
from mindmove import data, evaluate, transfer, viz
from mindmove.pipelines import OURS, PIPELINE_NAMES, make_pipeline

LOSO_KEY = "LOSO (unseen patient)"


def train_subject(subject, covs, ys):
    warnings.filterwarnings("ignore")
    XT, yT = data.load_subject(subject, sessions=("T",))["T"]
    models = {name: make_pipeline(name).fit(XT, yT) for name in PIPELINE_NAMES}
    src = [k for k in covs if k[0] != subject]
    models[LOSO_KEY] = transfer.RecenteredTSModel().fit([covs[k] for k in src], [ys[k] for k in src])
    joblib.dump(models, C.MODELS / f"A{subject:02d}.joblib", compress=3)

    # band importance: share of |LR coefficient| mass falling in each band
    fb = models[OURS].named_steps["fbts"]
    coef = np.abs(models[OURS].named_steps["head"].best_estimator_.coef_).sum(0)
    tot = coef.sum()
    imp = [{"subject": subject, "band": f"{lo}-{hi} Hz", "importance": coef[sl].sum() / tot}
           for (lo, hi), sl in zip(C.FBTS_BANDS, fb.band_slices())]
    pats = viz.ovr_csp_patterns(XT, yT)
    print(f"  A{subject:02d} trained", flush=True)
    return imp, pats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cov_file = C.DATA_PROCESSED / "covs_8_30.npz"
    if cov_file.exists():
        with np.load(cov_file) as z:
            covs = {(s, se): z[f"cov_{s}_{se}"] for s in C.SUBJECTS for se in C.SESSIONS}
            ys = {(s, se): z[f"y_{s}_{se}"] for s in C.SUBJECTS for se in C.SESSIONS}
    else:
        covs, ys = {}, {}
        for s in C.SUBJECTS:
            for se, (X, y) in data.load_subject(s).items():
                covs[(s, se)], ys[(s, se)] = transfer.covariances(X), y.astype(int)

    out = Parallel(n_jobs=args.jobs)(delayed(train_subject)(s, covs, ys) for s in C.SUBJECTS)

    # best pipeline per subject (session T within-session CV only)
    bench = pd.read_csv(C.RESULTS / "benchmark.csv")
    wt = bench[bench.protocol == "within_session_T"]
    best = {int(s): g.sort_values("kappa", ascending=False).pipeline.iloc[0]
            for s, g in wt.groupby("subject")}
    (C.MODELS / "best.json").write_text(json.dumps(best, indent=2))
    print("Best pipeline per subject (by session-T CV):", best)

    imp = pd.DataFrame([r for i, _ in out for r in i])
    imp.to_csv(C.RESULTS / "band_importance.csv", index=False)
    viz.plot_band_importance(imp).savefig(C.FIGURES / "band_importance.png", dpi=130)
    print(imp.groupby("band").importance.mean().round(3).to_string())

    pats = {c: np.stack([p[c] for _, p in out]) for c in C.CLASSES}
    np.savez(C.RESULTS / "csp_patterns.npz", **pats)
    ga = {c: v.mean(0) for c, v in pats.items()}
    viz.plot_topomaps(ga, data.make_info(), title="One-vs-rest CSP patterns (ERD direction, 8-30 Hz, mean of 9)",
                      cmap="Reds", symmetric=False, unit="|pattern| (normalised)").savefig(
        C.FIGURES / "csp_patterns.png", dpi=130)
    peak = {c: C.CHANNELS[int(np.argmax(v))] for c, v in ga.items()}
    print("Peak channel of grand-average CSP pattern per class:", peak)

    preds = pd.read_csv(C.RESULTS / "predictions_cross_session.csv")
    for name in (OURS, "FBCSP+SVM", "TS+LR"):
        p = preds[preds.pipeline == name]
        cm = evaluate.confusion(p.y_true, p.y_pred)
        viz.plot_confusion(cm, f"{name} — cross-session, all subjects").savefig(
            C.FIGURES / f"confusion_{name.replace('+', '_')}.png", dpi=130)
    print("Saved models/*.joblib, models/best.json and interpretability figures")


if __name__ == "__main__":
    main()
