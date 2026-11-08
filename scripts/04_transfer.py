"""Step 4 — cross-subject transfer ("new patient") experiments.

  data/processed/covs_8_30.npz   8-30 Hz OAS covariances for every subject/session (used live by the app)
  results/transfer.csv           LOSO naive vs re-centered (TS+LR and MDM), per target subject
  results/calibration.csv        accuracy vs k labelled calibration trials/class (3 methods, 5 repeats)
  results/tsne.csv               2-D t-SNE of tangent vectors before/after re-centering
  figures/calibration_curve.png, figures/loso.png, figures/tsne_recentering.png
"""
import argparse
import warnings

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import cohen_kappa_score

import config as C
from mindmove import data, transfer, viz

K_VALUES = [0, 2, 5, 10, 20, 36, 72]
METHODS = {"subject_only": "Subject only (standard calibration)",
           "subject_only_rc": "Subject only + re-centering",
           "transfer_rc": "Transfer from 8 others + re-centering (balanced)"}
COV_FILE = C.DATA_PROCESSED / "covs_8_30.npz"


def compute_covs():
    covs, ys = {}, {}
    for s in C.SUBJECTS:
        for sess, (X, y) in data.load_subject(s).items():
            covs[(s, sess)] = transfer.covariances(X)
            ys[(s, sess)] = y.astype(int)
    np.savez(COV_FILE, **{f"cov_{s}_{sess}": v for (s, sess), v in covs.items()},
             **{f"y_{s}_{sess}": v for (s, sess), v in ys.items()})
    return covs, ys


def load_covs():
    with np.load(COV_FILE) as z:
        covs = {(s, sess): z[f"cov_{s}_{sess}"] for s in C.SUBJECTS for sess in C.SESSIONS}
        ys = {(s, sess): z[f"y_{s}_{sess}"] for s in C.SUBJECTS for sess in C.SESSIONS}
    return covs, ys


def loso_target(covs, ys, target):
    warnings.filterwarnings("ignore")
    rows = []
    yE = ys[(target, "E")]
    for clf in ("TS+LR", "MDM"):
        for rc in (False, True):
            y_pred = transfer.loso(covs, ys, target, recentered=rc, clf=clf)
            rows.append({"subject": target, "classifier": clf,
                         "method": "re-centered" if rc else "naive",
                         "accuracy": float((y_pred == yE).mean()),
                         "kappa": cohen_kappa_score(yE, y_pred)})
    return rows


def calib_target(covs, ys, target, repeats):
    warnings.filterwarnings("ignore")
    rc = transfer.recenter_all(covs)
    rows = []
    for k in K_VALUES:
        n_rep = 1 if k in (0, 72) else repeats          # k=0 / k=72 are deterministic
        for r in range(n_rep):
            for m in METHODS:
                rng = np.random.default_rng(C.SEED + 1000 * target + 10 * k + r)
                acc = transfer.calibration_point(covs, ys, target, k, m, rng, rc=rc)
                if not np.isnan(acc):
                    rows.append({"subject": target, "k_per_class": k, "repeat": r,
                                 "method": METHODS[m], "accuracy": acc})
    print(f"  calibration A{target:02d} done", flush=True)
    return rows


def tsne(covs, ys):
    from sklearn.manifold import TSNE
    keys = [(s, "T") for s in C.SUBJECTS]
    subj = np.concatenate([[s] * len(ys[k]) for s, k in zip(C.SUBJECTS, keys)])
    cls = np.concatenate([ys[k] for k in keys])
    out = {}
    for rc in (False, True):
        V = transfer.tangent_vectors([covs[k] for k in keys], recentered=rc)
        out[rc] = TSNE(n_components=2, perplexity=40, random_state=C.SEED, init="pca").fit_transform(V)
    df = pd.DataFrame({"subject": subj, "class": np.asarray(C.CLASSES)[cls],
                       "x_before": out[False][:, 0], "y_before": out[False][:, 1],
                       "x_after": out[True][:, 0], "y_after": out[True][:, 1]})
    df.to_csv(C.RESULTS / "tsne.csv", index=False)
    viz.plot_tsne(out[False], out[True], subj,
                  title="Tangent vectors of 9 subjects (session T), coloured by subject").savefig(
        C.FIGURES / "tsne_recentering.png", dpi=130)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--skip-tsne", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    print("Computing 8-30 Hz covariances ...")
    covs, ys = compute_covs()

    print("LOSO naive vs re-centered ...")
    rows = Parallel(n_jobs=args.jobs)(delayed(loso_target)(covs, ys, t) for t in C.SUBJECTS)
    loso = pd.DataFrame([r for rr in rows for r in rr])
    loso.to_csv(C.RESULTS / "transfer.csv", index=False)
    print(loso.groupby(["classifier", "method"])[["accuracy", "kappa"]].agg(["mean", "std"]).round(3))
    piv = loso[loso.classifier == "TS+LR"].pivot(index="subject", columns="method", values="accuracy")
    print(f"TS+LR re-centered > naive for {(piv['re-centered'] > piv['naive']).sum()}/9 subjects")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.6))
    p2 = loso.pivot_table(index="subject", columns=["classifier", "method"], values="accuracy")
    p2.plot.bar(ax=ax)
    ax.axhline(0.25, color="0.5", ls=":")
    ax.set_ylabel("accuracy on unseen subject (session E)")
    ax.set_title("Leave-one-subject-out: naive pooling vs Riemannian re-centering")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "loso.png", dpi=130)

    print("Calibration curves ...")
    rows = Parallel(n_jobs=args.jobs)(delayed(calib_target)(covs, ys, t, args.repeats)
                                      for t in C.SUBJECTS)
    cal = pd.DataFrame([r for rr in rows for r in rr])
    cal.to_csv(C.RESULTS / "calibration.csv", index=False)
    print(cal.groupby(["method", "k_per_class"]).accuracy.mean().unstack().round(3).to_string())
    viz.plot_curve(cal, "k_per_class", "accuracy", "method",
                   "How many labelled calibration trials does a new patient need?",
                   "labelled calibration trials per class (from the new patient)").savefig(
        C.FIGURES / "calibration_curve.png", dpi=130)

    if not args.skip_tsne:
        print("t-SNE ...")
        tsne(covs, ys)
    print("Saved results/transfer.csv, calibration.csv, tsne.csv")


if __name__ == "__main__":
    main()
