"""Step 3 — benchmark every pipeline × subject × protocol.

Protocols: cross_session (T -> E, headline), within_session_T, within_session_E (5-fold CV).
Outputs:
  results/benchmark.csv                one row per subject × pipeline × protocol
  results/benchmark_summary.csv        mean ± std per pipeline × protocol
  results/predictions_cross_session.csv  per-trial predictions (confusion matrices in the app)
  results/stats.csv                    paired Wilcoxon tests across subjects
  figures/benchmark_*.png
Run: python scripts/03_benchmark.py [--jobs N] [--subjects 1 2 ...] [--pipelines ...]
"""
import argparse
import os
import warnings

import _bootstrap  # noqa: F401
import pandas as pd
from joblib import Parallel, delayed

import config as C
from mindmove import data, evaluate, viz
from mindmove.pipelines import OURS, PIPELINE_NAMES, make_pipeline


def run_one(subject, name):
    warnings.filterwarnings("ignore")
    d = data.load_subject(subject)
    (XT, yT), (XE, yE) = d["T"], d["E"]
    rows, preds = [], []
    res, y_pred, _ = evaluate.cross_session(make_pipeline(name), XT, yT, XE, yE)
    rows.append({"subject": subject, "pipeline": name, "protocol": "cross_session", **res})
    preds += [{"subject": subject, "pipeline": name, "trial": i, "y_true": int(t), "y_pred": int(p)}
              for i, (t, p) in enumerate(zip(yE, y_pred))]
    for sess, (X, y) in d.items():
        res, _ = evaluate.within_session_cv(make_pipeline(name), X, y)
        rows.append({"subject": subject, "pipeline": name, "protocol": f"within_session_{sess}", **res})
    k = rows[0]["kappa"]
    print(f"  A{subject:02d} {name:10s} cross-session kappa={k:.3f}", flush=True)
    return rows, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--subjects", type=int, nargs="*", default=C.SUBJECTS)
    ap.add_argument("--pipelines", nargs="*", default=PIPELINE_NAMES)
    args = ap.parse_args()

    tasks = [(s, p) for s in args.subjects for p in args.pipelines]
    print(f"Running {len(tasks)} subject×pipeline tasks on {args.jobs} workers ...")
    out = Parallel(n_jobs=args.jobs, verbose=0)(delayed(run_one)(s, p) for s, p in tasks)
    df = pd.DataFrame([r for rows, _ in out for r in rows])
    preds = pd.DataFrame([r for _, p in out for r in p])
    df.to_csv(C.RESULTS / "benchmark.csv", index=False)
    preds.to_csv(C.RESULTS / "predictions_cross_session.csv", index=False)

    summary = evaluate.summarize(df)
    summary.to_csv(C.RESULTS / "benchmark_summary.csv", index=False)
    print("\n", summary.to_string(index=False))

    comps = [(OURS, "FBCSP+SVM"), ("TS+LR", "CSP+LDA"), (OURS, "TS+LR"), ("TS+LR", "FBCSP+SVM")]
    stats = pd.DataFrame([evaluate.wilcoxon_compare(df, a, b) for a, b in comps
                          if a in args.pipelines and b in args.pipelines])
    if len(stats):
        stats.to_csv(C.RESULTS / "stats.csv", index=False)
        print("\n", stats.round(4).to_string(index=False))

    for proto in ("cross_session", "within_session_E"):
        viz.plot_benchmark_bar(df, proto).savefig(C.FIGURES / f"benchmark_{proto}.png", dpi=130)
    print("\nSaved results/benchmark.csv and figures/benchmark_*.png")


if __name__ == "__main__":
    main()
