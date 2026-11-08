"""Step 5 — robustness of decoders trained on session T, tested on corrupted session E.

  electrode failure: k ∈ {0,1,2,3,4,6,8} random broken channels, 20 repeats,
                     'broken' (left as noise) vs 'interpolated' (spherical spline)
  additive noise:    SNR ∈ {inf, 20, 10, 5, 0} dB, 5 repeats
Outputs results/robustness.csv, figures/robustness_electrodes.png, figures/robustness_noise.png
"""
import argparse
import warnings

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
from mindmove import data, robustness, viz
from mindmove.pipelines import make_pipeline

PIPES = ["CSP+LDA", "TS+LR", "FBTS+LR"]
K_BAD = [0, 1, 2, 3, 4, 6, 8]
SNRS = [np.inf, 20, 10, 5, 0]


def run_subject(subject, repeats, noise_repeats):
    warnings.filterwarnings("ignore")
    d = data.load_subject(subject)
    (XT, yT), (XE, yE) = d["T"], d["E"]
    models = {p: make_pipeline(p).fit(XT, yT) for p in PIPES}
    info = data.make_info()
    rows = []
    for k in K_BAD:
        for r in range(1 if k == 0 else repeats):
            for repair in (False, True):
                rng = np.random.default_rng(C.SEED + 100 * subject + 10 * k + r)   # same bad set for both
                Xc, _ = robustness.corrupt_electrodes(XE, k, rng, repair=repair, info=info)
                for p, m in models.items():
                    rows.append({"subject": subject, "pipeline": p, "test": "electrodes", "level": k,
                                 "condition": "interpolated" if repair else "broken", "repeat": r,
                                 "accuracy": float((m.predict(Xc) == yE).mean())})
    for snr in SNRS:
        for r in range(1 if np.isinf(snr) else noise_repeats):
            rng = np.random.default_rng(C.SEED + 100 * subject + r)
            Xn = robustness.add_noise(XE, snr, rng)
            for p, m in models.items():
                rows.append({"subject": subject, "pipeline": p, "test": "noise",
                             "level": 99.0 if np.isinf(snr) else float(snr),
                             "condition": "noise", "repeat": r,
                             "accuracy": float((m.predict(Xn) == yE).mean())})
    print(f"  A{subject:02d} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--noise-repeats", type=int, default=5)
    ap.add_argument("--subjects", type=int, nargs="*", default=C.SUBJECTS)
    args = ap.parse_args()

    out = Parallel(n_jobs=args.jobs)(delayed(run_subject)(s, args.repeats, args.noise_repeats)
                                     for s in args.subjects)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(C.RESULTS / "robustness.csv", index=False)

    el = df[df.test == "electrodes"].copy()
    el["series"] = el.pipeline + " — " + el.condition
    # average repeats within subject first, so error bars reflect between-subject spread
    el_s = el.groupby(["series", "level", "subject"], sort=False).accuracy.mean().reset_index()
    viz.plot_curve(el_s, "level", "accuracy", "series", "Broken electrodes at test time",
                   "number of broken electrodes (of 22)").savefig(
        C.FIGURES / "robustness_electrodes.png", dpi=130)
    no = df[df.test == "noise"].groupby(["pipeline", "level", "subject"]).accuracy.mean().reset_index()
    viz.plot_curve(no, "level", "accuracy", "pipeline", "Additive Gaussian noise (99 = clean)",
                   "SNR (dB)").savefig(C.FIGURES / "robustness_noise.png", dpi=130)

    print(el.groupby(["pipeline", "condition", "level"]).accuracy.mean().unstack().round(3).to_string())
    print(df[df.test == "noise"].groupby(["pipeline", "level"]).accuracy.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
