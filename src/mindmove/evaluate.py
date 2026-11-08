"""Evaluation protocols and statistics.

Protocols
  cross_session     train on session T, test on session E (official competition protocol)
  within_session_X  5-fold stratified CV inside one session (sanity / non-stationarity check)
  loso              leave-one-subject-out (see transfer.py)

Metric: Cohen's kappa (official BCI-IV metric: accuracy corrected for chance,
kappa = (acc - 0.25) / 0.75 for 4 balanced classes), plus accuracy.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import config as C


def scores(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "kappa": cohen_kappa_score(y_true, y_pred)}


def cross_session(pipe, X_train, y_train, X_test, y_test):
    """Fit on one session, test on the other. Returns (metrics dict, y_pred, fitted model)."""
    model = clone(pipe)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    y_pred = model.predict(X_test)
    res = scores(y_test, y_pred) | {"n_train": len(y_train), "n_test": len(y_test),
                                    "fit_time_s": fit_time}
    return res, y_pred, model


def within_session_cv(pipe, X, y, n_splits=5, seed=C.SEED):
    """Stratified k-fold inside one session; metrics on the pooled out-of-fold predictions."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = np.empty_like(y)
    t0 = time.perf_counter()
    for tr, te in cv.split(X, y):
        m = clone(pipe).fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
    res = scores(y, y_pred) | {"n_train": int(len(y) * (n_splits - 1) / n_splits),
                               "n_test": len(y),
                               "fit_time_s": (time.perf_counter() - t0) / n_splits}
    return res, y_pred


def confusion(y_true, y_pred, normalize="true"):
    return confusion_matrix(y_true, y_pred, labels=range(len(C.CLASSES)), normalize=normalize)


def wilcoxon_compare(df: pd.DataFrame, a: str, b: str, protocol="cross_session",
                     metric="kappa") -> dict:
    """Paired Wilcoxon signed-rank test across subjects (n=9 -> low power, report honestly)."""
    sub = df[df.protocol == protocol].pivot_table(index="subject", columns="pipeline",
                                                  values=metric)
    xa, xb = sub[a].values, sub[b].values
    diff = xa - xb
    try:
        stat, p = wilcoxon(xa, xb)
    except ValueError:          # all differences zero
        stat, p = np.nan, 1.0
    return {"protocol": protocol, "metric": metric, "a": a, "b": b,
            "mean_a": xa.mean(), "mean_b": xb.mean(), "mean_diff": diff.mean(),
            "a_wins": int((diff > 0).sum()), "n": len(diff), "statistic": stat, "p_value": p}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """mean ± std of kappa / accuracy per pipeline × protocol."""
    g = df.groupby(["protocol", "pipeline"])[["kappa", "accuracy"]]
    out = g.agg(["mean", "std"]).round(3)
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()
