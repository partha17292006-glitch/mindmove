"""The six decoding pipelines compared in MindMove.

Every pipeline consumes wide cached epochs (n_trials, 22, n_times) and is a
plain scikit-learn Pipeline, so cross-validation / joblib persistence work
out of the box.

Hyper-parameter tuning: the *head* (classifier / feature selector) is wrapped
in an inner GridSearchCV(cv=5). WHY: the expensive feature extraction (filter
bank + covariances) then runs once per fit instead of once per grid point,
which makes the full benchmark ~20x faster. The steps before the head are
either stateless (filters) or unsupervised (covariance, tangent-space
reference, scaler), so no label information leaks into the inner CV, and the
test session is never seen during tuning.
"""
from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config as C
from .features import (FilterBankCSP, FilterBankTangentSpace, FilterCrop,
                       mibif_score)

PIPELINE_NAMES = ["CSP+LDA", "FBCSP+SVM", "MDM", "TS+LR", "TS+SVM", "FBTS+LR"]
OURS = "FBTS+LR"


def _inner_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)


def _tuned(estimator, grid):
    return GridSearchCV(estimator, grid, cv=_inner_cv(), scoring="accuracy", n_jobs=1)


def _lr(C_=1.0, max_iter=2000):
    return LogisticRegression(C=C_, max_iter=max_iter)


def make_pipeline(name: str) -> Pipeline:
    from mne.decoding import CSP
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    lo, hi = C.MU_BETA
    if name == "CSP+LDA":
        return Pipeline([
            ("filter", FilterCrop(lo, hi)),
            ("csp", CSP(n_components=8, reg="ledoit_wolf", log=True)),
            ("lda", LDA(solver="lsqr", shrinkage="auto")),
        ])
    if name == "FBCSP+SVM":
        head = Pipeline([
            ("mibif", SelectKBest(mibif_score, k=16)),
            ("svm", SVC(kernel="linear", probability=True, random_state=C.SEED)),
        ])
        return Pipeline([
            ("fbcsp", FilterBankCSP(C.FBCSP_BANDS, n_components=4)),
            ("scale", StandardScaler()),
            ("head", _tuned(head, {"mibif__k": [8, 16, 24, 36],
                                   "svm__C": [0.01, 0.1, 1.0]})),
        ])
    if name == "MDM":
        return Pipeline([
            ("filter", FilterCrop(lo, hi)),
            ("cov", Covariances("oas")),
            ("mdm", MDM(metric="riemann")),
        ])
    if name == "TS+LR":
        return Pipeline([
            ("filter", FilterCrop(lo, hi)),
            ("cov", Covariances("oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("head", _tuned(_lr(), {"C": [0.01, 0.1, 1.0, 10.0]})),
        ])
    if name == "TS+SVM":
        return Pipeline([
            ("filter", FilterCrop(lo, hi)),
            ("cov", Covariances("oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("head", _tuned(SVC(kernel="linear", probability=True, random_state=C.SEED),
                            {"C": [0.01, 0.1, 1.0]})),
        ])
    if name == "FBTS+LR":
        return Pipeline([
            ("fbts", FilterBankTangentSpace(C.FBTS_BANDS)),
            ("scale", StandardScaler()),
            ("head", _tuned(_lr(), {"C": [0.001, 0.01, 0.1, 1.0]})),
        ])
    raise KeyError(f"Unknown pipeline {name!r}; choose from {PIPELINE_NAMES}")


def get_pipelines(names=None) -> dict[str, Pipeline]:
    """Return {name: fresh unfitted Pipeline}."""
    return {n: make_pipeline(n) for n in (names or PIPELINE_NAMES)}
