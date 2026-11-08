"""Cross-subject transfer with Riemannian re-centering (Zanini et al., 2018).

Idea: the dominant difference between two people (or two days) is a *shift*
of their whole cloud of covariance matrices on the SPD manifold. Moving each
domain's Riemannian mean M to the identity,

        C  ->  M^{-1/2} C M^{-1/2},

aligns the clouds. M is computed from *unlabelled* trials, so a new patient
needs no labelled calibration data for this step.

We implement it directly with pyriemann.utils (≈10 lines) instead of
pyriemann.transfer — WHY: the arithmetic is fully visible for the report /
viva, and it does not depend on the transfer-module API.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

import config as C
from .features import FilterCrop


# ----------------------------------------------------------------- geometry
def covariances(X, band=C.MU_BETA, estimator="oas"):
    """Wide epochs -> OAS covariance matrices of the band-passed classification window."""
    from pyriemann.estimation import Covariances
    Xf = FilterCrop(*band).transform(X)
    return Covariances(estimator).transform(Xf)


def riemann_mean(covs):
    from pyriemann.geometry.mean import mean_riemann
    return mean_riemann(covs)


def recenter(covs, reference=None):
    """Re-center covariances at the identity using `reference` (default: their own mean)."""
    from pyriemann.geometry.base import invsqrtm
    M = riemann_mean(covs) if reference is None else reference
    P = invsqrtm(M)
    return P @ covs @ P


# ----------------------------------------------------------------- classifiers on covs
def _fit_ts_lr(covs, y, C_=1.0, sample_weight=None):
    from pyriemann.tangentspace import TangentSpace
    ts = TangentSpace(metric="riemann").fit(covs)
    lr = LogisticRegression(C=C_, max_iter=2000).fit(ts.transform(covs), y, sample_weight=sample_weight)
    return ts, lr


def _fit_predict(train_covs, y_train, test_covs, clf="TS+LR", sample_weight=None):
    if clf == "TS+LR":
        ts, lr = _fit_ts_lr(train_covs, y_train, sample_weight=sample_weight)
        return lr.predict(ts.transform(test_covs))
    if clf == "MDM":
        from pyriemann.classification import MDM
        return MDM(metric="riemann").fit(train_covs, y_train).predict(test_covs)
    raise KeyError(clf)


# ----------------------------------------------------------------- experiments
def loso(covs: dict, ys: dict, target: int, recentered: bool, clf="TS+LR"):
    """Leave-one-subject-out: train on both sessions of the 8 other subjects,
    test on the target's session E.

    covs / ys: {(subject, session): array}. Returns y_pred for target E.
    """
    src = [k for k in covs if k[0] != target]
    if recentered:
        Xtr = np.concatenate([recenter(covs[k]) for k in src])
        Xte = recenter(covs[(target, "E")])            # unlabelled target data only
    else:
        Xtr = np.concatenate([covs[k] for k in src])
        Xte = covs[(target, "E")]
    ytr = np.concatenate([ys[k] for k in src])
    return _fit_predict(Xtr, ytr, Xte, clf)


def recenter_all(covs: dict) -> dict:
    """Re-center every domain (subject, session) at the identity with its own mean."""
    return {k: recenter(v) for k, v in covs.items()}


def sample_per_class(y, k, rng):
    """Indices of k random trials per class."""
    return np.concatenate([rng.choice(np.flatnonzero(y == c), k, replace=False)
                           for c in np.unique(y)])


def calibration_point(covs, ys, target, k, method, rng, rc=None):
    """Accuracy on target E when using k labelled target trials/class from session T.

    method:
      'subject_only'     k target trials, no re-centering (standard calibration)
      'subject_only_rc'  k target trials, each session re-centered (unsupervised)
      'transfer_rc'      k target trials + all 8 other subjects, everything re-centered.
                         Domain-balanced weights: the k*4 target trials together get the same
                         total weight as all source trials (WHY: otherwise 8 or 20 target trials
                         are drowned out by ~4600 source trials and extra calibration data
                         barely changes the model).
    rc: optional {domain: re-centered covs} cache (see recenter_all) to avoid recomputation.
    """
    rc = rc if rc is not None else recenter_all(covs)
    cT, yT = covs[(target, "T")], ys[(target, "T")]
    cE, yE = covs[(target, "E")], ys[(target, "E")]
    idx = sample_per_class(yT, k, rng) if k > 0 else np.array([], int)

    if method == "subject_only":
        if k == 0:
            return np.nan
        y_pred = _fit_predict(cT[idx], yT[idx], cE)
    elif method == "subject_only_rc":
        if k == 0:
            return np.nan
        y_pred = _fit_predict(rc[(target, 'T')][idx], yT[idx], rc[(target, 'E')])
    elif method == "transfer_rc":
        src = [kk for kk in covs if kk[0] != target]
        Xtr = [rc[kk] for kk in src]
        ytr = [ys[kk] for kk in src]
        n_src = sum(len(v) for v in ytr)
        w = [np.ones(n_src)]
        if k > 0:
            Xtr.append(rc[(target, "T")][idx])
            ytr.append(yT[idx])
            w.append(np.full(len(idx), n_src / len(idx)))
        w = np.concatenate(w)
        y_pred = _fit_predict(np.concatenate(Xtr), np.concatenate(ytr), rc[(target, "E")],
                              sample_weight=w / w.mean())
    else:
        raise KeyError(method)
    return float((y_pred == yE).mean())


def tangent_vectors(covs_list, recentered: bool):
    """Tangent-space vectors of several domains at their common mean (for t-SNE)."""
    from pyriemann.tangentspace import TangentSpace
    blocks = [recenter(c) if recentered else c for c in covs_list]
    allc = np.concatenate(blocks)
    return TangentSpace(metric="riemann").fit_transform(allc)


# ----------------------------------------------------------------- deployable model
class RecenteredTSModel:
    """'Unseen patient' decoder: trained on other subjects, re-centered.

    Usage: model.fit_reference(X_unlabelled_epochs_of_new_patient); model.predict_proba(X)
    """

    def __init__(self, band=C.MU_BETA):
        self.band = band
        self.reference_ = None

    def fit(self, covs_by_domain: list, ys: list):
        Xtr = np.concatenate([recenter(c) for c in covs_by_domain])
        self.ts_, self.lr_ = _fit_ts_lr(Xtr, np.concatenate(ys))
        self.classes_ = self.lr_.classes_
        return self

    def fit_reference(self, X):
        """Estimate the new patient's manifold centre from unlabelled epochs."""
        self.reference_ = riemann_mean(covariances(X, self.band))
        return self

    def predict_proba(self, X):
        covs = covariances(X, self.band)
        ref = self.reference_ if self.reference_ is not None else riemann_mean(covs)
        return self.lr_.predict_proba(self.ts_.transform(recenter(covs, ref)))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
