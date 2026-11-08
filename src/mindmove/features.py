"""Scikit-learn transformers for EEG epochs.

All transformers take the *wide cached epochs* (n_trials, n_channels, n_times)
spanning EPOCH_TMIN..EPOCH_TMAX around the cue. Band-pass filtering is done on
the wide epoch and only then cropped to the classification window — WHY: a
zero-phase IIR filter has edge transients; filtering the long epoch pushes
them outside the 0.5-2.5 s window we classify.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

import config as C


def _crop_indices(tmin, tmax, epoch_tmin=C.EPOCH_TMIN, sfreq=C.FS):
    start = int(round((tmin - epoch_tmin) * sfreq))
    stop = int(round((tmax - epoch_tmin) * sfreq)) + 1
    return start, stop


def bandpass(X, fmin, fmax, sfreq=C.FS, order=4):
    """Zero-phase Butterworth band-pass along the last axis."""
    sos = butter(order, [fmin, fmax], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


class FilterCrop(BaseEstimator, TransformerMixin):
    """Band-pass the wide epoch, then crop to the classification window. Stateless."""

    def __init__(self, fmin=8.0, fmax=30.0, tmin=C.CLF_TMIN, tmax=C.CLF_TMAX,
                 epoch_tmin=C.EPOCH_TMIN, sfreq=C.FS):
        self.fmin, self.fmax, self.tmin, self.tmax = fmin, fmax, tmin, tmax
        self.epoch_tmin, self.sfreq = epoch_tmin, sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xf = bandpass(X, self.fmin, self.fmax, self.sfreq)
        a, b = _crop_indices(self.tmin, self.tmax, self.epoch_tmin, self.sfreq)
        return np.ascontiguousarray(Xf[..., a:b])


class FilterBank(BaseEstimator, TransformerMixin):
    """Apply several FilterCrop bands -> array (n_bands, n_trials, n_ch, n_t). Stateless."""

    def __init__(self, bands=tuple(C.FBTS_BANDS), tmin=C.CLF_TMIN, tmax=C.CLF_TMAX,
                 epoch_tmin=C.EPOCH_TMIN, sfreq=C.FS):
        self.bands, self.tmin, self.tmax = bands, tmin, tmax
        self.epoch_tmin, self.sfreq = epoch_tmin, sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.stack([
            FilterCrop(lo, hi, self.tmin, self.tmax, self.epoch_tmin, self.sfreq).transform(X)
            for lo, hi in self.bands
        ])


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """FBCSP feature extractor (Ang et al., 2008).

    One CSP per frequency band on the band-passed epochs; log-variance features
    of all bands are concatenated. Feature selection (MIBIF) is a separate
    pipeline step so it can be tuned independently.
    """

    def __init__(self, bands=tuple(C.FBCSP_BANDS), n_components=4,
                 tmin=C.CLF_TMIN, tmax=C.CLF_TMAX, reg="ledoit_wolf"):
        self.bands, self.n_components = bands, n_components
        self.tmin, self.tmax, self.reg = tmin, tmax, reg

    def fit(self, X, y):
        from mne.decoding import CSP
        Xb = FilterBank(self.bands, self.tmin, self.tmax).transform(X)
        self.csps_ = []
        for xb in Xb:
            csp = CSP(n_components=self.n_components, reg=self.reg, log=True,
                      norm_trace=False)
            csp.fit(xb, y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        Xb = FilterBank(self.bands, self.tmin, self.tmax).transform(X)
        return np.concatenate([csp.transform(xb) for csp, xb in zip(self.csps_, Xb)], axis=1)


class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    """MindMove features: per-band OAS covariance -> Riemannian tangent space, concatenated.

    WHY: FBCSP showed that different people use different frequency bands;
    the tangent space keeps *all* covariance information (not just a few
    CSP directions) in a Euclidean space where a linear model works.
    """

    def __init__(self, bands=tuple(C.FBTS_BANDS), estimator="oas", metric="riemann",
                 tmin=C.CLF_TMIN, tmax=C.CLF_TMAX):
        self.bands, self.estimator, self.metric = bands, estimator, metric
        self.tmin, self.tmax = tmin, tmax

    def fit(self, X, y=None):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        Xb = FilterBank(self.bands, self.tmin, self.tmax).transform(X)
        self.cov_ = Covariances(self.estimator)
        self.ts_ = []
        for xb in Xb:
            covs = self.cov_.transform(xb)
            self.ts_.append(TangentSpace(metric=self.metric).fit(covs))
        return self

    def transform(self, X):
        Xb = FilterBank(self.bands, self.tmin, self.tmax).transform(X)
        return np.concatenate(
            [ts.transform(self.cov_.transform(xb)) for ts, xb in zip(self.ts_, Xb)], axis=1)

    def band_slices(self):
        """Feature index range belonging to each band (for band importance)."""
        n_ch = self.ts_[0].reference_.shape[0]
        d = n_ch * (n_ch + 1) // 2
        return [slice(i * d, (i + 1) * d) for i in range(len(self.bands))]


_MI_CACHE: dict = {}


def mibif_score(X, y):
    """Mutual-information score for SelectKBest (deterministic, picklable).

    Memoised on the data bytes: inside GridSearchCV the same fold is scored
    once per SVM C value, and kNN-based MI estimation is the slow part.
    """
    import hashlib
    key = hashlib.sha1(np.ascontiguousarray(X).tobytes() + np.asarray(y).tobytes()).hexdigest()
    if key not in _MI_CACHE:
        if len(_MI_CACHE) > 256:
            _MI_CACHE.clear()
        _MI_CACHE[key] = mutual_info_classif(X, y, random_state=C.SEED)
    return _MI_CACHE[key]
