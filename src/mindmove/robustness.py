"""Robustness: broken electrodes and additive noise at test time.

Electrode failure model: a detached / broken electrode records large
broadband noise unrelated to the brain. We replace k random channels with
white noise of 5x the channel's own standard deviation, then either
  * leave them ('broken')                  -> what a naive system would see
  * repair them ('interpolated')           -> mark as bad and rebuild from the
    neighbours with MNE spherical-spline interpolation (Perrin et al., 1989).
"""
from __future__ import annotations

import numpy as np

import config as C

NOISE_FACTOR = 5.0


def break_channels(X, bad_idx, rng, factor=NOISE_FACTOR):
    """Replace channels `bad_idx` with white noise (factor × their std)."""
    Xb = np.array(X, dtype=np.float64, copy=True)
    if len(bad_idx) == 0:
        return Xb
    std = X[:, bad_idx, :].std(axis=(0, 2))
    Xb[:, bad_idx, :] = rng.standard_normal(Xb[:, bad_idx, :].shape) * (factor * std)[None, :, None]
    return Xb


def interpolate_channels(X, bad_idx, info=None):
    """Spherical-spline interpolation of channels `bad_idx` (MNE)."""
    import mne

    from .data import make_info
    if len(bad_idx) == 0:
        return np.asarray(X, dtype=np.float64)
    info = (info or make_info()).copy()
    info["bads"] = [C.CHANNELS[i] for i in bad_idx]
    ep = mne.EpochsArray(np.asarray(X, dtype=np.float64), info, tmin=C.EPOCH_TMIN,
                         baseline=None, verbose=False)
    ep.interpolate_bads(reset_bads=True, verbose=False)
    return ep.get_data(copy=False)


def corrupt_electrodes(X, k, rng, repair: bool, info=None):
    bad = np.sort(rng.choice(len(C.CHANNELS), size=k, replace=False)) if k else np.array([], int)
    Xb = break_channels(X, bad, rng)
    return (interpolate_channels(Xb, bad, info) if repair else Xb), bad


def add_noise(X, snr_db, rng):
    """Additive white Gaussian noise at a given per-channel SNR (dB). snr_db=inf -> clean."""
    X = np.asarray(X, dtype=np.float64)
    if np.isinf(snr_db):
        return X.copy()
    p_sig = (X ** 2).mean(axis=(0, 2), keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return X + rng.standard_normal(X.shape) * np.sqrt(p_noise)
