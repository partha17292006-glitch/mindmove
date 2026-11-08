"""Figures: signals, ERD/ERS, topomaps, manifold view, results charts.

Every function returns a matplotlib Figure (scripts save them, the
Streamlit app displays them).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402

CLASS_COLORS = {"left_hand": "#2a78d6", "right_hand": "#e5484d", "feet": "#30a46c", "tongue": "#e09f3e"}
MOTOR_CHANNELS = ["C3", "Cz", "C4"]


def _ch(name):
    return C.CHANNELS.index(name)


# ----------------------------------------------------------------- signals
def plot_raw_trial(X_trial, times, channels=MOTOR_CHANNELS, title=""):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, ch in enumerate(channels):
        ax.plot(times, X_trial[_ch(ch)] * 1e6 + i * 40, lw=0.8, label=ch)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvspan(C.CLF_TMIN, C.CLF_TMAX, color="0.9", zorder=0)
    ax.set_yticks([i * 40 for i in range(len(channels))], channels)
    ax.set_xlabel("time relative to cue (s)")
    ax.set_title(title or "EEG trial (grey = classification window)")
    fig.tight_layout()
    return fig


def class_psd(X, y, channel="C3", fmin=4, fmax=40):
    """Welch PSD (dB) per class in the classification window."""
    from scipy.signal import welch
    a = int((C.CLF_TMIN - C.EPOCH_TMIN) * C.FS)
    b = int((C.CLF_TMAX - C.EPOCH_TMIN) * C.FS)
    f, P = welch(X[:, _ch(channel), a:b], fs=C.FS, nperseg=250, axis=-1)
    keep = (f >= fmin) & (f <= fmax)
    return f[keep], {c: 10 * np.log10(P[y == i][:, keep].mean(0)) for i, c in enumerate(C.CLASSES)}


def plot_psd(X, y, channels=MOTOR_CHANNELS, title="Power spectrum during imagery"):
    fig, axes = plt.subplots(1, len(channels), figsize=(12, 3.4), sharey=True)
    for ax, ch in zip(np.atleast_1d(axes), channels):
        f, psd = class_psd(X, y, ch)
        for c, p in psd.items():
            ax.plot(f, p, color=CLASS_COLORS[c], label=C.CLASS_LABELS[c], lw=1.4)
        ax.axvspan(8, 13, color="0.93", zorder=0)
        ax.set_title(ch)
        ax.set_xlabel("Hz")
    np.atleast_1d(axes)[0].set_ylabel("dB (V²/Hz)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- ERD / ERS
def compute_erd(X, y, channels=MOTOR_CHANNELS, freqs=None, decim=5):
    """Morlet TFR per class, % power change vs baseline (-1.5..-0.5 s).

    Returns freqs, times, {class: array (n_channels, n_freqs, n_times)}.
    """
    from mne.time_frequency import tfr_array_morlet
    freqs = np.arange(4, 41, 1.0) if freqs is None else freqs
    idx = [_ch(c) for c in channels]
    times = np.linspace(C.EPOCH_TMIN, C.EPOCH_TMAX, X.shape[-1])[::decim]
    bmask = (times >= C.BASELINE[0]) & (times <= C.BASELINE[1])
    out = {}
    for i, c in enumerate(C.CLASSES):
        data = X[y == i][:, idx].astype(np.float64)
        power = tfr_array_morlet(data, C.FS, freqs, n_cycles=freqs / 2.0, output="avg_power",
                                 decim=decim, n_jobs=1)
        base = power[..., bmask].mean(-1, keepdims=True)
        out[c] = 100 * (power - base) / base
    return freqs, times, out


def plot_erd(freqs, times, erd, channels=MOTOR_CHANNELS, title="ERD/ERS (% vs baseline)"):
    fig, axes = plt.subplots(len(erd), len(channels), figsize=(11, 2.3 * len(erd)),
                             sharex=True, sharey=True)
    vmax = 60
    for r, (c, pw) in enumerate(erd.items()):
        for k, ch in enumerate(channels):
            ax = axes[r, k]
            im = ax.pcolormesh(times, freqs, pw[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               shading="auto")
            ax.axvline(0, color="k", lw=0.8)
            if r == 0:
                ax.set_title(ch)
            if k == 0:
                ax.set_ylabel(f"{C.CLASS_LABELS[c]}\nHz")
            if r == len(erd) - 1:
                ax.set_xlabel("s")
    fig.colorbar(im, ax=axes, shrink=0.6, label="% power change (blue = ERD)")
    fig.suptitle(title)
    return fig


def band_erd_topo(X, y, band=(8, 13)):
    """Per-class, per-channel log-power change (dB) in `band`, imagery vs baseline."""
    from .features import bandpass
    Xf = bandpass(X.astype(np.float64), *band)
    t = np.linspace(C.EPOCH_TMIN, C.EPOCH_TMAX, X.shape[-1])
    act = (t >= C.CLF_TMIN) & (t <= C.CLF_TMAX)
    base = (t >= C.BASELINE[0]) & (t <= C.BASELINE[1])
    pa = (Xf[..., act] ** 2).mean(-1)
    pb = (Xf[..., base] ** 2).mean(-1)
    return {c: 10 * np.log10(pa[y == i].mean(0) / pb[y == i].mean(0))
            for i, c in enumerate(C.CLASSES)}


def plot_topomaps(values: dict, info, title="", cmap="RdBu_r", symmetric=True, unit=""):
    import mne
    fig, axes = plt.subplots(1, len(values), figsize=(3 * len(values), 3.2))
    allv = np.concatenate([np.ravel(v) for v in values.values()])
    vmax = np.abs(allv).max()
    vlim = (-vmax, vmax) if symmetric else (allv.min(), allv.max())
    for ax, (name, v) in zip(np.atleast_1d(axes), values.items()):
        im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap=cmap, vlim=vlim,
                                     names=C.CHANNELS if False else None, contours=4)
        ax.set_title(C.CLASS_LABELS.get(name, name))
    fig.colorbar(im, ax=axes, shrink=0.7, label=unit)
    fig.suptitle(title)
    return fig


def ovr_csp_patterns(X, y, band=C.MU_BETA):
    """One-vs-rest CSP pattern per class: the spatial direction where the class
    shows the strongest power *decrease* (ERD) relative to the other classes.

    Generalised eigenproblem  C_class w = λ (C_class + C_rest) w;  the smallest λ
    = least relative class variance. Pattern a = Σ w (Haufe et al., 2014) — the
    physiologically interpretable map (filters themselves are not).
    """
    from scipy.linalg import eigh

    from .features import FilterCrop
    Xf = FilterCrop(*band).transform(X)
    covs = np.einsum("nct,ndt->ncd", Xf, Xf) / Xf.shape[-1]
    covs /= np.trace(covs, axis1=1, axis2=2)[:, None, None]
    pats = {}
    for i, c in enumerate(C.CLASSES):
        Ca, Cb = covs[y == i].mean(0), covs[y != i].mean(0)
        _, W = eigh(Ca, Ca + Cb)
        w = W[:, 0]
        a = (Ca + Cb) @ w
        pats[c] = np.abs(a) / np.abs(a).max()
    return pats


# ----------------------------------------------------------------- manifold
def plot_tsne(emb_before, emb_after, labels, label_name="subject", title=""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labs = np.asarray(labels)
    uniq = list(dict.fromkeys(labs))
    cmap = plt.get_cmap("tab10")
    for ax, emb, ttl in zip(axes, (emb_before, emb_after), ("before re-centering", "after re-centering")):
        for j, u in enumerate(uniq):
            m = labs == u
            col = CLASS_COLORS.get(u, cmap(j % 10))
            ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.6, color=col,
                       label=C.CLASS_LABELS.get(u, f"{label_name} {u}"))
        ax.set_title(ttl)
        ax.set_xticks([]), ax.set_yticks([])
    axes[1].legend(markerscale=3, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle(title or f"t-SNE of tangent vectors, coloured by {label_name}")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- results
def plot_benchmark_bar(df: pd.DataFrame, protocol="cross_session", winner_kappa=0.57):
    sub = df[df.protocol == protocol]
    order = sub.groupby("pipeline").kappa.mean().sort_values().index.tolist()
    piv = sub.pivot_table(index="subject", columns="pipeline", values="kappa")[order]
    fig, ax = plt.subplots(figsize=(11, 4))
    n = len(order)
    w = 0.8 / n
    x = np.arange(len(piv.index))
    for j, p in enumerate(order):
        ax.bar(x + j * w - 0.4 + w / 2, piv[p], w, label=f"{p} (mean {piv[p].mean():.2f})")
    ax.axhline(winner_kappa, color="k", ls="--", lw=1, label=f"BCI-IV winner κ={winner_kappa}")
    ax.set_xticks(x, [f"A{s:02d}" for s in piv.index])
    ax.set_ylabel("Cohen's κ")
    ax.set_title(f"Per-subject kappa — {protocol.replace('_', ' ')}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_confusion(cm, title="Confusion matrix (row-normalised)"):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    labels = [C.CLASS_LABELS[c] for c in C.CLASSES]
    ax.set_xticks(range(4), labels, rotation=30)
    ax.set_yticks(range(4), labels)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_band_importance(importance: pd.DataFrame):
    """importance: columns band, subject, importance (L2 norm of LR coefficients)."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    order = [f"{lo}-{hi} Hz" for lo, hi in C.FBTS_BANDS]
    g = importance.groupby("band").importance
    m, s = g.mean().reindex(order), g.std().reindex(order)
    ax.bar(order, m, yerr=s, color="#2a78d6", capsize=3)
    ax.set_ylabel("share of |coef| (mean ± sd over subjects)")
    ax.set_title("FBTS+LR: which frequency bands drive the decision")
    fig.tight_layout()
    return fig


def plot_curve(df, x, y, hue, title, xlabel, ylabel="accuracy", chance=0.25):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for h, g in df.groupby(hue, sort=False):
        agg = g.groupby(x)[y].agg(["mean", "sem"]).reset_index()
        ax.errorbar(agg[x], agg["mean"], yerr=agg["sem"], marker="o", capsize=3, label=h)
    if chance is not None:
        ax.axhline(chance, color="0.5", ls=":", label="chance")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
