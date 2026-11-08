"""Data loading for BCI Competition IV 2a (BNCI2014_001).

One code path for both download routes: we always obtain *continuous* runs,
filter them 1-45 Hz (filtering continuous data avoids epoch-edge artefacts),
then cut wide epochs (-1.5 .. +4.0 s around the cue) ourselves.

    route B / local files : A0xT.mat / A0xE.mat parsed with scipy  (preferred when present)
    route A / MOABB       : BNCI2014_001().get_data()             (needs bnci-horizon-2020.eu)

WHY local .mat first: it needs no network, so loading never hangs on a
server timeout; MOABB is only used when no .mat files are on disk.

Cached result per subject: data/processed/A0x.npz with
    X_T, X_E   float32 (288, 22, 1376) volts
    y_T, y_E   int8   class index 0..3 (see config.CLASSES)
    art_T, art_E  bool  artefact flag from the dataset (trials are kept, flag stored)
    times      float  (1376,) seconds relative to cue
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import config as C

N_TIMES = int(round((C.EPOCH_TMAX - C.EPOCH_TMIN) * C.FS)) + 1
TIMES = np.linspace(C.EPOCH_TMIN, C.EPOCH_TMAX, N_TIMES)
CUE_OFFSET_S = 2.0          # 'trial' marks trial start (fixation); cue appears 2 s later


# ----------------------------------------------------------------- info / montage
def make_info():
    """MNE Info with the 22 standard 10-20 EEG channels + montage."""
    import mne
    info = mne.create_info(C.CHANNELS, C.FS, ch_types="eeg")
    info.set_montage("standard_1020")
    return info


# ----------------------------------------------------------------- raw sources
def find_mat_file(subject: int, session: str) -> Path | None:
    name = f"A{subject:02d}{session}.mat"
    candidates = [C.KAGGLE_DIR / "BCICIV-2a-mat" / name, C.KAGGLE_DIR / name,
                  C.MOABB_DIR / "MNE-bnci-data" / "database" / "data-sets" / "001-2014" / name]
    candidates += list(C.DATA_RAW.rglob(name))
    for p in candidates:
        if p.exists():
            return p
    return None


def load_bnci_mat(path) -> list[dict]:
    """Parse one A0x{T,E}.mat file -> list of motor-imagery runs.

    Each run: {'X': (n_samples, 22) volts, 'trial': cue-less trial-start sample
    indices (0-based), 'y': labels 1..4, 'artifacts': 0/1, 'fs': 250}.
    Runs with an empty 'trial' field are EOG calibration runs and are skipped.
    """
    import scipy.io as sio
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    runs = []
    for r in np.atleast_1d(m["data"]):
        trial = np.atleast_1d(r.trial)
        if trial.size == 0:
            continue
        X = np.asarray(r.X, dtype=np.float64)[:, :22] * 1e-6       # µV -> V, drop 3 EOG
        runs.append({
            "X": X,
            "trial": trial.astype(int) - 1,                         # MATLAB 1-based -> 0-based
            "y": np.atleast_1d(r.y).astype(int),
            "artifacts": np.atleast_1d(r.artifacts).astype(bool),
            "fs": int(r.fs),
        })
    return runs


def _runs_from_moabb(subject: int) -> dict[str, list[dict]]:
    """Route A: continuous runs through MOABB (requires the BNCI server)."""
    import mne
    import moabb.utils
    from moabb.datasets import BNCI2014_001
    moabb.utils.set_download_dir(str(C.MOABB_DIR))
    data = BNCI2014_001().get_data(subjects=[subject])[subject]
    code_to_label = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
    out = {}
    for sess_key, runs in data.items():
        session = "T" if ("train" in sess_key.lower() or sess_key.startswith("0")) else "E"
        out[session] = []
        for _, raw in sorted(runs.items()):
            raw = raw.copy().pick(C.CHANNELS)
            events, ev_id = mne.events_from_annotations(raw, verbose=False)
            inv = {v: k for k, v in ev_id.items()}
            keep = [i for i, e in enumerate(events) if inv[e[2]] in code_to_label]
            if not keep:
                continue
            ev = events[keep]
            out[session].append({
                "X": raw.get_data().T,
                "trial": ev[:, 0] - raw.first_samp,      # MOABB events mark trial start, like .mat
                "y": np.array([code_to_label[inv[c]] for c in ev[:, 2]]),
                "artifacts": np.zeros(len(ev), bool),
                "fs": int(raw.info["sfreq"]),
            })
    return out


def get_runs(subject: int) -> dict[str, list[dict]]:
    """Continuous MI runs per session, from local .mat if present else MOABB."""
    paths = {s: find_mat_file(subject, s) for s in C.SESSIONS}
    if all(paths.values()):
        return {s: load_bnci_mat(p) for s, p in paths.items()}
    return _runs_from_moabb(subject)


# ----------------------------------------------------------------- epoching
def _fill_nans(X):
    """A few samples are NaN (amplifier saturation) -> linear interpolation per channel."""
    if not np.isnan(X).any():
        return X
    X = X.copy()
    idx = np.arange(X.shape[0])
    for ch in range(X.shape[1]):
        bad = np.isnan(X[:, ch])
        if bad.all():
            X[:, ch] = 0.0
        elif bad.any():
            X[bad, ch] = np.interp(idx[bad], idx[~bad], X[~bad, ch])
    return X


def epoch_runs(runs: list[dict]):
    """Filter each continuous run 1-45 Hz and cut wide epochs around the cue."""
    import mne
    Xs, ys, arts = [], [], []
    a = int(round(C.EPOCH_TMIN * C.FS))
    for run in runs:
        assert run["fs"] == C.FS, f"unexpected fs {run['fs']}"
        sig = _fill_nans(run["X"]).T                                  # (22, n_samples)
        sig = mne.filter.filter_data(sig, C.FS, C.CACHE_L_FREQ, C.CACHE_H_FREQ,
                                     verbose=False)
        cues = run["trial"] + int(CUE_OFFSET_S * C.FS)
        for cue, lab, art in zip(cues, run["y"], run["artifacts"]):
            start = cue + a
            seg = sig[:, start:start + N_TIMES]
            if start < 0 or seg.shape[1] != N_TIMES:
                raise ValueError("epoch exceeds run boundaries")
            Xs.append(seg)
            ys.append(lab - 1)
            arts.append(art)
    return (np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int8),
            np.asarray(arts, dtype=bool))


# ----------------------------------------------------------------- cache API
def cache_path(subject: int) -> Path:
    return C.DATA_PROCESSED / f"A{subject:02d}.npz"


def build_cache(subject: int, force: bool = False) -> Path:
    path = cache_path(subject)
    if path.exists() and not force:
        return path
    runs = get_runs(subject)
    arrays = {"times": TIMES}
    for s in C.SESSIONS:
        X, y, art = epoch_runs(runs[s])
        arrays.update({f"X_{s}": X, f"y_{s}": y, f"art_{s}": art})
    np.savez(path, **arrays)
    return path


def load_subject(subject: int, sessions=tuple(C.SESSIONS)) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{session: (X volts (288, 22, n_times), y int 0..3)} — builds the cache on first use."""
    path = build_cache(subject)
    with np.load(path) as z:
        return {s: (z[f"X_{s}"], z[f"y_{s}"]) for s in sessions}


def load_artifacts(subject: int) -> dict[str, np.ndarray]:
    with np.load(build_cache(subject)) as z:
        return {s: z[f"art_{s}"] for s in C.SESSIONS}


def labels_to_names(y) -> np.ndarray:
    return np.asarray(C.CLASSES)[np.asarray(y, dtype=int)]


def verify_subject(subject: int) -> dict:
    """Check 288 trials/session, 72 per class, 22 channels."""
    d = load_subject(subject)
    report = {}
    for s, (X, y) in d.items():
        counts = np.bincount(y, minlength=4)
        ok = X.shape[0] == C.N_TRIALS_PER_SESSION and X.shape[1] == 22 and (counts == 72).all()
        report[s] = {"shape": X.shape, "per_class": counts.tolist(), "ok": bool(ok),
                     "finite": bool(np.isfinite(X).all())}
    return report
