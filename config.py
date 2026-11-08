"""Central configuration for MindMove: paths, signal constants, bands, seed.

Every script / module imports from here so a single change (e.g. the
classification window) propagates everywhere.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
KAGGLE_DIR = DATA_RAW / "kaggle"                      # route B (Kaggle mirror)
MOABB_DIR = DATA_RAW                                  # route A (MOABB cache root)
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- dataset
SUBJECTS = list(range(1, 10))           # A01 .. A09
SESSIONS = ["T", "E"]                   # T = training day, E = evaluation day
FS = 250                                # Hz
CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
CLASSES = ["left_hand", "right_hand", "feet", "tongue"]   # label codes 1..4 in the .mat files
CLASS_LABELS = {c: c.replace("_", " ").title() for c in CLASSES}
N_TRIALS_PER_SESSION = 288

# ---------------------------------------------------------------- epoching
# Wide cache window around the cue (t=0 = cue onset). Baseline (-1.5..-0.5 s)
# is needed for ERD maps; +4 s covers the whole imagery period for the stream.
EPOCH_TMIN, EPOCH_TMAX = -1.5, 4.0
CACHE_L_FREQ, CACHE_H_FREQ = 1.0, 45.0
BASELINE = (-1.5, -0.5)

# Classification window: skip the first 0.5 s (visual evoked response to the
# cue, not imagery); 2 s gives a stable covariance estimate.
CLF_TMIN, CLF_TMAX = 0.5, 2.5

# ---------------------------------------------------------------- bands
MU_BETA = (8.0, 30.0)                                           # single-band pipelines
FBCSP_BANDS = [(f, f + 4.0) for f in range(4, 40, 4)]           # 9 bands, 4-8 ... 36-40 Hz
FBTS_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 30), (30, 40)]  # 6 bands

SEED = 42

# BCI Competition IV 2a winner (FBCSP, Ang et al.), kappa per subject A01..A09 (competition results page)
WINNER_KAPPA = [0.68, 0.42, 0.75, 0.48, 0.40, 0.27, 0.77, 0.75, 0.61]
WINNER_KAPPA_MEAN = 0.57
