"""MindMove: decoding imagined movement from EEG with classical (Riemannian) ML."""
import sys
from pathlib import Path

# Make the project-root `config.py` importable from anywhere in the package.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # keep MNE quiet (CSP prints rank estimates for every fit)
    import mne as _mne
    _mne.set_log_level("ERROR")
except ImportError:  # pragma: no cover
    pass

__version__ = "1.0.0"
