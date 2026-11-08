import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="session")
def synthetic_epochs():
    """Wide epochs (n, 22, 1376) where each class boosts a 10-20 Hz rhythm on its own channels."""
    import config as C
    from mindmove.data import N_TIMES
    from mindmove.features import bandpass
    rng = np.random.default_rng(0)
    n_per_class = 16
    X, y = [], []
    for c in range(4):
        for _ in range(n_per_class):
            x = rng.standard_normal((22, N_TIMES))
            rhythm = bandpass(rng.standard_normal(N_TIMES), 10, 20)
            for ch in range(c * 5, c * 5 + 3):
                x[ch] += 10 * rhythm * rng.uniform(0.8, 1.2)
            X.append(x)
            y.append(c)
    X = np.asarray(X) * 1e-6
    return X.astype(np.float32), np.asarray(y)


@pytest.fixture(scope="session")
def random_spd():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((60, 8, 50))
    return np.einsum("nct,ndt->ncd", A, A) / 50 + 0.1 * np.eye(8)
