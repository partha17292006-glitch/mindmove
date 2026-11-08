import numpy as np
import pytest
import scipy.io as sio

import config as C
from mindmove import data


def _fake_mat(path, n_runs=2, trials_per_run=8, fs=250):
    rng = np.random.default_rng(0)
    runs = np.empty(n_runs + 1, dtype=object)
    # first run = EOG calibration run with no trials (must be skipped)
    runs[0] = {"X": rng.standard_normal((3000, 25)), "trial": np.zeros((0, 1)), "y": np.zeros((0, 1)),
               "fs": fs, "artifacts": np.zeros((0, 1))}
    for r in range(1, n_runs + 1):
        n = 2000 * trials_per_run + 3000
        trial = (np.arange(trials_per_run) * 2000 + 500 + 1).reshape(-1, 1)  # 1-based like MATLAB
        y = (np.arange(trials_per_run) % 4 + 1).reshape(-1, 1)
        runs[r] = {"X": rng.standard_normal((n, 25)), "trial": trial, "y": y, "fs": fs,
                   "artifacts": np.zeros((trials_per_run, 1))}
    sio.savemat(path, {"data": runs})


def test_load_bnci_mat_and_epoch(tmp_path):
    p = tmp_path / "A99T.mat"
    _fake_mat(p)
    runs = data.load_bnci_mat(p)
    assert len(runs) == 2                                  # EOG run skipped
    assert runs[0]["X"].shape[1] == 22                     # EOG channels dropped
    assert runs[0]["trial"][0] == 500                      # converted to 0-based
    X, y, art = data.epoch_runs(runs)
    assert X.shape == (16, 22, data.N_TIMES)
    assert set(y.tolist()) == {0, 1, 2, 3}
    assert art.dtype == bool and not art.any()


def test_times_axis():
    assert data.TIMES[0] == pytest.approx(C.EPOCH_TMIN)
    assert data.TIMES[-1] == pytest.approx(C.EPOCH_TMAX)
    assert len(data.TIMES) == 1376


@pytest.mark.skipif(not data.cache_path(1).exists(), reason="real-data cache not built")
def test_real_subject_shapes():
    rep = data.verify_subject(1)
    for sess in C.SESSIONS:
        assert rep[sess]["shape"] == (288, 22, 1376)
        assert rep[sess]["per_class"] == [72, 72, 72, 72]
        assert rep[sess]["finite"]
