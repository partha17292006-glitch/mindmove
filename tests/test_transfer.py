import numpy as np

from mindmove import transfer


def test_recenter_mean_is_identity(random_spd):
    rc = transfer.recenter(random_spd)
    M = transfer.riemann_mean(rc)
    assert np.allclose(M, np.eye(M.shape[0]), atol=1e-6)


def test_recenter_keeps_spd(random_spd):
    rc = transfer.recenter(random_spd)
    assert np.allclose(rc, np.transpose(rc, (0, 2, 1)))
    assert (np.linalg.eigvalsh(rc) > 0).all()


def test_recentered_model_runs(synthetic_epochs):
    X, y = synthetic_epochs
    covs = transfer.covariances(X)
    model = transfer.RecenteredTSModel().fit([covs[:32], covs[32:]], [y[:32], y[32:]])
    model.fit_reference(X)
    p = model.predict_proba(X[:5])
    assert p.shape == (5, 4) and np.allclose(p.sum(1), 1)
