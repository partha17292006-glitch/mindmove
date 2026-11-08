import numpy as np
import pytest

from mindmove.features import FilterBankTangentSpace, FilterCrop
from mindmove.pipelines import PIPELINE_NAMES, make_pipeline


def test_filtercrop_shape(synthetic_epochs):
    X, _ = synthetic_epochs
    out = FilterCrop(8, 30).transform(X[:4])
    assert out.shape == (4, 22, 501)                       # 0.5..2.5 s at 250 Hz


def test_fbts_feature_dim(synthetic_epochs):
    X, y = synthetic_epochs
    fb = FilterBankTangentSpace().fit(X, y)
    assert fb.transform(X[:3]).shape == (3, 6 * 22 * 23 // 2)


@pytest.mark.parametrize("name", PIPELINE_NAMES)
def test_pipeline_fit_predict(name, synthetic_epochs):
    X, y = synthetic_epochs
    tr = np.arange(len(y)) % 4 != 0
    pipe = make_pipeline(name).fit(X[tr], y[tr])
    pred = pipe.predict(X[~tr])
    assert pred.shape == ((~tr).sum(),)
    proba = pipe.predict_proba(X[~tr])
    assert np.allclose(proba.sum(1), 1)
    assert (pred == y[~tr]).mean() > 0.4                   # well above 4-class chance (0.25)
