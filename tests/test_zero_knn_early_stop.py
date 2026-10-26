"""Regression: explicit k_nn=0 must not divide by zero during early-stop setup."""
import numpy as np
import pytest

from pcc import ParticleCompetitionAndCooperation


@pytest.mark.parametrize("early_stop", [False, True])
def test_explicit_zero_knn_does_not_crash(early_stop):
    model = ParticleCompetitionAndCooperation(impl="numpy")
    neighbors = np.array([[1], [0], [3], [2]], dtype=np.int64)
    degrees = np.ones(4, dtype=np.int64)
    labels = np.array([0, -1, 1, -1], dtype=np.int64)
    model.set_graph(neighbors, degrees, k_nn=0)
    np.random.seed(12345)
    result = model.fit_predict(labels, max_iter=2, early_stop=early_stop, es_chk=1)
    assert result.shape == labels.shape
    np.testing.assert_array_equal(result[[0, 2]], labels[[0, 2]])
