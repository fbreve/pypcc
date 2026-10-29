"""Regression tests for opt-in public stopping metadata."""
import numpy as np
import pytest

import pcc
from pcc import ParticleCompetitionAndCooperation


@pytest.mark.parametrize("backend", ["numpy", "numba", "cython"])
@pytest.mark.parametrize("criterion", ["potential", "dominance_change"])
def test_return_info_preserves_predictions_state_and_rng(backend, criterion):
    if criterion == "dominance_change" and backend != "numpy":
        pytest.skip("experimental criterion is NumPy-only")
    if backend == "numba" and not pcc._HAS_NUMBA:
        pytest.skip("Numba not installed")
    if backend == "cython" and not pcc._HAS_CYTHON:
        pytest.skip("Cython extension not installed")

    graph = np.array([[1, 2], [0, 2], [1, 3], [2, 1]], dtype=np.int64)
    degrees = np.array([2, 2, 2, 2], dtype=np.int64)
    labels = np.array([0, -1, -1, 1], dtype=np.int64)

    def run(info):
        model = ParticleCompetitionAndCooperation(impl=backend)
        model.set_graph(graph, degrees, k_nn=2)
        result = model.fit_predict(labels, max_iter=40, early_stop=True,
                                   es_chk=20, seed=123, stop_criterion=criterion,
                                   return_info=info)
        prediction, metadata = result if info else (result, None)
        return (prediction, model.node.dominance.copy(),
                model.part.curnode.copy(), model.part.strength.copy(),
                model.part.dist_table.copy(), metadata)

    baseline = run(False)
    measured = run(True)
    for i in range(5):
        np.testing.assert_array_equal(baseline[i], measured[i])
    assert 0 < measured[5]["effective_iterations"] <= 40
    assert measured[5]["stop_reason"] in ("early_stop", "max_iter")
    assert measured[5]["reached_max_iter"] == (measured[5]["effective_iterations"] == 40)


def test_return_info_rejects_nonboolean_before_fit():
    model = ParticleCompetitionAndCooperation(impl="numpy")
    with pytest.raises(TypeError, match="return_info"):
        model.fit_predict(np.array([0, 1]), return_info="yes")
