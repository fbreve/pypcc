"""Regression tests for opt-in public stopping metadata (unittest compatible)."""
import unittest

import numpy as np
import pcc
from pcc import ParticleCompetitionAndCooperation


class ExperimentalPublicStopMetricsTests(unittest.TestCase):
    def test_return_info_preserves_predictions_and_state(self):
        for backend in ("numpy", "numba", "cython"):
            if backend == "numba" and not pcc._HAS_NUMBA:
                continue
            if backend == "cython" and not pcc._HAS_CYTHON:
                continue
            for criterion in ("potential", "dominance_change"):
                if criterion == "dominance_change" and backend != "numpy":
                    continue
                with self.subTest(backend=backend, criterion=criterion):
                    graph = np.array([[1, 2], [0, 2], [1, 3], [2, 1]], dtype=np.int64)
                    degrees = np.array([2, 2, 2, 2], dtype=np.int64)
                    labels = np.array([0, -1, -1, 1], dtype=np.int64)

                    def run(info):
                        model = ParticleCompetitionAndCooperation(impl=backend)
                        model.set_graph(graph, degrees, k_nn=2)
                        result = model.fit_predict(
                            labels, max_iter=40, early_stop=True, es_chk=20,
                            seed=123, stop_criterion=criterion, return_info=info,
                        )
                        prediction, metadata = result if info else (result, None)
                        return (
                            prediction, model.node.dominance.copy(),
                            model.part.curnode.copy(), model.part.strength.copy(),
                            model.part.dist_table.copy(), metadata,
                        )

                    baseline = run(False)
                    measured = run(True)
                    for index in range(5):
                        np.testing.assert_array_equal(baseline[index], measured[index])
                    self.assertGreater(measured[5]["effective_iterations"], 0)
                    self.assertLessEqual(measured[5]["effective_iterations"], 40)
                    self.assertIn(measured[5]["stop_reason"], ("early_stop", "max_iter"))
                    self.assertEqual(
                        measured[5]["reached_max_iter"],
                        measured[5]["effective_iterations"] == 40,
                    )

    def test_return_info_rejects_nonboolean_before_fit(self):
        model = ParticleCompetitionAndCooperation(impl="numpy")
        with self.assertRaisesRegex(TypeError, "return_info"):
            model.fit_predict(np.array([0, 1]), return_info="yes")


if __name__ == "__main__":
    unittest.main()
