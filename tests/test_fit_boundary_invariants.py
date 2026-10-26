"""Boundary invariants for fit initialization and stopping parameters.

These tests document current supported behavior without changing the kernels.
"""
import unittest

import numpy as np

from pcc import ParticleCompetitionAndCooperation


class FitBoundaryInvariants(unittest.TestCase):
    def setUp(self):
        self.neighbors = np.array([[1], [0], [3], [2]], dtype=np.int64)
        self.degrees = np.ones(4, dtype=np.int64)
        self.labels = np.array([0, -1, 1, -1], dtype=np.int64)

    def make_model(self, backend):
        model = ParticleCompetitionAndCooperation(impl=backend)
        model.set_graph(self.neighbors, self.degrees, k_nn=1)
        return model

    def test_zero_iterations_preserves_initial_particle_state(self):
        for backend in ("numpy", "numba"):
            with self.subTest(backend=backend):
                model = self.make_model(backend)
                result = model.fit_predict(
                    self.labels, max_iter=0, early_stop=False
                )
                np.testing.assert_array_equal(result[[0, 2]], [0, 1])
                np.testing.assert_array_equal(model.part.curnode, [0, 2])
                np.testing.assert_array_equal(model.part.strength, [1.0, 1.0])
                np.testing.assert_array_equal(
                    model.node.dominance[[0, 2]], [[1.0, 0.0], [0.0, 1.0]]
                )

    def test_single_iteration_with_early_stop(self):
        for backend in ("numpy", "numba"):
            with self.subTest(backend=backend):
                model = self.make_model(backend)
                result = model.fit_predict(
                    self.labels, max_iter=1, early_stop=True, es_chk=0
                )
                self.assertEqual(result.shape, self.labels.shape)
                np.testing.assert_array_equal(result[[0, 2]], [0, 1])


if __name__ == "__main__":
    unittest.main()
