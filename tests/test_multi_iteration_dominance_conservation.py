"""Dominance conservation and cross-backend state parity over many forced moves."""
import unittest
import numpy as np
from test_sequential_safety import buffers
from pcc_numpy import pcc_propagate_numpy

try:
    from pcc_numba import pcc_propagate_numba
except ImportError:
    pcc_propagate_numba = None
try:
    from pcc_step import pcc_propagate
except ImportError:
    pcc_propagate = None


class MultiIterationDominanceConservationTests(unittest.TestCase):
    def backends(self):
        result = [('numpy', pcc_propagate_numpy)]
        if pcc_propagate_numba is not None:
            result.append(('numba', pcc_propagate_numba))
        if pcc_propagate is not None:
            result.append(('cython', pcc_propagate))
        return result

    def run_case(self, fn, classes, delta_v, deltap, iterations):
        # Deterministic two-node cycle: no backend-specific RNG trajectory.
        edges = np.array([[1], [0]], dtype=np.int64)
        degrees = np.ones(2, dtype=np.int64)
        labels = np.full(2, -1, dtype=np.int64)
        dominance = np.full((2, classes), 1. / classes, dtype=np.float64)
        dominance[1] = np.arange(1, classes + 1, dtype=np.float64)
        dominance[1] /= dominance[1].sum()
        initial_sums = dominance.sum(axis=1).copy()
        positions = np.array([0], dtype=np.int64)
        strengths = np.array([.85], dtype=np.float64)
        distances = np.full((2, 1), 255, dtype=np.uint8)
        distances[0, 0] = 0
        ownership = np.zeros((2, classes), dtype=np.float64)
        a = [edges, degrees, labels, 0., delta_v, classes, np.zeros(classes),
             positions, np.array([0], dtype=np.int64), strengths, distances,
             dominance, ownership, deltap, 2.]
        fn(*a, *buffers(a), iterations, False, 2000, 1, seed=17)
        np.testing.assert_allclose(dominance.sum(axis=1), initial_sums,
                                   rtol=0, atol=2e-12)
        self.assertTrue(np.isfinite(dominance).all())
        self.assertGreaterEqual(dominance.min(), -1e-14)
        self.assertLessEqual(dominance.max(), 1. + 1e-14)
        self.assertTrue(np.isfinite(strengths).all())
        return [a[i].copy() for i in (7, 8, 9, 10, 11, 12)]

    def test_many_iterations_conserve_dominance_and_match_backends(self):
        for classes in (2, 3, 5):
            for delta_v in (.01, .5, 1.):
                for deltap in (.2, 1.):
                    for iterations in (50, 500):
                        with self.subTest(classes=classes, delta_v=delta_v,
                                          deltap=deltap, iterations=iterations):
                            expected = self.run_case(pcc_propagate_numpy, classes,
                                                     delta_v, deltap, iterations)
                            for backend, fn in self.backends():
                                actual = self.run_case(fn, classes, delta_v,
                                                       deltap, iterations)
                                for x, y in zip(actual, expected):
                                    np.testing.assert_allclose(x, y, rtol=1e-11,
                                                               atol=2e-12,
                                                               err_msg=backend)


if __name__ == '__main__':
    unittest.main()
