"""Dominance conservation under competing particles on shared destinations."""
import unittest
import numpy as np
from pcc_numpy import pcc_propagate_numpy

try:
    from pcc_numba import pcc_propagate_numba
except ImportError:
    pcc_propagate_numba = None
try:
    from pcc_step import pcc_propagate
except ImportError:
    pcc_propagate = None


class ContestedGraphDominanceTests(unittest.TestCase):
    def backends(self):
        result = [('numpy', pcc_propagate_numpy)]
        if pcc_propagate_numba is not None:
            result.append(('numba', pcc_propagate_numba))
        if pcc_propagate is not None:
            result.append(('cython', pcc_propagate))
        return result

    def run_case(self, fn, classes, delta_v, deltap, iterations):
        # Each particle starts at its own leaf; all leaves target the hub.
        # The hub targets leaf zero, making later movement nontrivial.
        n = classes + 1
        edges = np.full((n, classes), -1, dtype=np.int64)
        edges[0] = np.arange(1, n)
        edges[1:, 0] = 0
        degrees = np.array([classes] + [1] * classes, dtype=np.int64)
        labels = np.full(n, -1, dtype=np.int64)
        dominance = np.full((n, classes), 1. / classes, dtype=np.float64)
        dominance[0] = np.arange(1, classes + 1, dtype=np.float64)
        dominance[0] /= dominance[0].sum()
        initial_sums = dominance.sum(axis=1).copy()
        positions = np.arange(1, n, dtype=np.int64)
        particle_labels = np.arange(classes, dtype=np.int64)
        strengths = np.linspace(.25, .95, classes)
        distances = np.full((n, classes), 255, dtype=np.uint8)
        distances[positions, np.arange(classes)] = 0
        ownership = np.zeros((n, classes), dtype=np.float64)
        args = [edges, degrees, labels, 0., delta_v, classes,
                np.zeros(classes), positions, particle_labels, strengths,
                distances, dominance, ownership, deltap, 2.]
        scratch = [np.empty(classes), np.empty(classes)] + [np.empty(classes) for _ in range(4)] + [1. / (np.arange(257) + 1.) ** 2]
        fn(*args, *scratch, iterations, False, 2000, 1, seed=17)
        np.testing.assert_allclose(dominance.sum(axis=1), initial_sums,
                                   rtol=0, atol=5e-12)
        self.assertTrue(np.isfinite(dominance).all())
        self.assertGreaterEqual(dominance.min(), -1e-13)
        self.assertLessEqual(dominance.max(), 1. + 1e-13)
        self.assertTrue(np.isfinite(strengths).all())
        return dominance, strengths, positions, distances, ownership

    def test_contested_hub_conserves_dominance(self):
        for classes in (2, 3, 5):
            for delta_v in (.05, .5, 1.):
                for deltap in (.2, 1.):
                    for iterations in (50, 500):
                        with self.subTest(classes=classes, delta_v=delta_v,
                                          deltap=deltap, iterations=iterations):
                            for backend, fn in self.backends():
                                with self.subTest(backend=backend):
                                    self.run_case(fn, classes, delta_v,
                                                  deltap, iterations)


if __name__ == '__main__':
    unittest.main()
