"""Subnormal weights and underflow-driven greedy fallback across backends."""
import unittest
import numpy as np
from test_sequential_safety import call_step, pcc_step


class SubnormalGreedyFallbackTests(unittest.TestCase):
    def backends(self):
        return ['numpy', 'numba'] + (['cython'] if pcc_step is not None else [])

    def sample(self, backend, scale, count=4000):
        edges = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         dtype=np.int64)
        degrees = np.array([3, 0, 0, 0], dtype=np.int64)
        labels = np.array([-1, 1, 1, 1], dtype=np.int64)
        positions = np.zeros(count, dtype=np.int64)
        distances = np.full((4, count), 255, dtype=np.uint8)
        distances[0] = 0
        dominance = np.full((4, 2), .5, dtype=np.float64)
        dominance[1:, 0] = [.2, .3, .4]
        dominance[1:, 1] = 1. - dominance[1:, 0]
        dist_weights = np.ones(257, dtype=np.float64)
        dist_weights[255] = scale
        args = [edges, degrees, labels, 1., .1, 2, np.zeros(2),
                positions, np.zeros(count, dtype=np.int64), np.ones(count),
                distances, dominance, np.zeros((4, 2)), 1., 2.]
        np.random.seed(2026)
        call_step(backend, args, dist_weights=dist_weights)
        visited = distances[1:] == 1
        np.testing.assert_array_equal(visited.sum(axis=0), np.ones(count))
        np.testing.assert_array_equal(positions, np.zeros(count, dtype=np.int64))
        return visited.mean(axis=1)

    def test_subnormal_positive_total_preserves_ratios(self):
        # All three products are positive subnormals at this scale.
        scale = 1e-320
        self.assertTrue(np.all(np.array([.2, .3, .4]) * scale > 0))
        for backend in self.backends():
            with self.subTest(backend=backend):
                np.testing.assert_allclose(self.sample(backend, scale),
                                           [2/9, 1/3, 4/9], rtol=0, atol=.06)

    def test_complete_underflow_falls_back_to_uniform(self):
        # Smallest subnormal times each dominance value rounds to zero.
        scale = np.nextafter(0., 1.)
        np.testing.assert_array_equal(np.array([.2, .3, .4]) * scale,
                                      np.zeros(3))
        for backend in self.backends():
            with self.subTest(backend=backend):
                np.testing.assert_allclose(self.sample(backend, scale),
                                           [1/3, 1/3, 1/3], rtol=0, atol=.06)


if __name__ == '__main__':
    unittest.main()
