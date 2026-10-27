"""Diagnostic regression for finite greedy weights whose sum overflows.

This test is intentionally expected to expose a failure until the weighted
selector handles a nonfinite accumulated total without bias.
"""
import unittest
import numpy as np
from test_sequential_safety import call_step, pcc_step


class GreedyTotalOverflowTests(unittest.TestCase):
    def backends(self):
        return ['numpy', 'numba'] + (['cython'] if pcc_step is not None else [])

    def test_finite_equal_weights_with_overflowing_sum_are_uniform(self):
        count = 3000
        scale = 1.7e308
        weight = .4 * scale
        self.assertTrue(np.isfinite(weight))
        self.assertTrue(weight * 3 > np.finfo(np.float64).max)
        for backend in self.backends():
            with self.subTest(backend=backend):
                edges = np.array([[1, 2, 3], [0, 0, 0],
                                  [0, 0, 0], [0, 0, 0]], dtype=np.int64)
                degrees = np.array([3, 0, 0, 0], dtype=np.int64)
                labels = np.array([-1, 1, 1, 1], dtype=np.int64)
                positions = np.zeros(count, dtype=np.int64)
                distances = np.full((4, count), 255, dtype=np.uint8)
                distances[0] = 0
                dominance = np.full((4, 2), .5, dtype=np.float64)
                dominance[1:, 0] = .4
                dominance[1:, 1] = .6
                dist_weights = np.ones(257, dtype=np.float64)
                dist_weights[255] = scale
                args = [edges, degrees, labels, 1., .1, 2, np.zeros(2),
                        positions, np.zeros(count, dtype=np.int64),
                        np.ones(count), distances, dominance,
                        np.zeros((4, 2)), 1., 2.]
                np.random.seed(2026)
                call_step(backend, args, dist_weights=dist_weights)
                visited = distances[1:] == 1
                np.testing.assert_array_equal(visited.sum(axis=0),
                                              np.ones(count))
                np.testing.assert_allclose(visited.mean(axis=1),
                                           [1/3, 1/3, 1/3], rtol=0, atol=.06)


if __name__ == '__main__':
    unittest.main()
