"""Regression for weighted selection immediately below and above float64 sum overflow."""
import unittest
import numpy as np
from test_sequential_safety import call_step, pcc_step


class GreedyNearOverflowTests(unittest.TestCase):
    def backends(self):
        return ['numpy', 'numba'] + (['cython'] if pcc_step is not None else [])

    def sample(self, backend, weights, count=4000):
        edges = np.array([[1, 2, 3], [0, 0, 0],
                          [0, 0, 0], [0, 0, 0]], dtype=np.int64)
        degrees = np.array([3, 0, 0, 0], dtype=np.int64)
        labels = np.array([-1, 1, 1, 1], dtype=np.int64)
        positions = np.zeros(count, dtype=np.int64)
        distances = np.full((4, count), 255, dtype=np.uint8)
        distances[0] = 0
        dominance = np.full((4, 2), .5, dtype=np.float64)
        dominance[1:, 0] = weights
        dominance[1:, 1] = 1. - dominance[1:, 0]
        dist_weights = np.ones(257, dtype=np.float64)
        dist_weights[255] = 1.7e308
        args = [edges, degrees, labels, 1., .1, 2, np.zeros(2),
                positions, np.zeros(count, dtype=np.int64),
                np.ones(count), distances, dominance,
                np.zeros((4, 2)), 1., 2.]
        np.random.seed(2026)
        call_step(backend, args, dist_weights=dist_weights)
        visited = distances[1:] == 1
        np.testing.assert_array_equal(visited.sum(axis=0), np.ones(count))
        np.testing.assert_array_equal(positions, np.zeros(count, dtype=np.int64))
        return visited.mean(axis=1)

    def test_equal_weights_straddling_overflow(self):
        # 3 * .35 * 1.7e308 remains finite; 3 * .36 * 1.7e308 overflows.
        for fraction, overflowing in ((.35, False), (.36, True)):
            products = np.full(3, fraction) * 1.7e308
            self.assertEqual(bool(np.isinf(np.sum(products))), overflowing)
            for backend in self.backends():
                with self.subTest(fraction=fraction, backend=backend):
                    np.testing.assert_allclose(self.sample(backend, [fraction] * 3),
                                               [1/3, 1/3, 1/3], rtol=0, atol=.06)

    def test_unequal_weights_straddling_overflow(self):
        # Both sets have a 1:2:3 ratio, on opposite sides of the boundary.
        for factor, overflowing in ((.17, False), (.19, True)):
            weights = np.array([factor, 2 * factor, 3 * factor])
            products = weights * 1.7e308
            self.assertEqual(bool(np.isinf(np.sum(products))), overflowing)
            for backend in self.backends():
                with self.subTest(factor=factor, backend=backend):
                    np.testing.assert_allclose(self.sample(backend, weights),
                                               [1/6, 1/3, 1/2], rtol=0, atol=.06)


if __name__ == '__main__':
    unittest.main()
