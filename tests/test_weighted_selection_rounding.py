"""Selection distributions around floating-point cumulative-sum boundaries."""
import unittest
import numpy as np
from test_sequential_safety import call_step, pcc_step


class WeightedSelectionRoundingTests(unittest.TestCase):
    def backends(self):
        return ['numpy', 'numba'] + (['cython'] if pcc_step is not None else [])

    def sample(self, backend, weights, count=4000):
        edges = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         dtype=np.int64)
        degrees = np.array([3, 0, 0, 0], dtype=np.int64)
        labels = np.array([-1, 1, 1, 1], dtype=np.int64)
        positions = np.zeros(count, dtype=np.int64)
        distances = np.full((4, count), 255, dtype=np.uint8)
        distances[0] = 0
        dominance = np.full((4, 2), .5, dtype=np.float64)
        dominance[1:, 0] = weights
        dominance[1:, 1] = 1. - np.asarray(weights)
        args = [edges, degrees, labels, 1., .1, 2, np.zeros(2),
                positions, np.zeros(count, dtype=np.int64),
                np.ones(count), distances, dominance,
                np.zeros((4, 2)), 1., 2.]
        np.random.seed(2026)
        call_step(backend, args)
        visited = distances[1:] == 1
        np.testing.assert_array_equal(visited.sum(axis=0), np.ones(count))
        return visited.mean(axis=1)

    def test_rounding_sensitive_weight_ratios(self):
        # Adjacent representable float64 values near 0.25 and 0.5.
        next_quarter = np.nextafter(.25, 1.)
        prev_half = np.nextafter(.5, 0.)
        cases = (
            ('adjacent_floats', (.25, next_quarter, prev_half)),
            ('small_first', (1e-12, .25, .5)),
            ('small_middle', (.25, 1e-12, .5)),
            ('small_last', (.25, .5, 1e-12)),
        )
        for backend in self.backends():
            for name, weights in cases:
                with self.subTest(backend=backend, case=name):
                    expected = np.asarray(weights) / sum(weights)
                    observed = self.sample(backend, weights)
                    np.testing.assert_allclose(observed, expected,
                                               rtol=0, atol=.055)


if __name__ == '__main__':
    unittest.main()
