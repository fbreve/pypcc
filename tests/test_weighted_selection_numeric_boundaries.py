"""Numerical boundaries of weighted neighbor selection across backends."""
import unittest
import numpy as np
from test_sequential_safety import call_step, pcc_step


class WeightedSelectionNumericBoundaryTests(unittest.TestCase):
    def backends(self):
        return ['numpy', 'numba'] + (['cython'] if pcc_step is not None else [])

    def sample(self, backend, weights, neighbors=(1, 2, 3), count=4000):
        # Labeled targets preserve dominance; the distance table identifies
        # which target each particle visited even if its movement was rejected.
        edges = np.zeros((4, 3), dtype=np.int64)
        edges[0] = neighbors
        degrees = np.array([3, 0, 0, 0], dtype=np.int64)
        labels = np.array([-1, 1, 1, 1], dtype=np.int64)
        positions = np.zeros(count, dtype=np.int64)
        strengths = np.ones(count, dtype=np.float64)
        distances = np.full((4, count), 255, dtype=np.uint8)
        distances[0, :] = 0
        dominance = np.full((4, 2), .5, dtype=np.float64)
        dominance[1:, 0] = [.2, .3, .4]
        dominance[1:, 1] = 1. - dominance[1:, 0]
        ownership = np.zeros((4, 2), dtype=np.float64)
        # All target distances are 255, so this value scales every valid
        # selection weight without changing their relative probabilities.
        dist_weights = np.ones(257, dtype=np.float64)
        dist_weights[255] = weights
        args = [edges, degrees, labels, 1., .1, 2, np.zeros(2),
                positions, np.zeros(count, dtype=np.int64), strengths,
                distances, dominance, ownership, 1., 2.]
        np.random.seed(2026)
        call_step(backend, args, dist_weights=dist_weights)
        return (distances[1:, :] == 1).mean(axis=1), distances

    def test_tiny_positive_weights_keep_expected_ratios(self):
        for backend in self.backends():
            for scale in (1e-100, 1e-250):
                with self.subTest(backend=backend, scale=scale):
                    observed, _ = self.sample(backend, scale)
                    np.testing.assert_allclose(observed, [2/9, 1/3, 4/9],
                                               rtol=0, atol=.055)

    def test_invalid_neighbors_have_zero_greedy_probability(self):
        # Invalid slots are zero-weight; positive valid weights prevent
        # the all-zero fallback from selecting an invalid slot.
        for backend in self.backends():
            for invalid in (-1, 4):
                with self.subTest(backend=backend, invalid=invalid):
                    observed, distances = self.sample(
                        backend, 1., neighbors=(invalid, 2, 3))
                    self.assertEqual(observed[0], 0.)
                    np.testing.assert_allclose(observed[1:], [3/7, 4/7],
                                               rtol=0, atol=.055)
                    np.testing.assert_array_equal(
                        (distances[1:, :] == 1).sum(axis=0),
                        np.ones(distances.shape[1]))


if __name__ == '__main__':
    unittest.main()
