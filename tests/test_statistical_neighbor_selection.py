"""Statistical selection tests with distinct neighbors and backend-specific RNGs.

Use one step with many independent particles: Cython's private RNG is seeded
per call, so repeatedly invoking a single-particle step would not sample it.
"""
import unittest
import numpy as np
from test_sequential_safety import call_step, pcc_step


class StatisticalNeighborSelectionTests(unittest.TestCase):
    def backends(self):
        return ['numpy', 'numba'] + (['cython'] if pcc_step is not None else [])

    def sample(self, backend, greedy, count=3000):
        # Each particle starts at node 0, with two distinct labeled targets.
        # Both targets reject class-0 particles (class 1 has greater dominance),
        # but the resulting particle strength records which target was visited.
        edges = np.array([[1, 2], [0, 0], [0, 0]], dtype=np.int64)
        degrees = np.array([2, 0, 0], dtype=np.int64)
        labels = np.array([-1, 1, 1], dtype=np.int64)
        positions = np.zeros(count, dtype=np.int64)
        classes = np.zeros(count, dtype=np.int64)
        strengths = np.ones(count, dtype=np.float64)
        distances = np.full((3, count), 255, dtype=np.uint8)
        distances[0] = 0
        dominance = np.array([[.5, .5], [.2, .8], [.4, .6]], dtype=np.float64)
        ownership = np.zeros((3, 2), dtype=np.float64)
        args = [edges, degrees, labels, greedy, .1, 2,
                np.zeros(2), positions, classes, strengths, distances,
                dominance, ownership, 1., 2.]
        np.random.seed(2026)
        call_step(backend, args)
        np.testing.assert_array_equal(positions, np.zeros(count, dtype=np.int64))
        np.testing.assert_array_equal(dominance,
                                      [[.5, .5], [.2, .8], [.4, .6]])
        return np.count_nonzero(np.isclose(strengths, .4)) / count

    def test_distinct_neighbor_selection_distribution(self):
        # Uniform: 1/2 each. Greedy weights: .2 and .4, hence P(node 2)=2/3.
        # A fixed 0.055 absolute tolerance is > 5 standard errors at n=3000.
        for backend in self.backends():
            for greedy, expected in ((0., .5), (1., 2. / 3.)):
                with self.subTest(backend=backend, greedy=greedy):
                    observed = self.sample(backend, greedy)
                    self.assertAlmostEqual(observed, expected, delta=.055)


if __name__ == '__main__':
    unittest.main()
