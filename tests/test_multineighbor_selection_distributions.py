"""Statistical selection of three distinct neighbors, including zero weights."""
import unittest
import numpy as np
from test_sequential_safety import call_step, pcc_step


class MultineighborSelectionDistributionTests(unittest.TestCase):
    def backends(self):
        return ['numpy', 'numba'] + (['cython'] if pcc_step is not None else [])

    def sample(self, backend, greedy, class_zero_dominance, count=4000):
        # Targets are labeled class 1, so their dominance stays unchanged.
        # A visited target is recorded by its distance update for that particle.
        edges = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         dtype=np.int64)
        degrees = np.array([3, 0, 0, 0], dtype=np.int64)
        labels = np.array([-1, 1, 1, 1], dtype=np.int64)
        positions = np.zeros(count, dtype=np.int64)
        classes = np.zeros(count, dtype=np.int64)
        strengths = np.ones(count, dtype=np.float64)
        distances = np.full((4, count), 255, dtype=np.uint8)
        distances[0, :] = 0
        dominance = np.full((4, 2), .5, dtype=np.float64)
        dominance[1:, 0] = class_zero_dominance
        dominance[1:, 1] = 1. - np.asarray(class_zero_dominance)
        ownership = np.zeros((4, 2), dtype=np.float64)
        args = [edges, degrees, labels, greedy, .1, 2, np.zeros(2),
                positions, classes, strengths, distances, dominance,
                ownership, 1., 2.]
        np.random.seed(2026)
        call_step(backend, args)
        visited = distances[1:, :] == 1
        np.testing.assert_array_equal(visited.sum(axis=0), np.ones(count))
        np.testing.assert_array_equal(positions, np.zeros(count, dtype=np.int64))
        return visited.mean(axis=1)

    def test_three_neighbor_distributions(self):
        cases = (
            ('uniform', 0., (.1, .2, .3), (1/3, 1/3, 1/3)),
            ('weighted', 1., (.1, .2, .3), (1/6, 1/3, 1/2)),
            ('zero_weight', 1., (0., .2, .4), (0., 1/3, 2/3)),
            ('zero_total_fallback', 1., (0., 0., 0.), (1/3, 1/3, 1/3)),
        )
        for backend in self.backends():
            for name, greedy, weights, expected in cases:
                with self.subTest(backend=backend, case=name):
                    observed = self.sample(backend, greedy, weights)
                    np.testing.assert_allclose(observed, expected, rtol=0, atol=.055)


if __name__ == '__main__':
    unittest.main()
