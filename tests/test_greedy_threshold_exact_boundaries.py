"""Deterministic regression for exact greedy cumulative-weight boundaries.

The NumPy RNG is patched only in these tests; compiled backends have separate
RNG implementations and are not claimed to reproduce this controlled draw.
"""
import unittest
from unittest.mock import patch
import numpy as np
from test_sequential_safety import call_step


class ExactGreedyThresholdBoundaryTests(unittest.TestCase):
    def sample(self, draw, weights):
        edges = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         dtype=np.int64)
        degrees = np.array([3, 0, 0, 0], dtype=np.int64)
        labels = np.array([-1, 1, 1, 1], dtype=np.int64)
        positions = np.array([0], dtype=np.int64)
        distances = np.full((4, 1), 255, dtype=np.uint8)
        distances[0, 0] = 0
        dominance = np.full((4, 2), .5, dtype=np.float64)
        dominance[1:, 0] = weights
        dominance[1:, 1] = 1. - np.asarray(weights)
        args = [edges, degrees, labels, 1., .1, 2, np.zeros(2),
                positions, np.array([0], dtype=np.int64), np.ones(1),
                distances, dominance, np.zeros((4, 2)), 1., 2.]
        with patch('numpy.random.random', side_effect=[0., draw]) as random:
            call_step('numpy', args)
        self.assertEqual(random.call_count, 2)
        visited = np.flatnonzero(distances[1:, 0] == 1)
        self.assertEqual(visited.size, 1)
        return int(visited[0])

    def test_zero_threshold_skips_zero_weight_first_neighbor(self):
        # A zero draw should never choose a zero-probability target.
        self.assertEqual(self.sample(0., (0., .25, .5)), 1)

    def test_exact_cumulative_boundary_goes_to_next_positive_interval(self):
        # With weights .25, .25, .5, the draw .25 lies on a boundary.
        # Continuous weighted sampling assigns this boundary to the next bin.
        self.assertEqual(self.sample(.25, (.25, .25, .5)), 1)

    def test_zero_weight_middle_interval_is_not_selected(self):
        # At threshold .25, the middle neighbor has no probability mass.
        self.assertEqual(self.sample(.25, (.25, 0., .75)), 2)


if __name__ == '__main__':
    unittest.main()
