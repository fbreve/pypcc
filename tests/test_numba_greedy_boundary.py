"""Deterministic boundary semantics for the Numba cumulative selector.

This small njit probe exercises the same strict comparison used by the
production kernel without relying on inaccessible compiled RNG state.
"""
import unittest
import numpy as np
from numba import njit


@njit
def select_interval(slices, threshold):
    # Mirrors the production Numba weighted cumulative comparison.
    choice = 0
    for i in range(len(slices)):
        if threshold < slices[i]:
            choice = i
            break
    return choice


class NumbaGreedyBoundaryTests(unittest.TestCase):
    def test_exact_threshold_boundaries(self):
        cases = (
            ((0., .25, .75), 0., 1),
            ((.25, .5, 1.), .25, 1),
            ((.25, .25, 1.), .25, 2),
            ((.25, .25, 1.), 0., 0),
        )
        for cumulative, threshold, expected in cases:
            with self.subTest(cumulative=cumulative, threshold=threshold):
                self.assertEqual(select_interval(np.array(cumulative), threshold),
                                 expected)


if __name__ == '__main__':
    unittest.main()
