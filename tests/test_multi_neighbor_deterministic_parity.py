"""Multiple neighbor slots with deterministic destination, across backends.

Repeated neighbor indices exercise the degree>1 random and weighted-selection
paths without depending on distinct RNG streams. This is not a test of the
probability distribution over distinct neighbors.
"""
import unittest
import numpy as np
from test_sequential_safety import state, buffers
from pcc_numpy import pcc_propagate_numpy

try:
    from pcc_numba import pcc_propagate_numba
except ImportError:
    pcc_propagate_numba = None
try:
    from pcc_step import pcc_propagate
except ImportError:
    pcc_propagate = None


class MultiNeighborDeterministicParityTests(unittest.TestCase):
    def backends(self):
        result = [('numpy', pcc_propagate_numpy)]
        if pcc_propagate_numba is not None:
            result.append(('numba', pcc_propagate_numba))
        if pcc_propagate is not None:
            result.append(('cython', pcc_propagate))
        return result

    def run_case(self, fn, greedy, iterations, labeled, tie):
        a = state(degree=2)
        # Two slots at every node, but each pair names the same target.
        # Node 0 -> 2 -> 0; no RNG-dependent choice of destination.
        a[0][:] = 0
        a[0][0, :] = 2
        a[1][:] = 2
        a[2][2] = 1 if labeled else -1
        a[3] = greedy
        if tie:
            a[4] = 0.0  # A tie at node 2 stays a tie after visiting it.
            a[11][2] = [.5, .5]
        info = fn(*a, *buffers(a), iterations, False, 2000, 1,
                  return_info=True, seed=19)
        return [a[i].copy() for i in (7, 8, 9, 10, 11, 12)], info

    def test_repeated_neighbor_slots_match_across_backends(self):
        for greedy in (0., 1.):
            for iterations in (1, 2, 5):
                for labeled in (False, True):
                    for tie in (False, True):
                        expected, metrics = self.run_case(
                            pcc_propagate_numpy, greedy, iterations, labeled, tie)
                        for backend, fn in self.backends():
                            with self.subTest(backend=backend, greedy=greedy,
                                              iterations=iterations, labeled=labeled,
                                              tie=tie):
                                actual, info = self.run_case(
                                    fn, greedy, iterations, labeled, tie)
                                for x, y in zip(actual, expected):
                                    np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)
                                self.assertEqual(info, metrics)


if __name__ == '__main__':
    unittest.main()
