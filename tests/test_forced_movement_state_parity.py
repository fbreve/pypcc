"""Cross-backend state parity when each movement has exactly one destination."""
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


class ForcedMovementStateParityTests(unittest.TestCase):
    def backends(self):
        result = [('numpy', pcc_propagate_numpy)]
        if pcc_propagate_numba is not None:
            result.append(('numba', pcc_propagate_numba))
        if pcc_propagate is not None:
            result.append(('cython', pcc_propagate))
        return result

    def run_case(self, fn, greedy, deltap, iterations, labeled):
        a = state()
        # Node 0 -> node 2 -> node 0: each reachable node has one neighbor.
        a[0][2, 0] = 0
        a[1][:] = 1
        a[2][2] = 1 if labeled else -1
        a[3] = greedy
        a[13] = deltap
        info = fn(*a, *buffers(a), iterations, False, 2000, 1,
                  return_info=True, seed=17)
        return [a[i].copy() for i in (7, 8, 9, 10, 11, 12)], info

    def test_forced_movement_state_matches_numpy(self):
        for greedy in (0., 1.):
            for deltap in (1., .4):
                for labeled in (False, True):
                    for iterations in (1, 2, 5):
                        expected, expected_info = self.run_case(
                            pcc_propagate_numpy, greedy, deltap, iterations, labeled)
                        for backend, fn in self.backends():
                            with self.subTest(backend=backend, greedy=greedy,
                                              deltap=deltap, labeled=labeled,
                                              iterations=iterations):
                                actual, info = self.run_case(
                                    fn, greedy, deltap, iterations, labeled)
                                for x, y in zip(actual, expected):
                                    np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)
                                self.assertEqual(info, expected_info)


if __name__ == '__main__':
    unittest.main()
