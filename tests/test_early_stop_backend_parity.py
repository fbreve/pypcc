"""Lock down early-stop timing and metrics across propagation backends.

The stopping check runs at iteration indices 0, 10, 20, ... regardless
of es_chk; stop_max is already computed by the public API.
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


class EarlyStopBackendParityTests(unittest.TestCase):
    def backends(self):
        result = [('numpy', pcc_propagate_numpy)]
        if pcc_propagate_numba is not None:
            result.append(('numba', pcc_propagate_numba))
        if pcc_propagate is not None:
            result.append(('cython', pcc_propagate))
        return result

    def run_plateau(self, fn, cap, early, es_chk, stop_max):
        a = state()
        # No valid movement: every backend sees a constant dominance plateau.
        a[0][:] = 0
        a[2][0] = 0
        a[11][0] = [1., 0.]
        a[3] = 0.
        info = fn(*a, *buffers(a), cap, early, es_chk, stop_max,
                  return_info=True)
        return info, a[12][0, 0]

    def test_plateau_stop_schedule_and_metrics_match(self):
        cases = [
            (0, True, 2000, 1, 0, 0, 'max_iter'),
            (1, True, 2000, 1, 1, 1, 'max_iter'),
            (10, True, 2000, 1, 10, 1, 'max_iter'),
            (11, True, 2000, 1, 11, 2, 'max_iter'),
            (20, True, 2000, 1, 20, 2, 'max_iter'),
            (21, True, 2000, 1, 21, 3, 'early_stop'),
            (100, True, 2000, 1, 21, 3, 'early_stop'),
            (100, True, 1, 0, 11, 2, 'early_stop'),
            (30, False, 2000, 1, 30, 0, 'max_iter'),
        ]
        for backend, fn in self.backends():
            for cap, early, es_chk, stop_max, iterations, checks, reason in cases:
                with self.subTest(backend=backend, cap=cap, early=early,
                                  es_chk=es_chk, stop_max=stop_max):
                    info, counted = self.run_plateau(
                        fn, cap, early, es_chk, stop_max)
                    self.assertEqual(info['effective_iterations'], iterations)
                    self.assertEqual(info['checks'], checks)
                    self.assertEqual(info['stop_reason'], reason)
                    self.assertEqual(info['reached_max_iter'], iterations == cap)
                    self.assertEqual(info['stop_max'], stop_max)
                    self.assertEqual(counted, iterations)

    def test_es_chk_is_not_direct_check_interval(self):
        for backend, fn in self.backends():
            with self.subTest(backend=backend):
                baseline, _ = self.run_plateau(fn, 100, True, 2000, 1)
                changed, _ = self.run_plateau(fn, 100, True, 1, 1)
                self.assertEqual(changed, baseline)


if __name__ == '__main__':
    unittest.main()
