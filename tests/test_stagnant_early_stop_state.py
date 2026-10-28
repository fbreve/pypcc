"""Early stopping on stable dominance must match an equivalent fixed run."""
import unittest
import numpy as np
from pcc_numpy import pcc_propagate_numpy

try:
    from pcc_numba import pcc_propagate_numba
except ImportError:
    pcc_propagate_numba = None
try:
    from pcc_step import pcc_propagate
except ImportError:
    pcc_propagate = None


class StagnantEarlyStopStateTests(unittest.TestCase):
    def backends(self):
        result = [('numpy', pcc_propagate_numpy)]
        if pcc_propagate_numba is not None:
            result.append(('numba', pcc_propagate_numba))
        if pcc_propagate is not None:
            result.append(('cython', pcc_propagate))
        return result

    def run_case(self, fn, greedy, early_stop, max_iter, stop_max):
        # delta_v=0 keeps dominance exactly constant despite movement.
        # Multiple neighbors exercise stochastic selection and state changes.
        edges = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        degrees = np.full(3, 2, dtype=np.int64)
        labels = np.array([0, 1, -1], dtype=np.int64)
        dominance = np.array([[1., 0.], [0., 1.], [.4, .6]])
        positions = np.array([0, 1], dtype=np.int64)
        strengths = np.array([1., 1.])
        distances = np.full((3, 2), 255, dtype=np.uint8)
        distances[0, 0] = 0
        distances[1, 1] = 0
        ownership = np.zeros((3, 2), dtype=np.float64)
        args = [edges, degrees, labels, greedy, 0., 2, np.zeros(2),
                positions, np.array([0, 1], dtype=np.int64), strengths,
                distances, dominance, ownership, .4, 2.]
        scratch = [np.empty(2), np.empty(2)] + [np.empty(2) for _ in range(4)] + [1. / (np.arange(257) + 1.) ** 2]
        info = fn(*args, *scratch, max_iter, early_stop, 2000, stop_max,
                  return_info=True, seed=23)
        np.testing.assert_array_equal(dominance,
                                      [[1., 0.], [0., 1.], [.4, .6]])
        return info, [args[i].copy() for i in (7, 9, 10, 11, 12)]

    def test_stagnation_stop_matches_fixed_iteration_state(self):
        # First check at iteration 1 improves max_mmpot from zero.
        # Subsequent checks at 11, 21, 31... increment stop_cnt.
        for greedy in (0., .5, 1.):
            for stop_max in (0, 1, 2):
                expected_iterations = 11 + 10 * stop_max
                for backend, fn in self.backends():
                    with self.subTest(greedy=greedy, stop_max=stop_max,
                                      backend=backend):
                        info, state = self.run_case(fn, greedy, True, 100,
                                                    stop_max)
                        self.assertEqual(info['stop_reason'], 'early_stop')
                        self.assertEqual(info['effective_iterations'],
                                         expected_iterations)
                        self.assertEqual(info['stalled_checks'], stop_max + 1)
                        fixed_info, fixed_state = self.run_case(
                            fn, greedy, False, expected_iterations, stop_max)
                        self.assertEqual(fixed_info['effective_iterations'],
                                         expected_iterations)
                        for actual, expected in zip(state, fixed_state):
                            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
