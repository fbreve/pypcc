"""Characterize early stopping while class dominance is still changing."""
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


class EarlyStopOngoingDominanceTests(unittest.TestCase):
    def backends(self):
        result = [('numpy', pcc_propagate_numpy)]
        if pcc_propagate_numba is not None:
            result.append(('numba', pcc_propagate_numba))
        if pcc_propagate is not None:
            result.append(('cython', pcc_propagate))
        return result

    def run_case(self, fn, early_stop, iterations):
        # Both particles target the same unlabeled node each iteration.
        # Class 0 acts first, class 1 second; both can change its dominance.
        # Their starting nodes have no incoming edges, so positions remain fixed.
        edges = np.array([[2], [2], [-1]], dtype=np.int64)
        degrees = np.array([1, 1, 0], dtype=np.int64)
        labels = np.array([0, 1, -1], dtype=np.int64)
        dominance = np.array([[1., 0.], [0., 1.], [.5, .5]], dtype=np.float64)
        positions = np.array([0, 1], dtype=np.int64)
        strengths = np.array([1., 1.], dtype=np.float64)
        distances = np.full((3, 2), 255, dtype=np.uint8)
        distances[0, 0] = 0
        distances[1, 1] = 0
        ownership = np.zeros((3, 2), dtype=np.float64)
        args = [edges, degrees, labels, 0., .2, 2, np.zeros(2),
                positions, np.array([0, 1], dtype=np.int64), strengths,
                distances, dominance, ownership, 1., 2.]
        scratch = [np.empty(2), np.empty(2)] + [np.empty(1) for _ in range(4)] + [1. / (np.arange(257) + 1.) ** 2]
        info = fn(*args, *scratch, iterations, early_stop, 2000, 0,
                  return_info=True, seed=31)
        return info, dominance.copy()

    def test_early_stop_does_not_imply_frozen_dominance(self):
        for backend, fn in self.backends():
            with self.subTest(backend=backend):
                info, stopped = self.run_case(fn, True, 100)
                self.assertEqual(info['stop_reason'], 'early_stop')
                self.assertEqual(info['effective_iterations'], 11)
                _, next_step = self.run_case(fn, False, 12)
                self.assertGreater(np.max(np.abs(next_step - stopped)), 1e-10)
                np.testing.assert_allclose(stopped.sum(axis=1), 1., atol=1e-12)


if __name__ == '__main__':
    unittest.main()
