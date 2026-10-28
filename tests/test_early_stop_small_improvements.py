"""Early-stop bookkeeping when dominance improves by very small amounts."""
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


class EarlyStopSmallImprovementsTests(unittest.TestCase):
    def backends(self):
        result = [('numpy', pcc_propagate_numpy)]
        if pcc_propagate_numba is not None:
            result.append(('numba', pcc_propagate_numba))
        if pcc_propagate is not None:
            result.append(('cython', pcc_propagate))
        return result

    def run_case(self, fn, delta_v, early_stop, iterations):
        # A single forced destination eliminates differences between RNG streams.
        edges = np.array([[1], [0]], dtype=np.int64)
        degrees = np.ones(2, dtype=np.int64)
        labels = np.array([0, -1], dtype=np.int64)
        dominance = np.array([[1., 0.], [.45, .55]], dtype=np.float64)
        positions = np.array([0], dtype=np.int64)
        strengths = np.array([1.], dtype=np.float64)
        distances = np.array([[0], [255]], dtype=np.uint8)
        ownership = np.zeros((2, 2), dtype=np.float64)
        args = [edges, degrees, labels, 0., delta_v, 2, np.zeros(2),
                positions, np.array([0], dtype=np.int64), strengths,
                distances, dominance, ownership, 1., 2.]
        scratch = [np.empty(2), np.empty(2)] + [np.empty(1) for _ in range(4)] + [1. / (np.arange(257) + 1.) ** 2]
        info = fn(*args, *scratch, iterations, early_stop, 2000, 1,
                  return_info=True, seed=19)
        return info, [args[i].copy() for i in (7, 9, 10, 11, 12)]

    def test_small_improvements_preserve_stop_and_state_contract(self):
        for delta_v in (0., 1e-12, 1e-8, 1e-4):
            for backend, fn in self.backends():
                with self.subTest(delta_v=delta_v, backend=backend):
                    info, stopped_state = self.run_case(fn, delta_v, True, 100)
                    self.assertGreater(info['effective_iterations'], 0)
                    self.assertLessEqual(info['effective_iterations'], 100)
                    self.assertEqual(info['checks'],
                                     (info['effective_iterations'] + 9) // 10)
                    fixed_info, fixed_state = self.run_case(
                        fn, delta_v, False, info['effective_iterations'])
                    self.assertEqual(fixed_info['effective_iterations'],
                                     info['effective_iterations'])
                    for actual, expected in zip(stopped_state, fixed_state):
                        np.testing.assert_array_equal(actual, expected)
                    if info['stop_reason'] == 'early_stop':
                        self.assertEqual(info['stalled_checks'], 2)
                        self.assertLess(info['effective_iterations'], 100)
                    else:
                        self.assertEqual(info['effective_iterations'], 100)

    def test_maximum_mean_potential_does_not_measure_full_state(self):
        # Different class assignments can have identical mean maximum dominance.
        a = np.array([[.9, .1], [.2, .8]])
        b = np.array([[.1, .9], [.8, .2]])
        self.assertFalse(np.array_equal(a, b))
        self.assertEqual(float(np.mean(np.max(a, axis=1))),
                         float(np.mean(np.max(b, axis=1))))


if __name__ == '__main__':
    unittest.main()
