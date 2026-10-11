"""Optional metrics must observe the existing loop without changing its state/RNG."""
import copy
import unittest
from functools import partial

import numpy as np
from numba import njit

import pcc_numba
from pcc_numpy import pcc_propagate_numpy
from test_sequential_safety import state, buffers, next_random, pcc_step


@njit
def seed_only(seed):
    np.random.seed(seed)


class PropagationMetricsTests(unittest.TestCase):
    def backends(self):
        result = {'numpy': pcc_propagate_numpy,
                  'numpy_synchronous': partial(pcc_propagate_numpy, update_mode='synchronous'),
                  'numba': pcc_numba.pcc_propagate_numba}
        if pcc_step is not None:
            result['cython'] = pcc_step.pcc_propagate
        return result

    def test_counts_iterations_not_checks_and_preserves_default_state_rng(self):
        cases = [(100, True, 1, 21, 3, 'early_stop'),
                 (12, True, 1, 12, 2, 'max_iter'),
                 (30, False, 1, 30, 0, 'max_iter'),
                 (0, True, 1, 0, 0, 'max_iter'),
                 (1, True, 1, 1, 1, 'max_iter'),
                 (21, True, 1, 21, 3, 'early_stop'),
                 (100, True, 2, 31, 4, 'early_stop'),
                 (10000, True, 1198, 10000, 1000, 'max_iter'),
                 (30000, True, 1198, 11991, 1200, 'early_stop')]
        for backend, fn in self.backends().items():
            for cap, early, threshold, expected, checks, reason in cases:
                with self.subTest(backend=backend, cap=cap, early=early, threshold=threshold):
                    initial = state()
                    initial[0][:] = 0
                    initial[2][0] = 0
                    initial[11][0] = [1., 0.]
                    initial[3] = 0.
                    states, rng_outputs = [], []
                    for info in (None, False, True):
                        args = copy.deepcopy(initial)
                        np.random.seed(42)
                        seed_only(42)
                        returned = fn(*args, *buffers(args), cap, early, 2000, threshold,
                                      **({} if info is None else {'return_info': info}))
                        states.append(args)
                        rng_outputs.append((np.random.random(), next_random()))
                        self.assertEqual(args[12][0, 0], expected)
                        if info:
                            self.assertEqual(returned['effective_iterations'], expected)
                            self.assertEqual(returned['checks'], checks)
                            self.assertEqual(returned['stop_reason'], reason)
                            self.assertEqual(returned['stop_max'], threshold)
                            self.assertEqual(returned['reached_max_iter'], expected == cap)
                            self.assertEqual(returned['stalled_checks'], max(0, checks - 1))
                        else:
                            self.assertIsNone(returned)
                    for index in (7, 9, 10, 11, 12):
                        for observed in states[1:]:
                            np.testing.assert_array_equal(states[0][index], observed[index])
                    for observed in rng_outputs[1:]:
                        self.assertEqual(rng_outputs[0], observed)

    def test_metrics_preserve_nonconstant_dominance_and_rng(self):
        for backend, fn in self.backends().items():
            with self.subTest(backend=backend):
                states, rng_outputs = [], []
                for info in (False, True):
                    args = state()
                    args[3] = .5
                    np.random.seed(19)
                    seed_only(19)
                    fn(*args, *buffers(args), 50, True, 2000, 1, return_info=info)
                    states.append(args)
                    rng_outputs.append((np.random.random(), next_random()))
                for index in (7, 9, 10, 11, 12):
                    np.testing.assert_array_equal(states[0][index], states[1][index])
                self.assertEqual(rng_outputs[0], rng_outputs[1])


if __name__ == '__main__':
    unittest.main()
