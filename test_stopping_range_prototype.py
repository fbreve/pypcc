"""Small deterministic prototype checks; no long diagnostics in CI."""
import copy
import unittest
import numpy as np
import pcc_numba
from experiments.stopping_range import propagate_range, RangeObserver
from test_sequential_safety import state, buffers, next_random
from test_propagation_metrics import seed_only


class RangePrototypeTests(unittest.TestCase):
    def arguments(self, cap, threshold):
        a = state()
        a[0][:] = 0
        a[2][0] = 0
        a[11][0] = [1., 0.]
        a[3] = 0.
        return a+buffers(a)+[cap, True, 2000, threshold]

    def test_original_stop_and_capped_states_rng(self):
        for cap, threshold in ((100, 1), (101, 1198), (0, 1198)):
            a = self.arguments(cap, threshold)
            b = copy.deepcopy(a)
            seed_only(42)
            original = pcc_numba.pcc_propagate_numba(*a, return_info=True)
            rng = next_random()
            seed_only(42)
            candidate = propagate_range(*b)
            self.assertEqual(rng, next_random())
            for i in (7, 9, 10, 11, 12):
                np.testing.assert_array_equal(a[i], b[i])
            for key, value in original.items():
                self.assertEqual(value, candidate[key])

    def test_alternative_and_original_priority(self):
        for threshold, reason in ((1198, 'alternative_range'), (499, 'early_stop')):
            a = self.arguments(30000, threshold)
            b = copy.deepcopy(a)
            seed_only(71)
            info = propagate_range(*a)
            rng = next_random()
            self.assertEqual(info['effective_iterations'], 5001)
            self.assertEqual(info['stop_reason'], reason)
            b[22], b[23] = 5001, False
            seed_only(71)
            pcc_numba.pcc_propagate_numba(*b)
            self.assertEqual(rng, next_random())
            for i in (7, 9, 10, 11, 12):
                np.testing.assert_array_equal(a[i], b[i])

    def test_window_range_interior_excursion_resets_patience(self):
        observer = RangeObserver()
        for iteration in range(1, 5002, 10):
            metric = .9 + (.001 if iteration == 4501 else 0.)
            self.assertFalse(observer.observe(iteration, metric))
        # Excursion affects both windows sharing the endpoint 4501.
        self.assertEqual(observer.streak, 0)
        for iteration in range(5011, 6502, 10):
            triggered = observer.observe(iteration, .9)
            self.assertEqual(triggered, iteration == 6501)


if __name__ == '__main__':
    unittest.main()
