"""Compare read-only convergence signals on a known premature-stop case.

No alternative stopping policy is enabled by these tests.
"""
import unittest
import numpy as np
from test_early_stop_ongoing_dominance import EarlyStopOngoingDominanceTests


def maximum_mean_potential(dominance):
    return float(np.max(dominance, axis=1).mean())


def maximum_dominance_change(before, after):
    return float(np.max(np.abs(after - before)))


class EarlyStopMetricComparisonTests(unittest.TestCase):
    def test_declining_potential_does_not_mean_stagnation(self):
        fixture = EarlyStopOngoingDominanceTests()
        for backend, fn in fixture.backends():
            with self.subTest(backend=backend):
                _, at_one = fixture.run_case(fn, False, 1)
                _, at_eleven = fixture.run_case(fn, False, 11)
                _, at_twelve = fixture.run_case(fn, False, 12)
                info, stopped = fixture.run_case(fn, True, 100)
                self.assertEqual(info['effective_iterations'], 11)
                np.testing.assert_array_equal(stopped, at_eleven)
                self.assertLess(maximum_mean_potential(at_eleven),
                                maximum_mean_potential(at_one))
                self.assertGreater(maximum_dominance_change(at_eleven, at_twelve),
                                   1e-10)
                # The two measurements answer different questions: the
                # existing criterion tracks the best-ever mean maximum,
                # whereas this candidate tracks changes in the full matrix.
                self.assertGreater(maximum_dominance_change(at_one, at_eleven),
                                   1e-10)

    def test_maximum_potential_can_hide_class_reassignment(self):
        before = np.array([[.9, .1], [.2, .8]], dtype=np.float64)
        after = np.array([[.1, .9], [.8, .2]], dtype=np.float64)
        self.assertEqual(maximum_mean_potential(before),
                         maximum_mean_potential(after))
        self.assertGreater(maximum_dominance_change(before, after), .5)

    def test_change_metric_is_zero_for_identical_states(self):
        before = np.array([[1., 0.], [.4, .6]], dtype=np.float64)
        self.assertEqual(maximum_dominance_change(before, before.copy()), 0.)


if __name__ == '__main__':
    unittest.main()
