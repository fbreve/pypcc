"""Unit tests for opt-in dominance-change convergence observer."""
import unittest
import numpy as np
from pcc_metrics import DominanceChangeStopTracker
from test_early_stop_ongoing_dominance import EarlyStopOngoingDominanceTests


class DominanceChangeStopTrackerTests(unittest.TestCase):
    def test_ongoing_change_prevents_premature_stop(self):
        fixture = EarlyStopOngoingDominanceTests()
        for backend, fn in fixture.backends():
            with self.subTest(backend=backend):
                tracker = DominanceChangeStopTracker(tolerance=1e-10, stop_max=0)
                _, first = fixture.run_case(fn, False, 1)
                _, eleventh = fixture.run_case(fn, False, 11)
                self.assertFalse(tracker.observe(first))
                self.assertFalse(tracker.observe(eleventh))
                self.assertGreater(tracker.last_change, tracker.tolerance)
                self.assertEqual(tracker.stalled_checks, 0)

    def test_stable_state_obeys_patience(self):
        tracker = DominanceChangeStopTracker(tolerance=0., stop_max=1)
        state = np.array([[1., 0.], [.4, .6]])
        self.assertFalse(tracker.observe(state))
        self.assertFalse(tracker.observe(state))
        self.assertTrue(tracker.observe(state))
        self.assertEqual(tracker.checks, 3)
        self.assertEqual(tracker.stalled_checks, 2)

    def test_snapshot_does_not_alias_mutable_input(self):
        tracker = DominanceChangeStopTracker()
        state = np.array([[.4, .6]])
        self.assertFalse(tracker.observe(state))
        state[0] = [.5, .5]
        self.assertFalse(tracker.observe(state))
        self.assertGreater(tracker.last_change, 0.)

    def test_invalid_controls_and_shapes(self):
        for value in (-1., float('nan'), float('inf'), True):
            with self.subTest(tolerance=value), self.assertRaises(ValueError):
                DominanceChangeStopTracker(tolerance=value)
        for value in (-1, 1.5, True):
            with self.subTest(stop_max=value), self.assertRaises(ValueError):
                DominanceChangeStopTracker(stop_max=value)
        tracker = DominanceChangeStopTracker()
        with self.assertRaises(ValueError):
            tracker.observe(np.array([.5, .5]))
        self.assertFalse(tracker.observe(np.array([[.5, .5]])))
        with self.assertRaises(ValueError):
            tracker.observe(np.array([[.5, .5], [.4, .6]]))


if __name__ == '__main__':
    unittest.main()
