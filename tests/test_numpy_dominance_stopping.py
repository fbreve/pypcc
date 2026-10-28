"""Optional NumPy stopping criterion, without changing the legacy default."""
import unittest
import numpy as np
from pcc_numpy import pcc_propagate_numpy
from test_early_stop_ongoing_dominance import EarlyStopOngoingDominanceTests


class NumpyDominanceStoppingTests(unittest.TestCase):
    def run_case(self, criterion=None, tolerance=0.):
        edges = np.array([[1], [0]], dtype=np.int64)
        degrees = np.ones(2, dtype=np.int64)
        labels = np.array([0, -1], dtype=np.int64)
        dominance = np.array([[1., 0.], [.1, .9]], dtype=np.float64)
        positions = np.array([0], dtype=np.int64)
        strengths = np.array([1.], dtype=np.float64)
        distances = np.array([[0], [255]], dtype=np.uint8)
        ownership = np.zeros((2, 2), dtype=np.float64)
        args = [edges, degrees, labels, 0., .05, 2, np.zeros(2),
                positions, np.array([0], dtype=np.int64), strengths,
                distances, dominance, ownership, 1., 2.]
        scratch = [np.empty(2), np.empty(2)] + [np.empty(1) for _ in range(4)] + [1. / (np.arange(257) + 1.) ** 2]
        kwargs = {'return_info': True, 'seed': 31}
        if criterion is not None:
            kwargs.update(stop_criterion=criterion, dominance_tol=tolerance)
        info = pcc_propagate_numpy(*args, *scratch, 100, True, 2000, 0, **kwargs)
        return info, dominance

    def test_default_is_bitwise_identical_to_explicit_legacy(self):
        default_info, default_dom = self.run_case()
        legacy_info, legacy_dom = self.run_case('potential')
        self.assertEqual(default_info, legacy_info)
        np.testing.assert_array_equal(default_dom, legacy_dom)
        self.assertEqual(default_info['effective_iterations'], 11)

    def test_opt_in_avoids_known_premature_stop(self):
        legacy, _ = self.run_case()
        experimental, _ = self.run_case('dominance_change', 1e-10)
        self.assertGreater(experimental['effective_iterations'], legacy['effective_iterations'])

    def test_invalid_criterion_is_atomic(self):
        with self.assertRaisesRegex(ValueError, 'stop_criterion'):
            self.run_case('unknown')
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            self.run_case('dominance_change', -1.)


if __name__ == '__main__':
    unittest.main()
