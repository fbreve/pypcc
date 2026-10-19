"""Regression tests for overflowing label values in fit_predict."""
import unittest
import numpy as np
from pcc import ParticleCompetitionAndCooperation


class FitLabelInt64RangeTests(unittest.TestCase):
    def setUp(self):
        self.model = ParticleCompetitionAndCooperation(impl="numpy")
        self.model.set_graph(np.array([[1], [0]], dtype=np.int64),
                             np.array([1, 1], dtype=np.int64))

    def test_overflowing_labels_rejected_before_state_change(self):
        huge = 10 ** 100
        cases = (
            np.array([huge, 0], dtype=object),
            np.array([2**64 - 1, 0], dtype=np.uint64),
            np.array([1e30, 0.0]),
            np.array([-1e30, 0.0]),
        )
        for labels in cases:
            with self.subTest(labels=labels.tolist()):
                with self.assertRaises(ValueError):
                    self.model.fit_predict(labels)
                self.assertIsNone(self.model.labels)
                self.assertIsNone(self.model.part)
                self.assertIsNone(self.model.node)

    def test_in_range_fractional_labels_preserve_truncation(self):
        with self.assertWarns(UserWarning):
            result = self.model.fit_predict(np.array([2.9, 2.1]))
        np.testing.assert_array_equal(result, [2.9, 2.1])
        np.testing.assert_array_equal(self.model.labels, [2, 2])


if __name__ == "__main__":
    unittest.main()
