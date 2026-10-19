"""Synchronous NumPy entry points must reject invalid inputs before RNG or mutation."""
import unittest
from unittest.mock import patch
import numpy as np
from pcc_numpy import pcc_step_numpy, pcc_propagate_numpy


class SynchronousPreflightTests(unittest.TestCase):
    def make_args(self):
        return [
            np.array([[1], [0]], dtype=np.int64),  # neib_list
            np.array([1, 1], dtype=np.int64),  # neib_qt
            np.array([0, -1], dtype=np.int64),  # labels
            0.5, 0.1, 2, np.zeros(2),
            np.array([0], dtype=np.int64),  # current
            np.array([0], dtype=np.int64),  # particle labels
            np.array([1.0]),  # strength
            np.zeros((2, 1), dtype=np.uint8),  # distances
            np.full((2, 2), 0.5),  # dominance
            np.zeros((2, 2)),  # owndeg
        ]

    def call_entry(self, entry, args):
        if entry == "step":
            return pcc_step_numpy(*args, update_mode="synchronous")
        return pcc_propagate_numpy(
            *args, 1.0, 2.0, None, None, None, None, None, None,
            None, 1, False, 10, 1, update_mode="synchronous")

    def test_invalid_shape_rejected_before_rng_or_mutation(self):
        for entry in ("step", "propagate"):
            for field, invalid in ((1, np.array([2, 1])),
                                   (10, np.zeros((1, 1), dtype=np.uint8))):
                with self.subTest(entry=entry, field=field):
                    args = self.make_args()
                    args[field] = invalid
                    original = [a.copy() for a in args if isinstance(a, np.ndarray)]
                    with patch("numpy.random.random", side_effect=AssertionError("RNG consumed")):
                        with self.assertRaises(ValueError):
                            self.call_entry(entry, args)
                    for actual, expected in zip((a for a in args if isinstance(a, np.ndarray)), original):
                        np.testing.assert_array_equal(actual, expected)

    def test_step_rejects_nonfinite_scalar_and_readonly_state(self):
        for field, value in ((3, np.nan), (4, np.inf)):
            args = self.make_args()
            args[field] = value
            with patch("numpy.random.random", side_effect=AssertionError("RNG consumed")):
                with self.assertRaises(ValueError):
                    self.call_entry("step", args)
        args = self.make_args()
        args[11].flags.writeable = False
        with patch("numpy.random.random", side_effect=AssertionError("RNG consumed")):
            with self.assertRaises(ValueError):
                self.call_entry("step", args)


if __name__ == "__main__":
    unittest.main()
