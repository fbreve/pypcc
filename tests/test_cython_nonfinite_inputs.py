"""Nonfinite probability inputs must fail before compiled Cython mutates state."""
import unittest
import numpy as np

try:
    from pcc_step import pcc_step, pcc_propagate
except ImportError:
    pcc_step = pcc_propagate = None


@unittest.skipIf(pcc_step is None, "Cython extension unavailable")
class CythonNonfiniteInputsTests(unittest.TestCase):
    def make_args(self):
        return [np.array([[1], [0]], dtype=np.int64),
                np.array([1, 1], dtype=np.int64),
                np.array([0, -1], dtype=np.int64),
                1.0, 0.1, 2, np.zeros(2),
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.ones(1), np.zeros((2, 1), dtype=np.uint8),
                np.full((2, 2), 0.5), np.zeros((2, 2))]

    def test_step_rejects_nonfinite_probability_inputs(self):
        for index, location in ((9, (0,)), (11, (1, 0))):
            with self.subTest(index=index):
                args = self.make_args()
                args[index][location] = np.nan
                state = [x.copy() for x in args if isinstance(x, np.ndarray)]
                with self.assertRaisesRegex(ValueError, "finite"):
                    pcc_step(*args)
                for actual, before in zip((x for x in args if isinstance(x, np.ndarray)), state):
                    np.testing.assert_array_equal(actual, before)

    def test_step_rejects_nonfinite_distance_weights(self):
        args = self.make_args()
        weights = np.ones(257)
        weights[0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            pcc_step(*args, dist_weights=weights)

    def test_propagation_rejects_nonfinite_inputs(self):
        args = self.make_args()
        args[11][1, 0] = np.nan
        buffers = [np.zeros(2), np.zeros(2), np.zeros(1),
                   np.zeros(1), np.zeros(1), np.zeros(1)]
        with self.assertRaisesRegex(ValueError, "finite"):
            pcc_propagate(*args, 1.0, 2.0, *buffers,
                          np.ones(257), 1, 0, 10, 1)


if __name__ == "__main__":
    unittest.main()
