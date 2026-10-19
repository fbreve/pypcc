"""Read-only Numba mutation buffers must be rejected before JIT execution."""
import unittest
import numpy as np

try:
    from pcc_numba import pcc_step_numba, pcc_propagate_numba
except ImportError:
    pcc_step_numba = pcc_propagate_numba = None


@unittest.skipIf(pcc_step_numba is None, "Numba unavailable")
class NumbaWritableBufferTests(unittest.TestCase):
    def arguments(self):
        return [np.array([[1], [0]], dtype=np.int64),
                np.array([1, 1], dtype=np.int64),
                np.array([0, -1], dtype=np.int64),
                0.5, 0.1, 2, np.zeros(2),
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.array([1.0]),
                np.zeros((2, 1), dtype=np.uint8),
                np.full((2, 2), 0.5), np.zeros((2, 2))]

    def test_step_rejects_readonly_mutation_buffers(self):
        for index in (7, 9, 10, 11, 12):
            with self.subTest(index=index):
                args = self.arguments()
                args[index].flags.writeable = False
                with self.assertRaisesRegex(ValueError, "writable"):
                    pcc_step_numba(*args)

    def test_propagation_rejects_readonly_scratch_buffer(self):
        for index in range(6):
            with self.subTest(index=index):
                args = self.arguments()
                buffers = [np.empty(2), np.empty(2), np.empty(1),
                           np.empty(1), np.empty(1), np.empty(1)]
                buffers[index].flags.writeable = False
                with self.assertRaisesRegex(ValueError, "writable"):
                    pcc_propagate_numba(*args, 1.0, 2.0, *buffers,
                                        np.ones(257), 1, False, 10, 1)


if __name__ == "__main__":
    unittest.main()
