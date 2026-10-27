"""Compare backend particle state only where movement is RNG-independent."""
import unittest
import numpy as np

from test_sequential_safety import state, buffers
from pcc_numpy import pcc_propagate_numpy

try:
    from pcc_numba import pcc_propagate_numba
except ImportError:
    pcc_propagate_numba = None

try:
    from pcc_step import pcc_propagate
except ImportError:
    pcc_propagate = None


class DeterministicBackendStateParityTests(unittest.TestCase):
    def backends(self):
        result = [('numpy', pcc_propagate_numpy)]
        if pcc_propagate_numba is not None:
            result.append(('numba', pcc_propagate_numba))
        if pcc_propagate is not None:
            result.append(('cython', pcc_propagate))
        return result

    def run_case(self, fn, iterations, isolated):
        a = state()
        if isolated:
            # All particles start on an isolated node: no RNG draws or moves.
            a[1][0] = 0
        b = buffers(a)
        info = fn(*a, *b, iterations, False, 2000, 1,
                  return_info=True, seed=42)
        return [a[i].copy() for i in (7, 8, 9, 10, 11, 12)], info

    def test_zero_iterations_preserve_full_particle_state(self):
        reference, metrics = self.run_case(pcc_propagate_numpy, 0, False)
        for backend, fn in self.backends():
            with self.subTest(backend=backend):
                actual, info = self.run_case(fn, 0, False)
                for expected, observed in zip(reference, actual):
                    np.testing.assert_array_equal(observed, expected)
                self.assertEqual(info, metrics)

    def test_isolated_particles_preserve_full_state_across_iterations(self):
        for iterations in (1, 10, 21):
            reference, metrics = self.run_case(
                pcc_propagate_numpy, iterations, True)
            for backend, fn in self.backends():
                with self.subTest(backend=backend, iterations=iterations):
                    actual, info = self.run_case(fn, iterations, True)
                    for expected, observed in zip(reference, actual):
                        np.testing.assert_array_equal(observed, expected)
                    self.assertEqual(info, metrics)


if __name__ == '__main__':
    unittest.main()
