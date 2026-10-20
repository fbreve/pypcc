"""Regression coverage for Cython weighted-neighbor selection."""
import unittest
import numpy as np

try:
    from pcc_step import pcc_step
except ImportError:
    pcc_step = None


@unittest.skipIf(pcc_step is None, "Cython extension unavailable")
class CythonProbabilityTests(unittest.TestCase):
    def test_zero_weight_neighbor_is_not_selected(self):
        # Particle at node 0: neighbor 1 has no dominance in its class,
        # while neighbor 2 has positive dominance. Greedy movement must
        # select node 2, regardless of the random threshold.
        neib_list = np.array([[1, 2], [0, 0], [0, 0]], dtype=np.int64)
        neib_qt = np.array([2, 0, 0], dtype=np.int64)
        labels = np.array([0, 0, 0], dtype=np.int64)
        zerovec = np.zeros(2, dtype=np.float64)
        current = np.array([0], dtype=np.int64)
        particle_labels = np.array([0], dtype=np.int64)
        strength = np.array([1.0], dtype=np.float64)
        distances = np.zeros((3, 1), dtype=np.uint8)
        dominance = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        owndeg = np.zeros((3, 2), dtype=np.float64)
        pcc_step(neib_list, neib_qt, labels, 1.0, 0.1, 2, zerovec,
                 current, particle_labels, strength, distances,
                 dominance, owndeg)
        self.assertEqual(int(current[0]), 2)
        self.assertEqual(float(strength[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
