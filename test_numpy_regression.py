import unittest
import numpy as np

from pcc_numpy import pcc_step_numpy


class NumpyDominanceRegressionTests(unittest.TestCase):
    def test_competing_particles_do_not_create_negative_dominance(self):
        # Both particles must visit node 2; they represent opposing classes.
        neighbors = np.array([[2], [2], [0]], dtype=np.int64)
        degrees = np.array([1, 1, 1], dtype=np.int64)
        labels = np.array([0, 1, -1], dtype=np.int64)
        positions = np.array([0, 1], dtype=np.int64)
        classes = np.array([0, 1], dtype=np.int64)
        strength = np.ones(2, dtype=np.float64)
        distance = np.full((3, 2), 2, dtype=np.uint8)
        distance[0, 0] = 0
        distance[1, 1] = 0
        dominance = np.array([[1., 0.], [0., 1.], [.5, .5]])
        owndeg = np.zeros((3, 2), dtype=np.float64)
        weights = 1. / (np.arange(257) + 1.) ** 2

        pcc_step_numpy(
            neighbors, degrees, labels, 0., 1., 2, np.zeros(2),
            positions, classes, strength, distance, dominance, owndeg,
            1., 2., dist_weights=weights
        )
        self.assertTrue(np.all(dominance >= -1e-12))
        np.testing.assert_allclose(dominance.sum(axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(dominance[2], [0., 1.], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
