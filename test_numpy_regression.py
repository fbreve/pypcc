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
            1., 2., dist_weights=weights, update_mode='sequential'
        )
        self.assertTrue(np.all(dominance >= -1e-12))
        np.testing.assert_allclose(dominance.sum(axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(dominance[2], [0., 1.], atol=1e-12)


    def test_isolated_particle_is_unchanged(self):
        neighbors = np.array([[-1], [0]], dtype=np.int64)
        degrees = np.array([0, 1], dtype=np.int64)
        labels = np.array([0, -1], dtype=np.int64)
        positions = np.array([0], dtype=np.int64)
        classes = np.array([0], dtype=np.int64)
        strength = np.array([1.], dtype=np.float64)
        distance = np.array([[0], [1]], dtype=np.uint8)
        dominance = np.array([[1., 0.], [.5, .5]])
        owndeg = np.zeros((2, 2))
        pcc_step_numpy(neighbors, degrees, labels, 0.5, 0.1, 2,
                       np.zeros(2), positions, classes, strength,
                       distance, dominance, owndeg, update_mode='sequential')
        np.testing.assert_array_equal(positions, [0])
        np.testing.assert_array_equal(strength, [1.])
        np.testing.assert_array_equal(dominance[1], [.5, .5])

    def test_second_particle_sees_first_particles_updated_dominance(self):
        # Both particles have one possible target. The second particle's
        # strength must reflect its own visit, not a later batched update.
        neighbors = np.array([[2], [2], [0]], dtype=np.int64)
        degrees = np.array([1, 1, 1], dtype=np.int64)
        labels = np.array([0, 1, -1], dtype=np.int64)
        positions = np.array([0, 1], dtype=np.int64)
        classes = np.array([0, 1], dtype=np.int64)
        strength = np.ones(2)
        distance = np.array([[0, 2], [2, 0], [2, 2]], dtype=np.uint8)
        dominance = np.array([[1., 0.], [0., 1.], [.5, .5]])
        owndeg = np.zeros((3, 2))
        pcc_step_numpy(neighbors, degrees, labels, 0., 1., 2,
                       np.zeros(2), positions, classes, strength,
                       distance, dominance, owndeg, update_mode='sequential')
        np.testing.assert_allclose(strength, [1., 1.])
        np.testing.assert_allclose(owndeg[2], [1., 1.])
        np.testing.assert_allclose(dominance[2], [0., 1.])


    def test_parallel_collision_preserves_probability_simplex(self):
        neighbors = np.array([[2], [2], [0]], dtype=np.int64)
        degrees = np.ones(3, dtype=np.int64)
        labels = np.array([0, 1, -1], dtype=np.int64)
        positions = np.array([0, 1], dtype=np.int64)
        classes = np.array([0, 1], dtype=np.int64)
        strength = np.ones(2)
        distance = np.array([[0, 2], [2, 0], [2, 2]], dtype=np.uint8)
        dominance = np.array([[1., 0.], [0., 1.], [.5, .5]])
        owndeg = np.zeros((3, 2))
        pcc_step_numpy(neighbors, degrees, labels, 0., 1., 2,
                       np.zeros(2), positions, classes, strength,
                       distance, dominance, owndeg, update_mode="parallel")
        self.assertTrue(np.all(dominance >= -1e-12))
        np.testing.assert_allclose(dominance.sum(axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(dominance[2], [.5, .5], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
