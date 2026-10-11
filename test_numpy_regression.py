import unittest
from unittest.mock import patch
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
        np.testing.assert_array_equal(positions, [2, 2])
        np.testing.assert_array_equal(distance[2], [1, 1])
        np.testing.assert_array_equal(dominance[:2], [[1., 0.], [0., 1.]])
        self.assertTrue(np.isfinite(dominance).all())

    def test_same_class_collision_does_not_overdraw_dominance(self):
        neighbors = np.array([[2], [2], [0]], dtype=np.int64)
        positions = np.array([0, 1], dtype=np.int64)
        dominance = np.array([[1., 0.], [1., 0.], [.5, .5]])
        strength = np.ones(2)
        distance = np.array([[0, 2], [2, 0], [2, 2]], dtype=np.uint8)
        pcc_step_numpy(neighbors, np.ones(3, dtype=np.int64),
                       np.array([0, 0, -1]), 0., 1., 2, np.zeros(2),
                       positions, np.array([0, 0]), strength, distance,
                       dominance, np.zeros((3, 2)),
                       dist_weights=1. / (np.arange(257) + 1.) ** 2)
        np.testing.assert_array_equal(dominance[2], [1., 0.])
        np.testing.assert_array_equal(strength, [1., 1.])

    def test_later_greedy_choice_uses_updated_dominance(self):
        neighbors = np.array([[2, -1], [2, 3], [0, -1], [1, -1]], dtype=np.int64)
        positions = np.array([0, 1], dtype=np.int64)
        dominance = np.array([[1., 0.], [0., 1.], [.5, .5], [.5, .5]])
        distance = np.array([[0, 3], [3, 0], [3, 3], [3, 3]], dtype=np.uint8)
        with patch('pcc_numpy.np.random.random', return_value=0.1):
            pcc_step_numpy(neighbors, np.array([1, 2, 1, 1]),
                           np.array([0, 1, -1, -1]), 1., 1., 2, np.zeros(2),
                           positions, np.array([0, 1]), np.ones(2), distance,
                           dominance, np.zeros((4, 2)),
                           dist_weights=1. / (np.arange(257) + 1.) ** 2)
        np.testing.assert_array_equal(positions, [2, 3])
        np.testing.assert_array_equal(dominance[2:], [[1., 0.], [0., 1.]])

    def test_distance_255_does_not_wrap(self):
        distance = np.full((3, 2), 255, dtype=np.uint8)
        pcc_step_numpy(np.array([[2], [2], [0]]), np.ones(3, dtype=np.int64),
                       np.array([0, 1, -1]), 0., .1, 2, np.zeros(2),
                       np.array([0, 1]), np.array([0, 1]), np.ones(2), distance,
                       np.array([[1., 0.], [0., 1.], [.5, .5]]), np.zeros((3, 2)))
        np.testing.assert_array_equal(distance, np.full((3, 2), 255))


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
                       distance, dominance, owndeg)
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
                       distance, dominance, owndeg)
        np.testing.assert_allclose(strength, [1., 1.])
        np.testing.assert_allclose(owndeg[2], [1., 1.])
        np.testing.assert_allclose(dominance[2], [0., 1.])
        np.testing.assert_array_equal(positions, [2, 2])


    def test_numba_and_numpy_sequential_one_step_with_forced_neighbors(self):
        from pcc_numba import pcc_step_numba
        reference = None
        for backend in (pcc_step_numpy, pcc_step_numba):
            neighbors = np.array([[2], [2], [0]], dtype=np.int64)
            degrees = np.ones(3, dtype=np.int64)
            labels = np.array([0, 1, -1], dtype=np.int64)
            positions = np.array([0, 1], dtype=np.int64)
            classes = np.array([0, 1], dtype=np.int64)
            strength = np.ones(2, dtype=np.float64)
            distance = np.array([[0, 2], [2, 0], [2, 2]], dtype=np.uint8)
            dominance = np.array([[1., 0.], [0., 1.], [.5, .5]])
            owndeg = np.zeros((3, 2), dtype=np.float64)
            kwargs = {}
            backend(neighbors, degrees, labels, 0., 1., 2, np.zeros(2),
                    positions, classes, strength, distance, dominance, owndeg,
                    1., 2., **kwargs)
            state = (positions, strength, distance, dominance, owndeg)
            if reference is None:
                reference = tuple(x.copy() for x in state)
            else:
                for actual, expected in zip(state, reference):
                    np.testing.assert_allclose(actual, expected, atol=1e-12)


    def test_numba_numpy_sequential_multiple_forced_steps(self):
        from pcc_numba import pcc_step_numba
        snapshots = []
        for backend in (pcc_step_numpy, pcc_step_numba):
            neighbors = np.array([[2], [2], [0]], dtype=np.int64)
            degrees = np.ones(3, dtype=np.int64)
            labels = np.array([0, 1, -1], dtype=np.int64)
            positions = np.array([0, 1], dtype=np.int64)
            classes = np.array([0, 1], dtype=np.int64)
            strength = np.array([0.7, 0.9], dtype=np.float64)
            distance = np.array([[0, 2], [2, 0], [2, 2]], dtype=np.uint8)
            dominance = np.array([[1., 0.], [0., 1.], [.4, .6]])
            owndeg = np.zeros((3, 2), dtype=np.float64)
            trajectory = []
            for _ in range(5):
                kwargs = {}
                backend(neighbors, degrees, labels, 0., 0.25, 2, np.zeros(2),
                        positions, classes, strength, distance, dominance, owndeg,
                        0.5, 2., **kwargs)
                self.assertTrue(np.isfinite(dominance).all())
                self.assertTrue(np.all((dominance >= 0.) & (dominance <= 1.)))
                np.testing.assert_allclose(dominance.sum(axis=1), 1., atol=1e-12)
                np.testing.assert_array_equal(dominance[:2], [[1., 0.], [0., 1.]])
                self.assertTrue(np.all((positions >= 0) & (positions < 3)))
                np.testing.assert_array_equal(distance[[0, 1], [0, 1]], [0, 0])
                trajectory.append(tuple(a.copy() for a in (
                    positions, strength, distance, dominance, owndeg)))
            snapshots.append(trajectory)
        for numpy_state, numba_state in zip(*snapshots):
            for expected, actual in zip(numpy_state, numba_state):
                np.testing.assert_allclose(actual, expected, atol=1e-12)


    def test_compiled_and_numpy_sequential_forced_graph(self):
        # A single neighbor removes destination randomness; p_grd=0 or 1
        # also removes RNG-dependent greedy/random decisions.
        try:
            import pcc_step
        except ImportError:
            self.skipTest("Cython extension is not compiled")
        neighbors = np.array([[2], [2], [0]], dtype=np.int64)
        degrees = np.ones(3, dtype=np.int64)
        labels = np.array([0, 1, -1], dtype=np.int64)
        for p_grd in (0., 1.):
            for delta_v in (0.1, 0.25):
                with self.subTest(p_grd=p_grd, delta_v=delta_v):
                    self._check_compiled_forced_graph(neighbors, degrees, labels, p_grd, delta_v)

    def _check_compiled_forced_graph(self, neighbors, degrees, labels, p_grd, delta_v):
        from pcc import ParticleCompetitionAndCooperation
        states = {}
        for impl in ("cython", "numba", "numpy"):
            model = ParticleCompetitionAndCooperation(
                impl=impl)
            model.set_graph(neighbors, degrees)
            self.assertEqual(model._get_backend_fn().__module__,
                             {'numpy': 'pcc_numpy', 'numba': 'pcc_numba',
                              'cython': 'pcc_step'}[impl])
            predictions = model.fit_predict(
                labels, p_grd=p_grd, delta_v=delta_v, deltap=0.5,
                max_iter=5, early_stop=False)
            states[impl] = (
                predictions.copy(), model.part.curnode.copy(),
                model.part.strength.copy(), model.part.dist_table.copy(),
                model.node.dominance.copy(), model.owndeg.copy())
            np.testing.assert_array_equal(predictions[:2], labels[:2])
            self.assertTrue(np.isfinite(model.node.dominance).all())
            self.assertTrue(np.all((model.node.dominance >= 0.) &
                                  (model.node.dominance <= 1.)))
        for impl in ("cython", "numba"):
            for actual, expected in zip(states[impl], states["numpy"]):
                np.testing.assert_allclose(actual, expected, atol=1e-12)

if __name__ == "__main__":
    unittest.main()
