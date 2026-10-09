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


    def test_parallel_single_visit_matches_sequential_dominance(self):
        # A single visiting particle has no synchronous update conflict.
        for mode in ("parallel", "sequential"):
            neighbors = np.array([[2], [1], [0]], dtype=np.int64)
            degrees = np.ones(3, dtype=np.int64)
            labels = np.array([0, 1, -1], dtype=np.int64)
            positions = np.array([0], dtype=np.int64)
            classes = np.array([0], dtype=np.int64)
            strength = np.array([0.8])
            distance = np.array([[0], [2], [2]], dtype=np.uint8)
            dominance = np.array([[1., 0.], [0., 1.], [.3, .7]])
            owndeg = np.zeros((3, 2))
            pcc_step_numpy(neighbors, degrees, labels, 0., 0.5, 2,
                           np.zeros(2), positions, classes, strength,
                           distance, dominance, owndeg, update_mode=mode)
            np.testing.assert_allclose(dominance[2], [.7, .3], atol=1e-12)

    def test_parallel_isolated_particle_is_unchanged(self):
        neighbors = np.array([[-1], [0]], dtype=np.int64)
        degrees = np.array([0, 1], dtype=np.int64)
        labels = np.array([0, -1], dtype=np.int64)
        positions = np.array([0], dtype=np.int64)
        classes = np.array([0], dtype=np.int64)
        strength = np.array([1.])
        distance = np.array([[0], [1]], dtype=np.uint8)
        dominance = np.array([[1., 0.], [.5, .5]])
        owndeg = np.zeros((2, 2))
        pcc_step_numpy(neighbors, degrees, labels, 0.5, 0.1, 2,
                       np.zeros(2), positions, classes, strength,
                       distance, dominance, owndeg, update_mode="parallel")
        np.testing.assert_array_equal(positions, [0])
        np.testing.assert_array_equal(strength, [1.])
        np.testing.assert_array_equal(dominance[1], [.5, .5])


    def test_high_level_numpy_modes_execute(self):
        from pcc import ParticleCompetitionAndCooperation
        neighbors = np.array([[2, 3], [2, 3], [0, 1], [0, 1]], dtype=np.int64)
        degrees = np.full(4, 2, dtype=np.int64)
        labels = np.array([0, 1, -1, -1], dtype=np.int64)
        for mode in ("parallel", "sequential"):
            model = ParticleCompetitionAndCooperation(impl="numpy", update_mode=mode)
            model.set_graph(neighbors, degrees)
            result = model.fit_predict(labels, max_iter=3, early_stop=False)
            self.assertEqual(result.shape, labels.shape)
            np.testing.assert_array_equal(result[:2], labels[:2])
            self.assertTrue(np.all((result == 0) | (result == 1)))


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
            kwargs = {'update_mode': 'sequential'} if backend is pcc_step_numpy else {}
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
                kwargs = {'update_mode': 'sequential'} if backend is pcc_step_numpy else {}
                backend(neighbors, degrees, labels, 0., 0.25, 2, np.zeros(2),
                        positions, classes, strength, distance, dominance, owndeg,
                        0.5, 2., **kwargs)
                trajectory.append(tuple(a.copy() for a in (
                    positions, strength, distance, dominance, owndeg)))
            snapshots.append(trajectory)
        for numpy_state, numba_state in zip(*snapshots):
            for expected, actual in zip(numpy_state, numba_state):
                np.testing.assert_allclose(actual, expected, atol=1e-12)


    def test_numba_numpy_sequential_probabilistic_moves(self):
        # Seed both RNGs independently; compare a multi-neighbor stochastic
        # trajectory without assuming identical random-stream implementations.
        from numba import njit
        from pcc_numba import pcc_step_numba

        @njit
        def seed_numba(value):
            np.random.seed(value)

        def run(backend, seed, p_grd):
            neighbors = np.array([[2, 3], [2, 3], [0, 1], [0, 1]], dtype=np.int64)
            degrees = np.full(4, 2, dtype=np.int64)
            labels = np.array([0, 1, -1, -1], dtype=np.int64)
            positions = np.array([0, 1], dtype=np.int64)
            classes = np.array([0, 1], dtype=np.int64)
            strength = np.ones(2, dtype=np.float64)
            distance = np.array([[0, 2], [2, 0], [2, 2], [2, 2]], dtype=np.uint8)
            dominance = np.array([[1., 0.], [0., 1.], [.5, .5], [.5, .5]])
            owndeg = np.zeros((4, 2), dtype=np.float64)
            np.random.seed(seed)
            if backend is pcc_step_numba:
                seed_numba(seed)
            for _ in range(12):
                kwargs = {'update_mode': 'sequential'} if backend is pcc_step_numpy else {}
                backend(neighbors, degrees, labels, p_grd, 0.2, 2, np.zeros(2),
                        positions, classes, strength, distance, dominance, owndeg,
                        0.5, 2., **kwargs)
                np.testing.assert_allclose(dominance.sum(axis=1), 1., atol=1e-12)
                self.assertTrue(np.all(dominance >= -1e-12))
            return positions, strength, distance, dominance, owndeg

        for seed in (0, 1, 17, 42):
            for p_grd in (0., 0.5, 1.):
                with self.subTest(seed=seed, p_grd=p_grd):
                    numpy_state = run(pcc_step_numpy, seed, p_grd)
                    numba_state = run(pcc_step_numba, seed, p_grd)
                    for actual, expected in zip(numba_state, numpy_state):
                        np.testing.assert_allclose(actual, expected, atol=1e-12)



    def test_compiled_and_numpy_sequential_forced_graph(self):
        # A single neighbor per node removes backend RNG differences.
        from pcc import ParticleCompetitionAndCooperation
        try:
            import pcc_step
        except ImportError:
            self.skipTest("Cython extension is not compiled")
        neighbors = np.array([[2], [2], [0]], dtype=np.int64)
        degrees = np.ones(3, dtype=np.int64)
        labels = np.array([0, 1, -1], dtype=np.int64)
        states = {}
        for impl in ("cython", "numba", "numpy"):
            model = ParticleCompetitionAndCooperation(
                impl=impl, update_mode="sequential")
            model.set_graph(neighbors, degrees)
            predictions = model.fit_predict(
                labels, p_grd=0., delta_v=0.25, deltap=0.5,
                max_iter=5, early_stop=False)
            states[impl] = (
                predictions.copy(), model.part.curnode.copy(),
                model.part.strength.copy(), model.part.dist_table.copy(),
                model.node.dominance.copy(), model.owndeg.copy())
        for impl in ("cython", "numba"):
            for actual, expected in zip(states[impl], states["numpy"]):
                np.testing.assert_allclose(actual, expected, atol=1e-12)

if __name__ == "__main__":
    unittest.main()
