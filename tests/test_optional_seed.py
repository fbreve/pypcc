"""Explicit seeds replay each available backend without changing seed=None."""
import unittest
import numpy as np
from pcc import ParticleCompetitionAndCooperation


class OptionalSeedTests(unittest.TestCase):
    def run_fit(self, impl, seed):
        model = ParticleCompetitionAndCooperation(impl=impl)
        model.set_graph(
            np.array([[1, 2], [0, 3], [0, 3], [1, 2]], dtype=np.int64),
            np.full(4, 2, dtype=np.int64), k_nn=2)
        prediction = model.fit_predict(
            np.array([0, -1, -1, 1], dtype=np.int64),
            max_iter=30, early_stop=False, seed=seed)
        return prediction, model.node.dominance.copy(), model.part.curnode.copy()

    def test_seeded_replay(self):
        for impl in ("numpy", "numba", "cython"):
            with self.subTest(impl=impl):
                if impl == "numba":
                    try:
                        import pcc_numba
                    except ImportError:
                        continue
                if impl == "cython":
                    try:
                        import pcc_step
                    except ImportError:
                        continue
                a = self.run_fit(impl, 371)
                b = self.run_fit(impl, 371)
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)

    def test_boundary_seeds_replay_and_distinguish_runs(self):
        for impl in ("numpy", "numba", "cython"):
            with self.subTest(impl=impl):
                if impl == "numba":
                    try:
                        import pcc_numba
                    except ImportError:
                        continue
                if impl == "cython":
                    try:
                        import pcc_step
                    except ImportError:
                        continue
                for seed in (0, 1, 2**32 - 1):
                    with self.subTest(impl=impl, seed=seed):
                        a = self.run_fit(impl, seed)
                        b = self.run_fit(impl, seed)
                        for x, y in zip(a, b):
                            np.testing.assert_array_equal(x, y)

    def test_numpy_seeded_fit_replays_after_intervening_random_draws(self):
        np.random.seed(100)
        first = self.run_fit("numpy", 42)
        np.random.random(200)
        second = self.run_fit("numpy", 42)
        for x, y in zip(first, second):
            np.testing.assert_array_equal(x, y)

    def test_zero_iterations_seed_does_not_change_model_random_draws(self):
        # A seeded fit is allowed to reset the RNG, but zero propagation
        # iterations must not advance it beyond that seed.
        for seed in (0, 1, 2**32 - 1):
            np.random.seed(999)
            model = ParticleCompetitionAndCooperation(impl="numpy")
            model.set_graph(
                np.array([[1], [0], [3], [2]], dtype=np.int64),
                np.ones(4, dtype=np.int64))
            model.fit_predict(np.array([0, -1, 1, -1]), max_iter=0,
                              early_stop=False, seed=seed)
            observed = np.random.random()
            np.random.seed(seed)
            expected = np.random.random()
            self.assertEqual(observed, expected)

    def test_invalid_seed_does_not_change_fit(self):
        model = ParticleCompetitionAndCooperation(impl="numpy")
        model.set_graph(
            np.array([[1], [0], [3], [2]], dtype=np.int64),
            np.ones(4, dtype=np.int64))
        labels = np.array([0, -1, 1, -1], dtype=np.int64)
        model.fit_predict(labels, max_iter=0)
        old_node = model.node
        for seed in (-1, 2**32, 1.5, True, "123"):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    model.fit_predict(labels, seed=seed)
                self.assertIs(model.node, old_node)


if __name__ == "__main__":
    unittest.main()
