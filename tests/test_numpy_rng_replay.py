"""Reproducibility contracts for the NumPy propagation backend."""
import unittest

import numpy as np

from pcc import ParticleCompetitionAndCooperation


class NumpyRngReplayTests(unittest.TestCase):
    def make_model(self):
        model = ParticleCompetitionAndCooperation(impl="numpy")
        model.set_graph(
            np.array([[1, 2], [0, 3], [0, 3], [1, 2]], dtype=np.int64),
            np.array([2, 2, 2, 2], dtype=np.int64),
            k_nn=2,
        )
        return model

    def test_zero_iterations_do_not_advance_numpy_rng(self):
        labels = np.array([0, -1, -1, 1], dtype=np.int64)
        np.random.seed(371)
        expected = np.random.random()
        np.random.seed(371)
        self.make_model().fit_predict(labels, max_iter=0, early_stop=False)
        self.assertEqual(np.random.random(), expected)

    def test_seeded_numpy_runs_replay_exactly(self):
        labels = np.array([0, -1, -1, 1], dtype=np.int64)
        snapshots = []
        for _ in range(2):
            np.random.seed(371)
            model = self.make_model()
            prediction = model.fit_predict(labels, max_iter=25, early_stop=False)
            snapshots.append((
                prediction.copy(),
                model.node.dominance.copy(),
                model.part.curnode.copy(),
                model.part.strength.copy(),
                model.part.dist_table.copy(),
                np.random.random(),
            ))
        for first, second in zip(*snapshots):
            np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
