"""Cython's internal fixed-seed RNG should replay independently of NumPy's seed."""
import unittest

import numpy as np

from pcc import ParticleCompetitionAndCooperation

try:
    from pcc_step import pcc_propagate  # noqa: F401
except ImportError:
    CYTHON_AVAILABLE = False
else:
    CYTHON_AVAILABLE = True


@unittest.skipUnless(CYTHON_AVAILABLE, "Cython extension is not built")
class CythonRngReplayTests(unittest.TestCase):
    def test_cython_replays_independently_of_numpy_seed(self):
        labels = np.array([0, -1, -1, 1], dtype=np.int64)
        neighbors = np.array([[1, 2], [0, 3], [0, 3], [1, 2]], dtype=np.int64)
        degrees = np.full(4, 2, dtype=np.int64)
        snapshots = []
        for seed in (17, 983):
            np.random.seed(seed)
            model = ParticleCompetitionAndCooperation(impl="cython")
            model.set_graph(neighbors, degrees, k_nn=2)
            prediction = model.fit_predict(labels, max_iter=25, early_stop=False)
            snapshots.append((
                prediction.copy(),
                model.node.dominance.copy(),
                model.part.curnode.copy(),
                model.part.strength.copy(),
                model.part.dist_table.copy(),
            ))
        for first, second in zip(*snapshots):
            np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
