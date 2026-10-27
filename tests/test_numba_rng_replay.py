"""Numba RNG is seeded inside compiled code, not by NumPy's Python seed."""
import unittest

import numpy as np

from pcc import ParticleCompetitionAndCooperation

try:
    from numba import njit
    import pcc_numba  # noqa: F401
except ImportError:
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


if NUMBA_AVAILABLE:
    @njit
    def seed_numba(seed):
        np.random.seed(seed)


@unittest.skipUnless(NUMBA_AVAILABLE, "Numba is not installed")
class NumbaRngReplayTests(unittest.TestCase):
    def test_seeded_numba_runs_replay_exactly(self):
        labels = np.array([0, -1, -1, 1], dtype=np.int64)
        neighbors = np.array([[1, 2], [0, 3], [0, 3], [1, 2]], dtype=np.int64)
        degrees = np.full(4, 2, dtype=np.int64)
        snapshots = []
        for python_seed in (17, 983):
            np.random.seed(python_seed)
            seed_numba(371)
            model = ParticleCompetitionAndCooperation(impl="numba")
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
