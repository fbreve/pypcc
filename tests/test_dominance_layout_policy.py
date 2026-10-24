"""Regression checks for backend-specific dominance allocation."""
import numpy as np
import unittest
from pcc import ParticleCompetitionAndCooperation as PCC


CASES = [
    ("numba", 4, False),
    ("numba", 7, False),
    ("numba", 8, True),
    ("numba", 12, True),
    ("numba", 16, True),
    ("numba", 32, True),
    ("numpy", 8, False),
    ("numpy", 16, False),
    ("numpy", 32, False),
    ("cython", 8, False),
    ("cython", 16, False),
    ("cython", 32, False),
]


class DominanceLayoutPolicyTests(unittest.TestCase):
    def test_dominance_order(self):
        for impl, classes, expect_c in CASES:
            with self.subTest(impl=impl, classes=classes):
                check_order(impl, classes, expect_c)


def check_order(impl, classes, expect_c):
    n = max(128, classes * 4)
    model = PCC(impl=impl)
    offsets = np.array([-2, -1, 1, 2], dtype=np.int64)
    neighbors = (np.arange(n, dtype=np.int64)[:, None] + offsets) % n
    model.set_graph(neighbors, np.full(n, 4, dtype=np.int64), k_nn=4)
    labels = np.full(n, -1, dtype=np.int64)
    labels[::4] = np.arange(len(labels[::4])) % classes
    model.fit_predict(labels, max_iter=1, early_stop=False)
    assert model.node.dominance.flags.c_contiguous == expect_c
    assert model.node.dominance.flags.f_contiguous == (not expect_c)
