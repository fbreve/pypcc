"""Regression checks for direct sequential kernel input validation (F01-F03)."""

import unittest

import numpy as np

from pcc_numpy import pcc_step_numpy, pcc_propagate_numpy
from pcc_numba import pcc_step_numba, pcc_propagate_numba

try:
    from pcc_step import pcc_step as pcc_step_cython
    from pcc_step import pcc_propagate as pcc_propagate_cython
except ImportError:
    pcc_step_cython = None
    pcc_propagate_cython = None


def fixture():
    neighbors = np.array([[2], [3], [2], [3]], dtype=np.int64)
    degrees = np.ones(4, dtype=np.int64)
    labels = np.full(4, -1, dtype=np.int64)
    positions = np.array([0, 1], dtype=np.int64)
    classes = np.array([0, 1], dtype=np.int64)
    strength = np.ones(2, dtype=np.float64)
    distances = np.full((4, 2), 3, dtype=np.uint8)
    dominance = np.full((4, 2), 0.5, dtype=np.float64)
    own_degree = np.zeros((4, 2), dtype=np.float64)
    return dict(neib_list=neighbors, neib_qt=degrees, labels=labels,
                p_grd=0.0, delta_v=0.4, c=2,
                zerovec=np.zeros(2, dtype=np.float64),
                part_curnode=positions, part_label=classes,
                part_strength=strength, dist_table=distances,
                dominance=dominance, owndeg=own_degree)


def snapshot(case):
    return {key: value.copy() for key, value in case.items()
            if isinstance(value, np.ndarray)}


def assert_unchanged(test, case, before):
    for key, old in before.items():
        np.testing.assert_array_equal(case[key], old, err_msg=key)


def buffers(case):
    c = case["c"]
    k = case["neib_list"].shape[1]
    return dict(dom_row=np.empty(c), reduc=np.empty(c),
                dom_list=np.empty(k), dist_list=np.empty(k),
                prob=np.empty(k), slices=np.empty(k),
                dist_weights=1.0 / (np.arange(257) + 1.0) ** 2)


class DirectKernelValidation(unittest.TestCase):
    def check_rejected(self, kernel, case, extra=None, check_rng=False):
        extra = extra or {}
        before = snapshot(case)
        if check_rng:
            np.random.seed(123)
            expected = np.random.random()
            np.random.seed(123)
        with self.assertRaises(ValueError):
            kernel(**case, **extra)
        assert_unchanged(self, case, before)
        if check_rng:
            self.assertEqual(np.random.random(), expected)

    def test_short_labels_preflight_numpy(self):
        case = fixture()
        case["labels"] = case["labels"][:-1].copy()
        self.check_rejected(pcc_step_numpy, case, check_rng=True)
        self.check_rejected(pcc_propagate_numpy, case,
                            dict(max_iter=2, early_stop=False, es_chk=10,
                                 stop_max=10, **buffers(case)), check_rng=True)

    def test_integer_strength_rejected_numpy_numba(self):
        for kernel in (pcc_step_numpy, pcc_step_numba):
            with self.subTest(kernel=kernel.__name__):
                case = fixture()
                case["part_strength"] = case["part_strength"].astype(np.int64)
                self.check_rejected(kernel, case, check_rng=(kernel is pcc_step_numpy))

    def test_integer_dom_row_rejected_numba(self):
        case = fixture()
        self.check_rejected(pcc_step_numba, case,
                            dict(dom_row=np.zeros(2, dtype=np.int64)))

    def test_nonfinite_scalar_step_all_backends(self):
        kernels = [pcc_step_numpy, pcc_step_numba]
        if pcc_step_cython is not None:
            kernels.append(pcc_step_cython)
        for kernel in kernels:
            with self.subTest(kernel=kernel.__name__):
                case = fixture()
                case["delta_v"] = np.nan
                self.check_rejected(kernel, case, check_rng=(kernel is pcc_step_numpy))

    def test_nonfinite_scalar_propagate_all_backends(self):
        kernels = [pcc_propagate_numpy, pcc_propagate_numba]
        if pcc_propagate_cython is not None:
            kernels.append(pcc_propagate_cython)
        for kernel in kernels:
            with self.subTest(kernel=kernel.__name__):
                case = fixture()
                case["delta_v"] = np.nan
                extra = dict(max_iter=2, early_stop=False, es_chk=10,
                             stop_max=10, **buffers(case))
                self.check_rejected(kernel, case, extra,
                                    check_rng=(kernel is pcc_propagate_numpy))


if __name__ == "__main__":
    unittest.main()
