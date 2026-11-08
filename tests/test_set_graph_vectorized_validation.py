"""Focused regression for vectorized active-neighbor validation in set_graph.

The public setter must still copy only active entries, preserve int64 conversion
and atomic failure, and keep fitted results independent of input storage.
"""
from decimal import Decimal
import unittest
import warnings

import numpy as np

from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_graph import build_knn_graph


class SetGraphVectorizedValidationTests(unittest.TestCase):
    def check_installed_graph(self, neighbors, degrees, expected, k_nn=None):
        m = PCC(impl="numpy")
        m.set_graph(neighbors, degrees, k_nn=k_nn)
        np.testing.assert_array_equal(m.neib_list, expected)
        np.testing.assert_array_equal(m.neib_qt, np.asarray(degrees, dtype=np.int64))
        self.assertEqual(m.neib_list.dtype, np.int64)
        self.assertEqual(m.neib_qt.dtype, np.int64)
        self.assertFalse(np.shares_memory(m.neib_list, neighbors))
        self.assertFalse(np.shares_memory(m.neib_qt, degrees))
        return m

    def test_original_int64_graph_and_padding_retained(self):
        a = np.array([[1, -1, -1], [0, 2, -1], [1, -1, -1]], dtype=np.int64)
        d = np.array([1, 2, 1], dtype=np.int64)
        m = self.check_installed_graph(a, d,
                                        np.array([[1, 0, 0], [0, 2, 0], [1, 0, 0]]))
        before = m.neib_list.copy()
        a[0, 0] = 2
        np.testing.assert_array_equal(m.neib_list, before)

    def test_finite_casts_noncontiguous_and_ignored_invalid_padding(self):
        a = np.array([[1.9, np.nan], [0.9, np.inf]], order="F")
        self.check_installed_graph(a, np.array([1., 1.]),
                                   np.array([[1, 0], [0, 0]]))
        # Actual noncontiguous views and object-array padding are also legal.
        raw = np.array([[1, "bad", None, "unused"], [0, np.nan, None, np.inf]],
                       dtype=object)
        a = raw[:, ::2]
        self.assertFalse(a.flags.c_contiguous)
        self.check_installed_graph(a, np.array([1, 1]),
                                   np.array([[1, 0], [0, 0]]))
        s = np.array([["1", "ignored"], ["0", "ignored"]])
        self.check_installed_graph(s, np.array([1, 1]),
                                   np.array([[1, 0], [0, 0]]))

    def test_floats_and_empty_shapes_keep_original_conversion(self):
        a = np.array([[1.9, -1.], [0.9, -1.]])
        self.check_installed_graph(a, np.array([1.9, 1.9]),
                                   np.array([[1, 0], [0, 0]]), k_nn=1.9)
        self.check_installed_graph(np.empty((0, 3), dtype=np.int64),
                                   np.empty(0, dtype=np.int64),
                                   np.empty((0, 3), dtype=np.int64))
        self.check_installed_graph(np.empty((2, 0), dtype=np.int64),
                                   np.zeros(2, dtype=np.int64),
                                   np.empty((2, 0), dtype=np.int64))
        self.check_installed_graph(np.array([[1, 0], [0, 0]]),
                                   np.zeros(2, dtype=np.int64),
                                   np.zeros((2, 2), dtype=np.int64))

    def test_bad_active_values_preserve_previous_graph_and_fit(self):
        m = PCC(impl="numpy")
        m.set_graph(np.array([[2], [2], [0]], dtype=np.int64),
                    np.ones(3, dtype=np.int64))
        m.fit_predict(np.array([0, 1, -1]), max_iter=3, early_stop=False, seed=71)
        before = dict(m.__dict__)
        cases = [
            (np.array([[np.nan], [2.], [0.]]), "finite"),
            (np.array([[np.inf], [2.], [0.]]), "finite"),
            (np.array([[Decimal("NaN")], [2], [0]], dtype=object), "finite"),
            (np.array([[10**100], [2], [0]], dtype=object), "range"),
            (np.array([[2**64 - 1], [2], [0]], dtype=np.uint64), "range"),
            (np.array([[-1], [2], [0]], dtype=np.int64), "bounds"),
            (np.array([[3], [2], [0]], dtype=np.int64), "bounds"),
            (np.array([["bad"], [2], [0]], dtype=object), "conversion"),
        ]
        for array, kind in cases:
            with self.subTest(kind=kind, dtype=str(array.dtype)):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with self.assertRaises((TypeError, ValueError, OverflowError)):
                        m.set_graph(array, np.ones(3, dtype=np.int64))
                self.assertEqual(set(m.__dict__), set(before))
                for field, previous in before.items():
                    self.assertIs(m.__dict__[field], previous, field)

    def test_no_graph_change_to_seeded_native_propagation(self):
        data = np.array([[0.], [1.], [2.], [3.], [4.], [5.]])
        graph, degrees = build_knn_graph(data, k_nn=2)
        labels = np.array([0, -1, -1, -1, -1, 1])
        for backend in ("numpy", "numba", "cython"):
            with self.subTest(backend=backend):
                original = PCC(impl=backend)
                from_setter = PCC(impl=backend)
                original.build_graph(data, k_nn=2)
                from_setter.set_graph(graph, degrees, k_nn=2)
                self.assertEqual(original._get_backend_fn().__module__,
                                 from_setter._get_backend_fn().__module__)
                pred_a = original.fit_predict(labels, max_iter=17,
                                              early_stop=False, seed=34007)
                pred_b = from_setter.fit_predict(labels, max_iter=17,
                                                 early_stop=False, seed=34007)
                for a, b in ((pred_a, pred_b),
                             (original.node.dominance, from_setter.node.dominance),
                             (original.part.curnode, from_setter.part.curnode),
                             (original.part.strength, from_setter.part.strength),
                             (original.part.dist_table, from_setter.part.dist_table),
                             (original.owndeg, from_setter.owndeg)):
                    np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
