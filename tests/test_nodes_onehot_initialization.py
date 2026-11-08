"""Characterize vectorized dominance initialization without changing graph layout."""
import unittest

import numpy as np

from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_graph import build_knn_graph


class NodeOnehotInitializationTests(unittest.TestCase):
    def model_for_state(self, mapped, classes, backend):
        model = PCC(impl=backend)
        model.neib_list = np.empty((mapped.size, 1), dtype=np.int64)
        model.c = classes
        model.mapped_labels = mapped
        return model

    def assert_original_initialization(self, model, expected_order):
        mapped = model.mapped_labels
        n = mapped.size
        classes = model.c
        expected = np.full((n, classes), 1.0 / classes, dtype=np.float64,
                           order=expected_order)
        labeled = mapped != -1
        expected[labeled] = 0.0
        for cls in range(classes):
            expected[mapped == cls, cls] = 1.0

        nodes = model._ParticleCompetitionAndCooperation__genNodes()
        np.testing.assert_array_equal(nodes.dominance, expected)
        np.testing.assert_array_equal(nodes.label, mapped)
        self.assertFalse(np.shares_memory(nodes.label, mapped))
        self.assertEqual(nodes.dominance.dtype, np.float64)
        self.assertEqual(nodes.dominance.flags.f_contiguous if expected_order == "F"
                         else nodes.dominance.flags.c_contiguous, True)
        self.assertEqual(nodes.amount, n)
        self.assertEqual(model.owndeg.shape, (n, classes))
        np.testing.assert_array_equal(model.owndeg,
                                      np.zeros((n, classes), dtype=np.float64))
        self.assertTrue(model.owndeg.flags.f_contiguous)
        return nodes

    def test_empty_all_unlabeled_all_labeled_and_sparse_legacy(self):
        for c in (2, 3, 10):
            cases = [
                np.empty(0, dtype=np.int64),
                np.full(17, -1, dtype=np.int64),
                np.arange(33, dtype=np.int64) % c,
                np.concatenate((np.full(23, -1, dtype=np.int64),
                                np.arange(c, dtype=np.int64))),
                np.array([0, -1, c-1, -1, 0, c-1], dtype=np.int64),
            ]
            for backend in ("numpy", "numba", "cython"):
                expected_order = "C" if backend == "numba" and c >= 8 else "F"
                for mapped in cases:
                    with self.subTest(classes=c, backend=backend,
                                      nodes=len(mapped)):
                        model = self.model_for_state(mapped, c, backend)
                        selected = model._get_backend_fn().__module__
                        self.assertEqual(selected, {"numpy": "pcc_numpy",
                                                    "numba": "pcc_numba",
                                                    "cython": "pcc_step"}[backend])
                        self.assert_original_initialization(model, expected_order)

    def test_negative_external_labels_map_as_before(self):
        data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        labels = np.array([-8, -1, 41, -1, -8, 41], dtype=np.int64)
        graph, degrees = build_knn_graph(data, k_nn=2)
        for backend in ("numpy", "numba", "cython"):
            with self.subTest(backend=backend):
                m = PCC(impl=backend)
                m.set_graph(graph, degrees, k_nn=2)
                prediction = m.fit_predict(labels, max_iter=17,
                                           early_stop=False, seed=35059)
                np.testing.assert_array_equal(m.unique_labels, [-8, 41])
                np.testing.assert_array_equal(m.mapped_labels,
                                              [0, -1, 1, -1, 0, 1])
                self.assertEqual(prediction.shape, labels.shape)
                self.assertEqual(m.node.dominance.shape, (6, 2))
                np.testing.assert_array_equal(m.node.dominance[[0, 2, 4, 5]],
                                              [[1, 0], [0, 1], [1, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()
