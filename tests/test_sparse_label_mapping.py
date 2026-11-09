"""Exact public fit-label mapping around the 1024-node fast-path boundary."""
import unittest
import warnings

import numpy as np

from pcc import ParticleCompetitionAndCooperation as PCC


def ring(n):
    ids = np.arange(n, dtype=np.int64)
    return np.stack(((ids - 1) % n, (ids + 1) % n), axis=1)


class SparseLabelMappingTests(unittest.TestCase):
    def fit_check(self, n, values, backend="numpy", max_iter=1):
        values = np.asarray(values, dtype=np.int64)
        self.assertEqual(values.shape, (n,))
        classes = np.unique(values[values != -1])
        self.assertGreaterEqual(classes.size, 2)
        m = PCC(impl=backend)
        m.set_graph(ring(n), np.full(n, 2, dtype=np.int64), k_nn=2)
        expected = np.full(n, -1, dtype=np.int64)
        ordered_map = {lbl: i for i, lbl in enumerate(classes)}
        for lbl, idx in ordered_map.items():
            expected[values == lbl] = idx
        m.fit_predict(values, max_iter=max_iter, early_stop=False, seed=37059)
        np.testing.assert_array_equal(m.mapped_labels, expected)
        np.testing.assert_array_equal(m.unique_labels, classes)
        self.assertEqual(m.c, classes.size)
        self.assertEqual(m.label_to_idx, ordered_map)
        self.assertEqual(m.idx_to_label, {i: lbl for lbl, i in ordered_map.items()})
        self.assertEqual(m.mapped_labels.dtype, np.int64)
        self.assertTrue(m.mapped_labels.flags.c_contiguous)
        self.assertTrue(m.mapped_labels.flags.writeable)
        self.assertFalse(np.shares_memory(m.mapped_labels, values))
        np.testing.assert_array_equal(m.node.label[values != -1],
                                      expected[values != -1])

    def test_boundary_and_sparse_class_counts(self):
        for n, classes in ((150, 3), (569, 2), (1023, 10),
                           (1024, 10), (1797, 10), (2048, 50)):
            with self.subTest(nodes=n, classes=classes):
                labels = np.full(n, -1, dtype=np.int64)
                targets = 37 * np.arange(classes, dtype=np.int64) - 10000
                labels[:3*classes:3] = targets
                labels[1:3*classes:3] = targets
                labels[2:3*classes:3] = targets
                self.fit_check(n, labels)

    def test_extreme_int64_and_unknown_between_classes(self):
        high = np.iinfo(np.int64).max
        low = np.iinfo(np.int64).min
        for n in (1023, 1024, 2048):
            with self.subTest(nodes=n):
                labels = np.full(n, -1, dtype=np.int64)
                labels[0], labels[n//3], labels[n//2], labels[-1] = (
                    low, -10, 0, high)
                self.fit_check(n, labels)

    def test_many_labeled_and_all_labeled(self):
        for density in (0.5, 0.9, 1.0):
            with self.subTest(density=density):
                n = 1024
                labels = np.full(n, -1, dtype=np.int64)
                count = int(n * density)
                labels[:count] = np.arange(count) % 10 * 19 + 17
                self.fit_check(n, labels)

    def test_all_unknown_and_single_class_keep_original_nonfit(self):
        n = 1024
        m = PCC(impl="numpy")
        m.set_graph(ring(n), np.full(n, 2, dtype=np.int64), k_nn=2)
        for seed in (None, 37059):
            for classes in (0, 1):
                labels = np.full(n, -1, dtype=np.int64)
                if classes:
                    labels[0] = 37071
                with self.subTest(classes=classes, seed=seed):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        actual = m.fit_predict(labels, max_iter=0, seed=seed)
                    self.assertIs(actual, labels)
                    self.assertIsNone(m.part)
                    self.assertEqual(m.c, classes)
                    self.assertNotIn("mapped_labels", m.__dict__)

    def test_native_backend_keeps_mapping_and_predictions_for_sparse_inputs(self):
        n = 1024
        labels = np.full(n, -1, dtype=np.int64)
        labels[[0, 7, 11, 208, 360, 551]] = [99, 99, 99, -7, -7, -7]
        for backend in ("numpy", "numba", "cython"):
            with self.subTest(backend=backend):
                self.fit_check(n, labels, backend=backend, max_iter=5)


if __name__ == "__main__":
    unittest.main()
