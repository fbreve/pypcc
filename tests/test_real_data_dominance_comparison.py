"""Smoke tests for the real-data stopping comparison protocol."""
import unittest
import numpy as np
from experiments.compare_dominance_stopping import prepare, compare, DATASETS


class RealDataDominanceComparisonTests(unittest.TestCase):
    def test_graph_excludes_self_and_masks_labels(self):
        edges, degrees, labels, truth = prepare("iris", 11, neighbors=4)
        self.assertEqual(edges.shape, (150, 4))
        self.assertTrue(np.all(edges != np.arange(150)[:, None]))
        self.assertTrue(np.all(degrees == 4))
        self.assertEqual(np.count_nonzero(labels != -1), 6)
        np.testing.assert_array_equal(labels[labels != -1],
                                      truth[labels != -1])

    def test_reproducible_preparation(self):
        a = prepare("wine", 23, neighbors=3)
        b = prepare("wine", 23, neighbors=3)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_real_data_comparison_smoke(self):
        rows = compare("iris", 11, cap=2, neighbors=3)
        self.assertEqual([r["criterion"] for r in rows],
                         ["reference", "potential", "dominance_change",
                          "dominance_relative"])
        for row in rows:
            self.assertTrue(0 <= row["accuracy"] <= 1)
            self.assertTrue(0 <= row["reference_agreement"] <= 1)
            self.assertEqual(row["unlabeled_count"], 144)
            self.assertGreaterEqual(row["iterations"], 1)
            self.assertLessEqual(row["iterations"], 2)
            self.assertIn(row["stop_reason"], ("early_stop", "max_iter"))


if __name__ == "__main__":
    unittest.main()
