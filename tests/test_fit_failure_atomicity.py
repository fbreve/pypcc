"""Invalid fit arguments must not replace an existing fitted state."""
import unittest

import numpy as np

from pcc import ParticleCompetitionAndCooperation


class FitFailureAtomicityTests(unittest.TestCase):
    def test_invalid_inputs_preserve_previous_fit(self):
        neighbors = np.array([[1], [0], [3], [2]], dtype=np.int64)
        degrees = np.ones(4, dtype=np.int64)
        labels = np.array([0, -1, 1, -1], dtype=np.int64)
        model = ParticleCompetitionAndCooperation(impl="numpy")
        model.set_graph(neighbors, degrees)
        model.fit_predict(labels, max_iter=0, early_stop=False)
        original_part = model.part
        original_node = model.node
        original_labels = model.labels
        original_dominance = model.node.dominance.copy()

        bad_calls = [
            {"labels": np.array([0, -1, 1])},
            {"labels": labels, "p_grd": float("nan")},
            {"labels": labels, "delta_v": float("inf")},
            {"labels": labels, "deltap": float("-inf")},
            {"labels": labels, "dexp": float("nan")},
            {"labels": labels, "max_iter": float("inf")},
            {"labels": labels, "es_chk": float("nan")},
            {"labels": np.array([0, -1, 2**63, -1], dtype=object)},
        ]
        for kwargs in bad_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    model.fit_predict(**kwargs)
                self.assertIs(model.part, original_part)
                self.assertIs(model.node, original_node)
                self.assertIs(model.labels, original_labels)
                np.testing.assert_array_equal(model.node.dominance, original_dominance)


if __name__ == "__main__":
    unittest.main()
