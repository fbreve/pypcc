"""Backend resolution failures must not destroy an earlier fit."""
import unittest
from unittest.mock import patch

import numpy as np

from pcc import ParticleCompetitionAndCooperation


class FitBackendPreflightTests(unittest.TestCase):
    def test_backend_resolution_failure_preserves_previous_fit(self):
        model = ParticleCompetitionAndCooperation(impl="numpy")
        model.set_graph(
            np.array([[1], [0], [3], [2]], dtype=np.int64),
            np.ones(4, dtype=np.int64),
        )
        labels = np.array([0, -1, 1, -1], dtype=np.int64)
        model.fit_predict(labels, max_iter=0, early_stop=False)
        previous_labels = model.labels
        previous_part = model.part
        previous_node = model.node
        previous_dominance = model.node.dominance.copy()

        with patch.object(model, "_get_backend_fn", side_effect=RuntimeError("backend unavailable")):
            with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
                model.fit_predict(labels, max_iter=1)

        self.assertIs(model.labels, previous_labels)
        self.assertIs(model.part, previous_part)
        self.assertIs(model.node, previous_node)
        np.testing.assert_array_equal(model.node.dominance, previous_dominance)

    def test_one_class_bypasses_backend_resolution(self):
        model = ParticleCompetitionAndCooperation(impl="numpy")
        model.set_graph(
            np.array([[1], [0]], dtype=np.int64),
            np.ones(2, dtype=np.int64),
        )
        labels = np.array([0, -1], dtype=np.int64)
        with patch.object(model, "_get_backend_fn", side_effect=RuntimeError("backend unavailable")):
            with self.assertWarns(UserWarning):
                result = model.fit_predict(labels)
        self.assertIs(result, labels)


if __name__ == "__main__":
    unittest.main()
