"""Failed propagation must not replace the last successful fit."""
import unittest
from unittest.mock import patch

import numpy as np

from pcc import ParticleCompetitionAndCooperation


class FitExceptionRollbackTests(unittest.TestCase):
    def test_propagation_exception_restores_previous_state(self):
        model = ParticleCompetitionAndCooperation(impl="numpy")
        model.set_graph(
            np.array([[1], [0], [3], [2]], dtype=np.int64),
            np.ones(4, dtype=np.int64),
        )
        labels = np.array([0, -1, 1, -1], dtype=np.int64)
        model.fit_predict(labels, max_iter=0, early_stop=False)
        before = model.__dict__.copy()
        before_dom = model.node.dominance.copy()

        def fail_after_initialization():
            self.assertIsNot(model.node, before["node"])
            self.assertIsNot(model.part, before["part"])
            raise RuntimeError("simulated propagation failure")

        with patch.object(model, "_ParticleCompetitionAndCooperation__labelPropagation",
                          side_effect=fail_after_initialization):
            with self.assertRaisesRegex(RuntimeError, "simulated propagation failure"):
                model.fit_predict(labels, max_iter=1, early_stop=False)

        self.assertEqual(set(model.__dict__), set(before))
        for key, value in before.items():
            self.assertIs(model.__dict__[key], value, key)
        np.testing.assert_array_equal(model.node.dominance, before_dom)


if __name__ == "__main__":
    unittest.main()
