"""Characterize RNG behavior when fit fails after drawing randomness.

The model state is rolled back; NumPy's process-global RNG is not.
"""
import unittest
from unittest.mock import patch

import numpy as np

from pcc import ParticleCompetitionAndCooperation


class FitRngFailureTests(unittest.TestCase):
    def setUp(self):
        self.model = ParticleCompetitionAndCooperation(impl="numpy")
        self.model.set_graph(
            np.array([[1], [0], [3], [2]], dtype=np.int64),
            np.ones(4, dtype=np.int64),
        )
        self.labels = np.array([0, -1, 1, -1], dtype=np.int64)
        self.model.fit_predict(self.labels, max_iter=0, early_stop=False)

    def test_preflight_failure_does_not_consume_numpy_rng(self):
        np.random.seed(217)
        expected = np.random.random()
        np.random.seed(217)
        with patch.object(self.model, "_get_backend_fn", side_effect=RuntimeError("backend unavailable")):
            with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
                self.model.fit_predict(self.labels, max_iter=1)
        self.assertEqual(np.random.random(), expected)

    def test_propagation_failure_does_consume_global_numpy_rng(self):
        previous_node = self.model.node
        np.random.seed(217)
        expected_first = np.random.random()
        expected_second = np.random.random()
        np.random.seed(217)

        def consume_and_fail():
            self.assertEqual(np.random.random(), expected_first)
            raise RuntimeError("simulated failure after RNG draw")

        with patch.object(
            self.model,
            "_ParticleCompetitionAndCooperation__labelPropagation",
            side_effect=consume_and_fail,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                self.model.fit_predict(self.labels, max_iter=1)
        self.assertIs(self.model.node, previous_node)
        self.assertEqual(np.random.random(), expected_second)


    def test_seeded_propagation_failure_restores_model_not_numpy_rng(self):
        previous_node = self.model.node
        previous_part = self.model.part
        previous_dominance = previous_node.dominance.copy()
        seed = 371
        np.random.seed(seed)
        first = np.random.random()
        second = np.random.random()

        def consume_and_fail(*args, **kwargs):
            self.assertEqual(kwargs.get("seed"), seed)
            # The real NumPy backend applies the explicit seed immediately
            # before propagation. Simulate its first random draw and failure.
            np.random.seed(kwargs["seed"])
            self.assertEqual(np.random.random(), first)
            raise RuntimeError("seeded propagation failed")

        with patch.object(self.model, "_get_backend_fn",
                          return_value=consume_and_fail):
            with self.assertRaisesRegex(RuntimeError, "seeded propagation failed"):
                self.model.fit_predict(self.labels, max_iter=1,
                                       early_stop=False, seed=seed)

        self.assertIs(self.model.node, previous_node)
        self.assertIs(self.model.part, previous_part)
        np.testing.assert_array_equal(self.model.node.dominance,
                                      previous_dominance)
        self.assertEqual(np.random.random(), second)

    def test_seeded_preflight_failure_does_not_reset_numpy_rng(self):
        np.random.seed(217)
        expected = np.random.random()
        np.random.seed(217)
        previous_node = self.model.node
        with patch.object(self.model, "_get_backend_fn",
                          side_effect=RuntimeError("backend unavailable")):
            with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
                self.model.fit_predict(self.labels, max_iter=1, seed=371)
        self.assertIs(self.model.node, previous_node)
        self.assertEqual(np.random.random(), expected)


if __name__ == "__main__":
    unittest.main()
