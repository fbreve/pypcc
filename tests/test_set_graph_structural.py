"""Structural preflight for the public set_graph API."""
import unittest
import numpy as np
from pcc import ParticleCompetitionAndCooperation


class SetGraphStructuralTests(unittest.TestCase):
    def test_invalid_graph_does_not_replace_existing_state(self):
        model = ParticleCompetitionAndCooperation(impl="numpy")
        model.set_graph(np.array([[1], [0]], dtype=np.int64),
                        np.array([1, 1], dtype=np.int64))
        old_neighbors, old_degrees = model.neib_list, model.neib_qt
        old_k = model.k_nn
        cases = (
            (np.array([[1], [0]]), np.array([-1, 1])),
            (np.array([[1], [0]]), np.array([2, 1])),
            (np.array([[2], [0]]), np.array([1, 1])),
            (np.array([[-1], [0]]), np.array([1, 1])),
            (np.array([[1], [0]]), np.array([1])),
            (np.array([1, 0]), np.array([1, 1])),
            (np.array([[1], [0]]), np.array([[1], [1]])),
        )
        for neighbors, degrees in cases:
            with self.subTest(neighbors=neighbors.tolist(), degrees=degrees.tolist()):
                with self.assertRaises(ValueError):
                    model.set_graph(neighbors, degrees)
                self.assertIs(model.neib_list, old_neighbors)
                self.assertIs(model.neib_qt, old_degrees)
                self.assertEqual(model.k_nn, old_k)

    def test_padding_is_ignored_and_integer_conversion_preserved(self):
        model = ParticleCompetitionAndCooperation(impl="numpy")
        neighbors = np.array([[1.9, np.nan], [0.8, np.inf]])
        model.set_graph(neighbors, np.array([1.0, 1.0]))
        np.testing.assert_array_equal(model.neib_qt, [1, 1])
        np.testing.assert_array_equal(model.neib_list[:, 0], [1, 0])

    def test_empty_graph_supported_and_active_nonfinite_rejected(self):
        model = ParticleCompetitionAndCooperation(impl="numpy")
        model.set_graph(np.empty((0, 1), dtype=np.int64), np.empty(0, dtype=np.int64))
        self.assertEqual(model.k_nn, 0)
        self.assertEqual(model.neib_qt.size, 0)
        with self.assertRaises(ValueError):
            model.set_graph(np.array([[np.nan], [0.0]]), np.array([1, 1]))


if __name__ == "__main__":
    unittest.main()
