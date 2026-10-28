"""Public API opt-in is NumPy-only and preserves the legacy default."""
import unittest
import numpy as np
from pcc import ParticleCompetitionAndCooperation


class PublicDominanceStoppingTests(unittest.TestCase):
    def model(self, impl='numpy'):
        model = ParticleCompetitionAndCooperation(impl=impl)
        model.set_graph(np.array([[1], [0]], dtype=np.int64),
                        np.ones(2, dtype=np.int64))
        return model

    def test_default_and_explicit_legacy_match(self):
        labels = np.array([0, 1], dtype=np.int64)
        a, b = self.model(), self.model()
        first = a.fit_predict(labels, max_iter=15, seed=31)
        second = b.fit_predict(labels, max_iter=15, seed=31,
                               stop_criterion='potential')
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(a.node.dominance, b.node.dominance)

    def test_experimental_option_is_accessible(self):
        labels = np.array([0, 1], dtype=np.int64)
        model = self.model()
        result = model.fit_predict(labels, max_iter=15, seed=31,
                                   stop_criterion='dominance_change',
                                   dominance_tol=1e-10)
        self.assertEqual(result.shape, labels.shape)

    def test_invalid_option_rejected_before_fit(self):
        labels = np.array([0, 1], dtype=np.int64)
        for kwargs in ({'stop_criterion': 'bad'},
                       {'stop_criterion': 'dominance_change', 'dominance_tol': -1},):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model().fit_predict(labels, **kwargs)
        with self.assertRaisesRegex(ValueError, 'requires impl'):
            self.model('numba').fit_predict(labels, stop_criterion='dominance_change')


if __name__ == '__main__':
    unittest.main()
