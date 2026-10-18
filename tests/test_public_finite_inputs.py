"""Reject nonfinite public inputs before conversion/dispatch; retain finite coercions."""
from decimal import Decimal
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from numba import njit
from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_graph import build_knn_graph


@njit
def seed_numba():
    np.random.seed(19)


@njit
def draw_numba():
    return np.random.random()


def model(backend):
    m = PCC(impl=backend)
    expected = {'numpy': 'pcc_numpy', 'numba': 'pcc_numba', 'cython': 'pcc_step'}
    if m._get_backend_fn().__module__ != expected[backend]:
        raise AssertionError('public finite-input tests require all three backends')
    m.set_graph(np.array([[2], [2], [0]]), np.ones(3, dtype=np.int64), k_nn=1)
    return m


class PublicFiniteInputsTests(unittest.TestCase):
    def reject_without_dispatch(self, m, labels, message, **parameters):
        before = dict(m.__dict__)
        np.random.seed(19)
        seed_numba()
        expected_numpy = np.random.get_state()
        with patch.object(m, '_get_backend_fn') as dispatch:
            with self.assertRaisesRegex(ValueError, message):
                m.fit_predict(labels, **parameters)
            dispatch.assert_not_called()
        self.assertEqual(set(m.__dict__), set(before))
        for k, v in before.items():
            self.assertIs(m.__dict__[k], v, k)
        actual_numpy = np.random.get_state()
        np.testing.assert_array_equal(actual_numpy[1], expected_numpy[1])
        self.assertEqual(actual_numpy[2:], expected_numpy[2:])
        actual_numba = draw_numba()
        seed_numba()
        self.assertEqual(actual_numba, draw_numba())

    def test_nonfinite_fit_parameters_fail_before_mutation_or_rng(self):
        for backend in ('numpy', 'numba', 'cython'):
            for name in ('p_grd', 'delta_v', 'deltap', 'dexp', 'max_iter', 'es_chk', 'early_stop'):
                for value in (np.nan, np.inf, -np.inf, Decimal('NaN'), Decimal('Infinity')):
                    with self.subTest(backend=backend, parameter=name, value=value):
                        self.reject_without_dispatch(model(backend), [0, 1, -1], name, **{name: value})
            for name in ('p_grd', 'delta_v', 'deltap', 'dexp'):
                for value in ('nan', 'inf', '-inf'):
                    with self.subTest(backend=backend, parameter=name, value=value):
                        self.reject_without_dispatch(model(backend), [0, 1, -1], name, **{name: value})

    def test_nonfinite_labels_fail_before_integer_cast(self):
        for backend in ('numpy', 'numba', 'cython'):
            for dtype in (np.float32, np.float64, np.complex128, object):
                for value in (np.nan, np.inf, -np.inf):
                    labels = np.array([0, 1, value], dtype=dtype)
                    with self.subTest(backend=backend, dtype=dtype, value=value):
                        with warnings.catch_warnings():
                            warnings.simplefilter('error')
                            self.reject_without_dispatch(model(backend), labels, 'labels', max_iter=0)
            self.reject_without_dispatch(model(backend), [0, -1, np.nan], 'labels', max_iter=0)
            self.reject_without_dispatch(model(backend), np.array([0, 1, Decimal('NaN')], dtype=object), 'labels')

    def test_one_class_early_return_does_not_bypass_finite_contract(self):
        for backend in ('numpy', 'numba', 'cython'):
            self.reject_without_dispatch(model(backend), [0, -1, -1], 'delta_v', delta_v=np.nan)

    def test_rejected_nonfinite_input_leaves_an_existing_fit_untouched(self):
        for backend in ('numpy', 'numba', 'cython'):
            m = model(backend)
            m.fit_predict([0, 1, -1], max_iter=3, early_stop=False)
            self.reject_without_dispatch(m, [0, 1, -1], 'delta_v', delta_v=np.nan)

    def test_set_graph_checks_active_indices_and_degrees_before_cast(self):
        for field in ('neib_list', 'neib_qt', 'k_nn'):
            for value in (np.nan, np.inf, -np.inf):
                m = model('numpy')
                before = dict(m.__dict__)
                neighbors = np.array([[2.], [2.], [0.]])
                degrees = np.ones(3)
                k = 1
                if field == 'neib_list':
                    neighbors[0, 0] = value
                elif field == 'neib_qt':
                    degrees[0] = value
                else:
                    k = value
                with self.subTest(field=field, value=value), warnings.catch_warnings():
                    warnings.simplefilter('error')
                    with self.assertRaisesRegex(ValueError, field):
                        m.set_graph(neighbors, degrees, k_nn=k)
                for key, original in before.items():
                    self.assertIs(m.__dict__[key], original)

    def test_undefined_neighbor_padding_remains_ignored(self):
        m = PCC(impl='numpy')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            m.set_graph(np.array([[1., np.nan], [0., np.inf]]), np.ones(2), k_nn=1)
        np.testing.assert_array_equal(m.neib_list[:, 0], [1, 0])

    def test_knn_already_rejects_nonfinite_features(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_knn_graph(np.array([[0.], [1.], [value]]), k_nn=1)

    def test_finite_coercions_determinism_and_state_remain_compatible(self):
        variants = [([0, 1, -1], {}),
                    (np.array([.9, 1.9, -1.]), dict(p_grd='0.5', delta_v='0.1', deltap='1', dexp='2', max_iter=10.9, es_chk=2000.9, early_stop=1)),
                    (np.array([0, 1, -1], dtype=object), {}),
                    (['0', '1', '-1'], {}),
                    ([-3, 5, -1], {}),
                    ([0, 1, -1], dict(p_grd=2., delta_v=-.1, deltap=2., dexp=-1., max_iter=-2, es_chk=-20))]
        for backend in ('numpy', 'numba', 'cython'):
            for labels, params in variants:
                normalized = dict(p_grd=.5, delta_v=.1, deltap=1., dexp=2., max_iter=10, early_stop=True, es_chk=2000)
                normalized.update(params)
                results = []
                for repeated in (False, True):
                    m = model(backend)
                    actual = dict(normalized)
                    if repeated:
                        actual = {k: (bool(v) if k == 'early_stop' else int(v) if k in ('max_iter', 'es_chk') else float(v)) for k, v in normalized.items()}
                    np.random.seed(19)
                    seed_numba()
                    pred = m.fit_predict(labels if not repeated else np.asarray(labels, dtype=np.int64), **actual)
                    results.append(([a.copy() for a in (pred, m.node.dominance, m.part.curnode, m.part.strength, m.part.dist_table, m.owndeg)],
                                    float(np.random.random()), float(draw_numba())))
                for a, b in zip(results[0][0], results[1][0]):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(results[0][1:], results[1][1:])


if __name__ == '__main__':
    unittest.main()
