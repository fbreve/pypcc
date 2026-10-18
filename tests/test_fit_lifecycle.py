"""Lifecycle regressions use only tiny synthetic graphs."""
import unittest
from unittest.mock import patch
import numpy as np
from numba import njit
from pcc import ParticleCompetitionAndCooperation as PCC


@njit
def seed_numba():
    np.random.seed(31)


@njit
def draw_numba():
    return np.random.random(4)


class FitLifecycleTests(unittest.TestCase):
    backends = ('numpy', 'numba', 'cython')
    cleared = ('part', 'node', 'label_to_idx', 'idx_to_label', 'zerovec', 'owndeg',
               'p_grd', 'delta_v', 'deltap', 'dexp', 'max_iter', 'early_stop', 'es_chk')
    absent = ('mapped_labels', 'dist_weights', 'buf_dom_row', 'buf_reduc',
              'buf_dom_list', 'buf_dist_list', 'buf_prob', 'buf_slices')

    def model(self, backend):
        m = PCC(impl=backend)
        m.set_graph(np.array([[1, 3], [0, 2], [1, 3], [0, 2]]), np.full(4, 2))
        self.assertEqual(m._get_backend_fn().__module__,
                         {'numpy': 'pcc_numpy', 'numba': 'pcc_numba', 'cython': 'pcc_step'}[backend])
        return m

    def skip(self, m, labels):
        graph = (m.neib_list, m.neib_qt, m.k_nn, m.data)
        np.random.seed(31)
        expected_numpy = np.random.random(4)
        np.random.seed(31)
        seed_numba()
        expected_numba = draw_numba()
        seed_numba()
        with patch.object(m, '_get_backend_fn', side_effect=AssertionError('unexpected dispatch')):
            with self.assertWarnsRegex(UserWarning, 'Returning original labels'):
                returned = m.fit_predict(labels, max_iter=17, delta_v=.3)
        self.assertIs(returned, labels)
        np.testing.assert_array_equal(m.labels, labels)
        known = np.unique(labels)
        known = known[known != -1]
        np.testing.assert_array_equal(m.unique_labels, known)
        self.assertEqual(m.c, len(known))
        for name in self.cleared:
            self.assertIsNone(getattr(m, name), name)
        for name in self.absent:
            self.assertFalse(hasattr(m, name), name)
        for actual, expected in zip((m.neib_list, m.neib_qt, m.k_nn, m.data), graph):
            self.assertIs(actual, expected)
        np.testing.assert_array_equal(np.random.random(4), expected_numpy)
        np.testing.assert_array_equal(draw_numba(), expected_numba)

    def fit(self, m, labels):
        info = {}
        fn = m._get_backend_fn()
        def capture(*args):
            info.update(fn(*args, return_info=True))
        np.random.seed(31)
        seed_numba()
        with patch.object(m, '_get_backend_fn', return_value=capture):
            prediction = m.fit_predict(labels, max_iter=35, early_stop=True, es_chk=1)
        arrays = [a.copy() for a in (prediction, m.node.dominance, m.node.label,
                  m.part.homenode, m.part.curnode, m.part.label, m.part.strength,
                  m.part.dist_table, m.owndeg, m.mapped_labels, m.dist_weights)]
        arrays.extend([np.random.random(4), draw_numba()])
        return arrays, info

    def test_first_degenerate_call_zero_or_one_class(self):
        for be in self.backends:
            for labels in ([-1]*4, [7, -1, -1, -1], np.full(4, 7)):
                with self.subTest(backend=be, labels=labels):
                    self.skip(self.model(be), labels)

    def test_reuse_clears_previous_fit_without_mutating_external_references(self):
        for be in self.backends:
            with self.subTest(backend=be):
                m = self.model(be)
                self.fit(m, [7, -1, 12, -1])
                node, part = m.node, m.part
                dominance, positions = node.dominance.copy(), part.curnode.copy()
                self.skip(m, [7, -1, -1, -1])
                np.testing.assert_array_equal(node.dominance, dominance)
                np.testing.assert_array_equal(part.curnode, positions)

    def test_repeated_degenerate_calls_are_history_independent(self):
        for be in self.backends:
            with self.subTest(backend=be):
                reused, fresh = self.model(be), self.model(be)
                self.fit(reused, [7, -1, 12, -1])
                for labels in ([7, -1, -1, -1], [-1]*4, [12]*4, [-1]*4):
                    self.skip(reused, labels)
                    self.skip(fresh, labels)
                    self.assertEqual(set(reused.__dict__), set(fresh.__dict__))

    def test_recovery_matches_fresh_and_repeated_valid_fits(self):
        for be in self.backends:
            with self.subTest(backend=be):
                reused, fresh = self.model(be), self.model(be)
                self.fit(reused, [7, -1, 12, -1])
                self.skip(reused, [-1]*4)
                for labels in ([2, -1, 5, 9], [4, -1, 8, -1]):
                    actual, info = self.fit(reused, labels)
                    expected, expected_info = self.fit(fresh, labels)
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a, b)
                    self.assertEqual(info, expected_info)
