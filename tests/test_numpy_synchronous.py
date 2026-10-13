"""Deterministic tests for the optional NumPy update mode."""
import unittest
from unittest.mock import patch

import numpy as np

from pcc import ParticleCompetitionAndCooperation
from pcc_numpy import pcc_step_numpy, pcc_propagate_numpy


class UpdateModeAPITests(unittest.TestCase):
    def test_existing_constructor_calls_keep_sequential_default(self):
        for args in ((), ('numpy',), ('numpy', 1)):
            model = ParticleCompetitionAndCooperation(*args)
            self.assertEqual(model.update_mode, 'sequential')

    def test_synchronous_requires_explicit_numpy(self):
        for impl in ('auto', 'numba', 'cython', 'invalid', None):
            with self.subTest(impl=impl), self.assertRaisesRegex(ValueError, "requires impl='numpy'"):
                ParticleCompetitionAndCooperation(impl=impl, update_mode='synchronous')
        model = ParticleCompetitionAndCooperation(impl='numpy', update_mode='synchronous')
        self.assertEqual(model._get_backend_fn().__module__, 'pcc_numpy')

    def test_mutating_backend_cannot_silently_fall_back(self):
        model = ParticleCompetitionAndCooperation(impl='numpy', update_mode='synchronous')
        model.impl = 'auto'
        with self.assertRaisesRegex(ValueError, "requires impl='numpy'"):
            model._get_backend_fn()

    def test_invalid_mode_is_rejected(self):
        for mode in ('parallel', 'invalid', None):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ParticleCompetitionAndCooperation(impl='numpy', update_mode=mode)

    def test_default_and_explicit_sequential_have_identical_state_and_rng(self):
        snapshots = []
        for kwargs in ({}, {'update_mode': 'sequential'}):
            np.random.seed(17)
            model = ParticleCompetitionAndCooperation(impl='numpy', **kwargs)
            model.set_graph(np.array([[2, 3], [2, 3], [0, 1], [0, 1]]),
                            np.full(4, 2))
            predicted = model.fit_predict(np.array([10, 20, -1, -1]),
                                          max_iter=12, early_stop=False)
            snapshots.append((predicted, model.node.dominance, model.part.curnode,
                              model.part.strength, model.part.dist_table,
                              model.owndeg, np.random.random(8)))
        for default, explicit in zip(*snapshots):
            np.testing.assert_array_equal(default, explicit)


class SynchronousStepTests(unittest.TestCase):
    def fixture(self, classes=(0, 1), row=(.5, .5), strength=None):
        classes = np.array(classes, dtype=np.int64)
        count = classes.size
        c = len(row)
        dominance = np.vstack((np.eye(c)[classes], row))
        distance = np.full((count + 1, count), count, dtype=np.uint8)
        distance[np.arange(count), np.arange(count)] = 0
        return dict(neib_list=np.array([[count]] * count + [[0]], dtype=np.int64),
                    neib_qt=np.ones(count + 1, dtype=np.int64),
                    labels=np.r_[classes, -1], p_grd=0., delta_v=1., c=c,
                    zerovec=np.zeros(c), part_curnode=np.arange(count, dtype=np.int64),
                    part_label=classes, part_strength=np.ones(count) if strength is None
                    else np.array(strength, dtype=np.float64), dist_table=distance,
                    dominance=dominance, owndeg=np.zeros((count + 1, c)))

    def step(self, state):
        pcc_step_numpy(**state, update_mode='synchronous')

    def assert_invariants(self, state, labeled):
        dom = state['dominance']
        self.assertTrue(np.isfinite(dom).all())
        self.assertTrue(np.all((dom >= 0.) & (dom <= 1.)))
        np.testing.assert_allclose(dom.sum(axis=1), 1., atol=1e-14)
        np.testing.assert_array_equal(dom[:-1], labeled)
        self.assertTrue(np.isfinite(state['part_strength']).all())
        self.assertTrue(np.all((state['part_strength'] >= 0.) &
                              (state['part_strength'] <= 1.)))
        self.assertTrue(np.all((state['part_curnode'] >= 0) &
                              (state['part_curnode'] < len(dom))))
        self.assertEqual(state['dist_table'].dtype, np.uint8)
        np.testing.assert_array_equal(np.diag(state['dist_table'][:-1]), 0)

    def test_same_class_collision_saturates_without_overdrawing(self):
        state = self.fixture(classes=(0, 0, 0))
        self.step(state)
        np.testing.assert_array_equal(state['dominance'][-1], [1., 0.])
        np.testing.assert_array_equal(state['part_strength'], [1., 1., 1.])
        self.assert_invariants(state, np.array([[1., 0.]] * 3))

    def test_opposing_equal_influence_preserves_tie_and_moves_both_particles(self):
        state = self.fixture()
        self.step(state)
        np.testing.assert_array_equal(state['dominance'][-1], [.5, .5])
        np.testing.assert_array_equal(state['part_strength'], [.5, .5])
        np.testing.assert_array_equal(state['part_curnode'], [2, 2])
        np.testing.assert_array_equal(state['dist_table'][-1], [1, 1])
        np.testing.assert_array_equal(state['owndeg'][-1], [.5, .5])

    def test_competing_classes_share_each_donors_loss_without_self_refund(self):
        state = self.fixture(row=(.05, .35, .6), strength=(.8, .4))
        self.step(state)
        # H=(.4,.2,0), loss=(.05,.35,.6), gains=(.75,.25,0).
        np.testing.assert_allclose(state['dominance'][-1], [.75, .25, 0.], atol=1e-14)
        np.testing.assert_allclose(state['part_strength'], [.75, .25])
        np.testing.assert_array_equal(state['part_curnode'], [2, 1])
        # A rejected visit still updates distance and own-degree.
        np.testing.assert_array_equal(state['dist_table'][-1], [1, 1])
        np.testing.assert_allclose(state['owndeg'][-1], [.75, .25, 0.])

    def test_single_visit_matches_sequential_for_all_state_updates(self):
        states = []
        for mode in ('sequential', 'synchronous'):
            state = self.fixture(classes=(1,), row=(.2, .3, .5), strength=(.8,))
            state.update(delta_v=.3, deltap=.5)
            pcc_step_numpy(**state, update_mode=mode)
            states.append(state)
        for key in ('dominance', 'part_strength', 'part_curnode', 'dist_table', 'owndeg'):
            np.testing.assert_allclose(states[0][key], states[1][key], atol=1e-14)

    def test_particle_permutation_does_not_change_simultaneous_aggregation(self):
        reference = self.fixture(classes=(0, 1, 2, 0), row=(.1, .3, .6),
                                 strength=(.8, .2, .6, .4))
        permuted = {k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in reference.items()}
        order = np.array([2, 0, 3, 1])
        for key in ('part_curnode', 'part_label', 'part_strength'):
            permuted[key] = permuted[key][order]
        permuted['dist_table'] = permuted['dist_table'][:, order]
        self.step(reference)
        self.step(permuted)
        for key in ('dominance', 'owndeg'):
            np.testing.assert_allclose(permuted[key], reference[key], atol=1e-14)
        for key in ('part_curnode', 'part_strength'):
            np.testing.assert_allclose(permuted[key], reference[key][order], atol=1e-14)
        np.testing.assert_array_equal(permuted['dist_table'], reference['dist_table'][:, order])

    def test_all_greedy_destinations_use_initial_dominance(self):
        state = self.fixture()
        state.update(neib_list=np.array([[2, -1], [2, 3], [0, -1], [1, -1]]),
                     neib_qt=np.array([1, 2, 1, 1]), labels=np.array([0, 1, -1, -1]),
                     dominance=np.array([[1., 0.], [0., 1.], [.5, .5], [.5, .5]]),
                     dist_table=np.array([[0, 3], [3, 0], [3, 3], [3, 3]], dtype=np.uint8),
                     owndeg=np.zeros((4, 2)), p_grd=1.)
        with patch('pcc_numpy.np.random.random', side_effect=lambda size: np.full(size, .1)):
            self.step(state)
        # Sequential processing would erase class 1 at node 2 before its selection.
        np.testing.assert_array_equal(state['part_curnode'], [2, 2])
        np.testing.assert_array_equal(state['dominance'][2:], [[.5, .5], [.5, .5]])
        np.testing.assert_array_equal(state['owndeg'], 0.)

    def test_labeled_target_keeps_dominance_and_rejects_other_class(self):
        state = self.fixture()
        state['labels'][-1] = 0
        state['dominance'][-1] = [1., 0.]
        self.step(state)
        np.testing.assert_array_equal(state['dominance'][-1], [1., 0.])
        np.testing.assert_array_equal(state['part_curnode'], [2, 1])
        np.testing.assert_array_equal(state['part_strength'], [1., 0.])

    def test_zero_roulette_threshold_skips_zero_weight_invalid_neighbor(self):
        state = self.fixture(classes=(0,))
        state.update(neib_list=np.array([[-1, 1], [0, -1]]),
                     neib_qt=np.array([2, 1]), p_grd=1.)
        with patch('pcc_numpy.np.random.random', side_effect=lambda size: np.zeros(size)):
            self.step(state)
        np.testing.assert_array_equal(state['part_curnode'], [1])
        np.testing.assert_array_equal(state['dominance'][-1], [1., 0.])

    def test_zero_greedy_weight_falls_back_to_random_and_records_own_degree(self):
        state = self.fixture(classes=(0,), row=(0., 1.))
        state.update(p_grd=1., delta_v=.25)
        self.step(state)
        np.testing.assert_allclose(state['dominance'][-1], [.25, .75])
        np.testing.assert_allclose(state['owndeg'][-1], [.25, 0.])
        np.testing.assert_array_equal(state['part_curnode'], [0])

    def test_inactive_particles_and_invalid_neighbors_do_not_modify_state(self):
        for kind in ('isolated', 'position', 'class', 'neighbor', 'empty'):
            with self.subTest(kind=kind):
                state = self.fixture()
                state['p_grd'] = 1.
                if kind == 'isolated':
                    state['neib_qt'][:] = 0
                elif kind == 'position':
                    state['part_curnode'][:] = 99
                elif kind == 'class':
                    state['part_label'][:] = 99
                elif kind == 'neighbor':
                    state['neib_list'][:] = 99
                else:
                    for key in ('part_label', 'part_strength', 'part_curnode'):
                        state[key] = state[key][:0]
                    state['dist_table'] = state['dist_table'][:, :0]
                before = {k: v.copy() for k, v in state.items() if isinstance(v, np.ndarray)}
                self.step(state)
                for key, value in before.items():
                    np.testing.assert_array_equal(state[key], value)

    def test_distance_255_does_not_wrap(self):
        state = self.fixture()
        state['dist_table'][:] = 255
        self.step(state)
        np.testing.assert_array_equal(state['dist_table'], 255)

    def test_zero_influence_keeps_dominance(self):
        state = self.fixture(strength=(0., 0.))
        self.step(state)
        np.testing.assert_array_equal(state['dominance'][-1], [.5, .5])

    def test_repeated_collisions_preserve_all_invariants(self):
        state = self.fixture(classes=tuple(np.arange(24) % 3), row=(.1, .2, .7))
        state.update(p_grd=.5, delta_v=.1, deltap=.5)
        labeled = state['dominance'][:-1].copy()
        np.random.seed(42)
        for _ in range(100):
            self.step(state)
            self.assert_invariants(state, labeled)

    def test_low_level_invalid_modes_are_rejected_even_without_iterations(self):
        state = self.fixture()
        with self.assertRaises(ValueError):
            pcc_step_numpy(**state, update_mode='parallel')
        with self.assertRaises(ValueError):
            pcc_propagate_numpy(**state, deltap=1., dexp=2., dom_row=None, reduc=None,
                                dom_list=None, dist_list=None, prob=None, slices=None,
                                dist_weights=None, max_iter=0, early_stop=False,
                                es_chk=2000, stop_max=1, update_mode='parallel')

    def test_high_level_synchronous_executes_and_preserves_external_labels(self):
        model = ParticleCompetitionAndCooperation(impl='numpy', update_mode='synchronous')
        model.set_graph(np.array([[2], [2], [0]]), np.ones(3, dtype=np.int64))
        prediction = model.fit_predict(np.array([10, 20, -1]), p_grd=0., delta_v=1.,
                                       max_iter=1, early_stop=False)
        np.testing.assert_array_equal(prediction[:2], [10, 20])
        np.testing.assert_array_equal(model.node.dominance[-1], [.5, .5])


if __name__ == '__main__':
    unittest.main()
