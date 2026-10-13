"""Deterministic safety checks for the sequential backend entry points."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from numba import njit
import pcc_numba
from pcc_numpy import pcc_step_numpy, pcc_propagate_numpy
from pcc import ParticleCompetitionAndCooperation

try:
    import pcc_step
except ImportError:
    pcc_step = None


def state(degree=1):
    n = max(3, degree + 1)
    edges = np.full((n, degree), 2, dtype=np.int64)
    edges[0] = np.arange(1, degree + 1) if degree > 1 else [2]
    degrees = np.ones(n, dtype=np.int64)
    degrees[0] = degree
    dominance = np.full((n, 2), .5)
    if degree > 1:
        dominance[1:] = [0., 1.]
        dominance[-1] = [1., 0.]
    distance = np.full((n, 1), 255, dtype=np.uint8)
    distance[0] = 0
    return [edges, degrees, np.full(n, -1, dtype=np.int64), 1., .1, 2,
            np.zeros(2), np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64), np.ones(1), distance,
            dominance, np.zeros((n, 2)), 1., 2.]


def buffers(a):
    return [np.empty(2), np.empty(2)] + [np.empty(max(1, int(a[1].max())))
            for _ in range(4)] + [1. / (np.arange(257) + 1.) ** 2]


def call_step(backend, a, **kwargs):
    if backend == 'numpy':
        return pcc_step_numpy(*a, **kwargs)
    if backend == 'numba':
        return pcc_numba.pcc_step_numba(*a, **kwargs)
    return pcc_step.pcc_step(*a, **kwargs)


@njit
def seeded_random(seed):
    np.random.seed(seed)
    return np.random.random()


@njit
def next_random():
    return np.random.random()


class SequentialSafetyTests(unittest.TestCase):
    def backends(self):
        return ['numpy', 'numba'] + (['cython'] if pcc_step is not None else [])

    def test_degree_33_and_64_with_missing_buffers(self):
        for degree in (33, 64):
            expected = state(degree)
            call_step('numpy', expected)
            for backend in self.backends():
                with self.subTest(degree=degree, backend=backend):
                    a = state(degree)
                    call_step(backend, a)
                    for index in (7, 9, 10, 11, 12):
                        np.testing.assert_allclose(a[index], expected[index], atol=1e-14)
                    self.assertEqual(a[7][0], degree)

    def test_numba_allocates_each_missing_buffer_independently(self):
        a = state(33)
        call_step('numba', a, dom_row=np.empty(2), prob=np.empty(33))
        self.assertEqual(a[7][0], 33)

    def test_invalid_neighbors_are_ignored_without_accessing_node_state(self):
        invalid_nodes = (-1, -3, 3, np.iinfo(np.int64).min,
                         np.iinfo(np.int64).max)
        for backend in self.backends():
            for greedy in (0., 1.):
                for target in invalid_nodes:
                    with self.subTest(backend=backend, greedy=greedy, target=target):
                        a = state()
                        a[0][0, 0] = target
                        a[3] = greedy
                        before = copy.deepcopy(a)
                        call_step(backend, a)
                        for index in (7, 9, 10, 11, 12):
                            np.testing.assert_array_equal(a[index], before[index])

    def test_greedy_walk_masks_invalid_neighbors_at_degree_33(self):
        for backend in self.backends():
            with self.subTest(backend=backend):
                a = state()
                a[0] = np.full((3, 33), -1, dtype=np.int64)
                a[0][0, 1::2] = np.iinfo(np.int64).max
                a[0][0, -1] = 2
                a[1][0] = 33
                expected = state()
                call_step('numpy', expected)
                call_step(backend, a)
                for index in (7, 9, 10, 11, 12):
                    np.testing.assert_allclose(a[index], expected[index], atol=1e-14)

    def test_cython_allocates_each_missing_buffer_independently(self):
        if pcc_step is None:
            self.skipTest('Cython extension unavailable')
        for name in ('prob', 'slices'):
            a = state(33)
            call_step('cython', a, **{name: np.empty(33)})
            self.assertEqual(a[7][0], 33)

    def test_short_or_wrong_rank_step_buffers_are_rejected(self):
        for backend in self.backends():
            if backend == 'numpy':
                continue
            names = ['prob', 'slices', 'dist_weights']
            if backend == 'numba':
                names += ['dom_row', 'reduc', 'dom_list', 'dist_list']
            for name in names:
                for value in (np.empty(0), np.empty((1, 1))):
                    with self.subTest(backend=backend, buffer=name, shape=value.shape):
                        a = state(33)
                        before = copy.deepcopy(a)
                        with self.assertRaises(ValueError):
                            call_step(backend, a, **{name: value})
                        np.testing.assert_array_equal(a[11], before[11])

    def test_malformed_state_is_rejected_before_compiled_step(self):
        mutations = [lambda a: a.__setitem__(1, np.array([34, 1, 1], dtype=np.int64)),
                     lambda a: a.__setitem__(1, np.empty(0, dtype=np.int64)),
                     lambda a: a.__setitem__(2, np.empty(0, dtype=np.int64)),
                     lambda a: a.__setitem__(6, np.empty(0)),
                     lambda a: a.__setitem__(7, np.zeros((1, 1), dtype=np.int64)),
                     lambda a: a.__setitem__(8, np.empty(0, dtype=np.int64)),
                     lambda a: a.__setitem__(9, np.empty(0)),
                     lambda a: a.__setitem__(10, np.empty((3, 0), dtype=np.uint8)),
                     lambda a: a.__setitem__(11, np.empty((3, 1))),
                     lambda a: a.__setitem__(12, np.empty((3, 1))),
                     lambda a: a.__setitem__(5, 1)]
        for backend in self.backends():
            if backend == 'numpy':
                continue
            for i, mutate in enumerate(mutations):
                with self.subTest(backend=backend, mutation=i):
                    a = state()
                    mutate(a)
                    with self.assertRaises(ValueError):
                        call_step(backend, a)

    def test_propagation_validates_all_buffers_and_state(self):
        for backend in self.backends():
            if backend == 'numpy':
                continue
            fn = pcc_numba.pcc_propagate_numba if backend == 'numba' else pcc_step.pcc_propagate
            for index in range(7):
                with self.subTest(backend=backend, buffer=index):
                    a = state(33)
                    b = buffers(a)
                    b[index] = np.empty(0)
                    with self.assertRaises(ValueError):
                        fn(*a, *b, 1, False, 2000, 1)
            a = state()
            b = buffers(a)
            a[10] = np.empty((0, 1), dtype=np.uint8)
            with self.assertRaises(ValueError):
                fn(*a, *b, 1, False, 2000, 1)

    def test_invalid_class_does_not_consume_numba_rng(self):
        for label in (-1, 2):
            a = state()
            a[8][0] = label
            expected = seeded_random(42)
            # Reset without consuming the first draw.
            @njit
            def seed():
                np.random.seed(42)
            seed()
            before = copy.deepcopy(a)
            call_step('numba', a)
            self.assertEqual(next_random(), expected)
            for index in (7, 9, 10, 11, 12):
                np.testing.assert_array_equal(a[index], before[index])

    def test_invalid_classes_leave_all_backend_states_unchanged(self):
        for backend in self.backends():
            for label in (-1, 2):
                with self.subTest(backend=backend, label=label):
                    a = state()
                    a[8][0] = label
                    before = copy.deepcopy(a)
                    call_step(backend, a)
                    for index in (7, 9, 10, 11, 12):
                        np.testing.assert_array_equal(a[index], before[index])

    def test_valid_numba_rng_consumption_is_unchanged(self):
        @njit
        def reference_draw(greedy):
            np.random.seed(123)
            if np.random.random() < greedy:
                np.random.uniform(0., 1.)
            else:
                np.random.randint(1)
            return np.random.random()

        @njit
        def seed():
            np.random.seed(123)

        for greedy in (0., 1., .5):
            a = state()
            a[3] = greedy
            expected_draw = reference_draw(greedy)
            seed()
            call_step('numba', a)
            self.assertEqual(next_random(), expected_draw)
            expected = state()
            expected[3] = float(seeded_random(123) < greedy)
            call_step('numpy', expected)
            for index in (7, 9, 10, 11, 12):
                np.testing.assert_allclose(a[index], expected[index], atol=1e-14)

    def test_isolated_graph_uses_plateau_stop_without_division(self):
        for backend in self.backends():
            for early in (True, False):
                with self.subTest(backend=backend, early=early):
                    model = ParticleCompetitionAndCooperation(impl=backend)
                    model.set_graph(np.empty((3, 0), dtype=np.int64),
                                    np.zeros(3, dtype=np.int64))
                    labels = np.array([0, 1, -1])
                    fn = model._get_backend_fn()
                    seen = []
                    def observe(*args):
                        seen.append(args[-1])
                        return fn(*args)
                    with patch.object(model, '_get_backend_fn', return_value=observe):
                        result = model.fit_predict(labels, max_iter=30, early_stop=early)
                    self.assertEqual(seen, [1])
                    np.testing.assert_array_equal(result[:2], labels[:2])
                    np.testing.assert_array_equal(model.part.curnode, [0, 1])
                    np.testing.assert_array_equal(model.part.strength, [1., 1.])
                    np.testing.assert_allclose(model.node.dominance.sum(1), 1.)

    def test_plateau_stopping_and_max_iter_are_unchanged(self):
        for backend in self.backends():
            fn = {'numpy': pcc_propagate_numpy,
                  'numba': pcc_numba.pcc_propagate_numba,
                  'cython': pcc_step.pcc_propagate if pcc_step else None}[backend]
            for cap, early, expected in ((100, True, 21), (12, True, 12), (30, False, 30)):
                a = state()
                a[0][:] = 0
                a[2][0] = 0
                a[11][0] = [1., 0.]
                a[3] = 0.
                fn(*a, *buffers(a), cap, early, 2000, 1)
                self.assertEqual(a[12][0, 0], expected)


if __name__ == '__main__':
    unittest.main()
