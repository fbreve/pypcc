import unittest
import numpy as np
from numba import njit
from pcc_graph import build_graph_from_edge_index
from pcc import ParticleCompetitionAndCooperation as PCC


@njit
def seed_native():
    np.random.seed(23)


@njit
def draw_native():
    return np.random.random(5)


class DuplicateEdgeTests(unittest.TestCase):
    def assert_graph(self, n, edges, expected):
        neighbors, degrees = build_graph_from_edge_index(n, edges)
        self.assertEqual(neighbors.dtype, np.int64)
        self.assertEqual(degrees.dtype, np.int64)
        self.assertEqual(degrees.tolist(), [len(row) for row in expected])
        for i, row in enumerate(expected):
            self.assertEqual(neighbors[i, :degrees[i]].tolist(), row)
            self.assertTrue(np.all(neighbors[i, degrees[i]:] == -1))
        # Exact adjacency also establishes all connected components.
        return neighbors, degrees

    def test_duplicate_multiplicities(self):
        for count in (1, 2, 127, 128, 255, 256, 512):
            with self.subTest(count=count):
                self.assert_graph(2, np.tile([[0], [1]], (1, count)), [[1], [0]])

    def test_reciprocal_edges_self_loops_and_disconnected_nodes(self):
        edges = np.array([[0, 1, 1, 2, 3, 4, 6], [1, 0, 2, 2, 4, 3, 6]])
        expected = [[1], [0, 2], [1], [4], [3], [], []]
        for count in (1, 2, 127, 128, 255, 256):
            with self.subTest(count=count):
                self.assert_graph(7, np.repeat(edges, count, axis=1), expected)

    def test_empty_edges_and_zero_nodes(self):
        self.assert_graph(3, np.empty((2, 0), dtype=np.int64), [[], [], []])
        self.assert_graph(0, np.empty((2, 0), dtype=np.int64), [])

    def test_graph_construction_does_not_consume_rng(self):
        np.random.seed(23)
        expected = np.random.random(5)
        np.random.seed(23)
        build_graph_from_edge_index(2, np.tile([[0], [1]], (1, 256)))
        np.testing.assert_array_equal(np.random.random(5), expected)

    def test_backend_state_stopping_and_rng_match_distinct_edges(self):
        edges = np.array([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 0, 7]])
        labels = [0, -1, -1, 1, -1, -1, 0, 1, -1]
        for backend in ('numpy', 'numba', 'cython'):
            reference = None
            for count in (1, 2, 127, 128, 255, 256):
                with self.subTest(backend=backend, count=count):
                    neighbors, degrees = build_graph_from_edge_index(9, np.repeat(edges, count, axis=1))
                    model = PCC(impl=backend)
                    model.set_graph(neighbors, degrees)
                    fn = model._get_backend_fn()
                    self.assertEqual(fn.__module__, {'numpy': 'pcc_numpy', 'numba': 'pcc_numba', 'cython': 'pcc_step'}[backend])
                    info = {}
                    def capture(*args):
                        info.update(fn(*args, return_info=True))
                    model._get_backend_fn = lambda: capture
                    np.random.seed(23)
                    seed_native()
                    prediction = model.fit_predict(labels, max_iter=40, early_stop=False)
                    state = [a.copy() for a in (prediction, model.node.dominance, model.part.curnode,
                             model.part.strength, model.part.dist_table, model.owndeg)]
                    state.extend([np.random.random(5), draw_native()])
                    if reference is None:
                        reference = state, info.copy()
                    else:
                        for actual, expected in zip(state, reference[0]):
                            np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(info, reference[1])
