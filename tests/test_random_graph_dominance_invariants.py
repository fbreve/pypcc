"""Reproducible random-graph dominance invariants for all three backends."""
import unittest
import numpy as np
from pcc_numpy import pcc_propagate_numpy

try:
    from pcc_numba import pcc_propagate_numba
except ImportError:
    pcc_propagate_numba = None
try:
    from pcc_step import pcc_propagate
except ImportError:
    pcc_propagate = None


class RandomGraphDominanceInvariantTests(unittest.TestCase):
    def backends(self):
        result = [('numpy', pcc_propagate_numpy)]
        if pcc_propagate_numba is not None:
            result.append(('numba', pcc_propagate_numba))
        if pcc_propagate is not None:
            result.append(('cython', pcc_propagate))
        return result

    def run_case(self, fn, nodes, classes, degree, graph_seed, greedy):
        rng = np.random.default_rng(graph_seed)
        # Fixed degree and distinct neighbors; graph creation does not consume
        # any of the backend's movement RNG state.
        edges = np.empty((nodes, degree), dtype=np.int64)
        for node in range(nodes):
            candidates = np.concatenate((np.arange(node), np.arange(node + 1, nodes)))
            edges[node] = rng.choice(candidates, size=degree, replace=False)
        degrees = np.full(nodes, degree, dtype=np.int64)
        labels = np.full(nodes, -1, dtype=np.int64)
        particle_labels = np.arange(classes, dtype=np.int64)
        positions = np.arange(classes, dtype=np.int64)
        labels[positions] = particle_labels
        dominance = rng.dirichlet(np.ones(classes), size=nodes)
        dominance[positions] = np.eye(classes)
        initial_sums = dominance.sum(axis=1).copy()
        strengths = np.full(classes, .75, dtype=np.float64)
        distances = np.full((nodes, classes), 255, dtype=np.uint8)
        distances[positions, np.arange(classes)] = 0
        ownership = np.zeros((nodes, classes), dtype=np.float64)
        args = [edges, degrees, labels, greedy, .4, classes,
                np.zeros(classes), positions, particle_labels, strengths,
                distances, dominance, ownership, .4, 2.]
        scratch = [np.empty(classes), np.empty(classes)] + [np.empty(degree) for _ in range(4)] + [1. / (np.arange(257) + 1.) ** 2]
        fn(*args, *scratch, 200, False, 2000, 1, seed=graph_seed)
        np.testing.assert_allclose(dominance.sum(axis=1), initial_sums,
                                   rtol=0, atol=5e-12)
        np.testing.assert_array_equal(dominance[labels >= 0],
                                      np.eye(classes))
        self.assertTrue(np.isfinite(dominance).all())
        self.assertGreaterEqual(dominance.min(), -1e-13)
        self.assertLessEqual(dominance.max(), 1. + 1e-13)
        self.assertTrue(np.isfinite(strengths).all())
        self.assertTrue(np.isfinite(ownership).all())
        self.assertTrue(np.all((positions >= 0) & (positions < nodes)))

    def test_random_topologies_preserve_dominance(self):
        for nodes, classes, degree in ((12, 2, 2), (24, 3, 5), (40, 5, 10)):
            for graph_seed in (7, 19, 43):
                for greedy in (0., .5, 1.):
                    for backend, fn in self.backends():
                        with self.subTest(nodes=nodes, classes=classes,
                                          degree=degree, graph_seed=graph_seed,
                                          greedy=greedy, backend=backend):
                            self.run_case(fn, nodes, classes, degree,
                                          graph_seed, greedy)


if __name__ == '__main__':
    unittest.main()
