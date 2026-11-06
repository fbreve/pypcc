"""Public fit_predict parity for a default-on Cython rowmax cache.

A normal PCC user does not pass any Cython option. Compare that unchanged
high-level API against an explicit low-level opt-out adapter over the same
canonical undirected graph, seed, and initially labeled examples.
"""
import unittest

import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
)
from sklearn.preprocessing import StandardScaler

import pcc_step
from pcc import ParticleCompetitionAndCooperation
from pcc_graph import build_knn_graph


def unoptimized_cython(*args, **kwargs):
    return pcc_step.pcc_propagate(
        *args, **kwargs, experimental_rowmax_cache=False)


# Used by the high-level graph preparation to recognize the Cython backend
# and retain the same F-contiguous dominance representation.
unoptimized_cython.__name__ = "pcc_propagate"


class ExplicitLegacyCythonPCC(ParticleCompetitionAndCooperation):
    def _get_backend_fn(self):
        return unoptimized_cython


DATASETS = (
    ("iris", load_iris),
    ("wine", load_wine),
    ("breast_cancer", load_breast_cancer),
    ("digits", load_digits),
)
SEEDS = (8501, 8609, 8707)


class TestCythonDefaultOnPublicModel(unittest.TestCase):
    def test_high_level_fit_predict_exactly_matches_explicit_legacy(self):
        self.assertIs(
            ParticleCompetitionAndCooperation(impl="cython")
            ._get_backend_fn(),
            pcc_step.pcc_propagate,
        )
        for name, load in DATASETS:
            ds = load()
            x = StandardScaler().fit_transform(
                np.asarray(ds.data, dtype=np.float64))
            truth = np.asarray(ds.target, dtype=np.int64)
            graph, degrees = build_knn_graph(x, k_nn=10)
            neighbors = [
                set(map(int, graph[i, :int(degrees[i])]))
                for i in range(len(truth))
            ]
            self.assertTrue(all(
                i in neighbors[j] for i, adj in enumerate(neighbors)
                for j in adj), f"Noncanonical directed graph: {name}")

            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                labels = np.full(len(truth), -1, dtype=np.int64)
                for cls in np.unique(truth):
                    available = np.flatnonzero(truth == cls)
                    labels[rng.choice(available, 2, replace=False)] = cls
                for stop in (True, False):
                    with self.subTest(dataset=name,seed=seed,early_stop=stop):
                        modern = ParticleCompetitionAndCooperation(impl="cython")
                        legacy = ExplicitLegacyCythonPCC(impl="cython")
                        modern.set_graph(graph, degrees, k_nn=10)
                        legacy.set_graph(graph, degrees, k_nn=10)
                        kwargs=dict(
                            max_iter=10000, early_stop=stop, es_chk=2000,
                            p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0,
                            seed=seed,
                        )
                        modern_pred = modern.fit_predict(labels.copy(), **kwargs)
                        legacy_pred = legacy.fit_predict(labels.copy(), **kwargs)
                        for key, a, b in (
                            ("predictions", modern_pred, legacy_pred),
                            ("particle positions", modern.part.curnode,legacy.part.curnode),
                            ("particle strength", modern.part.strength,legacy.part.strength),
                            ("particle distances",modern.part.dist_table,legacy.part.dist_table),
                            ("node dominance",modern.node.dominance,legacy.node.dominance),
                            ("node ownership",modern.owndeg,legacy.owndeg),
                        ):
                            self.assertTrue(
                                np.array_equal(a,b),
                                f"{name} seed={seed} stop={stop}: mismatched {key}",
                            )

    def test_explicit_low_level_opt_out_remains_available(self):
        self.assertTrue(callable(unoptimized_cython))
        self.assertEqual(unoptimized_cython.__name__, "pcc_propagate")


if __name__ == "__main__":
    unittest.main()
