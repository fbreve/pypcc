"""Synthetic-only P2-04/05 tests: no official UCI data and no PCC fitting."""
import copy
import unittest

import numpy as np

from pcc_graph import build_knn_graph
from experiments import p2_04_05_graph_mask_no_fit as gate


def synthetic_X():
    # 18 deterministic 2-D points: asymmetric distances, no source records.
    return np.array([
        (i * 1.173 + (i % 3) * .037, (i*i % 17) * .217 + i*.013)
        for i in range(18)
    ], dtype=np.float64, order="C")


def synthetic_labels():
    return tuple(("Beta", "Alpha", "Gamma")[i//6] for i in range(18))


class GraphMaskSyntheticTests(unittest.TestCase):
    def test_original_production_build_graph_not_set_graph(self):
        features = synthetic_X()
        via_graph = gate.graph_through_model(features)
        neighbors, degrees = build_knn_graph(
            features, k_nn=10, metric="minkowski", p=2, n_jobs=1)
        raw = gate.verify_dense_undirected(neighbors, degrees, n=len(features), k=10)
        self.assertEqual(via_graph["exact_full_dense_graph_SHA256"],
                         raw["exact_full_dense_graph_SHA256"])
        self.assertEqual(via_graph["self_loops_found"], 0)
        self.assertEqual(via_graph["edges_without_reciprocals"], 0)
        self.assertGreater(via_graph["n_undirected_edges"], 0)

    def test_strict_padding_self_edges_degree_and_neighbor_order(self):
        neighbors, degrees = build_knn_graph(
            synthetic_X(), k_nn=10, metric="minkowski", p=2, n_jobs=1)
        def reject(mutate):
            bad_list, bad_deg = neighbors.copy(), degrees.copy()
            mutate(bad_list, bad_deg)
            with self.assertRaises(gate.GraphMaskBlocked):
                gate.verify_dense_undirected(bad_list, bad_deg, n=18, k=10)

        reject(lambda a, b: a.__setitem__((0, 0), 0))
        reject(lambda a, b: a.__setitem__((0, 0), 19))
        reject(lambda a, b: a.__setitem__((0, 1), a[0, 0]))
        reject(lambda a, b: a.__setitem__((0, 0), -1))
        reject(lambda a, b: b.__setitem__(0, 0))
        # Make an edge present on one side but absent on the reciprocal side.
        def break_reciprocal(a, b):
            other = int(a[0, 0])
            reverse = np.flatnonzero(a[other, :int(b[other])] == 0)
            assert reverse.size == 1
            a[other, int(reverse[0])] = -1
        reject(break_reciprocal)
        # Graphs whose width > row degree must keep literal -1 padding.
        padded = [int(i) for i, d in enumerate(degrees) if d < neighbors.shape[1]]
        self.assertTrue(padded, "Synthetic fixture unexpectedly lacks padded nodes")
        reject(lambda a, b: a.__setitem__((padded[0], int(b[padded[0]])), 0))

    def test_pcG64_mask_reproducibility_and_no_ground_truth_export(self):
        labels = synthetic_labels()
        classes = tuple(sorted(set(labels)))
        masks = [gate.make_mask(labels, classes, 1, i) for i in range(8)]
        repeated = [gate.make_mask(labels, classes, 1, i) for i in range(8)]
        self.assertEqual(masks, repeated)
        self.assertEqual(len({m["mask_sha256"] for m in masks}), 8)
        for i, m in enumerate(masks):
            self.assertEqual(m["replicate"], i)
            self.assertEqual(m["selected_total"], 9)
            self.assertEqual(m["unlabeled_total"], 9)
            self.assertEqual(m["per_class_labeled_counts"],
                             {"Alpha": 3, "Beta": 3, "Gamma": 3})
            self.assertFalse(m["mask_array_exported"])
            self.assertNotIn("selected_original_row_indices", m)
        alternative = gate.make_mask(labels, classes, 2, 0)
        self.assertNotEqual(alternative["mask_sha256"], masks[0]["mask_sha256"])

    def test_fail_on_bad_source_class_counts_for_mask(self):
        with self.assertRaises(gate.GraphMaskBlocked):
            gate.make_mask(("a", "a", "a", "b", "b"), ("a", "b"), 0, 0)
        with self.assertRaises(gate.GraphMaskBlocked):
            gate.make_mask(synthetic_labels(), ("Gamma", "Beta", "Alpha"), 0, 0)
        with self.assertRaises(gate.GraphMaskBlocked):
            gate.make_mask(synthetic_labels(), ("Alpha", "Beta", "Gamma"), 1, 8)

    def test_frozen_domain_separated_seeds_no_RNG_or_model_use(self):
        table = gate.seed_table_for_dataset(850, 1)
        self.assertEqual(len(table), 24)
        self.assertEqual(len({x["native_seed_uint32"] for x in table}), 24)
        self.assertEqual([r["backend"] for r in table[:3]], list(gate.BACKENDS))
        self.assertTrue(all(1 <= x["native_seed_uint32"] <= 2**32-1 for x in table))
        with self.assertRaises(gate.GraphMaskBlocked):
            gate.seed_table_for_dataset(212, 1)

    def test_canonical_serialization_changes_with_byte_order_and_padding(self):
        base = np.array([[1, 2, -1], [0, 2, 3]], dtype=np.int64)
        base_bytes = gate.canonical_int64(base)
        self.assertEqual(len(base_bytes), 6 * 8)
        changed = base.copy()
        changed[0, 2] = 0
        self.assertNotEqual(gate.sha256(base_bytes),
                            gate.sha256(gate.canonical_int64(changed)))
        with self.assertRaises(gate.GraphMaskBlocked):
            gate.canonical_int64(base.astype(np.int32))


if __name__ == "__main__":
    unittest.main()
