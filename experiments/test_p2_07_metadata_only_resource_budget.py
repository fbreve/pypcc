"""Adversarial synthetic-only P2-07 resource/accounting unit tests, zero PCC."""
import copy
import unittest

from experiments import p2_07_metadata_only_resource_budget as budget


def frozen_metadata_fixture():
    datasets, ids, mask_hashes, graph_hashes = [], [], {}, []
    for index, (uci_id, n, features, classes) in enumerate(budget.EXPECTED_DATASETS):
        width = 12 + index
        class_order = [str(i) for i in range(classes)]
        counts = {name: n//classes for name in class_order}
        counts[class_order[-1]] += n-sum(counts.values())
        masks = [
            {"replicate": rep,
             "mask_sha256": budget.sha256(f"mask:{uci_id}:{rep}".encode("ascii")),
             "selected_total": 3*classes,
             "unlabeled_total": n-3*classes,
             "per_class_labeled_counts": {cls: 3 for cls in class_order},
             "mask_array_exported": False}
            for rep in range(8)
        ]
        seeds = [
            {"dataset_id": uci_id, "registry_index": index,
             "replicate": rep, "backend": backend,
             "native_seed_uint32": budget.frozen_seed(uci_id, rep, backend)}
            for rep in range(8) for backend in budget.BACKENDS
        ]
        graph_digest = budget.GRAPH_DIGESTS[uci_id]
        graph_hashes.append(graph_digest)
        mask_hashes[str(uci_id)] = [m["mask_sha256"] for m in masks]
        datasets.append({
            "uci_id": uci_id, "registry_index": index,
            "original_source_row_count": n,
            "original_class_support": counts,
            "original_class_order": class_order,
            "verified_P2_03_raw_X_sha256": "a"*64,
            "verified_P2_03_transformed_X_sha256": "b"*64,
            "graph": {
                "exact_full_dense_graph_SHA256": graph_digest,
                "dense_unused_padding_all_minus_one": True,
                "edges_without_reciprocals": 0,
                "self_loops_found": 0,
                "graph_storage_bytes": 8*n*(width+1),
                "minimum_degree": 10, "maximum_degree": width,
                "hash_serialization": {
                    "neib_list_shape": [n, width],
                    "neib_qt_shape": [n], "padding": -1,
                },
            },
            "masks": masks,
            "propagation_seeds_not_executed": seeds,
        })
    data = {
        "schema": "pypcc-P2-04-05-frozen-UCI-graph-and-mask-preflight/v1",
        "status": "EXACT_THREE_KNN_GRAPHS_AND_24_MASKS_WITH_72_NATIVE_SEEDS_NO_PCC_FITS",
        "P1_scientific_SHA256": budget.P1_SHA,
        "selected_UCI_order": [212, 850, 292],
        "graphs_built_on_real_source_features": 3,
        "real_source_label_masks_generated": 24,
        "backend_domain_separated_seeds_derived_not_used": 72,
        "PCC_propagations_executed": 0,
        "PCC_real_data_fits_executed": 0,
        "P2_execution_authorized": False,
        "P2_execution_preflight_SHA256": None,
        "raw_graph_or_mask_arrays_exported": False,
        "raw_source_rows_or_labels_exported": False,
        "graph_build_and_install_route": {"NOT_using_set_graph": True},
        "graph_padding_preserved_minus_one": True,
        "graph_is_undirected": True,
        "datasets": datasets,
    }
    cross = {
        "schema": "pypcc-P2-04-05-cross-platform-hash-concordance/v1",
        "P2_04_05_cross_platform_hash_gate_pass": True,
        "P1_scientific_sha256": budget.P1_SHA,
        "selected_uci_ids": [212, 850, 292],
        "platforms": ["ubuntu-latest", "windows-latest"],
        "dense_graph_hashes": graph_hashes,
        "mask_hashes_by_dataset": mask_hashes,
        "PCC_propagations_executed": 0,
        "P2_execution_authorized": False,
        "P2_08_approved_execution_SHA256": None,
    }
    return data, cross


class P207MetaOnlyBudgetTests(unittest.TestCase):
    def test_72_case_snapshot_budget_and_backend_specific_rowmax(self):
        left, right = frozen_metadata_fixture()
        rows = budget.check_hash_report(left, right)
        result = budget.accounting(rows)
        self.assertEqual(result["total_intended_fits"], 72)
        self.assertEqual(result["total_proposed_PCC_iterations_all_fits"], 3600000)
        self.assertEqual(result["total_particle_update_opportunities_all_fits"], 25200000)
        self.assertEqual(len(result["datasets"]), 3)
        self.assertEqual(result["if_every_fit_consumed_entire_64MiB_snapshot_budget_bytes"],
                         72 * 64*1024*1024)
        for entry in result["datasets"]:
            m = entry["backend_observer_payloads"]
            self.assertTrue(all(x["snapshots"]["under_original_observer_64_MiB_cap"]
                                for x in m.values()))
            normal = m[budget.BACKENDS[0]]["snapshots"]["planned_eight_snapshots_bytes"]
            numba = m[budget.BACKENDS[1]]["snapshots"]["planned_eight_snapshots_bytes"]
            cython = m[budget.BACKENDS[2]]["snapshots"]["planned_eight_snapshots_bytes"]
            self.assertEqual(normal, numba)
            self.assertEqual(cython-normal,
                             8*(8*entry["n_original_rows"]+4))
            self.assertEqual(
                entry["sum_retained_array_payload_bytes"],
                sum(entry["retained_array_line_items_bytes"].values()))

    def test_snapshot_exceeds_cap_is_rejected_in_isolation(self):
        # A true 64MiB per-fit snapshot limit; no allocating huge arrays.
        huge = budget.snapshot_bytes(1200000, 3, budget.BACKENDS[0])
        self.assertFalse(huge["under_original_observer_64_MiB_cap"])
        tiny = budget.snapshot_bytes(900, 2, budget.BACKENDS[2])
        self.assertTrue(tiny["under_original_observer_64_MiB_cap"])

    def test_reject_drift_in_graph_width_padding_and_provenance(self):
        source, concordance = frozen_metadata_fixture()
        cases = [
            ("width mismatch",
             lambda s, c: s["datasets"][0]["graph"].__setitem__(
                 "graph_storage_bytes", 0)),
            ("unpadded graph",
             lambda s, c: s["datasets"][0]["graph"].__setitem__(
                 "dense_unused_padding_all_minus_one", False)),
            ("graph changed in one OS",
             lambda s, c: c["dense_graph_hashes"].__setitem__(0, "f"*64)),
            ("wrong dataset order",
             lambda s, c: s.__setitem__(
                 "selected_UCI_order", [850, 212, 292])),
            ("unapproved input",
             lambda s, c: s.__setitem__("P2_execution_authorized", True)),
        ]
        for name, mutation in cases:
            with self.subTest(reason=name):
                s, c = copy.deepcopy(source), copy.deepcopy(concordance)
                mutation(s, c)
                with self.assertRaises(budget.BudgetBlocked):
                    budget.check_hash_report(s, c)

    def test_reject_missing_masks_and_forged_seed(self):
        source, concordance = frozen_metadata_fixture()
        mutated = copy.deepcopy(source)
        mutated["datasets"][0]["masks"].pop()
        with self.assertRaises(budget.BudgetBlocked):
            budget.check_hash_report(mutated, concordance)
        mutated = copy.deepcopy(source)
        mutated["datasets"][1]["propagation_seeds_not_executed"][0][
            "native_seed_uint32"] += 1
        with self.assertRaises(budget.BudgetBlocked):
            budget.check_hash_report(mutated, concordance)

    def test_fail_closed_if_missing_full_resource_budget(self):
        source, cross = frozen_metadata_fixture()
        rows = budget.check_hash_report(source, cross)
        accounting = budget.accounting(rows)
        self.assertGreater(
            accounting["total_eight_checkpoint_array_payloads_all_72_fits_bytes"], 0)
        self.assertTrue(accounting["array_payload_not_a_process_RSS_upper_bound"])
        self.assertTrue(accounting["projection_not_a_CPU_time_benchmark"])
        self.assertTrue(accounting["snapshot_persistence_on_disk_is_not_approved"])


if __name__ == "__main__":
    unittest.main()
