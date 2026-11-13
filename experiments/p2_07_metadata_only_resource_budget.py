"""P2-07: reproducible metadata-only resource accounting, NO PCC or source rows.

Consumes only already archived P2-04/05 hash/aggregate JSON evidence from
GitHub Actions #43475652889. No dataset acquisition, matrix/graph construction,
mask sampling, binary compilation, particle state or fit_predict is performed.

Models exactly the five snapshot arrays (plus Cython rowmax/RNG) documented in
the three pinned original observer sources, and the main persistent NumPy
arrays from the unchanged pcc.py. This is NOT a bound on process RSS, JIT
compiler memory, sk-learn transient workspace, CPU runtime or wall-clock time.
Such missing evidence blocks P2-07 overall and any P3 real-data fit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
P1_SHA = "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166"
SOURCE_PR = 267
SOURCE_RUN_ID = 43475652889
SOURCE_HEAD = "1a86cf0b1f9350f6a98cd1287e9b77edd390c87a"
SOURCE_ARTIFACT_IDS = {"Linux": 14063840028, "Windows": 14063660306}
CONCORDANCE_ARTIFACT_ID = 14063161582
MAX_SNAPSHOT_BYTES_PER_FIT = 64 * 1024 * 1024
MILESTONES = (0, 1, 10, 100, 1000, 5000, 25000, 50000)
N_MASKS = 8
BACKENDS = ("numpy_sequential", "numba_sequential_nopython",
            "cython_sequential_cached_or_legacy_as_defined_by_early_stop")
EXPECTED_DATASETS = ((212, 310, 6, 3), (850, 900, 7, 2), (292, 440, 6, 2))
GRAPH_DIGESTS = {
    212: "434ec3bb811222964304ae163afa1bf3eb25f767139107b5bb0f05dc67db27f2",
    850: "b0e260ae198037e89e5e8b0691da15b4c7bfc4cf17e41adb1fe8209a7a975384",
    292: "63e502a5d3f981a20c8c1a80884aa740a25fa7ea7f13335ad7554ad6e7824ca6",
}
SOURCE_GIT_BLOBS = {
    ("1a86cf0b1f9350f6a98cd1287e9b77edd390c87a", "pcc.py"):
        "01606aaee860c3b9bc84710e7362a2ef81046d45",
    ("1a86cf0b1f9350f6a98cd1287e9b77edd390c87a", "pcc_graph.py"):
        "3e8943462555b4feafbd162125b51e41f29d7ab7",
    ("6d908e78878c4922674277e3313dbfe7f7924146", "experiments/live_numpy_milestones.py"):
        "d642bd0f329eceb4f3dc3111b5f8c91f0e085cac",
    ("4c8e6df4e01d7a4dfc4efc31eb73db04d6e212e5", "experiments/live_numba_milestones.py"):
        "e10942e96be623d47d5537d45bd365f366bba548",
    ("af34f50ebabb0a7204d3f30679adaf3f00169742", "experiments/live_cython_milestones.py"):
        "281daf0e66a567e587d0bd3e21dc34329a234b95",
}

class BudgetBlocked(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise BudgetBlocked(reason)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def is_hash(value):
    return type(value) is str and len(value) == 64 and all(
        letter in "0123456789abcdef" for letter in value)


def frozen_seed(uci_id, replicate, backend):
    require(uci_id in (212, 850, 292) and 0 <= replicate < N_MASKS and
            backend in BACKENDS, "Unexpected frozen backend seed identity")
    key = ("pypcc-P1-propagation-v1|uci:" + str(uci_id) +
           "|mask:" + str(replicate) + "|" + backend)
    return 1 + (int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:4],
                                "little") % (2**32 - 1))


def original_git_blobs_present():
    """Git blob content is pinned, never infer model memory from current HEAD."""
    verified = {}
    for (commit, path), expected in SOURCE_GIT_BLOBS.items():
        command = ["git", "-C", str(ROOT), "rev-parse", commit + ":" + path]
        result = subprocess.run(command, check=False, capture_output=True,
                                text=True, errors="replace")
        require(result.returncode == 0 and result.stdout.strip() == expected,
                "Missing or modified original model/observer Git blob: " + path)
        verified[path] = {"source_commit": commit, "git_blob_sha1": expected}
    return verified


def check_hash_report(source, concordance):
    """Require real previous CI concordance and exact immutable input, no guess."""
    require(source.get("schema") ==
            "pypcc-P2-04-05-frozen-UCI-graph-and-mask-preflight/v1" and
            source.get("status") ==
            "EXACT_THREE_KNN_GRAPHS_AND_24_MASKS_WITH_72_NATIVE_SEEDS_NO_PCC_FITS" and
            concordance.get("schema") ==
            "pypcc-P2-04-05-cross-platform-hash-concordance/v1" and
            concordance.get("P2_04_05_cross_platform_hash_gate_pass") is True,
            "Wrong or unapproved P2-04/05 metadata provenance")
    require(source.get("P1_scientific_SHA256") == P1_SHA ==
            concordance.get("P1_scientific_sha256"),
            "Frozen scientific protocol SHA mismatch")
    require(source.get("selected_UCI_order") == [212, 850, 292] ==
            concordance.get("selected_uci_ids") and
            concordance.get("platforms") == ["ubuntu-latest", "windows-latest"],
            "Wrong UCI roster or missing OS concordance")
    require(source.get("graphs_built_on_real_source_features") == 3 and
            source.get("real_source_label_masks_generated") == 24 and
            source.get("backend_domain_separated_seeds_derived_not_used") == 72,
            "Historical graph/mask/seed counts not frozen")
    require(source.get("PCC_propagations_executed") == 0 and
            source.get("PCC_real_data_fits_executed") == 0 and
            source.get("P2_execution_authorized") is False and
            source.get("P2_execution_preflight_SHA256") is None and
            source.get("raw_graph_or_mask_arrays_exported") is False and
            source.get("raw_source_rows_or_labels_exported") is False,
            "Source report cannot contain real PCC fit or raw scientific records")
    require(concordance.get("PCC_propagations_executed") == 0 and
            concordance.get("P2_execution_authorized") is False and
            concordance.get("P2_08_approved_execution_SHA256") is None,
            "The prior cross-platform gate did not authorize P2")
    require(source.get("graph_build_and_install_route", {}).get("NOT_using_set_graph") is True and
            source.get("graph_padding_preserved_minus_one") is True and
            source.get("graph_is_undirected") is True,
            "Original dense graph creation/storage protocol differs")
    datasets = source.get("datasets")
    require(type(datasets) is list and len(datasets) == 3,
            "Three source summaries are mandatory")
    concordance_graphs = concordance.get("dense_graph_hashes")
    concordance_masks = concordance.get("mask_hashes_by_dataset")
    require(type(concordance_graphs) is list and len(concordance_graphs) == 3 and
            type(concordance_masks) is dict and
            set(concordance_masks) == {"212", "850", "292"},
            "Prior cross-platform hash manifest incomplete")
    for ordinal, (d, (uci_id, n, features, c)) in enumerate(
            zip(datasets, EXPECTED_DATASETS)):
        graph = d.get("graph")
        require(d.get("uci_id") == uci_id and
                d.get("registry_index") == ordinal and
                d.get("original_source_row_count") == n and
                len(d.get("original_class_support", {})) == c and
                sum(d["original_class_support"].values()) == n and
                len(d.get("original_class_order", [])) == c,
                "Original source geometry/target cardinality differs")
        require(d.get("verified_P2_03_transformed_X_sha256") and
                is_hash(d["verified_P2_03_transformed_X_sha256"]) and
                is_hash(d["verified_P2_03_raw_X_sha256"]),
                "P2-03 matrix fingerprint absent")
        require(graph.get("exact_full_dense_graph_SHA256") == GRAPH_DIGESTS[uci_id] ==
                concordance_graphs[ordinal] and
                graph.get("dense_unused_padding_all_minus_one") is True and
                graph.get("edges_without_reciprocals") == 0 and
                graph.get("self_loops_found") == 0,
                "Prior dense kNN graph drifted")
        m = graph.get("hash_serialization", {})
        width = m.get("neib_list_shape", [None, None])[1]
        require(type(width) is int and n > width >= 10 and
                m.get("neib_list_shape") == [n, width] and
                m.get("neib_qt_shape") == [n] and
                m.get("padding") == -1 and
                graph.get("graph_storage_bytes") == 8*n*(width+1) and
                graph.get("minimum_degree") >= 10 and
                graph.get("maximum_degree") == width,
                "Dense int64 graph dimensions/degrees are not consistent")
        masks = d.get("masks")
        require(type(masks) is list and len(masks) == N_MASKS and
                type(concordance_masks[str(uci_id)]) is list and
                len(concordance_masks[str(uci_id)]) == N_MASKS,
                "Prior mask count/evidence not complete")
        p = 3*c
        for replicate, item in enumerate(masks):
            require(item.get("replicate") == replicate and
                    item.get("mask_sha256") == concordance_masks[str(uci_id)][replicate] and
                    is_hash(item["mask_sha256"]) and
                    item.get("selected_total") == p and
                    item.get("unlabeled_total") == n-p and
                    item.get("per_class_labeled_counts") ==
                    {label: 3 for label in d["original_class_order"]} and
                    item.get("mask_array_exported") is False,
                    "Original labeled count or per-mask SHA not frozen")
        seeds = d.get("propagation_seeds_not_executed")
        require(type(seeds) is list and len(seeds) == N_MASKS*3,
                "Prior propagation seed count/identity not complete")
        for index, entry in enumerate(seeds):
            replicate, backend = divmod(index, 3)
            require(entry.get("dataset_id") == uci_id and
                    entry.get("registry_index") == ordinal and
                    entry.get("replicate") == replicate and
                    entry.get("backend") == BACKENDS[backend] and
                    entry.get("native_seed_uint32") ==
                    frozen_seed(uci_id, replicate, BACKENDS[backend]),
                    "Prior approved P1 native propagation seed differs")
    return datasets


def model_persistent_array_bytes(n, features, classes, width):
    """Transparent hypothetical co-resident arrays, NOT an RSS cap/estimate."""
    p = 3*classes
    table = {
        "raw_X_float64": 8*n*features,
        "transformed_X_float64": 8*n*features,
        "dense_neib_list_int64": 8*n*width,
        "neib_qt_int64": 8*n,
        "input_labels_int64": 8*n,
        "mapped_labels_int64": 8*n,
        "node_labels_int64": 8*n,
        "final_labels_int64_transient": 8*n,
        "particle_homenode_int64": 8*p,
        "particle_curnode_int64": 8*p,
        "particle_label_int64": 8*p,
        "particle_strength_float64": 8*p,
        "particle_dist_table_uint8": n*p,
        "node_dominance_float64": 8*n*classes,
        "node_owndeg_float64": 8*n*classes,
        "zerovec_float64": 8*classes,
        "dom_row_and_reduc_float64": 2*8*classes,
        "four_neighbor_buffers_float64": 4*8*width,
        "distance_weights_float64_257": 257*8,
    }
    require(all(type(v) is int and v >= 0 for v in table.values()),
            "Invalid numeric retained-array model")
    return table


def snapshot_bytes(n, classes, observer):
    p = 3*classes
    per_checkpoint = {
        "positions_int64": 8*p,
        "strengths_float64": 8*p,
        "distance_table_uint8": n*p,
        "dominance_float64": 8*n*classes,
        "owndeg_float64": 8*n*classes,
    }
    if observer == "cython_sequential_cached_or_legacy_as_defined_by_early_stop":
        per_checkpoint["rowmax_float64"] = 8*n
        per_checkpoint["native_rng_uint32"] = 4
    require(observer in BACKENDS, "Unknown observer type")
    per_snapshot = sum(per_checkpoint.values())
    total = per_snapshot * len(MILESTONES)
    # Additional Cython allocation of schedule, final RNG and recorded-count
    # are noted separately from the native 64MiB observer check.
    bookkeeping = (8*len(MILESTONES) + 4 + 8) if observer == BACKENDS[2] else \
                  (8*len(MILESTONES) if observer == BACKENDS[1] else 0)
    return {
        "snapshot_arrays": per_checkpoint,
        "bytes_per_checkpoint": per_snapshot,
        "checkpoint_count": len(MILESTONES),
        "planned_eight_snapshots_bytes": total,
        "observer_cap_bytes": MAX_SNAPSHOT_BYTES_PER_FIT,
        "under_original_observer_64_MiB_cap": total <= MAX_SNAPSHOT_BYTES_PER_FIT,
        "additional_native_schedule_and_bookkeeping_bytes": bookkeeping,
        "observer_allocation_plus_bookkeeping_bytes": total + bookkeeping,
    }


def accounting(datasets):
    rows = []
    accumulated = 0
    overall_max_payload = 0
    for dataset, (_, n, features, c) in zip(datasets, EXPECTED_DATASETS):
        width = dataset["graph"]["hash_serialization"]["neib_list_shape"][1]
        retained = model_persistent_array_bytes(n, features, c, width)
        candidates = {}
        for observer in BACKENDS:
            snapshots = snapshot_bytes(n, c, observer)
            retained_plus_snapshot = sum(retained.values()) + snapshots[
                "observer_allocation_plus_bookkeeping_bytes"]
            require(snapshots["under_original_observer_64_MiB_cap"],
                    "Dataset exceeds original observer's 64MiB per-fit cap")
            candidates[observer] = {
                "retained_arrays_plus_observer_hypothetical_co_resident_bytes":
                    retained_plus_snapshot,
                "snapshots": snapshots,
                "fits_proposed_for_dataset_backend": N_MASKS,
                "total_snapshot_payload_bytes_if_all_eight_fits_preserved":
                    snapshots["planned_eight_snapshots_bytes"]*N_MASKS,
            }
            accumulated += snapshots["planned_eight_snapshots_bytes"]*N_MASKS
            overall_max_payload = max(overall_max_payload, retained_plus_snapshot)
        rows.append({
            "uci_id": dataset["uci_id"],
            "n_original_rows": n,
            "n_numeric_features": features,
            "n_classes": c,
            "n_original_labeled_per_fit": 3*c,
            "n_original_unlabeled_per_fit": n-3*c,
            "dense_neighbor_width": width,
            "verified_graph_storage_bytes": dataset["graph"]["graph_storage_bytes"],
            "proposed_particle_update_opportunities_per_fit": 50000*3*c,
            "retained_array_line_items_bytes": retained,
            "sum_retained_array_payload_bytes": sum(retained.values()),
            "backend_observer_payloads": candidates,
            "n_planned_fits_all_backends": N_MASKS*len(BACKENDS),
        })
    require(len(rows) == 3 and sum(x["n_planned_fits_all_backends"]
                                  for x in rows) == 72,
            "Study schedule changed or incomplete")
    total_moves = sum(x["proposed_particle_update_opportunities_per_fit"] *
                      x["n_planned_fits_all_backends"] for x in rows)
    return {
        "datasets": rows,
        "total_intended_fits": 72,
        "total_proposed_PCC_iterations_all_fits": 72 * 50000,
        "total_particle_update_opportunities_all_fits": total_moves,
        "total_eight_checkpoint_array_payloads_all_72_fits_bytes":
            accumulated,
        "maximum_hypothetical_co_resident_array_payload_bytes_any_one_fit":
            overall_max_payload,
        "if_every_fit_consumed_entire_64MiB_snapshot_budget_bytes":
            72*MAX_SNAPSHOT_BYTES_PER_FIT,
        "no_actual_PCC_particle_updates_executed": True,
        "projection_not_a_CPU_time_benchmark": True,
        "array_payload_not_a_process_RSS_upper_bound": True,
        "snapshot_persistence_on_disk_is_not_approved": True,
    }


def current_runner_observation():
    """Host facts are advisory, not a toolchain lock or credible PCC CPU budget."""
    total, used, free = shutil.disk_usage(Path.cwd())
    return {
        "os": platform.system(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "logical_cpu_count_if_exposed": os.cpu_count(),
        "working_volume_total_bytes": total,
        "working_volume_free_bytes_observed": free,
        "working_volume_used_bytes_observed": used,
        "gh_runner_os": os.environ.get("RUNNER_OS"),
        "os_runner_image_not_immutable": True,
        "process_RSS_peak_measured_with_real_PCC_fit": False,
        "production_50k_iteration_CPU_wall_time_measured": False,
        "actual_compiler_JIT_memory_overhead_proven": False,
    }


def build_report(meta, concordance):
    datasets = check_hash_report(meta, concordance)
    source_blobs = original_git_blobs_present()
    computed = accounting(datasets)
    assert_flags = {
        "P2_07_snapshot_payload_within_original_64MiB_per_fit_cap": all(
            entry["snapshots"]["under_original_observer_64_MiB_cap"]
            for row in computed["datasets"]
            for entry in row["backend_observer_payloads"].values()),
        "P2_07_total_CPU_wall_time_budget_verified": False,
        "P2_07_native_backend_peak_RSS_budget_verified": False,
        "P2_07_real_fit_artifact_disk_and_log_budget_verified": False,
        "P2_07_full_resources_gate_pass": False,
        "P2_08_manifest_finalized_or_approved": False,
    }
    require(assert_flags["P2_07_snapshot_payload_within_original_64MiB_per_fit_cap"],
            "P2-07 array snapshot cap violated")
    return {
        "schema": "pypcc-P2-07-meta-only-array-and-snapshot-budget/v1",
        "status": "SNAPSHOT_ARRAY_BYTES_WITHIN_CAP_REAL_EXECUTION_CPU_RSS_DISK_STILL_UNVERIFIED",
        "P1_scientific_sha256": P1_SHA,
        "parent_P2_04_05_PR": SOURCE_PR,
        "parent_P2_04_05_workflow_run": SOURCE_RUN_ID,
        "parent_P2_04_05_exact_HEAD": SOURCE_HEAD,
        "parent_archived_artifact_ids": SOURCE_ARTIFACT_IDS,
        "parent_archived_cross_platform_concordance_artifact_id":
            CONCORDANCE_ARTIFACT_ID,
        "original_frozen_model_and_observer_source_git_blobs": source_blobs,
        "actual_P2_04_05_graph_hashes":
            {str(x["uci_id"]): x["graph"]["exact_full_dense_graph_SHA256"]
             for x in datasets},
        "accounting": computed,
        "candidate_environment_observed_not_locked": current_runner_observation(),
        "safety_and_approval_gates": assert_flags,
        "unknown_and_blocking_items": [
            "Actual 50,000-iteration per-backend, per-dataset CPU/wall-time measurement",
            "Full-process peak resident RAM while Cython and Numba observers are active",
            "Compiled JIT/native toolchain and Python extension overhead on intended execution host",
            "Actual source graph building, sklearn workspaces and NumPy temporaries RSS",
            "Final approved whole-run CPU/deadline/parallelism limit for 72 cases",
            "Approved disk/log/export budget, checkpoint retention and privacy policy",
            "Final native locked toolchain, original or candidate wheel hashes, and exact P2 SHA",
            "Explicit separate maintainer authorization of final technical P2 SHA",
        ],
        "real_dataset_downloads_or_rows_loaded_in_P2_07": 0,
        "real_knn_graphs_built_in_P2_07": 0,
        "real_label_masks_sampled_in_P2_07": 0,
        "PCC_particle_propagations_executed": 0,
        "PCC_real_data_fits_executed": 0,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-0405-metadata", required=True, type=Path)
    parser.add_argument("--p2-0405-concordance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads(args.p2_0405_metadata.read_text(encoding="utf-8"))
    cross = json.loads(args.p2_0405_concordance.read_text(encoding="utf-8"))
    result = build_report(raw, cross)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    budget = result["accounting"]
    print(json.dumps({
        "status": result["status"],
        "platform": platform.system(),
        "datasets": [x["uci_id"] for x in budget["datasets"]],
        "fits_proposed_never_executed": budget["total_intended_fits"],
        "maximum_one_fit_array_payload_bytes":
            budget["maximum_hypothetical_co_resident_array_payload_bytes_any_one_fit"],
        "all_72_snapshot_arrays_bytes":
            budget["total_eight_checkpoint_array_payloads_all_72_fits_bytes"],
        "P2_07_full_resources_gate_pass": False,
        "P2_execution_authorized": False,
        "PCC_real_data_fits_executed": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
