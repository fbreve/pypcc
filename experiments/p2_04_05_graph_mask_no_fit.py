"""P2-04 + P2-05 no-fit prospective graph/mask preflight on frozen UCI sources.

Reads and checks exact P1 datasets and P2-03 feature fingerprints. Builds each
graph exactly ONCE using unchanged ParticleCompetitionAndCooperation.build_graph
(never set_graph or fit_predict). Validates undirected ordered active neighbors,
dense int64 padded -1 representation, and graph bytes.

Materializes exactly eight PCG64 masks per approved source, three labels/class,
using the approved frozen P1 recipe. Derives 72 backend-domain-separated seeds
without calling any backend. Writes ONLY schema, source/graph/mask/seed hashes
and aggregate counts, never patient/grain/customer rows, labels, graph/mask
arrays, predictions or PCC state.

NOTE: this P2 engineering preflight *does* create REAL KNN GRAPHS and MASKS,
but performs ZERO PCC propagations/fits; it is NOT P2/P3 execution permission.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from pcc import ParticleCompetitionAndCooperation
from pcc_graph import build_knn_graph

from experiments import p1_exploratory_reference as frozen_reference
from experiments import p2_02a_official_uci_byte_integrity as official
from experiments import p2_02b_03_ordered_features_no_fit as prepared

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_VERSIONS = {
    "numpy": "2.5.4",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.1",
    "openpyxl": "3.1.5",
}
EXPECTED_TRANSFORM_SHA = {
    212: "420ff646d5edd234f907a27b71b00f5d29e4732959b58f34f32315740247cf21",
    850: "6b6d9a0520fee1a7625467a1ce6618f738d4eb815a4c4a786603329196294cf2",
    292: "9a4b2e4a49e2dd4d96f9cb99381d8849e9cce7571124761505938fc24c8acbe7",
}
EXPECTED_RAW_SHA = {
    212: "b9360c6b5aca77d7f4d3166437f0d5c4e2e88feb74f65730695b0c013745a017",
    850: "a8c391e1bca70cfc8b7e711f26a824e38398f7e04402be8d07fc3b31d94d5bb7",
    292: "0b3ee09cde98c0e6ac7d3c735764540839342347f3659088078212071087550a",
}
EXPECTED_P1_REFERENCE_BLOB = "480777758a445ba42425df69bcc5a90b97d6d0dc"
BACKENDS = (
    "numpy_sequential",
    "numba_sequential_nopython",
    "cython_sequential_cached_or_legacy_as_defined_by_early_stop",
)
KNN = 10
N_REPLICATES = 8
N_PER_CLASS = 3


class GraphMaskBlocked(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise GraphMaskBlocked(why)


def sha256(x):
    return hashlib.sha256(x).hexdigest()


def canonical_json(x):
    return json.dumps(x, separators=(",", ":"), sort_keys=True,
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def canonical_int64(arr, *, shape=None, signed=True):
    require(isinstance(arr, np.ndarray) and arr.dtype == np.dtype("int64"),
            "Canonical graph/mask arrays must be native int64 NumPy matrices")
    if shape is not None:
        require(arr.shape == shape, "Canonical int64 shape mismatch")
    return np.asarray(arr, dtype="<i8", order="C").tobytes(order="C")


def original_reference_source_verified():
    p = ROOT / "experiments/p1_exploratory_reference.py"
    blob = prepared.git_blob_sha(
        p.read_bytes().replace(bytes((13, 10)), bytes((10,))))
    require(blob == EXPECTED_P1_REFERENCE_BLOB,
            "Original P1 seed/reference source changed")
    require(tuple(frozen_reference.IDS) == tuple(official.DATASET_IDS) and
            tuple(frozen_reference.BACKENDS) == BACKENDS and
            frozen_reference.N_REPLICATES == N_REPLICATES,
            "P1 reference seed IDs/backend labels/replicate count changed")
    return blob


def verify_build_graph_route():
    """Inspect actual unchanged source signature, not inferred sk-learn defaults."""
    sig = inspect.signature(build_knn_graph)
    require(sig.parameters["k_nn"].default == KNN and
            sig.parameters["metric"].default == "minkowski" and
            sig.parameters["p"].default == 2,
            "Original graph builder k/metric/p defaults changed")
    require(inspect.signature(ParticleCompetitionAndCooperation.build_graph)
            .parameters["k_nn"].default == KNN,
            "PCC original graph installation route changed")
    return {
        "entrypoint": "ParticleCompetitionAndCooperation.build_graph",
        "source_function": "pcc_graph.build_knn_graph",
        "k_nn": KNN, "metric": "minkowski", "p": 2,
        "nearest_neighbors_algorithm": "auto", "n_jobs": 1,
        "symmetrization": "undirected union A|A.T",
        "native_dense_neighbor_dtype": "int64",
        "padding": -1,
        "NOT_using_set_graph": True,
        "NOT_using_CSR_in_PCC_propagation": True,
        "graph_construction_source_was_modified": False,
    }


def verify_dense_undirected(neib_list, neib_qt, *, n, k):
    """Fail-closed on self edges, asymmetry, padding, order, degrees, duplicates."""
    require(isinstance(neib_list, np.ndarray) and
            isinstance(neib_qt, np.ndarray), "PCC graph not NumPy arrays")
    require(neib_list.ndim == 2 and neib_list.shape[0] == n and
            neib_qt.shape == (n,), "Graph node count/dense matrix shape mismatch")
    require(neib_list.dtype == np.dtype("int64") and
            neib_qt.dtype == np.dtype("int64"), "Graph dtype not native int64")
    require(neib_list.flags.c_contiguous and neib_qt.flags.c_contiguous,
            "Graph memory layout changed from C-contiguous")
    width = neib_list.shape[1]
    require(width >= k and int(neib_qt.min()) >= k and
            int(neib_qt.max()) == width and
            int(neib_qt.max()) < n, "Bad graph degree/row width or empty neighbors")
    require((neib_list[:, 0] >= 0).all(), "Every source node needs active neighbors")
    adjacency = []
    for node in range(n):
        count = int(neib_qt[node])
        active = neib_list[node, :count]
        padding = neib_list[node, count:]
        require(0 < count <= width and (padding == -1).all(),
                "Dense -1 padding absent or degree out of range")
        require((active >= 0).all() and (active < n).all() and
                not (active == node).any(),
                "Self-neighbor or invalid active endpoint")
        require((np.diff(active) > 0).all(),
                "Unsorted or duplicate active neighbors; original CSR row-order drift")
        adjacency.append(set(int(x) for x in active))
    for node, neighbors in enumerate(adjacency):
        require(all(node in adjacency[other] for other in neighbors),
                "kNN symmetrized graph contains directed-only edge")
    edges = sum(len(row) for row in adjacency)
    require(edges % 2 == 0 and edges >= n * k,
            "Undirected adjacency edge count invalid")
    digest_spec = {
        "schema": "pypcc-P2-04-ordered-dense-graph-int64/v1",
        "graph_nodes": n,
        "neib_list_shape": list(neib_list.shape),
        "neib_qt_shape": list(neib_qt.shape),
        "dtype": "<i8",
        "byte_order": "little",
        "layout": "C-rowmajor",
        "padding": -1,
        "active_neighbors_are_strictly_increasing": True,
        "symmetrized_undirected": True,
        "k_nn": k,
        "metric": "minkowski",
        "p": 2,
        "n_jobs": 1,
    }
    combined = (canonical_json(digest_spec) + bytes((0,)) +
                canonical_int64(neib_qt) + canonical_int64(neib_list))
    return {
        "hash_serialization": digest_spec,
        "exact_full_dense_graph_SHA256": sha256(combined),
        "neib_list_bytes_SHA256": sha256(canonical_int64(neib_list)),
        "neib_qt_bytes_SHA256": sha256(canonical_int64(neib_qt)),
        "n_undirected_edges": edges // 2,
        "n_active_directed_adjacency_entries": edges,
        "minimum_degree": int(neib_qt.min()),
        "maximum_degree": int(neib_qt.max()),
        "graph_storage_bytes": int(neib_list.nbytes + neib_qt.nbytes),
        "zero_nodes_with_no_neighbors": 0,
        "self_loops_found": 0,
        "edges_without_reciprocals": 0,
        "max_active_neighbors_respect_dense_width": True,
        "dense_unused_padding_all_minus_one": True,
    }


def graph_through_model(X):
    """One original build_graph per dataset; not set_graph or fit_predict."""
    require(X.ndim == 2 and X.dtype == np.dtype("float64") and
            X.flags.c_contiguous and np.isfinite(X).all(),
            "Only original frozen finite row-ordered z-score matrix allowed")
    model = ParticleCompetitionAndCooperation(impl="numpy", n_jobs=1)
    require(model.neib_list is None and model.neib_qt is None,
            "Unexpected preexisting model graph")
    model.build_graph(X, k_nn=KNN, n_jobs=1)
    require(model.data is X and model.k_nn == KNN and model.n_jobs == 1,
            "Model.build_graph not installed using approved inputs")
    require(model.node is None and model.part is None and
            model.labels is None and model.owndeg is None,
            "PCC particle/fit state unexpectedly allocated during graph construction")
    return verify_dense_undirected(model.neib_list, model.neib_qt,
                                   n=len(X), k=KNN)


def make_mask(labels, class_order, registry_index, replicate):
    """Frozen PCG64 SeedSequence independent for each dataset and replicate."""
    require(type(registry_index) is int and
            0 <= registry_index < len(official.DATASET_IDS) and
            type(replicate) is int and 0 <= replicate < N_REPLICATES,
            "Mask registry/replicate outside frozen P1 design")
    require(tuple(sorted(set(labels))) == tuple(class_order),
            "Mask class registry mismatch")
    require(len(class_order) >= 2, "PCC requires two or more source classes")
    words = frozen_reference.proposed_mask_seed_words(
        registry_index, replicate)
    require(words == [20261112, registry_index, replicate],
            "P1 SeedSequence words changed")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))
    actual = []
    labels_arr = np.asarray(labels, dtype=str)
    for cls in class_order:
        members = np.flatnonzero(labels_arr == cls)
        require(len(members) >= N_PER_CLASS and
                bool((np.diff(members) > 0).all()),
                "Not enough original labels or bad original index order")
        selection = rng.choice(members, size=N_PER_CLASS, replace=False)
        require(selection.size == N_PER_CLASS and
                np.unique(selection).size == N_PER_CLASS,
                "Mask sampling produced repeated source records")
        actual.extend(int(x) for x in selection)
    sorted_selected = np.array(sorted(actual), dtype=np.int64)
    require(sorted_selected.size == N_PER_CLASS*len(class_order) and
            bool((np.diff(sorted_selected) > 0).all()),
            "Mask selected source rows overlap between classes")
    encoded = {cls: i for i, cls in enumerate(class_order)}
    mask = np.full(len(labels), -1, dtype=np.int64)
    for row_id in sorted_selected:
        mask[row_id] = encoded[labels[row_id]]
    counts = {cls: int(np.count_nonzero(mask == i))
              for i, cls in enumerate(class_order)}
    require(all(x == N_PER_CLASS for x in counts.values()) and
            int(np.count_nonzero(mask >= 0)) == len(sorted_selected) and
            int(np.count_nonzero(mask == -1)) == len(labels)-len(sorted_selected),
            "Exact 3/class frozen mask support or unlabeled sentinel changed")
    spec = {
        "schema": "pypcc-P2-05-int64-original-row-order-label-mask/v1",
        "n_rows": len(labels), "class_order": list(class_order),
        "source_registry_index": registry_index, "replicate": replicate,
        "generator": "np.random.Generator(PCG64(SeedSequence(words)))",
        "seed_words": words, "class_selection": "original ascending class row index",
        "sampling": "Generator.choice(members,size=3,replace=False) once per class",
        "selected_order": "sorted ascending union for serialization",
        "encoded_class_order": "0..C-1 lexicographic original source value",
        "unknown_class_index": -1,
        "dtype": "<i8", "memory_order": "C-rowmajor",
    }
    mask_bytes = canonical_int64(mask, shape=(len(labels),))
    indices_bytes = canonical_int64(sorted_selected)
    return {
        "replicate": replicate,
        "mask_sha256": sha256(canonical_json(spec) + bytes((0,)) + mask_bytes),
        "selected_original_row_indices_sha256": sha256(indices_bytes),
        "encoded_mask_bytes_sha256": sha256(mask_bytes),
        "serialization": spec,
        "selected_total": len(sorted_selected),
        "unlabeled_total": int(np.count_nonzero(mask == -1)),
        "per_class_labeled_counts": counts,
        "mask_array_exported": False,
    }


def seed_table_for_dataset(uci_id, index):
    """Generate exactly 8*3 frozen deterministic seed table entries, no RNG draws."""
    require(uci_id == official.DATASET_IDS[index],
            "Unapproved registry order or dataset ID")
    return [
        {
            "dataset_id": uci_id,
            "registry_index": index,
            "replicate": replicate,
            "backend": backend,
            "native_seed_uint32": frozen_reference.proposed_propagation_seed(
                uci_id, replicate, backend),
        }
        for replicate in range(N_REPLICATES)
        for backend in BACKENDS
    ]


def evaluate_dataset(spec):
    # Independent source and original P1-preprocessing re-validation.
    outer, _ = official.download_official(spec)
    require(sha256(outer) == spec["zip_sha256"], "Official frozen ZIP SHA drift")
    member = prepared.load_member_bytes(outer, spec)
    columns, rows, _, indices, feature_count = prepared.parse_ordered(spec, member)
    names = (list(prepared.WHOLESALE_X) if spec["uci_id"] == 292
             else [columns[i] for i in indices])
    X, labels, counts = prepared.numeric_rows_and_labels(
        spec, columns, rows, indices)
    require(X.shape == (spec["original_n_rows"], spec["n_features"]) and
            feature_count == spec["n_features"], "Prepared source feature dimensions drift")
    raw_sha = prepared.array_fingerprint(X, names)[
        "shape_dtype_feature_order_and_numeric_bytes_sha256"]
    require(raw_sha == EXPECTED_RAW_SHA[spec["uci_id"]],
            "Original source numeric X differs from independently tested P2-03")
    Z, trans = prepared.approved_transform(X, wholesale=(spec["uci_id"] == 292))
    require(trans["pop_std_ddof"] == 0 and
            trans["labels_used_to_select_or_scale_features"] is False,
            "P1 transform policy was silently altered")
    transformed_sha = prepared.array_fingerprint(Z, names)[
        "shape_dtype_feature_order_and_numeric_bytes_sha256"]
    require(transformed_sha == EXPECTED_TRANSFORM_SHA[spec["uci_id"]],
            "Source transformed-X differs from independently tested P2-03")
    graph = graph_through_model(Z)
    classes = tuple(sorted(spec["class_support"]))
    masks = [make_mask(labels, classes, spec["registry_index"], replicate)
             for replicate in range(N_REPLICATES)]
    require(len({m["mask_sha256"] for m in masks}) == N_REPLICATES,
            "Replicate masks unexpectedly coincide")
    seeds = seed_table_for_dataset(spec["uci_id"], spec["registry_index"])
    require(len(seeds) == N_REPLICATES * len(BACKENDS) and
            all(type(x["native_seed_uint32"]) is int and
                1 <= x["native_seed_uint32"] <= 2**32-1 for x in seeds),
            "P1 propagation seed table incomplete/out of range")
    return {
        "uci_id": spec["uci_id"],
        "registry_index": spec["registry_index"],
        "approved_P1_archive_sha256": spec["zip_sha256"],
        "approved_P1_member_sha256": spec["file_sha256"],
        "verified_P2_03_raw_X_sha256": raw_sha,
        "verified_P2_03_transformed_X_sha256": transformed_sha,
        "original_source_row_count": spec["original_n_rows"],
        "original_class_support": counts,
        "original_class_order": list(classes),
        "graph": graph,
        "masks": masks,
        "propagation_seeds_not_executed": seeds,
        "graph_arrays_exported": False,
        "masks_or_original_record_indices_exported": False,
        "PCC_particle_propagation_executed": False,
    }


def audit():
    require(sys.version_info[:2] == (3, 12),
            "Only candidate CPython3.12 scientific environment is tested")
    for name, version in EXPECTED_VERSIONS.items():
        require(importlib.metadata.version(name) == version,
                "Unpinned package version in candidate environment: " + name)
    parser_blobs = prepared.parser_sources_checked()
    ref_blob = original_reference_source_verified()
    route = verify_build_graph_route()
    p1 = json.loads(official.P1_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(official.P1_MANIFEST.read_text(encoding="utf-8"))
    selected = official.validate_registry(p1, manifest)
    require(p1["graph"]["k_nn"] == KNN and
            p1["graph"]["metric"] == "minkowski" and
            p1["graph"]["p"] == 2 and
            p1["graph"]["n_jobs"] == 1 and
            p1["graph"]["graph_directed"] is False and
            p1["graph"]["prohibit_CSR_propagation_replacement"] is True and
            p1["fit"]["label_mask_replicates_per_dataset"] == N_REPLICATES and
            p1["fit"]["originally_labeled_per_class"] == N_PER_CLASS,
            "Frozen scientific graph or mask recipe changed")
    require(p1["rng_and_masks"]["mask_seed_recipe"] ==
            "np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261112,registry_index,replicate_index])))" and
            p1["fit"]["originally_labeled_per_class"] == N_PER_CLASS and
            p1["fit"]["label_mask_replicates_per_dataset"] == N_REPLICATES and
            p1["fit"]["backend_order"] == list(BACKENDS),
            "Frozen scientific P1 mask/propagation backend recipe drift")
    data = [evaluate_dataset(spec) for spec in selected]
    seeds = [s["native_seed_uint32"] for ds in data
             for s in ds["propagation_seeds_not_executed"]]
    require(len(seeds) == 72, "Expected exactly 72 frozen propagation seeds")
    return {
        "schema": "pypcc-P2-04-05-frozen-UCI-graph-and-mask-preflight/v1",
        "status": "EXACT_THREE_KNN_GRAPHS_AND_24_MASKS_WITH_72_NATIVE_SEEDS_NO_PCC_FITS",
        "P1_scientific_SHA256": official.P1_SHA,
        "source_parser_original_git_blobs": parser_blobs,
        "frozen_P1_seed_recipe_source_git_blob": ref_blob,
        "original_graph_function_git_blob": p1["source_version"]["production_graph_git_blob_sha1"],
        "graph_build_and_install_route": route,
        "exact_candidate_Python_major_minor": "3.12",
        "candidate_installed_versions": EXPECTED_VERSIONS,
        "selected_UCI_order": list(official.DATASET_IDS),
        "datasets": data,
        "graphs_built_on_real_source_features": len(data),
        "real_source_label_masks_generated": sum(len(x["masks"]) for x in data),
        "backend_domain_separated_seeds_derived_not_used": len(seeds),
        "graph_padding_preserved_minus_one": True,
        "graph_is_undirected": True,
        "graph_fits_and_early_stops_executed": 0,
        "PCC_propagations_executed": 0,
        "PCC_real_data_fits_executed": 0,
        "P2_04_graph_candidate_pass": True,
        "P2_05_mask_and_seeds_candidate_pass": True,
        "P2_04_05_cross_platform_hash_gate_pass": False,
        "P2_01_final_native_environment_lock_pass": False,
        "P2_07_memory_and_time_resources_budget_pass": False,
        "P2_08_sealed_execution_manifest_pass": False,
        "raw_source_rows_or_labels_exported": False,
        "raw_graph_or_mask_arrays_exported": False,
        "P2_execution_preflight_SHA256": None,
        "P2_execution_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "uci_ids": result["selected_UCI_order"],
        "graph_hashes": [
            x["graph"]["exact_full_dense_graph_SHA256"]
            for x in result["datasets"]],
        "n_masks": result["real_source_label_masks_generated"],
        "n_propagation_seeds": result[
            "backend_domain_separated_seeds_derived_not_used"],
        "PCC_real_data_fits_executed": 0,
        "P2_execution_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
