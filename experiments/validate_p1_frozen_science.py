"""Validate the explicitly approved exploratory P1 SCIENCE protocol, with NO FIT.

Approval of the scientific design does not authorize source downloads, row-level
data processing, label-mask sampling, graph building, native compilation or
particle propagation. The SHA-256 of canonical JSON is the immutable SCIENCE
payload identity; it is separate from any future P2 execution authorization.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "docs/current_p1_frozen_scientific_protocol_2026-11-12.json"
DRAFT = ROOT / "docs/current_p1_prospective_design_draft_2026-11-12.json"
DECISIONS = ROOT / "docs/current_p1_scientific_decision_packet_2026-11-12.json"
REVIEW = ROOT / "docs/current_p1_scientific_resolution_for_review_2026-11-12.json"
GRAPH = ROOT / "pcc_graph.py"
EXPECTED_BASE_SHA = "ea69f690208620d5563c8efece6b2e04369f52c0"
EXPECTED_GRAPH_BLOB = "3e8943462555b4feafbd162125b51e41f29d7ab7"
IDS = (212, 850, 292)
BACKENDS = (
    "numpy_sequential",
    "numba_sequential_nopython",
    "cython_sequential_cached_or_legacy_as_defined_by_early_stop",
)
MILESTONES = [0, 1, 10, 100, 1000, 5000, 25000, 50000]


class FrozenScienceInvalid(ValueError):
    pass


def require(ok, explanation):
    if not ok:
        raise FrozenScienceInvalid(explanation)


def canonical_sha256(document):
    return hashlib.sha256(json.dumps(
        document, ensure_ascii=True, sort_keys=True, allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")).hexdigest()


def git_blob_sha1(data):
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\x00" + data,
    ).hexdigest()


def validate(frozen, draft, decisions, review, graph_bytes):
    require(type(frozen) is dict and
            frozen.get("schema") == "pypcc-P1-scientific-protocol-frozen/v1" and
            frozen.get("status") ==
            "SCIENTIFICALLY_APPROVED_EXPLORATORY_P1_ONLY",
            "P1 is an explicitly approved EXPLORATORY science protocol only")
    origin = frozen.get("approval_origin", {})
    require(origin.get("mechanism") == "maintainer_explicit_in_conversation" and
            "Sim, prossiga." in origin.get("evidence", "") and
            len(origin.get("scope", [])) == 6 and
            origin.get("does_NOT_approve_P2_execution") is True and
            origin.get("cannot_claim_no_prior_local_experiments") is True and
            origin.get("local_unpublished_experiment_history", "").startswith(
                "UNKNOWN_NOT_DECLARED") and
            origin.get("source_patient_customer_batch_independence",
                       "").startswith("NOT_VERIFIED"),
            "A human science GO cannot be misread as P2 approval or novelty")
    expected_scopes = [
        "S1_DATASETS_AND_HISTORICAL_EXPOSURE",
        "S2_TARGET_FEATURES_AND_PROVENANCE",
        "S3_PREPROCESSING", "S4_GRAPH_REPRESENTATION",
        "S5_MASK_SEEDS_AND_OBSERVATION", "S6_ANALYSIS_AND_DECISION",
    ]
    require(origin["scope"] == expected_scopes and
            [x["id"] for x in review["recommendations"]] == expected_scopes and
            review["human_decision"]["maintainer_approval"] is None and
            review["status"] ==
            "RECOMMENDED_FOR_EXPLICIT_SCIENTIFIC_DECISION_NOT_APPROVED",
            "Historical reviewer draft must remain a historical UNAPPROVED artifact")
    version = frozen.get("source_version", {})
    require(version.get("baseline_master_sha") == EXPECTED_BASE_SHA and
            version.get("production_graph_git_blob_sha1") == EXPECTED_GRAPH_BLOB and
            version.get("scientific_design_review_PR") == 258 and
            version.get("source_quality_review_PR") == 260 and
            version.get("evidence_registry") == decisions["evidence_files"] and
            version.get("production_source_changes_authorized") is False,
            "Scientific review source identity or authorization drift")
    # Windows checkout may convert exact Git LF bytes to CRLF. Accept only
    # byte-for-byte original Git blob after that one reversible normalization.
    if git_blob_sha1(graph_bytes) != EXPECTED_GRAPH_BLOB:
        require(
            b"\r\n" in graph_bytes and
            git_blob_sha1(graph_bytes.replace(b"\r\n", b"\n")) ==
            EXPECTED_GRAPH_BLOB,
            "Production graph source modified from frozen Git blob",
        )
    selected = frozen.get("selected_datasets")
    require(type(selected) is list and
            [d["uci_id"] for d in selected] == list(IDS),
            "Precisely 3 chosen domains in approved order are required")
    for i, (item, proposal) in enumerate(zip(
            selected, decisions["candidate_shortlist_proposed_not_selected"])):
        for key, value in (
            ("registry_index", i),
            ("uci_id", proposal["dataset_id"]),
            ("source_name", proposal["name"]),
            ("official_source_url", proposal["source_url"]),
            ("zip_sha256", proposal["source_zip_sha256"]),
            ("file_name", proposal["selected_file_name"]),
            ("file_sha256", proposal["selected_file_sha256"]),
            ("original_n_rows", proposal["n_rows"]),
            ("n_features", proposal["n_features"]),
            ("class_support", proposal["class_support"]),
            ("domain", proposal["domain"]),
        ):
            require(item.get(key) == value,
                    "P1 source identity or class support drift: " + key)
        require(item.get("clinical_subject_or_acquisition_batch_independence_established")
                is False and
                item.get("documented_prior_local_PCC_use") ==
                "NOT_DISCLOSED_OR_VERIFIED" and
                item.get("experiment_type") == "EXPLORATORY_ONLY",
                "Cannot turn absent historical/group evidence into proof")
    require(selected[0]["actual_target"].startswith("Class") and
            selected[1]["actual_target"] == "Class" and
            selected[2]["actual_target"] == "Channel" and
            selected[2]["target_is_derived"] is True and
            selected[2]["excluded_predictors"] == ["Channel", "Region"] and
            selected[2]["predictive_features"] ==
            ["Fresh", "Milk", "Grocery", "Frozen",
             "Detergents_Paper", "Delicassen"],
            "Scientific source target, label or feature leakage")
    require([d["uci_id"] for d in frozen["excluded_sources"]] == [545, 379],
            "Only one grain-morphology source, no encoded phishing substitute")
    prep = frozen.get("preprocessing", {})
    for field in ("original_rows_preserved","no_label_leakage",
                  "no_missing_value_imputation","no_row_exclusion",
                  "no_outlier_clipping","no_label_based_scaling",
                  "unsupervised_transductive_all_X"):
        require(prep.get(field) is True, "Unexpected preprocessing change: " + field)
    require("ddof=0" in prep["population_zscore_rule"] and
            "log1p" in prep["wholesale_transform"] and
            prep["class_labels_used_for_preprocessing"] is False and
            "P2" in prep["versions_and_transformed_array_hashes"],
            "Invalid mathematical transformation or leak from outcomes")
    graph = frozen.get("graph", {})
    require(graph.get("source_module") == "pcc_graph.py" and
            graph.get("function") == "build_knn_graph" and
            graph.get("k_nn") == 10 and graph.get("metric") == "minkowski" and
            graph.get("p") == 2 and graph.get("n_jobs") == 1 and
            graph.get("nearest_neighbors_algorithm") == "auto" and
            graph.get("graph_directed") is False and
            graph.get("prohibit_CSR_propagation_replacement") is True and
            graph.get("installation_route", "").startswith("build_graph") and
            "neib_list" in graph.get("PCC_storage", "") and
            "-1" in graph.get("PCC_storage", "") and
            graph.get("graph_hash", "").startswith("P2 required"),
            "Source-only approved undirected dense graph must not drift")
    fit = frozen.get("fit", {})
    require(fit.get("backend_order") == list(BACKENDS) and
            fit.get("update_mode") == "sequential" and
            all(fit.get(k) == v for k,v in [
                ("p_grd",0.5),("delta_v",0.1),
                ("deltap",1.0),("dexp",2.0),
                ("max_iter",50000),("es_chk",2000),
            ]) and
            fit.get("early_stop") is False and
            fit.get("milestones") == MILESTONES and
            fit.get("label_mask_replicates_per_dataset") == 8 and
            fit.get("originally_labeled_per_class") == 3 and
            fit.get("total_proposed_fits") == 72 and
            fit.get("max_iter") ==
            draft["experimental_design"]["max_iterations_proposed"],
            "Approved science schedule/fit or kernel parameter drift")
    rng = frozen.get("rng_and_masks", {})
    require(
        "SeedSequence([20261112,registry_index,replicate_index])" in
        rng.get("mask_seed_recipe","") and
        "size=3,replace=False" in rng.get("sampling_recipe","") and
        "SHA256 ASCII" in rng.get("propagation_seed_recipe","") and
        rng.get("source_label_masks_generated") is False and
        "P2" in rng.get("mask_array_hashes",""),
        "Frozen mask/RNG scientific recipe changed or actual labels fabricated")
    analysis = frozen.get("analysis", {})
    require(
        analysis.get("only_initially_unlabeled_nodes") is True and
        analysis.get("primary_checkpoints") == [5000,50000] and
        analysis.get("initial_labelled_nodes_never_counted") is True and
        "EXPLORATORY_DESCRIPTIVE_ONLY_NO_CONFIDENCE_INTERVAL_PVALUES_EQUIVALENCE"
        in analysis.get("inference","") and
        analysis.get("nodes_and_masks_are_not_independent_application_domains")
        is True and
        "failed" in analysis.get("missingness",""),
        "Scientific exploratory-only estimand or failure-handling drift")
    ethics = frozen.get("ethics_and_limitations", {})
    require(
        ethics.get("previously_used_locally", "").startswith(
            "UNKNOWN_NOT_DECLARED") and
        "independence" in ethics.get("science_claim","").lower() is False,
        "Prior use or independence must remain unverified")
    approval = frozen.get("approvals_and_execution", {})
    require(approval.get("P1_scientific_approval") ==
            "EXPLICIT_FOR_S1_TO_S6_IN_CONVERSATION" and
            approval.get("frozen_scientific_payload") is True and
            approval.get("P1_sha256_is_digest_of_this_JSON_without_embedding_its_own_digest")
            is True and
            approval.get("P1_sha256_recorded_in_separate_p1_manifest") is True and
            approval.get("P2_preflight_status") == "NOT_COMPLETED" and
            approval.get("P2_preflight_sha256") is None and
            approval.get("P2_execution_approved") is False and
            approval.get("real_data_fits_authorized") == 0 and
            approval.get("real_data_fits_executed") == 0 and
            approval.get("labels_sampling_or_graph_building_for_real_datasets_authorized_at_P1")
            is False,
            "Scientific consent must not authorize P2 or produce any real data")
    return {
        "schema": "pypcc-P1-approved-scientific-source-only-validation/v1",
        "science_status": "FROZEN_S1_S6_EXPLORATORY_ONLY",
        "scientific_protocol_sha256": canonical_sha256(frozen),
        "selected_source_uci_ids_as_research_protocol": list(IDS),
        "trial_count_IF_future_P2_approved": 72,
        "datasets_or_real_labels_processed": 0,
        "PCC_fits_executed": 0,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol",type=Path,default=FROZEN)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=validate(
        json.loads(args.protocol.read_text(encoding="utf-8")),
        json.loads(DRAFT.read_text(encoding="utf-8")),
        json.loads(DECISIONS.read_text(encoding="utf-8")),
        json.loads(REVIEW.read_text(encoding="utf-8")),
        GRAPH.read_bytes(),
    )
    value=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.write_text(value,encoding="utf-8")
    print(value)


if __name__=="__main__":
    main()
