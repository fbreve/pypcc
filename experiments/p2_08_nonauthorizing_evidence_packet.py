"""P2-08 REVIEW PACKET ONLY: immutable source/graph/mask/synthetic CI evidence.

This is NOT the final execution-preflight manifest, and this code MUST NOT
approve, import, compile or run PCC, download UCI sources, rebuild graphs/masks,
sample labels, or generate any real experimental fit. Its distinct SHA is a
review-only dossier checksum, NEVER an execution authorization token.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from experiments import p2_07c_consolidated_execution_envelope_no_fit as prior

ROOT = Path(__file__).resolve().parents[1]
P1_SHA = prior.P1_SHA
REGISTRY = ROOT / "docs/current_p2_08_review_only_evidence_registry_2026-11-13.json"
PLAN = ROOT / "docs/current_p2_preflight_plan_2026-11-12.json"
PARENT_PR_HEAD = "ea3aedb734de4e542361b1fc30bcfcb304b9167e"
EXPECTED_ARTIFACTS = {
    "source_linux": (43453332637,14053501109),
    "source_windows": (43453332637,14053715816),
    "graph_linux": (43475652889,14063840028),
    "graph_windows": (43475652889,14063660306),
    "graph_concordance": (43475652889,14063161582),
    "parity_linux": (43444184702,14048080950),
    "parity_windows": (43444184702,14046528279),
    "resource_draft": (43491838488,14069414165),
    "synthetic_linux": (43495144447,14071123334),
    "synthetic_windows": (43495144447,14070854067),
    "synthetic_concordance": (43495144447,14071451884),
}
STEPS = (
    "P2_01_CODE_AND_VERSIONS",
    "P2_02_OFFICIAL_SOURCES",
    "P2_03_TRANSFORMED_FEATURES",
    "P2_04_KNN_GRAPH",
    "P2_05_MASKS_AND_SEEDS",
    "P2_06_NATIVE_OBSERVERS",
    "P2_07_RESOURCE_AND_EXPORT_LIMITS",
    "P2_08_SEALED_EXECUTION_GATE",
)
BLOCKERS = (
    "No approved immutable final OS host/image or complete native compiler/SDK/DSO dependency-byte closure",
    "No reviewed actual-graph 50k fit time/CPU/RSS/disk upper bound or committed production-grade supervisor",
    "Proposed 72-case host/RAM/wall/CPU/disk/export thresholds have not been approved by maintainer",
    "Final technical P2 execution manifest not sealed: P2_execution_preflight_sha256 remains null",
    "No distinct explicit maintainer authorization of the exact final technical P2 SHA before any real PCC fit",
)


class ReviewPacketBlocked(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise ReviewPacketBlocked(reason)


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":")).encode("ascii")


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def read_json(path):
    require(path.is_file(), "Required source-free archived evidence missing: " + str(path))
    data = path.read_bytes()
    require(len(data) <= 4*1024*1024, "Evidence JSON unexpectedly large")
    return json.loads(data), sha256(data)


def validate_registry(registry):
    require(registry.get("schema") == "pypcc-P2-08-review-only-evidence-registry/v1"
            and registry.get("P1_scientific_sha256") == P1_SHA and
            registry.get("proposed_source_parent_PR") == 273 and
            registry.get("candidate_p2_08_review_packet_sha256") is None and
            registry.get("final_approved_P2_execution_preflight_sha256") is None and
            registry.get("explicit_maintainer_approval_of_distinct_exact_P2_sha") is False and
            registry.get("final_P2_08_gate_pass") is False and
            registry.get("P2_execution_authorized") is False and
            registry.get("PCC_real_source_fits_executed") == 0 and
            registry.get("original_P2_planning_steps") == list(STEPS),
            "Review-only artifact registry altered its scientific or approval boundary")
    entries = registry.get("evidence", [])
    require(type(entries) is list and len(entries) == len(EXPECTED_ARTIFACTS) and
            {x.get("key") for x in entries} == set(EXPECTED_ARTIFACTS),
            "Exactly eleven original source-free evidence inputs required")
    for entry in entries:
        key = entry["key"]
        run_id, artifact_id = EXPECTED_ARTIFACTS[key]
        require(entry.get("run_id") == run_id and
                entry.get("artifact_id") == artifact_id and
                type(entry.get("artifact_name")) is str and
                entry.get("artifact_name").startswith("p2-") and
                re.fullmatch("[0-9a-f]{64}",
                             entry.get("archive_zip_sha256","")) is not None and
                re.fullmatch("[a-zA-Z0-9._-]+\\.json",
                             entry.get("json_basename","")) is not None and
                entry.get("extracted_destination") ==
                "p2-08-prior-evidence/"+key,
                "Original CI artifact registry identity/path drift: "+key)
    return entries


def no_approved_execution(x, *, label):
    require(x.get("P2_execution_authorized") is False and
            x.get("real_PCC_fits_executed",x.get("PCC_real_source_fits_executed",0)) == 0,
            "Prior evidence falsely claims execution approval or a real fit: "+label)


def source_feature_gate(linux,windows,p1):
    require(linux.get("schema") ==
            "pypcc-P2-02B-03-ordered-source-and-features-no-fit/v1" and
            windows.get("schema") == linux["schema"] and
            linux.get("status") ==
            "EXACT_SOURCE_SCHEMA_AND_APPROVED_FLOAT64_TRANSFORMS_HASHED_NO_PCC_FITS",
            "P2-02B/03 source and feature report schema/status missing")
    projections = []
    for os_name,report in (("Linux",linux),("Windows",windows)):
        no_approved_execution(report,label="P2-02B/03 "+os_name)
        require(report.get("P1_scientific_sha256") == P1_SHA and
                report.get("original_dataset_registry_order") == [212,850,292] and
                report.get("P2_02_complete_gate_pass") is True and
                report.get("P2_03_transformed_X_and_hashes_candidate_pass") is True and
                report.get("P2_03_complete_execution_environment_gate_pass") is False and
                report.get("source_raw_values_persisted_or_uploaded") is False and
                report.get("graphs_generated") == 0 and
                report.get("label_masks_generated") == 0 and
                report.get("PCC_propagations_executed") == 0 and
                report.get("PCC_real_source_fits_executed") == 0 and
                report.get("P2_execution_preflight_sha256") is None and
                len(report.get("datasets",[])) == 3,
                "P2-02/03 source data, ordered hashing or authorization drift")
        items = []
        for ds,expected in zip(report["datasets"],p1["selected_datasets"]):
            n=expected["original_n_rows"]
            features=expected["n_features"]
            require(ds["uci_id"] == expected["uci_id"] and
                    ds["registry_index"] == expected["registry_index"] and
                    ds["original_source_rows"] == n and
                    len(ds["ordered_numeric_predictor_names"]) == features and
                    ds["target_class_support"] == expected["class_support"] and
                    ds["source_zip_sha256"] == expected["zip_sha256"] and
                    ds["member_sha256"] == expected["file_sha256"] and
                    ds["source_original_order_retained_without_sort_or_dedup"] is True and
                    ds["graph_or_masks_generated"] is False and
                    ds["observation_data_rows_exported"] == 0 and
                    ds["raw_or_transformed_matrix_exported"] is False and
                    ds["transformation"]["labels_used_to_select_or_scale_features"] is False and
                    ds["original_target_index_order"]["label_rows"] == n,
                    "P2-02/03 original data schema/labels/columns/row order drift")
            raw,transformed=ds["raw_X"],ds["approved_transformed_X"]
            require(raw["representation"]["shape"] == [n,features] and
                    transformed["representation"]["shape"] == [n,features] and
                    raw["representation"]["ordered_predictor_names"] ==
                    ds["ordered_numeric_predictor_names"] and
                    transformed["representation"]["ordered_predictor_names"] ==
                    ds["ordered_numeric_predictor_names"],
                    "P2-03 raw/transformed dimension or column order drift")
            name=ds["ordered_numeric_predictor_names"]
            if ds["uci_id"] == 292:
                require(name == expected["predictive_features"] and
                        ds["excluded_model_predictors"] ==
                        expected["excluded_predictors"] and
                        ds["transformation"]["method"] ==
                        "float64_np_log1p_then_ddof0_population_zscore",
                        "Wholesale Channel/Region leakage or transformation drift")
            else:
                require(ds["transformation"]["method"] ==
                        "float64_ddof0_population_zscore",
                        "Original population zscore changed")
            hashes = {}
            for key,value in (("raw",raw),("transformed",transformed)):
                full=value["shape_dtype_feature_order_and_numeric_bytes_sha256"]
                require(re.fullmatch("[0-9a-f]{64}",full) is not None,
                        "Invalid approved source order hash")
                hashes[key]=full
            target=ds["original_target_index_order"]["original_target_order_sha256"]
            require(re.fullmatch("[0-9a-f]{64}",target) is not None,
                    "Target order digest absent")
            items.append({"uci_id":ds["uci_id"],
                          "n_rows":n,"n_features":features,
                          "original_class_support":ds["target_class_support"],
                          "feature_order":name,
                          "raw_ordered_features_sha256":hashes["raw"],
                          "transformed_ordered_features_sha256":hashes["transformed"],
                          "original_target_row_order_sha256":target,
                          "zip_sha256":ds["source_zip_sha256"],
                          "member_sha256":ds["member_sha256"]})
        projections.append(items)
    require(projections[0] == projections[1],
            "P2-02B/03 Linux and Windows ordered numeric/label content SHA drift")
    return projections[0]


def graph_mask_gate(linux,windows,concordance,p1,source):
    require(concordance.get("schema") ==
            "pypcc-P2-04-05-cross-platform-hash-concordance/v1" and
            concordance.get("P2_04_05_cross_platform_hash_gate_pass") is True and
            concordance.get("P1_scientific_sha256") == P1_SHA and
            concordance.get("P2_execution_authorized") is False and
            concordance.get("PCC_propagations_executed") == 0 and
            concordance.get("selected_uci_ids") == [212,850,292] and
            concordance.get("verified_native_seed_count") == 72,
            "Two-platform exact original graph/mask/seed concordance not proven")
    fingerprint = []
    for os_name,report in (("Linux",linux),("Windows",windows)):
        require(report.get("schema") ==
                "pypcc-P2-04-05-frozen-UCI-graph-and-mask-preflight/v1" and
                report.get("status") ==
                "EXACT_THREE_KNN_GRAPHS_AND_24_MASKS_WITH_72_NATIVE_SEEDS_NO_PCC_FITS" and
                report.get("selected_UCI_order") == [212,850,292] and
                report.get("P1_scientific_sha256") == P1_SHA and
                report.get("graphs_built_on_real_source_features") == 3 and
                report.get("real_source_label_masks_generated") == 24 and
                report.get("backend_domain_separated_seeds_derived_not_used") == 72 and
                report.get("P2_04_graph_candidate_pass") is True and
                report.get("P2_05_mask_and_seeds_candidate_pass") is True and
                report.get("P2_04_05_cross_platform_hash_gate_pass") is False and
                report.get("graph_is_undirected") is True and
                report.get("graph_padding_preserved_minus_one") is True and
                report.get("PCC_propagations_executed") == 0 and
                report.get("PCC_real_data_fits_executed") == 0 and
                report.get("P2_execution_authorized") is False and
                report.get("raw_source_rows_or_labels_exported") is False and
                report.get("raw_graph_or_mask_arrays_exported") is False,
                "P2-04/05 graph/mask no-fit evidence not strict")
        entries = []
        for idx,(item,expected,features) in enumerate(
                zip(report["datasets"],p1["selected_datasets"],source)):
            uci_id = expected["uci_id"]
            g=item["graph"]
            width=g["hash_serialization"]["neib_list_shape"][1]
            require(item["uci_id"] == uci_id and item["registry_index"] == idx and
                    item["original_source_row_count"] == expected["original_n_rows"] and
                    item["approved_P1_archive_sha256"] == expected["zip_sha256"] and
                    item["approved_P1_member_sha256"] == expected["file_sha256"] and
                    item["verified_P2_03_raw_X_sha256"] ==
                    features["raw_ordered_features_sha256"] and
                    item["verified_P2_03_transformed_X_sha256"] ==
                    features["transformed_ordered_features_sha256"] and
                    item["original_class_support"] == expected["class_support"] and
                    g["hash_serialization"]["neib_list_shape"] ==
                    [expected["original_n_rows"],width] and
                    g["hash_serialization"]["padding"] == -1 and
                    g["dense_unused_padding_all_minus_one"] is True and
                    g["edges_without_reciprocals"] == 0 and
                    g["self_loops_found"] == 0 and
                    g["minimum_degree"] >= 1 and
                    len(item["masks"]) == 8 and
                    len(item["propagation_seeds_not_executed"]) == 24,
                    "P2-04/05 source/graph/padding or masks drift")
            gh=g["exact_full_dense_graph_SHA256"]
            require(gh == concordance["dense_graph_hashes"][idx],
                    "Exact source-derived dense graph hash differs from cross-OS manifest")
            masks=[]
            for replicate,m in enumerate(item["masks"]):
                require(m["replicate"] == replicate and
                        m["selected_total"] ==
                        3*len(expected["class_support"]) and
                        m["mask_array_exported"] is False,
                        "P2-05 mask selection/replicate schedule changed")
                mh=m["mask_sha256"]
                require(mh == concordance["mask_hashes_by_dataset"][str(uci_id)][replicate],
                        "Exact full original-source mask SHA differs")
                masks.append(mh)
            seeds = item["propagation_seeds_not_executed"]
            expected_backends=p1["fit"]["backend_order"]
            for replicate in range(8):
                for bidx,be in enumerate(expected_backends):
                    seed=seeds[replicate*3+bidx]
                    require(seed["dataset_id"] == uci_id and
                            seed["replicate"] == replicate and
                            seed["backend"] == be and
                            seed["registry_index"] == idx and
                            type(seed["native_seed_uint32"]) is int and
                            0<=seed["native_seed_uint32"]<=2**32-1,
                            "Unapproved backend seed/domain/replicate drift")
            entries.append({
                "uci_id":uci_id,
                "graph_SHA256":gh,
                "graph_dense_width":width,
                "n_masks":8,
                "mask_SHA256_list":masks,
                "native_seeds_sha256":sha256(canonical(seeds)),
            })
        fingerprint.append(entries)
    require(fingerprint[0] == fingerprint[1],
            "Linux and Windows actual graph, mask or 72 seed fingerprints differ")
    return fingerprint[0]


def parity_gate(linux,windows):
    details=[]
    for platform,report in (("Linux",linux),("Windows",windows)):
        no_approved_execution(report,label="P2-06 "+platform)
        require(report.get("schema") ==
                "pypcc-P2-06-synthetic-pinned-backend-parity/v1" and
                report.get("P1_scientific_sha256") == P1_SHA and
                report.get("original_synthetic_suite_test_count") == 21 and
                report.get("test_errors") == 0 and
                report.get("test_failures") == 0 and
                report.get("P2_06_synthetic_parity_candidate_pass") is True and
                report.get("P2_06_full_execution_gate_pass") is False and
                report.get("Numba_native_IR_and_ABI_frozen_for_final_P2") is False and
                report.get("real_source_datasets_loaded") == 0 and
                report.get("real_PCC_fits_executed") == 0 and
                report.get("P2_execution_preflight_sha256") is None and
                report.get("synthetic_PCC_executed_in_tests") is True,
                "P2-06 synthetic observer source/parity or final-GO drift")
        details.append({
            "platform":platform,
            "original_three_original_synthetic_test_suites_pass":True,
            "original_synthetic_suite_test_count":21,
            "original_observer_git_heads":report["source_observer_commit_heads"],
            "historical_original_native_binary_sha256":report[
                "exact_native_binary_used"],
        })
    require(details[0]["original_observer_git_heads"] ==
            details[1]["original_observer_git_heads"],
            "Synthetic original observer code differs across OS")
    return details


def resource_gate(draft, linux,windows,cross,policy,p1):
    prior.validate_policy(policy,p1)
    require(draft.get("schema") ==
            "pypcc-P2-07C-proposed-72-case-resource-and-export-admission/v1" and
            draft.get("P1_scientific_sha256") == P1_SHA and
            draft.get("case_schedule",{}).get("count") == 72 and
            draft["case_schedule"]["all_cases_not_executed"] is True and
            draft.get("approved_execution_host_selected") is False and
            draft.get("candidate_real_PCC_process_RSS_peak_or_disk_measured") is False and
            draft.get("native_compiler_SDK_runtime_OS_complete_byte_lock_proved") is False and
            draft.get("actual_P2_07_resource_gate_pass") is False and
            draft.get("P2_execution_preflight_sha256") is None and
            draft.get("P2_execution_authorized") is False and
            draft.get("real_PCC_fits_executed") == 0 and
            draft.get("candidate_policy_sha256") == sha256(canonical(policy)) and
            draft.get("candidate_guardrails_not_approved") ==
            policy["candidate_guardrails"] and
            draft["case_schedule"]["canonical_JSON_SHA256"] ==
            sha256(canonical(prior.case_order(p1))),
            "P2-07C candidate resource policy/case-order cross-proof drift")
    require(cross.get("schema") ==
            "pypcc-P2-07D-two-OS-exact-50k-synthetic-watchdog-evidence/v1" and
            cross.get("frozen_P1_sha256") == P1_SHA and
            cross.get("same_9_artificial_workloads_on_both_OS") is True and
            cross.get("observed_array_snapshot_shapes_agree") is True and
            cross.get("per_OS_synthetic_50k_cases") == 9 and
            cross.get("total_synthetic_fits_across_both_OS") == 36 and
            cross.get("no_50k_real_UCI_resource_upper_bound_proved") is True and
            cross.get("P2_07_full_resources_gate_pass") is False and
            cross.get("P2_execution_authorized") is False and
            cross.get("real_PCC_fits_executed") == 0,
            "P2-07D synthetic OS watchdog concordance does not preserve blockers")
    details=[]
    workload=None
    for system,report in (("Linux",linux),("Windows",windows)):
        no_approved_execution(report,label="P2-07D "+system)
        require(report.get("schema") ==
                "pypcc-P2-07D-synthetic-50k-case-resource-enforcement/v1" and
                report.get("status") ==
                "NINE_THREE_BACKEND_50K_SYNTHETIC_RESOURCE_CASES_PASS_NO_REAL_DATA_GO" and
                report.get("P1_scientific_sha256") == P1_SHA and
                report.get("candidate_P2_07C_policy_sha256") ==
                draft["candidate_policy_sha256"] and
                report.get("synthetic_PCC_fits_executed") == 18 and
                report.get("real_UCI_data_loaded") is False and
                report.get("real_source_graph_or_masks_loaded") is False and
                report.get("synthetic_results_not_a_real_UCI_graph_resource_upper_bound") is True and
                report.get("P2_07_full_resources_gate_pass") is False and
                report.get("P2_01_final_execution_toolchain_lock_pass") is False and
                report.get("P2_execution_preflight_sha256") is None and
                report.get("P2_execution_authorized") is False and
                report.get("real_PCC_fits_executed") == 0 and
                report.get("one_synthetic_case_at_a_time") is True and
                report.get("full_P1_milestones_exactly") ==
                p1["fit"]["milestones"],
                "P2-07D artificial 50k load parity/resource identity changed")
        rows=report.get("synthetic_50k_original_vs_observed_results",[])
        ids=[(x["uci_dimension_reference_only"],x["backend"]) for x in rows]
        require(len(rows)==9 and ids ==
                [(ds["uci_id"],be) for ds in p1["selected_datasets"]
                 for be in ("numpy","numba","cython")],
                "Not all nine synthetic workload resource cases present")
        bounds=policy["candidate_guardrails"]
        for x in rows:
            require(x.get("same_backend_full_state_and_prediction_parity") is True and
                    x.get("synthetic_fit_count_original_plus_observed") == 2 and
                    x.get("real_PCC_fits_executed") == 0 and
                    x.get("whole_process_peak_bytes",0)>0 and
                    x["whole_process_peak_bytes"] <=
                    bounds["max_one_fit_peak_process_resident_bytes"] and
                    x.get("observed_eight_snapshots_array_bytes",0)>0 and
                    x["observed_eight_snapshots_array_bytes"] <=
                    bounds["max_one_fit_snapshot_array_bytes"] and
                    x.get("complete_synthetic_50k_fit_observed_wall_s",0)>0 and
                    x["complete_synthetic_50k_fit_observed_wall_s"] <=
                    bounds["max_one_fit_wall_seconds"],
                    "Synth 50k parity/snapshot/memory/wall cap breached")
        snapshot=[x["observed_eight_snapshots_array_bytes"] for x in rows]
        require(workload is None or workload==snapshot,
                "Cross-OS synthetic 50k checkpoint shapes differ")
        workload=snapshot
        details.append({
            "platform":system,"synthetic_50k_case_count":len(rows),
            "synthetic_full_fits_including_baselines":18,
            "peak_whole_process_bytes":max(x["whole_process_peak_bytes"] for x in rows),
            "maximum_observer_wall_seconds":max(
                x["complete_synthetic_50k_fit_observed_wall_s"] for x in rows),
            "max_eight_milestone_snapshot_bytes":max(snapshot),
            "resource_limit_is_real_UCI_upper_bound":False,
        })
    return details


def validate_original_p2_plan(plan):
    require(plan.get("schema") == "pypcc-P2-execution-preflight-plan/v1" and
            plan.get("parent_science",{}).get("P1_scientific_sha256") == P1_SHA and
            [x.get("id") for x in plan.get("technical_steps",[])] == list(STEPS) and
            plan.get("approval_boundary","").find("explicit")!=-1 and
            plan.get("current_checks",{}).get("preflight_SHA256") is None and
            plan["current_checks"].get("user_execution_approval_for_P2_SHA") is False and
            plan["current_checks"].get("real_PCC_fit_count") == 0,
            "P2 original final-SHA approval contract changed")
    return True


def assemble(registry, collected):
    p1=prior.read_approved_P1()
    validate_original_p2_plan(read_json(PLAN)[0])
    policy=prior.load(prior.POLICY)
    inputs={key:data for key,(data,digest) in collected.items()}
    source=source_feature_gate(
        inputs["source_linux"],inputs["source_windows"],p1)
    graph=graph_mask_gate(
        inputs["graph_linux"],inputs["graph_windows"],
        inputs["graph_concordance"],p1,source)
    parity=parity_gate(
        inputs["parity_linux"],inputs["parity_windows"])
    resources=resource_gate(
        inputs["resource_draft"],inputs["synthetic_linux"],
        inputs["synthetic_windows"],inputs["synthetic_concordance"],policy,p1)
    require(len(source)==len(graph)==3 and len(parity)==len(resources)==2,
            "Evidence categories incomplete")
    gates=[
        {"id":STEPS[0],"candidate_evidence":"P2-01F new native build/13 exact archived wheel replay checked through P2-07C",
         "review_status":"PARTIAL_NATIVE_REPRODUCED_FULL_HOST_ABI_LOCK_BLOCKED",
         "final_execution_gate_pass":False},
        {"id":STEPS[1],"candidate_evidence":"Original member hashes, rows, column/target order and class supports cross-OS exact",
         "review_status":"CANDIDATE_SOURCE_IDENTITY_PASS",
         "final_execution_gate_pass":False},
        {"id":STEPS[2],"candidate_evidence":"Approved raw and transformed float64 source-order byte SHA concordant across OS",
         "review_status":"CANDIDATE_FLOAT64_TRANSFORM_PASS",
         "final_execution_gate_pass":False},
        {"id":STEPS[3],"candidate_evidence":"Three true official-source dense symmetric kNN graphs and exact ordered-neighbor hashes agree cross-OS",
         "review_status":"CANDIDATE_REAL_SOURCE_GRAPH_HASH_PASS_NO_PCC_FIT",
         "final_execution_gate_pass":False},
        {"id":STEPS[4],"candidate_evidence":"24 exact real-source PCG64 mask hashes and 72 domain-separated native seed values verified; not used",
         "review_status":"CANDIDATE_MASK_AND_SEED_PASS_NOT_EXECUTED",
         "final_execution_gate_pass":False},
        {"id":STEPS[5],"candidate_evidence":"21/21 original within-backend synthetic observer tests per OS; no final native-specialization lock",
         "review_status":"CANDIDATE_SYNTHETIC_PARITY_PASS_FINAL_LOCK_BLOCKED",
         "final_execution_gate_pass":False},
        {"id":STEPS[6],"candidate_evidence":"2x9 50k artificial original-vs-observed cases, forced resource kill tests, proposed caps only",
         "review_status":"SYNTHETIC_RESOURCE_ENFORCEMENT_PASS_REAL_50K_BUDGET_BLOCKED",
         "final_execution_gate_pass":False},
        {"id":STEPS[7],"candidate_evidence":"Dossier for review only, no final technical manifest/host/budget/consent",
         "review_status":"NOT_SEALED_NO_EXECUTION_AUTHORIZATION",
         "final_execution_gate_pass":False},
    ]
    # Critical: "review_packet_sha256" is deliberately NOT a final P2 SHA.
    packet={
        "schema":"pypcc-P2-08-REVIEW-ONLY-nonauthorizing-evidence-packet/v1",
        "status":"SOURCE_GRAPH_MASK_SYNTHETIC_CANDIDATE_EVIDENCE_REVIEWED_FINAL_P2_BLOCKED",
        "frozen_P1_scientific_sha256":P1_SHA,
        "base_research_PR":273,
        "base_research_commit_git_sha1":PARENT_PR_HEAD,
        "stage_gate_assessment":gates,
        "verified_original_UCI_source_and_transformed_matrix_fingerprints":source,
        "verified_original_UCI_dense_graph_and_mask_fingerprints":graph,
        "verified_original_backend_synthetic_parity":parity,
        "verified_synthetic_50k_OS_process_resource_pilots":resources,
        "approved_scientific_design_fit_count":72,
        "proposed_host":{"Linux_primary_candidate":"CPython 3.12.16",
                         "Windows_crosscheck_candidate":"CPython 3.12.10",
                         "approved_immutable_final_host":None},
        "proposed_policy_canonical_sha256":sha256(canonical(policy)),
        "proposed_resource_bounds_not_approved":policy["candidate_guardrails"],
        "original_72_case_ID_only_schedule_sha256":
            sha256(canonical(prior.case_order(p1))),
        "source_evidence":{
            x["key"]:{"run_id":x["run_id"],
                       "artifact_id":x["artifact_id"],
                       "github_action_archive_zip_sha256_record":
                           x["archive_zip_sha256"],
                       "actual_extracted_JSON_sha256":
                           collected[x["key"]][1],
                       "source":x["artifact_name"]}
            for x in registry["evidence"]
        },
        "remaining_MANDATORY_blockers_before_final_execution_manifest":list(BLOCKERS),
        "full_native_toolchain_and_execution_host_immutable":False,
        "real_UCI_50k_CPU_RSS_disk_limits_verified":False,
        "approved_final_resource_and_export_policy":False,
        "scientific_72_case_execution_started":False,
        "final_execution_preflight_manifest_created":False,
        "final_execution_preflight_SHA256":None,
        "separate_exact_P2_SHA_maintainer_execution_consent_obtained":False,
        "all_eight_P2_execution_gates_pass":False,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
        "this_review_only_stage_UCI_downloads":0,
        "this_review_only_stage_real_graph_or_mask_construction":0,
        "this_review_only_stage_synthetic_or_real_PCC_fits":0,
        "P2_review_dossier_is_NOT_an_execution_preflight_SHA":True,
    }
    require(not any(g["final_execution_gate_pass"] for g in gates) and
            packet["real_PCC_fits_executed"]==0 and
            packet["final_execution_preflight_SHA256"] is None and
            packet["P2_execution_authorized"] is False,
            "Accidental or forged P2 approval must be impossible")
    return packet


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    registry=read_json(REGISTRY)[0]
    entries=validate_registry(registry)
    root=args.archive_root.resolve()
    collected={}
    for entry in entries:
        key=entry["key"]
        json_path=root/key/entry["json_basename"]
        collected[key]=read_json(json_path)
    packet=assemble(registry,collected)
    digest=sha256(canonical(packet))
    result={
        "schema":"pypcc-P2-08-REVIEW-PACKET-CHECKSUM-not-an-execution-seal/v1",
        "review_packet_SHA256_NOT_P2_execution_SHA":digest,
        "review_packet":packet,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    }
    args.output.write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "review_packet_SHA256_NOT_P2_execution_SHA":digest,
        "candidate_source_datasets":len(
            packet["verified_original_UCI_source_and_transformed_matrix_fingerprints"]),
        "original_real_data_knn_graphs_in_old_P2_04_evidence":3,
        "original_real_source_masks_in_old_P2_05_evidence":24,
        "synthetic_only_50k_cases_reused_from_older_CI":18,
        "remaining_final_P2_execution_blockers":len(BLOCKERS),
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))


if __name__=="__main__":
    main()
