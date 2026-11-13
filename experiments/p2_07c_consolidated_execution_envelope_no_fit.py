"""P2-07C prospective execution-envelope ADMISSION DRY RUN, never execute PCC.

Reads only previously passed archived P2-07A static array budgets, P2-07B
SYNTHETIC CPU/working-set pilot, and P2-01F prospective native-build proofs.
Checks source/production scientific pins and verifies exact newly compiled
binary bytes inside their archived native seals. Defines *candidate* resource
hard stops and a frozen 72-case ORDER WITHOUT loading source records, graphs,
masks or running any particle propagation.

A proposed limit or a passing synthetic run is NOT empirical proof that a real
50,000-iteration PCC fit would meet that limit, and cannot grant P2/P3 approval.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "docs/current_p2_07c_candidate_operational_envelope_2026-11-13.json"
P1 = ROOT / "docs/current_p1_frozen_scientific_protocol_2026-11-12.json"
P1_MANIFEST = ROOT / "docs/current_p1_scientific_sha_manifest_2026-11-12.json"
P1_SHA = "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166"

EXPECTED_BUDGET = {
    "maximum_hypothetical_co_resident_array_payload_bytes_any_one_fit": 650012,
    "total_eight_checkpoint_array_payloads_all_72_fits_bytes": 14079360,
    "total_intended_fits": 72,
    "total_proposed_PCC_iterations_all_fits": 3600000,
    "total_particle_update_opportunities_all_fits": 25200000,
}
SOURCES = (212, 850, 292)
BACKENDS = (
    "numpy_sequential",
    "numba_sequential_nopython",
    "cython_sequential_cached_or_legacy_as_defined_by_early_stop",
)
PLATFORMS = {
    "ubuntu-latest": {"os": "Linux", "python": "3.12.16"},
    "windows-latest": {"os": "Windows", "python": "3.12.10"},
}
NATIVE_SHA = {
    "Linux": {
        "production_pcc_step":
            "cec70629e10b1f6e9f53274d571a52ad2177d31b89d2c189840f3abda2944a8a",
        "isolated_pcc_step_live_research":
            "1ea9f1c151df795071b3fe449df2a3e35ab72bb79156b073df9f8297c66d899f",
    },
    "Windows": {
        "production_pcc_step":
            "ad585e568cba1cf18f26fa0def795cda3597e81ecf4338c54f3d12ea840fc077",
        "isolated_pcc_step_live_research":
            "55e0516366ae8b291b39dc39c3d013379b163f6f9a5f10a7ed41e012f57e0e85",
    },
}
ARTIFACT_RUNS = {
    "P2_07A": 43478839756,
    "P2_07B": 43483012485,
    "P2_01F": 43488066633,
}


class AdmissionBlocked(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AdmissionBlocked(message)


def canonical(data):
    return json.dumps(data, ensure_ascii=True, sort_keys=True,
                      allow_nan=False, separators=(",", ":")).encode("ascii")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    require(path.is_file(), "Required archived evidence/contract unavailable: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def read_approved_P1():
    p1, manifest = load(P1), load(P1_MANIFEST)
    require(sha256(canonical(p1)) == P1_SHA ==
            manifest.get("approved_P1_scientific_protocol_sha256"),
            "P1 scientific exact immutable digest does not match")
    require(p1.get("fit", {}).get("max_iter") == 50000 and
            p1["fit"]["early_stop"] is False and
            p1["fit"]["total_proposed_fits"] == 72 and
            p1.get("graph", {}).get("graph_directed") is False and
            p1["graph"]["PCC_storage"].startswith("dense int64 neib_list") and
            p1["approvals_and_execution"]["P2_execution_approved"] is False,
            "Approved P1 graph/fit/authorization differs")
    return p1


def validate_policy(policy, p1):
    require(policy.get("schema") ==
            "pypcc-P2-07C-candidate-operational-envelope/v1" and
            policy.get("status") == "DRAFT_FOR_MAINTAINER_REVIEW_NOT_EXECUTION_APPROVAL" and
            policy.get("P1_scientific_sha256") == P1_SHA and
            policy.get("P2_execution_authorized") is False and
            policy.get("real_PCC_fits_authorized") == 0 and
            policy.get("real_PCC_fits_executed") == 0 and
            policy.get("P2_execution_preflight_sha256") is None and
            policy.get("P2_08_technical_manifest_sha256") is None and
            policy.get("P2_07_full_resources_gate_pass") is False and
            policy.get("P2_01_final_execution_toolchain_lock_pass") is False,
            "Proposed resource envelope must NEVER assert execution authority")
    require(policy.get("evidence") == {
        "P2_07A_metadata_run_id": ARTIFACT_RUNS["P2_07A"],
        "P2_07B_synthetic_run_id": ARTIFACT_RUNS["P2_07B"],
        "P2_01F_new_native_run_id": ARTIFACT_RUNS["P2_01F"],
    }, "Prospective resource limits must reference known audited runs")
    host = policy.get("proposed_execution_host", {})
    require(host.get("physical_or_immutable_execution_host_selected") is False and
            host.get("host_machine_config_sha256") is None and
            host.get("compiler_SDK_system_byte_closure_complete") is False and
            host.get("candidate_primary", "").startswith("Linux CPython 3.12.16") and
            host.get("candidate_crosscheck", "").startswith("Windows CPython 3.12.10"),
            "Prospective candidate may not claim immutable host selection")
    plan = policy.get("fit_schedule", {})
    p1fit = p1["fit"]
    require(plan.get("datasets_uci_in_order") == list(SOURCES) and
            plan.get("backends_in_order") == list(BACKENDS) ==
            p1fit["backend_order"] and
            plan.get("n_fits") == p1fit["total_proposed_fits"] == 72 and
            plan.get("masks_per_dataset") == p1fit["label_mask_replicates_per_dataset"] == 8 and
            plan.get("max_iter_per_fit") == p1fit["max_iter"] == 50000 and
            plan.get("milestones") == p1fit["milestones"] and
            plan.get("early_stop") is False and
            plan.get("max_parallel_fits") == 1 and
            plan.get("scientific_retries_allowed") == 0 and
            plan.get("outcome_dependent_reordering") is False,
            "Proposed run order, concurrency or no-retry semantics differ from P1")
    g = policy.get("candidate_guardrails", {})
    required_exact = {
        "min_host_installed_ram_bytes": 8*1024**3,
        "max_one_fit_peak_process_resident_bytes": 2*1024**3,
        "min_free_disk_before_start_bytes": 12*1024**3,
        "max_one_fit_wall_seconds": 30*60,
        "max_one_fit_process_cpu_seconds": 60*60,
        "max_entire_study_wall_seconds": 48*60*60,
        "max_one_fit_snapshot_array_bytes": 64*1024**2,
        "max_one_fit_aggregate_json_bytes": 1024**2,
        "max_one_fit_text_log_bytes": 2*1024**2,
        "max_one_fit_scratch_bytes": 1024**3,
        "max_all_72_fits_persisted_evidence_bytes": 1024**3,
    }
    require(g == required_exact and
            72*g["max_one_fit_wall_seconds"] <=
            g["max_entire_study_wall_seconds"] and
            72*(g["max_one_fit_aggregate_json_bytes"] +
                g["max_one_fit_text_log_bytes"]) <
            g["max_all_72_fits_persisted_evidence_bytes"],
            "Candidate process, disk, wall, CPU or archival guardrails changed")
    exports = policy.get("export_and_failure_policy", {})
    expected_false = (
        "original_feature_rows_or_original_labels_in_public_CI",
        "graph_neighbor_arrays_or_mask_indices_in_public_CI",
        "per_node_checkpoint_vectors_exported",
        "failed_fit_may_be_silently_replaced",
        "early_stop_or_adaptive_retry_allowed",
    )
    expected_true = (
        "per_fit_aggregate_transition_counts_and_tie_counts_only",
        "aggregate_original_unlabeled_denominator_and_postfit_accuracy_allowed",
        "baseline_recorded_before_observed_backend_interpretation",
        "entire_72_fit_summary_requires_all_72_completed",
        "failures_append_only_with_original_case_id",
        "complete_per_backend_RNG_state_verification_required",
    )
    require(all(exports.get(x) is False for x in expected_false) and
            all(exports.get(x) is True for x in expected_true) and
            policy.get("candidate_budget_not_measured_upper_bound") is True and
            policy.get("approved_execution_host_and_resource_limits") is False,
            "Export or approval/failure policy may leak data or imply P2 GO")
    return g


def case_order(p1):
    # IDs ONLY: actual frozen 24 source masks are NOT materialized.
    case_ids = []
    for item in p1["selected_datasets"]:
        for replicate in range(8):
            for backend in BACKENDS:
                case_ids.append({
                    "uci_id": item["uci_id"],
                    "replicate": replicate,
                    "backend": backend,
                    "status": "NOT_RUN_NO_APPROVAL",
                })
    require(len(case_ids) == 72 and
            len({(x["uci_id"], x["replicate"], x["backend"])
                 for x in case_ids}) == 72 and
            [x["uci_id"] for x in case_ids[::24]] == list(SOURCES),
            "The no-fit 72-case ID schedule has drifted")
    return case_ids


def validate_arrays(report):
    require(report.get("schema") ==
            "pypcc-P2-07-meta-only-array-and-snapshot-budget/v1" and
            report.get("status") ==
            "SNAPSHOT_ARRAY_BYTES_WITHIN_CAP_REAL_EXECUTION_CPU_RSS_DISK_STILL_UNVERIFIED" and
            report.get("P1_scientific_sha256") == P1_SHA and
            report.get("parent_P2_04_05_workflow_run") == 43475652889 and
            report.get("real_dataset_downloads_or_rows_loaded_in_P2_07") == 0 and
            report.get("PCC_real_data_fits_executed") == 0 and
            report.get("P2_execution_authorized") is False and
            report.get("P2_execution_preflight_sha256") is None,
            "P2-07A resource evidence source/approval drift")
    accounting = report.get("accounting", {})
    for key, value in EXPECTED_BUDGET.items():
        require(accounting.get(key) == value,
                "Frozen snapshot and array budget arithmetic drifted: " + key)
    require(accounting.get("array_payload_not_a_process_RSS_upper_bound") is True and
            accounting.get("projection_not_a_CPU_time_benchmark") is True,
            "P2-07A array geometry misrepresented as actual resource bound")
    gates = report.get("safety_and_approval_gates", {})
    require(gates.get("P2_07_snapshot_payload_within_original_64MiB_per_fit_cap") is True and
            gates.get("P2_07_full_resources_gate_pass") is False and
            gates.get("P2_07_total_CPU_wall_time_budget_verified") is False and
            gates.get("P2_07_native_backend_peak_RSS_budget_verified") is False,
            "Existing preflight resource blocker was silently removed")
    widths = [(x["uci_id"], x["dense_neighbor_width"]) for x in accounting["datasets"]]
    require(widths == [(212, 22), (850, 20), (292, 24)],
            "Actual graph widths or source order not preserved")
    return {"source_run": ARTIFACT_RUNS["P2_07A"],
            "max_hypothetical_array_payload_bytes":
            accounting["maximum_hypothetical_co_resident_array_payload_bytes_any_one_fit"],
            "all_72_snapshot_array_bytes":
            accounting["total_eight_checkpoint_array_payloads_all_72_fits_bytes"],
            "source_graph_dense_neighbor_widths": dict((str(a), b) for a, b in widths)}


def validate_synthetic_pilot(report, system):
    require(report.get("schema") ==
            "pypcc-P2-07B-original-three-backend-synthetic-CPU-RSS-pilot/v1" and
            report.get("status") ==
            "THREE_SYNTHETIC_BACKEND_PROCESS_MEASUREMENTS_NO_REAL_P2_GO" and
            report.get("P1_scientific_sha256") == P1_SHA and
            report.get("no_real_UCI_data_loaded") is True and
            report.get("all_synthetic_observer_pair_parity_pass") is True and
            report.get("memory_measurement_not_equivalent_across_Windows_Linux") is True and
            report.get("NOT_a_reliable_upper_bound_on_P2_50k_runtime_or_RAM") is True and
            report.get("P2_07_full_resources_gate_pass") is False and
            report.get("real_PCC_fits_executed") == 0 and
            report.get("P2_execution_authorized") is False and
            report.get("P2_execution_preflight_sha256") is None and
            report.get("candidate_python_patch_and_OS", {}).get("os") == system and
            report["candidate_python_patch_and_OS"]["python"] ==
            ("3.12.16" if system == "Linux" else "3.12.10"),
            "Synthetic pilot did not preserve frozen P1 and false P2 GO")
    samples = report.get("samples", [])
    require([x.get("backend") for x in samples] == ["numpy", "numba", "cython"],
            "All three original synthetic benchmark backends required")
    expected_sources = {"numpy":"numpy","numba":"numba","cython":"cython"}
    actual = {}
    for x in samples:
        backend = x["backend"]
        require(x.get("synthetic_graph_only") is True and
                x.get("no_UCI_source_downloaded_or_parsed") is True and
                x.get("synthetic_nodes") == 500 and
                x.get("synthetic_pcc_iterations_per_fit") == 1000 and
                x.get("synthetic_PCC_fits_did_execute") is True and
                x.get("real_PCC_fits_executed") == 0 and
                x.get("peak_is_NOT_backend_fit_incremental_RAM") is True and
                x.get("verified_paired_cpu", {}).get(
                    "original_paired_benchmark_checked_within_own_backend") is True,
                "Original paired synthetic pilot is not valid no-real-data evidence")
        peak = x.get("child_process_peak_resident_or_working_set", {}).get(
            "peak_memory_bytes")
        normal = x["verified_paired_cpu"].get(
            "median_baseline_CPU_seconds_per_complete_synthetic_fit")
        observed = x["verified_paired_cpu"].get(
            "median_observer_CPU_seconds_per_complete_synthetic_fit")
        require(type(peak) is int and 0 < peak < 4*1024**3 and
                type(normal) in (int,float) and 0 < normal < 100 and
                type(observed) in (int,float) and 0 < observed < 100,
                "Invalid original CPU/whole-process RSS evidence")
        actual[backend] = {"peak_process_bytes": peak,
                           "paired_baseline_CPU_s": normal,
                           "paired_observer_CPU_s": observed,
                           "synthetic_iterations_per_fit": 1000}
    return {"platform": system,
            "source_run": ARTIFACT_RUNS["P2_07B"],
            "backend_synthetic_measurements": actual,
            "same_backend_comparison_only": True,
            "not_a_real_P2_50k_execution_bound": True}


def validate_new_native(report, folder, system):
    require(report.get("schema") ==
            "pypcc-P2-01F-prospective-native-toolchain-byte-ledger/v1" and
            report.get("status") ==
            "NEW_NATIVE_BINARIES_BUILT_COMPILER_BYTES_RECORDED_PARTIAL_SDK_LEDGER" and
            report.get("P1_scientific_sha256") == P1_SHA and
            report.get("candidate_platform", {}).get("system") == system and
            report["candidate_platform"].get("python") ==
            ("3.12.16" if system == "Linux" else "3.12.10") and
            report.get("original_production_source_and_P1_audited_in_CI") is True and
            report.get("new_native_exact_source_smoke_proved") is True and
            report.get("new_Cython_binaries_not_claimed_to_be_old_P2_01C_bytes") is True and
            report.get("complete_compiler_SDK_system_include_and_binary_link_closure_sealed") is False and
            report.get("P2_01_final_execution_toolchain_lock_pass") is False and
            report.get("real_PCC_fits_executed") == 0 and
            report.get("P2_execution_authorized") is False and
            report.get("P2_execution_preflight_sha256") is None,
            "New native candidate provenance cannot be treated as complete toolchain")
    packages = report.get("candidate_installed_package_versions", {})
    require(type(packages) is dict and len(packages) == 13,
            "Candidate wheel inventory must contain all 13 reviewed packages")
    # Frozen installed inventory uses normalized lowercase distribution keys.
    require(packages.get("numpy") == "2.5.4" and
            packages.get("cython") == "3.3.1" and
            packages.get("numba") == "0.68.1" and
            len(packages) == 13,
            "Candidate native build did not use exact reviewed package version pins")
    native = report.get("new_Cython_production_and_research_binaries", {})
    require(set(native) == set(NATIVE_SHA[system]),
            "Two separately compiled original/isolated Cython binaries required")
    binaries = {}
    for label, expected in NATIVE_SHA[system].items():
        entry = native[label]
        path = folder / "new-exact-native-seal" / entry["relative_path"]
        require(path.is_file() and
                entry["sha256"] == expected and
                entry["bytes"] == path.stat().st_size and
                file_sha(path) == expected,
                "New P2-01F candidate binary SHA differs from immutable archived output")
        binaries[label] = {"sha256": expected, "bytes": entry["bytes"],
                           "filename": path.name}
    toolchain = report.get("logged_compiler_and_linker_executable_byte_hashes", {})
    tools = toolchain.get("observed_compile_tool_paths", {})
    require(len(tools) == (5 if system == "Linux" else 2) and
            all(len(x.get("sha256","")) == 64 and x.get("bytes", 0) > 0
                for x in tools.values()) and
            len(report.get("actual_recorded_compile_and_link_commands", {})) == 2,
            "Missing actual compiler driver/linker executable byte audit")
    return {"platform": system, "python_patch": report["candidate_platform"]["python"],
            "runner_image_observed_not_immutable": report["candidate_platform"]["runner_image"],
            "runner_image_version_observed_not_immutable":
                report["candidate_platform"]["runner_image_version"],
            "candidate_installed_versions": packages,
            "new_native_byte_identity_verified": binaries,
            "invoked_compiler_linker_executable_files_hashed": len(tools),
            "OS_and_SDK_complete_binary_closure_proved": False,
            "original_P2_01C_build_wheel_bytes_proved": False}


def compute_report(policy, p1, arrays, pilots, natives):
    g = validate_policy(policy, p1)
    schedule = case_order(p1)
    max_pilot_peak = max(
        value["peak_process_bytes"]
        for report in pilots.values()
        for value in report["backend_synthetic_measurements"].values()
    )
    require(g["max_one_fit_peak_process_resident_bytes"] >=
            4*max_pilot_peak,
            "Proposed RSS cap lacks fourfold headroom even versus synthetic pilot")
    require(g["max_one_fit_snapshot_array_bytes"] >
            arrays["max_hypothetical_array_payload_bytes"] and
            72*(g["max_one_fit_text_log_bytes"] +
                g["max_one_fit_aggregate_json_bytes"]) <=
            g["max_all_72_fits_persisted_evidence_bytes"],
            "Proposed array/snapshot or 72-case archival reservation inadequate")
    require(natives["Linux"]["candidate_installed_versions"] ==
            natives["Windows"]["candidate_installed_versions"],
            "New native candidate dependency versions differ by OS")
    return {
        "schema": "pypcc-P2-07C-proposed-72-case-resource-and-export-admission/v1",
        "status": "CANDIDATE_GUARDRAILS_CONSISTENT_BUT_NO_ACTUAL_50K_RESOURCE_GO",
        "P1_scientific_sha256": P1_SHA,
        "source_evidence_actions_runs": ARTIFACT_RUNS,
        "candidate_policy_sha256": sha256(canonical(policy)),
        "case_schedule": {
            "count": len(schedule),
            "canonical_JSON_SHA256": sha256(canonical(schedule)),
            "preview_first_case_id": schedule[0],
            "preview_last_case_id": schedule[-1],
            "all_cases_not_executed": True,
            "label_masks_not_loaded_or_generated": True,
            "no_outcome_dependent_reordering": True,
        },
        "approved_immutable_P1_science_only": True,
        "P2_07A_source_array_accounting": arrays,
        "P2_07B_original_SYNTHETIC_measured_CPU_and_whole_process_RAM": pilots,
        "P2_01F_original_source_new_native_candidate": natives,
        "candidate_guardrails_not_approved": g,
        "candidate_ram_headroom_multiplier_vs_max_SYNTHETIC_only":
            g["max_one_fit_peak_process_resident_bytes"] / max_pilot_peak,
        "candidate_single_fit_wall_cap_seconds": g["max_one_fit_wall_seconds"],
        "candidate_72_fit_nominal_cap_seconds_excluding_overhead":
            72*g["max_one_fit_wall_seconds"],
        "candidate_full_study_wall_cap_seconds": g["max_entire_study_wall_seconds"],
        "candidate_72_aggregate_plus_log_cap_bytes":
            72*(g["max_one_fit_aggregate_json_bytes"] +
                g["max_one_fit_text_log_bytes"]),
        "approved_execution_host_selected": False,
        "candidate_per_fit_time_measured_on_real_50k_UCI_data": False,
        "candidate_real_PCC_process_RSS_peak_or_disk_measured": False,
        "native_compiler_SDK_runtime_OS_complete_byte_lock_proved": False,
        "final_label_handling_and_export_policy_reviewed_by_maintainer": False,
        "actual_P2_07_resource_gate_pass": False,
        "P2_08_technical_manifest_finalized": False,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
        "real_dataset_downloads_or_rows_loaded_in_this_stage": 0,
        "graph_or_label_masks_constructed_in_this_stage": 0,
        "synthetic_or_real_PCC_propagations_executed_in_this_stage": 0,
        "real_PCC_fits_executed": 0,
        "next_required_human_review": [
            "Choose a durable execution host and its reproducible native dependency closure",
            "Explicitly approve or revise candidate process/time/RAM/disk/export limits",
            "Establish defensible 50k per-backend resource upper bounds without unauthorized real fits",
            "Review immutable P2-08 technical preflight SHA and explicitly authorize first P3 real-data fit",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    root = args.evidence_root.resolve()
    policy, p1 = load(POLICY), read_approved_P1()
    arrays, pilots, natives = {}, {}, {}
    for platform, spec in PLATFORMS.items():
        system = spec["os"]
        budget_path = root / (
            "p2-07-metadata-only-snapshot-budget-"+platform+"-py312") / (
            "p2-07-static-snapshot-and-budget.json")
        arrays[system] = validate_arrays(load(budget_path))
        pilot_path = root / (
            "p2-07b-synthetic-three-backend-cpu-memory-"+platform+"-py312") / (
            "p2-07b-three-backend-synthetic-process-resource.json")
        pilots[system] = validate_synthetic_pilot(load(pilot_path), system)
        original_new = root / (
            "p2-01f-PROSPECTIVE-new-native-toolchain-"+platform+
            "-py312-run43488066633") / "p2-01f-new-native-candidate"
        natives[system] = validate_new_native(
            load(original_new/"p2-01f-prospective-build-ledger.json"),
            original_new, system)
    require(arrays["Linux"] == arrays["Windows"],
            "Source graph/node/snapshot array arithmetic changed between platforms")
    result = compute_report(policy, p1, arrays["Linux"], pilots, natives)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "P1_sha": P1_SHA,
        "real_fits_executed": 0,
        "candidate_cases_not_executed": result["case_schedule"]["count"],
        "synthetic_max_RSS_bytes": max(
            v["peak_process_bytes"] for x in pilots.values()
            for v in x["backend_synthetic_measurements"].values()),
        "candidate_max_RSS_bytes":
            result["candidate_guardrails_not_approved"]["max_one_fit_peak_process_resident_bytes"],
        "actual_P2_07_resource_gate_pass": False,
        "P2_execution_authorized": False,
    },sort_keys=True))


if __name__ == "__main__":
    main()
