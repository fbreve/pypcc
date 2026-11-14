"""P2-07K: source-free host/resource admission evidence and P2-08 blockers.

This is a read-only review artifact, NOT an executable scientific preflight.
It independently validates exact original P2-07J source-free archived evidence,
the original P1 identities and unapproved P2-07C numeric proposal, and produces
a deterministic, explicitly NON-authorizing resource/environment decision.

Never downloads original UCI data, generates masks/graphs/seeds, starts Docker,
imports a PCC kernel, or executes real or synthetic PCC propagation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments import p2_07j_synthetic_supervisor_studywide_stop_bridge_no_pcc as j
from experiments import p2_07c_consolidated_execution_envelope_no_fit as c

P1_SHA = c.P1_SHA
POLICY_SHA = j.i.DRAFT_POLICY_SHA
OCI_SHA = j.i.OCI_GZIP_SHA
OCI_ID = j.i.OCI_CONFIG_ID
J_RUN = 43550965075
J_ARTIFACT = 14095798755
J_REPORT_SHA = "1d1d9302d4a21d204f2c13ba6db971b4af660d7a4fe6a24083975e8b141b651f"
J_AUDIT_SHA = "0da808928330769ea234f5cc0a2a4db4e0492e29ae2300b3a34965c94a475c01"
J_PRODUCTION_REGRESSION_RUN = 43550965006

SCHEMA = "pypcc-P2-07K-source-free-resource-host-admission-decision/v1"
STATUS = "RESOURCE_AND_HOST_GATES_BLOCKED_72_ORIGINAL_SCIENTIFIC_CASES_NOT_RUN"
PENDING = (
    "FINAL_SELECTED_HOST_AND_COMPLETE_TOOLCHAIN_SOURCE_BYTE_CLOSURE_MISSING",
    "REAL_ORIGINAL_50K_GRAPH_FIT_CPU_WALL_RSS_AND_DISK_BOUNDS_UNPROVEN",
    "ORIGINAL_P2_07C_NUMERIC_LIMITS_AND_EXPORT_POLICY_NOT_MAINTAINER_APPROVED",
    "PRODUCTION_SCIENTIFIC_WORKER_MONITOR_NOT_VALIDATED_WITH_ORIGINAL_GRAPH",
    "DURABLE_RUNTIME_AND_EVIDENCE_RETENTION_NOT_INDEPENDENTLY_CONFIRMED",
    "DISTINCT_FINAL_P2_08_TECHNICAL_PREFLIGHT_SHA_AND_EXPLICIT_APPROVAL_MISSING",
)


class AdmissionBlocked(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AdmissionBlocked(message)


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(data):
    return hashlib.sha256(canonical(data)).hexdigest()


def exact_old_j(path, audit, journals_dir):
    require(path.is_file() and 0 < path.stat().st_size < 400_000,
            "Missing, oversized or empty exact original P2-07J proof")
    require(audit.is_file() and 0 < audit.stat().st_size < 350_000,
            "Missing, oversized or empty original P2-07J administrative journal")
    require(journals_dir.is_dir(), "Missing original five process-tree journals")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        require(digest(report) == J_REPORT_SHA, "Original J canonical report differs")
        require(j.validate_report(report), "Original J internal scientific or resource evidence invalid")
        require(report["studywide_administrative_journal_SHA256"] == J_AUDIT_SHA and
                hashlib.sha256(audit.read_bytes()).hexdigest() == J_AUDIT_SHA,
                "Original 504-row administrative journal SHA differs")
        require(j.verify_disk_artifacts(journals_dir, audit, report),
                "One of five old H process-tree journals or original IDs altered")
    except (ValueError, TypeError, OSError, KeyError) as exc:
        raise AdmissionBlocked("Old P2-07J complete source-free evidence failed closed") from exc
    expected_journal_names = {
        j.ledger_name(fault, index)
        for fault, index, mode in j.PLAN if mode is not None
    }
    require({p.name for p in journals_dir.iterdir()} == expected_journal_names,
            "Unexpected extra or replaced previous supervisor attempt journal")
    return report


def original_candidate():
    # Validation reopens the ORIGINAL P1 and proposal, no new scientific data.
    p1 = c.read_approved_P1()
    policy = c.load(c.POLICY)
    limits = c.validate_policy(policy, p1)
    require(digest(policy) == POLICY_SHA and
            policy["P1_scientific_sha256"] == P1_SHA and
            policy["approved_execution_host_and_resource_limits"] is False and
            policy["proposed_execution_host"]["physical_or_immutable_execution_host_selected"] is False and
            policy["proposed_execution_host"]["compiler_SDK_system_byte_closure_complete"] is False,
            "Original draft source/host/resource policy changed or was falsely approved")
    cases, i_budget = j.i.original_inputs()
    require(len(cases) == 72 and
            [x["index"] for x in cases] == list(range(72)),
            "Original 72 scientific case identities absent")
    return policy, limits, cases, i_budget


def budget_projection(limits, budget):
    # These are arithmetic caps, not measured real-graph feasibility or ETA.
    wall = 72 * limits["max_one_fit_wall_seconds"]
    cpu = 72 * limits["max_one_fit_process_cpu_seconds"]
    evidence = 72 * (limits["max_one_fit_aggregate_json_bytes"] +
                     limits["max_one_fit_text_log_bytes"])
    require(wall == 129600 and cpu == 259200 and
            limits["max_entire_study_wall_seconds"] == 172800 and
            evidence == 226492416 and
            evidence < limits["max_all_72_fits_persisted_evidence_bytes"] and
            budget["maximum_72_case_fit_wall_slots_seconds"] == wall and
            budget["maximum_72_case_CPU_slots_seconds"] == cpu,
            "Unapproved candidate budget arithmetic/source schedule drift")
    return {
        "candidate_sum_wall_slots_seconds_NOT_REAL_DURATION": wall,
        "candidate_total_cpu_slots_seconds_NOT_MEASURED": cpu,
        "candidate_study_wall_deadline_seconds_NOT_TESTED_AT_48H": 172800,
        "candidate_non_fit_overhead_seconds": 172800-wall,
        "candidate_json_and_log_max_bytes_all_cases": evidence,
        "candidate_total_persisted_evidence_cap_bytes": 
            limits["max_all_72_fits_persisted_evidence_bytes"],
        "actual_real_UCI_CPU_wall_RSS_scratch_upper_bounds_measured": False,
        "cap_arithmetic_is_not_worker_feasibility": True,
    }


def build_report(j_report):
    require(digest(j_report) == J_REPORT_SHA and j.validate_report(j_report),
            "Wrong old original complete synthetic supervisor result")
    policy, limits, cases, original_budget = original_candidate()
    require(j_report["original_72_scientific_IDs"] == cases and
            j_report["unapproved_original_candidate_budget"] == original_budget,
            "Original P1 original-case order changed")
    proposal = budget_projection(limits, original_budget)
    all_ids = [case["case_id"] for case in cases]
    report = {
        "schema": SCHEMA,
        "status": STATUS,
        "scientific_protocol_P1_SHA256": P1_SHA,
        "proposed_original_P2_07C_policy_canonical_SHA256": POLICY_SHA,
        "verified_original_OCI_payload_SHA256": OCI_SHA,
        "verified_original_OCI_config_id": OCI_ID,
        "original_P2_07J_source_free_run": J_RUN,
        "original_P2_07J_artifact": J_ARTIFACT,
        "original_P2_07J_canonical_report_SHA256": J_REPORT_SHA,
        "original_P2_07J_504row_journal_SHA256": J_AUDIT_SHA,
        "independent_original_PCC_regression_source_run": J_PRODUCTION_REGRESSION_RUN,
        "original_P1_scientific_case_IDs": all_ids,
        "candidate_host": policy["proposed_execution_host"],
        "candidate_guardrails_UNAPPROVED": limits,
        "candidate_resource_budget_arithmetic_ONLY": proposal,
        "verified_actual_source_free_synthetic_supervised_trees": 5,
        "verified_injected_administrative_prelaunch_stops": 2,
        "verified_original_P2_07J_administrative_ID_rows": 504,
        "verified_real_PCC_fit_count": 0,
        "verified_synthetic_PCC_propagation_count": 0,
        "observed_OCI_binary_import_only_and_artificial_process_control": True,
        "complete_source_reproducible_native_and_OS_build_closure_verified": False,
        "real_original_P1_50k_graph_resource_bound_verified": False,
        "candidate_resource_policy_approved_by_maintainer": False,
        "source_free_supervisor_is_an_actual_scientific_worker_runner": False,
        "durable_OCI_or_full_toolchain_offline_backup_verified": False,
        "final_P2_08_manifest_created": False,
        "final_P2_08_reviewed_exact_execution_SHA256": None,
        "missing_execution_requirements": list(PENDING),
        "missing_execution_requirements_count": len(PENDING),
        "P2_01_final_execution_toolchain_lock_pass": False,
        "P2_07_full_resources_gate_pass": False,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
        "real_PCC_fits_executed": 0,
    }
    validate_report(report)
    return report


def validate_report(report):
    require(type(report) is dict and report.get("schema") == SCHEMA and
            report.get("status") == STATUS and
            report.get("scientific_protocol_P1_SHA256") == P1_SHA and
            report.get("proposed_original_P2_07C_policy_canonical_SHA256") == POLICY_SHA and
            report.get("verified_original_OCI_payload_SHA256") == OCI_SHA and
            report.get("verified_original_OCI_config_id") == OCI_ID and
            report.get("original_P2_07J_source_free_run") == J_RUN and
            report.get("original_P2_07J_artifact") == J_ARTIFACT and
            report.get("original_P2_07J_canonical_report_SHA256") == J_REPORT_SHA and
            report.get("original_P2_07J_504row_journal_SHA256") == J_AUDIT_SHA and
            report.get("independent_original_PCC_regression_source_run") == J_PRODUCTION_REGRESSION_RUN,
            "Source hashes, runtime identity or independent regression provenance drift")
    policy, limits, cases, budget = original_candidate()
    require(report.get("original_P1_scientific_case_IDs") ==
            [x["case_id"] for x in cases] and
            report.get("candidate_host") == policy["proposed_execution_host"] and
            report.get("candidate_guardrails_UNAPPROVED") == limits and
            report.get("candidate_resource_budget_arithmetic_ONLY") ==
            budget_projection(limits, budget),
            "Original case identity or draft resource host/caps changed")
    for key, expected in (
        ("verified_actual_source_free_synthetic_supervised_trees", 5),
        ("verified_injected_administrative_prelaunch_stops", 2),
        ("verified_original_P2_07J_administrative_ID_rows", 504),
        ("verified_real_PCC_fit_count", 0),
        ("verified_synthetic_PCC_propagation_count", 0),
        ("observed_OCI_binary_import_only_and_artificial_process_control", True),
        ("complete_source_reproducible_native_and_OS_build_closure_verified", False),
        ("real_original_P1_50k_graph_resource_bound_verified", False),
        ("candidate_resource_policy_approved_by_maintainer", False),
        ("source_free_supervisor_is_an_actual_scientific_worker_runner", False),
        ("durable_OCI_or_full_toolchain_offline_backup_verified", False),
        ("final_P2_08_manifest_created", False),
        ("final_P2_08_reviewed_exact_execution_SHA256", None),
        ("missing_execution_requirements", list(PENDING)),
        ("missing_execution_requirements_count", len(PENDING)),
        ("P2_01_final_execution_toolchain_lock_pass", False),
        ("P2_07_full_resources_gate_pass", False),
        ("P2_execution_preflight_sha256", None),
        ("P2_execution_authorized", False),
        ("real_PCC_fits_executed", 0),
    ):
        require(report.get(key) == expected,
                "False scientific approval, hidden fit or missing blocker: " + key)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-p2-07j-report", type=Path, required=True)
    parser.add_argument("--original-p2-07j-audit-jsonl", type=Path, required=True)
    parser.add_argument("--original-p2-07j-H-journals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Never overwrite a prior no-GO review report")
    original = exact_old_j(args.original_p2_07j_report,
                           args.original_p2_07j_audit_jsonl,
                           args.original_p2_07j_H_journals)
    report = build_report(original)
    with args.output.open("xb") as file:
        file.write(json.dumps(report, indent=2, sort_keys=True).encode("utf-8")+b"\n")
        file.flush()
        os.fsync(file.fileno())
    require(json.loads(args.output.read_text(encoding="utf-8")) == report,
            "Failed to save full canonical source-free review")
    print(json.dumps({
        "status": STATUS,
        "source_free_synthetic_supervisor_trees": 5,
        "candidate_host_resource_gates_approved": False,
        "missing_execution_requirements": len(PENDING),
        "review_only_canonical_report_sha256_NOT_AUTHORIZATION": digest(report),
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
        "real_PCC_fits_executed": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
