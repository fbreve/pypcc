"""P2-07I: study-wide fail-closed ADMINISTRATIVE stop policy, **NO PCC**.

All 72 original frozen P1 case *identities*, all six P2-07H artificial
supervisor outcomes, and the original unapproved P2-07C resource proposal
are checked without loading any UCI row, real graph/mask, RNG seed, fit
kernel, Docker image, compiled dependency, or real/synthetic PCC model.

Every simulated fault *denies* processing the failing ID and all remaining
case IDs. The 72 rows in each journal are only **metadata about denied**
or schema-checked case identities, never 72 scientific process launches.
An exact hash chain + fsync'd JSONL audit is NOT a cryptographic signature.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments import p2_07f_sealed_72case_no_fit_administrative_dryrun as f
from experiments import p2_07h_integrated_OCI_resource_tree_supervisor_no_pcc as h

P1_SHA=f.P1_SHA
DRAFT_POLICY_SHA=f.ORIGINAL_POLICY_SHA
OCI_GZIP_SHA=h.IMAGE_SHA
OCI_CONFIG_ID=h.IMAGE_ID

# Exact successful original H *canonical JSON* receipt, NOT the GitHub ZIP SHA.
H_REPORT_SHA="0f96c4c91afbd0840f50adc430f42017207b0da71e3a6126c012d50d804388db"
H_JOURNAL_TAIL="964a40c45f124a1aba00bfec42603bca0136a426e1dcec282fb9ff2f3fafaf48"
H_RUN_ID=43547341278
H_ARTIFACT_ID=14094039842

# Six previously tested synthetic categories, plus proposed study-wide deadline.
# These are deliberate hypothetical *denials*, not actual resource violations
# produced by any original scientific worker or dataset.
FAULTS=(
    "WALL_DEADLINE",
    "CPU_TIME_BUDGET",
    "CGROUP_MEMORY_OOM",
    "TMPFS_SCRATCH_ENOSPC",
    "WORKER_ROOT_FAILED",
    "DESCENDANT_CLEANUP_UNVERIFIED",
    "STUDY_WALL_DEADLINE",
)
LOCATIONS=(0,35,71)
EVENTS=("ADMIN_ID_CHECKED_ONLY",
        "ADMIN_FAKE_STOP_NO_FIT",
        "ADMIN_CASE_UNREACHED_AFTER_FAKE_STOP")
GENESIS="0"*64
TOTAL_CASES=72
TOTAL_CAMPAIGNS=1+len(FAULTS)*len(LOCATIONS)
MAX_AUDIT_BYTES=1024**2


class StudyStopBlocked(ValueError):
    pass


def require(ok,message):
    if not ok:
        raise StudyStopBlocked(message)


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("ascii")


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def verify_old_h_receipt(receipt):
    try:
        h.validate_report(receipt)
    except (h.UnifiedBlocked,KeyError,ValueError,TypeError) as exc:
        raise StudyStopBlocked("Original six-case integrated OCI synthetic proof invalid") from exc
    require(digest(receipt)==H_REPORT_SHA and
            receipt.get("journal_final_sha256")==H_JOURNAL_TAIL and
            receipt.get("bounded_original_PID_identities_confirmed_gone")==18 and
            receipt.get("exact_OCI_binary_gzip_sha256")==OCI_GZIP_SHA and
            receipt.get("exact_OCI_config_id")==OCI_CONFIG_ID and
            receipt.get("P1_scientific_sha256")==P1_SHA and
            receipt.get("P2_execution_preflight_sha256") is None and
            receipt.get("P2_execution_authorized") is False and
            receipt.get("real_PCC_fits_executed")==0,
            "Wrong original successful P2-07H artifact or scientific GO claim")
    return {
        "original_source_run_id":H_RUN_ID,
        "original_source_artifact_id":H_ARTIFACT_ID,
        "verified_exact_P2_07H_report_SHA256":H_REPORT_SHA,
        "verified_original_P2_07H_30record_journal_tail":H_JOURNAL_TAIL,
        "original_18_synthetic_host_pid_starttime_identities_gone":True,
        "source_free_evidence_cannot_authorize_PCC":True,
    }


def original_inputs():
    p1,policy,limits,_=f.frozen_inputs()
    cases=f.schedule(p1)
    original_budget=f.exact_candidate_total_budget(limits)
    require(len(cases)==TOTAL_CASES and
            [x["index"] for x in cases]==list(range(72)) and
            original_budget["maximum_72_case_fit_wall_slots_seconds"]==129600 and
            original_budget["maximum_72_case_CPU_slots_seconds"]==259200 and
            original_budget["candidate_study_wall_seconds_limit"]==172800 and
            policy["approved_execution_host_and_resource_limits"] is False and
            policy["P2_execution_authorized"] is False,
            "Original candidate study-wide envelope/protocol changed")
    return cases,original_budget


def build_entry(case,scenario,position,action,previous):
    require(action in EVENTS and
            isinstance(scenario,str) and 1<=len(scenario)<=70 and
            len(previous)==64 and
            all(c in "0123456789abcdef" for c in previous) and
            type(position) is int and position==case["index"],
            "Unexpected source-free case event, scenario or chain")
    row={
        "scenario":scenario,
        "original_case_index":position,
        "original_case_ID":case["case_id"],
        "administrative_event":action,
        "original_scientific_fit_state":"NOT_RUN_NO_APPROVAL",
        "PCC_worker_invocations":0,
        "real_UCI_feature_rows_loaded":0,
        "previous_SHA256":previous,
    }
    row["record_SHA256"]=digest(row)
    return row


def scenario_spec():
    entries=[{"scenario":"NOMINAL_METADATA_SCHEMA_INSPECTION",
              "fault_reason":None,"failure_index":None}]
    for fault in FAULTS:
        for index in LOCATIONS:
            entries.append({
                "scenario":f"FAKE_{fault}_AT_INDEX_{index:02d}",
                "fault_reason":fault,
                "failure_index":index,
            })
    require(len(entries)==TOTAL_CAMPAIGNS and
            len({x["scenario"] for x in entries})==len(entries),
            "Missing or repeated study-wide simulated stop scenario")
    return entries


def campaign(cases,spec):
    require(type(cases) is list and len(cases)==TOTAL_CASES and
            spec in scenario_spec(),
            "Not an original 72-case isolated fake-stop campaign")
    fault=spec["failure_index"]
    rows=[]
    previous=GENESIS
    for idx,case in enumerate(cases):
        action=("ADMIN_ID_CHECKED_ONLY"
                if fault is None or idx<fault
                else "ADMIN_FAKE_STOP_NO_FIT" if idx==fault
                else "ADMIN_CASE_UNREACHED_AFTER_FAKE_STOP")
        row=build_entry(case,spec["scenario"],idx,action,previous)
        rows.append(row)
        previous=row["record_SHA256"]
    outcome={
        **spec,
        "administrative_records":rows,
        "original_72_case_IDs_present_as_METADATA_only":True,
        "before_fault_admin_ID_validations":72 if fault is None else fault,
        "failing_ID_has_never_run_any_PCC":True,
        "remaining_original_IDs_unreached":0 if fault is None else 71-fault,
        "one_fake_stop_then_no_further_validation":fault is not None,
        "no_retries_or_outcome_dependent_reordering":True,
        "scientific_worker_runs_in_campaign":0,
        "scientific_completed_case_count":0,
        "final_journal_SHA256":previous,
    }
    validate_campaign(cases,outcome)
    return outcome


def validate_campaign(cases,obj):
    require(type(obj) is dict and
            obj.get("scenario") in [s["scenario"] for s in scenario_spec()],
            "Unknown synthetic administrative scenario")
    spec=next(x for x in scenario_spec()
              if x["scenario"]==obj["scenario"])
    fault=spec["failure_index"]
    require(obj.get("failure_index")==fault and
            obj.get("fault_reason")==spec["fault_reason"] and
            obj.get("original_72_case_IDs_present_as_METADATA_only") is True and
            obj.get("before_fault_admin_ID_validations")==
            (72 if fault is None else fault) and
            obj.get("failing_ID_has_never_run_any_PCC") is True and
            obj.get("remaining_original_IDs_unreached")==
            (0 if fault is None else 71-fault) and
            obj.get("one_fake_stop_then_no_further_validation") is
            (fault is not None) and
            obj.get("no_retries_or_outcome_dependent_reordering") is True and
            obj.get("scientific_worker_runs_in_campaign")==0 and
            obj.get("scientific_completed_case_count")==0,
            "Campaign claims real worker execution, continuation or retries")
    logs=obj.get("administrative_records")
    require(type(logs) is list and len(logs)==len(cases)==TOTAL_CASES,
            "Truncated or duplicate records in one original 72-ID campaign")
    prev=GENESIS
    for idx,(case,actual) in enumerate(zip(cases,logs)):
        expected_event=("ADMIN_ID_CHECKED_ONLY"
                        if fault is None or idx<fault
                        else "ADMIN_FAKE_STOP_NO_FIT" if idx==fault
                        else "ADMIN_CASE_UNREACHED_AFTER_FAKE_STOP")
        expected=build_entry(case,spec["scenario"],idx,expected_event,prev)
        require(actual==expected,
                "Tampered SHA, changed case, false finish, after-failure check or record order")
        prev=expected["record_SHA256"]
    require(obj.get("final_journal_SHA256")==prev,
            "Journal tail does not match exact original case order")
    return prev


def pinned_h_proof():
    return {
        "original_source_run_id":H_RUN_ID,
        "original_source_artifact_id":H_ARTIFACT_ID,
        "verified_exact_P2_07H_report_SHA256":H_REPORT_SHA,
        "verified_original_P2_07H_30record_journal_tail":H_JOURNAL_TAIL,
        "original_18_synthetic_host_pid_starttime_identities_gone":True,
        "source_free_evidence_cannot_authorize_PCC":True,
    }


def canonical_audit_bytes(campaigns):
    content=b"".join(
        canonical(row)+b"\n"
        for campaign_report in campaigns
        for row in campaign_report["administrative_records"]
    )
    require(len(content)<MAX_AUDIT_BYTES,
            "Total 22 source-free administrative ledgers exceed 1MiB")
    return content


def build_report():
    cases,budget=original_inputs()
    # This report contains administrative case IDs ONLY. The old H report is
    # independently validated by CLI BEFORE constructing this report.
    campaigns=[campaign(cases,spec) for spec in scenario_spec()]
    report={
        "schema":"pypcc-P2-07I-studywide-failclosed-original-72case-administration-no-fit/v1",
        "status":"TWENTY_TWO_ORIGINAL_CASE_METADATA_ONLY_CAMPAIGNS_COMPLETE_PCC_FORBIDDEN",
        "P1_scientific_sha256":P1_SHA,
        "original_draft_P2_07C_resource_policy_SHA256":DRAFT_POLICY_SHA,
        "exact_locked_OCI_gzip_SHA256":OCI_GZIP_SHA,
        "exact_locked_OCI_config_id":OCI_CONFIG_ID,
        "original_P2_07H_source_free_report_canonical_SHA256":H_REPORT_SHA,
        "original_P2_07H_source_run_id":H_RUN_ID,
        "original_P2_07H_source_artifact_id":H_ARTIFACT_ID,
        "original_P2_07H_verified_source_free_proof":pinned_h_proof(),
        "source_free_audit_JSONL_SHA256":
            hashlib.sha256(canonical_audit_bytes(campaigns)).hexdigest(),
        "source_free_audit_JSONL_rows":TOTAL_CAMPAIGNS*TOTAL_CASES,
        "original_72_original_case_id_inventory":cases,
        "unapproved_candidate_study_wide_budget_arithmetic":budget,
        "administrative_campaigns":campaigns,
        "simulated_fault_classes":list(FAULTS),
        "simulated_fault_locations_zero_based":list(LOCATIONS),
        "metadata_only_campaign_count":TOTAL_CAMPAIGNS,
        "metadata_only_case_record_count":TOTAL_CAMPAIGNS*TOTAL_CASES,
        "original_72_case_states":"ALL_NOT_RUN_NO_APPROVAL",
        "administrative_failure_never_allows_next_case":True,
        "candidate_budget_is_NOT_actual_real_fit_resource_bound":True,
        "source_only_P2_07H_proves_no_actual_scientific_runner":True,
        "full_host_runtime_native_and_compiler_lock_approved":False,
        "real_graph_50k_resource_upper_bounds_verified":False,
        "this_module_has_no_real_scientific_worker_launcher":True,
        "real_UCI_dataset_rows_loaded":0,
        "graph_or_label_masks_generated":0,
        "synthetic_PCC_propagations_executed":0,
        "real_PCC_fits_executed":0,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
    }
    validate_report(report)
    return report


def validate_report(obj):
    require(obj.get("schema")==
            "pypcc-P2-07I-studywide-failclosed-original-72case-administration-no-fit/v1" and
            obj.get("status")==
            "TWENTY_TWO_ORIGINAL_CASE_METADATA_ONLY_CAMPAIGNS_COMPLETE_PCC_FORBIDDEN" and
            obj.get("P1_scientific_sha256")==P1_SHA and
            obj.get("original_draft_P2_07C_resource_policy_SHA256")==DRAFT_POLICY_SHA and
            obj.get("exact_locked_OCI_gzip_SHA256")==OCI_GZIP_SHA and
            obj.get("exact_locked_OCI_config_id")==OCI_CONFIG_ID and
            obj.get("original_P2_07H_source_free_report_canonical_SHA256")==H_REPORT_SHA and
            obj.get("original_P2_07H_source_run_id")==H_RUN_ID and
            obj.get("original_P2_07H_source_artifact_id")==H_ARTIFACT_ID and
            obj.get("original_P2_07H_verified_source_free_proof")==pinned_h_proof(),
            "P1/P2-07C/P2-07H frozen provenance drift")
    cases,budget=original_inputs()
    require(obj.get("original_72_original_case_id_inventory")==cases and
            obj.get("unapproved_candidate_study_wide_budget_arithmetic")==budget and
            obj.get("simulated_fault_classes")==list(FAULTS) and
            obj.get("simulated_fault_locations_zero_based")==list(LOCATIONS),
            "72 original scientific case IDs or draft budget altered")
    campaigns=obj.get("administrative_campaigns",[])
    specs=scenario_spec()
    require(type(campaigns) is list and len(campaigns)==len(specs),
            "Missing, extra or retried administrative campaign")
    for expected,c in zip(specs,campaigns):
        require(c.get("scenario")==expected["scenario"],
                "Outcome-dependent reordering or replaced fake study failure")
        validate_campaign(cases,c)
    require(obj.get("source_free_audit_JSONL_SHA256")==
            hashlib.sha256(canonical_audit_bytes(campaigns)).hexdigest() and
            obj.get("source_free_audit_JSONL_rows")==TOTAL_CAMPAIGNS*TOTAL_CASES,
            "Missing or altered exact complete source-free on-disk evidence digest")
    for key,expected in (
        ("metadata_only_campaign_count",TOTAL_CAMPAIGNS),
        ("metadata_only_case_record_count",TOTAL_CAMPAIGNS*TOTAL_CASES),
        ("original_72_case_states","ALL_NOT_RUN_NO_APPROVAL"),
        ("administrative_failure_never_allows_next_case",True),
        ("candidate_budget_is_NOT_actual_real_fit_resource_bound",True),
        ("source_only_P2_07H_proves_no_actual_scientific_runner",True),
        ("full_host_runtime_native_and_compiler_lock_approved",False),
        ("real_graph_50k_resource_upper_bounds_verified",False),
        ("this_module_has_no_real_scientific_worker_launcher",True),
        ("real_UCI_dataset_rows_loaded",0),
        ("graph_or_label_masks_generated",0),
        ("synthetic_PCC_propagations_executed",0),
        ("real_PCC_fits_executed",0),
        ("P2_01_final_execution_toolchain_lock_pass",False),
        ("P2_07_full_resources_gate_pass",False),
        ("P2_execution_preflight_sha256",None),
        ("P2_execution_authorized",False),
    ):
        require(obj.get(key)==expected,"False scientific completion/GO or invalid safety claim: "+key)
    return True


def append_audit_journal(path,report):
    require(not path.exists(),"Never overwrite an earlier administrative audit")
    count=0
    with path.open("xb") as dst:
        for campaign_report in report["administrative_campaigns"]:
            for record in campaign_report["administrative_records"]:
                line=canonical(record)+b"\n"
                require(len(line)<512,"Metadata-only bounded row limit exceeded")
                dst.write(line)
                dst.flush()
                os.fsync(dst.fileno())
                count+=1
    require(count==TOTAL_CAMPAIGNS*TOTAL_CASES and
            path.stat().st_size<MAX_AUDIT_BYTES,
            "Audit ledger exceeds source-free metadata budget")
    return count


def verify_audit_journal(path,report):
    require(path.is_file() and path.stat().st_size<MAX_AUDIT_BYTES,
            "Missing, oversized or overwritten original case audit")
    actual_lines=path.read_bytes().splitlines()
    expected_records=[
        row for scenario in report["administrative_campaigns"]
        for row in scenario["administrative_records"]
    ]
    require(len(actual_lines)==len(expected_records)==TOTAL_CAMPAIGNS*TOTAL_CASES,
            "Missing/extra complete original case metadata row")
    for original,actual in zip(expected_records,actual_lines):
        require(canonical(original)==actual,
                "Partial row or altered/jumbled append-only case metadata")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-p2-07h-source-free-report",required=True,type=Path)
    p.add_argument("--source-free-audit-jsonl",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    require(not args.output.exists() and not args.source_free_audit_jsonl.exists(),
            "Original attempt evidence must never be overwritten or retried")
    receipt=json.loads(args.original_p2_07h_source_free_report.read_text(encoding="utf-8"))
    proof=verify_old_h_receipt(receipt)
    report=build_report()
    require(proof==report["original_P2_07H_verified_source_free_proof"],
            "Exact archive input does not match frozen original H evidence fields")
    count=append_audit_journal(args.source_free_audit_jsonl,report)
    audit_sha=verify_audit_journal(args.source_free_audit_jsonl,report)
    require(audit_sha==report["source_free_audit_JSONL_SHA256"] and
            count==report["source_free_audit_JSONL_rows"],
            "Persisted source-free full study journal failed precise integrity proof")
    validate_report(report)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({
        "status":report["status"],
        "distinct_administrative_faults":len(FAULTS),
        "fault_injection_positions":list(LOCATIONS),
        "source_free_campaigns":TOTAL_CAMPAIGNS,
        "source_free_case_ID_log_rows":count,
        "audit_SHA256":audit_sha,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))


if __name__=="__main__":
    main()
