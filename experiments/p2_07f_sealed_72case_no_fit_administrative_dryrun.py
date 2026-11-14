"""P2-07F fail-closed ADMINISTRATIVE 72-case NO-FIT runbook and journal.

The P1 registry/masks/backend ordering and P2-07C resource envelope are
checked against original committed hashes. P2-07E exact OCI kernel quota
evidence is read only as *source-free metadata*. This module DOES NOT import
PCC, load UCI, run a scientific/synthetic fit, retrieve real seeds or masks,
call Docker, compile anything or admit actual scientific execution.

"ADMIN_CHECK" means checking the schema and candidate caps of a CASE ID,
not an executed, successful or even authorized PCC run. The complete set
of 72 cases ALWAYS remains NOT_RUN in every output.
"""
import argparse
import hashlib
import json
from pathlib import Path

from experiments import p2_07c_consolidated_execution_envelope_no_fit as prior
from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as linux
from experiments import p2_01i_verify_exact_exported_runtime_image as image

P1_SHA=prior.P1_SHA
ORIGINAL_POLICY_SHA="b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04"
ORIGINAL_IMAGE_ARCHIVE_SHA=image.EXPECTED_IMAGE_SHA
ORIGINAL_IMAGE_ID=image.EXPECTED_IMAGE_ID
P2_07E_RUN=43532796956
P2_07E_ARTIFACT=14088600777
BACKENDS=("numpy_sequential","numba_sequential_nopython",
          "cython_sequential_cached_or_legacy_as_defined_by_early_stop")
SOURCES=(212,850,292)
INJECTED_FAILURE_INDEX=(0,35,71)
GENESIS="0"*64


class DryRunBlocked(ValueError):
    pass


def require(condition,detail):
    if not condition:
        raise DryRunBlocked(detail)


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False,
                      separators=(",",":")).encode("ascii")


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def frozen_inputs():
    p1=prior.read_approved_P1()
    policy=prior.load(prior.POLICY)
    limits=prior.validate_policy(policy,p1)
    require(prior.sha256(prior.canonical(policy))==ORIGINAL_POLICY_SHA and
            p1["fit"]["backend_order"]==list(BACKENDS) and
            p1["fit"]["total_proposed_fits"]==72 and
            p1["fit"]["label_mask_replicates_per_dataset"]==8 and
            p1["fit"]["max_iter"]==50000 and
            p1["fit"]["milestones"]==[0,1,10,100,1000,5000,25000,50000] and
            p1["fit"]["early_stop"] is False and
            tuple(x["uci_id"] for x in p1["selected_datasets"])==SOURCES and
            p1["approvals_and_execution"]["P2_execution_approved"] is False,
            "Frozen P1 or original draft operational resource contract drift")
    lock=image.lock_checked(json.loads(image.LOCK_PATH.read_text(encoding="utf-8")))
    require(lock["docker_save_gzip_archive"]["sha256"]==ORIGINAL_IMAGE_ARCHIVE_SHA and
            lock["docker_save_gzip_archive"]["original_image_config_id"]==ORIGINAL_IMAGE_ID and
            lock["P2_execution_authorized"] is False,
            "Actual original OCI archive identification drift")
    return p1,policy,limits,lock


def validate_proven_cgroup_receipt(receipt):
    try:
        linux.validate_report(receipt)
    except linux.ResourceProbeBlocked as exc:
        raise DryRunBlocked("Original P2-07E source-free kernel enforcement report drift") from exc
    require(receipt["original_locked_OCI_gzip_sha256"]==ORIGINAL_IMAGE_ARCHIVE_SHA and
            receipt["verified_docker_image_config_SHA256"]==ORIGINAL_IMAGE_ID and
            receipt["draft_resource_policy_SHA256"]==ORIGINAL_POLICY_SHA and
            receipt["real_PCC_fits_executed"]==0 and
            receipt["P2_execution_preflight_sha256"] is None and
            receipt["P2_execution_authorized"] is False,
            "Previous cgroup tests may not authorize scientific fits")
    return {
        "source_run":P2_07E_RUN,
        "source_artifact":P2_07E_ARTIFACT,
        "source_report_canonical_sha256":sha(receipt),
        "five_actual_source_free_quota_and_kill_probes_passed":True,
        "no_actual_scientific_PCC_run_or_real_resource_bound_proven":True,
    }


def schedule(p1):
    original=prior.case_order(p1)
    result=[]
    for index,x in enumerate(original):
        expected_uci=SOURCES[index//24]
        expected_replicate=(index%24)//3
        expected_backend=BACKENDS[index%3]
        require((x["uci_id"],x["replicate"],x["backend"],x["status"])==
                (expected_uci,expected_replicate,expected_backend,
                 "NOT_RUN_NO_APPROVAL"),
                "Original P1 72 IDs, source order or backend labels drift")
        result.append({
            "index":index,
            "case_id":f"UCI{expected_uci}-M{expected_replicate:02d}-B{index%3}",
            "uci_id":expected_uci,
            "mask_replicate":expected_replicate,
            "backend":expected_backend,
            "case_state":"NOT_RUN_NO_APPROVAL",
        })
    require(len(result)==72 and
            len({row["case_id"] for row in result})==72,
            "Original case schedule must cover exactly 72 distinct IDs")
    return result


def envelope(proposed):
    # Invariants from #272: proposed per-fit quantities are not measurements.
    keys=("max_one_fit_peak_process_resident_bytes",
          "max_one_fit_wall_seconds",
          "max_one_fit_process_cpu_seconds",
          "max_one_fit_snapshot_array_bytes",
          "max_one_fit_aggregate_json_bytes",
          "max_one_fit_text_log_bytes",
          "max_one_fit_scratch_bytes")
    result={key:proposed[key] for key in keys}
    result.update({
        "candidate_not_approved":True,
        "real_data_resource_upper_bound_proven":False,
        "zero_parallel_fits_simulated":True,
        "real_PCC_fits_permitted":0,
        "real_PCC_fits_run":0,
    })
    require(result["max_one_fit_peak_process_resident_bytes"]==2*1024**3 and
            result["max_one_fit_wall_seconds"]==1800 and
            result["max_one_fit_process_cpu_seconds"]==3600 and
            result["max_one_fit_scratch_bytes"]==1024**3,
            "Cannot silently modify candidate per-fit budgets")
    return result


def exact_candidate_total_budget(limits, count=72):
    require(type(count) is int and count==72,"Only 72 scientific case IDs")
    # These are MAXIMUM theoretical slots if every case uses its full
    # proposed cap, not real resource usage or proof of feasibility.
    estimate={
        "planned_cases":count,
        "candidate_single_fit_wall_seconds":limits["max_one_fit_wall_seconds"],
        "candidate_single_fit_cpu_seconds":limits["max_one_fit_process_cpu_seconds"],
        "candidate_study_wall_seconds_limit":limits["max_entire_study_wall_seconds"],
        "maximum_72_case_fit_wall_slots_seconds":
            count*limits["max_one_fit_wall_seconds"],
        "maximum_72_case_CPU_slots_seconds":
            count*limits["max_one_fit_process_cpu_seconds"],
        "maximum_72_case_aggregate_JSON_limit_bytes":
            count*limits["max_one_fit_aggregate_json_bytes"],
        "maximum_72_case_text_logs_limit_bytes":
            count*limits["max_one_fit_text_log_bytes"],
        "maximum_72_case_snapshots_per_fit_bytes":
            limits["max_one_fit_snapshot_array_bytes"],
        "maximum_72_case_scratch_per_fit_bytes":
            limits["max_one_fit_scratch_bytes"],
        "total_exported_evidence_cap_bytes":
            limits["max_all_72_fits_persisted_evidence_bytes"],
        "candidate_time_budget_is_NOT_measured_host_elapsed_time":True,
        "candidate_memory_budget_is_NOT_real_graph_peak_RSS_bound":True,
        "candidate_CPU_slots_are_NOT_CPU_benchmark_or_scheduler_prediction":True,
        "candidate_disk_bytes_are_NOT_real_export_or_scratch_measurement":True,
    }
    require(estimate["maximum_72_case_fit_wall_slots_seconds"]==129600 and
            estimate["maximum_72_case_CPU_slots_seconds"]==259200 and
            estimate["maximum_72_case_aggregate_JSON_limit_bytes"]==
            72*1024**2 and
            estimate["maximum_72_case_text_logs_limit_bytes"]==
            144*1024**2 and
            estimate["maximum_72_case_fit_wall_slots_seconds"] <=
            estimate["candidate_study_wall_seconds_limit"] and
            estimate["maximum_72_case_aggregate_JSON_limit_bytes"]+
            estimate["maximum_72_case_text_logs_limit_bytes"] <
            estimate["total_exported_evidence_cap_bytes"],
            "Proposed 72-case accounting did not respect initial draft")
    return estimate


def journal_entry(prev, index, case_id, action):
    require(action in ("ADMIN_CASE_VALIDATED_ONLY","ADMIN_CASE_BLOCKED_FAKE",
                       "ADMIN_CASE_CHECK_NOT_ATTEMPTED_AFTER_FAILURE") and
            type(index) is int and 0<=index<72 and
            isinstance(case_id,str) and len(case_id)<50,
            "Only an administrative metadata-only event can be recorded")
    entry={
        "event_index":index,
        "case_id":case_id,
        "event":action,
        "prior_journal_SHA256":prev,
        "PCC_execution_count":0,
        "scientific_case_state":"NOT_RUN_NO_APPROVAL",
    }
    entry["journal_SHA256"]=sha(entry)
    return entry


def administrative_journal(cases,*,failure_index=None):
    require(len(cases)==72 and
            (failure_index is None or failure_index in INJECTED_FAILURE_INDEX),
            "Only full no-fit scan or one fixed adversarial failure injection")
    entries=[]
    prev=GENESIS
    for i,case in enumerate(cases):
        event=("ADMIN_CASE_VALIDATED_ONLY" if failure_index is None or
               i<failure_index else
               "ADMIN_CASE_BLOCKED_FAKE" if i==failure_index else
               "ADMIN_CASE_CHECK_NOT_ATTEMPTED_AFTER_FAILURE")
        row=journal_entry(prev,i,case["case_id"],event)
        entries.append(row)
        prev=row["journal_SHA256"]
    check_journal(cases,entries,failure_index=failure_index)
    return entries


def check_journal(cases,entries,*,failure_index=None):
    """Each 72-row journal is a simple ordered integrity chain (NOT a signature).

    Once a hypothetical administrative fault is injected, no subsequent case
    can be claimed validated. Real PCC fits remain explicitly NOT_RUN.
    """
    require(type(entries) is list and len(entries)==len(cases)==72 and
            (failure_index is None or failure_index in INJECTED_FAILURE_INDEX),
            "Missing, duplicate, extra or truncated 72-case journal")
    prev=GENESIS
    for i,(case,row) in enumerate(zip(cases,entries)):
        expected=("ADMIN_CASE_VALIDATED_ONLY" if failure_index is None or
                  i<failure_index else
                  "ADMIN_CASE_BLOCKED_FAKE" if i==failure_index else
                  "ADMIN_CASE_CHECK_NOT_ATTEMPTED_AFTER_FAILURE")
        require(row.get("event_index")==i and
                row.get("case_id")==case["case_id"] and
                row.get("event")==expected and
                row.get("prior_journal_SHA256")==prev and
                row.get("PCC_execution_count")==0 and
                row.get("scientific_case_state")=="NOT_RUN_NO_APPROVAL",
                "Case reordered/duplicated/retried or marked fit-completed")
        recreated=journal_entry(prev,i,case["case_id"],expected)
        require(row==recreated,
                "Hash-chain tampering or previous journal state mismatch")
        prev=recreated["journal_SHA256"]
    return prev


def compile_report(p207e):
    p1,policy,limits,lock=frozen_inputs()
    proof=validate_proven_cgroup_receipt(p207e)
    cases=schedule(p1)
    limits_map=envelope(limits)
    totals=exact_candidate_total_budget(limits)
    nominal=administrative_journal(cases)
    fault_reports=[]
    for failure in INJECTED_FAILURE_INDEX:
        events=administrative_journal(cases,failure_index=failure)
        fault_reports.append({
            "fake_failure_at_zero_based_case_index":failure,
            "previous_admin_validations_only":failure,
            "expected_fake_admission_denials":1,
            "remaining_unchecked_case_ID_count":71-failure,
            "all_72_scientific_cases_still_NOT_RUN":True,
            "journal_tail_SHA256":check_journal(
                cases,events,failure_index=failure),
            "case_log":events,
        })
    require(len(nominal)==72 and len(fault_reports)==3 and
            totals["planned_cases"]==len(cases),
            "Missing candidate case or fixed fault scenario")
    report={
        "schema":"pypcc-P2-07F-72case-failclosed-admin-only-dry-run/v1",
        "status":"72_CASE_ID_ADMINISTRATIVE_DRILL_COMPLETE_BUT_ALL_REAL_PCC_FITS_PROHIBITED",
        "P1_scientific_sha256":P1_SHA,
        "original_P2_07C_draft_resource_policy_SHA256":ORIGINAL_POLICY_SHA,
        "exact_archived_OCI_binary_SHA256":ORIGINAL_IMAGE_ARCHIVE_SHA,
        "original_OCI_image_config_id":ORIGINAL_IMAGE_ID,
        "P2_07E_source_free_Linux_cgroup_enforcement_evidence":proof,
        "case_inventory":cases,
        "nominal_admin_case_ID_check_journal":nominal,
        "three_forced_administrative_failure_scenarios":fault_reports,
        "candidate_per_case_unapproved_envelope":limits_map,
        "candidate_total_72_case_budget_arithmetic_NOT_measured":totals,
        "administrative_case_identity_count":72,
        "administrative_synthetic_drill_case_count":72*4,
        "scientific_fit_scheduling_callable_present":False,
        "real_OCI_Docker_execution_in_this_P2_07F_stage":False,
        "real_UCI_data_loaded":0,
        "graphs_or_masks_constructed":0,
        "particle_propagations_executed":0,
        "real_PCC_fits_executed":0,
        "all_72_original_scientific_cases_state":"NOT_RUN_NO_APPROVAL",
        "human_approved_candidate_operational_limits":False,
        "final_execution_host_and_whole_runtime_native_lock_complete":False,
        "real_graph_50k_resource_upper_bound_verified":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
    }
    validate_report(report)
    return report


def validate_report(report):
    require(report.get("schema")==
            "pypcc-P2-07F-72case-failclosed-admin-only-dry-run/v1" and
            report.get("status")==
            "72_CASE_ID_ADMINISTRATIVE_DRILL_COMPLETE_BUT_ALL_REAL_PCC_FITS_PROHIBITED" and
            report.get("P1_scientific_sha256")==P1_SHA and
            report.get("original_P2_07C_draft_resource_policy_SHA256")==
            ORIGINAL_POLICY_SHA and
            report.get("exact_archived_OCI_binary_SHA256")==
            ORIGINAL_IMAGE_ARCHIVE_SHA and
            report.get("original_OCI_image_config_id")==ORIGINAL_IMAGE_ID,
            "Original frozen scientific/operational image identity changed")
    p1,policy,limits,lock=frozen_inputs()
    proof=report.get("P2_07E_source_free_Linux_cgroup_enforcement_evidence",{})
    require(proof.get("source_run")==P2_07E_RUN and
            proof.get("source_artifact")==P2_07E_ARTIFACT and
            proof.get("five_actual_source_free_quota_and_kill_probes_passed") is True and
            proof.get("no_actual_scientific_PCC_run_or_real_resource_bound_proven") is True,
            "Original actual kernel cgroup proof is missing")
    cases=schedule(p1)
    require(report.get("case_inventory")==cases and
            report.get("candidate_per_case_unapproved_envelope")==envelope(limits) and
            report.get("candidate_total_72_case_budget_arithmetic_NOT_measured")==
            exact_candidate_total_budget(limits),
            "72-case ordering or original budget arithmetic diverged")
    require(check_journal(
        cases,report.get("nominal_admin_case_ID_check_journal"))==
        report["nominal_admin_case_ID_check_journal"][-1]["journal_SHA256"],
        "Nominal administrative journal malformed")
    faults=report.get("three_forced_administrative_failure_scenarios")
    require(type(faults) is list and len(faults)==3,
            "Missing three independent administrative fault checks")
    for i,index in enumerate(INJECTED_FAILURE_INDEX):
        fault=faults[i]
        log=fault.get("case_log")
        tail=check_journal(cases,log,failure_index=index)
        require(fault.get("fake_failure_at_zero_based_case_index")==index and
                fault.get("previous_admin_validations_only")==index and
                fault.get("expected_fake_admission_denials")==1 and
                fault.get("remaining_unchecked_case_ID_count")==71-index and
                fault.get("all_72_scientific_cases_still_NOT_RUN") is True and
                fault.get("journal_tail_SHA256")==tail,
                "Injected administrative fault was retried or cases replaced")
    for key,expected in (
        ("administrative_case_identity_count",72),
        ("administrative_synthetic_drill_case_count",288),
        ("scientific_fit_scheduling_callable_present",False),
        ("real_OCI_Docker_execution_in_this_P2_07F_stage",False),
        ("real_UCI_data_loaded",0),
        ("graphs_or_masks_constructed",0),
        ("particle_propagations_executed",0),
        ("real_PCC_fits_executed",0),
        ("all_72_original_scientific_cases_state","NOT_RUN_NO_APPROVAL"),
        ("human_approved_candidate_operational_limits",False),
        ("final_execution_host_and_whole_runtime_native_lock_complete",False),
        ("real_graph_50k_resource_upper_bound_verified",False),
        ("P2_07_full_resources_gate_pass",False),
        ("P2_01_final_execution_toolchain_lock_pass",False),
        ("P2_execution_preflight_sha256",None),
        ("P2_execution_authorized",False),
    ):
        require(report.get(key)==expected,
                "Source-free P2-07F report claimed unauthorized scientific run: "+key)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous-cgroup-receipt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    require(not args.output.exists(),"Never overwrite P2-07F original safe output")
    evidence=json.loads(args.previous_cgroup_receipt.read_text(encoding="utf-8"))
    result=compile_report(evidence)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        "status":result["status"],
        "candidate_72_case_ids_checked":len(result["case_inventory"]),
        "artificial_failure_positions":list(INJECTED_FAILURE_INDEX),
        "candidate_72_case_wall_slots_s":result[
            "candidate_total_72_case_budget_arithmetic_NOT_measured"][
            "maximum_72_case_fit_wall_slots_seconds"],
        "candidate_72_case_CPU_slots_s":result[
            "candidate_total_72_case_budget_arithmetic_NOT_measured"][
            "maximum_72_case_CPU_slots_seconds"],
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))


if __name__=="__main__":
    main()
