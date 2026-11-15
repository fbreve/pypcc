"""P2-07N: independent archived P2-07A vs P2-07M resource gap, NO PCC.

Reuses exact prior source-free evidence; no data, model, graph, seed, mask, OCI
transfer, runtime installation, JIT, propagation or fit. Quantifies where the
2-GiB resource proposal remains *unproven*, avoiding the common mistake of
treating graph-only peak RSS + hypothetical array payload as a full-fit bound.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments import p2_07_metadata_only_resource_budget as a
from experiments import p2_07m_exact_oci_original_graph_preparation_no_fit as m
from experiments import p2_07c_consolidated_execution_envelope_no_fit as c

SCHEMA = "pypcc-P2-07N-archived-original-OCI-graph-plus-static-arrays-unresolved-fit-resources/v1"
STATUS = "ORIGINAL_OCI_GRAPH_AND_ARRAY_EVIDENCE_RECONCILED_FULL_50K_FIT_BUDGET_UNPROVEN"
P1_SHA = c.P1_SHA
STATIC_RUN_ID = 43478839756
STATIC_ARTIFACT_ID = 14064456453
OCI_RUN_ID = 43664754999
OCI_ARTIFACT_ID = 14142136186
OCI_EVIDENCE_SHA = "0cb2b85ec100c558837042bc0b147f13597e6182a6453bab60a540f8fc0c9b97"
CAP = 2*1024**3
MISSING = (
    "ACTUAL_P1_50K_COMPLETE_PCC_NUMPY_NUMBA_CYTHON_RSS_AND_CGROUP_PEAK_UNMEASURED",
    "ACTUAL_P1_50K_COMPLETE_PCC_CPU_AND_WALL_DURATION_UNMEASURED",
    "NUMBA_REAL_KERNEL_JIT_NATIVE_BACKEND_PEAK_AND_LAZY_LIBRARIES_NOT_CLOSED",
    "ACTUAL_FULL_FIT_SCRATCH_LOG_AND_CHECKPOINT_DISK_BUDGET_UNMEASURED",
    "COMPLETE_SELECTED_HOST_SOURCE_NATIVE_COMPILER_SDK_OS_BYTE_CLOSURE_UNPROVEN",
    "FINAL_NUMERIC_POLICY_EXPORT_APPROVAL_AND_DISTINCT_P2_08_EXECUTION_SHA_MISSING",
)

class BudgetBlocked(ValueError):
    pass

def require(ok, reason):
    if not ok:
        raise BudgetBlocked(reason)

def canonical(o):
    return json.dumps(o,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("ascii")

def digest(o):
    return hashlib.sha256(canonical(o)).hexdigest()

def read_exact(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and
            0<path.stat().st_size<=180000,
            "Missing, unsafe or oversized original archived source-free evidence")
    return json.loads(path.read_text(encoding="utf-8"))

def check_old_static(doc):
    require(type(doc) is dict and
            doc.get("schema")=="pypcc-P2-07-meta-only-array-and-snapshot-budget/v1" and
            doc.get("status")==
              "SNAPSHOT_ARRAY_BYTES_WITHIN_CAP_REAL_EXECUTION_CPU_RSS_DISK_STILL_UNVERIFIED" and
            doc.get("P1_scientific_sha256")==P1_SHA and
            doc.get("parent_P2_04_05_PR")==267 and
            doc.get("parent_P2_04_05_workflow_run")==43475652889 and
            doc.get("P2_execution_preflight_sha256") is None and
            doc.get("P2_execution_authorized") is False and
            doc.get("PCC_real_data_fits_executed")==0 and
            doc.get("PCC_particle_propagations_executed")==0,
            "Old archival P2-07A not a source-only frozen no-GO report")
    budget=doc.get("accounting")
    require(type(budget) is dict and
            budget.get("total_intended_fits")==72 and
            budget.get("total_proposed_PCC_iterations_all_fits")==3600000 and
            budget.get("total_particle_update_opportunities_all_fits")==25200000 and
            budget.get("total_eight_checkpoint_array_payloads_all_72_fits_bytes")==14079360 and
            budget.get("maximum_hypothetical_co_resident_array_payload_bytes_any_one_fit")==650012 and
            budget.get("array_payload_not_a_process_RSS_upper_bound") is True and
            budget.get("projection_not_a_CPU_time_benchmark") is True,
            "Original P2-07A formulas/72-case budget changed")
    source_gate=doc.get("safety_and_approval_gates")
    require(type(source_gate) is dict and
            source_gate.get("P2_07_snapshot_payload_within_original_64MiB_per_fit_cap") is True and
            source_gate.get("P2_07_total_CPU_wall_time_budget_verified") is False and
            source_gate.get("P2_07_native_backend_peak_RSS_budget_verified") is False and
            source_gate.get("P2_07_real_fit_artifact_disk_and_log_budget_verified") is False and
            source_gate.get("P2_07_full_resources_gate_pass") is False,
            "Original frozen static audit claimed scientific execution readiness")
    rows=budget.get("datasets")
    require(type(rows) is list and len(rows)==3 and
            [x.get("uci_id") for x in rows]==list(m.IDS),
            "Original graph identity/order changed in static evidence")
    for row,(uci,n,f,classes) in zip(rows,a.EXPECTED_DATASETS):
        width=m.DIMENSIONS[uci][2]
        exact=a.model_persistent_array_bytes(n,f,classes,width)
        require(row.get("n_original_rows")==n and
                row.get("n_numeric_features")==f and
                row.get("n_classes")==classes and
                row.get("n_original_labeled_per_fit")==3*classes and
                row.get("dense_neighbor_width")==width and
                row.get("verified_graph_storage_bytes")==8*n*(width+1) and
                row.get("retained_array_line_items_bytes")==exact and
                row.get("sum_retained_array_payload_bytes")==sum(exact.values()) and
                row.get("proposed_particle_update_opportunities_per_fit")==50000*3*classes,
                "Original Python/Cython/Numba static array accounting drift")
        groups=row.get("backend_observer_payloads")
        require(type(groups) is dict and set(groups)==set(a.BACKENDS),
                "Expected all three original observer variants")
        for be in a.BACKENDS:
            snap=a.snapshot_bytes(n,classes,be)
            entry=groups[be]
            require(entry["snapshots"]==snap and
                    entry["retained_arrays_plus_observer_hypothetical_co_resident_bytes"]==
                    sum(exact.values())+snap["observer_allocation_plus_bookkeeping_bytes"] and
                    entry["fits_proposed_for_dataset_backend"]==8,
                    "Previously archived observer payload differs from original source formulas")
    return rows

def check_old_oci(doc):
    require(type(doc) is dict and m.validate_report(doc) and
            digest(doc)==OCI_EVIDENCE_SHA and
            doc["verified_original_OCI_archive_SHA256"]==m.IMAGE_SHA and
            doc["verified_original_OCI_config_id"]==m.IMAGE_ID and
            doc["P2_01_final_execution_toolchain_lock_pass"] is False and
            doc["P2_07_full_resources_gate_pass"] is False and
            doc["P2_execution_preflight_sha256"] is None and
            doc["P2_execution_authorized"] is False and
            doc["real_PCC_fits_executed"]==0,
            "Original archived P2-07M has wrong SHA/OCI, graphs or execution authority")
    return doc["exact_OCI_three_original_graph_only_resources"]

def build(old_static,old_oci):
    static_rows=check_old_static(old_static)
    real_rows=check_old_oci(old_oci)
    policy=c.load(c.POLICY)
    p1=c.read_approved_P1()
    c.validate_policy(policy,p1)
    require(c.sha256(c.canonical(policy))==
            "b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04" and
            policy["approved_execution_host_and_resource_limits"] is False and
            policy["candidate_guardrails"]["max_one_fit_peak_process_resident_bytes"]==CAP,
            "Original unapproved P2-07C proposal changed")
    items=[]
    for old,real in zip(static_rows,real_rows):
        uci=old["uci_id"]
        require(uci==real["uci_id"] and
                old_static["actual_P2_04_05_graph_hashes"][str(uci)]==
                real["original_graph_sha256"]==m.GRAPH_HASHES[uci] and
                real["original_graph_dense_minus_one_padding"] is True,
                "Unchanged original real dense graphs did not match cross-archive")
        most=max(x["retained_arrays_plus_observer_hypothetical_co_resident_bytes"]
                 for x in old["backend_observer_payloads"].values())
        highwater=real["process_peak_resident_bytes"]
        require(most>0 and most<=650012 and 0<highwater<CAP,
                "Unexpected array payload or actual graph-only RSS beyond draft cap")
        items.append({
            "uci_id":uci,
            "original_graph_SHA256":m.GRAPH_HASHES[uci],
            "original_node_count":old["n_original_rows"],
            "original_graph_dense_width":old["dense_neighbor_width"],
            "original_pcc_particles_per_approved_mask":old["n_original_labeled_per_fit"],
            "actual_OCI_graph_only_peak_process_RSS_bytes":highwater,
            "static_retained_array_payload_bytes_NOT_RSS_BOUND":
                old["sum_retained_array_payload_bytes"],
            "static_largest_backend_snapshot_plus_arrays_payload_NOT_RSS_BOUND":most,
            "candidate_cap_minus_observed_graph_only_RSS_bytes_NOT_FIT_HEADROOM_GUARANTEE":
                CAP-highwater,
            "graph_only_observed_below_draft_2GiB_cap":True,
            "this_does_NOT_include_PCC_propagation_JIT_checkpoint_live_peak":True,
        })
    report={
        "schema":SCHEMA,
        "status":STATUS,
        "P1_scientific_sha256":P1_SHA,
        "old_original_P2_07A_static_run":STATIC_RUN_ID,
        "old_original_P2_07A_static_artifact":STATIC_ARTIFACT_ID,
        "original_static_source_only_evidence_canonical_SHA256":digest(old_static),
        "old_original_P2_07M_exact_OCI_run":OCI_RUN_ID,
        "old_original_P2_07M_exact_OCI_artifact":OCI_ARTIFACT_ID,
        "original_P2_07M_canonical_SHA256":OCI_EVIDENCE_SHA,
        "original_OCI_archive_SHA256":m.IMAGE_SHA,
        "original_OCI_config_id":m.IMAGE_ID,
        "candidate_2GiB_process_peak_cap_UNAPPROVED_bytes":CAP,
        "original_graph_vs_static_array_crosschecks":items,
        "original_72_P1_scientific_cases_NOT_RUN":72,
        "actual_50k_complete_PCC_fit_resource_upper_bound_proven":False,
        "cannot_legitimately_add_graph_peak_to_array_payload_as_fit_RSS_upper_bound":True,
        "Numba_JIT_and_Cython_observer_live_RSS_not_measured":True,
        "candidate_per_fit_30min_wall_and_1hr_CPU_not_measured":True,
        "uncleared_final_P2_resource_and_toolchain_requirements":list(MISSING),
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    }
    validate(report)
    return report

def validate(report):
    require(type(report) is dict and report.get("schema")==SCHEMA and
            report.get("status")==STATUS and
            report.get("P1_scientific_sha256")==P1_SHA and
            report.get("old_original_P2_07A_static_run")==STATIC_RUN_ID and
            report.get("old_original_P2_07A_static_artifact")==STATIC_ARTIFACT_ID and
            report.get("old_original_P2_07M_exact_OCI_run")==OCI_RUN_ID and
            report.get("old_original_P2_07M_exact_OCI_artifact")==OCI_ARTIFACT_ID and
            report.get("original_P2_07M_canonical_SHA256")==OCI_EVIDENCE_SHA and
            report.get("original_OCI_archive_SHA256")==m.IMAGE_SHA and
            report.get("original_OCI_config_id")==m.IMAGE_ID and
            report.get("candidate_2GiB_process_peak_cap_UNAPPROVED_bytes")==CAP,
            "Unrecognized original source, P1, OCI or unapproved resource proposal")
    rows=report.get("original_graph_vs_static_array_crosschecks")
    require(type(rows) is list and len(rows)==3 and
            [x.get("uci_id") for x in rows]==list(m.IDS),
            "Three original real graphs, in exact P1 order, required")
    for row,(uci,n,f,classes) in zip(rows,a.EXPECTED_DATASETS):
        width=m.DIMENSIONS[uci][2]
        table=a.model_persistent_array_bytes(n,f,classes,width)
        maximum=max(
            sum(table.values())+
            a.snapshot_bytes(n,classes,be)["observer_allocation_plus_bookkeeping_bytes"]
            for be in a.BACKENDS)
        peak=row.get("actual_OCI_graph_only_peak_process_RSS_bytes")
        require(row.get("original_graph_SHA256")==m.GRAPH_HASHES[uci] and
                row.get("original_node_count")==n and
                row.get("original_graph_dense_width")==width and
                row.get("original_pcc_particles_per_approved_mask")==3*classes and
                row.get("static_retained_array_payload_bytes_NOT_RSS_BOUND")==sum(table.values()) and
                row.get("static_largest_backend_snapshot_plus_arrays_payload_NOT_RSS_BOUND")==maximum and
                type(peak) is int and 0<peak<CAP and
                row.get("candidate_cap_minus_observed_graph_only_RSS_bytes_NOT_FIT_HEADROOM_GUARANTEE")==CAP-peak and
                row.get("graph_only_observed_below_draft_2GiB_cap") is True and
                row.get("this_does_NOT_include_PCC_propagation_JIT_checkpoint_live_peak") is True,
                "Tampered P1 graph, array formulas, source RSS or false fit feasibility")
    for key,expected in (
        ("original_72_P1_scientific_cases_NOT_RUN",72),
        ("actual_50k_complete_PCC_fit_resource_upper_bound_proven",False),
        ("cannot_legitimately_add_graph_peak_to_array_payload_as_fit_RSS_upper_bound",True),
        ("Numba_JIT_and_Cython_observer_live_RSS_not_measured",True),
        ("candidate_per_fit_30min_wall_and_1hr_CPU_not_measured",True),
        ("uncleared_final_P2_resource_and_toolchain_requirements",list(MISSING)),
        ("P2_01_final_execution_toolchain_lock_pass",False),
        ("P2_07_full_resources_gate_pass",False),
        ("P2_execution_preflight_sha256",None),
        ("P2_execution_authorized",False),
        ("real_PCC_fits_executed",0),
    ):
        require(report.get(key)==expected,"Forbidden resource, provenance or P2 execution approval: "+key)
    return True

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original-static",required=True,type=Path)
    ap.add_argument("--original-OCI",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args()
    require(not args.output.exists(),"Never overwrite existing admission/evidence record")
    report=build(read_exact(args.original_static),read_exact(args.original_OCI))
    with args.output.open("xb") as dest:
        dest.write(json.dumps(report,sort_keys=True,indent=2).encode("utf-8")+b"\n")
        dest.flush()
        os.fsync(dest.fileno())
    require(validate(json.loads(args.output.read_text(encoding="utf-8"))),
            "Persisted resource review no-GO evidence incomplete")
    print(json.dumps({
        "status":STATUS,
        "source_free_review_sha256_NOT_EXECUTION":digest(report),
        "original_graph_only_OCI_peak_RSS_bytes":
            {str(r["uci_id"]):r["actual_OCI_graph_only_peak_process_RSS_bytes"]
             for r in report["original_graph_vs_static_array_crosschecks"]},
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    },sort_keys=True))

if __name__=="__main__":
    main()
