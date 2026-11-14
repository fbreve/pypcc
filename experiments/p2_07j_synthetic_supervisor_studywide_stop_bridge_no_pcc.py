"""P2-07J: actual synthetic H supervisor -> original-I study STOP bridge, NO PCC.

This is a research-only technical probe. The original 72 P1 scientific IDs
are metadata, not executable jobs. Exactly five synthetic H process trees
are launched in the complete live CLI run (one per adverse resource/root
mode); two additional faults are PRE-LAUNCH administrative injections.
No UCI data, graph, mask, real seed, or PCC implementation is imported/run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments import p2_07h_integrated_OCI_resource_tree_supervisor_no_pcc as h
from experiments import p2_07i_studywide_failclosed_72case_no_fit as i
from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as e

# One real source-free H tree per independently observed failure class;
# precise first/middle/last original P1 identity positions are covered.
# The two I-only administrative faults cannot truthfully be passed off as
# Linux-kernel-observed failures.
PLAN = (
    ("WALL_DEADLINE", 0, "wall"),
    ("CPU_TIME_BUDGET", 35, "cpu"),
    ("CGROUP_MEMORY_OOM", 71, "memory"),
    ("TMPFS_SCRATCH_ENOSPC", 0, "scratch"),
    ("WORKER_ROOT_FAILED", 35, "crash"),
    ("DESCENDANT_CLEANUP_UNVERIFIED", 71, None),
    ("STUDY_WALL_DEADLINE", 0, None),
)
REASONS = {
    "wall": "HOST_SAMPLED_WALL_THRESHOLD_STOP",
    "cpu": "HOST_SAMPLED_CGROUP_CPU_THRESHOLD_STOP",
    "memory": "KERNEL_OOM_EVENT_DESCENDANT_FORCED_FULL_TREE_STOP",
    "scratch": "KERNEL_TMPFS_ENOSPC_FORCED_FULL_TREE_STOP",
    "crash": "ROOT_EXIT_23_AND_CONTAINER_KILLED_DESCENDANTS",
}
EVENTS = (
    "STARTED", "TREE_AND_HARD_CAPS_VERIFIED",
    "RELEASED_SYNTHETIC_BARRIER", "TERMINATION_DECISION_RECORDED",
    "ALL_PIDS_AND_CONTAINER_GONE",
)
SCHEMA = "pypcc-P2-07J-integrated-synthetic-supervisor-studywide-stop-no-PCC/v1"
STATUS = "FIVE_LIVE_SYNTHETIC_STOPS_TWO_ADMIN_INJECTIONS_PCC_BLOCKED"


class BridgeBlocked(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise BridgeBlocked(message)


def ledger_name(fault, index):
    return "p2-07j-synthetic-" + fault.lower().replace("_", "-") + "-%02d.jsonl" % index


def spec_for(fault, index):
    spec = next((s for s in i.scenario_spec()
                 if s["fault_reason"] == fault and s["failure_index"] == index), None)
    require(spec is not None, "P2-07I has no such original-ID stop policy scenario")
    return spec


def validate_five_h_events(path, mode, last_sha):
    require(path.is_file() and path.stat().st_size < 12_000,
            "Absent/oversized actual synthetic H process-tree journal")
    try:
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        chain = h.validate_journal(path)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise BridgeBlocked("Invalid actual H resource supervisor journal") from exc
    require(len(rows) == 5 and
            tuple(row.get("event") for row in rows) == EVENTS and
            all(row.get("mode") == mode for row in rows) and
            chain["records"] == 5 and
            chain["tail_sha256"] == last_sha and
            rows[-1].get("exact_original_PID_starttime_identities_gone") == 3,
            "Wrong, incomplete or re-ordered actual supervisor termination/cleanup")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_live_result(mode, journal_path, outcome):
    require(mode in REASONS and type(outcome) is tuple and len(outcome) == 2,
            "Unknown synthetic mode or invalid actual H supervisor return")
    tail, proof = outcome
    require(type(proof) is dict and
            proof.get("mode") == mode and
            (proof.get("termination_reason") == REASONS[mode] or
             (mode == "memory" and proof.get("termination_reason") ==
              "KERNEL_CGROUP_OOM_KILLED_CONTAINER")) and
            proof.get("real_three_initial_host_PID_starttime_identities_gone") is True and
            proof.get("synthetic_PCC_fits_executed") == 0 and
            proof.get("real_PCC_fits_executed") == 0 and
            proof.get("hard_kernel_quotas", {}).get("hard_swap_max_bytes") == 0 and
            proof.get("hard_kernel_quotas", {}).get("hard_pids_max") == 128 and
            type(proof.get("root_exit_code")) is int and
            (proof.get("root_exit_code") == 23 if mode == "crash"
             else proof.get("root_exit_code") != 0) and
            (mode != "memory" or
             proof.get("actual_kernel_cgroup_oom_kill_event_observed") is True) and
            (mode != "scratch" or
             proof.get("actual_tmpfs_ENOSPC_observed") is True) and
            (mode != "cpu" or
             proof.get("cgroup_cpu_usage_seconds_at_last_poll", 0) >= h.CPU_THRESHOLD_SECONDS),
            "A successful/unclean/false-resource synthetic result cannot trigger authenticated STOP")
    audit_sha = validate_five_h_events(journal_path, mode, tail)
    return {
        "mode": mode,
        "verified_actual_synthetic_termination_reason": proof["termination_reason"],
        "original_host_pid_starttime_identities_gone": 3,
        "source_free_H_journal_final_SHA256": tail,
        "source_free_H_journal_bytes_SHA256": audit_sha,
        "synthetic_process_tree_only": True,
        "real_PCC_fits_executed": 0,
    }


def actual_h_launcher(mode, path):
    require(mode in REASONS and not path.exists(), "No arbitrary or retried H fixture")
    return h.one(mode, path, h.GENESIS)


def run_one(fault, index, mode, output_dir, launcher):
    """Scan all original IDs; invoke H precisely at the failing ID, STOP afterwards.

    In unit tests the launcher is a fake injected callback; the CLI always
    supplies actual_h_launcher, checked against original H file evidence.
    """
    require((fault, index, mode) in PLAN, "Unapproved synthetic integration plan")
    cases, _ = i.original_inputs()
    spec = spec_for(fault, index)
    prior = i.GENESIS
    records = []
    invocations = 0
    evidence = None
    for position, case in enumerate(cases):
        if position < index:
            action = "ADMIN_ID_CHECKED_ONLY"
        elif position == index:
            if mode is not None:
                require(invocations == 0, "Cannot retry an H worker")
                path = output_dir / ledger_name(fault, index)
                require(not path.exists(), "Never overwrite original supervisor evidence")
                # A failing/ambiguous H supervisor raises and prevents ALL
                # subsequent original-ID dispatch. No fallback launcher.
                result = launcher(mode, path)
                invocations += 1
                evidence = verify_live_result(mode, path, result)
            else:
                # Deliberate administrative faults, NOT claimed kernel proof.
                evidence = {
                    "injected_administrative_fault_only": fault,
                    "actual_synthetic_process_tree_started": False,
                    "real_PCC_fits_executed": 0,
                }
            action = "ADMIN_FAKE_STOP_NO_FIT"
        else:
            action = "ADMIN_CASE_UNREACHED_AFTER_FAKE_STOP"
        row = i.build_entry(case, spec["scenario"], position, action, prior)
        prior = row["record_SHA256"]
        records.append(row)

    # Compare the integrated dispatch ledger against the independently
    # validated prior I administrative-stop policy (not a PCC fit log).
    expected = i.campaign(cases, spec)
    require(records == expected["administrative_records"] and
            prior == expected["final_journal_SHA256"] and
            invocations == (1 if mode is not None else 0) and
            records[-1]["original_scientific_fit_state"] == "NOT_RUN_NO_APPROVAL",
            "Study-wide STOP changed original-I order or allowed later work")
    return {
        "administrative_policy": expected,
        "live_supervised_artificial_tree_count": invocations,
        "supervisor_or_injected_stop_evidence": evidence,
        "after_STOP_supervisor_dispatches": 0,
        "synthetic_PCC_propagations_executed": 0,
        "real_PCC_fits_executed": 0,
    }


def audit_bytes(campaigns):
    return b"".join(i.canonical(row) + b"\n"
                    for c in campaigns
                    for row in c["administrative_policy"]["administrative_records"])


def build_report(campaigns):
    cases, budget = i.original_inputs()
    require(len(campaigns) == len(PLAN), "Missing actual/injected synthetic STOP scenario")
    source_free_audit = audit_bytes(campaigns)
    require(len(source_free_audit) < 350_000, "Synthetic administrative records oversized")
    report = {
        "schema": SCHEMA,
        "status": STATUS,
        "P1_scientific_sha256": i.P1_SHA,
        "original_P2_07H_source_free_report_SHA256": i.H_REPORT_SHA,
        "exact_original_OCI_gzip_SHA256": i.OCI_GZIP_SHA,
        "exact_original_OCI_config_id": i.OCI_CONFIG_ID,
        "original_72_scientific_IDs": cases,
        "unapproved_original_candidate_budget": budget,
        "bridge_campaigns": campaigns,
        "studywide_administrative_case_rows": 72 * len(PLAN),
        "studywide_administrative_journal_SHA256":
            hashlib.sha256(source_free_audit).hexdigest(),
        "actual_synthetic_supervised_three_process_trees":
            sum(x["live_supervised_artificial_tree_count"] for x in campaigns),
        "deliberately_injected_non_kernel_stop_scenarios": 2,
        "original_scientific_case_states": "ALL_NOT_RUN_NO_APPROVAL",
        "scientific_case_runners_present": False,
        "real_UCI_feature_rows_loaded": 0,
        "real_graph_or_masks_generated": 0,
        "synthetic_PCC_propagations_executed": 0,
        "real_PCC_fits_executed": 0,
        "P2_01_final_execution_toolchain_lock_pass": False,
        "P2_07_full_resources_gate_pass": False,
        "P2_execution_preflight_sha256": None,
        "P2_execution_authorized": False,
    }
    validate_report(report)
    return report


def validate_report(report):
    require(type(report) is dict and report.get("schema") == SCHEMA and
            report.get("status") == STATUS and
            report.get("P1_scientific_sha256") == i.P1_SHA and
            report.get("original_P2_07H_source_free_report_SHA256") == i.H_REPORT_SHA and
            report.get("exact_original_OCI_gzip_SHA256") == i.OCI_GZIP_SHA and
            report.get("exact_original_OCI_config_id") == i.OCI_CONFIG_ID,
            "Source provenance invalid")
    cases, budget = i.original_inputs()
    require(report.get("original_72_scientific_IDs") == cases and
            report.get("unapproved_original_candidate_budget") == budget,
            "Real P1 identity or unapproved budget altered")
    actual = report.get("bridge_campaigns")
    require(type(actual) is list and len(actual) == len(PLAN),
            "Wrong integrated stop campaign count")
    for (fault, index, mode), campaign in zip(PLAN, actual):
        spec = spec_for(fault, index)
        expected = i.campaign(cases, spec)
        require(campaign.get("administrative_policy") == expected and
                campaign.get("live_supervised_artificial_tree_count") == (mode is not None) and
                campaign.get("after_STOP_supervisor_dispatches") == 0 and
                campaign.get("synthetic_PCC_propagations_executed") == 0 and
                campaign.get("real_PCC_fits_executed") == 0,
                "H-I bridge omitted or retried a case or marked a PCC run")
        proof = campaign.get("supervisor_or_injected_stop_evidence")
        require(type(proof) is dict and proof.get("real_PCC_fits_executed") == 0,
                "Missing source-free stop evidence")
        if mode is None:
            require(proof == {
                "injected_administrative_fault_only": fault,
                "actual_synthetic_process_tree_started": False,
                "real_PCC_fits_executed": 0,
            }, "Injected limit falsely claimed a live kernel/cgroup test")
        else:
            require(proof.get("mode") == mode and
                    (proof.get("verified_actual_synthetic_termination_reason") == REASONS[mode] or
                     (mode == "memory" and proof.get("verified_actual_synthetic_termination_reason") ==
                      "KERNEL_CGROUP_OOM_KILLED_CONTAINER")) and
                    proof.get("original_host_pid_starttime_identities_gone") == 3 and
                    proof.get("synthetic_process_tree_only") is True and
                    isinstance(proof.get("source_free_H_journal_bytes_SHA256"), str) and
                    len(proof["source_free_H_journal_bytes_SHA256"]) == 64 and
                    isinstance(proof.get("source_free_H_journal_final_SHA256"), str) and
                    len(proof["source_free_H_journal_final_SHA256"]) == 64,
                    "Missing true live resource/container/descendant closure claim")
    source_free_audit = audit_bytes(actual)
    require(len(source_free_audit) < 350_000 and
            report.get("studywide_administrative_case_rows") == len(PLAN)*72 and
            report.get("studywide_administrative_journal_SHA256") ==
                hashlib.sha256(source_free_audit).hexdigest() and
            report.get("actual_synthetic_supervised_three_process_trees") == 5 and
            report.get("deliberately_injected_non_kernel_stop_scenarios") == 2,
            "Audit content/worker count invalid")
    for key, value in (
        ("original_scientific_case_states", "ALL_NOT_RUN_NO_APPROVAL"),
        ("scientific_case_runners_present", False),
        ("real_UCI_feature_rows_loaded", 0),
        ("real_graph_or_masks_generated", 0),
        ("synthetic_PCC_propagations_executed", 0),
        ("real_PCC_fits_executed", 0),
        ("P2_01_final_execution_toolchain_lock_pass", False),
        ("P2_07_full_resources_gate_pass", False),
        ("P2_execution_preflight_sha256", None),
        ("P2_execution_authorized", False),
    ):
        require(report.get(key) == value, "False scientific GO: " + key)
    return True


def verify_disk_artifacts(output_dir, ledger, report):
    require(ledger.is_file() and ledger.stat().st_size < 350_000 and
            ledger.read_bytes() == audit_bytes(report["bridge_campaigns"]),
            "Full 504-row original case-ID administrative audit was altered")
    for (fault, index, mode), campaign in zip(PLAN, report["bridge_campaigns"]):
        if mode is not None:
            proof = campaign["supervisor_or_injected_stop_evidence"]
            path = output_dir / ledger_name(fault, index)
            require(validate_five_h_events(
                path, mode, proof["source_free_H_journal_final_SHA256"]) ==
                proof["source_free_H_journal_bytes_SHA256"],
                "Detached original H supervisor journal does not match campaign")
        else:
            require(not (output_dir / ledger_name(fault, index)).exists(),
                    "Administrative simulated fault silently spawned a process")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-p2-07h-source-free-report", required=True, type=Path)
    parser.add_argument("--existing-oci-artifact", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-jsonl", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists() and not args.audit_jsonl.exists() and
            not args.evidence_dir.exists(),
            "Never overwrite or resume an integrated original study-stop attempt")
    # Pinned archived report cannot replace a live process/OS resource test.
    old = json.loads(args.original_p2_07h_source_free_report.read_text(encoding="utf-8"))
    i.verify_old_h_receipt(old)
    host = e.verify_original_image(args.existing_oci_artifact.resolve())
    require(host["loaded_exact_original_image_ID"] == h.IMAGE_ID,
            "Not the exact archived P1/P2-01J original OCI payload")
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    campaigns = [run_one(fault, index, mode, args.evidence_dir, actual_h_launcher)
                 for fault, index, mode in PLAN]
    report = build_report(campaigns)
    content = audit_bytes(campaigns)
    with args.audit_jsonl.open("xb") as dst:
        for row in content.splitlines(keepends=True):
            dst.write(row)
            dst.flush()
            os.fsync(dst.fileno())
    verify_disk_artifacts(args.evidence_dir, args.audit_jsonl, report)
    with args.output.open("xb") as dst:
        dst.write(json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        dst.flush()
        os.fsync(dst.fileno())
    print(json.dumps({
        "status": STATUS,
        "actual_synthetic_supervised_three_process_trees": 5,
        "original_72case_metadata_rows": len(PLAN)*72,
        "injected_non_kernel_stops": 2,
        "real_PCC_fits_executed": 0,
        "P2_execution_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
