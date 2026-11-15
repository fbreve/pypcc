"""P2-07N source-only adversarial evidence reconciliation; zero Docker/data/PCC."""
import copy
import json
import unittest

from experiments import p2_07n_original_oci_vs_static_resource_gap_no_fit as n
from experiments import p2_07_metadata_only_resource_budget as a
from experiments import p2_07m_exact_oci_original_graph_preparation_no_fit as m


def original_static_stub():
    rows=[]
    for uci,node_count,features,classes in a.EXPECTED_DATASETS:
        width=m.DIMENSIONS[uci][2]
        table=a.model_persistent_array_bytes(node_count,features,classes,width)
        backends={}
        for backend in a.BACKENDS:
            snap=a.snapshot_bytes(node_count,classes,backend)
            backends[backend]={
                "snapshots":snap,
                "retained_arrays_plus_observer_hypothetical_co_resident_bytes":
                    sum(table.values())+snap["observer_allocation_plus_bookkeeping_bytes"],
                "fits_proposed_for_dataset_backend":8,
            }
        rows.append({
            "uci_id":uci,
            "n_original_rows":node_count,
            "n_numeric_features":features,
            "n_classes":classes,
            "n_original_labeled_per_fit":3*classes,
            "dense_neighbor_width":width,
            "verified_graph_storage_bytes":8*node_count*(width+1),
            "retained_array_line_items_bytes":table,
            "sum_retained_array_payload_bytes":sum(table.values()),
            "proposed_particle_update_opportunities_per_fit":50000*3*classes,
            "backend_observer_payloads":backends,
        })
    return {
        "schema":"pypcc-P2-07-meta-only-array-and-snapshot-budget/v1",
        "status":"SNAPSHOT_ARRAY_BYTES_WITHIN_CAP_REAL_EXECUTION_CPU_RSS_DISK_STILL_UNVERIFIED",
        "P1_scientific_sha256":n.P1_SHA,
        "parent_P2_04_05_PR":267,
        "parent_P2_04_05_workflow_run":43475652889,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "PCC_real_data_fits_executed":0,
        "PCC_particle_propagations_executed":0,
        "actual_P2_04_05_graph_hashes":
            {str(uci):m.GRAPH_HASHES[uci] for uci in m.IDS},
        "accounting":{
            "total_intended_fits":72,
            "total_proposed_PCC_iterations_all_fits":3600000,
            "total_particle_update_opportunities_all_fits":25200000,
            "total_eight_checkpoint_array_payloads_all_72_fits_bytes":14079360,
            "maximum_hypothetical_co_resident_array_payload_bytes_any_one_fit":650012,
            "array_payload_not_a_process_RSS_upper_bound":True,
            "projection_not_a_CPU_time_benchmark":True,
            "datasets":rows,
        },
        "safety_and_approval_gates":{
            "P2_07_snapshot_payload_within_original_64MiB_per_fit_cap":True,
            "P2_07_total_CPU_wall_time_budget_verified":False,
            "P2_07_native_backend_peak_RSS_budget_verified":False,
            "P2_07_real_fit_artifact_disk_and_log_budget_verified":False,
            "P2_07_full_resources_gate_pass":False,
        },
    }


def source_only_fake_report():
    rows=[]
    for uci,node_count,features,classes in a.EXPECTED_DATASETS:
        width=m.DIMENSIONS[uci][2]
        table=a.model_persistent_array_bytes(node_count,features,classes,width)
        maxbytes=max(sum(table.values())+
            a.snapshot_bytes(node_count,classes,be)["observer_allocation_plus_bookkeeping_bytes"]
            for be in a.BACKENDS)
        peak={212:189612032,850:189870080,292:188956672}[uci]
        rows.append({
            "uci_id":uci,
            "original_graph_SHA256":m.GRAPH_HASHES[uci],
            "original_node_count":node_count,
            "original_graph_dense_width":width,
            "original_pcc_particles_per_approved_mask":3*classes,
            "actual_OCI_graph_only_peak_process_RSS_bytes":peak,
            "static_retained_array_payload_bytes_NOT_RSS_BOUND":sum(table.values()),
            "static_largest_backend_snapshot_plus_arrays_payload_NOT_RSS_BOUND":maxbytes,
            "candidate_cap_minus_observed_graph_only_RSS_bytes_NOT_FIT_HEADROOM_GUARANTEE":
                n.CAP-peak,
            "graph_only_observed_below_draft_2GiB_cap":True,
            "this_does_NOT_include_PCC_propagation_JIT_checkpoint_live_peak":True,
        })
    return {
        "schema":n.SCHEMA,
        "status":n.STATUS,
        "P1_scientific_sha256":n.P1_SHA,
        "old_original_P2_07A_static_run":n.STATIC_RUN_ID,
        "old_original_P2_07A_static_artifact":n.STATIC_ARTIFACT_ID,
        "original_static_source_only_evidence_canonical_SHA256":n.digest(original_static_stub()),
        "old_original_P2_07M_exact_OCI_run":n.OCI_RUN_ID,
        "old_original_P2_07M_exact_OCI_artifact":n.OCI_ARTIFACT_ID,
        "original_P2_07M_canonical_SHA256":n.OCI_EVIDENCE_SHA,
        "original_OCI_archive_SHA256":m.IMAGE_SHA,
        "original_OCI_config_id":m.IMAGE_ID,
        "candidate_2GiB_process_peak_cap_UNAPPROVED_bytes":n.CAP,
        "original_graph_vs_static_array_crosschecks":rows,
        "original_72_P1_scientific_cases_NOT_RUN":72,
        "actual_50k_complete_PCC_fit_resource_upper_bound_proven":False,
        "cannot_legitimately_add_graph_peak_to_array_payload_as_fit_RSS_upper_bound":True,
        "Numba_JIT_and_Cython_observer_live_RSS_not_measured":True,
        "candidate_per_fit_30min_wall_and_1hr_CPU_not_measured":True,
        "uncleared_final_P2_resource_and_toolchain_requirements":list(n.MISSING),
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    }


class OriginalNOFITResourceGapTests(unittest.TestCase):
    def test_existing_immutable_original_72case_static_formula_is_valid(self):
        self.assertEqual(len(n.check_old_static(original_static_stub())),3)
        self.assertEqual(len(n.MISSING),6)

    def test_original_static_sha_and_wrong_72case_or_milestones_rejected(self):
        for key,badval in (
            ("total_intended_fits",71),
            ("total_proposed_PCC_iterations_all_fits",100),
            ("total_eight_checkpoint_array_payloads_all_72_fits_bytes",1),
            ("maximum_hypothetical_co_resident_array_payload_bytes_any_one_fit",650013),
            ("array_payload_not_a_process_RSS_upper_bound",False),
        ):
            with self.subTest(key=key):
                bad=original_static_stub()
                bad["accounting"][key]=badval
                with self.assertRaises(n.BudgetBlocked):
                    n.check_old_static(bad)

    def test_wrong_original_dense_graph_width_and_backend_arrays_rejected(self):
        for subkey,badval in (
            ("dense_neighbor_width",25),
            ("verified_graph_storage_bytes",1),
            ("n_original_labeled_per_fit",100),
            ("sum_retained_array_payload_bytes",17),
        ):
            with self.subTest(key=subkey):
                bad=original_static_stub()
                bad["accounting"]["datasets"][0][subkey]=badval
                with self.assertRaises(n.BudgetBlocked):
                    n.check_old_static(bad)
        bad=original_static_stub()
        bad["accounting"]["datasets"][0]["backend_observer_payloads"].pop(a.BACKENDS[0])
        with self.assertRaises(n.BudgetBlocked):
            n.check_old_static(bad)

    def test_static_original_fake_scientific_authorization_rejected(self):
        for key,value in (
            ("P2_execution_authorized",True),
            ("PCC_real_data_fits_executed",1),
            ("PCC_particle_propagations_executed",1),
            ("P2_execution_preflight_sha256","f"*64),
        ):
            with self.subTest(key=key):
                doc=original_static_stub()
                doc[key]=value
                with self.assertRaises(n.BudgetBlocked):
                    n.check_old_static(doc)

    def test_draft_per_fit_residual_is_not_full_fit_bound(self):
        r=source_only_fake_report()
        self.assertTrue(n.validate(r))
        for row in r["original_graph_vs_static_array_crosschecks"]:
            self.assertGreater(row["candidate_cap_minus_observed_graph_only_RSS_bytes_NOT_FIT_HEADROOM_GUARANTEE"],0)
            self.assertTrue(row["this_does_NOT_include_PCC_propagation_JIT_checkpoint_live_peak"])
        self.assertFalse(r["actual_50k_complete_PCC_fit_resource_upper_bound_proven"])

    def test_false_go_claims_never_accepted_in_consolidated_report(self):
        base=source_only_fake_report()
        for key,value in (
            ("actual_50k_complete_PCC_fit_resource_upper_bound_proven",True),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_preflight_sha256","a"*64),
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",72),
            ("original_72_P1_scientific_cases_NOT_RUN",0),
            ("candidate_per_fit_30min_wall_and_1hr_CPU_not_measured",False),
            ("Numba_JIT_and_Cython_observer_live_RSS_not_measured",False),
        ):
            with self.subTest(key=key):
                with self.assertRaises(n.BudgetBlocked):
                    n.validate({**base,key:value})

    def test_forged_graph_sha_memory_array_and_policy_cap_rejected(self):
        for key,value in (
            ("original_graph_SHA256","0"*64),
            ("actual_OCI_graph_only_peak_process_RSS_bytes",3*1024**3),
            ("static_retained_array_payload_bytes_NOT_RSS_BOUND",1),
            ("static_largest_backend_snapshot_plus_arrays_payload_NOT_RSS_BOUND",99),
            ("candidate_cap_minus_observed_graph_only_RSS_bytes_NOT_FIT_HEADROOM_GUARANTEE",1),
        ):
            with self.subTest(key=key):
                bad=source_only_fake_report()
                bad["original_graph_vs_static_array_crosschecks"][0][key]=value
                with self.assertRaises(n.BudgetBlocked):
                    n.validate(bad)
        bad=source_only_fake_report()
        bad["candidate_2GiB_process_peak_cap_UNAPPROVED_bytes"]=999
        with self.assertRaises(n.BudgetBlocked):
            n.validate(bad)

    def test_extra_missing_or_reordered_graphs_not_admitted(self):
        rows=source_only_fake_report()["original_graph_vs_static_array_crosschecks"]
        for badrows in (rows[:2],rows+[rows[0]],[rows[1],rows[0],rows[2]]):
            bad=source_only_fake_report()
            bad["original_graph_vs_static_array_crosschecks"]=badrows
            with self.assertRaises(n.BudgetBlocked):
                n.validate(bad)

    def test_incomplete_OCI_original_actual_evidence_rejected(self):
        with self.assertRaises((n.BudgetBlocked,ValueError)):
            n.check_old_oci({"schema":"FAKE","P2_execution_authorized":False})
        with self.assertRaises((n.BudgetBlocked,ValueError)):
            n.build(original_static_stub(),{"P2_execution_authorized":False})

    def test_sha_is_review_only_and_canonical_deterministic(self):
        report=source_only_fake_report()
        self.assertEqual(n.digest(report),n.digest(json.loads(json.dumps(report))))
        self.assertIsNone(report["P2_execution_preflight_sha256"])
        self.assertFalse(report["P2_execution_authorized"])
        self.assertTrue(n.validate(report))


if __name__=="__main__":
    unittest.main()
