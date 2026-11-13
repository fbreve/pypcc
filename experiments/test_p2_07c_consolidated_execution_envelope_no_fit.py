"""P2-07C synthetic metadata-only tests. NO dataset, graph, mask or PCC import."""
import copy
import json
import unittest

from experiments import p2_07c_consolidated_execution_envelope_no_fit as gate


def fake_array_report():
    return {
        "schema":"pypcc-P2-07-meta-only-array-and-snapshot-budget/v1",
        "status":"SNAPSHOT_ARRAY_BYTES_WITHIN_CAP_REAL_EXECUTION_CPU_RSS_DISK_STILL_UNVERIFIED",
        "P1_scientific_sha256":gate.P1_SHA,
        "parent_P2_04_05_workflow_run":43475652889,
        "real_dataset_downloads_or_rows_loaded_in_P2_07":0,
        "PCC_real_data_fits_executed":0,
        "P2_execution_authorized":False,
        "P2_execution_preflight_sha256":None,
        "safety_and_approval_gates":{
            "P2_07_snapshot_payload_within_original_64MiB_per_fit_cap":True,
            "P2_07_full_resources_gate_pass":False,
            "P2_07_total_CPU_wall_time_budget_verified":False,
            "P2_07_native_backend_peak_RSS_budget_verified":False,
        },
        "accounting":{
            **gate.EXPECTED_BUDGET,
            "array_payload_not_a_process_RSS_upper_bound":True,
            "projection_not_a_CPU_time_benchmark":True,
            "datasets":[
                {"uci_id":212,"dense_neighbor_width":22},
                {"uci_id":850,"dense_neighbor_width":20},
                {"uci_id":292,"dense_neighbor_width":24},
            ]
        },
    }


def fake_pilot():
    return {
        "schema":"pypcc-P2-07B-original-three-backend-synthetic-CPU-RSS-pilot/v1",
        "status":"THREE_SYNTHETIC_BACKEND_PROCESS_MEASUREMENTS_NO_REAL_P2_GO",
        "P1_scientific_sha256":gate.P1_SHA,
        "no_real_UCI_data_loaded":True,
        "all_synthetic_observer_pair_parity_pass":True,
        "memory_measurement_not_equivalent_across_Windows_Linux":True,
        "NOT_a_reliable_upper_bound_on_P2_50k_runtime_or_RAM":True,
        "P2_07_full_resources_gate_pass":False,
        "real_PCC_fits_executed":0,
        "P2_execution_authorized":False,
        "P2_execution_preflight_sha256":None,
        "candidate_python_patch_and_OS":{"os":"Linux","python":"3.12.16"},
        "samples":[{
            "backend":name,
            "synthetic_graph_only":True,
            "no_UCI_source_downloaded_or_parsed":True,
            "synthetic_nodes":500,
            "synthetic_pcc_iterations_per_fit":1000,
            "synthetic_PCC_fits_did_execute":True,
            "real_PCC_fits_executed":0,
            "peak_is_NOT_backend_fit_incremental_RAM":True,
            "child_process_peak_resident_or_working_set":{
                "peak_memory_bytes": 300000000+i*50000000,
            },
            "verified_paired_cpu":{
                "original_paired_benchmark_checked_within_own_backend":True,
                "median_baseline_CPU_seconds_per_complete_synthetic_fit":0.01,
                "median_observer_CPU_seconds_per_complete_synthetic_fit":0.012,
            },
        } for i, name in enumerate(("numpy","numba","cython"))],
    }


class ProposedEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.p1 = gate.read_approved_P1()
        self.policy = gate.load(gate.POLICY)

    def test_immutable_P1_exact_science_sha_and_72_run_order(self):
        self.assertEqual(gate.sha256(gate.canonical(self.p1)),gate.P1_SHA)
        schedule=gate.case_order(self.p1)
        self.assertEqual(len(schedule),72)
        self.assertEqual(schedule[0]["uci_id"],212)
        self.assertEqual(schedule[0]["replicate"],0)
        self.assertEqual(schedule[0]["backend"],gate.BACKENDS[0])
        self.assertEqual(schedule[-1]["uci_id"],292)
        self.assertEqual(schedule[-1]["replicate"],7)
        self.assertEqual(schedule[-1]["backend"],gate.BACKENDS[-1])
        self.assertEqual({s["status"] for s in schedule},{"NOT_RUN_NO_APPROVAL"})

    def test_proposed_host_limits_and_output_quota_consistent(self):
        budgets=gate.validate_policy(self.policy,self.p1)
        self.assertEqual(budgets["max_one_fit_peak_process_resident_bytes"],2*1024**3)
        self.assertEqual(budgets["max_parallel_fits"] if "max_parallel_fits" in budgets
                         else self.policy["fit_schedule"]["max_parallel_fits"],1)
        self.assertLess(72*3*1024**2,
                        budgets["max_all_72_fits_persisted_evidence_bytes"])
        self.assertLess(72*budgets["max_one_fit_wall_seconds"],
                        budgets["max_entire_study_wall_seconds"])

    def test_fail_closed_on_premature_P2_GOs_and_changed_concurrency(self):
        changes=[
            (("P2_execution_authorized",),True),
            (("P2_07_full_resources_gate_pass",),True),
            (("P2_01_final_execution_toolchain_lock_pass",),True),
            (("real_PCC_fits_authorized",),72),
            (("P2_execution_preflight_sha256",),"0"*64),
            (("fit_schedule","max_parallel_fits"),2),
            (("fit_schedule","scientific_retries_allowed"),1),
            (("candidate_guardrails","max_one_fit_peak_process_resident_bytes"),512*1024**2),
            (("approved_execution_host_and_resource_limits",),True),
            (("export_and_failure_policy","original_feature_rows_or_original_labels_in_public_CI"),True),
        ]
        for keys, value in changes:
            with self.subTest(keys=keys):
                policy=copy.deepcopy(self.policy)
                parent=policy
                for key in keys[:-1]:
                    parent=parent[key]
                parent[keys[-1]]=value
                with self.assertRaises(gate.AdmissionBlocked):
                    gate.validate_policy(policy,self.p1)

    def test_original_static_array_evidence_is_not_real_RSS_bound(self):
        x=gate.validate_arrays(fake_array_report())
        self.assertEqual(x["max_hypothetical_array_payload_bytes"],650012)
        self.assertEqual(x["all_72_snapshot_array_bytes"],14079360)
        forged=fake_array_report()
        forged["safety_and_approval_gates"]["P2_07_full_resources_gate_pass"]=True
        with self.assertRaises(gate.AdmissionBlocked):
            gate.validate_arrays(forged)

    def test_original_synthetic_three_backend_cpu_peak_can_be_used_only_as_pilot(self):
        result=gate.validate_synthetic_pilot(fake_pilot(),"Linux")
        self.assertTrue(result["not_a_real_P2_50k_execution_bound"])
        self.assertEqual(set(result["backend_synthetic_measurements"]),
                         {"numpy","numba","cython"})
        fake=fake_pilot()
        fake["samples"][0]["no_UCI_source_downloaded_or_parsed"]=False
        with self.assertRaises(gate.AdmissionBlocked):
            gate.validate_synthetic_pilot(fake,"Linux")
        fake=fake_pilot()
        fake["real_PCC_fits_executed"]=1
        with self.assertRaises(gate.AdmissionBlocked):
            gate.validate_synthetic_pilot(fake,"Linux")

    def test_native_requires_original_byte_sha_not_just_metadata(self):
        with self.assertRaises(gate.AdmissionBlocked):
            gate.validate_new_native(
                {"schema":"pypcc-P2-01F-prospective-native-toolchain-byte-ledger/v1",
                 "status":"NEW_NATIVE_BINARIES_BUILT_COMPILER_BYTES_RECORDED_PARTIAL_SDK_LEDGER",
                 "P1_scientific_sha256":gate.P1_SHA,
                 "P2_execution_authorized":True},
                gate.ROOT,"Linux")

    def test_synthetic_ledger_combines_without_claiming_real_resource_go(self):
        arrays=gate.validate_arrays(fake_array_report())
        p=gate.validate_synthetic_pilot(fake_pilot(),"Linux")
        pilots={"Linux":p,"Windows":p}
        natives={
            "Linux":{"candidate_installed_versions":{"numpy":"2.5.4"}},
            "Windows":{"candidate_installed_versions":{"numpy":"2.5.4"}},
        }
        output=gate.compute_report(self.policy,self.p1,arrays,pilots,natives)
        self.assertEqual(output["case_schedule"]["count"],72)
        self.assertFalse(output["actual_P2_07_resource_gate_pass"])
        self.assertFalse(output["P2_execution_authorized"])
        self.assertEqual(output["real_PCC_fits_executed"],0)
        self.assertGreater(output["candidate_ram_headroom_multiplier_vs_max_SYNTHETIC_only"],4.0)
        inflated=copy.deepcopy(p)
        inflated["backend_synthetic_measurements"]["numba"]["peak_process_bytes"]=700000000
        with self.assertRaises(gate.AdmissionBlocked):
            gate.compute_report(self.policy,self.p1,arrays,{"Linux":inflated,"Windows":p},natives)


if __name__=="__main__":
    unittest.main()
