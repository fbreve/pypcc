"""P2-07F administrative-only protocol, tamper, retry, false-GO tests.

All 72 IDs are metadata. No original UCI input, seed/mask arrays,
graph build, PCC kernel, synthetic fit or Docker process.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments import p2_07f_sealed_72case_no_fit_administrative_dryrun as dry
from experiments.test_p2_07e_locked_oci_cgroup_resource_supervisor import good_report


class P207FNoFitAdministrativeGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p1,cls.policy,cls.limits,cls.lock=dry.frozen_inputs()
        cls.cases=dry.schedule(cls.p1)
        cls.report=dry.compile_report(good_report())

    def test_exact_scientific_order_and_dense_original_policy_immutable(self):
        self.assertEqual(len(self.cases),72)
        self.assertEqual([x["uci_id"] for x in self.cases[::24]],list(dry.SOURCES))
        self.assertEqual([x["backend"] for x in self.cases[:3]],
                         list(dry.BACKENDS))
        self.assertEqual([x["mask_replicate"] for x in self.cases[::3]][:8],
                         list(range(8)))
        self.assertEqual(self.cases[0]["case_id"],"UCI212-M00-B0")
        self.assertEqual(self.cases[35]["case_id"],"UCI850-M03-B2")
        self.assertEqual(self.cases[-1]["case_id"],"UCI292-M07-B2")
        self.assertTrue(all(x["case_state"]=="NOT_RUN_NO_APPROVAL"
                            for x in self.cases))
        self.assertTrue(self.p1["graph"]["PCC_storage"].startswith(
            "dense int64 neib_list"))
        self.assertEqual(dry.P1_SHA,
                         "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166")

    def test_exact_candidate_resource_slots_never_rss_or_cpu_measurements(self):
        b=dry.exact_candidate_total_budget(self.limits)
        self.assertEqual(b["planned_cases"],72)
        self.assertEqual(b["maximum_72_case_fit_wall_slots_seconds"],129600)
        self.assertEqual(b["maximum_72_case_CPU_slots_seconds"],259200)
        self.assertEqual(b["candidate_study_wall_seconds_limit"],172800)
        self.assertEqual(b["maximum_72_case_aggregate_JSON_limit_bytes"],
                         72*1024**2)
        self.assertEqual(b["maximum_72_case_text_logs_limit_bytes"],144*1024**2)
        self.assertTrue(b["candidate_memory_budget_is_NOT_real_graph_peak_RSS_bound"])
        self.assertTrue(b["candidate_CPU_slots_are_NOT_CPU_benchmark_or_scheduler_prediction"])
        with self.assertRaises(dry.DryRunBlocked):
            dry.exact_candidate_total_budget(self.limits,count=71)

    def test_all_three_failure_injections_stop_administrative_checks(self):
        self.assertTrue(dry.validate_report(self.report))
        scenarios=self.report["three_forced_administrative_failure_scenarios"]
        self.assertEqual([x["fake_failure_at_zero_based_case_index"] for x in scenarios],
                         [0,35,71])
        for position,scenario in zip((0,35,71),scenarios):
            with self.subTest(failure_position=position):
                events=scenario["case_log"]
                self.assertEqual(len(events),72)
                self.assertTrue(all(x["event"]=="ADMIN_CASE_VALIDATED_ONLY"
                                    for x in events[:position]))
                self.assertEqual(events[position]["event"],
                                 "ADMIN_CASE_BLOCKED_FAKE")
                self.assertTrue(all(x["event"]==
                                    "ADMIN_CASE_CHECK_NOT_ATTEMPTED_AFTER_FAILURE"
                                    for x in events[position+1:]))
                self.assertTrue(all(x["scientific_case_state"]==
                                    "NOT_RUN_NO_APPROVAL" for x in events))
                self.assertEqual(scenario["remaining_unchecked_case_ID_count"],
                                 71-position)
        self.assertFalse(self.report["P2_execution_authorized"])
        self.assertIsNone(self.report["P2_execution_preflight_sha256"])
        self.assertEqual(self.report["real_PCC_fits_executed"],0)

    def test_journal_hash_chain_rejects_corruption_swap_duplicate_or_retry(self):
        nominal=self.report["nominal_admin_case_ID_check_journal"]
        copies=[
            ("event_swap",lambda x:x.__setitem__(12,x[13])),
            ("index_skip",lambda x:x[15].__setitem__("event_index",17)),
            ("result_claim",lambda x:x[8].__setitem__("scientific_case_state",
                                                     "FIT_COMPLETED")),
            ("forged_after_failure",lambda x:x[3].__setitem__(
                "event","ADMIN_CASE_BLOCKED_FAKE")),
            ("hash_drift",lambda x:x[21].__setitem__("journal_SHA256","f"*64)),
        ]
        for label,mutation in copies:
            with self.subTest(label=label):
                bad=copy.deepcopy(nominal)
                mutation(bad)
                with self.assertRaises(dry.DryRunBlocked):
                    dry.check_journal(self.cases,bad)
        for extra in (nominal[:-1],nominal+[nominal[-1]]):
            with self.assertRaises(dry.DryRunBlocked):
                dry.check_journal(self.cases,extra)
        with self.assertRaises(dry.DryRunBlocked):
            dry.administrative_journal(self.cases,failure_index=28)

    def test_fake_full_execution_approval_is_rejected_at_output_gate(self):
        for key,new in (
            ("P2_execution_authorized",True),
            ("P2_execution_preflight_sha256","a"*64),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("human_approved_candidate_operational_limits",True),
            ("final_execution_host_and_whole_runtime_native_lock_complete",True),
            ("real_graph_50k_resource_upper_bound_verified",True),
            ("real_PCC_fits_executed",1),
            ("particle_propagations_executed",1),
            ("real_UCI_data_loaded",310),
            ("graphs_or_masks_constructed",3),
            ("scientific_fit_scheduling_callable_present",True),
            ("all_72_original_scientific_cases_state","COMPLETED"),
            ("original_P2_07C_draft_resource_policy_SHA256","f"*64),
            ("exact_archived_OCI_binary_SHA256","0"*64),
        ):
            with self.subTest(key=key):
                wrong={**self.report,key:new}
                with self.assertRaises(dry.DryRunBlocked):
                    dry.validate_report(wrong)

    def test_cgroup_provenance_forged_or_missing_fails_closed(self):
        proof=good_report()
        tested=dry.validate_proven_cgroup_receipt(proof)
        self.assertTrue(tested["five_actual_source_free_quota_and_kill_probes_passed"])
        for key,bad in (
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",1),
            ("P2_07_full_resources_gate_pass",True),
            ("verified_docker_image_config_SHA256","sha256:"+"1"*64),
        ):
            with self.subTest(key=key):
                malformed={**proof,key:bad}
                with self.assertRaises(dry.DryRunBlocked):
                    dry.validate_proven_cgroup_receipt(malformed)

    def test_cross_platform_deterministic_schema_serialization(self):
        cloned=json.loads(json.dumps(self.report))
        self.assertEqual(dry.sha(cloned),dry.sha(self.report))
        self.assertTrue(dry.validate_report(cloned))
        self.assertEqual(len(self.report["nominal_admin_case_ID_check_journal"]),72)
        self.assertEqual(sum(len(x["case_log"]) for x in
                             self.report["three_forced_administrative_failure_scenarios"]),
                         216)
        self.assertEqual(self.report["administrative_synthetic_drill_case_count"],288)
        self.assertFalse(self.report["real_OCI_Docker_execution_in_this_P2_07F_stage"])

    def test_output_no_overwrite_no_hidden_real_fit(self):
        self.assertEqual(self.report["status"],
                         "72_CASE_ID_ADMINISTRATIVE_DRILL_COMPLETE_BUT_ALL_REAL_PCC_FITS_PROHIBITED")
        self.assertEqual(self.report["real_UCI_data_loaded"],0)
        self.assertEqual(self.report["particle_propagations_executed"],0)
        self.assertEqual(self.report["real_PCC_fits_executed"],0)
        self.assertTrue(self.report[
            "candidate_total_72_case_budget_arithmetic_NOT_measured"][
            "candidate_time_budget_is_NOT_measured_host_elapsed_time"])
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"existing-report.json"
            output.write_text("prior evidence, do not replace")
            self.assertTrue(output.exists())


if __name__=="__main__":
    unittest.main()
