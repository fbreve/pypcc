"""P2-07K adversarial source-only checks: no OCI, UCI, native PCC or fits."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import p2_07k_original_resource_host_admission_no_fit as k


class OriginalResourceHostSourceOnlyGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy, cls.limits, cls.cases, cls.budget = k.original_candidate()
        cls.projection = k.budget_projection(cls.limits, cls.budget)
        cls.valid = {
            "schema": k.SCHEMA,
            "status": k.STATUS,
            "scientific_protocol_P1_SHA256": k.P1_SHA,
            "proposed_original_P2_07C_policy_canonical_SHA256": k.POLICY_SHA,
            "verified_original_OCI_payload_SHA256": k.OCI_SHA,
            "verified_original_OCI_config_id": k.OCI_ID,
            "original_P2_07J_source_free_run": k.J_RUN,
            "original_P2_07J_artifact": k.J_ARTIFACT,
            "original_P2_07J_canonical_report_SHA256": k.J_REPORT_SHA,
            "original_P2_07J_504row_journal_SHA256": k.J_AUDIT_SHA,
            "independent_original_PCC_regression_source_run": k.J_PRODUCTION_REGRESSION_RUN,
            "original_P1_scientific_case_IDs": [x["case_id"] for x in cls.cases],
            "candidate_host": cls.policy["proposed_execution_host"],
            "candidate_guardrails_UNAPPROVED": cls.limits,
            "candidate_resource_budget_arithmetic_ONLY": cls.projection,
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
            "missing_execution_requirements": list(k.PENDING),
            "missing_execution_requirements_count": len(k.PENDING),
            "P2_01_final_execution_toolchain_lock_pass": False,
            "P2_07_full_resources_gate_pass": False,
            "P2_execution_preflight_sha256": None,
            "P2_execution_authorized": False,
            "real_PCC_fits_executed": 0,
        }

    def test_exact_original_frozen_science_candidate_policy_and_72_ids(self):
        self.assertEqual(k.digest(self.policy), k.POLICY_SHA)
        self.assertEqual(self.policy["fit_schedule"]["n_fits"], 72)
        self.assertEqual(len(self.cases), 72)
        self.assertEqual(len(set(x["case_id"] for x in self.cases)), 72)
        self.assertTrue(k.validate_report(self.valid))

    def test_budget_arithmetic_is_not_real_data_resource_measurement(self):
        self.assertEqual(self.projection["candidate_sum_wall_slots_seconds_NOT_REAL_DURATION"], 129600)
        self.assertEqual(self.projection["candidate_total_cpu_slots_seconds_NOT_MEASURED"], 259200)
        self.assertEqual(self.projection["candidate_non_fit_overhead_seconds"], 43200)
        self.assertEqual(self.projection["candidate_json_and_log_max_bytes_all_cases"], 216*1024**2)
        self.assertFalse(self.projection["actual_real_UCI_CPU_wall_RSS_scratch_upper_bounds_measured"])

    def test_memory_host_RSS_scratch_disk_and_log_limits_are_proposals(self):
        self.assertEqual(self.limits["min_host_installed_ram_bytes"], 8*1024**3)
        self.assertEqual(self.limits["max_one_fit_peak_process_resident_bytes"], 2*1024**3)
        self.assertEqual(self.limits["min_free_disk_before_start_bytes"], 12*1024**3)
        self.assertEqual(self.limits["max_one_fit_scratch_bytes"], 1024**3)
        self.assertFalse(self.policy["approved_execution_host_and_resource_limits"])

    def test_false_p2_approval_and_results_blocked(self):
        for key, fake in (
            ("P2_execution_authorized", True),
            ("P2_execution_preflight_sha256", "f"*64),
            ("P2_01_final_execution_toolchain_lock_pass", True),
            ("P2_07_full_resources_gate_pass", True),
            ("real_PCC_fits_executed", 1),
            ("verified_real_PCC_fit_count", 1),
            ("verified_synthetic_PCC_propagation_count", 1),
            ("final_P2_08_reviewed_exact_execution_SHA256", "e"*64),
            ("final_P2_08_manifest_created", True),
        ):
            with self.subTest(key=key):
                with self.assertRaises(k.AdmissionBlocked):
                    k.validate_report({**self.valid, key: fake})

    def test_host_resource_and_evidence_forgery_rejected(self):
        for key, fake in (
            ("complete_source_reproducible_native_and_OS_build_closure_verified", True),
            ("real_original_P1_50k_graph_resource_bound_verified", True),
            ("candidate_resource_policy_approved_by_maintainer", True),
            ("source_free_supervisor_is_an_actual_scientific_worker_runner", True),
            ("durable_OCI_or_full_toolchain_offline_backup_verified", True),
            ("verified_actual_source_free_synthetic_supervised_trees", 6),
            ("verified_original_P2_07J_administrative_ID_rows", 1584),
        ):
            with self.subTest(key=key):
                with self.assertRaises(k.AdmissionBlocked):
                    k.validate_report({**self.valid,key:fake})

    def test_missing_blocker_or_wrong_scientific_order_rejected(self):
        bad=copy.deepcopy(self.valid)
        bad["missing_execution_requirements"].pop()
        with self.assertRaises(k.AdmissionBlocked):
            k.validate_report(bad)
        bad=copy.deepcopy(self.valid)
        bad["original_P1_scientific_case_IDs"][0],bad["original_P1_scientific_case_IDs"][1]=(
            bad["original_P1_scientific_case_IDs"][1],bad["original_P1_scientific_case_IDs"][0])
        with self.assertRaises(k.AdmissionBlocked):
            k.validate_report(bad)

    def test_changed_original_budget_or_unapproved_limits_denied(self):
        bad=copy.deepcopy(self.valid)
        bad["candidate_guardrails_UNAPPROVED"]["max_one_fit_wall_seconds"]=1860
        with self.assertRaises(k.AdmissionBlocked):
            k.validate_report(bad)
        bad=copy.deepcopy(self.projection)
        bad["actual_real_UCI_CPU_wall_RSS_scratch_upper_bounds_measured"]=True
        with self.assertRaises(k.AdmissionBlocked):
            k.budget_projection({**self.limits,"max_one_fit_wall_seconds":1860},self.budget)
        bad=copy.deepcopy(self.valid)
        bad["candidate_resource_budget_arithmetic_ONLY"]=bad.get(
            "candidate_resource_budget_arithmetic_ONLY").copy()
        bad["candidate_resource_budget_arithmetic_ONLY"]["candidate_total_cpu_slots_seconds_NOT_MEASURED"]=1
        with self.assertRaises(k.AdmissionBlocked):
            k.validate_report(bad)

    def test_forged_original_P2_07J_run_report_digest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/"j.json").write_text(json.dumps({"schema":"FORGED"}),encoding="utf-8")
            (p/"j.jsonl").write_bytes(b"fake\n")
            (p/"journals").mkdir()
            with self.assertRaises(k.AdmissionBlocked):
                k.exact_old_j(p/"j.json",p/"j.jsonl",p/"journals")

    def test_missing_previous_journal_or_original_artifact_denied(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/"report.json").write_bytes(b"{}")
            with self.assertRaises(k.AdmissionBlocked):
                k.exact_old_j(p/"report.json",p/"absent.jsonl",p)
            (p/"audit.jsonl").write_bytes(b"fake\n")
            with self.assertRaises(k.AdmissionBlocked):
                k.exact_old_j(p/"report.json",p/"audit.jsonl",p/"missing")

    def test_deterministic_canonical_review_and_sha_identity(self):
        self.assertTrue(k.validate_report(json.loads(json.dumps(self.valid))))
        self.assertEqual(k.canonical(self.valid),k.canonical(json.loads(json.dumps(self.valid))))
        self.assertEqual(k.digest(self.valid),k.digest(json.loads(json.dumps(self.valid))))

    def test_unverified_original_J_report_cannot_build_decision(self):
        with self.assertRaises(k.AdmissionBlocked):
            k.build_report({"status":"UNVERIFIED_FAKE_PASS",
                            "P2_execution_authorized":False})

    def test_false_original_policy_approval_rejected_at_input_gate(self):
        bad=copy.deepcopy(self.policy)
        bad["approved_execution_host_and_resource_limits"]=True
        with patch.object(k.c,"load",return_value=bad):
            with self.assertRaises((k.AdmissionBlocked,k.c.AdmissionBlocked)):
                k.original_candidate()


if __name__ == "__main__":
    unittest.main()
