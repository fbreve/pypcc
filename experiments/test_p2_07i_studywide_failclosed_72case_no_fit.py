"""P2-07I source-only adversarial complete 72-case global STOP policy tests.

No Docker, original UCI feature rows, graphs, masks, native PCC kernels or
synthetic/real particle propagation in this module.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments import p2_07i_studywide_failclosed_72case_no_fit as i
from experiments.test_p2_07h_integrated_OCI_resource_tree_supervisor_no_pcc import example_report


class P207IStudyWideStopSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=i.build_report()
        cls.cases=cls.report["original_72_original_case_id_inventory"]

    def test_exact_22_campaigns_72_original_case_ids_with_no_PCC(self):
        r=self.report
        self.assertTrue(i.validate_report(r))
        self.assertEqual(r["metadata_only_campaign_count"],22)
        self.assertEqual(r["metadata_only_case_record_count"],1584)
        self.assertEqual(len(self.cases),72)
        self.assertEqual(self.cases[0]["case_id"],"UCI212-M00-B0")
        self.assertEqual(self.cases[35]["case_id"],"UCI850-M03-B2")
        self.assertEqual(self.cases[71]["case_id"],"UCI292-M07-B2")
        self.assertTrue(all(c["case_state"]=="NOT_RUN_NO_APPROVAL"
                            for c in self.cases))
        self.assertEqual(r["real_UCI_dataset_rows_loaded"],0)
        self.assertEqual(r["synthetic_PCC_propagations_executed"],0)
        self.assertEqual(r["real_PCC_fits_executed"],0)
        self.assertFalse(r["P2_execution_authorized"])

    def test_all_seven_faults_at_first_middle_last_denies_every_later_case(self):
        all_campaigns=self.report["administrative_campaigns"]
        self.assertEqual(len(all_campaigns),1+7*3)
        nominal=all_campaigns[0]
        self.assertIsNone(nominal["failure_index"])
        self.assertEqual(nominal["before_fault_admin_ID_validations"],72)
        self.assertTrue(all(x["administrative_event"]==
                            "ADMIN_ID_CHECKED_ONLY"
                            for x in nominal["administrative_records"]))
        for campaign in all_campaigns[1:]:
            index=campaign["failure_index"]
            with self.subTest(reason=campaign["fault_reason"],position=index):
                self.assertIn(index,(0,35,71))
                self.assertTrue(i.validate_campaign(self.cases,campaign))
                events=[row["administrative_event"]
                        for row in campaign["administrative_records"]]
                self.assertEqual(events[:index],
                                 ["ADMIN_ID_CHECKED_ONLY"]*index)
                self.assertEqual(events[index],"ADMIN_FAKE_STOP_NO_FIT")
                self.assertEqual(events[index+1:],
                                 ["ADMIN_CASE_UNREACHED_AFTER_FAKE_STOP"]*(71-index))
                self.assertEqual(campaign["scientific_completed_case_count"],0)
                self.assertEqual(campaign["scientific_worker_runs_in_campaign"],0)

    def test_candidate_total_48h_deadline_36h_slots_are_not_real_measurement(self):
        limits=self.report["unapproved_candidate_study_wide_budget_arithmetic"]
        self.assertEqual(limits["planned_cases"],72)
        self.assertEqual(limits["candidate_study_wall_seconds_limit"],48*3600)
        self.assertEqual(limits["maximum_72_case_fit_wall_slots_seconds"],36*3600)
        self.assertEqual(limits["maximum_72_case_CPU_slots_seconds"],72*3600)
        self.assertTrue(limits[
            "candidate_time_budget_is_NOT_measured_host_elapsed_time"])
        self.assertTrue(self.report[
            "candidate_budget_is_NOT_actual_real_fit_resource_bound"])

    def test_pinned_exact_original_P207H_source_only_receipt_enforced(self):
        self.assertEqual(i.H_REPORT_SHA,
                         "0f96c4c91afbd0840f50adc430f42017207b0da71e3a6126c012d50d804388db")
        self.assertEqual(i.H_JOURNAL_TAIL,
                         "964a40c45f124a1aba00bfec42603bca0136a426e1dcec282fb9ff2f3fafaf48")
        self.assertEqual(i.H_ARTIFACT_ID,14094039842)
        fake=example_report()
        self.assertTrue(i.h.validate_report(fake))
        with self.assertRaises(i.StudyStopBlocked):
            i.verify_old_h_receipt(fake)
        fake["P2_execution_authorized"]=True
        with self.assertRaises(i.StudyStopBlocked):
            i.verify_old_h_receipt(fake)

    def test_unauthorized_next_case_after_fault_rejected_even_with_new_hash(self):
        bad=copy.deepcopy(self.report["administrative_campaigns"][1])
        self.assertEqual(bad["failure_index"],0)
        # Forged "checked" at index 1, with manually reconstructed SHA.
        next_=bad["administrative_records"][1]
        next_["administrative_event"]="ADMIN_ID_CHECKED_ONLY"
        next_["record_SHA256"]=i.digest({
            key:value for key,value in next_.items() if key!="record_SHA256"})
        with self.assertRaises(i.StudyStopBlocked):
            i.validate_campaign(self.cases,bad)

    def test_truncated_reordered_duplicated_and_retried_ids_are_blocked(self):
        original=self.report["administrative_campaigns"][2]
        for alteration in ("truncated","reordered","duplicated","altered_case_id",
                           "false_completed","missing_tail"):
            with self.subTest(alteration=alteration):
                c=copy.deepcopy(original)
                rows=c["administrative_records"]
                if alteration=="truncated":
                    rows.pop()
                elif alteration=="reordered":
                    rows[2],rows[3]=rows[3],rows[2]
                elif alteration=="duplicated":
                    rows[2]=rows[1]
                elif alteration=="altered_case_id":
                    rows[0]["original_case_ID"]="UCI999-M00-B0"
                elif alteration=="false_completed":
                    rows[0]["original_scientific_fit_state"]="FIT_COMPLETED"
                else:
                    c["final_journal_SHA256"]="f"*64
                with self.assertRaises(i.StudyStopBlocked):
                    i.validate_campaign(self.cases,c)
        bad=copy.deepcopy(original)
        bad["failure_index"]=20
        with self.assertRaises(i.StudyStopBlocked):
            i.validate_campaign(self.cases,bad)

    def test_p2_scientific_GOs_source_drift_and_missing_campaigns_rejected(self):
        for key,value in (
            ("P2_execution_authorized",True),
            ("P2_execution_preflight_sha256","f"*64),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("real_PCC_fits_executed",1),
            ("synthetic_PCC_propagations_executed",1),
            ("real_UCI_dataset_rows_loaded",212),
            ("graph_or_label_masks_generated",24),
            ("original_72_case_states","SOME_COMPLETED"),
            ("administrative_failure_never_allows_next_case",False),
            ("real_graph_50k_resource_upper_bounds_verified",True),
            ("source_free_audit_JSONL_SHA256","0"*64),
            ("source_free_audit_JSONL_rows",1583),
            ("original_P2_07H_source_free_report_canonical_SHA256","a"*64),
            ("exact_locked_OCI_gzip_SHA256","b"*64),
        ):
            with self.subTest(field=key):
                malformed={**self.report,key:value}
                with self.assertRaises(i.StudyStopBlocked):
                    i.validate_report(malformed)
        truncated=copy.deepcopy(self.report)
        truncated["administrative_campaigns"].pop()
        with self.assertRaises(i.StudyStopBlocked):
            i.validate_report(truncated)

    def test_fsync_audit_canonical_SHA_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"p2-07i-no-fit-audit.jsonl"
            count=i.append_audit_journal(path,self.report)
            self.assertEqual(count,1584)
            self.assertEqual(i.verify_audit_journal(path,self.report),
                             self.report["source_free_audit_JSONL_SHA256"])
            self.assertLess(path.stat().st_size,1024**2)
            with self.assertRaises(i.StudyStopBlocked):
                i.append_audit_journal(path,self.report)
            content=path.read_bytes()
            path.write_bytes(content.replace(b"ADMIN_FAKE_STOP_NO_FIT",
                                             b"ADMIN_ID_CHECKED_ONLY",1))
            with self.assertRaises(i.StudyStopBlocked):
                i.verify_audit_journal(path,self.report)

    def test_nonstandard_fault_or_extra_attempt_and_invalid_hash_denied(self):
        self.assertEqual(len(i.FAULTS),7)
        self.assertEqual(i.LOCATIONS,(0,35,71))
        with self.assertRaises(i.StudyStopBlocked):
            i.campaign(self.cases,{
                "scenario":"REAL_FIT",
                "failure_index":0,
                "fault_reason":"RUN_PCC"})
        with self.assertRaises(i.StudyStopBlocked):
            i.build_entry(self.cases[0],"bad",0,"ACTUAL_FIT_DONE",i.GENESIS)
        with self.assertRaises(i.StudyStopBlocked):
            i.build_entry(self.cases[0],"bad",0,"ADMIN_ID_CHECKED_ONLY",
                          "not-a-sha")

    def test_cross_os_canonical_report_is_deterministic_without_time_or_random(self):
        reconstructed=i.build_report()
        self.assertEqual(i.digest(reconstructed),i.digest(self.report))
        self.assertEqual(reconstructed,self.report)
        self.assertEqual(
            hashlib.sha256(i.canonical_audit_bytes(
                reconstructed["administrative_campaigns"])).hexdigest(),
            self.report["source_free_audit_JSONL_SHA256"])
        self.assertTrue(i.validate_report(json.loads(json.dumps(reconstructed))))


if __name__=="__main__":
    unittest.main()
