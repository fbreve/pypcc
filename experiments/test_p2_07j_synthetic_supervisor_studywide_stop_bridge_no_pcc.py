"""P2-07J source-only bridge adversarial tests. No Docker/UCI/PCC calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments import p2_07j_synthetic_supervisor_studywide_stop_bridge_no_pcc as j
from experiments import p2_07h_integrated_OCI_resource_tree_supervisor_no_pcc as h
from experiments import p2_07i_studywide_failclosed_72case_no_fit as i


def fake_h_launcher(calls):
    def run(mode, path):
        calls.append((mode, path.name))
        tail = h.GENESIS
        for event in j.EVENTS:
            details = ({"exact_original_PID_starttime_identities_gone": 3}
                       if event == "ALL_PIDS_AND_CONTAINER_GONE" else {})
            tail = h.journal_append(path, tail, mode, event, **details)
        proof = {
            "mode": mode,
            "termination_reason": j.REASONS[mode],
            "real_three_initial_host_PID_starttime_identities_gone": True,
            "synthetic_PCC_fits_executed": 0,
            "real_PCC_fits_executed": 0,
            "hard_kernel_quotas": {
                "hard_swap_max_bytes": 0, "hard_pids_max": 128,
            },
            "root_exit_code": 23 if mode == "crash" else 137,
            "actual_kernel_cgroup_oom_kill_event_observed": mode == "memory",
            "actual_tmpfs_ENOSPC_observed": mode == "scratch",
            "cgroup_cpu_usage_seconds_at_last_poll": (
                h.CPU_THRESHOLD_SECONDS if mode == "cpu" else 0),
        }
        return tail, proof
    return run


class StudySupervisorStopBridgeNoPCC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)
        cls.calls = []
        cls.campaigns = [
            j.run_one(fault, index, mode, cls.path, fake_h_launcher(cls.calls))
            for fault, index, mode in j.PLAN
        ]
        cls.report = j.build_report(cls.campaigns)
        cls.audit = cls.path / "complete-admin-only-504.jsonl"
        cls.audit.write_bytes(j.audit_bytes(cls.campaigns))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_original_fault_mapping_first_middle_last(self):
        self.assertEqual(len(j.PLAN), 7)
        self.assertEqual({pos for _, pos, _ in j.PLAN}, {0,35,71})
        self.assertEqual({fault for fault,_,_ in j.PLAN}, set(i.FAULTS))
        self.assertEqual(len(j.REASONS), 5)

    def test_five_single_dispatches_zero_after_stop(self):
        self.assertEqual(len(self.calls), 5)
        self.assertEqual([x[0] for x in self.calls],
                         ["wall","cpu","memory","scratch","crash"])
        self.assertTrue(all(x["after_STOP_supervisor_dispatches"] == 0
                            for x in self.campaigns))
        self.assertEqual(self.report["actual_synthetic_supervised_three_process_trees"],5)

    def test_full_original_case_identity_order_and_no_fits(self):
        self.assertTrue(j.validate_report(self.report))
        for c in self.campaigns:
            policy = c["administrative_policy"]
            self.assertEqual(len(policy["administrative_records"]), 72)
            self.assertEqual(policy["scientific_worker_runs_in_campaign"], 0)
            self.assertEqual(policy["final_journal_SHA256"],
                             policy["administrative_records"][-1]["record_SHA256"])
            self.assertEqual({x["original_scientific_fit_state"]
                              for x in policy["administrative_records"]},
                             {"NOT_RUN_NO_APPROVAL"})

    def test_actual_later_ids_are_not_even_admin_checked(self):
        for (fault, position, _), campaign in zip(j.PLAN,self.campaigns):
            events=[row["administrative_event"] for row in
                    campaign["administrative_policy"]["administrative_records"]]
            self.assertEqual(events[:position],["ADMIN_ID_CHECKED_ONLY"]*position)
            self.assertEqual(events[position],"ADMIN_FAKE_STOP_NO_FIT")
            self.assertEqual(events[position+1:],
                             ["ADMIN_CASE_UNREACHED_AFTER_FAKE_STOP"]*(71-position))

    def test_both_prelaunch_injections_never_spawn_tree(self):
        for fault, pos, mode in j.PLAN[-2:]:
            self.assertIsNone(mode)
            campaign=next(c for (f,p,m),c in zip(j.PLAN,self.campaigns)
                          if (f,p,m)==(fault,pos,mode))
            self.assertEqual(campaign["live_supervised_artificial_tree_count"],0)
            self.assertFalse(campaign["supervisor_or_injected_stop_evidence"]
                             ["actual_synthetic_process_tree_started"])
        self.assertTrue(j.verify_disk_artifacts(self.path,self.audit,self.report))

    def test_integrated_host_journals_exact_five_expected_events(self):
        for fault, pos, mode in j.PLAN[:5]:
            path=self.path/j.ledger_name(fault,pos)
            self.assertEqual(len(path.read_bytes().splitlines()),5)
            self.assertEqual(j.validate_five_h_events(
                path,mode,h.validate_journal(path)["tail_sha256"]),
                next(c["supervisor_or_injected_stop_evidence"]
                     ["source_free_H_journal_bytes_SHA256"]
                     for (f,p,m),c in zip(j.PLAN,self.campaigns)
                     if (f,p,m)==(fault,pos,mode)))

    def test_bad_root_exit_or_missing_cleanup_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)
            for variant in ("root_exit","descendant_proof","scientific_fit","reason"):
                with self.subTest(variant=variant):
                    def bad(mode, file):
                        tail, proof=fake_h_launcher([])(mode,file)
                        if variant == "root_exit": proof["root_exit_code"]=0
                        elif variant == "descendant_proof":
                            proof["real_three_initial_host_PID_starttime_identities_gone"]=False
                        elif variant == "scientific_fit": proof["real_PCC_fits_executed"]=1
                        else: proof["termination_reason"]="NORMAL_ALL_THREE_PROCESS_EXIT_0"
                        return tail,proof
                    p=path/variant
                    p.mkdir()
                    with self.assertRaises(j.BridgeBlocked):
                        j.run_one("WALL_DEADLINE",0,"wall",p,bad)

    def test_partial_corrupt_or_reordered_H_chain_denied(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)
            tail,proof=fake_h_launcher([])("wall",path/"worker.jsonl")
            original=(path/"worker.jsonl").read_bytes().splitlines()
            for kind in ("truncated","extra","reordered","forge"):
                with self.subTest(kind=kind):
                    lines=list(original)
                    if kind=="truncated": lines.pop()
                    elif kind=="extra": lines.append(lines[-1])
                    elif kind=="reordered": lines[1],lines[2]=lines[2],lines[1]
                    else:
                        row=json.loads(lines[-1])
                        row["exact_original_PID_starttime_identities_gone"]=0
                        lines[-1]=i.canonical(row)
                    p=path/(kind+".jsonl")
                    p.write_bytes(b"\n".join(lines)+b"\n")
                    with self.assertRaises(j.BridgeBlocked):
                        j.verify_live_result("wall",p,(tail,proof))

    def test_different_case_retrial_or_bad_protocol_denied(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)
            calls=[]
            j.run_one("CPU_TIME_BUDGET",35,"cpu",path,fake_h_launcher(calls))
            with self.assertRaises(j.BridgeBlocked):
                j.run_one("CPU_TIME_BUDGET",35,"cpu",path,fake_h_launcher(calls))
            self.assertEqual(len(calls),1)
            with self.assertRaises(j.BridgeBlocked):
                j.run_one("REAL_RUN",0,"wall",path,fake_h_launcher(calls))

    def test_launcher_failure_never_retries_or_reports_success(self):
        calls=[]
        def broken(mode,path):
            calls.append(mode)
            raise RuntimeError("synthetic supervisor failed")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                j.run_one("CGROUP_MEMORY_OOM",71,"memory",Path(td),broken)
        self.assertEqual(calls,["memory"])

    def test_false_scientific_go_and_wrong_case_list_rejected(self):
        for key,value in (
            ("P2_execution_authorized",True),
            ("P2_execution_preflight_sha256","1"*64),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("real_PCC_fits_executed",1),
            ("synthetic_PCC_propagations_executed",1),
            ("studywide_administrative_case_rows",503),
            ("studywide_administrative_journal_SHA256","0"*64),
            ("original_P2_07H_source_free_report_SHA256","0"*64),
        ):
            with self.subTest(key=key):
                with self.assertRaises(j.BridgeBlocked):
                    j.validate_report({**self.report,key:value})
        bad=copy.deepcopy(self.report)
        bad["bridge_campaigns"][0]["administrative_policy"][
            "administrative_records"][1]["administrative_event"]="ADMIN_ID_CHECKED_ONLY"
        with self.assertRaises(j.BridgeBlocked):
            j.validate_report(bad)

    def test_duplicate_campaign_or_forged_live_receipt_denied(self):
        bad=copy.deepcopy(self.report)
        bad["bridge_campaigns"].append(copy.deepcopy(bad["bridge_campaigns"][-1]))
        with self.assertRaises(j.BridgeBlocked):
            j.validate_report(bad)
        bad=copy.deepcopy(self.report)
        bad["bridge_campaigns"][0]["supervisor_or_injected_stop_evidence"][
            "original_host_pid_starttime_identities_gone"]=0
        with self.assertRaises(j.BridgeBlocked):
            j.validate_report(bad)

    def test_byte_exact_persisted_504_ID_audit(self):
        self.assertTrue(j.verify_disk_artifacts(self.path,self.audit,self.report))
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/"changed.jsonl"
            content=self.audit.read_bytes()
            f.write_bytes(content[:-1])
            with self.assertRaises(j.BridgeBlocked):
                j.verify_disk_artifacts(self.path,f,self.report)


if __name__=="__main__":
    unittest.main()
