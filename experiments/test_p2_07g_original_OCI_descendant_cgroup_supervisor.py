"""P2-07G stdlib source-only adversarial no-PCC tests for Windows/Linux."""
import copy
from pathlib import Path
import tempfile
import unittest

from experiments import p2_07g_original_OCI_descendant_cgroup_supervisor as s


def fake_result():
    return {
        "schema":"pypcc-P2-07G-exact-OCI-descendant-termination-no-PCC/v1",
        "status":"FOUR_SYNTHETIC_CONTAINER_PROCESS_TREES_CLEANED_NO_PCC",
        "P1_scientific_sha256":s.P1_SHA,
        "original_OCI_gzip_sha256":s.IMAGE_SHA,
        "original_OCI_image_config_id":s.IMAGE_ID,
        "original_P2_07F_admin_report_sha256":s.F_REPORT_SHA,
        "cases":[
            {"mode":mode,"original_processes_observed":3,
             "root_exit_code":23 if mode=="synthetic_crash" else 137,
             "all_original_host_PID_starttime_identities_gone":True,
             "all_original_processes_in_same_cgroup":True,
             "extra_scientific_fits":0}
            for mode in s.MODES
        ],
        "journal_records":20,
        "journal_tail_sha256":"a"*64,
        "host_PID_reuse_checked_by_proc_starttime":True,
        "every_original_root_child_grandchild_verified_gone":True,
        "all_possible_future_process_escapes_proven_absent":False,
        "final_scientific_runner_present":False,
        "approved_real_graph_50k_resource_bounds":False,
        "real_UCI_source_rows_loaded":0,
        "graphs_or_masks_created":0,
        "synthetic_PCC_fits":0,
        "real_PCC_fits_executed":0,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
    }


class P207GSourceOnlyTests(unittest.TestCase):
    def test_four_actual_process_tree_proofs_required_but_no_scientific_go(self):
        result=fake_result()
        self.assertTrue(s.validate_result(result))
        self.assertEqual(s.MODES,("normal_tree","detached_tree",
                                  "stubborn_tree","synthetic_crash"))
        self.assertEqual(s.IMAGE_SHA,
                         "008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8")
        self.assertFalse(result["P2_execution_authorized"])

    def test_namespaced_original_container_cmd_no_network_no_caps_and_exact_image(self):
        for mode in s.MODES:
            with self.subTest(mode=mode):
                arg=s.arguments(mode,f"pypcc-p207g-{mode}-a012345678")
                self.assertEqual(arg[:3],["docker","run","-d"])
                for key,value in (
                    ("--network","none"),("--user","65534:65534"),
                    ("--cap-drop","ALL"),("--pids-limit","128"),
                    ("--memory","2g"),("--memory-swap","2g"),("--cpus","1.0"),
                ):
                    self.assertEqual(arg[arg.index(key)+1],value)
                self.assertIn("--read-only",arg)
                self.assertEqual(arg[-3:],
                                 [s.e.IMAGE_TAG,"/opt/p207g-probe.py",mode])
                self.assertNotIn("--privileged",arg)
                self.assertIn("size=64m",arg[arg.index("--tmpfs")+1])

    def test_reject_unexpected_name_or_real_fit_request(self):
        with self.assertRaises(s.TreeBlocked):
            s.arguments("fit_predict","pypcc-p207g-normal_tree-a012345678")
        with self.assertRaises(s.TreeBlocked):
            s.arguments("normal_tree","pcc-real-production")
        with self.assertRaises(s.TreeBlocked):
            s.arguments("normal_tree","pypcc-p207g-normal_tree-nothex")
        self.assertTrue(s.FIXTURE.is_file())

    def test_docker_top_three_processes_parsed_with_host_pid_identity(self):
        text=("PID PPID COMMAND\n"
              "4010 4000 python3.12\n"
              "4011 4010 python3.12\n"
              "4012 4011 python3.12\n")
        self.assertEqual(s.parse_top(text),[4010,4011,4012])
        for wrong in (
            "PID PPID COMMAND\n4010 1 python3.12\n",
            text+"4013 4012 python3.12\n",
            text.replace("4012 4011","4011 4011"),
            text.replace("python3.12","pcc_kernel"),
        ):
            with self.subTest(text=wrong[:45]):
                with self.assertRaises(s.TreeBlocked):
                    s.parse_top(wrong)

    def test_no_pid_reuse_ambiguity_and_proc_missing_identity(self):
        with self.assertRaises(s.TreeBlocked):
            s.proc_identity(-1)
        with self.assertRaises(s.TreeBlocked):
            s.proc_identity(1)
        self.assertIsNone(s.proc_identity(2**31-1))
        with self.assertRaises(s.TreeBlocked):
            s.ensure_all_gone([])

    def test_fail_closed_on_any_forged_P2_go_or_survivor(self):
        initial=fake_result()
        changes=[
            ("P2_execution_authorized",True),
            ("P2_execution_preflight_sha256","a"*64),
            ("real_PCC_fits_executed",1),
            ("synthetic_PCC_fits",1),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("real_UCI_source_rows_loaded",310),
            ("graphs_or_masks_created",24),
            ("all_possible_future_process_escapes_proven_absent",True),
            ("every_original_root_child_grandchild_verified_gone",False),
            ("original_OCI_gzip_sha256","a"*64),
            ("original_P2_07F_admin_report_sha256","b"*64),
        ]
        for field,value in changes:
            with self.subTest(field=field):
                with self.assertRaises(s.TreeBlocked):
                    s.validate_result({**initial,field:value})

    def test_reject_one_surviving_descendant_and_missing_case_or_crash(self):
        original=fake_result()
        for index,key,wrong in (
            (0,"all_original_host_PID_starttime_identities_gone",False),
            (1,"all_original_processes_in_same_cgroup",False),
            (2,"original_processes_observed",2),
            (3,"root_exit_code",0),
        ):
            with self.subTest(index=index):
                report=copy.deepcopy(original)
                report["cases"][index][key]=wrong
                with self.assertRaises(s.TreeBlocked):
                    s.validate_result(report)
        truncated=copy.deepcopy(original)
        truncated["cases"].pop()
        with self.assertRaises(s.TreeBlocked):
            s.validate_result(truncated)

    def test_append_only_fsync_chain_enforces_exact_five_events_each_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"no-fit-events.jsonl"
            prev=s.GENESIS
            expected=("STARTED","THREE_DESCENDANTS_SAME_CGROUP",
                      "TERMINATION_REQUESTED","ROOT_EXIT_NONZERO",
                      "ALL_ORIGINAL_PROCESS_IDENTITIES_GONE")
            for mode in s.MODES:
                for event in expected:
                    prev=s.append(file,prev,mode,event)
            chain=s.validate_journal(file,completed=True)
            self.assertEqual(chain["count"],20)
            self.assertEqual(chain["tail_sha256"],prev)
            with self.assertRaises(s.TreeBlocked):
                s.append(file,prev,"uci850-real","STARTED")
            file.write_bytes(file.read_bytes().replace(
                b'"event":"STARTED"',b'"event":"PCC_FIT"',1))
            with self.assertRaises(s.TreeBlocked):
                s.validate_journal(file)

    def test_journal_truncation_duplicate_false_fit_event_and_missing_tail_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"unfinished.jsonl"
            last=s.append(file,s.GENESIS,"normal_tree","STARTED")
            with self.assertRaises(s.TreeBlocked):
                s.validate_journal(file,completed=True)
            with self.assertRaises(s.TreeBlocked):
                s.append(file,last,"normal_tree","SCIENTIFIC_FIT_APPROVED")
            with self.assertRaises(s.TreeBlocked):
                s.append(file,"bad-digest","normal_tree","FAIL_CLOSED")


if __name__=="__main__":
    unittest.main()
