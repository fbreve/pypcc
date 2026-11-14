"""P2-07H adversarial source-only cross-OS tests; no Docker or PCC fits."""
import copy
from pathlib import Path
import tempfile
import unittest
from experiments import p2_07h_integrated_OCI_resource_tree_supervisor_no_pcc as h

def example_report():
    cases=[]
    for mode in h.MODES:
        limits={
            "hard_memory_max_bytes":128*1024**2 if mode=="memory" else 2*1024**3,
            "hard_swap_max_bytes":0,
            "hard_pids_max":128,
            "cpu_quota_cores_at_most":1,
            "hard_tmpfs_quota_bytes":1024**2 if mode=="scratch" else 64*1024**2,
        }
        reasons={
            "normal":"NORMAL_ALL_THREE_PROCESS_EXIT_0",
            "wall":"HOST_SAMPLED_WALL_THRESHOLD_STOP",
            "cpu":"HOST_SAMPLED_CGROUP_CPU_THRESHOLD_STOP",
            "memory":"KERNEL_OOM_EVENT_DESCENDANT_FORCED_FULL_TREE_STOP",
            "scratch":"KERNEL_TMPFS_ENOSPC_FORCED_FULL_TREE_STOP",
            "crash":"ROOT_EXIT_23_AND_CONTAINER_KILLED_DESCENDANTS",
        }
        cases.append({
            "mode":mode,"termination_reason":reasons[mode],
            "root_exit_code":0 if mode=="normal" else 23 if mode=="crash" else 137,
            "real_three_initial_host_PID_starttime_identities_gone":True,
            "actual_kernel_cgroup_oom_kill_event_observed":mode=="memory",
            "actual_tmpfs_ENOSPC_observed":mode=="scratch",
            "cgroup_cpu_usage_seconds_at_last_poll":.8,
            "cgroup_memory_peak_bytes_at_last_poll":128*1024**2,
            "hard_kernel_quotas":limits,
            "synthetic_PCC_fits_executed":0,
            "real_PCC_fits_executed":0,
        })
    return {
        "schema":"pypcc-P2-07H-unified-source-free-three-process-resource-supervisor/v1",
        "status":"SIX_INTEGRATED_KERNEL_AND_PARENT_PROBES_PASS_ZERO_PCC",
        "P1_scientific_sha256":h.P1_SHA,
        "exact_OCI_binary_gzip_sha256":h.IMAGE_SHA,
        "exact_OCI_config_id":h.IMAGE_ID,
        "source_72case_P2_07F_canonical_sha256":h.F_REPORT_SHA,
        "prior_P2_07G_actual_descendants_verified":True,
        "synthetic_three_process_integrated_cases":cases,
        "bounded_original_PID_identities_confirmed_gone":18,
        "journal_records":30,
        "journal_final_sha256":"f"*64,
        "frozen_P1_all_72_scientific_cases_NOT_RUN":True,
        "P2_07H_is_not_a_scientific_worker_execution_supervisor":True,
        "real_graph_50k_resource_upper_bounds_known":False,
        "full_host_OS_compiler_and_native_closure_complete":False,
        "host_cpu_watchdog_sampling_is_a_hard_limit":False,
        "per_fit_real_PCC_runner_included":False,
        "real_UCI_source_rows_loaded":0,
        "graphs_or_masks_built":0,
        "synthetic_PCC_fits_executed":0,
        "real_PCC_fits_executed":0,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
    }

class UnifiedSourceOnlyContract(unittest.TestCase):
    def test_six_modes_have_zero_scientific_fit_or_real_data(self):
        self.assertEqual(h.MODES,
                         ("normal","wall","cpu","memory","scratch","crash"))
        self.assertTrue(h.FIXTURE.is_file())
        self.assertTrue(h.validate_report(example_report()))
        self.assertFalse(example_report()["P2_execution_authorized"])

    def test_exact_OCI_config_SHA_and_immutability(self):
        self.assertEqual(h.IMAGE_SHA,
                         "008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8")
        self.assertEqual(h.IMAGE_ID,
                         "sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55")
        self.assertEqual(h.F_REPORT_SHA,
                         "d6c7fedc5abb3bf48395a3b223bb1112c11ba6a21593075c55e4a02afabd1328")

    def test_all_synthetic_tree_modes_require_cgroup_limits_and_no_network(self):
        for mode in h.MODES:
            with self.subTest(mode=mode):
                argv=h.docker_args(
                    mode,f"pypcc-p207h-{mode}-1234567890")
                self.assertEqual(argv[:3],["docker","run","-d"])
                self.assertEqual(argv[-4:],
                                 [h.e.IMAGE_TAG,"/opt/p207h-fixture.py","root",mode])
                self.assertIn("--read-only",argv)
                for key,value in (("--network","none"),("--user","65534:65534"),
                                  ("--cap-drop","ALL"),("--pids-limit","128"),
                                  ("--cpus","1.0")):
                    self.assertEqual(argv[argv.index(key)+1],value)
                limit="128m" if mode=="memory" else "2g"
                self.assertEqual(argv[argv.index("--memory")+1],limit)
                self.assertEqual(argv[argv.index("--memory-swap")+1],limit)
                tmpfs=argv[argv.index("--tmpfs")+1]
                self.assertIn("size=1m" if mode=="scratch" else "size=64m",tmpfs)
                self.assertNotIn("--privileged",argv)

    def test_invalid_case_or_container_cannot_be_executed(self):
        with self.assertRaises(h.UnifiedBlocked):
            h.docker_args("run_PCC","pypcc-p207h-normal-1234567890")
        with self.assertRaises(h.UnifiedBlocked):
            h.docker_args("normal","pypcc-real-UCI212")
        with self.assertRaises(h.UnifiedBlocked):
            h.docker_args("normal","pypcc-p207h-normal-nonhex")

    def test_cgroup_counter_probe_is_valid_python_before_any_docker_exec(self):
        source=h.cgroup_probe_source()
        compile(source,"<P2-07H-cgroup-v2-metadata-probe>","exec")
        self.assertIn("memory.events",source)
        self.assertIn("cpu.stat",source)
        self.assertIn("memory.swap.max",source)
        self.assertIn("sort_keys=True",source)
        self.assertNotIn("PCC",source)
        self.assertNotIn("UCI",source)

    def test_actual_cgroup_resource_limits_reject_fake_memory_swap_cpu(self):
        expected={
            "cpu_usage_usec":123456,"oom_kill":0,
            "memory_peak":40000000,
            "memory_max":"2147483648",
            "memory_swap_max":"0","pids_max":"128",
            "cpu_max":"100000 100000",
            "scratch_available_bytes":64*1024**2,
        }
        caps=h.check_caps(expected,"wall")
        self.assertEqual(caps["hard_memory_max_bytes"],2*1024**3)
        for k,bad in (("memory_max","max"),
                      ("memory_swap_max","2147483648"),
                      ("pids_max","max"),
                      ("cpu_max","200000 100000"),
                      ("scratch_available_bytes",1024**3)):
            with self.subTest(field=k):
                with self.assertRaises(h.UnifiedBlocked):
                    h.check_caps({**expected,k:bad},"wall")
        with self.assertRaises(h.UnifiedBlocked):
            h.check_caps(expected,"memory")

    def test_forged_scientific_approval_or_missing_resource_proof_fails(self):
        original=example_report()
        for field,wrong in (
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_authorized",True),
            ("P2_execution_preflight_sha256","0"*64),
            ("real_PCC_fits_executed",72),
            ("synthetic_PCC_fits_executed",1),
            ("real_UCI_source_rows_loaded",212),
            ("graphs_or_masks_built",24),
            ("host_cpu_watchdog_sampling_is_a_hard_limit",True),
            ("real_graph_50k_resource_upper_bounds_known",True),
            ("frozen_P1_all_72_scientific_cases_NOT_RUN",False),
            ("source_72case_P2_07F_canonical_sha256","f"*64),
            ("exact_OCI_binary_gzip_sha256","a"*64),
        ):
            with self.subTest(field=field):
                with self.assertRaises(h.UnifiedBlocked):
                    h.validate_report({**original,field:wrong})

    def test_missing_oom_enospc_or_descendant_cleanup_disallows_pass(self):
        for index,field,bad in (
            (0,"real_three_initial_host_PID_starttime_identities_gone",False),
            (1,"termination_reason","SUCCESS"),
            (2,"termination_reason","UNPROVEN_CPU"),
            (3,"actual_kernel_cgroup_oom_kill_event_observed",False),
            (4,"actual_tmpfs_ENOSPC_observed",False),
            (5,"root_exit_code",0),
        ):
            with self.subTest(mode=h.MODES[index]):
                report=example_report()
                report["synthetic_three_process_integrated_cases"][index][field]=bad
                with self.assertRaises(h.UnifiedBlocked):
                    h.validate_report(report)

    def test_integrity_chain_is_fail_closed_and_fsync_truncated_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"bounded-no-PCC-journal.jsonl"
            prev=h.GENESIS
            events=("STARTED","TREE_AND_HARD_CAPS_VERIFIED",
                    "RELEASED_SYNTHETIC_BARRIER",
                    "TERMINATION_DECISION_RECORDED",
                    "ALL_PIDS_AND_CONTAINER_GONE")
            for mode in h.MODES:
                for event in events:
                    prev=h.journal_append(p,prev,mode,event)
            j=h.validate_journal(p,complete=True)
            self.assertEqual(j["records"],30)
            self.assertEqual(j["tail_sha256"],prev)
            with self.assertRaises(h.UnifiedBlocked):
                h.journal_append(p,prev,"UCI212","STARTED")
            p.write_bytes(p.read_bytes().replace(
                b'"event":"STARTED"',b'"event":"PCC_FIT"',1))
            with self.assertRaises(h.UnifiedBlocked):
                h.validate_journal(p,complete=True)

    def test_case_failure_cannot_be_mistaken_for_complete_journal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"failed-source-free.jsonl"
            previous=h.journal_append(p,h.GENESIS,"normal","STARTED")
            h.journal_append(p,previous,"normal","FAIL_CLOSED")
            with self.assertRaises(h.UnifiedBlocked):
                h.validate_journal(p,complete=True)
            with self.assertRaises(h.UnifiedBlocked):
                h.journal_append(p,"BAD","normal","STARTED")


if __name__=="__main__":
    unittest.main()
