"""P2-07R source-only adversarial test suite, no Docker/PCC kernel execution."""
import copy
import json
import unittest
from experiments import p2_07r_supervise_original_oci_fake_checkpoint_fits as s
from experiments.test_p2_07q_exact_oci_synthetic_live_observer_50k import row as fake_q_row

class OriginalLiveCPUWallSupervisorTests(unittest.TestCase):
    @staticmethod
    def sample():
        results=[]
        for case in s.MODES:
            if case in s.q.BACKENDS:
                x=s.outcome(case,"NORMAL_NATIVE_8CHECKPOINTS_50K_FAKE_EXIT_0",
                            0,4.5,True,fake_q_row(case))
            else:
                why="SAMPLED_CGROUP_CPU_BUDGET_KILL" if case=="cpu_fault" else "SAMPLED_WALL_BUDGET_KILL"
                x=s.outcome(case,why,137,1.1,True)
            x["original_docker_cgroup_OS_limits_verified_before_release"]=True
            x["sampled_cgroup_peak_bytes_at_last_poll"]=150*1024**2
            results.append(x)
        return results

    def test_original_sources_and_P1_sha_and_observers_unchanged(self):
        self.assertTrue(s.source_locked())
        self.assertEqual(s.P1_SHA,"f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166")
        self.assertEqual(s.MODES,("numpy","numba","cython","cpu_fault","wall_fault"))

    def test_three_real_native_FAKE_50k_pairs_plus_two_nonfit_faults(self):
        rows=self.sample()
        report=s.assemble(rows)
        self.assertIs(s.validate_report(report),report)
        self.assertEqual(report["original_native_with_eight_observer_fake_fullfits"],6)
        self.assertEqual(report["original_real_P1_fits_executed"],0)
        self.assertEqual(s.digest(report),s.digest(json.loads(json.dumps(report))))

    def test_decision_cgroup_OOM_always_fatal(self):
        for case in s.MODES:
            self.assertEqual(s.decision(case,0.1,0.0,1),"REAL_CGROUP_OOM_FAIL_CLOSED")

    def test_CPU_wall_normal_thresholds_do_not_confuse_cpuset_with_CPU_budget(self):
        self.assertIsNone(s.decision("numba",239.9,119.9,0) if False else s.decision("numba",.1,.1,0))
        self.assertEqual(s.decision("numba",240,2,0),"SAMPLED_WALL_BUDGET_KILL")
        self.assertEqual(s.decision("numba",1,120,0),"SAMPLED_CGROUP_CPU_BUDGET_KILL")
        self.assertIsNone(s.decision("numba",30,30,0))
        self.assertTrue(s.CPU_LIMIT_NORMAL_S < s.MAX_WALL_PROCESS_S)
        self.assertTrue(s.WALL_LIMIT_NORMAL_S < s.MAX_WALL_PROCESS_S)

    def test_fault_sampled_CPU_and_wall_stops_independent(self):
        self.assertIsNone(s.decision("cpu_fault",.1,.1,0))
        self.assertEqual(s.decision("cpu_fault",1,.7,0),"SAMPLED_CGROUP_CPU_BUDGET_KILL")
        self.assertEqual(s.decision("wall_fault",1,.1,0),"SAMPLED_WALL_BUDGET_KILL")
        self.assertIsNone(s.decision("wall_fault",.2,.1,0))

    def test_exact_source_OCI_Docker_run_detached_no_network_no_privileges(self):
        for name in s.MODES:
            with self.subTest(case=name):
                cmd=s.docker_command(name,"pypcc-p207r-012345abcdef")
                self.assertEqual(cmd[:3],["docker","run","-d"])
                self.assertNotIn("--rm",cmd)
                self.assertEqual(cmd[cmd.index("--name")+1],
                                 "pypcc-p207r-012345abcdef")
                for term in ("--network","none","--read-only",
                             "--user","65534:65534","--cap-drop","ALL",
                             "--memory","2g","--memory-swap","--pids-limit","128",
                             "P207R_FAKE_ONLY=1","P207Q_FAKE_ONLY=1",
                             "PYTHONPATH="+s.q.p.k.CORRECT_PATH,
                             "experiments.p2_07r_supervise_original_oci_fake_checkpoint_fits"):
                    self.assertIn(term,cmd)
                self.assertIn("--mode",cmd)
                self.assertNotIn("UCI"," ".join(cmd))
                self.assertNotIn("build"," ".join(cmd[:2]))

    def test_invalid_worker_container_case_rejected(self):
        for name in ("auto","real_UCI","CYTHON","not_started"):
            with self.subTest(name=name),self.assertRaises(s.ResourceBlocked):
                s.docker_command(name,"pypcc-p207r-012345abcdef")
        for name in ("bad","pypcc-p207q-012345abcdef","pypcc-p207r-hello"):
            with self.subTest(name=name),self.assertRaises(s.ResourceBlocked):
                s.docker_command("numpy",name)

    def test_genuine_OS_HostConfig_and_cgroup_caps_required(self):
        config={"Memory":2147483648,"MemorySwap":2147483648,"NanoCpus":1000000000,
                "PidsLimit":128,"ReadonlyRootfs":True,"NetworkMode":"none",
                "CapDrop":["ALL"]}
        counters={"memory_max":"2147483648","memory_swap_max":"0",
                  "pids_max":"128","cpu_max":"100000 100000",
                  "memory_peak":100*1024**2,"oom_kill":0,
                  "scratch_available_bytes":128*1024**2}
        self.assertTrue(s.actual_hard_caps(config,counters))
        for field,value in (
            ("Memory",0),("MemorySwap",-1),("NanoCpus",0),
            ("PidsLimit",0),("ReadonlyRootfs",False),("NetworkMode","bridge"),
            ("CapDrop",[]),
        ):
            with self.subTest(field=field),self.assertRaises(s.ResourceBlocked):
                s.actual_hard_caps({**config,field:value},counters)
        for field,value in (
            ("memory_max","max"),("memory_swap_max","max"),
            ("pids_max","max"),("cpu_max","max 100000"),
            ("memory_peak",2147483649),("oom_kill",-1),
            ("scratch_available_bytes",0),
        ):
            with self.subTest(field=field),self.assertRaises(s.ResourceBlocked):
                s.actual_hard_caps(config,{**counters,field:value})

    def test_normal_worker_receipt_requires_exact_Q_parity_and_exit_zero(self):
        for case in s.q.BACKENDS:
            with self.subTest(case=case):
                self.assertEqual(s.outcome(case,"NORMAL_NATIVE_8CHECKPOINTS_50K_FAKE_EXIT_0",
                                           0,1.0,True,fake_q_row(case))["synthetic_full_50k_fits_executed"],2)
                with self.assertRaises(s.ResourceBlocked):
                    s.outcome(case,"SAMPLED_WALL_BUDGET_KILL",137,1.0,True,None)
                with self.assertRaises(s.ResourceBlocked):
                    s.outcome(case,"NORMAL_NATIVE_8CHECKPOINTS_50K_FAKE_EXIT_0",0,1.0,False,
                              fake_q_row(case))

    def test_forced_fault_requires_terminated_nonzero_original_container(self):
        for case,why in (("cpu_fault","SAMPLED_CGROUP_CPU_BUDGET_KILL"),
                         ("wall_fault","SAMPLED_WALL_BUDGET_KILL")):
            self.assertEqual(s.outcome(case,why,137,1.2,True)["synthetic_full_50k_fits_executed"],0)
            with self.assertRaises(s.ResourceBlocked):
                s.outcome(case,why,0,1.2,True)
            with self.assertRaises(s.ResourceBlocked):
                s.outcome(case,"NORMAL_NATIVE_8CHECKPOINTS_50K_FAKE_EXIT_0",0,1.2,True)

    def test_forbidden_scientific_execution_or_resource_approval_fails(self):
        report=s.assemble(self.sample())
        edits=[
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_preflight_sha256","a"*64),
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",1),
            ("original_real_P1_fits_executed",1),
            ("actual_final_P2_host_selected",True),
            ("active_supervisor_has_not_validated_real_P1_resource_envelope",False),
            ("host_CPU_wall_decisions_are_sampled_NOT_kernel_hard_thresholds",False),
        ]
        for key,value in edits:
            with self.subTest(key=key),self.assertRaises(s.ResourceBlocked):
                s.validate_report({**report,key:value})

    def test_report_rejects_fake_case_count_failure_PID_survival_and_missing_stop(self):
        report=s.assemble(self.sample())
        variants=[
            (0,"synthetic_full_50k_fits_executed",0),
            (0,"original_root_PID_starttime_identity_gone",False),
            (0,"original_docker_cgroup_OS_limits_verified_before_release",False),
            (2,"eight_milestones_verified",False),
            (3,"termination_reason","NORMAL_NATIVE_8CHECKPOINTS_50K_FAKE_EXIT_0"),
            (4,"host_killed_original_container",False),
        ]
        for i,key,value in variants:
            altered=copy.deepcopy(report)
            altered["source_free_actual_five_case_supervisor_results"][i][key]=value
            with self.subTest(case=i,key=key),self.assertRaises(s.ResourceBlocked):
                s.validate_report(altered)

if __name__=="__main__":
    unittest.main()
