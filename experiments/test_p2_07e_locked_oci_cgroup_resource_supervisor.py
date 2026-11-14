"""P2-07E synthetic supervisor fail-closed stdlib contract tests on both OS."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from experiments import p2_07e_locked_oci_cgroup_resource_supervisor as s


def good_report():
    rows=[]
    for mode in s.MODES:
        x={"case":mode,"event":"EXPECTED_RESULT",
           "exit_code":137 if mode!="baseline" else 0,
           "kernel_reported_OOMKilled":mode=="memory_violation",
           "termination_reason":None}
        if mode=="baseline":
            x["cgroup_v2_linux_hard_limits"]={"memory_bytes":2*1024**3}
        if mode=="wall_violation":
            x["termination_reason"]="SAMPLED_WALL_WATCHDOG_KILL"
        if mode=="cpu_violation":
            x["termination_reason"]="CGROUP_CPU_STAT_WATCHDOG_KILL"
        if mode=="scratch_violation":
            x["exit_code"]=42
        rows.append(x)
    return {
        "schema":"pypcc-P2-07E-locked-OCI-Linux-cgroup-enforcement-synthetic-probes/v1",
        "status":"FIVE_STANDALONE_CGROUP_TESTS_PASS_IMPORT_ONLY_OCI_NO_PCC",
        "P1_scientific_sha256":s.P1_SHA,
        "original_locked_OCI_gzip_sha256":s.IMAGE_SHA,
        "verified_docker_image_config_SHA256":s.IMAGE_ID,
        "draft_resource_policy_SHA256":s.POLICY_SHA,
        "Linux_cgroup_v2_only":True,
        "cgroup_synthetic_tests":rows,
        "actual_PCC_kernel_fits_executed":0,
        "UCI_datasets_loaded":0,
        "real_graphs_masks_reconstructed":0,
        "scientific_runner_present_in_OCI_image":False,
        "per_fit_real_UCI_50k_resource_upper_bound_proven":False,
        "cgroup_quota_is_not_total_CPU_seconds_cap":True,
        "polling_is_not_guaranteed_to_catch_subinterval_transients":True,
        "Linux_resources_policy_approved":False,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    }


class KernelResourceSupervisorContract(unittest.TestCase):
    def test_original_byte_locked_image_and_five_probes_are_source_free(self):
        self.assertEqual(s.IMAGE_SHA,
                         "008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8")
        self.assertEqual(s.IMAGE_ID,
                         "sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55")
        self.assertTrue(s.PROBE.is_file())
        self.assertEqual(s.MODES,("baseline","wall_violation","cpu_violation",
                                  "memory_violation","scratch_violation"))
        self.assertTrue(s.validate_report(good_report()))

    def test_container_always_has_no_network_no_privileges_read_only(self):
        for mode in s.MODES:
            with self.subTest(mode=mode):
                name="pypcc-p207e-"+mode+"-abcdef0123"
                argv=s.docker_args(mode,name)
                self.assertEqual(argv[:3],["docker","run","-d"])
                for key,value in (
                    ("--network","none"),("--user","65534:65534"),
                    ("--cap-drop","ALL"),("--read-only","--user"),
                    ("--pids-limit","128"),("--cpus","1.0"),
                ):
                    self.assertIn(key,argv)
                    if key!="--read-only":
                        self.assertEqual(argv[argv.index(key)+1],value)
                self.assertEqual(argv[-2:],
                                 ["/opt/p2-07e-probe.py",mode])
                self.assertNotIn("--privileged",argv)
                self.assertNotIn("--rm",argv)
                self.assertEqual(argv[argv.index("--memory")+1],
                                 argv[argv.index("--memory-swap")+1])
                self.assertEqual(argv[argv.index("--memory")+1],
                                 "128m" if mode=="memory_violation" else "2g")
                self.assertIn("size=1m" if mode=="scratch_violation"
                              else "size=64m",argv[argv.index("--tmpfs")+1])

    def test_reject_foreign_container_or_forbidden_test_mode(self):
        with self.assertRaises(s.ResourceProbeBlocked):
            s.docker_args("fit_predict","pypcc-p207e-baseline-abcdef0123")
        with self.assertRaises(s.ResourceProbeBlocked):
            s.docker_args("baseline","production-container")
        with self.assertRaises(s.ResourceProbeBlocked):
            s.docker_args("baseline","pypcc-p207e-baseline-something-invalid")

    def test_forged_scientific_go_real_fits_and_missing_caps_rejected(self):
        original=good_report()
        changes=[
            ("P2_execution_preflight_sha256","a"*64),
            ("P2_execution_authorized",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("Linux_resources_policy_approved",True),
            ("real_PCC_fits_executed",1),
            ("actual_PCC_kernel_fits_executed",1),
            ("UCI_datasets_loaded",1),
            ("real_graphs_masks_reconstructed",3),
            ("scientific_runner_present_in_OCI_image",True),
            ("per_fit_real_UCI_50k_resource_upper_bound_proven",True),
            ("cgroup_quota_is_not_total_CPU_seconds_cap",False),
            ("original_locked_OCI_gzip_sha256","f"*64),
            ("verified_docker_image_config_SHA256","sha256:"+"b"*64),
        ]
        for key,value in changes:
            with self.subTest(key=key):
                bad={**original,key:value}
                with self.assertRaises(s.ResourceProbeBlocked):
                    s.validate_report(bad)

    def test_missing_forced_oom_cpu_or_tmpfs_kills_cannot_pass(self):
        for mode,field,value in (
            ("memory_violation","kernel_reported_OOMKilled",False),
            ("wall_violation","termination_reason",None),
            ("cpu_violation","termination_reason","EXPIRED_WALL"),
            ("scratch_violation","exit_code",0),
            ("baseline","cgroup_v2_linux_hard_limits",
             {"memory_bytes":128*1024**2}),
        ):
            with self.subTest(mode=mode):
                report=good_report()
                target=next(x for x in report["cgroup_synthetic_tests"]
                            if x["case"]==mode)
                target[field]=value
                with self.assertRaises(s.ResourceProbeBlocked):
                    s.validate_report(report)
        report=good_report()
        report["cgroup_synthetic_tests"].pop()
        with self.assertRaises(s.ResourceProbeBlocked):
            s.validate_report(report)

    def test_append_ledger_refuses_arbitrary_data_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            dest=Path(name)/"synthetic-only-ledger.jsonl"
            s.ledger_write(dest,{"case":"baseline","event":"STARTED"})
            s.ledger_write(dest,{"case":"baseline","event":"EXPECTED_RESULT",
                                 "exit_code":0})
            rows=[json.loads(x) for x in dest.read_text().splitlines()]
            self.assertEqual([x["event"] for x in rows],
                             ["STARTED","EXPECTED_RESULT"])
            with self.assertRaises(s.ResourceProbeBlocked):
                s.ledger_write(dest,{"case":"uci-212-real-fit","event":"STARTED"})
            with self.assertRaises(s.ResourceProbeBlocked):
                s.ledger_write(dest,{"case":"baseline","event":"P2_APPROVED"})
            self.assertEqual(len(dest.read_text().splitlines()),2)


if __name__=="__main__":
    unittest.main()
