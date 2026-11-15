"""P2-07P strict SOURCE-ONLY 50k artificial original-OCI backend/resource checks.

No NumPy install, Docker, PCC fit, original official data/masks, or downloads
in this unittest module. The real artificial 50k runs occur only in the
separate explicitly opt-in Github Actions original OCI worker job.
"""
import copy
import json
import unittest

from experiments import p2_07p_exact_oci_synthetic_fullfit_resource_probe as p

def fake_row(name):
    return {
        "schema":p.WORKER,"backend":name,
        "data_kind":"DETERMINISTIC_ARTIFICIAL_ONLY_NO_UCI",
        "image_config_id":p.IMAGE_ID,"P1_scientific_sha256":p.P1_SHA,
        "synthetic_nodes":900,"synthetic_features":7,"synthetic_classes":2,
        "synthetic_known_labels":6,"synthetic_k_nn":10,
        "synthetic_graph_shape":[900,18],
        "synthetic_graph_sha256":"f"*64,
        "synthetic_full_fit_requested_iterations":50000,
        "synthetic_early_stop_enabled":False,
        "synthetic_original_seed_separate_from_P1":71,
        "synthetic_50k_full_fits_completed":1,
        "real_UCI_PCC_fits_completed":0,
        "original_checkpoint_observer_invocations":0,
        "selected_function_identity_verified":True,
        "synthetic_original_graph_builder_invoked":True,
        "graph_wall_seconds":.5,
        "graph_cpu_seconds":.3,
        "fit_wall_seconds":37.0,
        "fit_cpu_seconds":35.0,
        "process_peak_rss_bytes":255*1024**2,
        "cgroup_v2_peak_bytes":300*1024**2,
        "tmpfs_file_payload_bytes_end":2000,
        "candidate_2GiB_memory_limit_enforced":True,
        "synthetic_5min_failclosed_wall_limit_enforced":True,
        "actual_P1_resource_upper_bounds_proven":False,
        "P2_execution_authorized":False,
    }

def fake_report():
    return p.assemble([fake_row(name) for name in p.BACKENDS])

class OriginalOCI50kSyntheticFullfitHardSafety(unittest.TestCase):
    def test_original_scientific_sha_and_original_source_unchanged(self):
        self.assertTrue(p.original_P1_locked_NO_GO())
        self.assertEqual(p.ITERATIONS,50000)
        self.assertEqual(p.BACKENDS,("numpy","numba","cython"))

    def test_synthetic_fixture_only_and_no_original_P1_mask_or_seed(self):
        self.assertEqual((p.NODES,p.FEATURES,p.CLASSES,p.KNOWN_LABELS),
                         (900,7,2,6))
        self.assertEqual(p.FAKE_SEED,71)
        self.assertEqual(p.MAX_MEMORY_BYTES,2*1024**3)
        self.assertEqual(p.MAX_SYNTHETIC_WALL_SECONDS,300)

    def test_valid_native_backend_rows_must_be_explicitly_complete(self):
        for backend in p.BACKENDS:
            with self.subTest(backend=backend):
                self.assertTrue(p.validate_worker(fake_row(backend),backend))

    def test_invalid_real_or_observed_scientific_fits_fail_closed(self):
        for field,value in (
            ("data_kind","ORIGINAL_UCI"),
            ("synthetic_50k_full_fits_completed",72),
            ("real_UCI_PCC_fits_completed",1),
            ("original_checkpoint_observer_invocations",1),
            ("actual_P1_resource_upper_bounds_proven",True),
            ("P2_execution_authorized",True),
            ("synthetic_full_fit_requested_iterations",100),
            ("synthetic_early_stop_enabled",True),
            ("synthetic_original_seed_separate_from_P1",123),
            ("selected_function_identity_verified",False),
            ("synthetic_original_graph_builder_invoked",False),
        ):
            row=fake_row("cython")
            row[field]=value
            with self.subTest(field=field),self.assertRaises(p.SyntheticFullfitBlocked):
                p.validate_worker(row,"cython")

    def test_invalid_memory_time_and_scratch_bounds_fail_closed(self):
        for field,value in (
            ("process_peak_rss_bytes",0),
            ("process_peak_rss_bytes",p.MAX_MEMORY_BYTES+1),
            ("cgroup_v2_peak_bytes",p.MAX_MEMORY_BYTES+1),
            ("graph_wall_seconds",0),
            ("fit_wall_seconds",p.MAX_SYNTHETIC_WALL_SECONDS),
            ("fit_cpu_seconds",-1),
            ("tmpfs_file_payload_bytes_end",p.TMPFS_MAX+1),
            ("candidate_2GiB_memory_limit_enforced",False),
            ("synthetic_5min_failclosed_wall_limit_enforced",False),
        ):
            with self.subTest(field=field):
                row=fake_row("numba")
                row[field]=value
                with self.assertRaises(p.SyntheticFullfitBlocked):
                    p.validate_worker(row,"numba")

    def test_graph_must_be_identical_for_all_three_original_backends(self):
        rows=[fake_row(name) for name in p.BACKENDS]
        rows[1]["synthetic_graph_sha256"]="b"*64
        with self.assertRaises(p.SyntheticFullfitBlocked):
            p.assemble(rows)
        rows=[fake_row(name) for name in p.BACKENDS]
        rows[2]["synthetic_graph_shape"]=[900,20]
        with self.assertRaises(p.SyntheticFullfitBlocked):
            p.assemble(rows)

    def test_backend_reordering_aliases_or_incomplete_experiment_rejected(self):
        for rows in (
            [fake_row("cython"),fake_row("numba"),fake_row("numpy")],
            [fake_row("numpy"),fake_row("numba")],
            [fake_row("numpy"),fake_row("numba"),fake_row("numba")],
        ):
            with self.subTest(rows=len(rows)),self.assertRaises(p.SyntheticFullfitBlocked):
                p.assemble(rows)
        with self.assertRaises(p.SyntheticFullfitBlocked):
            p.validate_worker(fake_row("numba"),"cython")

    def test_source_free_report_exact_NO_GO(self):
        r=fake_report()
        self.assertTrue(p.validate_report(r))
        self.assertEqual(p.digest(r),p.digest(json.loads(json.dumps(r))))
        self.assertEqual(r["synthetic_50k_full_fits_executed"],3)
        self.assertEqual(r["real_UCI_PCC_fits_executed"],0)
        self.assertIsNone(r["P2_execution_preflight_sha256"])

    def test_false_approval_or_real_resource_bound_never_passes(self):
        report=fake_report()
        for field,value in (
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_preflight_sha256","a"*64),
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",1),
            ("actual_P1_real_knn_graph_used",True),
            ("original_source_UCI_fetched_or_transformed",True),
            ("source_real_labels_masks_or_P1_seeds_created",True),
            ("real_full_fit_instrumented_50k_resource_upper_bound_proven",True),
            ("proposed_2GiB_container_is_NOT_approved_scientific_envelope",False),
        ):
            with self.subTest(field=field),self.assertRaises(p.SyntheticFullfitBlocked):
                p.validate_report({**report,field:value})

    def test_command_is_fixed_original_OCI_nonroot_offline_with_cython_path(self):
        cmd=p.docker_args("cython","pypcc-p207p-012345abcdef")
        for flag in ("--read-only","--network","none","--user","65534:65534",
                     "--memory","2g","--memory-swap","--cap-drop","ALL",
                     "--tmpfs","P207P_SYNTHETIC_ONLY=1",
                     "PYTHONPATH="+p.k.CORRECT_PATH,
                     "--entrypoint",p.PYTHON_IN_OCI):
            self.assertIn(flag,cmd)
        self.assertEqual(cmd[-1],"cython")
        self.assertNotIn("UCI", " ".join(cmd))
        self.assertNotIn("fit_predict"," ".join(cmd))
        self.assertNotIn("docker pull"," ".join(cmd))
        for name in ("bad","pypcc-p207m-012345abcdef",
                     "pypcc-p207p-012345678901;rm"):
            with self.subTest(name=name),self.assertRaises(p.SyntheticFullfitBlocked):
                p.docker_args("numpy",name)

    def test_arbitrary_backend_and_non_worker_inputs_rejected(self):
        for name in ("auto","NUMBA","cython_sequential","original_UCI_fit"):
            with self.subTest(name=name),self.assertRaises(p.SyntheticFullfitBlocked):
                p.docker_args(name,"pypcc-p207p-012345abcdef")
        with self.assertRaises(p.SyntheticFullfitBlocked):
            p.validate_worker(fake_row("numpy"),"bad")

if __name__=="__main__":
    unittest.main()
