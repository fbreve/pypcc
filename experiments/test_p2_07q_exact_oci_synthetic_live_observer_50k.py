"""P2-07Q: strict source-only fixtures for original 8-milestone synthetic OCI parity."""
import copy
import json
import unittest

from experiments import p2_07q_exact_oci_synthetic_live_observer_50k as q

def row(backend):
    return {
        "schema":q.WORKER,"backend":backend,
        "data_kind":"DETERMINISTIC_ARTIFICIAL_ONLY_NO_UCI",
        "original_Cython_native_sha256":
            q.p.k.lock.NEW_NATIVE["production_pcc_step"],
        "original_isolated_Cython_observer_native_sha256":
            q.p.k.lock.NEW_NATIVE["isolated_pcc_step_live_research"],
        "P1_scientific_sha256":q.P1_SHA,"exact_original_OCI_config_id":q.OCI_ID,
        "synthetic_nodes":900,"synthetic_classes":2,"synthetic_fake_label_count":6,
        "original_native_fullfit_and_historical_observer":[
            {"mode":mode,"graph_shape":[900,20],
             "synthetic_graph_sha256":"a"*64,
             "graph_cpu_seconds":0.2,"graph_wall_seconds":0.3,
             "fit_cpu_seconds":4.0,"fit_wall_seconds":4.1,
             "snapshot_array_bytes":500_000 if i else 0,
             "milestones":list(q.MILESTONES) if i else [],
             "internal_effective_iterations":50000,
             "PCC_fits_synthetic":1,"PCC_fits_real":0}
            for i,mode in enumerate(("original","observed"))
        ],
        "same_backend_full_mutable_state_parity_verified":True,
        "same_backend_prediction_parity_verified":True,
        "numpy_or_numba_next_rng_parity_verified":backend!="cython",
        "cython_baseline_native_rng_exposed_and_compared":False,
        "exact_eight_observations_captured_in_full_propagation":True,
        "actual_synthetic_fits_completed":2,
        "real_original_P1_fits_completed":0,
        "synthetic_pair_process_peak_RSS_bytes":320*1024**2,
        "synthetic_pair_cgroup_v2_peak_bytes":190*1024**2,
        "synthetic_pair_tmpfs_file_payload_bytes":1000,
        "real_P1_50k_checkpoint_resource_upper_bounds_proven":False,
        "P2_execution_authorized":False
    }

def report():
    return q.assemble([row(name) for name in q.BACKENDS])

class OriginalOCIEightCheckpointNoRealScience(unittest.TestCase):
    def test_hist_observers_and_helper_original_Git_blobs_locked(self):
        self.assertTrue(q.source_locked())
        self.assertEqual(len(q.OBSERVER_BLOBS),5)
        self.assertEqual(len(q.MILESTONES),8)

    def test_original_P1_50k_and_fake_900n_locked(self):
        self.assertEqual(q.P1_SHA,"f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166")
        self.assertEqual(q.MILESTONES,(0,1,10,100,1000,5000,25000,50000))
        self.assertEqual((q.p.NODES,q.p.CLASSES,q.p.KNOWN_LABELS),(900,2,6))
        self.assertEqual(q.MAX_SNAPSHOT_BYTES,64*1024**2)

    def test_three_complete_backend_worker_receipts_valid(self):
        for name in q.BACKENDS:
            with self.subTest(name=name):
                self.assertTrue(q.validate_worker(row(name),name))

    def test_full_review_only_report_has_persistent_P2_NO_GO(self):
        r=report()
        self.assertTrue(q.validate_report(r))
        self.assertEqual(r["synthetic_50k_complete_original_and_observed_fits_executed"],6)
        self.assertEqual(r["real_original_P1_PCC_fits_executed"],0)
        self.assertEqual(r["P2_execution_preflight_sha256"],None)
        self.assertEqual(q.digest(r),q.digest(json.loads(json.dumps(r))))

    def test_eight_actual_observation_indices_required(self):
        for mutation in (
            [0,1,10,100,1000,5000,25000],
            [0,1,10,100,1000,5000,25000,49999],
            list(reversed(q.MILESTONES)),
            list(q.MILESTONES)+[50000],
        ):
            current=row("cython")
            current["original_native_fullfit_and_historical_observer"][1]["milestones"]=mutation
            with self.subTest(mutation=mutation),self.assertRaises(q.ObserverBlocked):
                q.validate_worker(current,"cython")

    def test_real_fit_or_early_stop_resource_completion_forgery_rejected(self):
        for key,value in (
            ("real_original_P1_fits_completed",1),
            ("same_backend_full_mutable_state_parity_verified",False),
            ("same_backend_prediction_parity_verified",False),
            ("exact_eight_observations_captured_in_full_propagation",False),
            ("actual_synthetic_fits_completed",72),
            ("real_P1_50k_checkpoint_resource_upper_bounds_proven",True),
            ("P2_execution_authorized",True),
            ("synthetic_pair_process_peak_RSS_bytes",q.p.MAX_MEMORY_BYTES+1),
            ("synthetic_pair_cgroup_v2_peak_bytes",q.p.MAX_MEMORY_BYTES+1),
            ("synthetic_pair_tmpfs_file_payload_bytes",q.p.TMPFS_MAX+1),
        ):
            current=row("numpy")
            current[key]=value
            with self.subTest(key=key),self.assertRaises(q.ObserverBlocked):
                q.validate_worker(current,"numpy")

    def test_cython_rng_is_never_falsely_claimed_verified(self):
        current=row("cython")
        current["cython_baseline_native_rng_exposed_and_compared"]=True
        with self.assertRaises(q.ObserverBlocked):
            q.validate_worker(current,"cython")
        current=row("cython")
        current["numpy_or_numba_next_rng_parity_verified"]=True
        with self.assertRaises(q.ObserverBlocked):
            q.validate_worker(current,"cython")

    def test_numpy_numba_next_rng_verification_mandatory(self):
        for name in ("numpy","numba"):
            current=row(name)
            current["numpy_or_numba_next_rng_parity_verified"]=False
            with self.subTest(name=name),self.assertRaises(q.ObserverBlocked):
                q.validate_worker(current,name)

    def test_timing_snapshot_and_case_shape_bounds_fail_closed(self):
        for key,value,index in (
            ("internal_effective_iterations",100,1),
            ("PCC_fits_synthetic",2,0),
            ("PCC_fits_real",1,1),
            ("fit_wall_seconds",q.MAX_SYNTHETIC_PAIR_WALL_SECONDS,1),
            ("graph_cpu_seconds",-1,0),
            ("snapshot_array_bytes",q.MAX_SNAPSHOT_BYTES+1,1),
            ("snapshot_array_bytes",10,0),
            ("graph_shape",[900,901],0),
            ("synthetic_graph_sha256","invalid",1),
        ):
            current=row("numpy")
            current["original_native_fullfit_and_historical_observer"][index][key]=value
            with self.subTest(key=key,index=index),self.assertRaises(q.ObserverBlocked):
                q.validate_worker(current,"numpy")

    def test_no_science_GO_report_claims_permitted(self):
        base=report()
        for key,value in (
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_preflight_sha256","f"*64),
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",72),
            ("original_UCI_data_graph_or_P1_masks_seeds_accessed",True),
            ("full_real_UCI_graph_mask_checkpoint_50k_CPU_RSS_disk_upper_bound_proven",True),
            ("original_Cython_baseline_rng_state_observable",True),
            ("native_OS_compiler_SDK_full_byte_closure_proven",True),
        ):
            current=copy.deepcopy(base)
            current[key]=value
            with self.subTest(key=key),self.assertRaises(q.ObserverBlocked):
                q.validate_report(current)

    def test_strict_OCI_worker_only_fake_data_and_expected_native_paths(self):
        cmd=q.docker_args("cython","pypcc-p207q-012345abcdef")
        for field in ("--network","none","--read-only","--user","65534:65534",
                      "--memory","2g","--memory-swap","--cap-drop","ALL",
                      "P207Q_FAKE_ONLY=1",
                      "PYTHONPATH="+q.p.k.CORRECT_PATH,
                      "experiments.p2_07q_exact_oci_synthetic_live_observer_50k"):
            self.assertIn(field,cmd)
        self.assertNotIn("P207P_SYNTHETIC_ONLY=1",cmd)
        self.assertNotIn("fit_predict"," ".join(cmd))
        for candidate in ("auto","real","NUMPY","cython+graph"):
            with self.assertRaises(q.ObserverBlocked):
                q.docker_args(candidate,"pypcc-p207q-012345abcdef")

    def test_three_worker_coverage_and_same_fake_graph_required(self):
        rows=[row(name) for name in q.BACKENDS]
        rows[1]["original_native_fullfit_and_historical_observer"][0]["synthetic_graph_sha256"]="b"*64
        with self.assertRaises(q.ObserverBlocked):
            q.assemble(rows)
        rows=[row(name) for name in q.BACKENDS]
        rows[1]["backend"]="numpy"
        with self.assertRaises(q.ObserverBlocked):
            q.assemble(rows)
        with self.assertRaises(q.ObserverBlocked):
            q.assemble([row("numpy"),row("numba")])

if __name__=="__main__":
    unittest.main()
