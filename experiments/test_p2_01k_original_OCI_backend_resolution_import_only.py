"""P2-01K: source-only strict import/backend-path contract tests, no native imports."""
import ast
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from experiments import p2_01k_original_OCI_backend_resolution_import_only as s


def receipt(mode):
    corrected=mode==s.MODES[1]
    return {
        "schema":s.WORKER,
        "mode":mode,
        "P1_scientific_sha256":s.P1_SHA,
        "existing_exact_OCI_config_id":s.IMAGE_ID,
        "original_Cython_ELF_stored_in_image_SHA256":s.lock.NEW_NATIVE["production_pcc_step"],
        "original_observer_ELF_stored_in_image_SHA256":s.lock.NEW_NATIVE["isolated_pcc_step_live_research"],
        "cython_module_available_on_PYTHONPATH":corrected,
        "explicit_cython_request_resolved_actual_backend":"cython" if corrected else "numba",
        "numpy_backend_correct":True,
        "numba_backend_correct":True,
        "cython_backend_correct":corrected,
        "Cython_live_observer_imported":corrected,
        "model_graphs_built":0,
        "Numba_JIT_specializations_compiled":0,
        "PCC_fits_and_kernel_propagations_executed":0,
        "P2_execution_authorized":False,
    }

class P201KBackendResolutionNoFit(unittest.TestCase):
    def test_original_production_sources_remain_byte_locked(self):
        self.assertTrue(s.m.verify_source_blobs())
        self.assertEqual(len(s.lock.NEW_NATIVE),2)

    def test_original_fallback_is_present_in_source_and_not_mislabelled(self):
        code=(s.ROOT/"pcc.py").read_text(encoding="utf-8")
        tree=ast.parse(code)
        fn=next(x for x in tree.body if isinstance(x,ast.ClassDef) and
                x.name=="ParticleCompetitionAndCooperation")
        method=next(x for x in fn.body if isinstance(x,ast.FunctionDef) and
                    x.name=="_get_backend_fn")
        method_source=ast.get_source_segment(code,method)
        self.assertIn('["cython", "numba", "numpy"]',method_source)
        self.assertIn('if be == "cython" and _HAS_CYTHON',method_source)
        self.assertIn('if be == "numba" and _HAS_NUMBA',method_source)

    def test_historical_graph_only_path_omits_Cython_and_corrected_includes_both(self):
        self.assertEqual(s.FALLBACK_PATH,"/workspace")
        self.assertEqual(s.CORRECT_PATH,
                         "/workspace:/opt/p2-native:/opt/p2-source-helpers")
        self.assertNotIn("/opt/p2-native",s.FALLBACK_PATH)
        self.assertIn("/opt/p2-native",s.CORRECT_PATH)

    def test_dual_baseline_corrected_worker_receipts_validate(self):
        for mode in s.MODES:
            with self.subTest(mode=mode):
                self.assertTrue(s.validate_worker(receipt(mode),mode))
        self.assertTrue(s.validate_report(s.assemble([receipt(x) for x in s.MODES])))

    def test_no_false_execution_or_wrong_binary_hash_allowed(self):
        for key,value in (
            ("P2_execution_authorized",True),
            ("PCC_fits_and_kernel_propagations_executed",1),
            ("Numba_JIT_specializations_compiled",1),
            ("model_graphs_built",1),
            ("original_Cython_ELF_stored_in_image_SHA256","a"*64),
            ("original_observer_ELF_stored_in_image_SHA256","b"*64),
            ("existing_exact_OCI_config_id","sha256:"+"0"*64),
        ):
            with self.subTest(key=key):
                row=receipt(s.MODES[1])
                row[key]=value
                with self.assertRaises(s.BackendLockBlocked):
                    s.validate_worker(row,s.MODES[1])

    def test_false_backend_status_or_fallback_hide_blocked(self):
        for key,value in (
            ("cython_backend_correct",False),
            ("Cython_live_observer_imported",False),
            ("explicit_cython_request_resolved_actual_backend","numba"),
            ("cython_module_available_on_PYTHONPATH",False),
            ("numba_backend_correct",False),
            ("numpy_backend_correct",False),
        ):
            with self.subTest(key=key):
                row=receipt(s.MODES[1])
                row[key]=value
                with self.assertRaises(s.BackendLockBlocked):
                    s.validate_worker(row,s.MODES[1])

    def test_invalid_baseline_does_not_pass(self):
        row=receipt(s.MODES[0])
        row["cython_backend_correct"]=True
        with self.assertRaises(s.BackendLockBlocked):
            s.validate_worker(row,s.MODES[0])

    def test_wrong_mode_and_unsafe_container_names_fail_closed(self):
        for name in ("bad","pypcc-p201k-012345","pypcc-p201k-012345abcdef;rm"):
            with self.subTest(name=name):
                with self.assertRaises(s.BackendLockBlocked):
                    s.docker_args(s.MODES[0],name)
        with self.assertRaises(s.BackendLockBlocked):
            s.docker_args("run-real-PCC","pypcc-p201k-012345abcdef")

    def test_actual_docker_command_is_offline_readonly_nonroot_with_corrected_path(self):
        old=s.docker_args(s.MODES[0],"pypcc-p201k-012345abcdef")
        new=s.docker_args(s.MODES[1],"pypcc-p201k-012345abcdef")
        for term in ("--read-only","--network","none","65534:65534",
                     "--cap-drop","ALL","--memory","2g","--memory-swap",
                     "PYTHONPATH="+s.CORRECT_PATH,"--entrypoint"):
            self.assertIn(term,new)
        self.assertIn("PYTHONPATH="+s.FALLBACK_PATH,old)
        self.assertIn("readonly"," ".join(new))
        self.assertNotIn("docker build"," ".join(new))
        self.assertNotIn("docker pull"," ".join(new))
        self.assertNotIn("fit_predict"," ".join(new))

    def test_swap_reorder_or_missing_mode_never_admitted(self):
        for rows in ([receipt(s.MODES[0])],
                     [receipt(s.MODES[1]),receipt(s.MODES[0])],
                     [receipt(s.MODES[0]),receipt(s.MODES[0])]):
            with self.assertRaises(s.BackendLockBlocked):
                s.assemble(rows)

    def test_false_final_toolchain_and_P2_08_claims_fail(self):
        report=s.assemble([receipt(x) for x in s.MODES])
        for key,value in (
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_preflight_sha256","f"*64),
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",72),
            ("image_durable_off_GHA_archive_copy_proven",True),
            ("actual_P1_50000_iteration_Cython_and_NumPy_and_Numba_full_fit_observer_dispatch_proven",True),
        ):
            with self.subTest(key=key):
                with self.assertRaises(s.BackendLockBlocked):
                    s.validate_report({**report,key:value})

    def test_no_fit_canonical_report_is_source_safe_review_only(self):
        r=s.assemble([receipt(x) for x in s.MODES])
        self.assertEqual(s.digest(r),s.digest(json.loads(json.dumps(r))))
        self.assertEqual(r["real_PCC_fits_executed"],0)
        self.assertIsNone(r["P2_execution_preflight_sha256"])


if __name__=="__main__":
    unittest.main()
