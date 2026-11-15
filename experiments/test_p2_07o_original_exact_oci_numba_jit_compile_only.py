"""P2-07O strict source-only Numba compile-contract tests (NO JIT run)."""
import ast
import json
import subprocess
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import p2_07o_original_exact_oci_numba_jit_compile_only as o


def fake_jit_receipt():
    return {
        "schema":o.WORKER_SCHEMA,
        "status":o.STATUS,
        "P1_scientific_sha256":o.P1_SHA,
        "exact_existing_image_config_id":o.IMAGE_ID,
        "original_production_numba_git_blob":o.m.PRODUCTION_BLOBS["pcc_numba.py"],
        "original_source_only_argument_types_spec_SHA256":o.digest(o.TYPE_SPEC),
        "original_propagation_JIT_compiled_nopython_dispatcher_signatures":1,
        "original_nested_step_JIT_nopython_signatures_at_least_one":True,
        "original_seed_JIT_nopython_signatures":1,
        "JIT_signature_compilation_wall_seconds":10.5,
        "JIT_signature_compilation_process_CPU_seconds":9.2,
        "JIT_process_peak_resident_bytes":360*1024**2,
        "JIT_process_RSS_after_compile_bytes":330*1024**2,
        "imported_file_backed_objects_before_JIT":78,
        "imported_file_backed_objects_after_JIT":82,
        "imported_file_backed_objects_after_JIT_manifest_SHA256":"a"*64,
        "new_file_backed_objects_mapped_during_JIT_count":4,
        "new_file_backed_objects_mapped_during_JIT_manifest_SHA256":"b"*64,
        "this_IS_NOT_actual_PCC_50k_fit_peak_RSS_or_CPU":True,
        "original_scientific_matrices_graphs_masks_or_seeds_used":0,
        "Numba_dispatchers_invoked_for_propagation_or_RNG":0,
        "PCC_fits_or_propagations_executed":0,
        "P2_execution_authorized":False,
    }

class NumbaExactOCICompileNoFitTests(unittest.TestCase):
    def test_original_code_byte_identity_and_27_P1_arg_types(self):
        self.assertTrue(o.m.verify_source_blobs())
        self.assertEqual(len(o.TYPE_SPEC),27)
        self.assertEqual(o.TYPE_SPEC[10],"u8_mat_F")
        self.assertEqual(o.TYPE_SPEC[11:13],("f64_mat_F","f64_mat_F"))
        self.assertEqual(o.TYPE_SPEC[-1],"bool")

    def test_original_numba_propagation_is_27_args_not_executed(self):
        text=(o.ROOT/"pcc_numba.py").read_text(encoding="utf-8")
        tree=ast.parse(text)
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)
                and x.name=="_pcc_propagate_numba_sequential")
        self.assertEqual(len(fn.args.args),27)
        self.assertGreaterEqual(len(fn.decorator_list),1)
        self.assertIn("_pcc_step_numba_sequential",text)

    def test_type_spec_has_no_scientific_array_values(self):
        self.assertEqual(set(o.TYPE_SPEC),{
            "i64_mat_C","i64_vec_C","f64","i64","f64_vec_C",
            "u8_mat_F","f64_mat_F","bool",
        })
        self.assertNotIn("fit_predict",o.TYPE_SPEC)

    def test_frozen_original_source_worker_check(self):
        self.assertTrue(o.validate_worker(fake_jit_receipt()))
        self.assertTrue(o.validate_report(o.assemble(fake_jit_receipt())))

    def test_false_execution_and_mutated_source_blocked(self):
        for key,value in (
            ("P2_execution_authorized",True),
            ("PCC_fits_or_propagations_executed",1),
            ("Numba_dispatchers_invoked_for_propagation_or_RNG",1),
            ("original_scientific_matrices_graphs_masks_or_seeds_used",1),
            ("original_production_numba_git_blob","0"*40),
            ("exact_existing_image_config_id","sha256:"+"0"*64),
            ("original_source_only_argument_types_spec_SHA256","0"*64),
        ):
            with self.subTest(key=key):
                row=fake_jit_receipt()
                row[key]=value
                with self.assertRaises(o.CompileOnlyBlocked):
                    o.validate_worker(row)

    def test_invalid_CPU_RSS_native_mapping_totals_blocked(self):
        for key,value in (
            ("JIT_signature_compilation_wall_seconds",240),
            ("JIT_signature_compilation_process_CPU_seconds",-1),
            ("JIT_process_peak_resident_bytes",0),
            ("JIT_process_RSS_after_compile_bytes",500*1024**2),
            ("imported_file_backed_objects_before_JIT",1000),
            ("new_file_backed_objects_mapped_during_JIT_count",99),
            ("new_file_backed_objects_mapped_during_JIT_manifest_SHA256","0"),
        ):
            with self.subTest(key=key):
                row=fake_jit_receipt()
                row[key]=value
                with self.assertRaises(o.CompileOnlyBlocked):
                    o.validate_worker(row)

    def test_host_restricts_original_OCI_offline_readonly_and_tmpfs(self):
        command=o.docker_args("pypcc-p207o-012345abcdef")
        for term in ("--read-only","--network","none","65534:65534",
                     "--cap-drop","ALL","--memory","2g","--memory-swap",
                     "--entrypoint",o.PYTHON_IN_OCI,
                     "NUMBA_CACHE_DIR=/tmp/numba-cache",
                     "--pids-limit","128"):
            self.assertIn(term,command)
        self.assertTrue(any("mode=1777,size=128m" in x for x in command))
        self.assertIn("readonly"," ".join(command))
        self.assertNotIn("fit_predict"," ".join(command))
        self.assertNotIn("docker build"," ".join(command))
        self.assertNotIn("docker pull"," ".join(command))

    def test_wrong_container_names_blocked(self):
        for name in ("foo","pypcc-p207o-123","pypcc-p207o-012345abcdef;rm"):
            with self.subTest(name=name):
                with self.assertRaises(o.CompileOnlyBlocked):
                    o.docker_args(name)

    def test_zero_fits_and_resource_full_GO_cannot_be_inferred(self):
        original=o.assemble(fake_jit_receipt())
        for key,value in (
            ("complete_runtime_OS_compiler_SDK_build_byte_closure_proven",True),
            ("full_50k_original_PCC_fit_CPU_wall_RSS_disk_resource_bound_proven",True),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_preflight_sha256","a"*64),
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",72),
            ("JIT_compilation_only_is_NOT_actual_50000_iteration_full_fit_resource_bound",False),
        ):
            with self.subTest(key=key):
                with self.assertRaises(o.CompileOnlyBlocked):
                    o.validate_report({**original,key:value})

    def test_JIT_failure_is_failclosed_and_container_cleans_up(self):
        failed=subprocess.CompletedProcess(["docker"],1,"","P207O_SAFE_ERROR|code.py|55|CompileOnlyBlocked")
        with patch.object(o.subprocess,"run",return_value=
                          subprocess.CompletedProcess(["docker","rm"],1,"","")) as cleanup:
            with self.assertRaises(o.CompileOnlyBlocked):
                o.run_compile_only(runner=lambda *a,**kw:failed)
            self.assertTrue(cleanup.called)

    def test_canonical_report_sha_is_not_P2_execution_SHA(self):
        doc=o.assemble(fake_jit_receipt())
        self.assertEqual(o.digest(doc),o.digest(json.loads(json.dumps(doc))))
        self.assertIsNone(doc["P2_execution_preflight_sha256"])
        self.assertFalse(doc["P2_execution_authorized"])
        self.assertEqual(doc["real_PCC_fits_executed"],0)


if __name__=="__main__":
    unittest.main()
