"""P2-01I: source-only adversarial immutable OCI lock tests; no Docker/fit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments import p2_01i_verify_exact_exported_runtime_image as frozen


class ExactOCILockTests(unittest.TestCase):
    def setUp(self):
        self.lock=json.loads(frozen.LOCK_PATH.read_text(encoding="utf-8"))

    def test_actual_github_archive_digest_and_image_config_are_locked(self):
        result=frozen.lock_checked(self.lock)
        self.assertEqual(result["docker_save_gzip_archive"]["sha256"],
                         frozen.EXPECTED_IMAGE_SHA)
        self.assertEqual(result["docker_save_gzip_archive"]["original_image_config_id"],
                         frozen.EXPECTED_IMAGE_ID)
        self.assertEqual(result["copied_CP312_runtime_tree_SHA256"],
                         frozen.EXPECTED_RUNTIME_TREE)
        self.assertNotEqual(frozen.OLD_RUNTIME_TREE,frozen.EXPECTED_RUNTIME_TREE)
        self.assertFalse(result["P2_execution_authorized"])
        self.assertIsNone(result["P2_execution_preflight_sha256"])

    def test_refuse_mutated_artifact_id_zip_sha_base_or_python_bytes(self):
        cases=[
            (("source_action_artifact","id"),1),
            (("source_action_artifact","github_archive_sha256"),"f"*64),
            (("validated_source_workflow_run",),1),
            (("docker_save_gzip_archive","sha256"),"f"*64),
            (("docker_save_gzip_archive","original_image_config_id"),
             "sha256:"+"0"*64),
            (("base_oci_ref_exact",),"ubuntu:26.04"),
            (("copied_CP312_runtime_tree_SHA256",),frozen.OLD_RUNTIME_TREE),
            (("floating_base_tag_re_resolution_for_THIS_binary_replay_allowed",),
             True),
        ]
        for keys,wrong in cases:
            with self.subTest(keys=keys):
                modified=copy.deepcopy(self.lock)
                part=modified
                for key in keys[:-1]:
                    part=part[key]
                part[keys[-1]]=wrong
                with self.assertRaises(frozen.FrozenImageBlocked):
                    frozen.lock_checked(modified)

    def test_false_P2_go_and_rebuild_claims_never_pass(self):
        for key,value in (
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_preflight_sha256","0"*64),
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",1),
            ("full_CPython_source_build_reproducibility_proven",True),
            ("full_OS_compiler_SDK_Numba_native_specialization_lock_proven",True),
            ("durable_image_binary_archive_beyond_GHA_retention",True),
        ):
            with self.subTest(key=key):
                altered={**self.lock,key:value}
                with self.assertRaises(frozen.FrozenImageBlocked):
                    frozen.lock_checked(altered)

    def test_original_Cython_native_sha_and_two_import_only_helpers_locked(self):
        self.assertEqual(set(self.lock["exact_new_P2_01F_Cython_native_SHA256"]),
                         set(frozen.NEW_NATIVE))
        self.assertEqual(self.lock["exact_new_P2_01F_Cython_native_SHA256"],
                         frozen.NEW_NATIVE)
        self.assertEqual(self.lock["original_native_python_import_helper_Git_blob_SHA1"],
                         frozen.IMPORT_HELPERS)

    def test_resolve_github_flattened_or_nested_report_uniquely(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            nested=root/"a"/"b"
            nested.mkdir(parents=True)
            x=nested/"report.json"
            x.write_text('{"valid":true}',encoding="utf-8")
            self.assertEqual(frozen.only_file(root,"report.json"),x)
            (root/"report.json").write_text("{}",encoding="utf-8")
            with self.assertRaises(frozen.FrozenImageBlocked):
                frozen.only_file(root,"report.json")

    def test_valid_import_only_receipt_rejects_forged_real_fits(self):
        obj={
            "schema":"pypcc-P2-01H-pinned-container-import-only-receipt/v1",
            "status":"PINNED_UBUNTU26_BASE_AND_EXACT_PYTHON_WHEELS_CYTHON_IMPORTED_OFFLINE_NO_PCC",
            "P1_scientific_sha256":frozen.P1_SHA,
            "OCI_base_pinned_index_digest":frozen.EXPECTED_BASE_REF,
            "copied_python_runtime_tree_sha256":frozen.EXPECTED_RUNTIME_TREE,
            "copied_python_runtime_file_count":19199,
            "python_version":"3.12.16",
            "candidate_package_versions":{"pkg"+str(i):"1.0" for i in range(13)},
            "number_of_loaded_mapped_native_files":170,
            "frozen_Cython_import_helper_git_blob_sha1":frozen.IMPORT_HELPERS,
            "P2_01F_exact_new_native_imports":{
                k:{"sha256":h,"import_only":True} for k,h in frozen.NEW_NATIVE.items()},
            "copied_separately_sealed_host_libffi_SHA256":"a"*64,
            "libffi_was_not_part_of_pinned_minimal_ubuntu_base":True,
            "not_a_final_scientific_execution_image":True,
            "P2_01_final_execution_toolchain_lock_pass":False,
            "P2_07_full_resources_gate_pass":False,
            "P2_execution_preflight_sha256":None,
            "P2_execution_authorized":False,
            "real_UCI_source_rows_loaded":0,
            "graph_or_mask_built":0,
            "synthetic_PCC_fits_executed":0,
            "real_PCC_fits_executed":0,
        }
        frozen.evidence_receipt_checked(obj,self.lock)
        for key,bad in (
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",1),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("copied_python_runtime_tree_sha256","b"*64),
        ):
            with self.subTest(key=key):
                with self.assertRaises(frozen.FrozenImageBlocked):
                    frozen.evidence_receipt_checked({**obj,key:bad},self.lock)


if __name__=="__main__":
    unittest.main()
