"""Pure stdlib P2-01G Linux host receipt failure-mode tests. NO PCC."""
import copy
from pathlib import Path
import tempfile
import unittest

from experiments import p2_01g_linux_host_runtime_receipt as host


def fake_receipt(path, *, native="a"*64):
    return {
        "schema":"pypcc-P2-01G-linux-exact-native-runtime-DSO-receipt/v1",
        "status":"LINUX_CANDIDATE_EXACT_RUNTIME_MAPPING_AND_RESOURCES_OBSERVED_NOT_FULL_P2_LOCK",
        "P1_scientific_sha256":host.P1_SHA,
        "prospective_native_P2_01F_build_run":43488066633,
        "P2_01F_archive_artifact_id":14068871078,
        "P2_01D_candidate_wheel_artifact_id":14049265732,
        "P2_01D_later_candidate_wheels_not_original_P2_01C":True,
        "Python":{"version":"3.12.16","system":"Linux","machine":"x86_64",
                  "executable_file":{"path":str(path),"bytes":11,"sha256":native}},
        "exact_file_backed_mapped_native_and_executable_hashes":[
            {"path":str(path),"bytes":11,"sha256":native}],
        "exact_new_native_Cython_modules_imported_without_fit":{
            "production_pcc_step":{"module_name":"pcc_step","native_file_sha256":native},
            "isolated_pcc_step_live_research":{
                "module_name":"pcc_step_live_research","native_file_sha256":native}},
        "host_observed_candidate_resource_capacity":{
            "physical_RAM_bytes":16*1024**3,
            "free_space_on_work_dir_bytes":30*1024**3,
            "candidate_limits_from_P2_07C_have_not_been_approved":True},
        "candidate_exact_13_wheel_distribution_versions":{
            "pkg"+str(x):"1.0" for x in range(13)},
        "candidate_policy_canonical_sha256":host.POLICY_SHA,
        "environment_source_artifact_digest_is_metadata_not_a_recheck_of_ZIP":True,
        "P2_01G_host_observation_not_an_immutable_OS_image":True,
        "P2_01G_complete_lazy_dynamic_dependency_closure_hashed":False,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "official_UCI_sources_loaded":0,"graph_or_mask_created":0,
        "synthetic_PCC_fits_executed":0,"real_PCC_fits_executed":0,
    }


class HostRuntimeReceiptTests(unittest.TestCase):
    def test_proc_maps_parser_ignores_anonymous_but_reports_deleted_files(self):
        paths,deleted=host.mapped_file_paths([
            "001-002 r-xp 00000000 00:00 0 /usr/lib/x86_64-linux-gnu/libc.so.6",
            "002-003 r--p 00000000 00:00 0 /usr/lib/x86_64-linux-gnu/libc.so.6",
            "004-005 rw-p 00000000 00:00 0 [heap]",
            "006-007 r-xp 00000000 00:00 0 /tmp/libold.so (deleted)",
        ])
        self.assertEqual(paths,["/usr/lib/x86_64-linux-gnu/libc.so.6"])
        self.assertEqual(deleted,["/tmp/libold.so (deleted)"])

    def test_real_file_hash_and_byte_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"mock-native.so"
            p.write_bytes(b"12345678901")
            observed=host.file_identity(p)
            self.assertEqual(observed["bytes"],11)
            self.assertEqual(len(observed["sha256"]),64)
            p.write_bytes(b"01234567890")
            self.assertNotEqual(observed["sha256"],host.file_identity(p)["sha256"])
            p.write_bytes(b"")
            with self.assertRaises(host.RuntimeReceiptBlocked):
                host.file_identity(p)

    def test_valid_synthetic_contract_but_NEVER_full_P2_approval(self):
        x=fake_receipt("/tmp/mock.so")
        self.assertTrue(host.validate_receipt(x))
        self.assertFalse(x["P2_execution_authorized"])
        self.assertFalse(x["P2_01_final_execution_toolchain_lock_pass"])
        self.assertIsNone(x["P2_execution_preflight_sha256"])

    def test_forbid_forged_execution_GO_and_real_fits(self):
        original=fake_receipt("/tmp/mock.so")
        for key,bad in (
            ("P2_execution_authorized",True),
            ("P2_execution_preflight_sha256","f"*64),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("real_PCC_fits_executed",1),
            ("synthetic_PCC_fits_executed",1),
            ("official_UCI_sources_loaded",1),
            ("graph_or_mask_created",1),
            ("P2_01G_host_observation_not_an_immutable_OS_image",False),
            ("P2_01G_complete_lazy_dynamic_dependency_closure_hashed",True),
        ):
            with self.subTest(key=key):
                changed=copy.deepcopy(original)
                changed[key]=bad
                with self.assertRaises(host.RuntimeReceiptBlocked):
                    host.validate_receipt(changed)

    def test_reject_original_native_or_wheel_source_conflation(self):
        original=fake_receipt("/tmp/mock.so")
        for key,bad in (
            ("P2_01F_archive_artifact_id",14041882611),
            ("P2_01D_candidate_wheel_artifact_id",14047753965),
            ("P2_01D_later_candidate_wheels_not_original_P2_01C",False),
            ("prospective_native_P2_01F_build_run",43434870932),
            ("P1_scientific_sha256","c"*64),
            ("candidate_policy_canonical_sha256","e"*64),
        ):
            with self.subTest(key=key):
                changed=copy.deepcopy(original)
                changed[key]=bad
                with self.assertRaises(host.RuntimeReceiptBlocked):
                    host.validate_receipt(changed)

    def test_reject_unbounded_or_ambiguous_dynamic_library_receipt(self):
        original=fake_receipt("/tmp/mock.so")
        changes=[
            {"exact_file_backed_mapped_native_and_executable_hashes":[]},
            {"exact_file_backed_mapped_native_and_executable_hashes":[
                {"path":"/tmp/libc.so","bytes":1,"sha256":"broken"}]},
            {"exact_file_backed_mapped_native_and_executable_hashes":[
                original["exact_file_backed_mapped_native_and_executable_hashes"][0]]*2},
            {"exact_new_native_Cython_modules_imported_without_fit":{
                "production_pcc_step":{"module_name":"pcc_step","native_file_sha256":"a"*64}}},
        ]
        for patch in changes:
            with self.subTest(patch=next(iter(patch))):
                with self.assertRaises(host.RuntimeReceiptBlocked):
                    host.validate_receipt({**original,**patch})


if __name__=="__main__":
    unittest.main()
