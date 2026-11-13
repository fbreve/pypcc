"""P2-01H adversarial source-only tests; no Docker pull, no package imports."""
import copy
from pathlib import Path
import tempfile
import unittest

from experiments import p2_01h_content_addressed_linux_image as candidate
from experiments import p2_01h_import_only_oci_selfcheck as image_check


class ContentAddressedCandidateTests(unittest.TestCase):
    def test_docker_registry_base_requires_digest_and_real_ubuntu26(self):
        original={
            "Os":"linux","Architecture":"amd64",
            "RepoDigests":["ubuntu@sha256:"+"a"*64],
            "Id":"sha256:"+"b"*64,
        }
        actual=candidate.base_identity(original,'NAME="Ubuntu"\nVERSION_ID="26.04"\n')
        self.assertEqual(actual["reference"],"ubuntu@sha256:"+"a"*64)
        self.assertTrue(actual["base_is_content_addressed_but_not_complete_runtime_image"])
        for changed,release in (
            ({**original,"Architecture":"arm64"},'VERSION_ID="26.04"\n'),
            ({**original,"RepoDigests":["ubuntu:26.04"]},'VERSION_ID="26.04"\n'),
            ({**original,"Id":"sha256:bad"},'VERSION_ID="26.04"\n'),
            (original,'VERSION_ID="24.04"\n'),
        ):
            with self.assertRaises(candidate.OCIBaseBlocked):
                candidate.base_identity(changed,release)

    def test_prior_digest_metadata_does_not_authorize_P2(self):
        discovered={
            "schema":"pypcc-P2-01H-immutable-OCI-Ubuntu26-base-discovery/v1",
            "status":"UBUNTU_26_AMD64_CONTENT_ADDRESSED_BASE_DISCOVERED_NO_PCC",
            "P1_scientific_sha256":candidate.P1_SHA,
            "OCI_base":{
                "reference":"ubuntu@sha256:"+"a"*64,
                "config_image_sha256":"b"*64,
                "platform":"linux/amd64",
                "observed_ubuntu_version_id":"26.04",
                "source_tag_used_only_for_initial_resolution":"ubuntu:26.04",
                "base_is_content_addressed_but_not_complete_runtime_image":True,
            },
            "P2_execution_preflight_sha256":None,
            "P2_execution_authorized":False,
            "real_PCC_fits_executed":0,
        }
        candidate.validate_base_doc(discovered)
        for keys,value in (
            (("P2_execution_preflight_sha256",),"a"*64),
            (("P2_execution_authorized",),True),
            (("real_PCC_fits_executed",),1),
            (("OCI_base","source_tag_used_only_for_initial_resolution"),"latest"),
            (("OCI_base","reference"),"ubuntu:26.04"),
        ):
            with self.subTest(keys=keys):
                wrong=copy.deepcopy(discovered)
                field=wrong
                for key in keys[:-1]:
                    field=field[key]
                field[keys[-1]]=value
                with self.assertRaises(candidate.OCIBaseBlocked):
                    candidate.validate_base_doc(wrong)

    def test_real_file_and_relative_symlink_tree_hashes_are_stable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"lib").mkdir()
            file=root/"lib"/"libpython.so.1.0"
            file.write_bytes(b"dummy native bytes only")
            (root/"lib"/"libpython.so").symlink_to("libpython.so.1.0")
            first=image_check.runtime_tree_digest(root)
            self.assertEqual(first["filesystem_entries"],3)
            self.assertEqual(first,image_check.runtime_tree_digest(root))
            file.write_bytes(b"changed native bytes")
            self.assertNotEqual(first["sha256"],
                                image_check.runtime_tree_digest(root)["sha256"])
            (root/"lib"/"escape.so").symlink_to("/tmp/outside.so")
            with self.assertRaises(image_check.OCIImportBlocked):
                image_check.runtime_tree_digest(root)

    def test_candidate_image_must_be_amd64_with_complete_false_GO_fields(self):
        manifest={"OCI_base":{
            "reference":"ubuntu@sha256:"+"a"*64,
            "config_image_sha256":"b"*64,
            },
            "copied_CP312_runtime_tree":{"sha256":"c"*64},
            "copied_host_native_runtime_libraries":{"libffi.so.8":{"sha256":"f"*64}},
        }
        receipt={
            "schema":"pypcc-P2-01H-pinned-container-import-only-receipt/v1",
            "status":"PINNED_UBUNTU26_BASE_AND_EXACT_PYTHON_WHEELS_CYTHON_IMPORTED_OFFLINE_NO_PCC",
            "P1_scientific_sha256":candidate.P1_SHA,
            "OCI_base_pinned_index_digest":manifest["OCI_base"]["reference"],
            "OCI_base_pinned_local_config_SHA256":"b"*64,
            "copied_python_runtime_tree_sha256":"c"*64,
            "python_version":"3.12.16",
            "number_of_loaded_mapped_native_files":5,
            "copied_separately_sealed_host_libffi_SHA256":"f"*64,
            "candidate_package_versions":{"pkg"+str(i):"1.0" for i in range(13)},
            "P2_01F_exact_new_native_imports":{
                "production_pcc_step":{},"isolated_pcc_step_live_research":{}},
            "real_UCI_source_rows_loaded":0,
            "graph_or_mask_built":0,
            "synthetic_PCC_fits_executed":0,
            "real_PCC_fits_executed":0,
            "P2_01_final_execution_toolchain_lock_pass":False,
            "P2_07_full_resources_gate_pass":False,
            "P2_execution_preflight_sha256":None,
            "P2_execution_authorized":False,
        }
        inspection={"Os":"linux","Architecture":"amd64","Id":"sha256:"+"d"*64}
        candidate.validate_image_receipt(receipt,manifest,inspection)
        for key,value in (
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",72),
            ("P2_execution_preflight_sha256","e"*64),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("copied_python_runtime_tree_sha256","z"*64),
            ("OCI_base_pinned_index_digest","ubuntu:26.04"),
        ):
            with self.subTest(key=key):
                mutated={**receipt,key:value}
                with self.assertRaises(candidate.OCIBaseBlocked):
                    candidate.validate_image_receipt(mutated,manifest,inspection)

    def test_import_only_manifest_full_approval_boundary_frozen(self):
        self.assertEqual(candidate.P1_SHA,
                         "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166")
        self.assertEqual(candidate.SOURCE_RUN,43488066633)
        self.assertEqual(candidate.BASE_TAG,"ubuntu:26.04")
        self.assertEqual(candidate.BASE_ARCH,"linux/amd64")
        self.assertEqual(candidate.SOURCE_RUN,43488066633)
        self.assertFalse(candidate.POLICY_SHA == candidate.P1_SHA)


if __name__=="__main__":
    unittest.main()
