"""P2-01J safe archive policy and forged-GO adversarial stdlib tests."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from experiments import p2_01j_verified_interim_oci_archive as archive


def candidate():
    return {
        "schema":"pypcc-P2-01J-exact-archived-image-interim-backup-receipt/v1",
        "status":"EXACT_PREVIOUS_CANDIDATE_OCI_BYTES_VERIFIED_FOR_TEMPORARY_ACTIONS_ARTIFACT",
        "P1_scientific_sha256":archive.P1_SHA,
        "locked_P2_01I_manifest_sha256":archive.LOCK_SHA,
        "original_P2_01H_successful_run":archive.ORIGINAL_SOURCE_RUN,
        "original_full_OCI_artifact_id":archive.ORIGINAL_ARTIFACT_ID,
        "current_repository":"fbreve/pypcc",
        "archival_destination":"GitHub Actions artifact only",
        "source_image_original_filename":"p2-01h-exact-runtime-image.tar.gz",
        "preserved_docker_save_gzip_SHA256":archive.EXPECTED_ORIGINAL_TAR_SHA,
        "preserved_docker_save_gzip_bytes":290_850_000,
        "original_OCI_image_config_id":archive.EXPECTED_CONFIG_SHA,
        "source_OCI_base_digest":archive.original.EXPECTED_BASE_REF,
        "source_CP312_tree_sha256":archive.original.EXPECTED_RUNTIME_TREE,
        "two_prior_hosts_binary_import_evidence_verified":True,
        "repo_artifact_retention_policy":archive.retention_policy(30,90),
        "new_GitHub_artifact_is_durable_forever":False,
        "new_GitHub_artifact_replaces_original":False,
        "new_GitHub_artifact_is_private_only_in_actions_UI":False,
        "archive_does_not_push_any_external_registry":True,
        "archive_does_not_change_GitHub_repo_settings":True,
        "one_additional_billed_artifact_copy_possible":True,
        "network_or_registry_Docker_pull":False,
        "Docker_rebuild_or_scientific_runner_included":False,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "official_UCI_source_rows_read":0,
        "graphs_or_masks_generated":0,
        "synthetic_PCC_fits_executed":0,
        "real_PCC_fits_executed":0,
    }


class InterimArchiveReceiptTests(unittest.TestCase):
    def test_exact_source_archive_contract_is_still_non_authorizing(self):
        x=candidate()
        self.assertTrue(archive.validate_receipt(x))
        self.assertIsNone(x["P2_execution_preflight_sha256"])
        self.assertFalse(x["P2_execution_authorized"])
        self.assertFalse(x["new_GitHub_artifact_is_durable_forever"])

    def test_policy_respects_current_repo_setting_and_public_ceiling(self):
        self.assertEqual(archive.retention_policy(30,90)[
            "requested_new_artifact_retention_days"],30)
        self.assertEqual(archive.retention_policy(90,90)[
            "requested_new_artifact_retention_days"],90)
        self.assertEqual(archive.retention_policy(400,400)[
            "requested_new_artifact_retention_days"],90)
        self.assertEqual(archive.retention_policy(20,10)[
            "requested_new_artifact_retention_days"],10)
        self.assertFalse(archive.retention_policy(30,90)[
            "new_artifact_90day_retention_available"])
        tentative=archive.unverified_policy_request()
        self.assertEqual(tentative["requested_new_artifact_retention_days"],90)
        self.assertIsNone(tentative["repo_configured_retention_days"])
        self.assertFalse(tentative["new_artifact_90day_retention_available"])
        self.assertTrue(tentative[
            "actual_expiry_is_unverified_until_GitHub_returns_it"])
        unknown=candidate()
        unknown["repo_artifact_retention_policy"]=tentative
        self.assertTrue(archive.validate_receipt(unknown))
        with self.assertRaises(archive.InterimArchiveBlocked):
            archive.unverified_policy_request(91)
        for setting,maximum in ((0,90),(-1,90),(None,90),(30,"90"),(401,90)):
            with self.subTest(setting=setting,maximum=maximum):
                with self.assertRaises(archive.InterimArchiveBlocked):
                    archive.retention_policy(setting,maximum)

    def test_all_source_SHA_and_identity_mutations_are_fail_closed(self):
        source=candidate()
        for key,wrong in (
            ("P1_scientific_sha256","a"*64),
            ("locked_P2_01I_manifest_sha256","f"*64),
            ("original_P2_01H_successful_run",1),
            ("original_full_OCI_artifact_id",2),
            ("source_image_original_filename","other.tar.gz"),
            ("preserved_docker_save_gzip_SHA256","0"*64),
            ("original_OCI_image_config_id","sha256:"+"0"*64),
            ("source_OCI_base_digest","ubuntu:26.04"),
            ("source_CP312_tree_sha256","1"*64),
            ("preserved_docker_save_gzip_bytes",100),
        ):
            with self.subTest(field=key):
                copy_=copy.deepcopy(source)
                copy_[key]=wrong
                with self.assertRaises(archive.InterimArchiveBlocked):
                    archive.validate_receipt(copy_)

    def test_never_mark_a_public_repo_artifact_private_or_durable(self):
        source=candidate()
        for field,wrong in (
            ("new_GitHub_artifact_is_private_only_in_actions_UI",True),
            ("new_GitHub_artifact_is_durable_forever",True),
            ("new_GitHub_artifact_replaces_original",True),
            ("current_repository","somewhere/else"),
            ("archival_destination","public release"),
            ("archive_does_not_push_any_external_registry",False),
            ("archive_does_not_change_GitHub_repo_settings",False),
            ("one_additional_billed_artifact_copy_possible",False),
        ):
            with self.subTest(field=field):
                altered={**source,field:wrong}
                with self.assertRaises(archive.InterimArchiveBlocked):
                    archive.validate_receipt(altered)

    def test_reject_forged_scientific_fits_or_execution_authorization(self):
        source=candidate()
        for field,wrong in (
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_preflight_sha256","e"*64),
            ("P2_execution_authorized",True),
            ("official_UCI_source_rows_read",1),
            ("graphs_or_masks_generated",3),
            ("synthetic_PCC_fits_executed",1),
            ("real_PCC_fits_executed",72),
            ("Docker_rebuild_or_scientific_runner_included",True),
            ("network_or_registry_Docker_pull",True),
        ):
            with self.subTest(field=field):
                with self.assertRaises(archive.InterimArchiveBlocked):
                    archive.validate_receipt({**source,field:wrong})

    def test_retention_policy_missing_original_archive_and_byte_drift(self):
        source=candidate()
        changed=copy.deepcopy(source)
        changed["repo_artifact_retention_policy"]["requested_new_artifact_retention_days"]=90
        with self.assertRaises(archive.InterimArchiveBlocked):
            archive.validate_receipt(changed)
        with tempfile.TemporaryDirectory() as dirname:
            path=Path(dirname)/"p2-01h-exact-runtime-image.tar.gz"
            path.write_bytes(b"wrong synthetic fixture, not real archive")
            self.assertNotEqual(archive.original.sha_file(path),
                                archive.EXPECTED_ORIGINAL_TAR_SHA)
            path.unlink()
            with self.assertRaises(archive.original.FrozenImageBlocked):
                archive.original.only_file(dirname,path.name)


if __name__=="__main__":
    unittest.main()
