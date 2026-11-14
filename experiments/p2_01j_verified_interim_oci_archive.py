"""P2-01J: exact pre-existing OCI archive backup receipts (ZERO PCC).

Never create/build/pull an OCI image. Never read any UCI samples, construct
graphs/masks, run synthetic/real PCC, change the production package or grant P2.
All original binary bytes originate from successful P2-01H Actions run
43516085668, with the same SHA validated independently by P2-01I #277.
GitHub Actions copies are TEMPORARY; their actual reported expiry must be
checked after upload. Durable private archive requires a separately chosen
destination, e.g. the optional manually invoked local PowerShell helper.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiments import p2_01i_verify_exact_exported_runtime_image as original

P1_SHA=original.P1_SHA
LOCK=original.LOCK_PATH
ORIGINAL_ARTIFACT_ID=14081565237
ORIGINAL_SOURCE_RUN=43516085668
LOCK_SHA="7fa90329779980f1b8739428796617a2da1ba872dac2e4c02d23fb07db3f5340"
EXPECTED_ORIGINAL_TAR_SHA=original.EXPECTED_IMAGE_SHA
EXPECTED_CONFIG_SHA=original.EXPECTED_IMAGE_ID
MAX_PUBLIC_REPO_DAYS=90


class InterimArchiveBlocked(ValueError):
    pass


def require(condition, msg):
    if not condition:
        raise InterimArchiveBlocked(msg)


def retention_policy(observed_setting_days, observed_maximum_days):
    """A new upload cannot exceed the present repository retention setting.
    90 days is a public-repo ceiling, never a guarantee or a durable backup.
    """
    for value in (observed_setting_days,observed_maximum_days):
        require(type(value) is int and 1<=value<=400,
                "Repository artifact retention policy not verified")
    allowed=min(MAX_PUBLIC_REPO_DAYS,
                observed_setting_days,observed_maximum_days)
    require(1<=allowed<=90,
            "The allowed retention must fit public repository rules")
    return {
        "repo_configured_retention_days":observed_setting_days,
        "repo_maximum_allowed_days":observed_maximum_days,
        "requested_new_artifact_retention_days":allowed,
        "new_artifact_90day_retention_available":allowed==90,
        "not_permanent_storage":True,
        "not_a_retroactive_extension_of_original_artifact":True,
    }


def unverified_policy_request(requested_days=90):
    """Endpoint may deny GITHUB_TOKEN (HTTP 403); 90 is a REQUEST, not policy.
    Only GitHub's actual upload+artifact.expires_at can confirm acceptance.
    """
    require(type(requested_days) is int and requested_days==90,
            "Unknown repository setting only permits explicit 90-day TRY")
    return {
        "repo_configured_retention_days":None,
        "repo_maximum_allowed_days":MAX_PUBLIC_REPO_DAYS,
        "requested_new_artifact_retention_days":requested_days,
        "new_artifact_90day_retention_available":False,
        "actual_expiry_is_unverified_until_GitHub_returns_it":True,
        "GITHUB_TOKEN_could_not_read_repository_retention_setting":True,
        "not_permanent_storage":True,
        "not_a_retroactive_extension_of_original_artifact":True,
    }


def validate_receipt(receipt):
    require(receipt.get("schema")==
            "pypcc-P2-01J-exact-archived-image-interim-backup-receipt/v1"
            and receipt.get("status")==
            "EXACT_PREVIOUS_CANDIDATE_OCI_BYTES_VERIFIED_FOR_TEMPORARY_ACTIONS_ARTIFACT",
            "Not an original P2-01J research-only safe archive receipt")
    require(receipt.get("P1_scientific_sha256")==P1_SHA and
            receipt.get("locked_P2_01I_manifest_sha256")==LOCK_SHA and
            receipt.get("original_P2_01H_successful_run")==ORIGINAL_SOURCE_RUN and
            receipt.get("original_full_OCI_artifact_id")==ORIGINAL_ARTIFACT_ID and
            receipt.get("preserved_docker_save_gzip_SHA256")==EXPECTED_ORIGINAL_TAR_SHA and
            receipt.get("original_OCI_image_config_id")==EXPECTED_CONFIG_SHA and
            receipt.get("source_OCI_base_digest")==original.EXPECTED_BASE_REF and
            receipt.get("source_CP312_tree_sha256")==original.EXPECTED_RUNTIME_TREE and
            receipt.get("source_image_original_filename")==
            "p2-01h-exact-runtime-image.tar.gz",
            "Original archived image/hash source drift")
    file_size=receipt.get("preserved_docker_save_gzip_bytes")
    require(type(file_size) is int and 200*1024**2<=file_size<2*1024**3,
            "Whole original OCI image bytes absent or nonsensical size")
    policy=receipt.get("repo_artifact_retention_policy",{})
    require(policy.get("repo_configured_retention_days") is None or
            type(policy.get("repo_configured_retention_days")) is int,
            "Repository config must be an integer only if verified")
    for key in ("repo_maximum_allowed_days",
                "requested_new_artifact_retention_days"):
        require(type(policy.get(key)) is int, "Retention setting missing")
    if policy["repo_configured_retention_days"] is None:
        expected=unverified_policy_request(policy["requested_new_artifact_retention_days"])
    else:
        expected=retention_policy(policy["repo_configured_retention_days"],
                                  policy["repo_maximum_allowed_days"])
    require(policy==expected, "Repository retention budget was silently modified")
    require(receipt.get("new_GitHub_artifact_is_durable_forever") is False and
            receipt.get("new_GitHub_artifact_replaces_original") is False and
            receipt.get("new_GitHub_artifact_is_private_only_in_actions_UI") is False and
            receipt.get("archive_does_not_push_any_external_registry") is True and
            receipt.get("archive_does_not_change_GitHub_repo_settings") is True and
            receipt.get("one_additional_billed_artifact_copy_possible") is True and
            receipt.get("network_or_registry_Docker_pull") is False and
            receipt.get("Docker_rebuild_or_scientific_runner_included") is False and
            receipt.get("P2_01_final_execution_toolchain_lock_pass") is False and
            receipt.get("P2_07_full_resources_gate_pass") is False and
            receipt.get("P2_execution_preflight_sha256") is None and
            receipt.get("P2_execution_authorized") is False and
            receipt.get("official_UCI_source_rows_read")==0 and
            receipt.get("graphs_or_masks_generated")==0 and
            receipt.get("synthetic_PCC_fits_executed")==0 and
            receipt.get("real_PCC_fits_executed")==0,
            "Artifact receipt claims unauthorized or scientifically impossible action")
    require(receipt.get("current_repository")=="fbreve/pypcc"
            and receipt.get("archival_destination")=="GitHub Actions artifact only",
            "Artifact was uploaded to a different service/repository")
    return True


def verify_source(artifact_root,prior_replay_root,setting_days,max_days):
    lock=original.lock_checked(json.loads(LOCK.read_text(encoding="utf-8")))
    require(hashlib.sha256(original.canonical(lock)).hexdigest()==LOCK_SHA,
            "Previously validated P2-01I archive lock changed")
    proofs=original.compare_archive_and_reports(
        lock, Path(artifact_root), Path(prior_replay_root))
    path=original.only_file(artifact_root,"p2-01h-exact-runtime-image.tar.gz")
    require(path.stat().st_size>=200*1024**2,
            "Invalid imported original Docker archive size")
    policy=(unverified_policy_request(90) if setting_days is None
            else retention_policy(setting_days,max_days))
    report={
        "schema":"pypcc-P2-01J-exact-archived-image-interim-backup-receipt/v1",
        "status":"EXACT_PREVIOUS_CANDIDATE_OCI_BYTES_VERIFIED_FOR_TEMPORARY_ACTIONS_ARTIFACT",
        "P1_scientific_sha256":P1_SHA,
        "locked_P2_01I_manifest_sha256":LOCK_SHA,
        "original_P2_01H_successful_run":ORIGINAL_SOURCE_RUN,
        "original_full_OCI_artifact_id":ORIGINAL_ARTIFACT_ID,
        "current_repository":"fbreve/pypcc",
        "archival_destination":"GitHub Actions artifact only",
        "source_image_original_filename":path.name,
        "preserved_docker_save_gzip_SHA256":proofs["OCI_bundle_tar_gz_sha256"],
        "preserved_docker_save_gzip_bytes":path.stat().st_size,
        "original_OCI_image_config_id":EXPECTED_CONFIG_SHA,
        "source_OCI_base_digest":original.EXPECTED_BASE_REF,
        "source_CP312_tree_sha256":original.EXPECTED_RUNTIME_TREE,
        "two_prior_hosts_binary_import_evidence_verified":proofs[
            "all_prior_binary_import_no_fit_claims_validated"],
        "repo_artifact_retention_policy":policy,
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
        "real_PCC_fits_executed":0
    }
    validate_receipt(report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-artifact-dir",required=True,type=Path)
    p.add_argument("--prior-replay-dir",required=True,type=Path)
    p.add_argument("--repo-retention-days",type=int,default=None,
                   help="Required only if verified by repo policy REST; omit after 403")
    p.add_argument("--repo-max-days",type=int,default=90)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    report=verify_source(args.source_artifact_dir,
                         args.prior_replay_dir,
                         args.repo_retention_days,args.repo_max_days)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        "status":report["status"],
        "preserved_docker_save_gzip_sha256":report["preserved_docker_save_gzip_SHA256"],
        "preserved_image_bytes":report["preserved_docker_save_gzip_bytes"],
        "REQUESTED_NOT_YET_CONFIRMED_retention_days":report["repo_artifact_retention_policy"][
            "requested_new_artifact_retention_days"],
        "permanent_copy_created":False,
        "P2_execution_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
