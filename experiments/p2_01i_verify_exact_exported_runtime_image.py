"""P2-01I verify EXACT already-exported and independently imported P2-01H OCI bytes.

Unlike P2-01H, this workflow NEVER touches a floating Ubuntu registry tag,
re-resolves an OCI base, recompiles Python, downloads a new wheel, or creates
a Docker image from a recipe. It validates and offline docker-loads the single
previously reviewed tar.gz SHA, checks the identical config ID, and repeats
an import-only selfcheck without network or writable rootfs. No UCI/PCC fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from experiments import p2_07c_consolidated_execution_envelope_no_fit as prior

ROOT=Path(__file__).resolve().parents[1]
LOCK_PATH=ROOT/"docs/current_p2_01i_exact_OCI_binary_replay_lock_2026-11-13.json"
P1_SHA=prior.P1_SHA
EXPECTED_IMAGE_SHA="008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8"
EXPECTED_IMAGE_ID="sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55"
EXPECTED_BASE_REF="ubuntu@sha256:065aa8626841f16057b78b027b464cd59c4fac36c204c72dcbf48d66f5fa8a96"
EXPECTED_RUNTIME_TREE="cd4be74d169340445a87c34c1dace8e92598ab87b1625ea52ee8d31bad1590a7"
OLD_RUNTIME_TREE="496e9d7318d30d9300c949b75cf1709d65a73c2281289ee8786aaddb777f455d"
NEW_NATIVE={
    "production_pcc_step":"cec70629e10b1f6e9f53274d571a52ad2177d31b89d2c189840f3abda2944a8a",
    "isolated_pcc_step_live_research":"1ea9f1c151df795071b3fe449df2a3e35ab72bb79156b073df9f8297c66d899f",
}
IMPORT_HELPERS={
    "pcc_metrics.py":"967353fc0f6ee88c8f26bbde6266645e6de9aa3b",
    "pcc_validation.py":"b98510431c2281463988321fb7f1b6b159a5f77d",
}


class FrozenImageBlocked(ValueError):
    pass


def require(ok,msg):
    if not ok:
        raise FrozenImageBlocked(msg)


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for buf in iter(lambda:stream.read(1024*1024),b""):
            h.update(buf)
    return h.hexdigest()


def canonical(x):
    return json.dumps(x,ensure_ascii=True,allow_nan=False,sort_keys=True,
                      separators=(",",":")).encode("ascii")


def lock_checked(lock):
    require(lock.get("schema")==
            "pypcc-P2-01I-verified-OCI-binary-only-replay-lock/v1" and
            lock.get("status")==
            "LOCK_EXACT_ALREADY_TRANSPORTED_OCI_ARCHIVE_NOT_A_NEW_OS_BUILD_OR_P2_GO",
            "Unrecognized immutable binary runtime lock")
    require(lock.get("P1_scientific_sha256")==P1_SHA and
            lock.get("source_PR")==276 and
            lock.get("source_PR_head")==
            "70ddf6b398df5ddf0cc89766c5e01c7822fdbdf5" and
            lock.get("validated_source_workflow_run")==43516085668,
            "Original science or exact passing P2-01H source workflow drift")
    artifact=lock.get("source_action_artifact",{})
    receipt=lock.get("validated_separate_host_receipt_artifact",{})
    require(artifact.get("id")==14081565237 and
            artifact.get("name")=="p2-01h-transportable-OCI-image-and-safe-provenance" and
            artifact.get("github_archive_sha256")==
            "cbca5754de9b32048254b0875bffb30ac3bd6510df3168eeff6fb44fe203dead" and
            receipt.get("id")==14080014146 and
            receipt.get("name")==
            "p2-01h-separate-host-reloaded-container-import-only-receipt" and
            receipt.get("github_archive_sha256")==
            "826a6e89cf9e6bc9eb778ddd199d34fb92c7960e4c54eb757d3b0f72832af0dd",
            "Only exact original P2-01H saved source/replay artifacts accepted")
    exported=lock.get("docker_save_gzip_archive",{})
    require(exported.get("filename")=="p2-01h-exact-runtime-image.tar.gz" and
            exported.get("sha256")==EXPECTED_IMAGE_SHA and
            exported.get("original_image_config_id")==EXPECTED_IMAGE_ID and
            lock.get("base_oci_ref_exact")==EXPECTED_BASE_REF and
            lock.get("copied_CP312_runtime_tree_SHA256")==EXPECTED_RUNTIME_TREE and
            lock.get("copied_CP312_runtime_file_entries")==19199 and
            lock.get("candidate_python_version")=="3.12.16" and
            lock.get("later_P2_01D_wheels_count")==13 and
            lock.get("exact_new_P2_01F_Cython_native_SHA256")==NEW_NATIVE and
            lock.get("original_native_python_import_helper_Git_blob_SHA1")==IMPORT_HELPERS and
            lock.get("last_successful_import_only_mapped_file_count")==170,
            "Binary OCI exact content digest, native or Python tree does not match")
    require(lock.get("exact_archived_image_reimport_from_second_CI_runner_pass") is True and
            lock.get("different_image_rebuilds_had_different_Python_tree_SHA256") is True and
            OLD_RUNTIME_TREE!=EXPECTED_RUNTIME_TREE and
            lock.get("floating_base_tag_re_resolution_for_THIS_binary_replay_allowed") is False and
            lock.get("candidate_base_digest_locked_to_exported_binary") is True and
            lock.get("full_CPython_source_build_reproducibility_proven") is False and
            lock.get("full_OS_compiler_SDK_Numba_native_specialization_lock_proven") is False and
            lock.get("approved_execution_host_selected") is False and
            lock.get("real_UCI_50k_resource_bound_proven") is False and
            lock.get("durable_image_binary_archive_beyond_GHA_retention") is False and
            lock.get("P2_01_final_execution_toolchain_lock_pass") is False and
            lock.get("P2_07_full_resources_gate_pass") is False and
            lock.get("P2_08_final_execution_manifest_SHA256") is None and
            lock.get("P2_execution_preflight_sha256") is None and
            lock.get("P2_execution_authorized") is False and
            lock.get("real_PCC_fits_executed")==0,
            "Cannot promote a candidate OCI binary receipt to P2 approval")
    return lock


def only_file(root,name):
    paths=list(Path(root).rglob(name))
    require(len(paths)==1 and paths[0].is_file(),
            "Expected exactly one archived file named "+name)
    return paths[0]


def evidence_receipt_checked(receipt,lock):
    require(receipt.get("schema")==
            "pypcc-P2-01H-pinned-container-import-only-receipt/v1" and
            receipt.get("status")==
            "PINNED_UBUNTU26_BASE_AND_EXACT_PYTHON_WHEELS_CYTHON_IMPORTED_OFFLINE_NO_PCC" and
            receipt.get("P1_scientific_sha256")==P1_SHA and
            receipt.get("OCI_base_pinned_index_digest")==lock["base_oci_ref_exact"] and
            receipt.get("copied_python_runtime_tree_sha256")==
            lock["copied_CP312_runtime_tree_SHA256"] and
            receipt.get("copied_python_runtime_file_count")==19199 and
            receipt.get("python_version")=="3.12.16" and
            len(receipt.get("candidate_package_versions",{}))==13 and
            receipt.get("number_of_loaded_mapped_native_files")==170 and
            receipt.get("frozen_Cython_import_helper_git_blob_sha1")==IMPORT_HELPERS,
            "Wrong previous/transported native import-only provenance")
    native=receipt.get("P2_01F_exact_new_native_imports",{})
    require(set(native)==set(NEW_NATIVE) and
            all(native[k].get("sha256")==value and native[k].get("import_only") is True
                for k,value in NEW_NATIVE.items()),
            "New native modules are not precisely the already verified pair")
    require(re.fullmatch("[0-9a-f]{64}",
                         receipt.get("copied_separately_sealed_host_libffi_SHA256","")) is not None and
            receipt.get("libffi_was_not_part_of_pinned_minimal_ubuntu_base") is True and
            receipt.get("not_a_final_scientific_execution_image") is True and
            receipt.get("P2_01_final_execution_toolchain_lock_pass") is False and
            receipt.get("P2_07_full_resources_gate_pass") is False and
            receipt.get("P2_execution_preflight_sha256") is None and
            receipt.get("P2_execution_authorized") is False and
            receipt.get("real_UCI_source_rows_loaded")==0 and
            receipt.get("graph_or_mask_built")==0 and
            receipt.get("synthetic_PCC_fits_executed")==0 and
            receipt.get("real_PCC_fits_executed")==0,
            "Archived image makes a forbidden scientific execution claim")
    return True


def compare_archive_and_reports(lock,artifact_root,first_replay_root):
    bundle=only_file(artifact_root,"p2-01h-exact-runtime-image.tar.gz")
    require(sha_file(bundle)==lock["docker_save_gzip_archive"]["sha256"],
            "Full exported OCI archive bytes differ from exact locked artifact")
    att=json.loads(only_file(
        artifact_root,"p2-01h-image-build-and-import-attestation.json"
    ).read_text(encoding="utf-8"))
    require(att.get("schema")==
            "pypcc-P2-01H-OCI-build-and-offline-import-only-attestation/v1" and
            att.get("status")==
            "CONTENT_ADDRESSED_BASE_COPIED_EXACT_CP312_AND_13_WHEELS_NEW_CYTHON_IMPORT_PASS" and
            att.get("newly_built_candidate_image_config_id")==EXPECTED_IMAGE_ID and
            att.get("candidate_base_image_ref")==EXPECTED_BASE_REF and
            att.get("copied_CPython_binary_tree_sha256")==EXPECTED_RUNTIME_TREE and
            att.get("P2_execution_preflight_sha256") is None and
            att.get("P2_execution_authorized") is False and
            att.get("real_PCC_fits_executed")==0 and
            att.get("exported_full_OCI_runtime_archive",{}).get("sha256")==
            EXPECTED_IMAGE_SHA and
            att["exported_full_OCI_runtime_archive"].get("filename")==bundle.name and
            att["exported_full_OCI_runtime_archive"].get("bytes")==bundle.stat().st_size,
            "Original final P2-01H archive/export attestation differs")
    first=json.loads(only_file(
        artifact_root,"p2-01h-first-import-only-receipt.json"
    ).read_text(encoding="utf-8"))
    second=json.loads(only_file(
        first_replay_root,"replayed-image-import-only-receipt.json"
    ).read_text(encoding="utf-8"))
    for receipt in (first,second):
        evidence_receipt_checked(receipt,lock)
    for field in ("copied_python_runtime_tree_sha256",
                  "candidate_package_versions",
                  "P2_01F_exact_new_native_imports",
                  "copied_separately_sealed_host_libffi_SHA256",
                  "frozen_Cython_import_helper_git_blob_sha1"):
        require(first[field]==second[field],
                "Two old CI runner import receipts differ: "+field)
    return {"OCI_bundle_tar_gz_sha256":sha_file(bundle),
            "original_export_receipt_sha256":sha_file(only_file(
                artifact_root,"p2-01h-image-build-and-import-attestation.json")),
            "original_import_receipt_sha256":sha_file(only_file(
                artifact_root,"p2-01h-first-import-only-receipt.json")),
            "second_original_runner_import_receipt_sha256":sha_file(only_file(
                first_replay_root,"replayed-image-import-only-receipt.json")),
            "copied_python_tree_sha256":EXPECTED_RUNTIME_TREE,
            "candidate_OCI_image_config_SHA":EXPECTED_IMAGE_ID,
            "all_prior_binary_import_no_fit_claims_validated":True}


def validate_new_replay(new_receipt,lock):
    evidence_receipt_checked(new_receipt,lock)
    require(new_receipt.get("OCI_image_selfcheck_network_disabled_and_read_only") is True,
            "New locked image replay did not preserve network-free read-only policy")
    actual=json.loads(subprocess.run([
        "docker","image","inspect","pypcc-p2-01h-candidate:immutable"],
        capture_output=True,text=True,check=True).stdout)
    require(len(actual)==1 and actual[0]["Id"]==EXPECTED_IMAGE_ID and
            actual[0]["Architecture"]=="amd64" and actual[0]["Os"]=="linux",
            "Docker load restored different image than exact archived original")
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root",type=Path,required=True)
    p.add_argument("--previous-replay-root",type=Path,required=True)
    p.add_argument("--new-receipt",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    lock=lock_checked(json.loads(LOCK_PATH.read_text(encoding="utf-8")))
    proofs=compare_archive_and_reports(
        lock,args.artifact_root.resolve(),args.previous_replay_root.resolve())
    if args.new_receipt:
        validate_new_replay(json.loads(
            args.new_receipt.read_text(encoding="utf-8")),lock)
    report={
        "schema":"pypcc-P2-01I-exact-verified-OCI-archive-lock-check/v1",
        "status":("EXACT_BINARY_ARCHIVE_RELOADED_IN_NEW_HOST_IMPORT_ONLY"
                  if args.new_receipt else
                  "EXACT_PREVIOUS_ARCHIVE_AND_TWO_RUNNER_IMPORT_PROOFS_MATCH"),
        "P1_scientific_sha256":P1_SHA,
        "P2_01I_locked_policy_SHA256":
            hashlib.sha256(canonical(lock)).hexdigest(),
        "source_P2_01H_run":lock["validated_source_workflow_run"],
        "locked_original_container_config_id":EXPECTED_IMAGE_ID,
        "locked_exact_exported_runtime_tar_SHA256":EXPECTED_IMAGE_SHA,
        "locked_ubuntu26_base_digest":EXPECTED_BASE_REF,
        "replayed_original_image_without_floating_tag":True,
        "new_OCI_binary_import_only_pass":bool(args.new_receipt),
        "original_provenance":proofs,
        "not_source_rebuild_reproducibility":True,
        "host_native_abi_or_complete_72_fit_resource_budget_proven":False,
        "archive_not_durable_beyond_original_GHA_retention":True,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_UCI_data_loaded":False,
        "synthetic_PCC_fits_executed":0,
        "real_PCC_fits_executed":0,
    }
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps({"status":report["status"],
                      "lock_digest":report["P2_01I_locked_policy_SHA256"],
                      "OCI_binary":EXPECTED_IMAGE_ID,
                      "replayed_an_existing_archive":True,
                      "P2_execution_authorized":False},sort_keys=True))


if __name__=="__main__":
    main()
