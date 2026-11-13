"""P2-01H: resolve immutable Ubuntu26 OCI base and stage exact Python/native bytes.

This creates a transportable *candidate runtime substrate*, never a complete
scientific execution image. No UCI data, PCC model/kernel calls, graphs, masks
or synthetic or real PCC fits. Image built with FROM a validated sha256 digest.
The existing Python CP312 runtime is a COPIED, SHA-fingerprinted binary tree:
its original interpreter build inputs are not reproducible from archived
source, and the full OS/toolchain closure is NOT proven by this receipt.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from experiments import p2_07c_consolidated_execution_envelope_no_fit as prior
from experiments import p2_01e_replay_archived_wheelhouse as wheels
from experiments import p2_01h_import_only_oci_selfcheck as check

ROOT = Path(__file__).resolve().parents[1]
P1_SHA = prior.P1_SHA
BASE_TAG = "ubuntu:26.04"
BASE_ARCH = "linux/amd64"
SOURCE_RUN = 43488066633
POLICY_SHA = "b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04"


class OCIBaseBlocked(ValueError):
    pass


def require(ok, msg):
    if not ok:
        raise OCIBaseBlocked(msg)


def run(argv, *, check=True):
    p = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True,
                       errors="replace", timeout=240)
    if check:
        require(p.returncode == 0,
                "Docker/runtime command failed: " + " ".join(argv) +
                " stderr: " + p.stderr[-1600:])
    return p


def sha_file(path):
    return prior.file_sha(path)


def base_identity(inspection, os_release):
    require(type(inspection) is dict and inspection.get("Os")=="linux" and
            inspection.get("Architecture")=="amd64", "Wrong OCI platform")
    digests = [x for x in inspection.get("RepoDigests",[])
               if re.fullmatch(r"ubuntu@sha256:[0-9a-f]{64}",x)]
    require(len(set(digests))==1, "No unique content-addressed Ubuntu OCI base ref")
    config=inspection.get("Id","")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}",config) is not None,
            "No exact base config SHA")
    require(re.search(r'(?m)^VERSION_ID="?26\.04"?\s*$',os_release) is not None,
            "Container did not attest Ubuntu 26.04")
    return {"reference":digests[0],
            "config_image_sha256":config.removeprefix("sha256:"),
            "platform":BASE_ARCH,
            "observed_ubuntu_version_id":"26.04",
            "source_tag_used_only_for_initial_resolution":BASE_TAG,
            "base_is_content_addressed_but_not_complete_runtime_image":True}


def resolve_base():
    require(platform.system()=="Linux" and
            platform.machine().lower() in ("x86_64","amd64"),
            "Base resolver requires Linux AMD64 Docker host")
    run(["docker","pull","--platform",BASE_ARCH,BASE_TAG])
    inspect=json.loads(run(["docker","image","inspect",BASE_TAG]).stdout)
    require(len(inspect)==1,"Docker inspect must return one image")
    original=inspect[0]
    refs=[x for x in original.get("RepoDigests",[])
          if x.startswith("ubuntu@sha256:")]
    require(len(refs)==1, "Exactly one Ubuntu reference required")
    release=run(["docker","run","--rm","--network","none","--read-only",
                 "--platform",BASE_ARCH,refs[0],
                 "/bin/sh","-c","cat /etc/os-release"]).stdout
    identity=base_identity(original,release)
    by_digest=json.loads(run([
        "docker","image","inspect",identity["reference"]
    ]).stdout)
    require(len(by_digest)==1 and by_digest[0]["Id"]==original["Id"],
            "Digest pull did not resolve the same original rootfs/config image")
    return {
        "schema":"pypcc-P2-01H-immutable-OCI-Ubuntu26-base-discovery/v1",
        "status":"UBUNTU_26_AMD64_CONTENT_ADDRESSED_BASE_DISCOVERED_NO_PCC",
        "P1_scientific_sha256":P1_SHA,
        "OCI_base":identity,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    }


def validate_base_doc(doc):
    require(doc.get("schema")==
            "pypcc-P2-01H-immutable-OCI-Ubuntu26-base-discovery/v1" and
            doc.get("status")==
            "UBUNTU_26_AMD64_CONTENT_ADDRESSED_BASE_DISCOVERED_NO_PCC" and
            doc.get("P1_scientific_sha256")==P1_SHA and
            doc.get("P2_execution_preflight_sha256") is None and
            doc.get("P2_execution_authorized") is False and
            doc.get("real_PCC_fits_executed")==0, "Invalid base discover no-GO")
    base=doc.get("OCI_base",{})
    require(re.fullmatch(r"ubuntu@sha256:[0-9a-f]{64}",
                         base.get("reference","")) is not None and
            re.fullmatch(r"[0-9a-f]{64}",
                         base.get("config_image_sha256","")) is not None and
            base.get("platform")==BASE_ARCH and
            base.get("observed_ubuntu_version_id")=="26.04" and
            base.get("source_tag_used_only_for_initial_resolution")==BASE_TAG and
            base.get("base_is_content_addressed_but_not_complete_runtime_image") is True,
            "Resolved digest is not a frozen Ubuntu26 AMD64 base")
    return base


def prepare_context(base_doc, wheelhouse, hash_lock, sealed_native, context):
    base=validate_base_doc(base_doc)
    require(not context.exists(), "Cannot overwrite an existing OCI image context")
    require(platform.system()=="Linux" and
            platform.python_version()=="3.12.16" and
            Path(sys.prefix).resolve()==Path("/opt/hostedtoolcache/Python/3.12.16/x64"),
            "Exact cached CPython3.12.16 toolchain expected")
    prior.read_approved_P1()
    policy=prior.load(prior.POLICY)
    prior.validate_policy(policy,prior.read_approved_P1())
    require(prior.sha256(prior.canonical(policy))==POLICY_SHA,
            "Draft P2-07C resource policy drift")
    checked=wheels.prepare(wheelhouse,hash_lock)
    require(checked["platform"]=="Linux" and
            len(checked["candidate_archived_wheels_sha256"])==13,
            "Archived 13 candidate wheel files are not identical")
    expected={key:value["version"] for key,value in
              checked["candidate_archived_wheels_sha256"].items()}
    installed={key:importlib.metadata.version(key) for key in expected}
    require(installed==expected,"Installed candidate version pin drift")
    root=sealed_native/"p2-01f-new-native-candidate"
    receipt=prior.validate_new_native(
        prior.load(root/"p2-01f-prospective-build-ledger.json"),root,"Linux")
    require(receipt["candidate_installed_versions"]==expected,
            "Prior newly compiled native Python package environment drift")
    # Copy only the current interpreter tree and the TWO exact archived new
    # native files. Do NOT bundle original UCI source or project datasets.
    context.mkdir(parents=True)
    target=context/"python-runtime"
    shutil.copytree(sys.prefix,target,symlinks=True)
    original_tree=check.runtime_tree_digest(sys.prefix)
    image_context_tree=check.runtime_tree_digest(target)
    require(original_tree==image_context_tree,
            "Exact copied CP312 interpreter binary tree differs from host")
    native_dir=context/"new-native"
    native_dir.mkdir()
    sources={}
    for key,data in receipt["new_native_byte_identity_verified"].items():
        path=(root/"new-exact-native-seal"/"native"/data["filename"])
        require(path.is_file() and sha_file(path)==data["sha256"],
                "New P2-01F module not byte verified before OCI COPY")
        copy=native_dir/path.name
        shutil.copy2(path,copy)
        require(sha_file(copy)==data["sha256"],
                "New P2-01F module changed when staged for OCI COPY")
        sources[key]={"filename":copy.name,"sha256":data["sha256"],
                      "bytes":copy.stat().st_size}
    require(set(sources)=={"production_pcc_step",
                           "isolated_pcc_step_live_research"},
            "Both NEW native modules mandatory")
    # Minimal Ubuntu 26 OCI base omits libffi.so.8, which the copied
    # CPython _ctypes extension requires. Copy ONLY its exact host bytes,
    # without apt/network/build; seal and disclose it as an additional
    # host-sourced dependency, NOT as part of the pinned Ubuntu base.
    ffi_candidates = (
        Path("/usr/lib/x86_64-linux-gnu/libffi.so.8"),
        Path("/lib/x86_64-linux-gnu/libffi.so.8"),
    )
    libffi = next((x for x in ffi_candidates if x.is_file()), None)
    require(libffi is not None,
            "Host has no existing exact libffi.so.8; do not use unpinned apt")
    host_libraries = context/"selected-host-libraries"
    host_libraries.mkdir()
    copied_libffi = host_libraries/"libffi.so.8"
    shutil.copy2(libffi.resolve(strict=True), copied_libffi)
    require(sha_file(copied_libffi)==sha_file(libffi),
            "Selected host libffi changed on COPY")
    selected_host_libraries = {
        "libffi.so.8": {
            "sha256":sha_file(copied_libffi),
            "bytes":copied_libffi.stat().st_size,
            "host_original_resolved_path":str(libffi.resolve(strict=True)),
            "not_in_original_content_addressed_minimal_ubuntu_base":True,
        }
    }
    # Original Cython modules import these two original read-only Python
    # helpers on module initialization. Seal exact original Git objects;
    # never copy experiment runners, UCI rows or propagation source.
    helper_sources = {
        "pcc_metrics.py":"967353fc0f6ee88c8f26bbde6266645e6de9aa3b",
        "pcc_validation.py":"b98510431c2281463988321fb7f1b6b159a5f77d",
    }
    helper_dir=context/"source-only-native-import-helpers"
    helper_dir.mkdir()
    imported_helpers={}
    for name,blob in helper_sources.items():
        data=(ROOT/name).read_bytes()
        actual=hashlib.sha1(
            b"blob "+str(len(data)).encode("ascii")+b"\0"+data).hexdigest()
        require(actual==blob,"Original Cython import helper Git blob drift: "+name)
        target=helper_dir/name
        target.write_bytes(data)
        imported_helpers[name]={
            "original_git_blob_SHA1":blob,
            "source_file_SHA256":sha_file(target),
            "bytes":len(data),
        }
    shutil.copy2(
        ROOT/"experiments/p2_01h_import_only_oci_selfcheck.py",
        context/"oci-import-only-selfcheck.py")
    manifest={
        "schema":"pypcc-P2-01H-content-addressed-OCI-runtime-substrate/v1",
        "P1_scientific_sha256":P1_SHA,
        "OCI_base":base,
        "copied_CP312_runtime_tree":original_tree,
        "CPython_binary_tree_copied_not_rebuilt":True,
        "candidate_python_patch":"3.12.16",
        "exact_later_candidate_13_distributions":{
            key:{"version":value["version"],"wheel_sha256":value["sha256"],
                 "original_wheel_filename":value["wheel_filename"]}
            for key,value in checked["candidate_archived_wheels_sha256"].items()},
        "P2_01D_later_candidate_wheels_not_original_P2_01C":True,
        "P2_01F_exact_new_native":sources,
        "exact_frozen_python_native_import_helpers":imported_helpers,
        "copied_host_native_runtime_libraries":selected_host_libraries,
        "host_libffi_bytes_copied_not_installed_or_recompiled":True,
        "native_modules_are_NEW_P2_01F_not_historical":True,
        "P2_01H_selfcheck_source_sha256":
            sha_file(context/"oci-import-only-selfcheck.py"),
        "source_P2_01F_build_run":SOURCE_RUN,
        "proposed_P2_07C_policy_sha256":POLICY_SHA,
        "no_original_UCI_rows_or_PCC_scientific_runner_bundled":True,
        "P2_01_final_execution_toolchain_lock_pass":False,
        "P2_07_full_resources_gate_pass":False,
        "P2_execution_preflight_sha256":None,
        "P2_execution_authorized":False,
        "real_PCC_fits_executed":0,
    }
    (context/"runtime-manifest.json").write_text(
        json.dumps(manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    dockerfile=(
        "FROM "+base["reference"]+"\n"
        "COPY python-runtime/ /opt/hostedtoolcache/Python/3.12.16/x64/\n"
        "COPY new-native/ /opt/p2-native/\n"
        "COPY selected-host-libraries/ /opt/p2-hostlibs/\n"
        "COPY source-only-native-import-helpers/ /opt/p2-source-helpers/\n"
        "COPY runtime-manifest.json /opt/p2-runtime-manifest.json\n"
        "COPY oci-import-only-selfcheck.py /opt/p2-oci-selfcheck.py\n"
        "ENV PATH=/opt/hostedtoolcache/Python/3.12.16/x64/bin:$PATH\n"
        "ENV LD_LIBRARY_PATH=/opt/p2-hostlibs:/opt/hostedtoolcache/Python/3.12.16/x64/lib\n"
        "ENV PYTHONPATH=/opt/p2-source-helpers\n"
        "ENV PYTHONDONTWRITEBYTECODE=1\n"
        "ENV PYTHONNOUSERSITE=1\n"
        "WORKDIR /opt\n"
        'ENTRYPOINT ["/opt/hostedtoolcache/Python/3.12.16/x64/bin/python3.12",'
        '"/opt/p2-oci-selfcheck.py"]\n'
    )
    (context/"Dockerfile").write_text(dockerfile,encoding="utf-8")
    return {"manifest":manifest,"source_python_runtime_tree":original_tree,
            "dockerfile_sha256":sha_file(context/"Dockerfile"),
            "OCI_base":base}


def validate_image_receipt(receipt, manifest, image_inspection):
    require(receipt.get("schema")==
            "pypcc-P2-01H-pinned-container-import-only-receipt/v1" and
            receipt.get("status")==
            "PINNED_UBUNTU26_BASE_AND_EXACT_PYTHON_WHEELS_CYTHON_IMPORTED_OFFLINE_NO_PCC",
            "Pinned image selfcheck did not produce exact successful receipt")
    require(receipt.get("P1_scientific_sha256")==P1_SHA and
            receipt.get("OCI_base_pinned_index_digest")==
            manifest["OCI_base"]["reference"] and
            receipt.get("OCI_base_pinned_local_config_SHA256")==
            manifest["OCI_base"]["config_image_sha256"] and
            receipt.get("copied_python_runtime_tree_sha256")==
            manifest["copied_CP312_runtime_tree"]["sha256"] and
            receipt.get("python_version")=="3.12.16" and
            receipt.get("frozen_Cython_import_helper_git_blob_sha1")==
            {k:v["original_git_blob_SHA1"] for k,v in
             manifest["exact_frozen_python_native_import_helpers"].items()} and
            receipt.get("number_of_loaded_mapped_native_files",0)>=5 and
            receipt.get("copied_separately_sealed_host_libffi_SHA256")==
            manifest["copied_host_native_runtime_libraries"]["libffi.so.8"]["sha256"] and
            len(receipt.get("candidate_package_versions",{}))==13 and
            set(receipt.get("P2_01F_exact_new_native_imports",{}))==
            {"production_pcc_step","isolated_pcc_step_live_research"} and
            receipt.get("real_UCI_source_rows_loaded")==0 and
            receipt.get("graph_or_mask_built")==0 and
            receipt.get("synthetic_PCC_fits_executed")==0 and
            receipt.get("real_PCC_fits_executed")==0 and
            receipt.get("P2_01_final_execution_toolchain_lock_pass") is False and
            receipt.get("P2_07_full_resources_gate_pass") is False and
            receipt.get("P2_execution_preflight_sha256") is None and
            receipt.get("P2_execution_authorized") is False,
            "Container may not represent new P2 approval or different environment")
    require(image_inspection.get("Os")=="linux" and
            image_inspection.get("Architecture")=="amd64" and
            re.fullmatch(r"sha256:[0-9a-f]{64}",
                         image_inspection.get("Id","")) is not None,
            "Built OCI config image ID must be content addressed")
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("resolve","prepare","verify"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--base-json",type=Path)
    parser.add_argument("--wheelhouse",type=Path)
    parser.add_argument("--hash-requirements",type=Path)
    parser.add_argument("--native-artifact",type=Path)
    parser.add_argument("--context-dir",type=Path)
    parser.add_argument("--container-selfcheck",type=Path)
    parser.add_argument("--image-tag",default="pypcc-p2-01h-candidate:immutable")
    args=parser.parse_args()
    if args.operation=="resolve":
        result=resolve_base()
        print(json.dumps({"status":result["status"],
                          "OCI_base":result["OCI_base"]},sort_keys=True))
    elif args.operation=="prepare":
        require(all(x is not None for x in (
            args.base_json,args.wheelhouse,args.hash_requirements,
            args.native_artifact,args.context_dir)),
            "Prepare requires exact base, offline candidate wheel and original new native")
        doc=json.loads(args.base_json.read_text(encoding="utf-8"))
        result=prepare_context(doc,args.wheelhouse.resolve(),
                               args.hash_requirements.resolve(),
                               args.native_artifact.resolve(),
                               args.context_dir.resolve())
        print(json.dumps({"status":"EXACT_CONTAINER_BUILD_CONTEXT_STAGED_NO_PCC",
                          "copied_python_files":
                          result["source_python_runtime_tree"]["filesystem_entries"],
                          "OCI_base":result["OCI_base"]["reference"],
                          "P2_execution_authorized":False},sort_keys=True))
    else:
        require(args.context_dir and args.container_selfcheck,
                "Verify requires a checked context and real container import log")
        manifest=json.loads(
            (args.context_dir/"runtime-manifest.json").read_text(encoding="utf-8"))
        receipt=json.loads(args.container_selfcheck.read_text(encoding="utf-8"))
        images=json.loads(run(["docker","image","inspect",args.image_tag]).stdout)
        require(len(images)==1,"Exactly one complete pinned OCI candidate image expected")
        validate_image_receipt(receipt,manifest,images[0])
        result={
            "schema":"pypcc-P2-01H-OCI-build-and-offline-import-only-attestation/v1",
            "status":"CONTENT_ADDRESSED_BASE_COPIED_EXACT_CP312_AND_13_WHEELS_NEW_CYTHON_IMPORT_PASS",
            "P1_scientific_sha256":P1_SHA,
            "candidate_base_image_ref":manifest["OCI_base"]["reference"],
            "candidate_base_image_config_sha256":
                manifest["OCI_base"]["config_image_sha256"],
            "newly_built_candidate_image_config_id":images[0]["Id"],
            "candidate_image_is_linux_amd64":True,
            "copied_CPython_binary_tree_sha256":
                manifest["copied_CP312_runtime_tree"]["sha256"],
            "copied_CPython_runtime_file_count":
                manifest["copied_CP312_runtime_tree"]["filesystem_entries"],
            "exact_13_archived_wheel_distribution_sha256":{
                k:v["wheel_sha256"] for k,v in
                manifest["exact_later_candidate_13_distributions"].items()},
            "new_original_Cython_binary_sha256":{
                k:v["sha256"] for k,v in manifest["P2_01F_exact_new_native"].items()},
            "container_import_only_selfcheck_sha256":
                sha_file(args.container_selfcheck),
            "container_observed_mapped_native_files":
                receipt["number_of_loaded_mapped_native_files"],
            "container_image_tar_archive_bytes_sha256_not_included_yet":True,
            "candidate_runtime_binary_tree_copied_not_rebuilt":True,
            "ubuntu_base_digest_pinned_but_not_whole_OS_source_build_proven":True,
            "not_the_complete_final_PCC_scientific_execution_image":True,
            "P2_01_final_execution_toolchain_lock_pass":False,
            "P2_07_full_resources_gate_pass":False,
            "P2_execution_preflight_sha256":None,
            "P2_execution_authorized":False,
            "official_UCI_rows_loaded":0,"graph_or_mask_constructed":0,
            "synthetic_PCC_fits_executed":0,"real_PCC_fits_executed":0,
        }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                           encoding="utf-8")


if __name__=="__main__":
    main()
